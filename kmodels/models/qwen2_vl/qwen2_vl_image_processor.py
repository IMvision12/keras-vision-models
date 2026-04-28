import math
from typing import List, Tuple

import numpy as np
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be <= 200, got {max(height, width) / min(height, width)}"
        )

    def _round_by(v, f):
        return max(f, round(v / f) * f)

    def _floor_by(v, f):
        return max(f, math.floor(v / f) * f)

    def _ceil_by(v, f):
        return max(f, math.ceil(v / f) * f)

    h_bar = _round_by(height, factor)
    w_bar = _round_by(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by(height / beta, factor)
        w_bar = _floor_by(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by(height * beta, factor)
        w_bar = _ceil_by(width * beta, factor)
    return h_bar, w_bar


class Qwen2VLImageProcessor:
    def __init__(
        self,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        image_mean=CLIP_MEAN,
        image_std=CLIP_STD,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.image_mean = np.asarray(image_mean, dtype=np.float32)
        self.image_std = np.asarray(image_std, dtype=np.float32)
        self.factor = patch_size * merge_size

    def _preprocess_single(
        self, img: Image.Image
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        img = img.convert("RGB")
        w, h = img.size
        h_new, w_new = _smart_resize(
            h, w, self.factor, self.min_pixels, self.max_pixels
        )
        img = img.resize((w_new, h_new), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.image_mean) / self.image_std
        arr = arr.transpose(2, 0, 1)

        frames = np.stack([arr] * self.temporal_patch_size, axis=0)

        grid_t = frames.shape[0] // self.temporal_patch_size
        grid_h = h_new // self.patch_size
        grid_w = w_new // self.patch_size
        ms = self.merge_size
        ps = self.patch_size
        T = self.temporal_patch_size

        patches = frames.reshape(
            grid_t, T, 3, grid_h // ms, ms, ps, grid_w // ms, ms, ps
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(
            grid_t * grid_h * grid_w,
            3 * T * ps * ps,
        ).astype(np.float32)
        return flat, (grid_t, grid_h, grid_w)

    def __call__(self, images: List[Image.Image]):
        if isinstance(images, Image.Image):
            images = [images]
        all_pv = []
        all_thw = []
        for img in images:
            pv, thw = self._preprocess_single(img)
            all_pv.append(pv)
            all_thw.append(thw)
        pixel_values = np.concatenate(all_pv, axis=0)
        grid_thw = np.asarray(all_thw, dtype=np.int32)
        return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
