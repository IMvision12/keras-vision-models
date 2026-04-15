from __future__ import annotations

import io
import os
from typing import List, Optional, Sequence, Tuple, Union

import keras
import numpy as np
import PIL.Image
from keras import ops
from PIL import Image

ImageInput = Union[str, bytes, bytearray, np.ndarray, "PIL.Image.Image"]
BatchImageInput = Union[ImageInput, Sequence[ImageInput]]
SizeLike = Union[int, Tuple[int, int]]


def get_data_format(data_format: Optional[str] = None) -> str:
    """Return a concrete data format string.

    Args:
        data_format: Either ``"channels_first"``, ``"channels_last"``, or
            ``None``. When ``None``, defaults to the global Keras setting from
            ``keras.config.image_data_format()``.
    """
    if data_format is None:
        return keras.config.image_data_format()
    if data_format not in ("channels_first", "channels_last"):
        raise ValueError(
            "data_format must be 'channels_first', 'channels_last', or None; "
            f"got {data_format!r}."
        )
    return data_format


def load_image(image: ImageInput) -> np.ndarray:
    """Load an image from common sources into an ``(H, W, 3)`` uint8 RGB array.

    Accepted inputs:
        * ``str`` — a local file path or an ``http(s)://`` URL.
        * ``bytes`` / ``bytearray`` — raw encoded image bytes.
        * ``PIL.Image.Image`` — returned as a copy converted to RGB.
        * ``np.ndarray`` — assumed to already be an HWC RGB image. 2D arrays
          are broadcast across 3 channels; 4-channel arrays are truncated to
          RGB; float arrays in [0, 1] are scaled to uint8.
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape {arr.shape}.")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, got shape {arr.shape}.")
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    if isinstance(image, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))

    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(image) as response:
                data = response.read()
            return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        return np.asarray(Image.open(image).convert("RGB"))

    raise TypeError(f"Unsupported image input type: {type(image).__name__}.")


def normalize_image(
    x,
    mean: Sequence[float],
    std: Sequence[float],
    data_format: Optional[str] = None,
):
    """Normalize a tensor by ``(x - mean) / std`` along the channel axis.

    Works for rank-3 (unbatched) and rank-4 (batched) tensors.
    """
    data_format = get_data_format(data_format)
    mean = ops.convert_to_tensor(mean, dtype="float32")
    std = ops.convert_to_tensor(std, dtype="float32")

    rank = len(x.shape)
    if data_format == "channels_first":
        shape = (1, -1, 1, 1) if rank == 4 else (-1, 1, 1)
    else:
        shape = (1, 1, 1, -1) if rank == 4 else (1, 1, -1)

    mean = ops.reshape(mean, shape)
    std = ops.reshape(std, shape)
    return (x - mean) / std


def _as_image_list(images: BatchImageInput) -> List:
    """Normalize the ``images`` argument of :func:`preprocess_image` to a list.

    * ``list`` / ``tuple`` -> returned as a list unchanged.
    * ``np.ndarray`` with 4 dims -> iterated along axis 0.
    * Anything else (single path / bytes / PIL / 2-D or 3-D ndarray) -> wrapped
      in a one-element list.
    """
    if isinstance(images, (list, tuple)):
        return list(images)
    if isinstance(images, np.ndarray) and images.ndim == 4:
        return [images[i] for i in range(images.shape[0])]
    return [images]


def preprocess_image(
    images: BatchImageInput,
    target_size: SizeLike,
    image_mean: Optional[Sequence[float]] = None,
    image_std: Optional[Sequence[float]] = None,
    rescale: bool = True,
    interpolation: str = "bilinear",
    antialias: bool = True,
    data_format: Optional[str] = None,
):
    """One-shot batched preprocessing pipeline.

    Runs: load -> resize -> (optional) rescale to [0, 1] -> (optional)
    normalize by mean/std -> transpose to the requested data format.

    Accepts either a single image (any type :func:`load_image` understands)
    or a list / tuple / 4-D ndarray batch of images. The return shape is
    consistent across both forms: the tensor is always rank-4 with an
    ``N`` leading dim, and ``original_sizes`` is always a list of
    ``(h, w)`` tuples with ``len == N``. Single-image callers still get
    ``N == 1``.

    Args:
        images: One image or a sequence of images. See :func:`load_image`
            for accepted element types.
        target_size: Either an ``int`` (square) or a ``(H, W)`` tuple.
            All images are resized to this size before stacking.
        image_mean: Per-channel mean. If ``None``, skips normalization.
        image_std: Per-channel std. Required when ``image_mean`` is given.
        rescale: Divide pixel values by 255 before normalization.
        interpolation: Passed to ``keras.ops.image.resize``.
        antialias: Passed to ``keras.ops.image.resize``.
        data_format: Output data format. ``None`` uses the global setting.

    Returns:
        Tuple ``(tensor, original_sizes, target_hw, data_format)``:

        * ``tensor`` — rank-4 batched tensor shaped ``(N, H, W, C)`` or
          ``(N, C, H, W)`` depending on ``data_format``.
        * ``original_sizes`` — ``[(h, w), ...]`` of length ``N``, in input
          order. Useful for mapping model outputs back to source resolution.
        * ``target_hw`` — ``(target_h, target_w)`` tuple.
        * ``data_format`` — the resolved data format string.
    """
    data_format = get_data_format(data_format)

    items = _as_image_list(images)
    if not items:
        raise ValueError("`images` must contain at least one image.")

    loaded = [load_image(img) for img in items]
    original_sizes: List[Tuple[int, int]] = [
        (int(arr.shape[0]), int(arr.shape[1])) for arr in loaded
    ]

    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    per_image = []
    for arr in loaded:
        t = ops.convert_to_tensor(arr, dtype="float32")
        t = ops.expand_dims(t, axis=0)
        t = ops.image.resize(
            t,
            size=(target_h, target_w),
            interpolation=interpolation,
            antialias=antialias,
            data_format="channels_last",
        )
        per_image.append(t)

    x = ops.concatenate(per_image, axis=0)

    if rescale:
        x = x / 255.0

    if image_mean is not None:
        if image_std is None:
            raise ValueError("image_std must be provided when image_mean is set.")
        x = normalize_image(x, image_mean, image_std, data_format="channels_last")

    if data_format == "channels_first":
        x = ops.transpose(x, (0, 3, 1, 2))

    return x, original_sizes, (target_h, target_w), data_format
