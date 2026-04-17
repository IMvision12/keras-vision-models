from typing import Any

import keras
import numpy as np
from keras import ops
from PIL import Image

from kmodels.models.clip.clip_image_processor import CLIPImageProcessor
from kmodels.utils.image import load_image


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2ImageProcessor(CLIPImageProcessor):
    """Image processor for MetaCLIP 2.

    Matches HF MetaCLIP 2's default config: **direct square resize** to
    ``image_resolution`` (no shortest-edge scaling + center crop), followed
    by rescale to ``[0, 1]`` and OpenAI-CLIP normalization. This is the
    main preprocessing difference from OpenAI CLIP itself, which uses
    shortest-edge=resolution + center crop.

    Resize uses PIL.BICUBIC to match HF's ``resample=3`` exactly.
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format=None,
        **kwargs,
    ):
        super().__init__(
            image_resolution=image_resolution,
            mean=list(mean),
            std=list(std),
            do_center_crop=False,
            do_normalize=do_normalize,
            do_resize=do_resize,
            data_format=data_format,
            **kwargs,
        )

    def process_path(self, image_path: str) -> Any:
        arr = load_image(image_path)
        if self.do_resize:
            pil = Image.fromarray(arr.astype(np.uint8))
            pil = pil.resize(
                (self.image_resolution, self.image_resolution), Image.BICUBIC
            )
            arr = np.array(pil)
        image = arr.astype(np.float32) * np.float32(1.0 / 255.0)
        image = ops.convert_to_tensor(image, dtype="float32")
        if self.do_normalize:
            image = (image - self.mean) / self.std
        return image
