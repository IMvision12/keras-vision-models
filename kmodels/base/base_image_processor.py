from typing import Optional

import keras


class BaseImageProcessor(keras.layers.Layer):
    """Base class for kmodels image preprocessors.

    Carries the kwargs that nearly every image preprocessor in kmodels
    duplicates today — target resolution, normalization mean / std, and
    the do_resize / do_center_crop / do_normalize toggles. Subclasses
    implement ``call(images)`` and may add their own kwargs (e.g.
    interpolation mode, patch size for ViT-style models).
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        do_resize: bool = True,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_resolution = image_resolution
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.data_format = data_format or keras.backend.image_data_format()

    def call(self, images):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(images)`."
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_resolution": self.image_resolution,
                "mean": list(self.mean),
                "std": list(self.std),
                "do_resize": self.do_resize,
                "do_center_crop": self.do_center_crop,
                "do_normalize": self.do_normalize,
                "data_format": self.data_format,
            }
        )
        return config
