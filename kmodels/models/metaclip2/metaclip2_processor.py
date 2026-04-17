import keras

from .metaclip2_image_processor import MetaClip2ImageProcessor
from .metaclip2_tokenizer import MetaClip2Tokenizer


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2Processor(keras.layers.Layer):
    """Combined image + text processor for MetaCLIP 2.

    Wraps :class:`MetaClip2ImageProcessor` and :class:`MetaClip2Tokenizer`.
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format=None,
        sentencepiece_model_file: str = None,
        context_length: int = 77,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_processor = MetaClip2ImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
            data_format=data_format,
        )
        self.tokenizer = MetaClip2Tokenizer(
            sentencepiece_model_file=sentencepiece_model_file,
            context_length=context_length,
        )

    def call(self, text=None, images=None, image_paths=None):
        if text is None and images is None and image_paths is None:
            raise ValueError(
                "At least one of 'text', 'images', or 'image_paths' must be provided"
            )
        if images is not None and image_paths is not None:
            raise ValueError("Cannot specify both 'images' and 'image_paths'")

        encoding = {}
        if text is not None:
            encoding.update(self.tokenizer(inputs=text))
        if images is not None:
            encoding.update(self.image_processor(inputs=images))
        if image_paths is not None:
            encoding.update(self.image_processor(image_paths=image_paths))
        return encoding
