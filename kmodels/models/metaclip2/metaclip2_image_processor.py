import keras

from kmodels.models.clip.clip_image_processor import CLIPImageProcessor


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2ImageProcessor(CLIPImageProcessor):
    """Image processor for MetaCLIP 2.

    Identical preprocessing to OpenAI CLIP (same OpenAI ImageNet-ish mean/std,
    bicubic resize with shortest-edge scaling, center crop). MetaCLIP 2 shares
    the full image pipeline with CLIP; only the text tokenizer differs.
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
        **kwargs,
    ):
        super().__init__(
            image_resolution=image_resolution,
            mean=list(mean),
            std=list(std),
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
            data_format=data_format,
            **kwargs,
        )
