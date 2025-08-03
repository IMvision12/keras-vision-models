from typing import List

import keras

from kvmm.models.siglip.siglip_image_processor import SigLIPImageProcessor


@keras.saving.register_keras_serializable(package="kvmm")
class SigLIP2ImageProcessor(SigLIPImageProcessor):
    """
    Image processor for SigLIP2 models, inheriting all functionality from SigLIPImageProcessor.

    This processor inherits all preprocessing capabilities from SigLIPImageProcessor,
    including resizing, center cropping, and normalization. It can be extended with
    additional functionality specific to SigLIP2 models.

    Inherits all attributes from SigLIPImageProcessor:
        image_resolution (int): Target resolution for the processed images.
        mean (keras.ops.Tensor): Mean values for RGB channels used in normalization.
        std (keras.ops.Tensor): Standard deviation values for RGB channels used in normalization.
        do_center_crop (bool): Whether to perform center cropping.
        do_normalize (bool): Whether to normalize the image using mean and std values.
        do_resize (bool): Whether to resize the image to the target resolution.

    Examples:
        Basic usage (same as SigLIPImageProcessor):
        ```python
        import keras
        from keras import ops

        # Create the processor
        processor = SigLIP2ImageProcessor(image_resolution=224)

        # Process a single image
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        result = processor(image_array)
        processed_image = result["images"]  # Shape: (1, 224, 224, 3)

        # Process a batch of images
        batch_size = 4
        random_images = ops.random.uniform((batch_size, 256, 256, 3))
        result = processor(random_images)
        processed_batch = result["images"]  # Shape: (4, 224, 224, 3)
        ```

        Custom configuration for SigLIP2:
        ```python
        # Create processor with custom settings
        custom_processor = SigLIP2ImageProcessor(
            image_resolution=384,  # Higher resolution for SigLIP2
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],   # ImageNet stds
            do_center_crop=True,
        )
        ```
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        **kwargs,
    ):
        super().__init__(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
            **kwargs,
        )
