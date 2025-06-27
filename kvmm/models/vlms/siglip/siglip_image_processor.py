from typing import Any, List, Union

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="kvmm")
class SigLIPImageProcessor(keras.layers.Layer):
    """
    Image processor for SigLIP (Sigmoid Loss for Language Image Pre-training) models.
    This processor handles various preprocessing steps for images to be used with SigLIP models,
    including resizing, center cropping, and normalization.

    Attributes:
        image_resolution (int): Target resolution for the processed images.
        mean (keras.ops.Tensor): Mean values for RGB channels used in normalization.
        std (keras.ops.Tensor): Standard deviation values for RGB channels used in normalization.
        do_center_crop (bool): Whether to perform center cropping.
        do_normalize (bool): Whether to normalize the image using mean and std values.
        do_resize (bool): Whether to resize the image to the target resolution.

    Examples:
        Basic usage with an image tensor:
        ```python
        import keras
        from keras import ops

        # Create the processor
        processor = SigLIPImageProcessor(image_resolution=224)

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

        Process images from file paths:
        ```python
        # Process a single image path
        result = processor(image_paths="path/to/image.jpg")
        processed_image = result["images"]  # Shape: (1, 224, 224, 3)

        # Process multiple image paths
        image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
        result = processor(image_paths=image_paths)
        processed_images = result["images"]  # Shape: (3, 224, 224, 3)
        ```

        Custom processing configuration:
        ```python
        # Create processor with custom settings
        custom_processor = SigLIPImageProcessor(
            image_resolution=384,  # Higher resolution
            mean=[0.5, 0.5, 0.5],  # Different normalization
            std=[0.5, 0.5, 0.5],
            do_center_crop=False,  # Skip center cropping
        )

        # Use with SigLIP model
        siglip_model = load_siglip_model()
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        processed = custom_processor(image_array)
        image_features = siglip_model(processed)
        ```

        Integration with image augmentation:
        ```python
        # Define augmentation layer
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ])

        # Apply augmentation before SigLIP processing
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        image_array = image_array / 255.0  # Normalize to [0,1]

        # Augment and add batch dimension
        augmented = augmentation(ops.expand_dims(image_array, 0))

        # Process augmented image
        processor = SigLIPImageProcessor()
        result = processor(augmented)
        processed_image = result["images"]
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
        super().__init__(**kwargs)
        self.image_resolution = image_resolution
        self.mean = ops.array(mean, dtype="float32")
        self.std = ops.array(std, dtype="float32")
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.do_resize = do_resize

    def preprocess(self, image: Any) -> Any:
        shape = ops.shape(image)
        num_channels = shape[-1]

        if num_channels == 1:
            # Convert grayscale to RGB by repeating the single channel
            image = ops.repeat(image, 3, axis=-1)
        elif num_channels == 4:
            # Convert RGBA to RGB by dropping the alpha channel
            image = image[..., :3]
        elif num_channels == 3:
            pass
        else:
            raise ValueError(f"Unsupported number of image channels: {num_channels}")

        image = ops.cast(image, "float32")
        image = ops.where(ops.greater(ops.max(image), 1.0), image / 255.0, image)

        if self.do_resize:
            image = ops.image.resize(
                image,
                (self.image_resolution, self.image_resolution),
                interpolation="bicubic",
                antialias=True,
            )

        if self.do_center_crop:
            image = self._center_crop(image)

        if self.do_normalize:
            image = (image - self.mean) / self.std

        return image

    def _center_crop(self, image: Any) -> Any:
        shape = ops.shape(image)
        height, width = shape[0], shape[1]
        target_size = self.image_resolution

        scale_h = ops.cast(target_size, "float32") / ops.cast(height, "float32")
        scale_w = ops.cast(target_size, "float32") / ops.cast(width, "float32")
        scale = ops.maximum(scale_h, scale_w)

        new_height = ops.cast(ops.cast(height, "float32") * scale, "int32")
        new_width = ops.cast(ops.cast(width, "float32") * scale, "int32")

        new_height = ops.maximum(new_height, target_size)
        new_width = ops.maximum(new_width, target_size)

        resized_image = ops.image.resize(
            image,
            (new_height, new_width),
            interpolation="bicubic",
            antialias=True,
        )

        y_start = (new_height - target_size) // 2
        x_start = (new_width - target_size) // 2

        cropped_image = ops.slice(
            resized_image,
            [y_start, x_start, 0],
            [target_size, target_size, 3],
        )

        return cropped_image

    def process_path(self, image_path: str) -> Any:
        image = keras.utils.load_img(image_path)
        image = keras.utils.img_to_array(image)
        return self.preprocess(image)

    def call(
        self,
        inputs: Any = None,
        image_paths: Union[str, List[str]] = None,
    ) -> Any:
        if image_paths is not None:
            if inputs is not None:
                raise ValueError("Cannot specify both 'inputs' and 'image_paths'")

            if isinstance(image_paths, str):
                processed_image = self.process_path(image_paths)
                return ops.expand_dims(processed_image, axis=0)
            else:
                if len(image_paths) == 0:
                    raise ValueError("image_paths list cannot be empty")

                processed_images = []
                for path in image_paths:
                    processed_images.append(self.process_path(path))
                return ops.stack(processed_images)

        if inputs is None:
            raise ValueError("Must provide either 'inputs' or 'image_paths'")

        if len(ops.shape(inputs)) == 3:
            processed_image = self.preprocess(inputs)
            return ops.expand_dims(processed_image, axis=0)

        elif len(ops.shape(inputs)) == 4:
            processed_images = ops.vectorized_map(self.preprocess, inputs)
            return processed_images

        else:
            raise ValueError(
                f"Input images must have 3 dimensions (H, W, C) or 4 dimensions (B, H, W, C), "
                f"got shape: {ops.shape(inputs)}"
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_resolution": self.image_resolution,
                "mean": self.mean.tolist()
                if hasattr(self.mean, "tolist")
                else self.mean,
                "std": self.std.tolist() if hasattr(self.std, "tolist") else self.std,
                "do_center_crop": self.do_center_crop,
                "do_normalize": self.do_normalize,
                "do_resize": self.do_resize,
            }
        )
        return config
