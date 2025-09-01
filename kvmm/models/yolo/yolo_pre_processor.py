from typing import Any, Dict, List, Union

import keras
from keras import ops

from kvmm.layers import Letterbox


@keras.saving.register_keras_serializable(package="kvmm")
class YoloPreProcessor(keras.layers.Layer):
    """
    Image preprocessor for YOLO models with letterbox resizing.

    This processor handles preprocessing steps for images to be used with YOLO models,
    including letterbox resizing (maintaining aspect ratio with padding), normalization,
    and format conversion.

    This processor uses the existing Letterbox layer which provides:
    - Maintains the original aspect ratio of the image
    - Resizes the image to fit within the target dimensions
    - Pads the remaining space with a constant color (typically [114, 114, 114] for YOLO)
    - Preserves object proportions for better detection accuracy
    - Optional stride-aware padding for network compatibility

    Attributes:
        image_size (int): Target size for the processed images (width and height).
        mean (keras.ops.Tensor): Mean values for RGB channels used in normalization.
        std (keras.ops.Tensor): Standard deviation values for RGB channels used in normalization.
        letterbox_color (List[int]): RGB color values for letterbox padding (typically [114, 114, 114] for YOLO).
        do_normalize (bool): Whether to normalize the image using mean and std values.
        do_letterbox (bool): Whether to apply letterbox resizing.
        letterbox_auto (bool): Whether to auto-adjust padding for stride compatibility.
        letterbox_stride (int): Stride value for automatic padding adjustment.

    Examples:
        Basic usage with an image tensor:
        ```python
        import keras
        from keras import ops

        # Create the processor
        processor = YoloPreProcessor(image_size=640)

        # Process a single image
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        result = processor(image_array)
        processed_image = result["images"]  # Shape: (1, 640, 640, 3)

        # Process a batch of images
        batch_size = 4
        random_images = ops.random.uniform((batch_size, 480, 640, 3))
        result = processor(random_images)
        processed_batch = result["images"]  # Shape: (4, 640, 640, 3)
        ```

        Process images from file paths:
        ```python
        # Process a single image path
        result = processor(image_paths="path/to/image.jpg")
        processed_image = result["images"]  # Shape: (1, 640, 640, 3)

        # Process multiple image paths
        image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
        result = processor(image_paths=image_paths)
        processed_images = result["images"]  # Shape: (2, 640, 640, 3)
        ```

        Custom processing configuration:
        ```python
        # Create processor with custom settings
        custom_processor = YoloPreProcessor(
            image_size=1024,  # Higher resolution
            letterbox_color=[128, 128, 128],  # Different padding color
            do_normalize=False,  # Skip normalization
        )

        # Use with YOLO model
        yolo_model = YoloV5s()
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        processed = custom_processor(image_array)
        detections = yolo_model(processed["images"])
        ```

        Integration with data augmentation:
        ```python
        # Define augmentation layer
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomBrightness(0.1),
            keras.layers.RandomContrast(0.1),
        ])

        # Apply augmentation before YOLO processing
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        image_array = image_array / 255.0  # Normalize to [0,1]

        # Augment and add batch dimension
        augmented = augmentation(ops.expand_dims(image_array, 0))

        # Process augmented image
        processor = YoloPreProcessor()
        result = processor(augmented)
        processed_image = result["images"]
        ```
    """

    def __init__(
        self,
        image_size: int = 640,
        mean: List[float] = [0.0, 0.0, 0.0],
        std: List[float] = [1.0, 1.0, 1.0],
        letterbox_color: List[int] = [114, 114, 114],
        do_normalize: bool = True,
        do_letterbox: bool = True,
        letterbox_auto: bool = True,
        letterbox_stride: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.mean = ops.array(mean, dtype="float32")
        self.std = ops.array(std, dtype="float32")
        self.letterbox_color = letterbox_color
        self.do_normalize = do_normalize
        self.do_letterbox = do_letterbox
        self.letterbox_auto = letterbox_auto
        self.letterbox_stride = letterbox_stride

        if self.do_letterbox:
            self.letterbox_layer = Letterbox(
                new_shape=(image_size, image_size),
                color=letterbox_color,
                auto=letterbox_auto,
                stride=letterbox_stride,
            )

    def preprocess(self, image: Any) -> Any:
        shape = ops.shape(image)
        num_channels = shape[-1]

        if num_channels == 1:
            image = ops.repeat(image, 3, axis=-1)
        elif num_channels == 4:
            image = image[..., :3]
        elif num_channels == 3:
            pass
        else:
            raise ValueError(f"Unsupported number of image channels: {num_channels}")

        image = ops.cast(image, "float32")
        image = ops.where(ops.greater(ops.max(image), 1.0), image / 255.0, image)

        if self.do_letterbox:
            image, _, _ = self.letterbox_layer(image)

        if self.do_normalize:
            image = (image - self.mean) / self.std

        return image

    def process_path(self, image_path: str) -> Any:
        image = keras.utils.load_img(image_path)
        image = keras.utils.img_to_array(image)
        return self.preprocess(image)

    def call(
        self,
        inputs: Any = None,
        image_paths: Union[str, List[str]] = None,
    ) -> Dict[str, Any]:
        if image_paths is not None:
            if inputs is not None:
                raise ValueError("Cannot specify both 'inputs' and 'image_paths'")

            if isinstance(image_paths, str):
                processed_image = self.process_path(image_paths)
                return {"images": ops.expand_dims(processed_image, axis=0)}
            else:
                if len(image_paths) == 0:
                    raise ValueError("image_paths list cannot be empty")

                processed_images = []
                for path in image_paths:
                    processed_images.append(self.process_path(path))
                return {"images": ops.stack(processed_images)}

        if inputs is None:
            raise ValueError("Must provide either 'inputs' or 'image_paths'")

        if len(ops.shape(inputs)) == 3:
            processed_image = self.preprocess(inputs)
            return {"images": ops.expand_dims(processed_image, axis=0)}

        elif len(ops.shape(inputs)) == 4:
            processed_images = ops.vectorized_map(self.preprocess, inputs)
            return {"images": processed_images}

        else:
            raise ValueError(
                f"Input images must have 3 dimensions (H, W, C) or 4 dimensions (B, H, W, C), "
                f"got shape: {ops.shape(inputs)}"
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "mean": self.mean.tolist()
                if hasattr(self.mean, "tolist")
                else self.mean,
                "std": self.std.tolist() if hasattr(self.std, "tolist") else self.std,
                "letterbox_color": self.letterbox_color,
                "do_normalize": self.do_normalize,
                "do_letterbox": self.do_letterbox,
                "letterbox_auto": self.letterbox_auto,
                "letterbox_stride": self.letterbox_stride,
            }
        )
        return config
