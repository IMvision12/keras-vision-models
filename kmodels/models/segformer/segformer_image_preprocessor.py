from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from kmodels.utils.image import preprocess_image

ADE20K_CLASSES = [
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
]

CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def SegFormerImageProcessor(
    image: Union[str, np.ndarray, Image.Image, keras.KerasTensor],
    do_resize: bool = True,
    size: Dict[str, int] = None,
    resample: str = "bilinear",
    do_rescale: bool = True,
    rescale_factor: float = 1 / 255,
    do_normalize: bool = True,
    image_mean: Optional[Union[float, tuple]] = None,
    image_std: Optional[Union[float, tuple]] = None,
    return_tensor: bool = True,
) -> Union[keras.KerasTensor, np.ndarray]:
    """
    Comprehensive image preprocessing function for SegFormer model input using Keras ops.
    Implements functionality equivalent to HuggingFace's SegformerImageProcessor.

    Args:
        image: Input image (file path, numpy array, PIL Image, or Keras tensor)
        do_resize: Whether to resize the image
        size: Dict with 'height' and 'width' keys for target size (default: {height: 512, width: 512})
        resample: Interpolation method ('nearest', 'bilinear', 'bicubic')
        do_rescale: Whether to rescale pixel values
        rescale_factor: Factor to rescale pixel values by
        do_normalize: Whether to normalize with mean and std
        image_mean: RGB mean values for normalization (default: (0.485, 0.456, 0.406))
        image_std: RGB standard deviation values (default: (0.229, 0.224, 0.225))
        return_tensor: Whether to return a Keras tensor (True) or numpy array (False)

    Returns:
        Preprocessed image as Keras tensor or numpy array
    """
    if size is None:
        size = {"height": 512, "width": 512}
    if image_mean is None:
        image_mean = (0.485, 0.456, 0.406)
    if image_std is None:
        image_std = (0.229, 0.224, 0.225)

    if size["height"] <= 0 or size["width"] <= 0:
        raise ValueError("Size dimensions must be positive")
    if resample not in ["nearest", "bilinear", "bicubic"]:
        raise ValueError("Resample method must be 'nearest', 'bilinear', or 'bicubic'")
    if rescale_factor < 0:
        raise ValueError("Rescale factor must be non-negative")

    # Keras-tensor input keeps its original bespoke path (range validation +
    # eager scaling). Every other input type routes through the shared
    # `preprocess_image` helper.
    is_keras_tensor = (
        not isinstance(image, (str, np.ndarray, Image.Image))
        and hasattr(image, "shape")
        and hasattr(image, "dtype")
    )
    if is_keras_tensor:
        if len(image.shape) == 4:
            image = image[0]
        if len(image.shape) != 3:
            raise ValueError("Input tensor must have shape (H, W, C)")

        image_float = keras.ops.cast(image, dtype="float32")
        max_val_py = keras.ops.convert_to_numpy(keras.ops.max(image_float)).item()
        min_val_py = keras.ops.convert_to_numpy(keras.ops.min(image_float)).item()

        if max_val_py <= 1.0 and min_val_py >= 0.0:
            image = image_float * 255.0
        elif not (min_val_py >= 0 and max_val_py <= 255):
            raise ValueError("Tensor values must be in [0,1] or [0,255] range")
        else:
            image = image_float

        image = keras.ops.expand_dims(image, axis=0)
        if do_resize:
            target_size = (size["height"], size["width"])
            if image.shape[1:3] != target_size:
                image = keras.ops.image.resize(
                    image, size=target_size, interpolation=resample
                )
        if do_rescale:
            image = image * rescale_factor
        if do_normalize:
            mean = keras.ops.reshape(
                keras.ops.convert_to_tensor(image_mean, dtype="float32"), (1, 1, 1, 3)
            )
            std = keras.ops.reshape(
                keras.ops.convert_to_tensor(image_std, dtype="float32"), (1, 1, 1, 3)
            )
            image = (image - mean) / std
    else:
        image, _, _, _ = preprocess_image(
            image,
            target_size=(size["height"], size["width"]),
            image_mean=image_mean if do_normalize else None,
            image_std=image_std if do_normalize else None,
            rescale=do_rescale,
            interpolation=resample,
            antialias=False,
        )
        if do_rescale and rescale_factor != 1 / 255:
            image = image * (rescale_factor * 255)

    if not return_tensor:
        image = keras.ops.convert_to_numpy(image)

    return image


def SegFormerPostProcessor(
    outputs: "keras.KerasTensor",
    target_size: Optional[Tuple[int, int]] = None,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Post-process raw SegFormer outputs into semantic segmentation results.

    Takes the raw logits from SegFormer, computes the argmax class map,
    optionally resizes to the original image size, and maps class indices
    to human-readable names.

    Args:
        outputs: Raw model output tensor of shape ``(1, H, W, num_classes)``.
        target_size: Original image ``(height, width)`` for resizing the
            prediction mask. If ``None``, the mask is returned at model
            output resolution.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to ADE20K class names (150
            classes). Provide this when using a model fine-tuned on a
            custom dataset (e.g. Cityscapes names via
            ``CITYSCAPES_CLASSES``).

    Returns:
        Dict with:
            - ``"segmentation"``: Integer array of shape ``(H, W)`` with
              class indices.
            - ``"class_names"``: List of unique class names detected in the
              image.
            - ``"unique_classes"``: Array of unique class indices.

    Example:
        ```python
        from kmodels.models.segformer import (
            SegFormerB0, SegFormerImageProcessor, SegFormerPostProcessor,
        )

        model = SegFormerB0(weights="ade20k_512", input_shape=(512, 512, 3))
        img = SegFormerImageProcessor("photo.jpg")
        output = model(img, training=False)
        result = SegFormerPostProcessor(output, target_size=(orig_h, orig_w))
        print(result["class_names"])
        ```
    """
    _names = label_names if label_names is not None else ADE20K_CLASSES

    logits = keras.ops.convert_to_numpy(outputs)
    pred_mask = np.argmax(logits[0], axis=-1)  # (H, W)

    if target_size is not None:
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8)).resize(
                (target_size[1], target_size[0]), Image.NEAREST
            )
        )

    unique_classes = np.unique(pred_mask)
    class_names = [
        _names[c] if c < len(_names) else f"class_{c}" for c in unique_classes
    ]

    return {
        "segmentation": pred_mask,
        "class_names": class_names,
        "unique_classes": unique_classes,
    }
