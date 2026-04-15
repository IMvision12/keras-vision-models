"""Preprocessing and postprocessing for DeepLabV3 semantic segmentation."""

from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from kmodels.utils.image import preprocess_image

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


def DeepLabV3ImageProcessor(
    image: Union[str, np.ndarray, "Image.Image"],
    size: Optional[Dict[str, int]] = None,
    resample: str = "bilinear",
    do_rescale: bool = True,
    rescale_factor: float = 1 / 255,
    do_normalize: bool = True,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    return_tensor: bool = True,
) -> Union["keras.KerasTensor", np.ndarray]:
    """Preprocess an image for DeepLabV3 inference.

    Handles loading, resizing, rescaling, and ImageNet normalization to match
    the preprocessing used during DeepLabV3 training (torchvision convention).

    Args:
        image: Input image as a file path, numpy array, or PIL Image.
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 520, "width": 520}``.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``).
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_normalize: Whether to apply ImageNet normalization.
        image_mean: Per-channel mean for normalization.
            Default: ``(0.485, 0.456, 0.406)``.
        image_std: Per-channel std for normalization.
            Default: ``(0.229, 0.224, 0.225)``.
        return_tensor: If True return a Keras tensor, otherwise numpy array.

    Returns:
        Preprocessed image with shape ``(1, H, W, 3)`` ready for model input.

    Example:
        ```python
        from kmodels.models.deeplabv3 import DeepLabV3ImageProcessor, DeepLabV3ResNet50

        model = DeepLabV3ResNet50(weights="voc")
        img = DeepLabV3ImageProcessor("photo.jpg")
        output = model(img, training=False)
        ```
    """
    if size is None:
        size = {"height": 520, "width": 520}
    if image_mean is None:
        image_mean = (0.485, 0.456, 0.406)
    if image_std is None:
        image_std = (0.229, 0.224, 0.225)

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


def DeepLabV3PostProcessor(
    outputs: "keras.KerasTensor",
    target_size: Optional[Tuple[int, int]] = None,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Post-process raw DeepLabV3 outputs into semantic segmentation results.

    Takes the raw logits from DeepLabV3, computes the argmax class map,
    optionally resizes to the original image size, and maps class indices
    to human-readable names.

    Args:
        outputs: Raw model output tensor of shape ``(1, H, W, num_classes)``.
        target_size: Original image ``(height, width)`` for resizing the
            prediction mask. If ``None``, the mask is returned at model
            output resolution.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to Pascal VOC class names (21
            classes). Provide this when using a model fine-tuned on a
            custom dataset.

    Returns:
        Dict with:
            - ``"segmentation"``: Integer array of shape ``(H, W)`` with
              class indices.
            - ``"class_names"``: List of unique class names detected in the
              image.
            - ``"unique_classes"``: Array of unique class indices.

    Example:
        ```python
        from kmodels.models.deeplabv3 import (
            DeepLabV3ResNet50, DeepLabV3ImageProcessor, DeepLabV3PostProcessor,
        )

        model = DeepLabV3ResNet50(weights="voc")
        img = DeepLabV3ImageProcessor("photo.jpg")
        output = model(img, training=False)
        result = DeepLabV3PostProcessor(output, target_size=(orig_h, orig_w))
        print(result["class_names"])
        ```
    """
    _names = label_names if label_names is not None else VOC_CLASSES

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
