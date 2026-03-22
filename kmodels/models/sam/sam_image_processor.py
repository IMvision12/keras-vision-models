from typing import Dict, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_preprocess_shape(
    old_h: int, old_w: int, long_side_length: int
) -> Tuple[int, int]:
    """Compute resized (h, w) that scales the longest side to ``long_side_length``."""
    scale = long_side_length / max(old_h, old_w)
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)
    return new_h, new_w


def SAMImageProcessor(
    image: Union[str, np.ndarray, "Image.Image"],
    target_length: int = 1024,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
) -> Dict[str, "keras.KerasTensor"]:
    """Preprocess an image for SAM inference.

    Resizes the image so its longest side equals ``target_length``, applies
    ImageNet normalization, and pads to a square. Also prepares default prompt
    inputs (empty points, labels, boxes, masks) so the model can run with just
    an image.

    Args:
        image: Input image as a file path, numpy array, or PIL Image.
        target_length: Target size for the longest side (default 1024).
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.

    Returns:
        Dict with keys:
            - ``"pixel_values"``: ``(1, target_length, target_length, 3)``
            - ``"input_points"``: ``(1, 1, 0, 2)`` empty placeholder
            - ``"input_labels"``: ``(1, 1, 0)`` empty placeholder
            - ``"input_boxes"``: ``(1, 0, 4)`` empty placeholder
            - ``"input_masks"``: ``(1, 0, target_length, target_length, 1)``
            - ``"original_size"``: ``(orig_h, orig_w)``
            - ``"reshaped_size"``: ``(new_h, new_w)``

    Example:
        ```python
        from kmodels.models.sam import SAM_ViT_Huge, SAMImageProcessor

        model = SAM_ViT_Huge(weights="sa1b")
        inputs = SAMImageProcessor("photo.jpg")
        outputs = model(inputs)
        ```
    """
    if image_mean is None:
        image_mean = IMAGENET_MEAN
    if image_std is None:
        image_std = IMAGENET_STD

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        image = np.array(image, dtype=np.float32)
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.float32)
        if image.ndim == 4:
            image = image[0]
    else:
        raise TypeError("Input must be a file path (str), numpy array, or PIL Image.")

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

    orig_h, orig_w = image.shape[:2]
    new_h, new_w = _get_preprocess_shape(orig_h, orig_w, target_length)

    image = keras.ops.convert_to_tensor(image, dtype="float32")
    image = keras.ops.expand_dims(image, axis=0)
    image = keras.ops.image.resize(image, (new_h, new_w), interpolation="bilinear")

    image = image / 255.0

    mean = keras.ops.reshape(
        keras.ops.convert_to_tensor(image_mean, dtype="float32"), (1, 1, 1, 3)
    )
    std = keras.ops.reshape(
        keras.ops.convert_to_tensor(image_std, dtype="float32"), (1, 1, 1, 3)
    )
    image = (image - mean) / std

    padded = keras.ops.zeros((1, target_length, target_length, 3), dtype="float32")
    padded = keras.ops.slice_update(padded, (0, 0, 0, 0), image)

    empty_points = keras.ops.zeros((1, 1, 0, 2), dtype="float32")
    empty_labels = keras.ops.zeros((1, 1, 0), dtype="int32")
    empty_boxes = keras.ops.zeros((1, 0, 4), dtype="float32")

    return {
        "pixel_values": padded,
        "input_points": empty_points,
        "input_labels": empty_labels,
        "input_boxes": empty_boxes,
        "original_size": (orig_h, orig_w),
        "reshaped_size": (new_h, new_w),
    }


def SAMImageProcessorWithPrompts(
    image: Union[str, np.ndarray, "Image.Image"],
    input_points: Optional[np.ndarray] = None,
    input_labels: Optional[np.ndarray] = None,
    input_boxes: Optional[np.ndarray] = None,
    target_length: int = 1024,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
) -> Dict[str, "keras.KerasTensor"]:
    """Preprocess an image and prompts for SAM inference.

    Extends ``SAMImageProcessor`` by also encoding prompt inputs (points,
    labels, boxes) into the correct tensor format expected by the model.

    Args:
        image: Input image as a file path, numpy array, or PIL Image.
        input_points: Point prompts of shape ``(num_point_sets, num_points, 2)``
            in ``(x, y)`` pixel coordinates.  Wrapped with a batch dim automatically.
        input_labels: Point labels matching ``input_points`` shape
            ``(num_point_sets, num_points)``.  ``1`` = foreground, ``0`` = background.
        input_boxes: Box prompts of shape ``(num_boxes, 4)`` as
            ``(x1, y1, x2, y2)`` pixel coordinates.
        target_length: Target size for the longest side (default 1024).
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.

    Returns:
        Dict with keys matching ``SAMImageProcessor`` output, but with
        populated prompt tensors.

    Example:
        ```python
        from kmodels.models.sam import SAM_ViT_Huge
        from kmodels.models.sam.sam_image_processor import SAMImageProcessorWithPrompts

        model = SAM_ViT_Huge(weights="sa1b")
        inputs = SAMImageProcessorWithPrompts(
            "photo.jpg",
            input_points=np.array([[[450, 600]]]),
            input_labels=np.array([[1]]),
        )
        outputs = model(inputs)
        ```
    """
    result = SAMImageProcessor(image, target_length, image_mean, image_std)

    if input_points is not None:
        points = np.array(input_points, dtype=np.float32)
        if points.ndim == 2:
            points = points[np.newaxis, :]
        if points.ndim == 3:
            points = points[np.newaxis, :]
        result["input_points"] = keras.ops.convert_to_tensor(points, dtype="float32")

    if input_labels is not None:
        labels = np.array(input_labels, dtype=np.int32)
        if labels.ndim == 1:
            labels = labels[np.newaxis, :]
        if labels.ndim == 2:
            labels = labels[np.newaxis, :]
        result["input_labels"] = keras.ops.convert_to_tensor(labels, dtype="int32")

    if input_boxes is not None:
        boxes = np.array(input_boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes[np.newaxis, :]
        if boxes.ndim == 2:
            boxes = boxes[np.newaxis, :]
        result["input_boxes"] = keras.ops.convert_to_tensor(boxes, dtype="float32")

    return result


def SAMPostProcessMasks(
    pred_masks: "keras.KerasTensor",
    original_size: Tuple[int, int],
    reshaped_size: Tuple[int, int],
    target_length: int = 1024,
) -> np.ndarray:
    """Post-process predicted masks to original image resolution.

    Upsamples low-resolution masks to the model's input size, crops out the
    padding region, and resizes to the original image dimensions.

    Args:
        pred_masks: Predicted masks of shape
            ``(batch, point_batch, num_masks, mask_h, mask_w)``.
        original_size: Original image ``(height, width)``.
        reshaped_size: Resized image ``(height, width)`` before padding.
        target_length: Model input resolution (default 1024).

    Returns:
        Numpy array of masks with shape
        ``(batch, point_batch, num_masks, orig_h, orig_w)``.

    Example:
        ```python
        masks_np = SAMPostProcessMasks(
            outputs["pred_masks"],
            original_size=inputs["original_size"],
            reshaped_size=inputs["reshaped_size"],
        )
        ```
    """
    pred_masks = keras.ops.convert_to_tensor(pred_masks, dtype="float32")

    batch = keras.ops.shape(pred_masks)[0]
    point_batch = keras.ops.shape(pred_masks)[1]
    num_masks = keras.ops.shape(pred_masks)[2]
    mask_h = keras.ops.shape(pred_masks)[3]
    mask_w = keras.ops.shape(pred_masks)[4]

    masks_4d = keras.ops.reshape(
        pred_masks, (batch * point_batch * num_masks, mask_h, mask_w, 1)
    )

    masks_upsampled = keras.ops.image.resize(
        masks_4d, (target_length, target_length), interpolation="bilinear"
    )

    new_h, new_w = reshaped_size
    masks_cropped = masks_upsampled[:, :new_h, :new_w, :]

    orig_h, orig_w = original_size
    masks_final = keras.ops.image.resize(
        masks_cropped, (orig_h, orig_w), interpolation="bilinear"
    )

    masks_final = keras.ops.reshape(
        masks_final, (batch, point_batch, num_masks, orig_h, orig_w)
    )

    return keras.ops.convert_to_numpy(masks_final)
