from typing import Dict, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def Sam2VideoImageProcessor(
    image: Union[str, np.ndarray, "Image.Image"],
    target_length: int = 1024,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
) -> Dict[str, "keras.KerasTensor"]:
    """Preprocess a single frame for Sam2Video inference.

    Resizes the frame to ``(target_length, target_length)`` with
    antialiased bilinear interpolation (no aspect-ratio preservation,
    matching HF ``Sam2VideoProcessor``), applies ImageNet normalization,
    and prepares default prompt placeholders so the model can run with
    just a frame.

    Args:
        image: Input frame as a file path, NumPy array ``(H, W, 3)``,
            or PIL Image.
        target_length: Target spatial size for both axes. Defaults to
            ``1024``.
        image_mean: Per-channel mean for normalization. Defaults to
            ImageNet statistics.
        image_std: Per-channel std for normalization. Defaults to
            ImageNet statistics.

    Returns:
        Dict with keys:
            - ``"pixel_values"``: ``(1, target_length, target_length, 3)``
            - ``"input_points"``: ``(1, 1, 0, 2)`` empty placeholder
            - ``"input_labels"``: ``(1, 1, 0)`` empty placeholder
            - ``"original_size"``: ``(orig_h, orig_w)``
            - ``"reshaped_size"``: ``(target_length, target_length)``

    Example:
        ```python
        from kmodels.models.sam2_video import Sam2VideoSmall, Sam2VideoImageProcessor

        model = Sam2VideoSmall(weights="sav")
        inputs = Sam2VideoImageProcessor("frame.jpg")
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

    image = keras.ops.convert_to_tensor(image, dtype="float32")
    image = keras.ops.expand_dims(image, axis=0)
    image = keras.ops.image.resize(
        image,
        (target_length, target_length),
        interpolation="bilinear",
        antialias=True,
    )

    image = image / 255.0

    mean = keras.ops.reshape(
        keras.ops.convert_to_tensor(image_mean, dtype="float32"), (1, 1, 1, 3)
    )
    std = keras.ops.reshape(
        keras.ops.convert_to_tensor(image_std, dtype="float32"), (1, 1, 1, 3)
    )
    image = (image - mean) / std

    empty_points = keras.ops.zeros((1, 1, 0, 2), dtype="float32")
    empty_labels = keras.ops.zeros((1, 1, 0), dtype="int32")

    return {
        "pixel_values": image,
        "input_points": empty_points,
        "input_labels": empty_labels,
        "original_size": (orig_h, orig_w),
        "reshaped_size": (target_length, target_length),
    }


def Sam2VideoImageProcessorWithPrompts(
    image: Union[str, np.ndarray, "Image.Image"],
    input_points: Optional[np.ndarray] = None,
    input_labels: Optional[np.ndarray] = None,
    target_length: int = 1024,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
) -> Dict[str, "keras.KerasTensor"]:
    """Preprocess a single frame and point prompts for Sam2Video inference.

    Extends :func:`Sam2VideoImageProcessor` by also encoding point
    prompts. Since SAM 2 Video stretches frames independently per
    axis, point coordinates are scaled per axis as well
    (``x_new = x * target / orig_w``, ``y_new = y * target / orig_h``).

    Args:
        image: Input frame as a file path, NumPy array, or PIL Image.
        input_points: Point prompts shaped
            ``(num_point_sets, num_points, 2)`` in ``(x, y)`` pixel
            coordinates of the original frame. Wrapped with a batch
            dim automatically.
        input_labels: Point labels matching ``input_points`` shape
            ``(num_point_sets, num_points)``. ``1`` = foreground,
            ``0`` = background.
        target_length: Target spatial size for both axes. Defaults to
            ``1024``.
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.

    Returns:
        Dict with keys matching :func:`Sam2VideoImageProcessor` output,
        but with populated prompt tensors.

    Example:
        ```python
        import numpy as np
        from kmodels.models.sam2_video import (
            Sam2VideoSmall, Sam2VideoImageProcessorWithPrompts,
        )

        model = Sam2VideoSmall(weights="sav")
        inputs = Sam2VideoImageProcessorWithPrompts(
            "frame.jpg",
            input_points=np.array([[[450, 600]]]),
            input_labels=np.array([[1]]),
        )
        outputs = model(inputs)
        ```
    """
    result = Sam2VideoImageProcessor(image, target_length, image_mean, image_std)

    orig_h, orig_w = result["original_size"]
    scale_x = target_length / float(orig_w)
    scale_y = target_length / float(orig_h)

    if input_points is not None:
        points = np.array(input_points, dtype=np.float64)
        points[..., 0] = points[..., 0] * scale_x
        points[..., 1] = points[..., 1] * scale_y
        points = keras.ops.convert_to_tensor(points, dtype="float32")
        if keras.ops.ndim(points) == 2:
            points = keras.ops.expand_dims(points, axis=0)
        if keras.ops.ndim(points) == 3:
            points = keras.ops.expand_dims(points, axis=0)
        result["input_points"] = points

    if input_labels is not None:
        labels = keras.ops.convert_to_tensor(input_labels, dtype="int32")
        if keras.ops.ndim(labels) == 1:
            labels = keras.ops.expand_dims(labels, axis=0)
        if keras.ops.ndim(labels) == 2:
            labels = keras.ops.expand_dims(labels, axis=0)
        result["input_labels"] = labels

    return result


def Sam2VideoPostProcessMasks(
    pred_masks: "keras.KerasTensor",
    original_size: Tuple[int, int],
    target_length: int = 1024,
) -> "keras.KerasTensor":
    """Resize predicted Sam2Video masks back to original frame resolution.

    Since SAM 2 Video stretches frames directly to the target square
    (no aspect-ratio preservation, no padding), reversing it is just a
    bilinear resize from the decoder mask resolution to
    ``original_size``.

    Args:
        pred_masks: Predicted masks of shape
            ``(batch, point_batch, num_masks, mask_h, mask_w)``.
        original_size: Original frame ``(height, width)``.
        target_length: Model input resolution. Unused by this
            implementation but kept for API parity with
            :func:`SAMPostProcessMasks`. Defaults to ``1024``.

    Returns:
        Keras tensor of masks shaped
        ``(batch, point_batch, num_masks, orig_h, orig_w)``.

    Example:
        ```python
        from kmodels.models.sam2_video import Sam2VideoPostProcessMasks

        masks = Sam2VideoPostProcessMasks(
            outputs["pred_masks"],
            original_size=inputs["original_size"],
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

    orig_h, orig_w = original_size
    masks_final = keras.ops.image.resize(
        masks_4d,
        (orig_h, orig_w),
        interpolation="bilinear",
        antialias=True,
    )

    masks_final = keras.ops.reshape(
        masks_final, (batch, point_batch, num_masks, orig_h, orig_w)
    )

    return masks_final
