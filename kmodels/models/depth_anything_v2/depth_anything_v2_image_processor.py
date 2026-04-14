from typing import Dict, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def DepthAnythingV2ImageProcessor(
    image: Union[str, np.ndarray, "Image.Image"],
    target_size: Union[int, Tuple[int, int]] = 518,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
) -> Dict[str, "keras.KerasTensor"]:
    """Preprocess a single image for DepthAnythingV2 inference.

    Resizes the image to ``target_size`` with bicubic interpolation
    (mirroring the HF ``DPTImageProcessor`` resample setting),
    rescales to ``[0, 1]``, and applies ImageNet normalization. Unlike
    HF ``DPTImageProcessor`` (which preserves aspect ratio and
    produces a variable-shape output), this processor stretches the
    image directly to the target size, so the shape must match the
    one the Keras model was built with.

    Both target dims should be multiples of the DINOv2 patch size
    (``14``). The pretrained ``518 x 518`` pos embeds are
    bilinearly interpolated to the model's grid on weight load, so a
    non-518 square (e.g. ``392``) or non-square pair
    (e.g. ``(392, 784)``) both work as long as the model was built
    with that same shape.

    Args:
        image: Input image as a file path, NumPy array ``(H, W, 3)``,
            or PIL Image.
        target_size: Target spatial size. Either a single ``int``
            (square output) or a ``(height, width)`` tuple. Defaults
            to ``518``.
        image_mean: Per-channel mean for normalization. Defaults to
            ImageNet statistics.
        image_std: Per-channel std for normalization. Defaults to
            ImageNet statistics.

    Returns:
        Dict with keys:
            - ``"pixel_values"``: ``(1, target_h, target_w, 3)``
              (or ``(1, 3, target_h, target_w)`` when
              ``keras.config.image_data_format() == "channels_first"``)
            - ``"original_size"``: ``(orig_h, orig_w)``
            - ``"reshaped_size"``: ``(target_h, target_w)``

    Example:
        ```python
        from kmodels.models.depth_anything_v2 import (
            DepthAnythingV2Small, DepthAnythingV2ImageProcessor,
        )

        model = DepthAnythingV2Small(
            input_shape=(392, 784, 3), weights="depth_anything"
        )
        inputs = DepthAnythingV2ImageProcessor(
            "photo.jpg", target_size=(392, 784)
        )
        depth = model(inputs["pixel_values"])
        ```
    """
    if isinstance(target_size, int):
        target_h, target_w = target_size, target_size
    else:
        target_h, target_w = int(target_size[0]), int(target_size[1])
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
        (target_h, target_w),
        interpolation="bicubic",
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

    if keras.config.image_data_format() == "channels_first":
        image = keras.ops.transpose(image, (0, 3, 1, 2))

    return {
        "pixel_values": image,
        "original_size": (orig_h, orig_w),
        "reshaped_size": (target_h, target_w),
    }


def DepthAnythingV2PostProcessDepth(
    predicted_depth: "keras.KerasTensor",
    original_size: Tuple[int, int],
) -> "keras.KerasTensor":
    """Resize predicted DepthAnythingV2 depth back to original image resolution.

    Mirrors HF ``DPTImageProcessor.post_process_depth_estimation``:
    bilinear-interpolate the predicted depth map to the original image
    size and squeeze the channel dim. Accepts Keras model output in
    either channels-last ``(B, H, W, 1)`` or channels-first
    ``(B, 1, H, W)`` format, or a 3D depth tensor ``(B, H, W)``.

    Args:
        predicted_depth: Depth tensor shaped ``(B, H, W)``,
            ``(B, H, W, 1)``, or ``(B, 1, H, W)``.
        original_size: Original image ``(height, width)``.

    Returns:
        Keras tensor of shape ``(B, orig_h, orig_w)`` — the depth map
        resampled to the original image resolution.

    Example:
        ```python
        depth = model(inputs["pixel_values"])
        depth_full = DepthAnythingV2PostProcessDepth(
            depth, original_size=inputs["original_size"]
        )
        ```
    """
    pd = keras.ops.convert_to_tensor(predicted_depth, dtype="float32")
    ndim = keras.ops.ndim(pd)

    if ndim == 3:
        pd = keras.ops.expand_dims(pd, axis=-1)
    elif ndim == 4:
        shape = keras.ops.shape(pd)
        if keras.config.image_data_format() == "channels_first" or shape[1] == 1:
            pd = keras.ops.transpose(pd, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected predicted_depth with 3 or 4 dims, got ndim={ndim}")

    orig_h, orig_w = original_size
    pd = keras.ops.image.resize(
        pd,
        (orig_h, orig_w),
        interpolation="bilinear",
        antialias=False,
    )
    pd = keras.ops.squeeze(pd, axis=-1)
    return pd
