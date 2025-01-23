from typing import Dict, Optional, Union

import keras
import numpy as np
from PIL import Image


def SegFormerImageProcessor(
    image: Union[str, np.ndarray, Image.Image],
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
        image: Input image (file path, numpy array, or PIL Image)
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

    if isinstance(image, str):
        try:
            image = Image.open(image).convert("RGB")
            image = np.array(image)
        except Exception as e:
            raise ValueError(f"Error loading image from path: {e}")
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if np.max(image) <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                raise ValueError("NumPy array must be uint8 or float32 in [0,1] range")
        if len(image.shape) == 4:
            image = image[0]
        if len(image.shape) != 3:
            raise ValueError("Input array must have shape (H, W, C)")
    else:
        raise TypeError("Input must be a file path, numpy array, or PIL Image")

    image = keras.ops.convert_to_tensor(image, dtype="float32")
    if len(image.shape) == 3:
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
        mean = keras.ops.convert_to_tensor(image_mean, dtype="float32")
        std = keras.ops.convert_to_tensor(image_std, dtype="float32")

        mean = keras.ops.reshape(mean, (1, 1, 1, 3))
        std = keras.ops.reshape(std, (1, 1, 1, 3))

        image = (image - mean) / std

    if not return_tensor:
        image = keras.ops.convert_to_numpy(image)

    return image
