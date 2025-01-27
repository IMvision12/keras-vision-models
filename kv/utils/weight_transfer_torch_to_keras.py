"""
Weight Transfer Utility for Converting PyTorch Model Weights to Keras

This module provides utility functions for transferring weights between PyTorch and Keras
neural network layers, handling various layer types and weight transformations.

Key Features:
- Supports conversion of weights for different layer types:
  - Convolutional layers (1D and 2D)
  - Dense/Linear layers
  - RNN layers (LSTM and GRU)
  - Embedding layers
  - Layer Normalization
  - Attention mechanisms

Dependencies:
- numpy
- keras
- torch

Example:
    # Assuming you have PyTorch and Keras model weights
    transfer_weights(
        keras_name='conv1_weights',
        keras_weight=keras_model.layers[0].weights[0],
        torch_weight=torch_model.conv1.weight.numpy()
    )
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import keras
import numpy as np
import torch


class WeightType(Enum):
    KERNEL = ("kernel", "weight")
    BIAS = ("bias", "bias")

    GAMMA = ("gamma", "weight")
    BETA = ("beta", "bias")
    MOVING_MEAN = ("moving_mean", "running_mean")
    MOVING_VARIANCE = ("moving_variance", "running_var")

    QUERY = ("query", "q_proj")
    KEY = ("key", "k_proj")
    VALUE = ("value", "v_proj")
    ATTENTION = ("attention", "attn")
    FFN = ("dense", "fc")
    OUTPUT = ("output", "out_proj")

    LAYER_NORM_GAMMA = ("layer_norm_gamma", "weight")
    LAYER_NORM_BETA = ("layer_norm_beta", "bias")

    EMBED_TOKEN = ("embed_token", "embed_tokens")
    EMBED_POS = ("embed_positions", "pos_embed")
    EMBED_PATCH = ("patch_embed", "patch_embed")

    CONV_KERNEL = ("conv_kernel", "conv.weight")
    CONV_BIAS = ("conv_bias", "conv.bias")

    POOL = ("pool", "pool")
    PROJ = ("projection", "proj")

    @classmethod
    def find_weight_type(cls, keras_name: str) -> Optional["WeightType"]:
        """Find the matching weight type for a given Keras weight name."""
        for weight_type in cls:
            if keras_name.endswith(weight_type.value[0]):
                return weight_type
        return None


class WeightMismatchError(Exception):
    """Custom exception for weight comparison mismatches."""

    pass


def validate_input_weights(
    keras_weight: Any, torch_weight: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
    # Ensure torch_weight is numpy array
    if isinstance(torch_weight, torch.Tensor):
        torch_weight = torch_weight.numpy()

    keras_shape = keras_weight.shape
    torch_shape = torch_weight.shape

    if not keras_shape or not torch_shape:
        raise ValueError(
            f"Empty shapes not allowed. Keras shape: {keras_shape}, "
            f"Torch shape: {torch_shape}"
        )

    return torch_weight, keras_shape, torch_shape


def transform_conv_weights(
    keras_name: str,
    torch_weight: np.ndarray,
) -> np.ndarray:
    if "conv1d" in keras_name.lower():  # [width, in_channels, out_channels]
        return np.transpose(torch_weight, [2, 1, 0])

    elif any(
        substring in keras_name.lower()
        for substring in [
            "depthwise",
            "dwconv2d",
            "dwconv",
        ]
    ):
        return np.transpose(torch_weight, [2, 3, 0, 1])

    elif any(
        substring in keras_name.lower()
        for substring in ["conv", "conv2d", "pointwise", "downsample", "sr"]
    ):
        # Standard 2D convolution
        return np.transpose(torch_weight, [2, 3, 1, 0])

    elif "grn" in keras_name:  # For ConvNextV2
        return np.expand_dims(
            np.expand_dims(np.expand_dims(torch_weight, axis=0), axis=0), axis=0
        )


def transform_dense_weights(
    keras_name: str, torch_weight: np.ndarray, keras_shape: Tuple[int, ...]
) -> np.ndarray:
    if "se" in keras_name and torch_weight.ndim == 4:  # SE block
        torch_weight = torch_weight.squeeze()

    if keras_shape[1] == torch_weight.shape[0]:
        return np.transpose(torch_weight)

    raise ValueError(
        f"Shape mismatch in Dense/SE layer {keras_name}. "
        f"Keras shape={keras_shape}, Torch shape={torch_weight.shape}"
    )


def transform_rnn_weights(
    keras_name: str, torch_weight: np.ndarray, rnn_type: str
) -> np.ndarray:
    if not ("kernel" in keras_name or "recurrent_kernel" in keras_name):
        return torch_weight

    if rnn_type == "lstm":
        split_size = torch_weight.shape[1] // 4
        return np.concatenate(
            [
                torch_weight[:, :split_size],  # input gate
                torch_weight[:, split_size : 2 * split_size],  # forget gate
                torch_weight[:, -split_size:],  # cell gate
                torch_weight[:, 2 * split_size : -split_size],  # output gate
            ],
            axis=1,
        )

    elif rnn_type == "gru":
        split_size = torch_weight.shape[1] // 3
        return np.concatenate(
            [
                torch_weight[:, split_size : 2 * split_size],  # reset gate
                torch_weight[:, :split_size],  # update gate
                torch_weight[:, 2 * split_size :],  # new gate
            ],
            axis=1,
        )

    raise ValueError(f"Unsupported RNN type: {rnn_type}")


def transfer_weights(
    keras_name: str, keras_weight: keras.Variable, torch_weight: np.ndarray
) -> None:
    """
    Transfer weights from PyTorch to Keras based on layer type.

    Handles weight transformation for various neural network layer types.

    Args:
        keras_name (str): Name of the Keras weight for type detection.
        keras_weight (keras.Variable): Keras weight variable to update.
        torch_weight (np.ndarray): PyTorch weight tensor to transfer.

    Raises:
        ValueError: If the layer type or weight shapes are unsupported.
    """
    torch_weight, keras_shape, torch_shape = validate_input_weights(
        keras_weight, torch_weight
    )

    # Determine transformation based on weight dimensionality
    if len(keras_shape) == 4:  # conv2d, depthwise conv
        transformed = transform_conv_weights(keras_name, torch_weight)

    elif len(keras_shape) == 2:
        if "embedding" in keras_name.lower():
            transformed = torch_weight
        else:
            transformed = transform_dense_weights(keras_name, torch_weight, keras_shape)

    elif len(keras_shape) == 1:
        if "layernorm" in keras_name.lower():
            if any(
                x in keras_name.lower() for x in ["gamma", "weight", "beta", "bias"]
            ):
                transformed = torch_weight
        elif "bias" in keras_name.lower() or "batchnorm" in keras_name.lower():
            transformed = torch_weight
        elif keras_shape == torch_shape:
            transformed = torch_weight
        else:
            raise ValueError(
                f"Shape mismatch in 1D weight {keras_name}. "
                f"Keras shape={keras_shape}, Torch shape={torch_shape}"
            )

    elif len(keras_shape) == 0 and len(torch_shape) == 1:
        transformed = torch_weight[0]

    elif "lstm" in keras_name.lower():
        transformed = transform_rnn_weights(keras_name, torch_weight, "lstm")

    elif "gru" in keras_name.lower():
        transformed = transform_rnn_weights(keras_name, torch_weight, "gru")

    else:
        raise ValueError(
            f"Unsupported layer type or shape mismatch for {keras_name}. "
            f"Keras shape={keras_shape}, Torch shape={torch_shape}"
        )

    keras_weight.assign(transformed)


def transfer_attention_weights(
    keras_name: str,
    keras_weight: keras.Variable,
    torch_weights_dict: Dict[str, torch.Tensor],
    name_replacements: Dict[str, str] = None,
) -> None:
    """
    Transfer attention mechanism weights from PyTorch to Keras.

    Maps PyTorch attention layer weights to corresponding Keras weights.

    Args:
        keras_name (str): Name of the Keras weight.
        keras_weight (keras.Variable): Keras weight variable to update.
        torch_weight_name (str): Name of the corresponding PyTorch weight.
        torch_weights_dict (Dict[str, torch.Tensor]): Dictionary of PyTorch weights.
        name_replacements (Dict[str, str], optional): Dictionary of custom name replacements
            to apply after replacing "_" with ".". Keys are strings to replace, values are
            their replacements. Defaults to None.

    Raises:
        ValueError: If the PyTorch weight is missing or weight type is unexpected.
    """
    keras_layer_path = keras_weight.path
    layer_name = keras_layer_path.split("/")[-2].replace("_", ".")

    if name_replacements:
        for old_name, new_name in name_replacements.items():
            layer_name = layer_name.replace(old_name, new_name)

    if "kernel" in keras_name:
        torch_name = f"{layer_name}.weight"
    elif "bias" in keras_name:
        torch_name = f"{layer_name}.bias"
    elif "gamma" in keras_name:
        torch_name = f"{layer_name}.weight"
    elif "beta" in keras_name:
        torch_name = f"{layer_name}.bias"
    elif "moving_mean" in keras_name:
        torch_name = f"{layer_name}.running_mean"
    elif "moving_variance" in keras_name:
        torch_name = f"{layer_name}.running_var"
    else:
        raise ValueError(f"Unexpected weight type in attention layer: {keras_name}")

    try:
        torch_weights = torch_weights_dict[torch_name]
        transfer_weights(torch_name, keras_weight, torch_weights)
    except KeyError:
        raise ValueError(
            f"Missing PyTorch weight '{torch_name}' for Keras weight '{keras_name}'"
        )


def compare_keras_torch_names(
    keras_name: str,
    keras_weights: Union[keras.Variable, np.ndarray],
    torch_name: str,
    torch_weights: Union[torch.Tensor, np.ndarray],
    verbose: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    check_values: bool = False,
) -> bool:
    """
    Enhanced comparison of Keras and PyTorch weights with comprehensive error reporting.

    Args:
        keras_name: Name of the Keras weights
        keras_weights: Keras weights as Variable or numpy array
        torch_name: Name of the PyTorch weights
        torch_weights: PyTorch weights as Tensor or numpy array
        verbose: Whether to print mismatch details (default: True)
        rtol: Relative tolerance for value comparison
        atol: Absolute tolerance for value comparison
        check_values: Whether to check actual weight values (default: False)

    Returns:
        Boolean indicating if weights match

    Raises:
        WeightMismatchError: When weights don't match and detailed error information
    """

    def _format_mismatch(error_type: str, details: str) -> str:
        return (
            f"Weight Mismatch Detected:\n"
            f"  Keras name: {keras_name}\n"
            f"  Torch name: {torch_name}\n"
            f"  Keras shape: {keras_weights_np.shape}\n"
            f"  Torch shape: {torch_weights_np.shape}\n"
            f"  Type: {error_type}\n"
            f"  Details: {details}\n"
            f"{'-' * 50}"
        )

    def _handle_mismatch(error_type: str, details: str) -> bool:
        message = _format_mismatch(error_type, details)
        if verbose:
            print(message)
        return False

    keras_weights_np = (
        keras_weights.numpy() if hasattr(keras_weights, "numpy") else keras_weights
    )
    torch_weights_np = (
        torch_weights.detach().cpu().numpy()
        if isinstance(torch_weights, torch.Tensor)
        else torch_weights
    )

    keras_size = np.prod(keras_weights_np.shape)
    torch_size = np.prod(torch_weights_np.shape)

    if keras_size != torch_size:
        if (
            (keras_size == 0 and torch_size == 1)
            or (keras_size == 1 and torch_size > 1)
            or (torch_size == 1 and keras_size > 1)
        ):
            return True

        return _handle_mismatch(
            "shape",
            f"Element count mismatch: Keras={keras_size} ({keras_weights_np.shape}), "
            f"Torch={torch_size} ({torch_weights_np.shape})",
        )

    weight_type = WeightType.find_weight_type(keras_name)
    if weight_type:
        keras_suffix, torch_suffix = weight_type.value
        if not torch_name.endswith(torch_suffix):
            return _handle_mismatch(
                "type",
                f"Expected Torch suffix '{torch_suffix}' for Keras '{keras_suffix}'",
            )

    if check_values:
        try:
            if not np.allclose(
                keras_weights_np, torch_weights_np, rtol=rtol, atol=atol
            ):
                max_diff = np.max(np.abs(keras_weights_np - torch_weights_np))
                return _handle_mismatch(
                    "values", f"Weight values differ (max diff: {max_diff:.6f})"
                )
        except Exception as e:
            return _handle_mismatch(
                "comparison", f"Error during value comparison: {str(e)}"
            )

    return True
