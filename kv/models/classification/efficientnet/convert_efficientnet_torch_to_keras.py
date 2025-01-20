import re
from typing import Any, Dict, List, Optional, Union

import keras
import timm
import torch
from tqdm import tqdm

from kv.models import EfficientNetB0
from kv.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kv.utils.model_equivalence_tester import verify_cls_model_equivalence
from kv.utils.weight_split_torch_and_keras import split_model_weights
from kv.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

block_mappings = {}

for i in range(6):
    block_prefix = f"blocks.0.{i}"
    # Conv layer mappings
    block_mappings[f"{block_prefix}.conv_pwl"] = f"{block_prefix}.conv_pw"
    # BN layer mappings
    block_mappings[f"{block_prefix}.bn2"] = f"{block_prefix}.bn1"
    block_mappings[f"{block_prefix}.bn3"] = f"{block_prefix}.bn2"

base_mappings = {
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "se_": "se.",
    "batchnorm_1": "bn1",
    "batchnorm_2": "bn2",
    "batchnorm_3": "bn3",
    "conv2d_1": "conv_pw",
    "dwconv2d": "conv_dw",
    "conv2d_2": "conv_pwl",
    "predictions": "classifier",
}

weight_name_mapping = {**base_mappings, **block_mappings}

model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": EfficientNetB0,
    "torch_model_name": "tf_efficientnet_b0.ns_jft_in1k",
    "input_shape": [
        224,
        224,
        3,
    ],  # Change as per the default given for different models
    "num_classes": 1000,
    "include_top": True,
    "include_preprocessing": False,
    "classifier_activation": "linear",
}


def create_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> Union[keras.Model, torch.nn.Module, None]:
    """
    Creates a Keras or Torch model based on the model_type.

    Args:
        model_type (str): Type of model to create ('keras' or 'torch').
        config (dict, optional): Configuration for Keras model (if model_type is 'keras').
        model_name (str, optional): Name of the Torch model (if model_type is 'torch').

    Returns:
        model: The created Keras or Torch model, or None if an error occurred.
    """
    if model_type == "keras" and config:
        return config["keras_model_cls"](
            weights=None,
            num_classes=config["num_classes"],
            include_top=config["include_top"],
            include_preprocessing=config["include_preprocessing"],
            input_shape=config["input_shape"],
            classifier_activation=config["classifier_activation"],
        )
    elif model_type == "torch" and model_name:
        try:
            return timm.create_model(model_name, pretrained=True).eval()
        except Exception as e:
            print(f"Error loading Torch model '{model_name}': {e}")
            return None
    else:
        print("Invalid model type or missing configuration.")
        return None


keras_model: keras.Model = create_model("keras", config=model_config)
torch_model: torch.nn.Module = create_model(
    "torch", model_name=model_config["torch_model_name"]
)

trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
    torch_model
)
trainable_keras_weights, non_trainable_keras_weights = split_model_weights(keras_model)

for keras_weight, keras_weight_name in tqdm(
    trainable_keras_weights + non_trainable_keras_weights,
    total=len(trainable_keras_weights + non_trainable_keras_weights),
    desc="Transferring weights",
):
    torch_weight_name: str = keras_weight_name
    for keras_name_part, torch_name_part in weight_name_mapping.items():
        torch_weight_name = re.sub(
            r"blocks_(\d+)_(\d+)_",
            lambda m: f"blocks.{m.group(1)}.{m.group(2)}.",
            torch_weight_name,
        )
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)

    torch_weights_dict: Dict[str, torch.Tensor] = {
        **trainable_torch_weights,
        **non_trainable_torch_weights,
    }

    if torch_weight_name not in torch_weights_dict:
        raise WeightMappingError(keras_weight_name, torch_weight_name)

    torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

    if not compare_keras_torch_names(
        keras_weight_name, keras_weight, torch_weight_name, torch_weight
    ):
        raise WeightShapeMismatchError(
            keras_weight_name, keras_weight.shape, torch_weight_name, torch_weight.shape
        )

    transfer_weights(keras_weight_name, keras_weight, torch_weight)


results = verify_cls_model_equivalence(
    model_a=torch_model,
    model_b=keras_model,
    input_shape=(224, 224, 3),
    output_specs={"num_classes": 1000},
    run_performance=False,
)


if not results["standard_input"]:
    raise ValueError(
        "Model equivalence test failed - model outputs do not match for standard input"
    )

model_filename: str = f"{model_config['torch_model_name'].replace('.', '_')}.keras"
keras_model.save(model_filename)
print(f"Model saved successfully as {model_filename}")
