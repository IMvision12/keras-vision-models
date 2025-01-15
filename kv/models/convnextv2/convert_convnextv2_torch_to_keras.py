import re
from typing import Any, Dict, List, Optional, Union

import keras
import timm
import torch
from tqdm import tqdm

from kv.models.convnextv2 import ConvNeXtV2Atto
from kv.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kv.utils.model_equivalence_tester import verify_cls_model_equivalence
from kv.utils.weight_split_torch_and_keras import split_model_weights
from kv.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "stem_conv_": "stem.0.",
    "stem_layernorm_": "stem.1.",
    "_": ".",
    "layernorm": "norm",
    "depthwise.conv": "conv_dw",
    "grn": "mlp.grn",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "conv.1": "mlp.fc1",
    "conv.2": "mlp.fc2",
    "downsampling.norm": "downsample.0",
    "downsampling.conv": "downsample.1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "final.norm": "head.norm",
    "predictions": "head.fc",
}

model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": ConvNeXtV2Atto,
    "torch_model_name": "convnextv2_atto.fcmae_ft_in1k",
    "input_shape": [224, 224, 3],
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
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)
    torch_weight_name = re.sub(
        r"layer\.scale\.variable(\.\d+)?", "gamma", torch_weight_name
    )

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
