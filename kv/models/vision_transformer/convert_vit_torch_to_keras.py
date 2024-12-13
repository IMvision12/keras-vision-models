import re
from typing import Any, Dict, List, Optional, Union

import keras
import timm
import torch

from tqdm import tqdm
from kv.models.vision_transformer import *
from kv.utils.custom_exception import *
from kv.utils.test_keras_models import run_all_tests
from kv.utils.weight_split_torch_keras import split_model_weights
from kv.utils.weight_transfer_torch_to_keras import *

weight_name_mapping = {
    "_": ".",
    "conv1": "patch_embed.proj",
    "pos.embed.pos.embed": "pos_embed",
    "cls.token.cls.token": "cls_token",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head",
}


model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": ViTLarge32,
    "torch_model_name": "vit_large_patch32_384.orig_in21k_ft_in1k",
    "input_shape": [384, 384, 3],
    "num_classes": 1000,
    "include_top": True,
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
        rf"pos_embed_variable_\d+$", "pos_embed", torch_weight_name
    )
    torch_weight_name = re.sub(
        rf"cls_token_variable_\d+$", "cls_token", torch_weight_name
    )

    torch_weights_dict: Dict[str, torch.Tensor] = {
        **trainable_torch_weights,
        **non_trainable_torch_weights,
    }

    if "attention" in torch_weight_name:
        transfer_attention_weights(keras_weight_name, keras_weight, torch_weights_dict)
        continue

    if torch_weight_name not in torch_weights_dict:
        raise WeightMappingError(keras_weight_name, torch_weight_name)

    torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

    if torch_weight_name == "cls_token":
        keras_weight.assign(torch_weight)
        continue

    if torch_weight_name == "pos_embed":
        if torch_weight.shape[1] == keras_weight.shape[1] + 1:
            torch_weight = torch_weight[:, 1:, :]
        keras_weight.assign(torch_weight)
        continue

    if not compare_keras_torch_names(
        keras_weight_name, keras_weight, torch_weight_name, torch_weight
    ):
        raise WeightShapeMismatchError(
            keras_weight_name, keras_weight.shape, torch_weight_name, torch_weight.shape
        )

    transfer_weights(keras_weight_name, keras_weight, torch_weight)

results = run_all_tests(
    keras_model=keras_model,
    torch_model=torch_model,
    input_shape=(384, 384, 3),
    output_specs={"num_classes": 1000},
    run_performance=False,
)

if results["standard_input"]:
    # Save model
    model_filename: str = f"{model_config['torch_model_name'].replace('.', '_')}.keras"
    keras_model.save(model_filename)
    print(f"Model saved successfully as {model_filename}")
