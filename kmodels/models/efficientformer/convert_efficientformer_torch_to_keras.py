"""
Script to convert EfficientFormer PyTorch weights to Keras 3 format.

Usage:
    python convert_efficientformer_torch_to_keras.py
"""

import re
from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kmodels.models.efficientformer import (
    EfficientFormerL1,
)
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.model_equivalence_tester import verify_cls_model_equivalence
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "stem.conv1": "stem.conv1",
    "stem.norm1": "stem.norm1",
    "stem.conv2": "stem.conv2",
    "stem.norm2": "stem.norm2",
    "downsample.conv": "downsample.conv",
    "downsample.norm": "downsample.norm",
    "pool.pool": "token_mixer.pool",
    "mlp.conv.1": "mlp.fc1",
    "mlp.norm.1": "mlp.norm1",
    "mlp.conv.2": "mlp.fc2",
    "mlp.norm.2": "mlp.norm2",
    "mlp.dense.1": "mlp.fc1",
    "mlp.dense.2": "mlp.fc2",
    "attn.qkv": "token_mixer.qkv",
    "attn.proj": "token_mixer.proj",
    "norm1": "norm1",
    "norm2": "norm2",
    "final.norm": "norm",
    "kernel": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "head.": "head.",
    "head.dist.": "head_dist.",
}


model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": EfficientFormerL1,
    "torch_model_name": "efficientformer_l1.snap_dist_in1k",
    "input_shape": [224, 224, 3],
    "num_classes": 1000,
    "include_top": True,
    "include_normalization": False,
    "classifier_activation": "linear",
}


keras_model: keras.Model = model_config["keras_model_cls"](
    include_top=model_config["include_top"],
    input_shape=model_config["input_shape"],
    classifier_activation=model_config["classifier_activation"],
    num_classes=model_config["num_classes"],
    include_normalization=model_config["include_normalization"],
    weights=None,
)

torch_model: torch.nn.Module = timm.create_model(
    model_config["torch_model_name"], pretrained=True
).eval()

trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
    torch_model
)
trainable_keras_weights, non_trainable_keras_weights = split_model_weights(keras_model)

torch_weights_dict: Dict[str, torch.Tensor] = {
    **trainable_torch_weights,
    **non_trainable_torch_weights,
}

for keras_weight, keras_weight_name in tqdm(
    trainable_keras_weights + non_trainable_keras_weights,
    total=len(trainable_keras_weights + non_trainable_keras_weights),
    desc="Transferring weights",
):
    torch_weight_name: str = keras_weight_name

    torch_weight_name = re.sub(r"_variable(_\d+)?$", "_gamma", torch_weight_name)

    for keras_name_part, torch_name_part in weight_name_mapping.items():
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)

    if ".gamma" in torch_weight_name and ".ls" not in torch_weight_name:
        torch_weight_name = torch_weight_name.replace(".gamma", ".weight")

    if "stages.3.blocks.3" in torch_weight_name:
        torch_weight_name = torch_weight_name.replace(
            "stages.3.blocks.3", "stages.3.blocks.4"
        )

    if "attn.attention.biases" in torch_weight_name:
        torch_weight_name = torch_weight_name.replace(
            ".attn.attention.biases", ".token_mixer.attention_biases"
        )

    if "attention_bias_idxs" in torch_weight_name:
        continue

    if torch_weight_name not in torch_weights_dict:
        raise WeightMappingError(keras_weight_name, torch_weight_name)

    torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

    if "attention_biases" in keras_weight_name:
        keras_weight.assign(
            torch_weight.numpy() if hasattr(torch_weight, "numpy") else torch_weight
        )
        continue

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

model_filename: str = f"{model_config['torch_model_name'].replace('.', '_')}.weights.h5"
keras_model.save_weights(model_filename)
print(f"Model saved successfully as {model_filename}")
