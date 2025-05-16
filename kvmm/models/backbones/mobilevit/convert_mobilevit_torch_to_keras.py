from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kvmm.models import mobilevit
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.model_equivalence_tester import verify_cls_model_equivalence
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "batchnorm": "bn",
    "ir.conv.1": "conv1_1x1.conv",
    "ir.bn.1": "conv1_1x1.bn",
    "ir.dwconv": "conv2_kxk.conv",
    "ir.bn.2": "conv2_kxk.bn",
    "ir.conv.2": "conv3_1x1.conv",
    "ir.bn.3": "conv3_1x1.bn",
    "mv.conv.1": "conv_kxk.conv",
    "mv.bn.1": "conv_kxk.bn",
    "mv.conv.2": "conv_1x1",
    "layernorm": "norm",
    "norm.1": "norm1",
    "norm.2": "norm2",
    "mv.conv.3": "conv_proj.conv",
    "mv.bn.2": "conv_proj.bn",
    "mv.conv.4": "conv_fusion.conv",
    "mv.bn.3": "conv_fusion.bn",
    "final.conv": "final_conv.conv",
    "final.bn": "final_conv.bn",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}


model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": mobilevit.MobileViTXXS,
    "torch_model_name": "mobilevit_xxs.cvnets_in1k",
    "input_shape": [256, 256, 3],
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

for keras_weight, keras_weight_name in tqdm(
    trainable_keras_weights + non_trainable_keras_weights,
    total=len(trainable_keras_weights + non_trainable_keras_weights),
    desc="Transferring weights",
):
    torch_weight_name: str = keras_weight_name
    for keras_name_part, torch_name_part in weight_name_mapping.items():
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)

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
    input_shape=(256, 256, 3),
    output_specs={"num_classes": 1000},
    run_performance=False,
    atol=1e-3,
    rtol=1e-3,
)

if not results["standard_input"]:
    raise ValueError(
        "Model equivalence test failed - model outputs do not match for standard input"
    )

model_filename: str = f"{model_config['torch_model_name'].replace('.', '_')}.weights.h5"
keras_model.save_weights(model_filename)
print(f"Model saved successfully as {model_filename}")
