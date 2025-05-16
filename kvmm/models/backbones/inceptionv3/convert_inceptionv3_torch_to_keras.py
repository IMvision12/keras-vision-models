import re
from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kvmm.models import inceptionv3
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.model_equivalence_tester import verify_cls_model_equivalence
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)


def convert_mixed_block_names(name: str) -> str:
    """
    Converts Mixed block layer names from underscore to dot notation.
    Example: Mixed_5b_branch1x1 -> Mixed_5b.branch1x1
    """
    pattern = r"(Mixed_[0-9][a-e])_(.+)"
    match = re.match(pattern, name)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return name


weight_name_mapping: Dict[str, str] = {
    "_conv2d_kernel": ".conv.weight",
    "_batchnorm_gamma": ".bn.weight",
    "_batchnorm_beta": ".bn.bias",
    "_batchnorm_moving_mean": ".bn.running_mean",
    "_batchnorm_moving_variance": ".bn.running_var",
    "classifier_kernel": "fc.weight",
    "classifier_bias": "fc.bias",
}

model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": inceptionv3.InceptionV3,
    "torch_model_name": "inception_v3.tf_adv_in1k",
    "input_shape": [299, 299, 3],
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
        torch_weight_name = convert_mixed_block_names(torch_weight_name)

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
    input_shape=(299, 299, 3),
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
