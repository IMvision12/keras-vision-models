from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kvmm.models import inception_resnetv2
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.model_equivalence_tester import verify_cls_model_equivalence
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping = {
    "_conv": ".conv",
    "_batchnorm": ".bn",
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "mixed_5b_": "mixed_5b.",
    "mixed_6a_": "mixed_6a.",
    "mixed_7a_": "mixed_7a.",
    "repeats_1_": "repeat_1.",
    "repeats_2_": "repeat_2.",
    "branch1_0": "branch1.0",
    "branch1_1": "branch1.1",
    "branch1_2": "branch1.2",
    "branch2_0": "branch2.0",
    "branch2_1": "branch2.1",
    "branch2_2": "branch2.2",
    "branch3_1": "branch3.1",
    "branch0_0": "branch0.0",
    "branch0_1": "branch0.1",
    "block8_": "block8.",
    "predictions": "classif",
}


def generate_repeat_mappings():
    mappings = {}

    for i in range(10):
        mappings[f"repeat_{i}_"] = f"repeat.{i}."
        mappings[f"repeat_{i}"] = f"repeat.{i}"

    for i in range(20):
        base = f"repeat.1.{i}"
        keras_base = f"repeat_1.{i}"
        mappings[f"{base}_branch1.0"] = f"{keras_base}.branch1.0"
        mappings[f"{base}_branch1.1"] = f"{keras_base}.branch1.1"
        mappings[f"{base}_branch1.2"] = f"{keras_base}.branch1.2"
        mappings[f"{base}_branch0"] = f"{keras_base}.branch0"
        mappings[f"{base}.conv2d"] = f"{keras_base}.conv2d"

    for i in range(9):
        base = f"repeat.2.{i}"
        keras_base = f"repeat_2.{i}"
        mappings[f"{base}_branch1.0"] = f"{keras_base}.branch1.0"
        mappings[f"{base}_branch1.1"] = f"{keras_base}.branch1.1"
        mappings[f"{base}_branch1.2"] = f"{keras_base}.branch1.2"
        mappings[f"{base}_branch0"] = f"{keras_base}.branch0"
        mappings[f"{base}.conv2d"] = f"{keras_base}.conv2d"

    return mappings


weight_name_mapping.update(generate_repeat_mappings())

model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": inception_resnetv2.InceptionResNetV2,
    "torch_model_name": "inception_resnet_v2.tf_ens_adv_in1k",
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
    torch_weight_name = keras_weight_name
    for keras_pattern, torch_pattern in weight_name_mapping.items():
        torch_weight_name = torch_weight_name.replace(keras_pattern, torch_pattern)

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


if results["standard_input"]:
    model_filename: str = (
        f"{model_config['torch_model_name'].replace('.', '_')}.weights.h5"
    )
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")
