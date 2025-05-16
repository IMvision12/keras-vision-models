import re
from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kvmm.models import inceptionv4
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.model_equivalence_tester import verify_cls_model_equivalence
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping = {
    "features_": "features.",
    "_conv": ".conv",
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_bn": ".bn",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "predictions": "last_linear",
}

model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": inceptionv4.InceptionV4,
    "torch_model_name": "inception_v4",
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

    torch_weight_name = re.sub(r"features\.(\d+)_", r"features.\1.", torch_weight_name)

    if re.match(r"features\.(19|20|21)\.", torch_weight_name):
        if "branch3_1" in torch_weight_name:
            torch_weight_name = torch_weight_name.replace("branch3_1", "branch3.1")
        else:
            torch_weight_name = re.sub(
                r"\.branch([12])\.([0-9]+[ab]?)", r".branch\1_\2", torch_weight_name
            )
    else:
        torch_weight_name = re.sub(
            r"\.branch([0-9])_([0-9][ab]?)", r".branch\1.\2", torch_weight_name
        )
        torch_weight_name = torch_weight_name.replace("branch3_1", "branch3.1")

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
    # Save model
    model_filename: str = (
        f"{model_config['torch_model_name'].replace('.', '_')}.weights.h5"
    )
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")
