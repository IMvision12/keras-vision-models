from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kv.models.res2net import Res2Net50_26w_4s
from kv.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kv.utils.model_equivalence_tester import verify_cls_model_equivalence
from kv.utils.weight_split_torch_and_keras import split_model_weights
from kv.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "_": ".",
    "batchnorm.1": "bn1",
    "batchnorm.s": "bns",
    "batchnorm.3": "bn3",
    "conv.1": "conv1",
    "conv.3": "conv3",
    "conv.s": "convs",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "fc",
}

model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": Res2Net50_26w_4s,
    "torch_model_name": "res2net50_26w_4s",
    "input_shape": [224, 224, 3],
    "num_classes": 1000,
    "include_top": True,
    "include_preprocessing": False,
    "classifier_activation": "linear",
}


keras_model: keras.Model = model_config["keras_model_cls"](
    include_top=model_config["include_top"],
    input_shape=model_config["input_shape"],
    classifier_activation=model_config["classifier_activation"],
    num_classes=model_config["num_classes"],
    include_preprocessing=model_config["include_preprocessing"],
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

test_keras_with_weights = Res2Net50_26w_4s(
    weights=None,
    num_classes=model_config["num_classes"],
    include_top=model_config["include_top"],
    include_preprocessing=True,
    input_shape=model_config["input_shape"],
    classifier_activation="softmax",
)
test_keras_with_weights.set_weights(keras_model.get_weights())

results = verify_cls_model_equivalence(
    model_a=None,
    model_b=test_keras_with_weights,
    input_shape=(224, 224, 3),
    output_specs={"num_classes": 1000},
    run_performance=False,
    test_imagenet_image=True,
)

if not results["imagenet_test"]["all_passed"]:
    raise ValueError(
        "Model equivalence test failed - model outputs do not match for standard input"
    )

model_filename: str = f"{model_config['torch_model_name'].replace('.', '_')}.keras"
keras_model.save(model_filename)
print(f"Model saved successfully as {model_filename}")
