from typing import Dict, List, Union

import keras
import torch
from tqdm import tqdm

from kvmm.models import mix_transformer
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.model_equivalence_tester import verify_cls_model_equivalence
from kvmm.utils.model_weights_util import download_weights
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "dense.1": "fc1",
    "dense.2": "fc2",
    "layernorm": "norm",
    ".conv": ".proj",
    "dwconv": "dwconv.dwconv",
    "overlap.patch.embed": "patch_embed",
    "kernel": "weight",  # conv2d
    "gamma": "weight",  # batchnorm weight
    "beta": "bias",  # batchnorm bias
    "bias": "bias",
    "predictions": "head",
}

model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": mix_transformer.MiT_B0,
    "torch_model_name": "mit_b0",
    "input_shape": [224, 224, 3],
    "num_classes": 1000,
    "include_top": True,
    "include_normalization": False,
    "classifier_activation": "linear",
}

# For Github Actions to run
torch_model_path_mit_b0 = "https://huggingface.co/IMvision12/Test/resolve/main/mit_b0.pth"  # TODO: change once repo is open sourced


keras_model: keras.Model = model_config["keras_model_cls"](
    include_top=model_config["include_top"],
    input_shape=model_config["input_shape"],
    classifier_activation=model_config["classifier_activation"],
    num_classes=model_config["num_classes"],
    include_normalization=model_config["include_normalization"],
    weights=None,
)
temp_path = download_weights(torch_model_path_mit_b0)
torch_model: torch.nn.Module = torch.load(temp_path, map_location="cpu")

trainable_keras_weights, non_trainable_keras_weights = split_model_weights(keras_model)

for keras_weight, keras_weight_name in tqdm(
    trainable_keras_weights + non_trainable_keras_weights,
    total=len(trainable_keras_weights + non_trainable_keras_weights),
    desc="Transferring weights",
):
    torch_weight_name: str = keras_weight_name
    for keras_name_part, torch_name_part in weight_name_mapping.items():
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)

    torch_weights_dict: Dict[str, torch.Tensor] = torch_model

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

test_keras_with_weights = mix_transformer.MiT_B0(
    weights=None,
    num_classes=model_config["num_classes"],
    include_top=model_config["include_top"],
    include_normalization=True,
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
    prediction_threshold=0.68,
)

if not results["imagenet_test"]["all_passed"]:
    raise ValueError(
        "Model equivalence test failed - model outputs do not match for standard input"
    )

model_filename: str = f"{model_config['torch_model_name'].replace('.', '_')}.keras"
keras_model.save(model_filename)
print(f"Model saved successfully as {model_filename}")
