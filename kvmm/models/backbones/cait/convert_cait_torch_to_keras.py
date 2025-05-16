import re
from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kvmm.models import cait
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
    "stem.conv": "patch_embed.proj",
    "cls.token.cls.token": "cls_token",
    "pos.embed.pos.embed": "pos_embed",
    "layernorm.": "norm",
    "dense.1": "fc1",
    "dense.2": "fc2",
    "blocks.token.only": "blocks_token_only",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
    "final.norm": "norm.",
    "predictions": "head",
}

attn_weight_replacement = {
    "proj.l": "proj_l",
    "proj.w": "proj_w",
    "blocks.token.only": "blocks_token_only",
}
model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": cait.CaiTXXS24,
    "torch_model_name": "cait_xxs24_224.fb_dist_in1k",
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

for keras_weight, keras_weight_name in tqdm(
    trainable_keras_weights + non_trainable_keras_weights,
    total=len(trainable_keras_weights + non_trainable_keras_weights),
    desc="Transferring weights",
):
    torch_weight_name: str = keras_weight_name
    for keras_name_part, torch_name_part in weight_name_mapping.items():
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)
    torch_weight_name = re.sub(
        r"layerscale\.(\d+)\.variable(?:\.\d+)?", r"gamma_\1", torch_weight_name
    )

    torch_weights_dict: Dict[str, torch.Tensor] = {
        **trainable_torch_weights,
        **non_trainable_torch_weights,
    }

    if "attention" in torch_weight_name:
        transfer_attention_weights(
            keras_weight_name, keras_weight, torch_weights_dict, attn_weight_replacement
        )
        continue

    if torch_weight_name not in torch_weights_dict:
        raise WeightMappingError(keras_weight_name, torch_weight_name)

    torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

    if torch_weight_name == "cls_token":
        keras_weight.assign(torch_weight)
        continue

    if torch_weight_name == "pos_embed":
        keras_weight.assign(torch_weight)
        continue

    if not compare_keras_torch_names(
        keras_weight_name, keras_weight, torch_weight_name, torch_weight
    ):
        raise WeightShapeMismatchError(
            keras_weight_name, keras_weight.shape, torch_weight_name, torch_weight.shape
        )

    transfer_weights(keras_weight_name, keras_weight, torch_weight)

test_keras_with_weights = model_config["keras_model_cls"](
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
)

if not results["imagenet_test"]["all_passed"]:
    raise ValueError(
        "Model equivalence test failed - model outputs do not match for standard input"
    )

model_filename: str = f"{model_config['torch_model_name'].replace('.', '_')}.weights.h5"
keras_model.save_weights(model_filename)
print(f"Model saved successfully as {model_filename}")
