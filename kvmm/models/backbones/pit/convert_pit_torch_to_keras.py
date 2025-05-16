import re
from typing import Dict, List, Union

import keras
import numpy as np
import timm
import torch
from tqdm import tqdm

from kvmm.models import pit
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
    "pit": "transformers",
    "patch.embed": "patch_embed",
    "pos.embed.pos.embed": "pos_embed",
    "class.dist.token.cls.token": "cls_token",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "layerscale.1": "ls1",
    "layerscale.2": "ls2",
    "pool.dense": "pool.fc",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head",
    "head.dist": "head_dist",
}


model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": pit.PiT_XS,
    "torch_model_name": "pit_xs_224.in1k",
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
        r"pos_embed_variable_\d+$", "pos_embed", torch_weight_name
    )
    torch_weight_name = re.sub(
        r"cls_token_variable_\d+$", "cls_token", torch_weight_name
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
        torch_weight = torch_weight.numpy()

        if torch_weight.ndim == 4:
            C, H, W = torch_weight.shape[1:]
            torch_weight = torch_weight.transpose(0, 2, 3, 1)
            torch_weight = torch_weight.reshape(1, H * W, C)

            if keras_weight.shape[1] > H * W:
                num_extra_tokens = keras_weight.shape[1] - H * W
                class_pos_embed = np.zeros((1, num_extra_tokens, C))
                torch_weight = np.concatenate([class_pos_embed, torch_weight], axis=1)

        keras_weight.assign(torch_weight)
        continue

    if not compare_keras_torch_names(
        keras_weight_name, keras_weight, torch_weight_name, torch_weight
    ):
        raise WeightShapeMismatchError(
            keras_weight_name, keras_weight.shape, torch_weight_name, torch_weight.shape
        )

    transfer_weights(keras_weight_name, keras_weight, torch_weight)

if "distilled" in str(model_config["keras_model_cls"]).lower():
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


else:
    results = verify_cls_model_equivalence(
        model_a=torch_model,
        model_b=keras_model,
        input_shape=(224, 224, 3),
        output_specs={"num_classes": 1000},
        run_performance=False,
    )

if ("standard_input" in results and not results["standard_input"]) or (
    "imagenet_test" in results and not results["imagenet_test"]["all_passed"]
):
    raise ValueError(
        "Model equivalence test failed - model outputs do not match for standard input"
    )

model_filename: str = f"{model_config['torch_model_name'].replace('.', '_')}.weights.h5"
keras_model.save_weights(model_filename)
print(f"Model saved successfully as {model_filename}")
