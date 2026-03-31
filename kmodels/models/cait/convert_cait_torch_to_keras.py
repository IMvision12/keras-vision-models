import re
from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kmodels.models import cait
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.model_equivalence_tester import verify_cls_model_equivalence
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
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

attn_weight_replacement: Dict[str, str] = {
    "proj.l": "proj_l",
    "proj.w": "proj_w",
    "blocks.token.only": "blocks_token_only",
}

model_configs: List[Dict[str, Union[str, type]]] = [
    {
        "keras_cls": cait.CaiTXXS24,
        "torch_name": "cait_xxs24_224.fb_dist_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTXXS24,
        "torch_name": "cait_xxs24_384.fb_dist_in1k",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTXXS36,
        "torch_name": "cait_xxs36_224.fb_dist_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTXXS36,
        "torch_name": "cait_xxs36_384.fb_dist_in1k",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTXS24,
        "torch_name": "cait_xs24_384.fb_dist_in1k",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTS24,
        "torch_name": "cait_s24_224.fb_dist_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTS24,
        "torch_name": "cait_s24_384.fb_dist_in1k",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTS36,
        "torch_name": "cait_s36_384.fb_dist_in1k",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTM36,
        "torch_name": "cait_m36_384.fb_dist_in1k",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": cait.CaiTM48,
        "torch_name": "cait_m48_448.fb_dist_in1k",
        "input_shape": [448, 448, 3],
        "num_classes": 1000,
    },
]

for model_config in model_configs:
    torch_model_name: str = model_config["torch_name"]
    print(f"\n{'=' * 60}")
    print(f"Converting {torch_model_name}...")
    print(f"{'=' * 60}")

    keras_model: keras.Model = model_config["keras_cls"](
        include_top=True,
        input_shape=model_config["input_shape"],
        classifier_activation="linear",
        num_classes=model_config["num_classes"],
        include_normalization=False,
        weights=None,
    )

    torch_model: torch.nn.Module = timm.create_model(
        torch_model_name, pretrained=True
    ).eval()

    trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
        torch_model
    )
    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model
    )

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
        for keras_name_part, torch_name_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )
        torch_weight_name = re.sub(
            r"layerscale\.(\d+)\.variable(?:\.\d+)?", r"gamma_\1", torch_weight_name
        )

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name,
                keras_weight,
                torch_weights_dict,
                attn_weight_replacement,
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
                keras_weight_name,
                keras_weight.shape,
                torch_weight_name,
                torch_weight.shape,
            )

        transfer_weights(keras_weight_name, keras_weight, torch_weight)

    results = verify_cls_model_equivalence(
        model_a=torch_model,
        model_b=keras_model,
        input_shape=tuple(model_config["input_shape"]),
        output_specs={"num_classes": model_config["num_classes"]},
        run_performance=False,
        atol=1e-4,
        rtol=1e-4,
    )

    if not results["standard_input"]:
        raise ValueError(
            "Model equivalence test failed - model outputs do not match for standard input"
        )

    model_filename: str = f"{torch_model_name.replace('.', '_')}.weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")

    del keras_model, torch_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
