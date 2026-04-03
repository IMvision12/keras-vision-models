import gc
from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kmodels.models import swinv2
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.model_equivalence_tester import verify_cls_model_equivalence
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "cpb_mlp": "CPB§MLPx",
    "logit_scale": "LOGIT§SCALE",
    "q_bias": "Q§BIAS",
    "v_bias": "V§BIAS",
    "relative_coords_table": "REL§COORDS§TABLE",
    "relative_position_index": "REL§POS§INDEX",
    "moving_variance": "MOVVAR",
    "moving_mean": "MOVMEAN",
    "_": ".",
    "CPB§MLPx": "cpb_mlp",
    "LOGIT§SCALE": "logit_scale",
    "Q§BIAS": "q_bias",
    "V§BIAS": "v_bias",
    "REL§COORDS§TABLE": "relative_coords_table",
    "REL§POS§INDEX": "relative_position_index",
    "MOVVAR": "running_var",
    "MOVMEAN": "running_mean",
    "stem.conv": "patch_embed.proj",
    "stem.norm": "patch_embed.norm",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "dense.1": "fc1",
    "dense.2": "fc2",
    "pm.layernorm": "norm",
    "pm.dense": "reduction",
    "final.norm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "predictions": "head.fc",
}

KERAS_MODEL_CLS_MAP = {
    "SwinV2TinyW8": swinv2.SwinV2TinyW8,
    "SwinV2TinyW16": swinv2.SwinV2TinyW16,
    "SwinV2SmallW8": swinv2.SwinV2SmallW8,
    "SwinV2SmallW16": swinv2.SwinV2SmallW16,
    "SwinV2BaseW8": swinv2.SwinV2BaseW8,
    "SwinV2BaseW12": swinv2.SwinV2BaseW12,
    "SwinV2BaseW16": swinv2.SwinV2BaseW16,
    "SwinV2LargeW12": swinv2.SwinV2LargeW12,
}

SIZE_NAME_MAP = {
    "SwinV2TinyW8": ("tiny", 8),
    "SwinV2TinyW16": ("tiny", 16),
    "SwinV2SmallW8": ("small", 8),
    "SwinV2SmallW16": ("small", 16),
    "SwinV2BaseW8": ("base", 8),
    "SwinV2BaseW12": ("base", 12),
    "SwinV2BaseW16": ("base", 16),
    "SwinV2LargeW12": ("large", 12),
}

# ft_in1k target resolution -> window_size override
FT_WINDOW_SIZE_MAP = {256: 16, 384: 24}

ATTN_NAME_REPLACEMENTS = {
    "cpb.mlp": "cpb_mlp",
}


def build_model_configs() -> List[Dict[str, Union[type, str, List[int], int, bool]]]:
    """Build a list of model configs from SWINV2_WEIGHTS_CONFIG."""
    from kmodels.models.swinv2.config import SWINV2_MODEL_CONFIG, SWINV2_WEIGHTS_CONFIG

    configs = []

    for model_name, variants in SWINV2_WEIGHTS_CONFIG.items():
        model_cfg = SWINV2_MODEL_CONFIG[model_name]
        size, ws = SIZE_NAME_MAP[model_name]
        res = model_cfg["pretrain_size"]

        for variant_name in variants:
            if variant_name == "ms_in1k":
                torch_model_name = f"swinv2_{size}_window{ws}_{res}.ms_in1k"
                input_size = res
                overrides = {}
            elif variant_name.startswith("ms_in1k_ft_in1k_"):
                target_res = int(variant_name.split("_")[-1])
                target_ws = FT_WINDOW_SIZE_MAP[target_res]
                torch_model_name = (
                    f"swinv2_{size}_window{ws}to{target_ws}"
                    f"_{res}to{target_res}.ms_in1k_ft_in1k"
                )
                input_size = target_res
                overrides = {"window_size": target_ws}
            else:
                raise ValueError(f"Unknown variant: {variant_name}")

            configs.append(
                {
                    "keras_model_cls": KERAS_MODEL_CLS_MAP[model_name],
                    "torch_model_name": torch_model_name,
                    "input_shape": [input_size, input_size, 3],
                    "num_classes": 1000,
                    "include_top": True,
                    "include_normalization": False,
                    "classifier_activation": "linear",
                    "overrides": overrides,
                }
            )

    return configs


def convert_model(
    model_config: Dict[str, Union[type, str, List[int], int, bool]],
) -> None:
    torch_model_name = model_config["torch_model_name"]
    input_shape = model_config["input_shape"]
    num_classes = model_config["num_classes"]
    overrides = model_config.get("overrides", {})

    print(f"\n{'=' * 60}")
    print(f"Converting: {torch_model_name}")
    print(f"  Input shape: {input_shape}, Num classes: {num_classes}")
    if overrides:
        print(f"  Overrides: {overrides}")
    print(f"{'=' * 60}")

    keras_model: keras.Model = model_config["keras_model_cls"](
        include_top=model_config["include_top"],
        input_shape=model_config["input_shape"],
        classifier_activation=model_config["classifier_activation"],
        num_classes=model_config["num_classes"],
        include_normalization=model_config["include_normalization"],
        weights=None,
        **overrides,
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
        if "relative_coords_table" in keras_weight_name:
            continue
        if "relative_position_index" in keras_weight_name:
            continue

        if "swin_v2_attention" in keras_weight_name:
            path_parts = keras_weight.path.split("/")

            if len(path_parts) == 2:
                weight_name = path_parts[-1]
                torch_name = weight_name.replace("_", ".")
                for old, new in ATTN_NAME_REPLACEMENTS.items():
                    torch_name = torch_name.replace(old, new)
                torch_name = torch_name.replace("logit.scale", "logit_scale")
                torch_name = torch_name.replace("q.bias", "q_bias")
                torch_name = torch_name.replace("v.bias", "v_bias")

                torch_w = torch_weights_dict[torch_name]
                if isinstance(torch_w, torch.Tensor):
                    torch_w = torch_w.detach().cpu().numpy()
                if "logit_scale" in weight_name:
                    keras_weight.assign(torch_w.squeeze())
                else:
                    keras_weight.assign(torch_w)
            else:
                transfer_attention_weights(
                    keras_weight_name,
                    keras_weight,
                    torch_weights_dict,
                    name_replacements=ATTN_NAME_REPLACEMENTS,
                )
            continue

        torch_weight_name: str = keras_weight_name
        for keras_name_part, torch_name_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )

        if torch_weight_name not in torch_weights_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

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
        input_shape=tuple(input_shape),
        output_specs={"num_classes": num_classes},
        run_performance=False,
    )

    if not results["standard_input"]:
        raise ValueError(
            f"Model equivalence test failed for {torch_model_name} "
            "- model outputs do not match for standard input"
        )

    model_filename: str = f"{torch_model_name.replace('.', '_')}.weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")

    del keras_model, torch_model, torch_weights_dict
    del trainable_torch_weights, non_trainable_torch_weights
    del trainable_keras_weights, non_trainable_keras_weights
    keras.backend.clear_session()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    all_configs = build_model_configs()
    print(f"Total models to convert: {len(all_configs)}")
    for config in all_configs:
        convert_model(config)
    print("\nAll models converted successfully!")
