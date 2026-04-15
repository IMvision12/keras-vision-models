import gc
from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kmodels.models import swin
from kmodels.models.swin.config import SWIN_MODEL_CONFIG, SWIN_WEIGHTS_CONFIG
from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.model_equivalence_tester import verify_cls_model_equivalence
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
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
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}

KERAS_MODEL_CLS_MAP = {
    "SwinTinyP4W7": swin.SwinTinyP4W7,
    "SwinSmallP4W7": swin.SwinSmallP4W7,
    "SwinBaseP4W7": swin.SwinBaseP4W7,
    "SwinBaseP4W12": swin.SwinBaseP4W12,
    "SwinLargeP4W7": swin.SwinLargeP4W7,
    "SwinLargeP4W12": swin.SwinLargeP4W12,
}

SIZE_NAME_MAP = {
    "SwinTinyP4W7": "tiny",
    "SwinSmallP4W7": "small",
    "SwinBaseP4W7": "base",
    "SwinBaseP4W12": "base",
    "SwinLargeP4W7": "large",
    "SwinLargeP4W12": "large",
}


def build_model_configs() -> List[Dict[str, Union[type, str, List[int], int, bool]]]:
    """Build a list of model configs from SWIN_WEIGHTS_CONFIG."""
    configs = []
    for model_name, variants in SWIN_WEIGHTS_CONFIG.items():
        model_cfg = SWIN_MODEL_CONFIG[model_name]
        size = SIZE_NAME_MAP[model_name]
        res = model_cfg["pretrain_size"]
        ws = model_cfg["window_size"]

        for variant_name in variants:
            num_classes = 21841 if variant_name.endswith("in22k") else 1000
            torch_model_name = f"swin_{size}_patch4_window{ws}_{res}.{variant_name}"
            configs.append(
                {
                    "keras_model_cls": KERAS_MODEL_CLS_MAP[model_name],
                    "torch_model_name": torch_model_name,
                    "input_shape": [res, res, 3],
                    "num_classes": num_classes,
                    "include_top": True,
                    "include_normalization": False,
                    "classifier_activation": "linear",
                }
            )
    return configs


def convert_model(
    model_config: Dict[str, Union[type, str, List[int], int, bool]],
) -> None:
    """Convert a single Swin variant from PyTorch to Keras."""
    torch_model_name = model_config["torch_model_name"]
    input_shape = model_config["input_shape"]
    num_classes = model_config["num_classes"]

    print(f"\n{'=' * 60}")
    print(f"Converting: {torch_model_name}")
    print(f"  Input shape: {input_shape}, Num classes: {num_classes}")
    print(f"{'=' * 60}")

    keras_model: keras.Model = model_config["keras_model_cls"](
        include_top=model_config["include_top"],
        input_shape=model_config["input_shape"],
        classifier_activation=model_config["classifier_activation"],
        num_classes=model_config["num_classes"],
        include_normalization=model_config["include_normalization"],
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

        if "relative.position.bias.table" in torch_weight_name:
            layer_name = keras_weight.path.split("/")[-1]
            layer_name = layer_name.replace("_", ".").replace(
                "relative.position.bias.table", "relative_position_bias_table"
            )
            relative_weight = torch_weights_dict[layer_name]
            keras_weight.assign(relative_weight)
            continue

        if "window.attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, torch_weights_dict
            )
            continue

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
