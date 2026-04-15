from typing import Dict, List, Union

import keras
import numpy as np
import timm
import torch
from tqdm import tqdm

from kmodels.models import maxvit
from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.model_equivalence_tester import verify_cls_model_equivalence
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "relative_position_bias_table": "RPBT",
    "moving_variance": "MOVVAR",
    "moving_mean": "MOVMEAN",
    "attn_block": "ATTNBLOCK",
    "attn_grid": "ATTNGRID",
    "shortcut_expand": "SHORTCUTEXPAND",
    "pre_logits": "PRELOGITS",
    "pre_norm": "PRENORM",
    "conv1_1x1": "CONV11X1",
    "conv2_kxk": "CONV2KXK",
    "conv3_1x1": "CONV31X1",
    "rel_pos": "RELPOS",
    "se_fc": "SEFC",
    "attn_qkv": "ATTNQKV",
    "attn_proj": "ATTNPROJ",
    "mlp_fc": "MLPFC",
    "_": ".",
    "RPBT": "relative_position_bias_table",
    "MOVVAR": "running_var",
    "MOVMEAN": "running_mean",
    "ATTNBLOCK": "attn_block",
    "ATTNGRID": "attn_grid",
    "SHORTCUTEXPAND": "shortcut.expand",
    "PRELOGITS": "pre_logits",
    "PRENORM": "pre_norm",
    "CONV11X1": "conv1_1x1",
    "CONV2KXK": "conv2_kxk",
    "CONV31X1": "conv3_1x1",
    "RELPOS": "rel_pos",
    "SEFC": "conv.se.fc",
    "ATTNQKV": "attn.qkv",
    "ATTNPROJ": "attn.proj",
    "MLPFC": "mlp.fc",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

_WS = {224: 7, 384: 12, 512: 16}


def _cfg(cls, torch_name, res, num_classes=1000):
    return {
        "keras_cls": cls,
        "torch_name": torch_name,
        "input_shape": [res, res, 3],
        "num_classes": num_classes,
        "window_size": _WS[res],
    }


model_configs: List[Dict[str, Union[type, str, list, int]]] = [
    _cfg(maxvit.MaxViTXLarge, "maxvit_xlarge_tf_384.in21k_ft_in1k", 384),
    _cfg(maxvit.MaxViTXLarge, "maxvit_xlarge_tf_512.in21k_ft_in1k", 512),
]

if __name__ == "__main__":
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
            window_size=model_config["window_size"],
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

        torch_weights_dict: Dict[str, "np.ndarray"] = {}
        for k, v in {**trainable_torch_weights, **non_trainable_torch_weights}.items():
            torch_weights_dict[k] = (
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            )

        all_keras_weights = []
        for layer in keras_model.layers:
            for w in layer.weights:
                path = w.path
                parts = path.split("/")
                layer_name = parts[-2] if len(parts) >= 2 else parts[0]
                weight_suffix = parts[-1]
                keras_weight_name = f"{layer_name}_{weight_suffix}"
                all_keras_weights.append((w, keras_weight_name))

        for keras_weight, keras_weight_name in tqdm(
            all_keras_weights,
            total=len(all_keras_weights),
            desc="Transferring weights",
        ):
            torch_weight_name: str = keras_weight_name
            for keras_name_part, torch_name_part in weight_name_mapping.items():
                torch_weight_name = torch_weight_name.replace(
                    keras_name_part, torch_name_part
                )

            if torch_weight_name not in torch_weights_dict:
                raise WeightMappingError(keras_weight_name, torch_weight_name)

            torch_weight = torch_weights_dict[torch_weight_name]

            if not compare_keras_torch_names(
                keras_weight_name, keras_weight, torch_weight_name, torch_weight
            ):
                raise WeightShapeMismatchError(
                    keras_weight_name,
                    keras_weight.shape,
                    torch_weight_name,
                    torch_weight.shape,
                )

            transfer_name = keras_weight_name
            if "conv2_kxk" in keras_weight_name:
                transfer_name = "dwconv_" + keras_weight_name
            elif "se_fc" in keras_weight_name:
                transfer_name = "conv_" + keras_weight_name
            transfer_weights(transfer_name, keras_weight, torch_weight)

        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=tuple(model_config["input_shape"]),
            output_specs={"num_classes": model_config["num_classes"]},
            run_performance=False,
            atol=1e-3,
            rtol=1e-3,
        )

        if not results["standard_input"]:
            raise ValueError(
                f"Model equivalence test failed for {torch_model_name} - "
                "model outputs do not match for standard input"
            )

        model_filename: str = f"{torch_model_name.replace('.', '_')}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"Model saved successfully as {model_filename}")

        del keras_model, torch_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
