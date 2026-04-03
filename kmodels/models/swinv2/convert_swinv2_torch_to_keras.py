import re
from typing import Dict, List, Union

import keras
import numpy as np
import timm
import torch
from tqdm import tqdm

from kmodels.models import swinv2
from kmodels.utils.model_equivalence_tester import verify_cls_model_equivalence


def _convert_layer_name(name):
    """Convert a keras layer/sublayer name to torch module path."""
    if name == "stem_conv":
        return "patch_embed.proj"
    if name == "stem_norm":
        return "patch_embed.norm"
    if name == "final_norm":
        return "norm"
    if name == "predictions":
        return "head.fc"

    m = re.match(r"layers_(\d+)_downsample_pm_dense", name)
    if m:
        return f"layers.{m.group(1)}.downsample.reduction"

    m = re.match(r"layers_(\d+)_downsample_pm_layernorm", name)
    if m:
        return f"layers.{m.group(1)}.downsample.norm"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_layernorm_(\d+)", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.norm{m.group(3)}"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_mlp_dense_(\d+)", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.mlp.fc{m.group(3)}"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_attn_qkv", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.attn.qkv"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_attn_proj", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.attn.proj"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_attn_cpb_mlp_(\d+)", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.attn.cpb_mlp.{m.group(3)}"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_attn_logit_scale", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.attn.logit_scale"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_attn_q_bias", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.attn.q_bias"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_attn_v_bias", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.attn.v_bias"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_attn_relative_coords_table", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.attn.relative_coords_table"

    m = re.match(r"layers_(\d+)_blocks_(\d+)_attn_relative_position_index", name)
    if m:
        return f"layers.{m.group(1)}.blocks.{m.group(2)}.attn.relative_position_index"

    raise ValueError(f"Cannot convert layer name: {name}")


def _path_to_torch(path):
    """Convert keras weight path to torch state_dict key."""
    parts = path.split("/")
    wtype_map = {"kernel": "weight", "bias": "bias", "gamma": "weight", "beta": "bias"}

    if len(parts) == 3:
        _layer, sublayer, wtype = parts
        torch_wtype = wtype_map.get(wtype, wtype)
        torch_name = _convert_layer_name(sublayer)
        return f"{torch_name}.{torch_wtype}"

    elif len(parts) == 2:
        layer, wtype = parts

        if layer.startswith("swin_v2_attention"):
            if "relative_coords_table" in wtype or "relative_position_index" in wtype:
                return "__skip__"
            torch_name = _convert_layer_name(wtype)
            return torch_name
        else:
            torch_wtype = wtype_map.get(wtype, wtype)
            torch_name = _convert_layer_name(layer)
            return f"{torch_name}.{torch_wtype}"

    else:
        raise ValueError(f"Unexpected path: {path}")


def _assign_weight(keras_weight, path, torch_numpy):
    """Assign a torch numpy weight to a keras weight, handling transpositions."""
    k_shape = keras_weight.shape
    t_shape = torch_numpy.shape

    if "logit_scale" in path:
        keras_weight.assign(torch_numpy.squeeze())
        return

    if "q_bias" in path or "v_bias" in path:
        keras_weight.assign(torch_numpy)
        return

    if len(k_shape) == 4 and len(t_shape) == 4:
        keras_weight.assign(np.transpose(torch_numpy, (2, 3, 1, 0)))
        return

    if len(k_shape) == 2 and len(t_shape) == 2:
        if k_shape[0] == t_shape[1] and k_shape[1] == t_shape[0]:
            keras_weight.assign(torch_numpy.T)
        elif k_shape == t_shape:
            keras_weight.assign(torch_numpy)
        else:
            raise ValueError(
                f"Shape mismatch for {path}: keras={k_shape} torch={t_shape}"
            )
        return

    if len(k_shape) == 1 and len(t_shape) == 1:
        keras_weight.assign(torch_numpy)
        return

    if k_shape == t_shape:
        keras_weight.assign(torch_numpy)
        return

    raise ValueError(
        f"Cannot handle shapes for {path}: keras={k_shape} torch={t_shape}"
    )


def _cfg(cls, torch_name, input_size, num_classes=1000, **overrides):
    return {
        "keras_cls": cls,
        "torch_name": torch_name,
        "input_size": input_size,
        "num_classes": num_classes,
        "overrides": overrides,
    }


model_configs: List[Dict[str, Union[type, str, int, dict]]] = [
    # Tiny
    _cfg(swinv2.SwinV2TinyW8, "swinv2_tiny_window8_256.ms_in1k", 256),
    _cfg(swinv2.SwinV2TinyW16, "swinv2_tiny_window16_256.ms_in1k", 256),
    # Small
    _cfg(swinv2.SwinV2SmallW8, "swinv2_small_window8_256.ms_in1k", 256),
    _cfg(swinv2.SwinV2SmallW16, "swinv2_small_window16_256.ms_in1k", 256),
    # Base
    _cfg(swinv2.SwinV2BaseW8, "swinv2_base_window8_256.ms_in1k", 256),
    _cfg(swinv2.SwinV2BaseW12, "swinv2_base_window12_192.ms_in1k", 192),
    _cfg(
        swinv2.SwinV2BaseW12,
        "swinv2_base_window12to16_192to256.ms_in1k_ft_in1k",
        256,
        window_size=16,
    ),
    _cfg(
        swinv2.SwinV2BaseW12,
        "swinv2_base_window12to24_192to384.ms_in1k_ft_in1k",
        384,
        window_size=24,
    ),
    _cfg(swinv2.SwinV2BaseW16, "swinv2_base_window16_256.ms_in1k", 256),
    # Large
    _cfg(swinv2.SwinV2LargeW12, "swinv2_large_window12_192.ms_in1k", 192),
    _cfg(
        swinv2.SwinV2LargeW12,
        "swinv2_large_window12to16_192to256.ms_in1k_ft_in1k",
        256,
        window_size=16,
    ),
    _cfg(
        swinv2.SwinV2LargeW12,
        "swinv2_large_window12to24_192to384.ms_in1k_ft_in1k",
        384,
        window_size=24,
    ),
]


if __name__ == "__main__":
    for model_config in model_configs:
        torch_model_name = model_config["torch_name"]
        input_size = model_config["input_size"]
        num_classes = model_config["num_classes"]
        overrides = model_config["overrides"]

        print(f"\n{'=' * 60}")
        print(f"Converting {torch_model_name}...")
        print(f"{'=' * 60}")

        keras_model: keras.Model = model_config["keras_cls"](
            include_top=True,
            input_shape=[input_size, input_size, 3],
            classifier_activation="linear",
            num_classes=num_classes,
            include_normalization=False,
            weights=None,
            **overrides,
        )

        torch_model: torch.nn.Module = timm.create_model(
            torch_model_name, pretrained=True
        ).eval()

        torch_state = dict(torch_model.state_dict())
        for bname, buf in torch_model.named_buffers():
            if bname not in torch_state:
                torch_state[bname] = buf

        for w in tqdm(keras_model.weights, desc="Transferring weights"):
            path = w.path
            torch_name = _path_to_torch(path)

            if torch_name == "__skip__":
                continue

            if torch_name not in torch_state:
                raise ValueError(
                    f"Could not find torch weight '{torch_name}' "
                    f"for keras path '{path}'"
                )

            torch_w = torch_state[torch_name]
            if isinstance(torch_w, torch.Tensor):
                torch_w = torch_w.detach().cpu().numpy()

            _assign_weight(w, path, torch_w)

        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=(input_size, input_size, 3),
            output_specs={"num_classes": num_classes},
            run_performance=False,
        )

        if not results["standard_input"]:
            raise ValueError(
                f"Model equivalence test failed for {torch_model_name} - "
                "model outputs do not match for standard input"
            )

        model_filename = f"{torch_model_name.replace('.', '_')}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"Model saved successfully as {model_filename}")

        del keras_model, torch_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
