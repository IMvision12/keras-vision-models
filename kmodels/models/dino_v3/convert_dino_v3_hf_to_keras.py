import re
from typing import Dict

import numpy as np

from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

DINOV3_VIT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "patch.embed": "embeddings.patch_embeddings",
    "blocks.": "model.layer.",
    "dense.1": "mlp.up_proj",
    "dense.2": "mlp.down_proj",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def _resolve_vit_attention(keras_weight_path: str):
    parts = keras_weight_path.split("/")
    layer_name = parts[-2]
    var_name = parts[-1]

    m = re.match(r"blocks_(\d+)_attn_(q|k|v)_proj$", layer_name)
    if m:
        idx = int(m.group(1))
        suffix = "weight" if "kernel" in var_name else "bias"
        return f"model.layer.{idx}.attention.{m.group(2)}_proj.{suffix}"

    m = re.match(r"blocks_(\d+)_attn_proj$", layer_name)
    if m:
        idx = int(m.group(1))
        suffix = "weight" if "kernel" in var_name else "bias"
        return f"model.layer.{idx}.attention.o_proj.{suffix}"

    return None


def _resolve_vit_layer_scale(keras_weight_path: str):
    layer_name = keras_weight_path.split("/")[-2]
    m = re.match(r"blocks_(\d+)_layerscale_(1|2)$", layer_name)
    if m is None:
        return None
    idx = int(m.group(1))
    which = m.group(2)
    return f"model.layer.{idx}.layer_scale{which}.lambda1"


def transfer_dinov3_vit_weights(keras_model, hf_state_dict):
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        path = keras_weight.path

        if "cls_token" in path and "cls_token" in path.split("/")[-1]:
            hf_key = "embeddings.cls_token"
            if hf_key in hf_state_dict:
                keras_weight.assign(hf_state_dict[hf_key])
                continue

        if "register_tokens" in path:
            hf_key = "embeddings.register_tokens"
            if hf_key in hf_state_dict:
                keras_weight.assign(hf_state_dict[hf_key])
                continue

        if "_attn_" in path:
            hf_key = _resolve_vit_attention(path)
            if hf_key is None or hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(hf_key))
            transfer_weights(keras_weight_name, keras_weight, hf_state_dict[hf_key])
            continue

        if "_layerscale_" in path:
            hf_key = _resolve_vit_layer_scale(path)
            if hf_key is None or hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(hf_key))
            keras_weight.assign(hf_state_dict[hf_key])
            continue

        if "patch_embed" in path and len(keras_weight.shape) == 4:
            hf_key = "embeddings.patch_embeddings.weight"
            if hf_key in hf_state_dict:
                transfer_weights("conv_kernel", keras_weight, hf_state_dict[hf_key])
                continue

        torch_name = keras_weight_name
        for old, new in DINOV3_VIT_NAME_MAPPING.items():
            torch_name = torch_name.replace(old, new)

        if torch_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_name)

        torch_weight = hf_state_dict[torch_name]
        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name, keras_weight.shape, torch_name, torch_weight.shape
            )
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


_VAR_MAP = {"kernel": "weight", "gamma": "weight", "bias": "bias", "beta": "bias"}


def _keras_to_hf_convnext(keras_name: str):
    m = re.match(r"stem_conv_(kernel|bias)", keras_name)
    if m:
        return f"model.stages.0.downsample_layers.0.{_VAR_MAP[m.group(1)]}"

    m = re.match(r"stem_layernorm_(gamma|beta)", keras_name)
    if m:
        return f"model.stages.0.downsample_layers.1.{_VAR_MAP[m.group(1)]}"

    m = re.match(r"stages_(\d+)_downsampling_layernorm_(gamma|beta)", keras_name)
    if m:
        stage = int(m.group(1))
        return f"model.stages.{stage}.downsample_layers.0.{_VAR_MAP[m.group(2)]}"

    m = re.match(r"stages_(\d+)_downsampling_conv_(kernel|bias)", keras_name)
    if m:
        stage = int(m.group(1))
        return f"model.stages.{stage}.downsample_layers.1.{_VAR_MAP[m.group(2)]}"

    m = re.match(r"stages_(\d+)_blocks_(\d+)_layer_scale_variable", keras_name)
    if m:
        return f"model.stages.{m.group(1)}.layers.{m.group(2)}.gamma"

    m = re.match(r"stages_(\d+)_blocks_(\d+)_depthwise_conv_(kernel|bias)", keras_name)
    if m:
        return f"model.stages.{m.group(1)}.layers.{m.group(2)}.depthwise_conv.{_VAR_MAP[m.group(3)]}"

    m = re.match(r"stages_(\d+)_blocks_(\d+)_layernorm_(gamma|beta)", keras_name)
    if m:
        return f"model.stages.{m.group(1)}.layers.{m.group(2)}.layer_norm.{_VAR_MAP[m.group(3)]}"

    m = re.match(r"stages_(\d+)_blocks_(\d+)_conv_1_(kernel|bias)", keras_name)
    if m:
        return f"model.stages.{m.group(1)}.layers.{m.group(2)}.pointwise_conv1.{_VAR_MAP[m.group(3)]}"

    m = re.match(r"stages_(\d+)_blocks_(\d+)_conv_2_(kernel|bias)", keras_name)
    if m:
        return f"model.stages.{m.group(1)}.layers.{m.group(2)}.pointwise_conv2.{_VAR_MAP[m.group(3)]}"

    m = re.match(r"final_layernorm_(gamma|beta)", keras_name)
    if m:
        return f"layer_norm.{_VAR_MAP[m.group(1)]}"

    return None


def transfer_dinov3_convnext_weights(keras_model, hf_state_dict):
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        hf_key = _keras_to_hf_convnext(keras_weight_name)
        if hf_key is None or hf_key not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, str(hf_key))

        hf_w = hf_state_dict[hf_key]

        if "layer_scale" in keras_weight_name:
            keras_weight.assign(hf_w)

        elif (
            "pointwise" in hf_key
            and len(hf_w.shape) == 2
            and len(keras_weight.shape) == 4
        ):
            keras_weight.assign(hf_w.T[np.newaxis, np.newaxis, :, :])
        else:
            transfer_weights(keras_weight_name, keras_weight, hf_w)


if __name__ == "__main__":
    import gc
    import os

    import keras
    import numpy as np
    import torch
    from transformers import AutoModel

    from kmodels.models.dino_v3.config import DINOV3_HF_MODEL_IDS
    from kmodels.models.dino_v3.dino_v3_model import (
        DinoV3ConvNeXtBase,
        DinoV3ConvNeXtLarge,
        DinoV3ConvNeXtSmall,
        DinoV3ConvNeXtTiny,
        DinoV3ViTBase16,
        DinoV3ViTLarge16,
        DinoV3ViTSmall16,
    )

    HF_TOKEN = os.environ.get("HF_TOKEN")

    VIT_VARIANTS = [
        ("DinoV3ViTSmall16", DinoV3ViTSmall16, "dinov3_vits16"),
        ("DinoV3ViTBase16", DinoV3ViTBase16, "dinov3_vitb16"),
        ("DinoV3ViTLarge16", DinoV3ViTLarge16, "dinov3_vitl16"),
    ]

    CONVNEXT_VARIANTS = [
        ("DinoV3ConvNeXtTiny", DinoV3ConvNeXtTiny, "dinov3_convnext_tiny"),
        ("DinoV3ConvNeXtSmall", DinoV3ConvNeXtSmall, "dinov3_convnext_small"),
        ("DinoV3ConvNeXtBase", DinoV3ConvNeXtBase, "dinov3_convnext_base"),
        ("DinoV3ConvNeXtLarge", DinoV3ConvNeXtLarge, "dinov3_convnext_large"),
    ]

    for name, ctor, save_name in VIT_VARIANTS:
        hf_id = DINOV3_HF_MODEL_IDS[name]
        print(f"\n{'=' * 60}")
        print(f"Converting ViT: {name}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = AutoModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = ctor(
            include_top=False,
            include_normalization=False,
            input_shape=(224, 224, 3),
            weights=None,
        )
        transfer_dinov3_vit_weights(keras_model, hf_sd)

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.no_grad():
            hf_out = (
                hf_model(pixel_values=torch.from_numpy(x_np))
                .last_hidden_state.cpu()
                .numpy()
            )
        k_in = np.transpose(x_np, (0, 2, 3, 1))
        k_out = keras_model(k_in, training=False)
        k_out = (
            k_out.detach().cpu().numpy()
            if hasattr(k_out, "detach")
            else np.asarray(k_out)
        )
        diff = float(np.abs(k_out - hf_out).max())
        assert diff < 1e-2, f"{name}: max diff {diff:.2e}"
        print(f"  Verification OK (max diff = {diff:.2e})")

        out = f"{save_name}.weights.h5"
        keras_model.save_weights(out)
        print(f"  Saved -> {out}")

        del keras_model, hf_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for name, ctor, save_name in CONVNEXT_VARIANTS:
        hf_id = DINOV3_HF_MODEL_IDS[name]
        print(f"\n{'=' * 60}")
        print(f"Converting ConvNeXt: {name}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = AutoModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = ctor(
            include_top=False,
            include_normalization=False,
            input_shape=(224, 224, 3),
            weights=None,
        )
        transfer_dinov3_convnext_weights(keras_model, hf_sd)

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.no_grad():
            hf_out_obj = hf_model(
                pixel_values=torch.from_numpy(x_np), output_hidden_states=True
            )

            hf_feat = hf_out_obj.hidden_states[-1].permute(0, 2, 3, 1).cpu().numpy()
        k_in = np.transpose(x_np, (0, 2, 3, 1))
        k_out = keras_model(k_in, training=False)
        k_out = (
            k_out.detach().cpu().numpy()
            if hasattr(k_out, "detach")
            else np.asarray(k_out)
        )
        diff = float(np.abs(k_out - hf_feat).max())
        assert diff < 1e-2, f"{name}: max diff {diff:.2e}"
        print(f"  Verification OK (max diff = {diff:.2e})")

        out = f"{save_name}.weights.h5"
        keras_model.save_weights(out)
        print(f"  Saved -> {out}")

        del keras_model, hf_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
