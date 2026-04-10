"""Convert HuggingFace DINOv3 checkpoints to kmodels Keras weights.

ViT variants from ``facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m``
ConvNeXt variants from ``facebook/dinov3-convnext-{tiny,small,base,large}-pretrain-lvd1689m``

The DINOv3 ViT uses separate Q/K/V projections (Q bias, no K bias, V bias)
unlike standard ViT which fuses them.  Register tokens and 2D RoPE are also
handled.  RoPE weights are not stored (they are computed at runtime).

Usage:
    python convert_dino_v3_hf_to_keras.py
"""

import gc
import re
from typing import Dict

import keras
import numpy as np
import torch
from tqdm import tqdm

from kmodels.models import dino_v3 as dino_v3_kmodels
from kmodels.models.dino_v3.config import DINOV3_HF_MODEL_IDS
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

# --------------------------------------------------------------------------- #
# Checkpoint registry
# --------------------------------------------------------------------------- #

DINOV3_VIT_VARIANTS = {
    "DinoV3ViTSmall16": {
        "hf_id": DINOV3_HF_MODEL_IDS["DinoV3ViTSmall16"],
        "ctor": dino_v3_kmodels.DinoV3ViTSmall16,
        "input_shape": (224, 224, 3),
        "save_name": "dinov3_vits16",
    },
    "DinoV3ViTBase16": {
        "hf_id": DINOV3_HF_MODEL_IDS["DinoV3ViTBase16"],
        "ctor": dino_v3_kmodels.DinoV3ViTBase16,
        "input_shape": (224, 224, 3),
        "save_name": "dinov3_vitb16",
    },
    "DinoV3ViTLarge16": {
        "hf_id": DINOV3_HF_MODEL_IDS["DinoV3ViTLarge16"],
        "ctor": dino_v3_kmodels.DinoV3ViTLarge16,
        "input_shape": (224, 224, 3),
        "save_name": "dinov3_vitl16",
    },
}

DINOV3_CONVNEXT_VARIANTS = {
    "DinoV3ConvNeXtTiny": {
        "hf_id": DINOV3_HF_MODEL_IDS["DinoV3ConvNeXtTiny"],
        "ctor": dino_v3_kmodels.DinoV3ConvNeXtTiny,
        "input_shape": (224, 224, 3),
        "save_name": "dinov3_convnext_tiny",
    },
    "DinoV3ConvNeXtSmall": {
        "hf_id": DINOV3_HF_MODEL_IDS["DinoV3ConvNeXtSmall"],
        "ctor": dino_v3_kmodels.DinoV3ConvNeXtSmall,
        "input_shape": (224, 224, 3),
        "save_name": "dinov3_convnext_small",
    },
    "DinoV3ConvNeXtBase": {
        "hf_id": DINOV3_HF_MODEL_IDS["DinoV3ConvNeXtBase"],
        "ctor": dino_v3_kmodels.DinoV3ConvNeXtBase,
        "input_shape": (224, 224, 3),
        "save_name": "dinov3_convnext_base",
    },
    "DinoV3ConvNeXtLarge": {
        "hf_id": DINOV3_HF_MODEL_IDS["DinoV3ConvNeXtLarge"],
        "ctor": dino_v3_kmodels.DinoV3ConvNeXtLarge,
        "input_shape": (224, 224, 3),
        "save_name": "dinov3_convnext_large",
    },
}

# --------------------------------------------------------------------------- #
# ViT name mapping helpers
# --------------------------------------------------------------------------- #

# Generic (non-attention) Keras -> HF name mapping for ViT.
# Applied sequentially to the keras_weight_name string.
DINOV3_VIT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "patch.embed": "embeddings.patch_embeddings.projection",
    "blocks.": "encoder.layer.",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "layernorm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def _resolve_vit_attention(keras_weight_path: str):
    """Map Q/K/V/proj weights to HF attention keys.

    DINOv3 Keras uses separate projections::

        blocks_{i}_attn_q_proj/{kernel,bias}
        blocks_{i}_attn_k_proj/kernel  (no bias)
        blocks_{i}_attn_v_proj/{kernel,bias}
        blocks_{i}_attn_proj/{kernel,bias}

    HF uses::

        encoder.layer.{i}.attention.attention.{query,key,value}.{weight,bias}
        encoder.layer.{i}.attention.output.dense.{weight,bias}
    """
    parts = keras_weight_path.split("/")
    layer_name = parts[-2]  # e.g. blocks_0_attn_q_proj
    var_name = parts[-1]  # e.g. kernel or bias

    m = re.match(r"blocks_(\d+)_attn_(q|k|v)_proj$", layer_name)
    if m:
        idx = int(m.group(1))
        qkv_map = {"q": "query", "k": "key", "v": "value"}
        role = qkv_map[m.group(2)]
        suffix = "weight" if "kernel" in var_name else "bias"
        return f"encoder.layer.{idx}.attention.attention.{role}.{suffix}"

    m = re.match(r"blocks_(\d+)_attn_proj$", layer_name)
    if m:
        idx = int(m.group(1))
        suffix = "weight" if "kernel" in var_name else "bias"
        return f"encoder.layer.{idx}.attention.output.dense.{suffix}"

    return None


def _resolve_vit_layer_scale(keras_weight_path: str):
    """Map ``blocks_{i}_layerscale_{1,2}/variable*`` to HF lambda1."""
    layer_name = keras_weight_path.split("/")[-2]
    m = re.match(r"blocks_(\d+)_layerscale_(1|2)$", layer_name)
    if m is None:
        return None
    idx = int(m.group(1))
    which = m.group(2)
    return f"encoder.layer.{idx}.layer_scale{which}.lambda1"


def _resolve_vit_cls_token(keras_weight_path: str):
    """Detect cls_token weight."""
    if (
        "cls_token" in keras_weight_path
        and "cls_token" in keras_weight_path.split("/")[-1]
    ):
        return "embeddings.cls_token"
    return None


def _resolve_vit_register_tokens(keras_weight_path: str):
    """Detect register_tokens weight."""
    if "register_tokens" in keras_weight_path:
        return "embeddings.register_tokens"
    return None


# --------------------------------------------------------------------------- #
# ViT weight transfer
# --------------------------------------------------------------------------- #


def _transfer_vit(
    keras_model: keras.Model, hf_state_dict: Dict[str, torch.Tensor]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable,
        total=len(trainable + non_trainable),
        desc="  ViT weights",
    ):
        path = keras_weight.path

        # --- CLS token ---
        hf_key = _resolve_vit_cls_token(path)
        if hf_key is not None:
            if hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, hf_key)
            keras_weight.assign(hf_state_dict[hf_key].numpy())
            continue

        # --- Register tokens ---
        hf_key = _resolve_vit_register_tokens(path)
        if hf_key is not None:
            if hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, hf_key)
            keras_weight.assign(hf_state_dict[hf_key].numpy())
            continue

        # --- Attention Q/K/V/proj ---
        if "_attn_" in path:
            hf_key = _resolve_vit_attention(path)
            if hf_key is None or hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(hf_key))
            transfer_weights(keras_weight_name, keras_weight, hf_state_dict[hf_key])
            continue

        # --- LayerScale ---
        if "_layerscale_" in path:
            hf_key = _resolve_vit_layer_scale(path)
            if hf_key is None or hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(hf_key))
            keras_weight.assign(hf_state_dict[hf_key].numpy())
            continue

        # --- Generic mapping (patch_embed, layer norms, MLP, final norm) ---
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


# --------------------------------------------------------------------------- #
# ConvNeXt name mapping
# --------------------------------------------------------------------------- #

DINOV3_CONVNEXT_NAME_MAPPING: Dict[str, str] = {
    "stem_conv_": "stem.0.",
    "stem_layernorm_": "stem.1.",
    "_": ".",
    "layernorm": "norm",
    "depthwise.conv": "conv_dw",
    "grn": "mlp.grn",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "conv.1": "mlp.fc1",
    "conv.2": "mlp.fc2",
    "downsampling.norm": "downsample.0",
    "downsampling.conv": "downsample.1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "final.norm": "head.norm",
    "predictions": "head.fc",
}


def _transfer_convnext(
    keras_model: keras.Model, hf_state_dict: Dict[str, torch.Tensor]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable,
        total=len(trainable + non_trainable),
        desc="  ConvNeXt weights",
    ):
        torch_name = keras_weight_name
        for old, new in DINOV3_CONVNEXT_NAME_MAPPING.items():
            torch_name = torch_name.replace(old, new)

        # LayerScale variable name
        torch_name = re.sub(r"layer\.scale\.variable(\.\d+)?", "gamma", torch_name)

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


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def _verify_vit(
    name: str,
    keras_model: keras.Model,
    hf_model: torch.nn.Module,
    input_shape,
) -> None:
    h, w, c = input_shape
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, c, h, w)).astype(np.float32)

    hf_model.eval()
    with torch.no_grad():
        hf_out = (
            hf_model(pixel_values=torch.from_numpy(x_np))
            .last_hidden_state.cpu()
            .numpy()
        )

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    keras_raw = keras_model(keras_in, training=False)
    keras_out = (
        keras_raw.detach().cpu().numpy()
        if hasattr(keras_raw, "detach")
        else np.asarray(keras_raw)
    )

    diff = float(np.abs(keras_out - hf_out).max())
    assert diff < 1e-3, f"{name}: max diff {diff:.2e}"
    print(f"  {name} OK (max diff = {diff:.2e})")


def _verify_convnext(
    name: str,
    keras_model: keras.Model,
    hf_model: torch.nn.Module,
    input_shape,
) -> None:
    h, w, c = input_shape
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, c, h, w)).astype(np.float32)

    hf_model.eval()
    with torch.no_grad():
        hf_out = (
            hf_model(pixel_values=torch.from_numpy(x_np))
            .last_hidden_state.cpu()
            .numpy()
        )

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    keras_raw = keras_model(keras_in, training=False)
    keras_out = (
        keras_raw.detach().cpu().numpy()
        if hasattr(keras_raw, "detach")
        else np.asarray(keras_raw)
    )

    diff = float(np.abs(keras_out - hf_out).max())
    assert diff < 1e-3, f"{name}: max diff {diff:.2e}"
    print(f"  {name} OK (max diff = {diff:.2e})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    # Import HF model classes lazily
    from transformers import AutoModel

    # --- ViT variants ---
    for name, info in DINOV3_VIT_VARIANTS.items():
        print("\n" + "=" * 60)
        print(f"Converting ViT: {name}  <-  {info['hf_id']}")
        print("=" * 60)

        hf_model = AutoModel.from_pretrained(info["hf_id"]).eval()
        hf_state_dict = dict(hf_model.state_dict())

        keras_model = info["ctor"](
            include_top=False,
            include_normalization=False,
            input_shape=info["input_shape"],
            weights=None,
        )

        _transfer_vit(keras_model, hf_state_dict)
        _verify_vit(name, keras_model, hf_model, info["input_shape"])

        out_path = f"{info['save_name']}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, hf_model, hf_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- ConvNeXt variants ---
    for name, info in DINOV3_CONVNEXT_VARIANTS.items():
        print("\n" + "=" * 60)
        print(f"Converting ConvNeXt: {name}  <-  {info['hf_id']}")
        print("=" * 60)

        hf_model = AutoModel.from_pretrained(info["hf_id"]).eval()
        hf_state_dict = dict(hf_model.state_dict())

        keras_model = info["ctor"](
            include_top=False,
            include_normalization=False,
            input_shape=info["input_shape"],
            weights=None,
        )

        _transfer_convnext(keras_model, hf_state_dict)
        _verify_convnext(name, keras_model, hf_model, info["input_shape"])

        out_path = f"{info['save_name']}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, hf_model, hf_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
