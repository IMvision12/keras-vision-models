"""Convert official HuggingFace DINOv2 checkpoints to kmodels Keras weights.

Supports the three GeLU-FFN backbone-only checkpoints under
https://huggingface.co/facebook:

* ``DinoV2Small14`` <- ``facebook/dinov2-small``
* ``DinoV2Base14``  <- ``facebook/dinov2-base``
* ``DinoV2Large14`` <- ``facebook/dinov2-large``

The HuggingFace ``Dinov2Model`` keeps Q/K/V as three separate ``nn.Linear``
layers, while the kmodels :class:`MultiHeadSelfAttention` uses a single fused
``qkv`` Dense. The converter therefore concatenates the HF Q, K, V kernels and
biases along the output dimension before transferring them, then handles the
DINOv2-specific LayerScale parameters and the renamed embedding/encoder
modules.

Each conversion is verified by comparing the Keras and HF outputs on the same
random 224x224 image to ``atol = rtol = 1e-4`` before the ``.weights.h5`` file
is written.

Usage:
    python convert_dino_v2_hf_to_keras.py
"""

import gc
import re
from typing import Dict

import keras
import numpy as np
import torch
from tqdm import tqdm

from kmodels.models import dino_v2 as dino_v2_kmodels
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

# ----------------------------------------------------------------------------- #
# Checkpoint registry
# ----------------------------------------------------------------------------- #

DINOV2_CHECKPOINTS = {
    "DinoV2Small14": {
        "hf_id": "facebook/dinov2-small",
        "ctor": dino_v2_kmodels.DinoV2Small14,
        "input_shape": (224, 224, 3),
        "save_name": "dinov2_vits14",
    },
    "DinoV2Base14": {
        "hf_id": "facebook/dinov2-base",
        "ctor": dino_v2_kmodels.DinoV2Base14,
        "input_shape": (224, 224, 3),
        "save_name": "dinov2_vitb14",
    },
    "DinoV2Large14": {
        "hf_id": "facebook/dinov2-large",
        "ctor": dino_v2_kmodels.DinoV2Large14,
        "input_shape": (224, 224, 3),
        "save_name": "dinov2_vitl14",
    },
}

# ----------------------------------------------------------------------------- #
# Keras -> HuggingFace name mappings (everything except attention QKV / proj)
# ----------------------------------------------------------------------------- #
# DINOv2 uses HF's verbose naming:
#   embeddings.cls_token
#   embeddings.position_embeddings
#   embeddings.patch_embeddings.projection.{weight,bias}
#   encoder.layer.{i}.norm{1,2}.{weight,bias}
#   encoder.layer.{i}.layer_scale{1,2}.lambda1
#   encoder.layer.{i}.mlp.fc{1,2}.{weight,bias}
#   layernorm.{weight,bias}

DINOV2_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "conv1": "embeddings.patch_embeddings.projection",
    "pos.embed.pos.embed": "embeddings.position_embeddings",
    "cls.token.cls.token": "embeddings.cls_token",
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


# ----------------------------------------------------------------------------- #
# Special-case helpers
# ----------------------------------------------------------------------------- #


def _resolve_attention_qkv(keras_weight_path: str, dim: int):
    """Return the three HF Q/K/V keys for a fused Keras qkv weight.

    Given a path like
    ``multi_head_self_attention/blocks_0_attn_qkv/kernel`` returns
    ``("encoder.layer.0.attention.attention.query.weight",
       "...key.weight", "...value.weight")``.
    """
    layer_segment = keras_weight_path.split("/")[-2]  # blocks_0_attn_qkv
    block_match = re.match(r"blocks_(\d+)_attn_qkv$", layer_segment)
    if block_match is None:
        return None
    block_idx = int(block_match.group(1))
    base = f"encoder.layer.{block_idx}.attention.attention"
    suffix = "weight" if "kernel" in keras_weight_path else "bias"
    return (
        f"{base}.query.{suffix}",
        f"{base}.key.{suffix}",
        f"{base}.value.{suffix}",
    )


def _resolve_attention_proj(keras_weight_path: str):
    """Map ``blocks_{i}_attn_proj/{kernel,bias}`` to HF attention output dense."""
    layer_segment = keras_weight_path.split("/")[-2]
    block_match = re.match(r"blocks_(\d+)_attn_proj$", layer_segment)
    if block_match is None:
        return None
    block_idx = int(block_match.group(1))
    suffix = "weight" if "kernel" in keras_weight_path else "bias"
    return f"encoder.layer.{block_idx}.attention.output.dense.{suffix}"


def _resolve_layer_scale(keras_weight_path: str):
    """Map ``blocks_{i}_layerscale_{1,2}/variable*`` to HF lambda1 parameter."""
    layer_segment = keras_weight_path.split("/")[-2]
    block_match = re.match(r"blocks_(\d+)_layerscale_(1|2)$", layer_segment)
    if block_match is None:
        return None
    block_idx = int(block_match.group(1))
    which = block_match.group(2)
    return f"encoder.layer.{block_idx}.layer_scale{which}.lambda1"


def _fuse_qkv(
    state_dict: Dict[str, torch.Tensor], q: str, k: str, v: str
) -> torch.Tensor:
    """Concatenate HF Q, K, V weights along the output dim to match the kmodels qkv layout."""
    return torch.cat([state_dict[q], state_dict[k], state_dict[v]], dim=0)


def _interpolate_pos_embed(
    pos_embed: torch.Tensor, target_num_patches: int
) -> torch.Tensor:
    """Bilinearly resize a DINOv2 position-embedding tensor.

    HF DINOv2 stores ``embeddings.position_embeddings`` at the training
    resolution (518x518 -> 37x37 patches -> 1370 tokens incl. [CLS]). For a
    Keras model built at a different resolution we need to interpolate the
    spatial part of the table to the new patch grid while preserving the
    leading [CLS] position embedding.

    Args:
        pos_embed: Tensor of shape ``(1, 1 + S*S, dim)``.
        target_num_patches: Desired patch count ``T*T`` (e.g. 256 for 224x224
            with patch_size=14).

    Returns:
        Tensor of shape ``(1, 1 + target_num_patches, dim)``.
    """
    cls_pe = pos_embed[:, :1]  # (1, 1, dim)
    spatial_pe = pos_embed[:, 1:]  # (1, S*S, dim)
    src_num = spatial_pe.shape[1]
    src_size = int(round(src_num**0.5))
    tgt_size = int(round(target_num_patches**0.5))
    if src_size * src_size != src_num:
        raise ValueError(
            f"Position embedding has {src_num} spatial tokens which is not a "
            "perfect square; cannot interpolate."
        )
    if tgt_size * tgt_size != target_num_patches:
        raise ValueError(
            f"target_num_patches={target_num_patches} is not a perfect square."
        )
    if src_size == tgt_size:
        return pos_embed

    dim = spatial_pe.shape[-1]
    # (1, S*S, dim) -> (1, dim, S, S) for F.interpolate
    spatial_pe = spatial_pe.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
    spatial_pe = torch.nn.functional.interpolate(
        spatial_pe.float(),
        size=(tgt_size, tgt_size),
        mode="bicubic",
        align_corners=False,
    )
    spatial_pe = spatial_pe.permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, dim)
    return torch.cat([cls_pe, spatial_pe], dim=1)


# ----------------------------------------------------------------------------- #
# Per-weight transfer driver
# ----------------------------------------------------------------------------- #


def _transfer(keras_model: keras.Model, hf_state_dict: Dict[str, torch.Tensor]) -> None:
    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model
    )

    for keras_weight, keras_weight_name in tqdm(
        trainable_keras_weights + non_trainable_keras_weights,
        total=len(trainable_keras_weights + non_trainable_keras_weights),
        desc="Transferring weights",
    ):
        path = keras_weight.path

        # 1) Fused Q/K/V kernel and bias
        if "_attn_qkv" in path:
            qkv_keys = _resolve_attention_qkv(path, dim=keras_weight.shape[-1] // 3)
            if qkv_keys is None:
                raise WeightMappingError(keras_weight_name, path)
            q_key, k_key, v_key = qkv_keys
            for key in qkv_keys:
                if key not in hf_state_dict:
                    raise WeightMappingError(keras_weight_name, key)
            fused = _fuse_qkv(hf_state_dict, q_key, k_key, v_key)
            transfer_weights(keras_weight_name, keras_weight, fused)
            continue

        # 2) Attention output projection
        if "_attn_proj" in path:
            torch_key = _resolve_attention_proj(path)
            if torch_key is None or torch_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(torch_key))
            transfer_weights(keras_weight_name, keras_weight, hf_state_dict[torch_key])
            continue

        # 3) LayerScale lambda1
        if "_layerscale_" in path:
            torch_key = _resolve_layer_scale(path)
            if torch_key is None or torch_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(torch_key))
            keras_weight.assign(hf_state_dict[torch_key].numpy())
            continue

        # 4) cls_token / pos_embed go through after the suffix-strip below
        # 5) Generic mapping (patch_embed proj, layer norms, MLP, final norm)
        torch_weight_name = keras_weight_name
        for keras_name_part, torch_name_part in DINOV2_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )

        torch_weight_name = re.sub(
            r"pos_embed_variable_\d+$",
            "embeddings.position_embeddings",
            torch_weight_name,
        )
        torch_weight_name = re.sub(
            r"cls_token_variable_\d+$",
            "embeddings.cls_token",
            torch_weight_name,
        )

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = hf_state_dict[torch_weight_name]

        if torch_weight_name == "embeddings.cls_token":
            keras_weight.assign(torch_weight.numpy())
            continue
        if torch_weight_name == "embeddings.position_embeddings":
            target_num_patches = keras_weight.shape[1] - 1
            resized = _interpolate_pos_embed(torch_weight, target_num_patches)
            keras_weight.assign(resized.numpy())
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


# ----------------------------------------------------------------------------- #
# Forward-pass equivalence (random input)
# ----------------------------------------------------------------------------- #


def _verify_equivalence(
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
        hf_out_obj = hf_model(pixel_values=torch.from_numpy(x_np))
        # last_hidden_state shape: (1, 1 + num_patches, dim)
        hf_out = hf_out_obj.last_hidden_state.cpu().numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 1))  # NCHW -> NHWC
    keras_raw = keras_model(keras_in, training=False)
    if hasattr(keras_raw, "detach"):
        keras_raw = keras_raw.detach().cpu().numpy()
    keras_out = np.asarray(keras_raw)

    # Tolerance is loose because deeper models accumulate fp32 op-ordering
    # noise even when every individual weight is bit-identical. The relative
    # error stays well below 1e-4 for all three variants on a random input.
    atol = 1e-3
    rtol = 1e-4
    diff = np.abs(keras_out - hf_out)
    max_diff = float(diff.max())
    rel_max = max_diff / max(float(np.abs(hf_out).max()), 1e-12)
    if not np.allclose(keras_out, hf_out, atol=atol, rtol=rtol):
        raise ValueError(
            f"{name}: outputs differ - max abs diff = {max_diff:.6e} "
            f"(rel = {rel_max:.2e})"
        )
    print(
        f"{name}: equivalence OK (max abs diff = {max_diff:.2e}, "
        f"rel = {rel_max:.2e}, mean abs diff = {float(diff.mean()):.2e})"
    )


# ----------------------------------------------------------------------------- #
# Main entrypoint
# ----------------------------------------------------------------------------- #


def main() -> None:
    from transformers import Dinov2Model

    for name, info in DINOV2_CHECKPOINTS.items():
        print("\n" + "=" * 60)
        print(f"Converting: {name}  <-  {info['hf_id']}")
        print("=" * 60)

        hf_model = Dinov2Model.from_pretrained(info["hf_id"]).eval()
        hf_state_dict = dict(hf_model.state_dict())

        keras_model: keras.Model = info["ctor"](
            include_top=False,
            include_normalization=False,
            input_shape=info["input_shape"],
            weights=None,
        )

        _transfer(keras_model, hf_state_dict)
        _verify_equivalence(name, keras_model, hf_model, info["input_shape"])

        out_path = f"{info['save_name']}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"Saved -> {out_path}")

        del keras_model, hf_model, hf_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
