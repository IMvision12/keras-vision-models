import gc
import re
from typing import Dict, List, Tuple, Type

import keras
import numpy as np
import torch
from tqdm import tqdm
from transformers import Dinov2Model

from kmodels.models import dino_v2
from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
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

DINOV2_WEIGHTS_CONFIG: List[Tuple[Type[keras.Model], str, int, str]] = [
    (dino_v2.DinoV2Small14, "facebook/dinov2-small", 224, "dinov2_vits14"),
    (dino_v2.DinoV2Base14, "facebook/dinov2-base", 224, "dinov2_vitb14"),
    (dino_v2.DinoV2Large14, "facebook/dinov2-large", 224, "dinov2_vitl14"),
]


def _resolve_attention_qkv(keras_weight_path: str):
    """Return the three HF Q/K/V keys for a fused Keras qkv weight."""
    layer_segment = keras_weight_path.split("/")[-2]
    m = re.match(r"blocks_(\d+)_attn_qkv$", layer_segment)
    if m is None:
        return None
    idx = int(m.group(1))
    base = f"encoder.layer.{idx}.attention.attention"
    suffix = "weight" if "kernel" in keras_weight_path else "bias"
    return (f"{base}.query.{suffix}", f"{base}.key.{suffix}", f"{base}.value.{suffix}")


def _resolve_attention_proj(keras_weight_path: str):
    """Map ``blocks_{i}_attn_proj`` to HF attention output dense."""
    layer_segment = keras_weight_path.split("/")[-2]
    m = re.match(r"blocks_(\d+)_attn_proj$", layer_segment)
    if m is None:
        return None
    idx = int(m.group(1))
    suffix = "weight" if "kernel" in keras_weight_path else "bias"
    return f"encoder.layer.{idx}.attention.output.dense.{suffix}"


def _resolve_layer_scale(keras_weight_path: str):
    """Map ``blocks_{i}_layerscale_{1,2}/variable*`` to HF lambda1."""
    layer_segment = keras_weight_path.split("/")[-2]
    m = re.match(r"blocks_(\d+)_layerscale_(1|2)$", layer_segment)
    if m is None:
        return None
    idx = int(m.group(1))
    which = m.group(2)
    return f"encoder.layer.{idx}.layer_scale{which}.lambda1"


def _fuse_qkv(
    state_dict: Dict[str, torch.Tensor], q: str, k: str, v: str
) -> torch.Tensor:
    """Concatenate HF Q, K, V weights along output dim to match fused qkv."""
    return torch.cat([state_dict[q], state_dict[k], state_dict[v]], dim=0)


def _interpolate_pos_embed(
    pos_embed: torch.Tensor, target_num_patches: int
) -> torch.Tensor:
    """Bilinearly resize a DINOv2 position-embedding tensor.

    HF DINOv2 stores position embeddings at the training resolution
    (518x518 -> 37x37 patches -> 1370 tokens incl. [CLS]). For a Keras
    model at a different resolution we interpolate the spatial part.
    """
    cls_pe = pos_embed[:, :1]
    spatial_pe = pos_embed[:, 1:]
    src_num = spatial_pe.shape[1]
    src_size = int(round(src_num**0.5))
    tgt_size = int(round(target_num_patches**0.5))
    if src_size == tgt_size:
        return pos_embed

    dim = spatial_pe.shape[-1]
    spatial_pe = spatial_pe.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
    spatial_pe = torch.nn.functional.interpolate(
        spatial_pe.float(),
        size=(tgt_size, tgt_size),
        mode="bicubic",
        align_corners=False,
    )
    spatial_pe = spatial_pe.permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, dim)
    return torch.cat([cls_pe, spatial_pe], dim=1)


for keras_model_cls, hf_id, resolution, save_name in DINOV2_WEIGHTS_CONFIG:
    input_shape = [resolution, resolution, 3]

    print(f"\n{'=' * 60}")
    print(f"Converting: {save_name}  <-  {hf_id}")
    print(f"{'=' * 60}")

    hf_model = Dinov2Model.from_pretrained(hf_id).eval()
    hf_state_dict = dict(hf_model.state_dict())

    keras_model: keras.Model = keras_model_cls(
        include_top=False,
        include_normalization=False,
        input_shape=input_shape,
        weights=None,
    )

    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model
    )

    for keras_weight, keras_weight_name in tqdm(
        trainable_keras_weights + non_trainable_keras_weights,
        total=len(trainable_keras_weights + non_trainable_keras_weights),
        desc="Transferring weights",
    ):
        path = keras_weight.path

        if "_attn_qkv" in path:
            qkv_keys = _resolve_attention_qkv(path)
            if qkv_keys is None:
                raise WeightMappingError(keras_weight_name, path)
            for key in qkv_keys:
                if key not in hf_state_dict:
                    raise WeightMappingError(keras_weight_name, key)
            fused = _fuse_qkv(hf_state_dict, *qkv_keys)
            transfer_weights(keras_weight_name, keras_weight, fused)
            continue

        if "_attn_proj" in path:
            hf_key = _resolve_attention_proj(path)
            if hf_key is None or hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(hf_key))
            transfer_weights(keras_weight_name, keras_weight, hf_state_dict[hf_key])
            continue

        if "_layerscale_" in path:
            hf_key = _resolve_layer_scale(path)
            if hf_key is None or hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(hf_key))
            keras_weight.assign(hf_state_dict[hf_key].numpy())
            continue

        torch_weight_name = keras_weight_name
        for old, new in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(old, new)

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
    assert diff < 1e-3, f"{save_name}: max diff {diff:.2e}"
    print(f"  Verification OK (max diff = {diff:.2e})")

    model_filename = f"{save_name}.weights.h5"
    keras_model.save_weights(model_filename)
    print(f"  Saved -> {model_filename}")

    del keras_model, hf_model, hf_state_dict
    del trainable_keras_weights, non_trainable_keras_weights
    keras.backend.clear_session()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
