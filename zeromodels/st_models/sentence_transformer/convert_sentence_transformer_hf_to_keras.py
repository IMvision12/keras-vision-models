"""Transfer upstream Sentence Transformer weights into Keras models."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights
from zeromodels.models.bert.convert_bert_hf_to_keras import transfer_bert_weights
from zeromodels.models.mpnet.convert_mpnet_hf_to_keras import (
    normalize_hf_key,
    transfer_mpnet_weights,
)
from zeromodels.models.roberta.convert_roberta_hf_to_keras import (
    transfer_roberta_weights,
)
from zeromodels.models.xlm_roberta.convert_xlm_roberta_hf_to_keras import (
    transfer_xlm_roberta_weights,
)


def strip_st_prefix(key: str) -> str:
    """Normalize ST / HF key prefixes down to bare MPNet (or Dense) names.

    Handles common layouts:

    * ``0.auto_model.*`` — SentenceTransformer module stack
    * ``auto_model.*``
    * ``mpnet.*`` — HF ``MPNetModel`` wrapper
    """
    for prefix in (
        "0.auto_model.",
        "auto_model.",
        "0.transformer.",
    ):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return normalize_hf_key(key)


def _find_dense_tensors(
    hf_state_dict: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Locate an ST ``Dense`` module's ``linear.weight`` / ``linear.bias``."""
    kernel = bias = None
    for k, v in hf_state_dict.items():
        if "auto_model" in k or ".layer." in k or k.startswith("embeddings."):
            continue
        if k.endswith("linear.weight"):
            kernel = np.asarray(v)
        elif k.endswith("linear.bias"):
            bias = np.asarray(v)
    return kernel, bias


def transfer_sentence_transformer_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    """Copy backbone (+ optional Dense projection) weights into ``keras_model``.

    Pooling and L2-normalize have no parameters. Unused HF pooler / MLM head
    tensors are skipped via the MPNet optional-weight rules.
    """
    hf = {strip_st_prefix(k): v for k, v in hf_state_dict.items()}
    kind = keras_model.backbone_type
    if kind == "mpnet":
        transfer_mpnet_weights(keras_model, hf)
    elif kind == "bert":
        transfer_bert_weights(keras_model, hf)
    elif kind == "roberta":
        transfer_roberta_weights(keras_model, hf)
    elif kind == "xlm-roberta":
        transfer_xlm_roberta_weights(keras_model, hf)
    elif kind in ("distilbert", "albert"):
        _transfer_custom_backbone(keras_model, hf)
    else:
        raise ValueError(f"Unsupported sentence embedding backbone: {kind}")

    needs_proj = any(w.path.startswith("projection/") for w in keras_model.weights)
    if not needs_proj:
        return

    dense_state = getattr(keras_model, "_st_dense_state", hf_state_dict)
    proj_kernel, proj_bias = _find_dense_tensors(dense_state)
    for weight in keras_model.weights:
        path = weight.path
        if path == "projection/kernel":
            if proj_kernel is None:
                raise WeightMappingError(path, "*.linear.weight")
            transfer_weights(path, weight, proj_kernel)
        elif path == "projection/bias":
            if proj_bias is None:
                raise WeightMappingError(path, "*.linear.bias")
            transfer_weights(path, weight, proj_bias)


def _custom_hf_name(path, backbone_type):
    """Map the two ST-only backbone families to their HF tensor names."""
    suffix = path.rsplit("/", 1)[-1]
    suffix = {
        "kernel": "weight",
        "embeddings": "weight",
        "gamma": "weight",
        "beta": "bias",
    }.get(suffix, suffix)
    if backbone_type == "distilbert":
        fixed = {
            "distil_word_embeddings": "embeddings.word_embeddings",
            "distil_position_embeddings": "embeddings.position_embeddings",
            "distil_embeddings_norm": "embeddings.LayerNorm",
        }
        if "/" not in path:
            return None
        outer, inner = path.split("/", 1)
        if outer in fixed:
            return f"{fixed[outer]}.{suffix}"
        if not outer.startswith("distil_layer_"):
            return None
        index = int(outer.removeprefix("distil_layer_"))
        part = inner.split("/", 1)[0]
        part = {
            "q_lin": "attention.q_lin",
            "k_lin": "attention.k_lin",
            "v_lin": "attention.v_lin",
            "out_lin": "attention.out_lin",
            "ffn_lin1": "ffn.lin1",
            "ffn_lin2": "ffn.lin2",
        }.get(part, part)
        return f"transformer.layer.{index}.{part}.{suffix}"

    fixed = {
        "albert_word_embeddings": "embeddings.word_embeddings",
        "albert_position_embeddings": "embeddings.position_embeddings",
        "albert_token_type_embeddings": "embeddings.token_type_embeddings",
        "albert_embeddings_norm": "embeddings.LayerNorm",
        "albert_embedding_projection": "encoder.embedding_hidden_mapping_in",
    }
    if "/" not in path:
        return None
    outer, inner = path.split("/", 1)
    if outer in fixed:
        return f"{fixed[outer]}.{suffix}"
    if outer != "albert_shared_block":
        return None
    part = inner.split("/", 1)[0]
    part = {
        "query": "attention.query",
        "key": "attention.key",
        "value": "attention.value",
        "attention_dense": "attention.dense",
        "attention_layer_norm": "attention.LayerNorm",
        "full_layer_norm": "full_layer_layer_norm",
    }.get(part, part)
    return f"encoder.albert_layer_groups.0.albert_layers.0.{part}.{suffix}"


def _transfer_custom_backbone(keras_model, hf_state_dict):
    kind = keras_model.backbone_type
    prefix = "distilbert." if kind == "distilbert" else "albert."
    names = {k.removeprefix(prefix): k for k in hf_state_dict}
    for weight in keras_model.weights:
        hf_name = _custom_hf_name(weight.path, kind)
        if hf_name is None:
            continue
        if hf_name not in names:
            raise WeightMappingError(weight.path, hf_name)
        source = hf_state_dict[names[hf_name]]
        if weight.path == "albert_embedding_projection/kernel":
            # The generic transfer helper treats any path containing
            # "embedding" as an Embedding table, but this is a Dense kernel.
            weight.assign(np.asarray(source).T)
        else:
            transfer_weights(weight.path, weight, source)


def load_dense_state_dict(hf_id, rel_path, *, token=None, revision=None):
    """Read the trained Dense module from its declared ST subdirectory."""
    if not rel_path or rel_path.startswith("/") or ".." in rel_path.split("/"):
        raise ValueError(
            f"Unsafe Sentence Transformers Dense module path: {rel_path!r}"
        )
    try:
        path = hf_hub_download(
            hf_id, f"{rel_path}/model.safetensors", token=token, revision=revision
        )
    except EntryNotFoundError:
        path = hf_hub_download(
            hf_id, f"{rel_path}/pytorch_model.bin", token=token, revision=revision
        )
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        return {key: value.detach().cpu().numpy() for key, value in state.items()}
    with safe_open(path, framework="np") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def pooling_mode_from_config(pooling_cfg: dict) -> str:
    if pooling_cfg.get("pooling_mode_cls_token"):
        return "cls"
    if pooling_cfg.get("pooling_mode_max_tokens"):
        return "max"
    return "mean"


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import AutoModel, AutoTokenizer

    from zeromodels.conversion.zm_config import write_zm_config
    from zeromodels.st_models import (
        SentenceTransformerModel,
        SentenceTransformerTokenizer,
    )

    HF_TOKEN = os.environ.get("HF_TOKEN")
    VARIANTS = {
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    }

    texts = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The weather is lovely today.",
    ]

    for variant, hf_id in VARIANTS.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        keras_model = SentenceTransformerModel.from_hf(hf_id)
        tok = SentenceTransformerTokenizer.from_hf(
            hf_id, max_seq_len=keras_model.max_seq_len
        )
        k_inputs = tok(texts)
        k_emb = keras.ops.convert_to_numpy(
            keras_model(k_inputs, training=False)["embeddings"]
        )

        # Reference: transformers backbone + mean pool + L2 (matches ST default).
        hf_tok = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN)
        hf_enc = AutoModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        enc = hf_tok(
            texts,
            padding=True,
            truncation=True,
            max_length=keras_model.max_seq_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = hf_enc(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            pooled = summed / counts
            ref = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()

        diff = float(np.abs(k_emb - ref).max())
        cos = float(np.sum(k_emb * ref, axis=-1).mean())
        print(f"  max abs diff vs mean-pool+L2 reference: {diff:.3e}")
        print(f"  mean cosine(keras, ref): {cos:.6f}")
        if diff > 1e-3:
            raise ValueError(f"{variant}: parity failed (diff={diff})")

        dest = variant
        os.makedirs(dest, exist_ok=True)
        out_path = os.path.join(dest, "model.weights.h5")
        keras_model.save_weights(out_path)
        write_zm_config(
            dest,
            SentenceTransformerModel,
            variant,
            keras_model.config,
            weights_filename="model.weights.h5",
        )
        print(f"  Saved -> {dest}/ (model.weights.h5 + zm_config.json)")

        del keras_model, hf_enc
        keras.backend.clear_session()
        gc.collect()
