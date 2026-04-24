"""Port HF Whisper PyTorch weights into the kmodels Keras Whisper.

Usage (from repo root):

    python -m kmodels.models.whisper.convert_whisper_torch_to_keras \\
        --variant tiny --out whisper_tiny_openai.weights.h5

Produces a single ``.weights.h5`` file that can be loaded via the saved
encoder/decoder split: the encoder weights are named ``encoder_*`` and
decoder weights ``decoder_*``, so one file can hydrate both sub-models.
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from kmodels.models.whisper.whisper_model import (
    WhisperBase,
    WhisperLarge,
    WhisperLargeV2,
    WhisperLargeV3,
    WhisperLargeV3Turbo,
    WhisperMedium,
    WhisperSmall,
    WhisperTiny,
)

HF_CHECKPOINT = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}
BUILDER = {
    "tiny": WhisperTiny,
    "base": WhisperBase,
    "small": WhisperSmall,
    "medium": WhisperMedium,
    "large": WhisperLarge,
    "large-v2": WhisperLargeV2,
    "large-v3": WhisperLargeV3,
    "large-v3-turbo": WhisperLargeV3Turbo,
}


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _layer_weight_map(
    torch_state: Dict[str, torch.Tensor],
    num_encoder_layers: int,
    num_decoder_layers: int,
) -> Dict[str, np.ndarray]:
    """Build {keras_layer_name: {variable_name: ndarray}} mapping."""
    kmap: Dict[str, Dict[str, np.ndarray]] = {}

    def _put(layer_name: str, var_name: str, arr: np.ndarray):
        kmap.setdefault(layer_name, {})[var_name] = arr

    # ---- Encoder conv stem ----
    # PyTorch Conv1d weight shape: (out, in, k). Keras Conv1D kernel: (k, in, out).
    for i in (1, 2):
        w = _torch_to_numpy(torch_state[f"model.encoder.conv{i}.weight"])
        b = _torch_to_numpy(torch_state[f"model.encoder.conv{i}.bias"])
        _put(f"encoder_conv{i}", "kernel", np.transpose(w, (2, 1, 0)))
        _put(f"encoder_conv{i}", "bias", b)

    # ---- Encoder sinusoidal position embedding (non-trainable, but we still set it) ----
    _put(
        "encoder_embed_positions",
        "weight",
        _torch_to_numpy(torch_state["model.encoder.embed_positions.weight"]),
    )

    # ---- Encoder blocks ----
    for i in range(num_encoder_layers):
        base = f"model.encoder.layers.{i}"
        kprefix = f"encoder_layers_{i}"

        # self-attn (k has no bias)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            w = _torch_to_numpy(torch_state[f"{base}.self_attn.{proj}.weight"])
            _put(f"{kprefix}_self_attn_{proj}", "kernel", np.transpose(w, (1, 0)))
            bias_key = f"{base}.self_attn.{proj}.bias"
            if bias_key in torch_state:
                _put(
                    f"{kprefix}_self_attn_{proj}",
                    "bias",
                    _torch_to_numpy(torch_state[bias_key]),
                )

        # self-attn LN
        _put(
            f"{kprefix}_self_attn_layer_norm",
            "gamma",
            _torch_to_numpy(torch_state[f"{base}.self_attn_layer_norm.weight"]),
        )
        _put(
            f"{kprefix}_self_attn_layer_norm",
            "beta",
            _torch_to_numpy(torch_state[f"{base}.self_attn_layer_norm.bias"]),
        )

        # FFN
        for fc in ("fc1", "fc2"):
            w = _torch_to_numpy(torch_state[f"{base}.{fc}.weight"])
            b = _torch_to_numpy(torch_state[f"{base}.{fc}.bias"])
            _put(f"{kprefix}_{fc}", "kernel", np.transpose(w, (1, 0)))
            _put(f"{kprefix}_{fc}", "bias", b)

        # final LN
        _put(
            f"{kprefix}_final_layer_norm",
            "gamma",
            _torch_to_numpy(torch_state[f"{base}.final_layer_norm.weight"]),
        )
        _put(
            f"{kprefix}_final_layer_norm",
            "beta",
            _torch_to_numpy(torch_state[f"{base}.final_layer_norm.bias"]),
        )

    # ---- Encoder final LN ----
    _put(
        "encoder_layer_norm",
        "gamma",
        _torch_to_numpy(torch_state["model.encoder.layer_norm.weight"]),
    )
    _put(
        "encoder_layer_norm",
        "beta",
        _torch_to_numpy(torch_state["model.encoder.layer_norm.bias"]),
    )

    # ---- Decoder token + pos embeddings ----
    _put(
        "decoder_embed_tokens",
        "embeddings",
        _torch_to_numpy(torch_state["model.decoder.embed_tokens.weight"]),
    )
    _put(
        "decoder_embed_positions",
        "weight",
        _torch_to_numpy(torch_state["model.decoder.embed_positions.weight"]),
    )

    # ---- Decoder blocks ----
    for i in range(num_decoder_layers):
        base = f"model.decoder.layers.{i}"
        kprefix = f"decoder_layers_{i}"

        # self-attn
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            w = _torch_to_numpy(torch_state[f"{base}.self_attn.{proj}.weight"])
            _put(f"{kprefix}_self_attn_{proj}", "kernel", np.transpose(w, (1, 0)))
            bias_key = f"{base}.self_attn.{proj}.bias"
            if bias_key in torch_state:
                _put(
                    f"{kprefix}_self_attn_{proj}",
                    "bias",
                    _torch_to_numpy(torch_state[bias_key]),
                )

        # self-attn LN
        _put(
            f"{kprefix}_self_attn_layer_norm",
            "gamma",
            _torch_to_numpy(torch_state[f"{base}.self_attn_layer_norm.weight"]),
        )
        _put(
            f"{kprefix}_self_attn_layer_norm",
            "beta",
            _torch_to_numpy(torch_state[f"{base}.self_attn_layer_norm.bias"]),
        )

        # cross-attn (encoder_attn in HF)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            w = _torch_to_numpy(torch_state[f"{base}.encoder_attn.{proj}.weight"])
            _put(f"{kprefix}_encoder_attn_{proj}", "kernel", np.transpose(w, (1, 0)))
            bias_key = f"{base}.encoder_attn.{proj}.bias"
            if bias_key in torch_state:
                _put(
                    f"{kprefix}_encoder_attn_{proj}",
                    "bias",
                    _torch_to_numpy(torch_state[bias_key]),
                )

        # cross-attn LN
        _put(
            f"{kprefix}_encoder_attn_layer_norm",
            "gamma",
            _torch_to_numpy(torch_state[f"{base}.encoder_attn_layer_norm.weight"]),
        )
        _put(
            f"{kprefix}_encoder_attn_layer_norm",
            "beta",
            _torch_to_numpy(torch_state[f"{base}.encoder_attn_layer_norm.bias"]),
        )

        # FFN
        for fc in ("fc1", "fc2"):
            w = _torch_to_numpy(torch_state[f"{base}.{fc}.weight"])
            b = _torch_to_numpy(torch_state[f"{base}.{fc}.bias"])
            _put(f"{kprefix}_{fc}", "kernel", np.transpose(w, (1, 0)))
            _put(f"{kprefix}_{fc}", "bias", b)

        # final LN
        _put(
            f"{kprefix}_final_layer_norm",
            "gamma",
            _torch_to_numpy(torch_state[f"{base}.final_layer_norm.weight"]),
        )
        _put(
            f"{kprefix}_final_layer_norm",
            "beta",
            _torch_to_numpy(torch_state[f"{base}.final_layer_norm.bias"]),
        )

    # ---- Decoder final LN ----
    _put(
        "decoder_layer_norm",
        "gamma",
        _torch_to_numpy(torch_state["model.decoder.layer_norm.weight"]),
    )
    _put(
        "decoder_layer_norm",
        "beta",
        _torch_to_numpy(torch_state["model.decoder.layer_norm.bias"]),
    )

    return kmap


def _iter_all_layers(model):
    """Yield every layer including those nested inside custom Layer subclasses."""
    import keras

    seen = set()

    def _walk(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        yield layer
        for attr in vars(layer).values():
            if isinstance(attr, keras.layers.Layer):
                yield from _walk(attr)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, keras.layers.Layer):
                        yield from _walk(item)

    for layer in model.layers:
        yield from _walk(layer)


def _assign_to_model(model, kmap: Dict[str, Dict[str, np.ndarray]]) -> Tuple[int, int]:
    assigned, missing = 0, 0
    for layer in _iter_all_layers(model):
        if layer.name not in kmap:
            continue
        layer_weights = kmap[layer.name]
        new_values = []
        for w in layer.weights:
            short = w.path.rsplit("/", 1)[-1].split(":")[0]
            if short in layer_weights:
                arr = layer_weights[short]
                if tuple(arr.shape) != tuple(w.shape):
                    raise ValueError(
                        f"Shape mismatch for {layer.name}/{short}: "
                        f"got {arr.shape}, expected {tuple(w.shape)}"
                    )
                new_values.append(arr)
                assigned += 1
            else:
                new_values.append(w.numpy())
                missing += 1
        if new_values:
            layer.set_weights(new_values)
    return assigned, missing


_SLUG = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "large-v2": "large_v2",
    "large-v3": "large_v3",
    "large-v3-turbo": "large_v3_turbo",
}


def convert(variant: str, out_dir: str = "."):
    if variant not in HF_CHECKPOINT:
        raise ValueError(
            f"Unknown variant {variant}. Choose from {list(HF_CHECKPOINT)}."
        )

    print(f"[1/4] Loading HF checkpoint: {HF_CHECKPOINT[variant]}")
    torch_model = WhisperForConditionalGeneration.from_pretrained(
        HF_CHECKPOINT[variant]
    ).eval()
    state = torch_model.state_dict()
    cfg = torch_model.config

    print(f"[2/4] Building Keras {variant} models")
    built = BUILDER[variant]()
    encoder, decoder = built["encoder"], built["decoder"]

    print("[3/4] Mapping weights")
    kmap = _layer_weight_map(state, cfg.encoder_layers, cfg.decoder_layers)
    enc_a, enc_m = _assign_to_model(encoder, kmap)
    dec_a, dec_m = _assign_to_model(decoder, kmap)
    print(f"  encoder assigned={enc_a} missing={enc_m}")
    print(f"  decoder assigned={dec_a} missing={dec_m}")

    slug = _SLUG[variant]
    enc_path = os.path.join(out_dir, f"{slug}_encoder.weights.h5")
    dec_path = os.path.join(out_dir, f"{slug}_decoder.weights.h5")
    print(f"[4/4] Saving split weights under {out_dir}/")
    encoder.save_weights(enc_path)
    decoder.save_weights(dec_path)
    print(f"  wrote {enc_path}")
    print(f"  wrote {dec_path}")
    return encoder, decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="tiny", choices=list(HF_CHECKPOINT))
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to drop the {slug}_encoder/{slug}_decoder weights files.",
    )
    args = parser.parse_args()
    convert(args.variant, args.out_dir)
