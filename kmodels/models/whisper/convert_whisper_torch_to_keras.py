import gc

import numpy as np
import torch
from keras import ops
from transformers import WhisperForConditionalGeneration

from kmodels.models.whisper.whisper_layers import WhisperAttention
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
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    transfer_nested_layer_weights,
    transfer_weights,
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
SLUG = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "large-v2": "largev2",
    "large-v3": "largev3",
    "large-v3-turbo": "largev3turbo",
}

DENSE_MAP = {"kernel": "weight"}
LN_MAP = {"gamma": "weight", "beta": "bias"}
EMBED_MAP = {"embeddings": "weight"}

SHARD_THRESHOLD_PARAMS = 500_000_000


def transfer_encoder(encoder, state, num_layers):
    for i in (1, 2):
        conv = encoder.get_layer(f"encoder_conv{i}")
        conv.kernel.assign(
            np.transpose(state[f"model.encoder.conv{i}.weight"], (2, 1, 0))
        )
        transfer_weights("bias", conv.bias, state[f"model.encoder.conv{i}.bias"])

    encoder.get_layer("encoder_embed_positions").pos_embed.assign(
        state["model.encoder.embed_positions.weight"]
    )

    attns = {
        layer.name_prefix: layer
        for layer in encoder.layers
        if isinstance(layer, WhisperAttention)
    }

    for i in range(num_layers):
        kp = f"encoder_layers_{i}"
        hp = f"model.encoder.layers.{i}"

        sa_kp = f"{kp}_self_attn"
        transfer_nested_layer_weights(
            attns[sa_kp],
            state,
            f"{hp}.self_attn",
            name_mapping={f"{sa_kp}_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_self_attn_layer_norm"),
            state,
            f"{hp}.self_attn_layer_norm",
            name_mapping=LN_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_fc1"),
            state,
            f"{hp}.fc1",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_fc2"),
            state,
            f"{hp}.fc2",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_final_layer_norm"),
            state,
            f"{hp}.final_layer_norm",
            name_mapping=LN_MAP,
        )

    transfer_nested_layer_weights(
        encoder.get_layer("encoder_layer_norm"),
        state,
        "model.encoder.layer_norm",
        name_mapping=LN_MAP,
    )


def transfer_decoder(decoder, state, num_layers):
    transfer_nested_layer_weights(
        decoder.get_layer("decoder_embed_tokens"),
        state,
        "model.decoder.embed_tokens",
        name_mapping=EMBED_MAP,
    )

    decoder.get_layer("decoder_embed_positions").pos_embed.assign(
        state["model.decoder.embed_positions.weight"]
    )

    attns = {
        layer.name_prefix: layer
        for layer in decoder.layers
        if isinstance(layer, WhisperAttention)
    }

    for i in range(num_layers):
        kp = f"decoder_layers_{i}"
        hp = f"model.decoder.layers.{i}"

        sa_kp = f"{kp}_self_attn"
        transfer_nested_layer_weights(
            attns[sa_kp],
            state,
            f"{hp}.self_attn",
            name_mapping={f"{sa_kp}_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_self_attn_layer_norm"),
            state,
            f"{hp}.self_attn_layer_norm",
            name_mapping=LN_MAP,
        )

        ca_kp = f"{kp}_encoder_attn"
        transfer_nested_layer_weights(
            attns[ca_kp],
            state,
            f"{hp}.encoder_attn",
            name_mapping={f"{ca_kp}_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_encoder_attn_layer_norm"),
            state,
            f"{hp}.encoder_attn_layer_norm",
            name_mapping=LN_MAP,
        )

        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_fc1"),
            state,
            f"{hp}.fc1",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_fc2"),
            state,
            f"{hp}.fc2",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_final_layer_norm"),
            state,
            f"{hp}.final_layer_norm",
            name_mapping=LN_MAP,
        )

    transfer_nested_layer_weights(
        decoder.get_layer("decoder_layer_norm"),
        state,
        "model.decoder.layer_norm",
        name_mapping=LN_MAP,
    )


for variant, hf_name in HF_CHECKPOINT.items():
    print(f"\n{'=' * 60}")
    print(f"Converting {hf_name}")
    print(f"{'=' * 60}")

    print(f"[1/4] Loading {hf_name}")
    torch_model = WhisperForConditionalGeneration.from_pretrained(hf_name).eval()
    state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
    cfg = torch_model.config

    print(f"[2/4] Building Keras {variant}")
    model = BUILDER[variant](weights=None)

    print("[3/4] Transferring weights")
    transfer_encoder(model.encoder, state, cfg.encoder_layers)
    transfer_decoder(model.decoder, state, cfg.decoder_layers)

    print("[4/4] Verifying parity with HF")
    np.random.seed(0)
    test_mel = np.random.randn(1, cfg.num_mel_bins, 3000).astype(np.float32)
    test_ids = np.array(
        [[cfg.decoder_start_token_id, cfg.decoder_start_token_id + 1]],
        dtype=np.int32,
    )
    keras_logits = ops.convert_to_numpy(
        model({"input_features": test_mel, "decoder_input_ids": test_ids})["logits"]
    )
    with torch.no_grad():
        hf_logits = (
            torch_model(
                input_features=torch.from_numpy(test_mel),
                decoder_input_ids=torch.from_numpy(test_ids),
            )
            .logits.detach()
            .cpu()
            .numpy()
        )
    diff = float(np.max(np.abs(keras_logits - hf_logits)))
    print(f"  max abs logit diff: {diff:.6e}")
    assert diff < 1e-3, f"{variant}: logit diff too high: {diff:.6e}"

    base = f"whisper{SLUG[variant]}_openai"
    if model.count_params() > SHARD_THRESHOLD_PARAMS:
        out_path = f"{base}.weights.json"
        model.save_weights(out_path, max_shard_size=1.5)
    else:
        out_path = f"{base}.weights.h5"
        model.save_weights(out_path)
    print(f"Saved -> {out_path}")

    del torch_model, model, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
