import keras
import numpy as np
from keras import layers, ops

from kmodels.weight_utils import download_file

from .config import (
    WHISPER_BEGIN_SUPPRESS_TOKENS,
    WHISPER_MODEL_CONFIG,
    WHISPER_SUPPRESS_TOKENS,
    WHISPER_WEIGHTS_CONFIG,
)
from .whisper_layers import (
    LearnedPositionEmbedding,
    SinusoidalPositionEmbedding,
    WhisperAttention,
)


def _gelu(x):
    return keras.activations.gelu(x, approximate=False)


def encoder_block(x, d_model, num_heads, ffn_dim, layer_idx):
    prefix = f"encoder_layers_{layer_idx}"

    ln_1 = layers.LayerNormalization(
        epsilon=1e-5, name=f"{prefix}_self_attn_layer_norm"
    )(x)
    attn_out = WhisperAttention(
        proj_dim=d_model,
        num_heads=num_heads,
        name_prefix=f"{prefix}_self_attn",
    )(ln_1)
    x = layers.Add()([x, attn_out])

    ln_2 = layers.LayerNormalization(epsilon=1e-5, name=f"{prefix}_final_layer_norm")(x)
    h = layers.Dense(ffn_dim, name=f"{prefix}_fc1")(ln_2)
    h = layers.Lambda(_gelu, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(d_model, name=f"{prefix}_fc2")(h)
    x = layers.Add()([x, h])
    return x


def decoder_block(
    x, encoder_hidden_states, causal_mask, d_model, num_heads, ffn_dim, layer_idx
):
    prefix = f"decoder_layers_{layer_idx}"

    ln_1 = layers.LayerNormalization(
        epsilon=1e-5, name=f"{prefix}_self_attn_layer_norm"
    )(x)
    self_attn_out = WhisperAttention(
        proj_dim=d_model,
        num_heads=num_heads,
        name_prefix=f"{prefix}_self_attn",
    )(ln_1, attention_mask=causal_mask)
    x = layers.Add()([x, self_attn_out])

    ln_2 = layers.LayerNormalization(
        epsilon=1e-5, name=f"{prefix}_encoder_attn_layer_norm"
    )(x)
    cross_attn_out = WhisperAttention(
        proj_dim=d_model,
        num_heads=num_heads,
        name_prefix=f"{prefix}_encoder_attn",
    )(ln_2, key_value_states=encoder_hidden_states)
    x = layers.Add()([x, cross_attn_out])

    ln_3 = layers.LayerNormalization(epsilon=1e-5, name=f"{prefix}_final_layer_norm")(x)
    h = layers.Dense(ffn_dim, name=f"{prefix}_fc1")(ln_3)
    h = layers.Lambda(_gelu, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(d_model, name=f"{prefix}_fc2")(h)
    x = layers.Add()([x, h])
    return x


def build_encoder(cfg, name="encoder"):
    d_model = cfg["d_model"]
    num_mel_bins = cfg["num_mel_bins"]
    max_src = cfg["max_source_positions"]

    # Input: (batch, n_mels, time) matching HF's (B, 80, 3000) convention.
    mel = layers.Input(shape=(num_mel_bins, None), name="input_features")
    # Move to channels-last for conv1d: (B, time, n_mels)
    x = layers.Permute((2, 1), name="encoder_permute_in")(mel)

    x = layers.ZeroPadding1D(padding=1, name="encoder_conv1_pad")(x)
    x = layers.Conv1D(
        filters=d_model,
        kernel_size=3,
        strides=1,
        padding="valid",
        name="encoder_conv1",
    )(x)
    x = layers.Lambda(_gelu, name="encoder_conv1_act")(x)
    x = layers.ZeroPadding1D(padding=1, name="encoder_conv2_pad")(x)
    x = layers.Conv1D(
        filters=d_model,
        kernel_size=3,
        strides=2,
        padding="valid",
        name="encoder_conv2",
    )(x)
    x = layers.Lambda(_gelu, name="encoder_conv2_act")(x)

    x = SinusoidalPositionEmbedding(
        max_source_positions=max_src,
        d_model=d_model,
        name="encoder_embed_positions",
    )(x)

    for i in range(cfg["encoder_layers"]):
        x = encoder_block(
            x,
            d_model=d_model,
            num_heads=cfg["encoder_attention_heads"],
            ffn_dim=cfg["encoder_ffn_dim"],
            layer_idx=i,
        )

    x = layers.LayerNormalization(epsilon=1e-5, name="encoder_layer_norm")(x)
    return keras.Model(inputs=mel, outputs=x, name=name)


def _make_causal_mask_from_ids(decoder_input_ids):
    seq_len = ops.shape(decoder_input_ids)[1]
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    mask = ops.cast(j > i, "float32") * -1e9
    return mask[None, None, :, :]


def build_decoder(cfg, name="decoder"):
    d_model = cfg["d_model"]
    max_tgt = cfg["max_target_positions"]
    vocab_size = cfg["vocab_size"]

    decoder_input_ids = layers.Input(
        shape=(None,), dtype="int32", name="decoder_input_ids"
    )
    encoder_hidden_states = layers.Input(
        shape=(None, d_model), name="encoder_hidden_states"
    )

    tok_embed = layers.Embedding(
        input_dim=vocab_size, output_dim=d_model, name="decoder_embed_tokens"
    )
    x = tok_embed(decoder_input_ids)
    x = LearnedPositionEmbedding(
        max_target_positions=max_tgt,
        d_model=d_model,
        name="decoder_embed_positions",
    )(x)

    causal_mask = layers.Lambda(
        _make_causal_mask_from_ids,
        name="decoder_causal_mask",
        output_shape=lambda s: (1, 1, s[1], s[1]),
    )(decoder_input_ids)

    for i in range(cfg["decoder_layers"]):
        x = decoder_block(
            x,
            encoder_hidden_states=encoder_hidden_states,
            causal_mask=causal_mask,
            d_model=d_model,
            num_heads=cfg["decoder_attention_heads"],
            ffn_dim=cfg["decoder_ffn_dim"],
            layer_idx=i,
        )

    x = layers.LayerNormalization(epsilon=1e-5, name="decoder_layer_norm")(x)

    # Tied embedding: logits = x @ embed_tokens.T
    embed_weight = tok_embed.embeddings
    logits = layers.Lambda(
        lambda t, w=embed_weight: ops.matmul(t, ops.transpose(w, (1, 0))),
        name="lm_head",
    )(x)

    return keras.Model(
        inputs={
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        },
        outputs=logits,
        name=name,
    )


def _load_pretrained(encoder, decoder, variant_key: str, weights: str):
    """Download + load the ``{variant_key}`` weights keyed by ``weights``."""
    if variant_key not in WHISPER_WEIGHTS_CONFIG:
        raise ValueError(f"No weights config for {variant_key}")
    variants = WHISPER_WEIGHTS_CONFIG[variant_key]
    if weights not in variants:
        raise ValueError(
            f"Unknown weights preset {weights!r} for {variant_key}. "
            f"Available: {list(variants)}"
        )
    urls = variants[weights]
    enc_path = download_file(urls["encoder_url"])
    dec_path = download_file(urls["decoder_url"])
    encoder.load_weights(enc_path)
    decoder.load_weights(dec_path)


def _build_variant(variant_key: str, slug: str, weights=None):
    cfg = WHISPER_MODEL_CONFIG[variant_key]
    encoder = build_encoder(cfg, name=f"whisper_{slug}_encoder")
    decoder = build_decoder(cfg, name=f"whisper_{slug}_decoder")
    if weights is not None:
        _load_pretrained(encoder, decoder, variant_key, weights)
    return {"encoder": encoder, "decoder": decoder, "config": cfg}


def WhisperTiny(weights=None):
    return _build_variant("WhisperTiny", "tiny", weights)


def WhisperBase(weights=None):
    return _build_variant("WhisperBase", "base", weights)


def WhisperSmall(weights=None):
    return _build_variant("WhisperSmall", "small", weights)


def WhisperMedium(weights=None):
    return _build_variant("WhisperMedium", "medium", weights)


def WhisperLarge(weights=None):
    return _build_variant("WhisperLarge", "large", weights)


def WhisperLargeV2(weights=None):
    return _build_variant("WhisperLargeV2", "large_v2", weights)


def WhisperLargeV3(weights=None):
    return _build_variant("WhisperLargeV3", "large_v3", weights)


def WhisperLargeV3Turbo(weights=None):
    return _build_variant("WhisperLargeV3Turbo", "large_v3_turbo", weights)


def whisper_generate(
    encoder_model,
    decoder_model,
    input_features,
    forced_decoder_ids: list = None,
    decoder_start_token_id: int = 50258,
    eos_token_id: int = 50257,
    max_new_tokens: int = 100,
    suppress_tokens: list = None,
    begin_suppress_tokens: list = None,
):
    """Greedy decoding matching HF's Whisper generate().

    Mirrors the key logit processors used by HF:
      * ``forced_decoder_ids``: at generation step ``k``, force the output to
        the given id. Typically ``[(1, lang_id), (2, task_id), (3, 50363)]``
        for English no-timestamps transcription.
      * ``suppress_tokens``: permanently forbid this set of token ids.
      * ``begin_suppress_tokens``: suppress these only at the very first
        generated step (e.g. blank/silent tokens).
    """
    enc_out = encoder_model(input_features)
    enc_np = (
        ops.convert_to_numpy(enc_out)
        if not isinstance(enc_out, np.ndarray)
        else enc_out
    )
    batch = enc_np.shape[0]

    forced = dict(forced_decoder_ids or [])
    suppress_set = set(
        suppress_tokens if suppress_tokens is not None else WHISPER_SUPPRESS_TOKENS
    )
    begin_suppress_set = set(
        begin_suppress_tokens
        if begin_suppress_tokens is not None
        else WHISPER_BEGIN_SUPPRESS_TOKENS
    )

    generated = np.full((batch, 1), decoder_start_token_id, dtype=np.int32)
    done = np.zeros(batch, dtype=bool)

    for step in range(max_new_tokens):
        cur_pos = generated.shape[1]
        if cur_pos in forced:
            forced_id = forced[cur_pos]
            next_ids = np.full((batch,), forced_id, dtype=np.int32)
        else:
            logits = decoder_model(
                {
                    "decoder_input_ids": generated,
                    "encoder_hidden_states": enc_np,
                }
            )
            next_logits = ops.convert_to_numpy(logits)[:, -1, :].copy()
            if suppress_set:
                next_logits[:, list(suppress_set)] = -1e9
            if step == 0 and begin_suppress_set:
                next_logits[:, list(begin_suppress_set)] = -1e9
            next_ids = np.argmax(next_logits, axis=-1).astype(np.int32)

        next_ids = np.where(done, eos_token_id, next_ids)
        generated = np.concatenate([generated, next_ids[:, None]], axis=1)
        done = done | (next_ids == eos_token_id)
        if done.all():
            break

    return [list(row) for row in generated]
