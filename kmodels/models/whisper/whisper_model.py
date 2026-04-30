import keras
from keras import layers, ops

from kmodels.model_registry import register_model
from kmodels.weight_utils import get_all_weight_names, load_weights_from_config

from .config import WHISPER_MODEL_CONFIG, WHISPER_WEIGHTS_CONFIG
from .whisper_layers import (
    LearnedPositionEmbedding,
    SinusoidalPositionEmbedding,
    WhisperAttention,
)


def _gelu(x):
    """Exact GELU (``approximate=False``) — the activation Whisper uses.

    HF and OpenAI's Whisper use the error-function form of GELU, not the
    tanh approximation that many transformer codebases default to.
    Wrapping it as a module-level function lets it be passed to
    :class:`keras.layers.Lambda` and round-trip through serialization.
    """
    return keras.activations.gelu(x, approximate=False)


def whisper_encoder_block(x, d_model, num_heads, ffn_dim, layer_idx):
    """One pre-LN encoder block: self-attention + MLP with residuals.

    Layout matches the HF Whisper encoder layer:

    1. ``LayerNorm`` → :class:`WhisperAttention` (self-attn) → residual.
    2. ``LayerNorm`` → ``Dense(ffn_dim)`` → exact GELU →
       ``Dense(d_model)`` → residual.

    All sub-layer names follow the HF convention
    (``encoder_layers_{i}_self_attn_layer_norm``, ``..._fc1``,
    ``..._fc2``, ...) so converted PyTorch weights drop in cleanly.

    Args:
        x: Input tensor of shape ``(B, T, d_model)``.
        d_model: Hidden / embedding dimension.
        num_heads: Number of self-attention heads.
        ffn_dim: MLP intermediate dimension (conventionally
            ``4 * d_model``).
        layer_idx: Index of the block within the encoder; used to name
            sub-layers.

    Returns:
        Output tensor of shape ``(B, T, d_model)``.
    """
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


def whisper_decoder_block(
    x, encoder_hidden_states, causal_mask, d_model, num_heads, ffn_dim, layer_idx
):
    """One pre-LN decoder block: self-attn + cross-attn + MLP with residuals.

    Layout matches the HF Whisper decoder layer:

    1. ``LayerNorm`` → :class:`WhisperAttention` self-attn (with
       ``causal_mask``) → residual.
    2. ``LayerNorm`` → :class:`WhisperAttention` cross-attn over
       ``encoder_hidden_states`` → residual.
    3. ``LayerNorm`` → ``Dense(ffn_dim)`` → exact GELU →
       ``Dense(d_model)`` → residual.

    All sub-layer names follow the HF convention
    (``decoder_layers_{i}_self_attn_layer_norm``,
    ``..._encoder_attn_layer_norm``, ...) so converted PyTorch weights
    drop in cleanly.

    Args:
        x: Decoder input tensor of shape ``(B, L, d_model)``.
        encoder_hidden_states: Encoder output tensor of shape
            ``(B, T, d_model)`` consumed by the cross-attention K / V
            projections.
        causal_mask: Additive mask of shape ``(1, 1, L, L)`` enforcing
            left-to-right attention in the self-attention sub-layer.
        d_model: Hidden / embedding dimension.
        num_heads: Number of attention heads (shared between self-attn
            and cross-attn).
        ffn_dim: MLP intermediate dimension.
        layer_idx: Index of the block within the decoder; used to name
            sub-layers.

    Returns:
        Output tensor of shape ``(B, L, d_model)``.
    """
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


def whisper_encoder(
    d_model,
    num_mel_bins,
    max_source_positions,
    encoder_layers,
    encoder_attention_heads,
    encoder_ffn_dim,
    name="encoder",
):
    """Build the Whisper encoder as a Functional :class:`keras.Model`.

    Architecture (matches HF / OpenAI exactly):

    1. ``input_features`` of shape ``(B, num_mel_bins, T_audio)``
       permuted to ``(B, T_audio, num_mel_bins)`` for ``Conv1D``.
    2. Two ``Conv1D`` layers (kernel 3, stride 1 then stride 2) with
       exact GELU activations and explicit ``ZeroPadding1D(1)``,
       halving the time axis once.
    3. :class:`SinusoidalPositionEmbedding` adding fixed position
       encodings.
    4. ``encoder_layers`` repetitions of :func:`whisper_encoder_block`.
    5. Final ``LayerNorm``.

    Args:
        d_model: Hidden / embedding dimension.
        num_mel_bins: Number of mel bins in the input log-mel
            spectrogram (``80`` for v1, ``128`` for v3).
        max_source_positions: Length of the encoder output time axis
            after the stride-2 conv. Always ``1500`` for Whisper.
        encoder_layers: Number of stacked transformer blocks.
        encoder_attention_heads: Heads per self-attention block.
        encoder_ffn_dim: MLP intermediate dimension (conventionally
            ``4 * d_model``).
        name: Returned model name.

    Returns:
        ``keras.Model`` mapping
        ``input_features: (B, num_mel_bins, T_audio)`` →
        ``(B, max_source_positions, d_model)``.
    """
    mel = layers.Input(shape=(num_mel_bins, None), name="input_features")
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
        max_source_positions=max_source_positions,
        d_model=d_model,
        name="encoder_embed_positions",
    )(x)

    for i in range(encoder_layers):
        x = whisper_encoder_block(
            x,
            d_model=d_model,
            num_heads=encoder_attention_heads,
            ffn_dim=encoder_ffn_dim,
            layer_idx=i,
        )

    x = layers.LayerNormalization(epsilon=1e-5, name="encoder_layer_norm")(x)
    return keras.Model(inputs=mel, outputs=x, name=name)


def _make_causal_mask_from_ids(decoder_input_ids):
    """Build an additive causal attention mask matching the input length.

    Returns a tensor of shape ``(1, 1, T, T)`` whose ``(i, j)`` entries
    are ``-1e9`` when ``j > i`` and ``0`` otherwise. Suitable for being
    added before softmax in :class:`WhisperAttention`.

    Wrapped in a ``keras.layers.Lambda`` inside the decoder graph so
    the mask shape stays dynamic with the decoded sequence length
    ``T``.
    """
    seq_len = ops.shape(decoder_input_ids)[1]
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    mask = ops.cast(j > i, "float32") * -1e9
    return mask[None, None, :, :]


def whisper_decoder(
    d_model,
    max_target_positions,
    vocab_size,
    decoder_layers,
    decoder_attention_heads,
    decoder_ffn_dim,
    name="decoder",
):
    """Build the Whisper decoder as a Functional :class:`keras.Model`.

    Architecture (matches HF / OpenAI exactly):

    1. ``decoder_input_ids`` of shape ``(B, L)`` → token embedding.
    2. :class:`LearnedPositionEmbedding` added in place.
    3. ``decoder_layers`` repetitions of :func:`whisper_decoder_block`,
       each cross-attending to ``encoder_hidden_states``.
    4. Final ``LayerNorm``.
    5. Tied LM head: ``logits = x @ embedding_matrix.T`` (no separate
       output projection — the input embedding weights are reused).

    Args:
        d_model: Hidden / embedding dimension.
        max_target_positions: Maximum decoded sequence length the
            position table supports. Always ``448`` for Whisper.
        vocab_size: Token vocabulary size (``51865`` for v1,
            ``51866`` for v3).
        decoder_layers: Number of stacked transformer blocks.
        decoder_attention_heads: Heads per attention sub-layer (shared
            between self-attn and cross-attn).
        decoder_ffn_dim: MLP intermediate dimension.
        name: Returned model name.

    Returns:
        ``keras.Model`` whose dict-input
        ``{"decoder_input_ids": (B, L),
        "encoder_hidden_states": (B, T, d_model)}`` maps to
        ``(B, L, vocab_size)`` logits.
    """
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
        max_target_positions=max_target_positions,
        d_model=d_model,
        name="decoder_embed_positions",
    )(x)

    causal_mask = layers.Lambda(
        _make_causal_mask_from_ids,
        name="decoder_causal_mask",
        output_shape=lambda s: (1, 1, s[1], s[1]),
    )(decoder_input_ids)

    for i in range(decoder_layers):
        x = whisper_decoder_block(
            x,
            encoder_hidden_states=encoder_hidden_states,
            causal_mask=causal_mask,
            d_model=d_model,
            num_heads=decoder_attention_heads,
            ffn_dim=decoder_ffn_dim,
            layer_idx=i,
        )

    x = layers.LayerNormalization(epsilon=1e-5, name="decoder_layer_norm")(x)

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


class Whisper(keras.Model):
    """Whisper encoder-decoder transformer for ASR / translation.

    Wires :func:`whisper_encoder` and :func:`whisper_decoder` into a single
    Functional graph so the full model can be called with one dict:

    >>> out = model({"input_features": mel, "decoder_input_ids": ids})
    >>> out["encoder_hidden_states"]   # (B, T, d_model)
    >>> out["logits"]                  # (B, L, vocab_size)

    This is the teacher-forced training path. For autoregressive
    inference use the :class:`WhisperGenerate` wrapper, which calls the
    encoder once and the decoder per step via the ``model.encoder`` and
    ``model.decoder`` attributes.

    .. note::
        Unlike vision models in kmodels, Whisper has a **fixed input
        shape** dictated by the audio pipeline: log-mel spectrograms
        of ``(num_mel_bins, max_source_positions * 2)`` —
        ``(80, 3000)`` for v1/v2 variants, ``(128, 3000)`` for
        large-v3 / large-v3-turbo. There is no ``input_shape`` kwarg;
        feed :class:`WhisperFeatureExtractor` output directly.

    Args:
        d_model: Hidden / embedding dimension. ``384`` (tiny) →
            ``1280`` (large).
        encoder_layers: Number of encoder transformer blocks.
        decoder_layers: Number of decoder transformer blocks.
        encoder_attention_heads: Encoder self-attn head count.
        decoder_attention_heads: Decoder self-attn / cross-attn head
            count.
        encoder_ffn_dim: Encoder MLP hidden dim. Conventionally
            ``4 * d_model``.
        decoder_ffn_dim: Decoder MLP hidden dim. Conventionally
            ``4 * d_model``.
        num_mel_bins: Mel bin count of the input log-mel spectrogram.
            ``80`` for v1 variants, ``128`` for large-v3 /
            large-v3-turbo.
        max_source_positions: Max encoder position (post-stride-2 conv).
            Always ``1500`` for Whisper (= 30 s @ 16 kHz / 320 hop).
        max_target_positions: Max decoded length, including special
            prompt prefix. Always ``448``.
        vocab_size: Token vocabulary size. ``51865`` for v1 variants,
            ``51866`` for v3 (adds Cantonese language id).
        name: Model name. Defaults to ``"Whisper"``.
        **kwargs: Additional ``keras.Model`` keyword arguments.
    """

    def __init__(
        self,
        d_model=384,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=6,
        decoder_attention_heads=6,
        encoder_ffn_dim=1536,
        decoder_ffn_dim=1536,
        num_mel_bins=80,
        max_source_positions=1500,
        max_target_positions=448,
        vocab_size=51865,
        name="Whisper",
        **kwargs,
    ):
        encoder = whisper_encoder(
            d_model=d_model,
            num_mel_bins=num_mel_bins,
            max_source_positions=max_source_positions,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            name=f"{name}_encoder",
        )
        decoder = whisper_decoder(
            d_model=d_model,
            max_target_positions=max_target_positions,
            vocab_size=vocab_size,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            name=f"{name}_decoder",
        )

        input_features = layers.Input(shape=(num_mel_bins, None), name="input_features")
        decoder_input_ids = layers.Input(
            shape=(None,), dtype="int32", name="decoder_input_ids"
        )
        encoder_hidden_states = encoder(input_features)
        logits = decoder(
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
            }
        )

        super().__init__(
            inputs={
                "input_features": input_features,
                "decoder_input_ids": decoder_input_ids,
            },
            outputs={
                "encoder_hidden_states": encoder_hidden_states,
                "logits": logits,
            },
            name=name,
            **kwargs,
        )

        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.vocab_size = vocab_size

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "encoder_layers": self.encoder_layers,
                "decoder_layers": self.decoder_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "decoder_attention_heads": self.decoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "num_mel_bins": self.num_mel_bins,
                "max_source_positions": self.max_source_positions,
                "max_target_positions": self.max_target_positions,
                "vocab_size": self.vocab_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def WhisperTiny(weights="openai", name="WhisperTiny", **kwargs):
    model = Whisper(**WHISPER_MODEL_CONFIG["WhisperTiny"], name=name, **kwargs)

    if weights in get_all_weight_names(WHISPER_WEIGHTS_CONFIG):
        load_weights_from_config("WhisperTiny", weights, model, WHISPER_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def WhisperBase(weights="openai", name="WhisperBase", **kwargs):
    model = Whisper(**WHISPER_MODEL_CONFIG["WhisperBase"], name=name, **kwargs)

    if weights in get_all_weight_names(WHISPER_WEIGHTS_CONFIG):
        load_weights_from_config("WhisperBase", weights, model, WHISPER_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def WhisperSmall(weights="openai", name="WhisperSmall", **kwargs):
    model = Whisper(**WHISPER_MODEL_CONFIG["WhisperSmall"], name=name, **kwargs)

    if weights in get_all_weight_names(WHISPER_WEIGHTS_CONFIG):
        load_weights_from_config("WhisperSmall", weights, model, WHISPER_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def WhisperMedium(weights="openai", name="WhisperMedium", **kwargs):
    model = Whisper(**WHISPER_MODEL_CONFIG["WhisperMedium"], name=name, **kwargs)

    if weights in get_all_weight_names(WHISPER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "WhisperMedium", weights, model, WHISPER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def WhisperLarge(weights="openai", name="WhisperLarge", **kwargs):
    model = Whisper(**WHISPER_MODEL_CONFIG["WhisperLarge"], name=name, **kwargs)

    if weights in get_all_weight_names(WHISPER_WEIGHTS_CONFIG):
        load_weights_from_config("WhisperLarge", weights, model, WHISPER_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def WhisperLargeV2(weights="openai", name="WhisperLargeV2", **kwargs):
    model = Whisper(**WHISPER_MODEL_CONFIG["WhisperLargeV2"], name=name, **kwargs)

    if weights in get_all_weight_names(WHISPER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "WhisperLargeV2", weights, model, WHISPER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def WhisperLargeV3(weights="openai", name="WhisperLargeV3", **kwargs):
    model = Whisper(**WHISPER_MODEL_CONFIG["WhisperLargeV3"], name=name, **kwargs)

    if weights in get_all_weight_names(WHISPER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "WhisperLargeV3", weights, model, WHISPER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def WhisperLargeV3Turbo(weights="openai", name="WhisperLargeV3Turbo", **kwargs):
    model = Whisper(**WHISPER_MODEL_CONFIG["WhisperLargeV3Turbo"], name=name, **kwargs)

    if weights in get_all_weight_names(WHISPER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "WhisperLargeV3Turbo", weights, model, WHISPER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
