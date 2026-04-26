import keras
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable(package="kmodels")
class WhisperAttention(keras.layers.Layer):
    """Multi-head attention shared between Whisper self-attention and cross-attention.

    Reproduces HF ``WhisperAttention`` bit-for-bit. Each instance owns
    four ``Dense`` projections — Q, K, V, output — with biases on
    Q / V / output and no bias on K. Scaling by ``1 / sqrt(head_dim)``
    is applied to the Q output **before** the scaled dot-product
    (``q *= scale`` then ``q @ k.T``), matching the reference
    implementation.

    The same layer handles two attention modes via the optional
    ``key_value_states`` argument to ``call``:

    * **Self-attention** (default): ``key_value_states is None`` — keys
      and values are projected from ``hidden_states``.
    * **Cross-attention**: ``key_value_states`` is the encoder output —
      queries come from ``hidden_states`` (the decoder input), keys and
      values come from the encoder.

    A causal / padding mask of any shape broadcastable to
    ``(B, num_heads, T_q, T_kv)`` may be added to the pre-softmax
    scores via ``attention_mask``.

    Args:
        proj_dim: Total projection dimension (``d_model``). Must be
            divisible by ``num_heads``.
        num_heads: Number of attention heads.
        name_prefix: Optional string prepended to the inner ``Dense``
            layer names. Whisper uses this to mirror the HF naming
            convention (e.g. ``"encoder_layers_0_self_attn_q_proj"``).
            When ``None``, the inner layers are named ``q_proj``,
            ``k_proj``, ``v_proj``, ``out_proj``.
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input Shape:
        - ``hidden_states``: ``(B, T_q, proj_dim)``.
        - ``key_value_states`` (optional): ``(B, T_kv, proj_dim)``.
        - ``attention_mask`` (optional): broadcastable to
          ``(B, num_heads, T_q, T_kv)``.

    Output Shape:
        ``(B, T_q, proj_dim)``.
    """

    def __init__(self, proj_dim, num_heads, name_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.name_prefix = name_prefix
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim**-0.5
        assert proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"

        q_name = f"{name_prefix}_q_proj" if name_prefix else "q_proj"
        k_name = f"{name_prefix}_k_proj" if name_prefix else "k_proj"
        v_name = f"{name_prefix}_v_proj" if name_prefix else "v_proj"
        o_name = f"{name_prefix}_out_proj" if name_prefix else "out_proj"

        self.q_proj = keras.layers.Dense(proj_dim, use_bias=True, name=q_name)
        self.k_proj = keras.layers.Dense(proj_dim, use_bias=False, name=k_name)
        self.v_proj = keras.layers.Dense(proj_dim, use_bias=True, name=v_name)
        self.out_proj = keras.layers.Dense(proj_dim, use_bias=True, name=o_name)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.q_proj.build((None, input_dim))
        self.k_proj.build((None, input_dim))
        self.v_proj.build((None, input_dim))
        self.out_proj.build((None, self.proj_dim))
        self.built = True

    def _split_heads(self, x):
        """Reshape ``(B, T, proj_dim)`` to ``(B, num_heads, T, head_dim)``."""
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        x = ops.reshape(x, (b, t, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, key_value_states=None, attention_mask=None):
        """Run scaled dot-product attention.

        Args:
            hidden_states: Query tensor of shape
                ``(B, T_q, proj_dim)``.
            key_value_states: Optional ``(B, T_kv, proj_dim)`` tensor.
                When supplied, switches the layer to cross-attention
                mode (K and V are projected from this tensor instead of
                from ``hidden_states``).
            attention_mask: Optional additive mask broadcastable to
                ``(B, num_heads, T_q, T_kv)``. Large negative entries
                (typically ``-1e9``) zero out positions after softmax;
                ``0`` entries pass through unchanged.

        Returns:
            Attention output of shape ``(B, T_q, proj_dim)``.
        """
        batch_size = ops.shape(hidden_states)[0]
        kv = key_value_states if key_value_states is not None else hidden_states

        q = self.q_proj(hidden_states) * self.scale
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        if attention_mask is not None:
            scores = scores + ops.cast(attention_mask, scores.dtype)

        attn = ops.softmax(scores, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, -1, self.proj_dim))
        out = self.out_proj(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proj_dim": self.proj_dim,
                "num_heads": self.num_heads,
                "name_prefix": self.name_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SinusoidalPositionEmbedding(keras.layers.Layer):
    """Fixed sinusoidal position embedding for the Whisper encoder.

    Builds a non-trainable ``(max_source_positions, d_model)`` embedding
    table from the original "Attention Is All You Need" sinusoid
    formulation: the first half of the channels carry sines, the second
    half carry cosines, with timescales geometrically interpolated from
    ``1`` to ``10000``. The first ``T`` rows are sliced and added to
    the input, where ``T`` is the encoder sequence length (post
    stride-2 conv stem, so ``T == 1500`` for a full 30-second chunk).

    Used only by :func:`whisper_encoder` — the decoder uses
    :class:`LearnedPositionEmbedding` instead.

    Args:
        max_source_positions: Number of position rows to materialize.
            Always ``1500`` for Whisper (= 30 s of 16 kHz audio with
            320-sample stride after the conv stem).
        d_model: Embedding dimension. Must be even (split into a sine
            half and a cosine half).
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input Shape:
        ``(B, T, d_model)`` with ``T <= max_source_positions``.

    Output Shape:
        Same as input.
    """

    def __init__(self, max_source_positions, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_source_positions = max_source_positions
        self.d_model = d_model

    def build(self, input_shape):
        half = self.d_model // 2
        log_timescale = np.log(10000.0) / (half - 1)
        inv_timescales = np.exp(-log_timescale * np.arange(half))
        positions = np.arange(self.max_source_positions)[:, None]
        scaled = positions * inv_timescales[None, :]
        embed = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(
            np.float32
        )
        self.pos_embed = self.add_weight(
            shape=(self.max_source_positions, self.d_model),
            initializer=keras.initializers.Constant(embed),
            trainable=False,
            name="weight",
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = ops.shape(inputs)[1]
        pe = self.pos_embed[:seq_len]
        return inputs + pe

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_source_positions": self.max_source_positions,
                "d_model": self.d_model,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class LearnedPositionEmbedding(keras.layers.Layer):
    """Trainable position embedding table for the Whisper decoder.

    Mirrors the HF ``nn.Embedding(max_target_positions, d_model)`` used
    in the Whisper decoder: a ``(max_target_positions, d_model)`` weight
    is initialized to zero and learned during training. At call time,
    rows ``[start : start + T]`` are added to the token embeddings,
    where ``start = past_key_values_length``. With the current
    no-cache implementation ``start`` is always ``0``; the parameter is
    kept in the signature for API symmetry with HF's KV-cache path.

    Args:
        max_target_positions: Number of position rows in the table.
            Always ``448`` for Whisper, the maximum supported decoded
            sequence length including the prompt prefix.
        d_model: Embedding dimension.
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input Shape:
        ``(B, T, d_model)``.

    Output Shape:
        Same as input.
    """

    def __init__(self, max_target_positions, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_target_positions = max_target_positions
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            shape=(self.max_target_positions, self.d_model),
            initializer="zeros",
            trainable=True,
            name="weight",
        )
        super().build(input_shape)

    def call(self, inputs, past_key_values_length=0):
        """Add the position embedding rows for the current decoder window.

        Args:
            inputs: Token-embedded decoder input of shape
                ``(B, T, d_model)``.
            past_key_values_length: Offset into the position table; the
                slice ``[start : start + T]`` is added to ``inputs``.
                When running without a KV cache this is always ``0``.

        Returns:
            ``(B, T, d_model)`` tensor with positional information
            added in place.
        """
        seq_len = ops.shape(inputs)[1]
        start = past_key_values_length
        pe = self.pos_embed[start : start + seq_len]
        return inputs + pe

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_target_positions": self.max_target_positions,
                "d_model": self.d_model,
            }
        )
        return config
