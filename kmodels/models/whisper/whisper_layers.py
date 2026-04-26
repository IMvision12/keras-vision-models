import keras
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable(package="kmodels")
class WhisperAttention(keras.layers.Layer):
    """Multi-head attention for Whisper, shared between self-attn and cross-attn.

    Matches HF ``WhisperAttention`` exactly: Q/O have bias, K has no bias,
    V has bias. When ``key_value_states`` is passed to ``call``, the layer
    operates in cross-attention mode (K/V come from encoder states instead
    of hidden_states).

    Note: HF uses standard scaled dot-product attention with scale = 1/sqrt(head_dim)
    applied to the Q projection output (i.e. ``q *= scale`` before matmul).

    Args:
        proj_dim: Total projection dimension (``d_model``).
        num_heads: Number of attention heads.
        name_prefix: Optional prefix used for sub-layer names (matches HF).
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
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        x = ops.reshape(x, (b, t, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, key_value_states=None, attention_mask=None):
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
    """Whisper's fixed sinusoidal position embedding for the encoder.

    Produces a ``(max_source_positions, d_model)`` embedding table following
    the original Whisper (and "Attention Is All You Need") formulation.

    Used only by the encoder — the decoder uses a learned table.
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
    """Whisper's learned position embedding table for the decoder.

    Matches ``nn.Embedding(max_target_positions, d_model)`` with a fixed
    offset of 0 and is added to the token embeddings at position ``[:T]``.
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
