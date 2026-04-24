"""Custom Keras 3 layers for Qwen2-VL (LLM + vision tower)."""

import math

import keras
from keras import ops

# ============================================================================
# Qwen2 LLM layers (RMSNorm, GQA attention with M-RoPE, SwiGLU MLP)
# ============================================================================


@keras.saving.register_keras_serializable(package="kmodels")
class Qwen2RMSNorm(keras.layers.Layer):
    """RMSNorm matching HF Qwen2RMSNorm: x * w / sqrt(mean(x^2) + eps)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.hidden_size,),
            initializer="ones",
            trainable=True,
            name="weight",
        )
        super().build(input_shape)

    def call(self, x):
        orig_dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(x * x, axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        return ops.cast(x, orig_dtype) * self.weight

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_size": self.hidden_size, "eps": self.eps})
        return cfg


def _rotate_half(x):
    half = ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concatenate([-x2, x1], axis=-1)


def apply_mrope(q, k, cos, sin):
    """Apply pre-assembled M-RoPE (cos/sin already mrope-interleaved).

    ``q`` / ``k`` shape: ``(B, num_heads, T, head_dim)``.
    ``cos`` / ``sin`` shape: ``(B, 1, T, head_dim)``.
    """
    q_emb = q * cos + _rotate_half(q) * sin
    k_emb = k * cos + _rotate_half(k) * sin
    return q_emb, k_emb


def build_mrope_cos_sin(
    position_ids,  # (3, B, T) int32
    inv_freq,  # (head_dim/2,) float32
    mrope_section,  # list[int] length 3, sum == head_dim/2
):
    """Build (B, 1, T, head_dim) cos & sin tensors for Qwen2-VL M-RoPE."""
    # freqs per axis: (3, B, T, half_dim)
    pos = ops.cast(position_ids, "float32")[..., None]  # (3, B, T, 1)
    freqs = pos * inv_freq[None, None, None, :]  # (3, B, T, half)
    # Duplicate to head_dim via "split" RoPE convention.
    emb = ops.concatenate([freqs, freqs], axis=-1)  # (3, B, T, head)
    cos_all = ops.cos(emb)
    sin_all = ops.sin(emb)

    # Doubled section list: each section appears twice (once per half).
    doubled = list(mrope_section) + list(mrope_section)
    cos_parts, sin_parts = [], []
    offset = 0
    for i, size in enumerate(doubled):
        axis = i % 3
        cos_parts.append(cos_all[axis, :, :, offset : offset + size])
        sin_parts.append(sin_all[axis, :, :, offset : offset + size])
        offset += size
    cos = ops.concatenate(cos_parts, axis=-1)  # (B, T, head)
    sin = ops.concatenate(sin_parts, axis=-1)
    cos = cos[:, None, :, :]  # (B, 1, T, head)
    sin = sin[:, None, :, :]
    return cos, sin


@keras.saving.register_keras_serializable(package="kmodels")
class Qwen2Attention(keras.layers.Layer):
    """GQA attention with Q/K/V biases (Qwen2-specific) + M-RoPE.

    ``num_attention_heads`` query heads, ``num_key_value_heads`` KV heads.
    Q/K/V have bias; O has no bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        name_prefix: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.name_prefix = name_prefix

        p = name_prefix
        self.q_proj = keras.layers.Dense(
            num_attention_heads * self.head_dim,
            use_bias=True,
            name=f"{p}_q_proj" if p else "q_proj",
        )
        self.k_proj = keras.layers.Dense(
            num_key_value_heads * self.head_dim,
            use_bias=True,
            name=f"{p}_k_proj" if p else "k_proj",
        )
        self.v_proj = keras.layers.Dense(
            num_key_value_heads * self.head_dim,
            use_bias=True,
            name=f"{p}_v_proj" if p else "v_proj",
        )
        self.o_proj = keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name=f"{p}_o_proj" if p else "o_proj",
        )

    def build(self, input_shape):
        d = input_shape[-1]
        self.q_proj.build((None, d))
        self.k_proj.build((None, d))
        self.v_proj.build((None, d))
        self.o_proj.build((None, self.hidden_size))
        self.built = True

    def call(self, hidden_states, cos, sin, attention_mask=None):
        b = ops.shape(hidden_states)[0]
        t = ops.shape(hidden_states)[1]

        q = self.q_proj(hidden_states)  # (B, T, H*d)
        k = self.k_proj(hidden_states)  # (B, T, Hkv*d)
        v = self.v_proj(hidden_states)

        q = ops.reshape(q, (b, t, self.num_attention_heads, self.head_dim))
        k = ops.reshape(k, (b, t, self.num_key_value_heads, self.head_dim))
        v = ops.reshape(v, (b, t, self.num_key_value_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))  # (B, H, T, d)
        k = ops.transpose(k, (0, 2, 1, 3))  # (B, Hkv, T, d)
        v = ops.transpose(v, (0, 2, 1, 3))

        q, k = apply_mrope(q, k, cos, sin)

        # Repeat K, V to match num_attention_heads for GQA.
        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if attention_mask is not None:
            scores = scores + ops.cast(attention_mask, scores.dtype)
        attn = ops.softmax(scores, axis=-1)
        out = ops.matmul(attn, v)  # (B, H, T, d)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, t, self.hidden_size))
        return self.o_proj(out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "name_prefix": self.name_prefix,
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="kmodels")
class Qwen2MLP(keras.layers.Layer):
    """SwiGLU MLP: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        name_prefix: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.name_prefix = name_prefix
        p = name_prefix
        self.gate_proj = keras.layers.Dense(
            intermediate_size,
            use_bias=False,
            name=f"{p}_gate_proj" if p else "gate_proj",
        )
        self.up_proj = keras.layers.Dense(
            intermediate_size,
            use_bias=False,
            name=f"{p}_up_proj" if p else "up_proj",
        )
        self.down_proj = keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name=f"{p}_down_proj" if p else "down_proj",
        )

    def build(self, input_shape):
        d = input_shape[-1]
        self.gate_proj.build((None, d))
        self.up_proj.build((None, d))
        self.down_proj.build((None, self.intermediate_size))
        self.built = True

    def call(self, x):
        return self.down_proj(
            keras.activations.silu(self.gate_proj(x)) * self.up_proj(x)
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "name_prefix": self.name_prefix,
            }
        )
        return cfg


# ============================================================================
# Vision tower layers (ViT with 2D RoPE and concat-QKV)
# ============================================================================


def build_vision_rope_cos_sin(
    grid_thw,  # (num_grid, 3) int32: per-image (t, h, w)
    inv_freq,  # (head_dim/4,) float32
    head_dim: int,
    spatial_merge_size: int = 2,
):
    """Build 2D vision RoPE cos/sin for the full patch sequence.

    Position ids are shuffled into merge-size groups to match HF's
    ``rot_pos_emb``: the (h, w) grid is partitioned into ``merge_size x
    merge_size`` tiles, permuted so each tile is contiguous, then flattened.
    This makes the positional encoding consistent with the image processor's
    patch ordering.

    Returns ``(total_patches, head_dim)`` cos & sin tensors for
    "split"-style RoPE.
    """
    import numpy as np

    grid_thw_np = (
        ops.convert_to_numpy(grid_thw)
        if not isinstance(grid_thw, np.ndarray)
        else grid_thw
    )
    inv_freq_np = (
        ops.convert_to_numpy(inv_freq)
        if not isinstance(inv_freq, np.ndarray)
        else inv_freq
    )
    ms = spatial_merge_size

    all_pos = []
    for t, h, w in grid_thw_np:
        hpos = np.arange(h)[:, None].repeat(w, axis=1)  # (h, w)
        hpos = hpos.reshape(h // ms, ms, w // ms, ms)
        hpos = hpos.transpose(0, 2, 1, 3).reshape(-1)  # (h*w,)
        wpos = np.arange(w)[None, :].repeat(h, axis=0)
        wpos = wpos.reshape(h // ms, ms, w // ms, ms)
        wpos = wpos.transpose(0, 2, 1, 3).reshape(-1)
        pos = np.stack([hpos, wpos], axis=-1)  # (h*w, 2)
        pos = np.tile(pos, (t, 1))
        all_pos.append(pos)
    pos = np.concatenate(all_pos, axis=0).astype(np.float32)

    freqs_h = pos[:, 0:1] * inv_freq_np[None, :]  # (N, head_dim/4)
    freqs_w = pos[:, 1:2] * inv_freq_np[None, :]
    freqs = np.concatenate([freqs_h, freqs_w], axis=-1)  # (N, head_dim/2)
    emb = np.concatenate([freqs, freqs], axis=-1)  # (N, head_dim)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return ops.convert_to_tensor(cos), ops.convert_to_tensor(sin)


@keras.saving.register_keras_serializable(package="kmodels")
class VisionAttention(keras.layers.Layer):
    """ViT attention for Qwen2-VL. Uses packed QKV weight + 2D RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, name_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.name_prefix = name_prefix
        p = name_prefix
        self.qkv = keras.layers.Dense(
            3 * hidden_size,
            use_bias=True,
            name=f"{p}_qkv" if p else "qkv",
        )
        self.proj = keras.layers.Dense(
            hidden_size,
            use_bias=True,
            name=f"{p}_proj" if p else "proj",
        )

    def build(self, input_shape):
        d = input_shape[-1]
        self.qkv.build((None, d))
        self.proj.build((None, self.hidden_size))
        self.built = True

    def call(self, x, cos, sin, attention_mask=None):
        # x: (B, N, C) - but Qwen2VL vision is typically unbatched (B=1 packed)
        b = ops.shape(x)[0]
        n = ops.shape(x)[1]
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE on q,k. cos/sin shape: (N, head_dim) -> broadcast to (1, 1, N, head_dim)
        cos_b = cos[None, None, :, :]
        sin_b = sin[None, None, :, :]
        q = q * cos_b + _rotate_half(q) * sin_b
        k = k * cos_b + _rotate_half(k) * sin_b

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if attention_mask is not None:
            scores = scores + ops.cast(attention_mask, scores.dtype)
        attn = ops.softmax(scores, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n, self.hidden_size))
        return self.proj(out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "name_prefix": self.name_prefix,
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="kmodels")
class VisionMLP(keras.layers.Layer):
    """ViT MLP: fc1 -> quick_gelu -> fc2."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, name_prefix=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.name_prefix = name_prefix
        p = name_prefix
        self.fc1 = keras.layers.Dense(
            intermediate_size, use_bias=True, name=f"{p}_fc1" if p else "fc1"
        )
        self.fc2 = keras.layers.Dense(
            hidden_size, use_bias=True, name=f"{p}_fc2" if p else "fc2"
        )

    def build(self, input_shape):
        d = input_shape[-1]
        self.fc1.build((None, d))
        self.fc2.build((None, self.intermediate_size))
        self.built = True

    def call(self, x):
        h = self.fc1(x)
        h = h * ops.sigmoid(1.702 * h)  # quick_gelu
        return self.fc2(h)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "name_prefix": self.name_prefix,
            }
        )
        return cfg
