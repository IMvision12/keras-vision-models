import math

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="kmodels")
class Qwen2RMSNorm(keras.layers.Layer):
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
    q_emb = q * cos + _rotate_half(q) * sin
    k_emb = k * cos + _rotate_half(k) * sin
    return q_emb, k_emb


def build_mrope_cos_sin(
    position_ids,
    inv_freq,
    mrope_section,
):
    pos = ops.cast(position_ids, "float32")[..., None]
    freqs = pos * inv_freq[None, None, None, :]
    emb = ops.concatenate([freqs, freqs], axis=-1)
    cos_all = ops.cos(emb)
    sin_all = ops.sin(emb)

    doubled = list(mrope_section) + list(mrope_section)
    cos_parts, sin_parts = [], []
    offset = 0
    for i, size in enumerate(doubled):
        axis = i % 3
        cos_parts.append(cos_all[axis, :, :, offset : offset + size])
        sin_parts.append(sin_all[axis, :, :, offset : offset + size])
        offset += size
    cos = ops.concatenate(cos_parts, axis=-1)
    sin = ops.concatenate(sin_parts, axis=-1)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return cos, sin


@keras.saving.register_keras_serializable(package="kmodels")
class Qwen2Attention(keras.layers.Layer):
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

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = ops.reshape(q, (b, t, self.num_attention_heads, self.head_dim))
        k = ops.reshape(k, (b, t, self.num_key_value_heads, self.head_dim))
        v = ops.reshape(v, (b, t, self.num_key_value_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        q, k = apply_mrope(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if attention_mask is not None:
            scores = scores + ops.cast(attention_mask, scores.dtype)
        attn = ops.softmax(scores, axis=-1)
        out = ops.matmul(attn, v)
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


def build_vision_rope_cos_sin(
    grid_thw,
    inv_freq,
    head_dim: int,
    spatial_merge_size: int = 2,
):
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
        hpos = np.arange(h)[:, None].repeat(w, axis=1)
        hpos = hpos.reshape(h // ms, ms, w // ms, ms)
        hpos = hpos.transpose(0, 2, 1, 3).reshape(-1)
        wpos = np.arange(w)[None, :].repeat(h, axis=0)
        wpos = wpos.reshape(h // ms, ms, w // ms, ms)
        wpos = wpos.transpose(0, 2, 1, 3).reshape(-1)
        pos = np.stack([hpos, wpos], axis=-1)
        pos = np.tile(pos, (t, 1))
        all_pos.append(pos)
    pos = np.concatenate(all_pos, axis=0).astype(np.float32)

    freqs_h = pos[:, 0:1] * inv_freq_np[None, :]
    freqs_w = pos[:, 1:2] * inv_freq_np[None, :]
    freqs = np.concatenate([freqs_h, freqs_w], axis=-1)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return ops.convert_to_tensor(cos), ops.convert_to_tensor(sin)


@keras.saving.register_keras_serializable(package="kmodels")
class VisionAttention(keras.layers.Layer):
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
        b = ops.shape(x)[0]
        n = ops.shape(x)[1]
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

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
        h = h * ops.sigmoid(1.702 * h)
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
