import math

import keras
import numpy as np
from keras import layers, ops


def _inverse_sigmoid(x, eps=1e-3):
    x = ops.clip(x, eps, 1.0 - eps)
    return ops.log(x / (1.0 - x))


def _box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = ops.split(boxes, 4, axis=-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return ops.concatenate([x0, y0, x1, y1], axis=-1)


def _rotate_pairwise(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = ops.stack([-x2, x1], axis=-1)
    shape = ops.shape(x)
    return ops.reshape(rotated, shape)


def _apply_rotary_pos_emb_2d(q, k, cos, sin):
    q_embed = q * cos + _rotate_pairwise(q) * sin
    k_embed = k * cos + _rotate_pairwise(k) * sin
    return q_embed, k_embed


def _sine_encode_boxes(boxes, num_pos_feats=128, temperature=10000):
    """Encode box coords matching HF: interleaved sin/cos, order (y, x, w, h)."""
    scale = 2 * 3.141592653589793
    dim_t = ops.cast(ops.arange(num_pos_feats), dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_pos_feats)

    def _encode_coord(coord):
        c = coord * scale
        c = ops.expand_dims(c, axis=-1) / dim_t
        c_sin = ops.sin(c[..., 0::2])
        c_cos = ops.cos(c[..., 1::2])
        half = num_pos_feats // 2
        parts = []
        for j in range(half):
            parts.append(c_sin[:, :, j : j + 1])
            parts.append(c_cos[:, :, j : j + 1])
        return ops.concatenate(parts, axis=-1)

    pos_y = _encode_coord(boxes[:, :, 1])
    pos_x = _encode_coord(boxes[:, :, 0])
    pos_w = _encode_coord(boxes[:, :, 2])
    pos_h = _encode_coord(boxes[:, :, 3])
    return ops.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)


def _compute_sine_pos_encoding(
    height, width, num_pos_feats, temperature=10000, normalize=True
):
    scale = 2 * math.pi
    y_embed = np.cumsum(np.ones((1, height, width), dtype=np.float32), axis=1)
    x_embed = np.cumsum(np.ones((1, height, width), dtype=np.float32), axis=2)

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = temperature ** (2 * np.floor(dim_t / 2) / num_pos_feats)

    pos_x = x_embed[..., np.newaxis] / dim_t
    pos_y = y_embed[..., np.newaxis] / dim_t

    pos_x_sin = np.sin(pos_x[:, :, :, 0::2])
    pos_x_cos = np.cos(pos_x[:, :, :, 1::2])
    pos_y_sin = np.sin(pos_y[:, :, :, 0::2])
    pos_y_cos = np.cos(pos_y[:, :, :, 1::2])

    pos_x = np.stack([pos_x_sin, pos_x_cos], axis=4).reshape(
        1, height, width, num_pos_feats
    )
    pos_y = np.stack([pos_y_sin, pos_y_cos], axis=4).reshape(
        1, height, width, num_pos_feats
    )

    pos = np.concatenate([pos_y, pos_x], axis=-1)
    pos = pos.transpose(0, 3, 1, 2)
    return pos


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3LearnableEmbedding(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, apply_sigmoid=False, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.apply_sigmoid = apply_sigmoid

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer="zeros",
        )
        self.built = True

    def call(self, batch_ref):
        batch_size = ops.shape(batch_ref)[0]
        emb = self.embeddings
        if self.apply_sigmoid:
            emb = ops.sigmoid(emb)
        emb = ops.expand_dims(emb, 0)
        return ops.broadcast_to(emb, (batch_size,) + ops.shape(emb)[1:])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "apply_sigmoid": self.apply_sigmoid,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, hidden_size, pretrain_grid, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.pretrain_grid = pretrain_grid
        self.grid_size = grid_size

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_patches, self.hidden_size),
            initializer="zeros",
        )
        self.built = True

    def call(self, x):
        pos = ops.reshape(
            self.embeddings,
            (1, self.pretrain_grid, self.pretrain_grid, self.hidden_size),
        )
        if self.grid_size != self.pretrain_grid:
            # Match HF: tile then crop (NCHW tile, then back to NHWC)
            pos = ops.transpose(pos, (0, 3, 1, 2))  # (1, C, H, W)
            repeat_h = self.grid_size // self.pretrain_grid + 1
            repeat_w = self.grid_size // self.pretrain_grid + 1
            pos = ops.tile(pos, (1, 1, repeat_h, repeat_w))
            pos = pos[:, :, : self.grid_size, : self.grid_size]
            pos = ops.transpose(pos, (0, 2, 3, 1))  # (1, H, W, C)
        pos = ops.reshape(pos, (1, self.grid_size * self.grid_size, self.hidden_size))
        return x + pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "hidden_size": self.hidden_size,
                "pretrain_grid": self.pretrain_grid,
                "grid_size": self.grid_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3ViTRotaryEmbedding(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        end_x,
        end_y,
        rope_theta=10000.0,
        scale=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.end_x = end_x
        self.end_y = end_y
        self.rope_theta = rope_theta
        self.scale = scale
        self.head_dim = hidden_size // num_attention_heads

        dim = self.head_dim
        freqs = 1.0 / (
            rope_theta ** (np.arange(0, dim, 4, dtype="float32")[: dim // 4] / dim)
        )

        # Match HF: flattened positions with x varying fastest (row-major)
        flat_idx = np.arange(end_x * end_y, dtype="float32")
        x_positions = (flat_idx % end_x) * scale
        y_positions = (flat_idx // end_x) * scale

        freqs_x = np.outer(x_positions, freqs)  # (N, dim//4)
        freqs_y = np.outer(y_positions, freqs)  # (N, dim//4)

        inv_freq = np.concatenate([freqs_x, freqs_y], axis=-1)  # (N, dim//2)
        inv_freq = np.repeat(inv_freq, 2, axis=-1)  # (N, dim)

        self._cos = np.cos(inv_freq).astype("float32")
        self._sin = np.sin(inv_freq).astype("float32")

    def call(self, inputs=None):
        cos = ops.convert_to_tensor(self._cos)
        sin = ops.convert_to_tensor(self._sin)
        return cos, sin

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "end_x": self.end_x,
                "end_y": self.end_y,
                "rope_theta": self.rope_theta,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3ViTRoPEAttention(layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.hidden_size, name="q_proj")
        self.q_proj.build(input_shape)
        self.k_proj = layers.Dense(self.hidden_size, name="k_proj")
        self.k_proj.build(input_shape)
        self.v_proj = layers.Dense(self.hidden_size, name="v_proj")
        self.v_proj.build(input_shape)
        self.o_proj = layers.Dense(self.hidden_size, name="o_proj")
        self.o_proj.build(input_shape)
        self.built = True

    def call(self, hidden_states, position_embeddings):
        cos, sin = position_embeddings
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        seq_len = shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = ops.reshape(
            q, (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        )
        k = ops.reshape(
            k, (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        )
        v = ops.reshape(
            v, (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        )

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(ops.expand_dims(cos, 0), 0)
        sin = ops.expand_dims(ops.expand_dims(sin, 0), 0)
        q, k = _apply_rotary_pos_emb_2d(q, k, cos, sin)

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, v)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = self.o_proj(attn_output)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3ViTLayerScale(layers.Layer):
    def __init__(self, hidden_size, init_value, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.init_value = init_value

    def build(self, input_shape):
        self.lambda1 = self.add_weight(
            name="lambda1",
            shape=(self.hidden_size,),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        self.built = True

    def call(self, x):
        return x * self.lambda1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "init_value": self.init_value,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3ViTLayer(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        window_size=0,
        image_size=72,
        layer_norm_eps=1e-6,
        layer_scale_init_value=None,
        rope_theta=10000.0,
        config_window_size=24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.rope_theta = rope_theta
        self.config_window_size = config_window_size

    def build(self, input_shape):
        seq_shape = (None, None, self.hidden_size)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.attn = SAM3ViTRoPEAttention(
            self.hidden_size, self.num_attention_heads, name="attention"
        )
        self.attn.build(seq_shape)

        if self.window_size > 0:
            end = self.window_size
            rope_scale = 1.0
        else:
            end = self.image_size
            rope_scale = self.config_window_size / end
        self.rotary_emb = SAM3ViTRotaryEmbedding(
            self.hidden_size,
            self.num_attention_heads,
            end_x=end,
            end_y=end,
            rope_theta=self.rope_theta,
            scale=rope_scale,
            name="rotary_emb",
        )

        self.mlp_fc1 = layers.Dense(self.intermediate_size, name="mlp_fc1")
        self.mlp_fc1.build(seq_shape)
        self.mlp_fc2 = layers.Dense(self.hidden_size, name="mlp_fc2")
        self.mlp_fc2.build((None, None, self.intermediate_size))

        if self.layer_scale_init_value is not None:
            self.layer_scale1 = SAM3ViTLayerScale(
                self.hidden_size, self.layer_scale_init_value, name="layer_scale1"
            )
            self.layer_scale1.build(seq_shape)
            self.layer_scale2 = SAM3ViTLayerScale(
                self.hidden_size, self.layer_scale_init_value, name="layer_scale2"
            )
            self.layer_scale2.build(seq_shape)

        self.built = True

    def _window_partition(self, x, window_size):
        shape = ops.shape(x)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        padded_h = height + pad_h
        padded_w = width + pad_w

        x = ops.reshape(
            x,
            (
                batch_size,
                padded_h // window_size,
                window_size,
                padded_w // window_size,
                window_size,
                channels,
            ),
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (-1, window_size * window_size, channels))
        return x, (padded_h, padded_w)

    def _window_unpartition(self, windows, window_size, pad_hw, original_hw):
        padded_h, padded_w = pad_hw
        height, width = original_hw
        num_h = padded_h // window_size
        num_w = padded_w // window_size

        channels = ops.shape(windows)[-1]
        batch_size = ops.shape(windows)[0] // (num_h * num_w)

        x = ops.reshape(
            windows, (batch_size, num_h, num_w, window_size, window_size, channels)
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (batch_size, padded_h, padded_w, channels))
        x = x[:, :height, :width, :]
        return x

    def call(self, hidden_states):
        shape = ops.shape(hidden_states)
        batch_size, height, width = shape[0], shape[1], shape[2]

        residual = hidden_states

        x = ops.reshape(hidden_states, (batch_size, height * width, self.hidden_size))
        x = self.layer_norm1(x)

        if self.window_size > 0:
            x = ops.reshape(x, (batch_size, height, width, self.hidden_size))
            x, pad_hw = self._window_partition(x, self.window_size)
        else:
            pass

        pos_emb = self.rotary_emb()
        x = self.attn(x, pos_emb)

        if self.window_size > 0:
            x = self._window_unpartition(x, self.window_size, pad_hw, (height, width))
            x = ops.reshape(x, (batch_size, height * width, self.hidden_size))

        if self.layer_scale_init_value is not None:
            x = self.layer_scale1(x)

        x = ops.reshape(x, (batch_size, height, width, self.hidden_size))
        hidden_states = residual + x

        residual = hidden_states
        x = ops.reshape(hidden_states, (batch_size, height * width, self.hidden_size))
        x = self.layer_norm2(x)
        x = self.mlp_fc1(x)
        x = ops.nn.gelu(x, approximate=False)
        x = self.mlp_fc2(x)

        if self.layer_scale_init_value is not None:
            x = self.layer_scale2(x)

        x = ops.reshape(x, (batch_size, height, width, self.hidden_size))
        hidden_states = residual + x

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "window_size": self.window_size,
                "image_size": self.image_size,
                "layer_norm_eps": self.layer_norm_eps,
                "layer_scale_init_value": self.layer_scale_init_value,
                "config_window_size": self.config_window_size,
                "rope_theta": self.rope_theta,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3SinePositionEmbedding(layers.Layer):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, **kwargs
    ):
        super().__init__(**kwargs)
        if scale is None:
            scale = 2 * math.pi
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def call(self, height, width, batch_size=1):
        not_mask = ops.ones((batch_size, height, width), dtype="float32")
        y_embed = ops.cumsum(not_mask, axis=1)
        x_embed = ops.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.cast(ops.arange(self.num_pos_feats), dtype="float32")
        dim_t = self.temperature ** (2 * ops.floor(dim_t / 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        pos_x_sin = ops.sin(pos_x[:, :, :, 0::2])
        pos_x_cos = ops.cos(pos_x[:, :, :, 1::2])
        pos_y_sin = ops.sin(pos_y[:, :, :, 0::2])
        pos_y_cos = ops.cos(pos_y[:, :, :, 1::2])

        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=4),
            (batch_size, height, width, self.num_pos_feats),
        )
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=4),
            (batch_size, height, width, self.num_pos_feats),
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.transpose(pos, (0, 3, 1, 2))
        return pos

    def encode_boxes(self, boxes):
        dim_t = ops.cast(ops.arange(self.num_pos_feats), dtype="float32")
        dim_t = self.temperature ** (2 * ops.floor(dim_t / 2) / self.num_pos_feats)

        boxes_scaled = boxes * self.scale
        pos = ops.expand_dims(boxes_scaled, axis=-1) / dim_t

        pos_sin = ops.sin(pos[..., 0::2])
        pos_cos = ops.cos(pos[..., 1::2])
        pos_embed = ops.reshape(
            ops.stack([pos_sin, pos_cos], axis=-1),
            ops.shape(pos)[:-1] + (self.num_pos_feats,),
        )
        shape = ops.shape(pos_embed)
        return ops.reshape(pos_embed, (shape[0], shape[1], shape[2] * shape[3]))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3MultiHeadAttention(layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout

    def build(self, input_shape):
        dim = self.hidden_size
        self.q_proj = layers.Dense(dim, name="q_proj")
        self.q_proj.build((None, None, dim))
        self.k_proj = layers.Dense(dim, name="k_proj")
        self.k_proj.build((None, None, dim))
        self.v_proj = layers.Dense(dim, name="v_proj")
        self.v_proj.build((None, None, dim))
        self.o_proj = layers.Dense(dim, name="o_proj")
        self.o_proj.build((None, None, dim))
        self.attn_dropout = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(self, query, key, value, attention_mask=None, training=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q_shape = ops.shape(q)
        k_shape = ops.shape(k)

        q = ops.reshape(
            q, (q_shape[0], q_shape[1], self.num_attention_heads, self.head_dim)
        )
        k = ops.reshape(
            k, (k_shape[0], k_shape[1], self.num_attention_heads, self.head_dim)
        )
        v = ops.reshape(
            v, (k_shape[0], k_shape[1], self.num_attention_heads, self.head_dim)
        )

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (q_shape[0], q_shape[1], self.hidden_size)
        )
        return self.o_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3DetrEncoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_size
        seq_shape = (None, None, dim)

        self.self_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="self_attn"
        )
        self.self_attn.build(seq_shape)
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.cross_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="cross_attn"
        )
        self.cross_attn.build(seq_shape)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.fc1 = layers.Dense(self.intermediate_size, name="fc1")
        self.fc1.build(seq_shape)
        self.fc2 = layers.Dense(dim, name="fc2")
        self.fc2.build((None, None, self.intermediate_size))
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm3"
        )
        self.layer_norm3.build(seq_shape)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(
        self,
        vision_feats,
        text_feats,
        vision_pos,
        text_mask=None,
        training=None,
    ):
        # Pre-norm self-attention: pos encoding on Q/K only, not V
        residual = vision_feats
        x = self.layer_norm1(vision_feats)
        q = k = x + vision_pos
        x = self.self_attn(q, k, x, training=training)
        vision_feats = self.dropout1(x, training=training) + residual

        # Pre-norm cross-attention: no pos encoding on query
        residual = vision_feats
        x = self.layer_norm2(vision_feats)
        x = self.cross_attn(
            x,
            text_feats,
            text_feats,
            attention_mask=text_mask,
            training=training,
        )
        vision_feats = self.dropout2(x, training=training) + residual

        # Pre-norm MLP
        residual = vision_feats
        x = self.layer_norm3(vision_feats)
        x = self.fc1(x)
        x = ops.nn.relu(x)
        x = self.dropout3(x, training=training)
        x = self.fc2(x)
        vision_feats = x + residual

        return vision_feats

    def compute_output_spec(
        self, vision_feats, text_feats, vision_pos, text_mask=None, training=None
    ):
        return keras.KerasTensor(shape=vision_feats.shape, dtype=vision_feats.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3DetrDecoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_size
        seq_shape = (None, None, dim)

        self.self_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="self_attn"
        )
        self.self_attn.build(seq_shape)
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.text_cross_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="text_cross_attn"
        )
        self.text_cross_attn.build(seq_shape)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.vision_cross_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="vision_cross_attn"
        )
        self.vision_cross_attn.build(seq_shape)
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm3"
        )
        self.layer_norm3.build(seq_shape)

        self.fc1 = layers.Dense(self.intermediate_size, name="fc1")
        self.fc1.build(seq_shape)
        self.fc2 = layers.Dense(dim, name="fc2")
        self.fc2.build((None, None, self.intermediate_size))
        self.layer_norm4 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm4"
        )
        self.layer_norm4.build(seq_shape)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.dropout4 = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(
        self,
        hidden_states,
        query_pos,
        text_feats,
        vision_feats,
        vision_pos,
        text_mask=None,
        vision_mask=None,
        training=None,
    ):
        # Post-norm self-attention: query_pos on Q and K, not V
        q = k = hidden_states + query_pos
        x = self.self_attn(q, k, hidden_states, training=training)
        x = self.dropout1(x, training=training)
        hidden_states = self.layer_norm1(hidden_states + x)

        # Post-norm text cross-attention: query_pos on Q
        x = self.text_cross_attn(
            hidden_states + query_pos,
            text_feats,
            text_feats,
            attention_mask=text_mask,
            training=training,
        )
        x = self.dropout2(x, training=training)
        hidden_states = self.layer_norm2(hidden_states + x)

        # Post-norm vision cross-attention: query_pos on Q, vision_pos on K
        x = self.vision_cross_attn(
            hidden_states + query_pos,
            vision_feats + vision_pos,
            vision_feats,
            attention_mask=vision_mask,
            training=training,
        )
        x = self.dropout3(x, training=training)
        hidden_states = self.layer_norm3(hidden_states + x)

        # Post-norm MLP
        x = self.fc1(hidden_states)
        x = ops.nn.relu(x)
        x = self.dropout4(x, training=training)
        x = self.fc2(x)
        hidden_states = self.layer_norm4(hidden_states + x)

        return hidden_states

    def compute_output_spec(
        self,
        hidden_states,
        query_pos,
        text_feats,
        vision_feats,
        vision_pos,
        text_mask=None,
        vision_mask=None,
        training=None,
    ):
        return keras.KerasTensor(shape=hidden_states.shape, dtype=hidden_states.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3GeometryEncoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_size
        seq_shape = (None, None, dim)

        self.self_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="self_attn"
        )
        self.self_attn.build(seq_shape)
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.cross_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="cross_attn"
        )
        self.cross_attn.build(seq_shape)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.fc1 = layers.Dense(self.intermediate_size, name="fc1")
        self.fc1.build(seq_shape)
        self.fc2 = layers.Dense(dim, name="fc2")
        self.fc2.build((None, None, self.intermediate_size))
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm3"
        )
        self.layer_norm3.build(seq_shape)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(
        self,
        prompt_feats,
        vision_feats,
        vision_pos,
        prompt_mask=None,
        training=None,
    ):
        x = self.self_attn(
            prompt_feats,
            prompt_feats,
            prompt_feats,
            attention_mask=prompt_mask,
            training=training,
        )
        x = self.dropout1(x, training=training)
        prompt_feats = self.layer_norm1(prompt_feats + x)

        k = vision_feats + vision_pos
        x = self.cross_attn(
            prompt_feats,
            k,
            vision_feats,
            training=training,
        )
        x = self.dropout2(x, training=training)
        prompt_feats = self.layer_norm2(prompt_feats + x)

        x = self.fc1(prompt_feats)
        x = ops.nn.relu(x)
        x = self.dropout3(x, training=training)
        x = self.fc2(x)
        prompt_feats = self.layer_norm3(prompt_feats + x)

        return prompt_feats

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
@keras.saving.register_keras_serializable(package="kmodels")
class SAM3DotProductScoring(layers.Layer):
    def __init__(
        self,
        hidden_size=256,
        intermediate_size=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def build(self, input_shape):
        dim = self.hidden_size
        seq_shape = (None, None, dim)

        self.text_mlp_fc1 = layers.Dense(self.intermediate_size, name="text_mlp_fc1")
        self.text_mlp_fc1.build(seq_shape)
        self.text_mlp_fc2 = layers.Dense(dim, name="text_mlp_fc2")
        self.text_mlp_fc2.build((None, None, self.intermediate_size))
        self.text_mlp_out_norm = layers.LayerNormalization(
            epsilon=1e-6, name="text_mlp_out_norm"
        )
        self.text_mlp_out_norm.build(seq_shape)

        self.text_proj = layers.Dense(dim, name="text_proj")
        self.text_proj.build((None, dim))
        self.query_proj = layers.Dense(dim, name="query_proj")
        self.query_proj.build(seq_shape)

        self._scale = dim**-0.5
        self.built = True

    def call(self, decoder_hidden_states, text_features, text_mask=None):
        x = self.text_mlp_fc1(text_features)
        x = ops.nn.relu(x)
        x = self.text_mlp_fc2(x)
        text_feats = self.text_mlp_out_norm(text_features + x)

        if text_mask is not None:
            mask_expanded = ops.expand_dims(ops.cast(text_mask, "float32"), axis=-1)
            text_pooled = ops.sum(text_feats * mask_expanded, axis=1) / (
                ops.sum(mask_expanded, axis=1) + 1e-8
            )
        else:
            text_pooled = ops.mean(text_feats, axis=1)

        text_proj = self.text_proj(text_pooled)  # (B, D)
        query_proj = self.query_proj(decoder_hidden_states)  # (B, Q, D)

        # Dot product (no L2 normalization — matching HF)
        logits = ops.matmul(query_proj, ops.expand_dims(text_proj, axis=-1))
        logits = ops.squeeze(logits, axis=-1) * self._scale
        logits = ops.clip(logits, -12.0, 12.0)
        return logits

    def compute_output_spec(self, decoder_hidden_states, text_features, text_mask=None):
        batch = decoder_hidden_states.shape[0]
        num_queries = decoder_hidden_states.shape[1]
        return keras.KerasTensor(
            shape=(batch, num_queries), dtype=decoder_hidden_states.dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3DecoderMLP(layers.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

    def build(self, input_shape):
        dims = (
            [self.input_dim]
            + [self.hidden_dim] * (self.num_layers - 1)
            + [self.output_dim]
        )
        self.dense_layers = []
        for i in range(self.num_layers):
            dense = layers.Dense(dims[i + 1], name=f"dense_{i}")
            dense.build((None, dims[i]) if i == 0 else (None, None, dims[i]))
            self.dense_layers.append(dense)
        self.built = True

    def call(self, x):
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = ops.nn.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3BoxRPB(layers.Layer):
    """Computes box-conditioned relative position bias for vision cross-attention."""

    def __init__(
        self, hidden_size, num_attention_heads, spatial_h, spatial_w, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

    def build(self, input_shape):
        self.box_rpb_embed_x = SAM3DecoderMLP(
            2,
            self.hidden_size,
            self.num_attention_heads,
            num_layers=2,
            name="box_rpb_embed_x",
        )
        self.box_rpb_embed_x.build((None, None, None, 2))
        self.box_rpb_embed_y = SAM3DecoderMLP(
            2,
            self.hidden_size,
            self.num_attention_heads,
            num_layers=2,
            name="box_rpb_embed_y",
        )
        self.box_rpb_embed_y.build((None, None, None, 2))
        self.built = True

    def call(self, reference_boxes):
        # reference_boxes: (B, Q, 4) in cxcywh format
        cx = reference_boxes[..., 0:1]
        cy = reference_boxes[..., 1:2]
        w = reference_boxes[..., 2:3]
        h = reference_boxes[..., 3:4]
        x0 = cx - 0.5 * w
        y0 = cy - 0.5 * h
        x1 = cx + 0.5 * w
        y1 = cy + 0.5 * h

        coords_w = ops.cast(ops.arange(self.spatial_w), "float32") / self.spatial_w
        coords_h = ops.cast(ops.arange(self.spatial_h), "float32") / self.spatial_h

        coords_w = ops.reshape(coords_w, (1, 1, self.spatial_w, 1))
        coords_h = ops.reshape(coords_h, (1, 1, self.spatial_h, 1))
        x0 = ops.expand_dims(x0, 2)  # (B, Q, 1, 1)
        x1 = ops.expand_dims(x1, 2)
        y0 = ops.expand_dims(y0, 2)
        y1 = ops.expand_dims(y1, 2)

        deltas_x = ops.concatenate([coords_w - x0, coords_w - x1], axis=-1)
        deltas_y = ops.concatenate([coords_h - y0, coords_h - y1], axis=-1)

        log2_8 = ops.cast(math.log2(8.0), "float32")
        deltas_x = ops.sign(deltas_x) * ops.log2(ops.abs(deltas_x * 8.0) + 1.0) / log2_8
        deltas_y = ops.sign(deltas_y) * ops.log2(ops.abs(deltas_y * 8.0) + 1.0) / log2_8

        rpb_x = self.box_rpb_embed_x(deltas_x)  # (B, Q, W, heads)
        rpb_y = self.box_rpb_embed_y(deltas_y)  # (B, Q, H, heads)

        # Outer sum: (B, Q, H, W, heads)
        rpb = ops.expand_dims(rpb_y, 3) + ops.expand_dims(rpb_x, 2)

        # Reshape (B, Q, H, W, heads) -> (B, Q, H*W, heads) -> (B, heads, Q, H*W)
        # Split H dim and concat to avoid dynamic reshape on batch
        hw_slices = [rpb[:, :, hi, :, :] for hi in range(self.spatial_h)]
        rpb_flat = ops.concatenate(hw_slices, axis=2)  # (B, Q, H*W, heads)
        rpb_out = ops.transpose(rpb_flat, (0, 3, 1, 2))  # (B, heads, Q, H*W)

        # Pad for presence token: prepend zero row on Q dim
        zeros = ops.zeros_like(rpb_out[:, :, :1, :])
        rpb_out = ops.concatenate([zeros, rpb_out], axis=2)  # (B, heads, Q+1, H*W)
        return rpb_out

    def compute_output_spec(self, reference_boxes):
        batch = reference_boxes.shape[0]
        nq = reference_boxes.shape[1]
        q1 = nq + 1 if nq is not None else None
        hw = self.spatial_h * self.spatial_w
        return keras.KerasTensor(
            shape=(batch, self.num_attention_heads, q1, hw),
            dtype=reference_boxes.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "spatial_h": self.spatial_h,
                "spatial_w": self.spatial_w,
            }
        )
        return config


# ═══════════════════════════════════════════════════════════════════
#  Geometry Encoder (box prompt encoding)
# ═══════════════════════════════════════════════════════════════════


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3GeometryEncoder(layers.Layer):
    """Encodes bounding box prompts into features for the DETR decoder.

    Combines direct box projection + sine position encoding + label embeddings,
    refined through transformer layers with cross-attention to vision features.
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        roi_size=7,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers_val = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout
        self.roi_size = roi_size
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_size
        self.boxes_direct_project = layers.Dense(dim, name="boxes_direct_project")
        self.boxes_direct_project.build((None, None, 4))
        self.boxes_pos_enc_project = layers.Dense(dim, name="boxes_pos_enc_project")
        self.boxes_pos_enc_project.build((None, None, dim + 2))
        self.boxes_pool_project = layers.Conv2D(
            dim,
            kernel_size=self.roi_size,
            padding="valid",
            name="boxes_pool_project",
        )
        self.boxes_pool_project.build((None, self.roi_size, self.roi_size, dim))
        self.label_embed = layers.Embedding(2, dim, name="label_embed")
        self.label_embed.build((None,))
        self.cls_embed = layers.Embedding(1, dim, name="cls_embed")
        self.cls_embed.build((None,))
        self.vision_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="vision_layer_norm"
        )
        self.vision_layer_norm.build((None, None, dim))
        self.prompt_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="prompt_layer_norm"
        )
        self.prompt_layer_norm.build((None, None, dim))
        self.output_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="output_layer_norm"
        )
        self.output_layer_norm.build((None, None, dim))
        self.final_proj = layers.Dense(dim, name="final_proj")
        self.final_proj.build((None, None, dim))
        self.transformer_layers = []
        for i in range(self.num_layers_val):
            layer = SAM3GeometryEncoderLayer(
                hidden_size=dim,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                dropout=self.dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
                name=f"layers_{i}",
            )
            layer.build((None, None, dim))
            self.transformer_layers.append(layer)
        self.built = True

    def call(self, boxes, box_labels, box_mask, vision_features_flat, vision_pos_flat):
        """
        Args:
            boxes: (B, num_boxes, 4) cxcywh normalized
            box_labels: (B, num_boxes) int
            box_mask: (B, num_boxes) float, 1=valid 0=padding
            vision_features_flat: (B, H*W, C)
            vision_pos_flat: (B, H*W, C)
        Returns:
            prompt_features: (B, num_boxes+1, C)
            prompt_mask: (B, num_boxes+1) float
        """
        batch_size = ops.shape(boxes)[0]

        # Direct projection
        direct = self.boxes_direct_project(boxes)

        # Sine position encoding of center + h, w
        cx = boxes[..., 0:1]
        cy = boxes[..., 1:2]
        w_box = boxes[..., 2:3]
        h_box = boxes[..., 3:4]
        # Simple sine encoding of center coordinates
        num_feats = self.hidden_size // 2
        dim_t = np.arange(num_feats, dtype="float32")
        dim_t = 10000.0 ** (2.0 * np.floor(dim_t / 2) / num_feats)
        dim_t = ops.convert_to_tensor(dim_t)
        center = ops.concatenate([cx, cy], axis=-1)  # (B, nb, 2)
        pos = ops.expand_dims(center, -1) / dim_t  # (B, nb, 2, num_feats)
        pos_sin = ops.sin(pos[..., 0::2])
        pos_cos = ops.cos(pos[..., 1::2])
        # Interleave
        half = pos_sin.shape[-1] if hasattr(pos_sin, "shape") else num_feats // 2
        parts = []
        for c in range(2):
            for j in range(half):
                parts.append(pos_sin[:, :, c, j : j + 1])
                parts.append(pos_cos[:, :, c, j : j + 1])
        center_sine = ops.concatenate(parts, axis=-1)  # (B, nb, hidden_size)

        pos_with_hw = ops.concatenate([center_sine, h_box, w_box], axis=-1)
        pos_encoded = self.boxes_pos_enc_project(pos_with_hw)

        # Label embedding
        label_emb = self.label_embed(box_labels)

        prompt_embeds = direct + pos_encoded + label_emb

        # Add CLS token
        cls_token = self.cls_embed(ops.zeros((batch_size, 1), dtype="int32"))
        prompt_embeds = ops.concatenate([cls_token, prompt_embeds], axis=1)

        cls_mask = ops.ones((batch_size, 1), dtype="float32")
        prompt_mask_out = ops.concatenate([cls_mask, box_mask], axis=1)

        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))

        for layer in self.transformer_layers:
            prompt_embeds = layer(prompt_embeds, vision_features_flat, vision_pos_flat)

        prompt_embeds = self.output_layer_norm(prompt_embeds)
        return prompt_embeds, prompt_mask_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers_val,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout_rate,
                "roi_size": self.roi_size,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config
