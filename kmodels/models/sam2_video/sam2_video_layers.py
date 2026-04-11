import keras
import numpy as np
from keras import layers, ops


def build_rope_2d_axial_cache(end_x, end_y, dim, theta=10000.0):
    """
    Precompute 2D axial RoPE cos/sin tables for memory attention.

    Uses pairwise rotation: frequencies are repeat-interleaved by 2 so
    each consecutive pair of dimensions shares the same frequency.

    Args:
        end_x: Integer, spatial width of the grid.
        end_y: Integer, spatial height of the grid.
        dim: Integer, total embedding dimension (must be divisible by 4).
        theta: Float, RoPE frequency base. Defaults to ``10000.0``.

    Returns:
        Tuple ``(cos, sin)``, each a NumPy array of shape
        ``(end_x * end_y, dim)``.
    """
    i = np.arange(0, dim, 4, dtype=np.float32)[: dim // 4]
    freqs = 1.0 / (theta ** (i / dim))

    flat = np.arange(end_x * end_y, dtype=np.float32)
    x_pos = flat % end_x
    y_pos = flat // end_x

    freqs_x = np.outer(x_pos, freqs)
    freqs_y = np.outer(y_pos, freqs)

    inv_freq = np.concatenate([freqs_x, freqs_y], axis=-1)
    inv_freq = np.repeat(inv_freq, 2, axis=-1)

    return np.cos(inv_freq).astype(np.float32), np.sin(inv_freq).astype(np.float32)


def _rotate_pairwise(x):
    shape = ops.shape(x)
    x = ops.reshape(x, [*shape[:-1], -1, 2])
    x1 = x[..., 0]
    x2 = x[..., 1]
    x = ops.stack([-x2, x1], axis=-1)
    return ops.reshape(x, shape)


def _apply_rope_2d(q, k, cos, sin, num_k_exclude_rope=0, rope_k_repeat=False):
    if num_k_exclude_rope > 0:
        k_rot = k[..., : ops.shape(k)[-2] - num_k_exclude_rope, :]
        k_pass = k[..., ops.shape(k)[-2] - num_k_exclude_rope :, :]
    else:
        k_rot = k
        k_pass = None

    q_embed = q * cos + _rotate_pairwise(q) * sin

    if rope_k_repeat and k_rot.shape[-2] != q.shape[-2]:
        repeat_factor = k_rot.shape[-2] // q.shape[-2]
        cos_k = ops.repeat(cos, repeat_factor, axis=2)
        sin_k = ops.repeat(sin, repeat_factor, axis=2)
    else:
        cos_k = cos
        sin_k = sin

    k_embed = k_rot * cos_k + _rotate_pairwise(k_rot) * sin_k

    if k_pass is not None:
        k_embed = ops.concatenate([k_embed, k_pass], axis=-2)

    return q_embed, k_embed


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2VideoRoPEAttention(layers.Layer):
    """
    Multi-head attention with 2D axial RoPE for memory attention.

    Uses separate Q, K, V projections with optional different key/value
    input dimension (for cross-attention with lower-dimensional memory).

    Args:
        hidden_size: Integer, query/output dimension. Defaults to ``256``.
        kv_in_dim: Integer, key/value input dimension. If None, same as
            hidden_size. Defaults to ``None``.
        num_heads: Integer, number of attention heads. Defaults to ``1``.
        rope_k_repeat: Boolean, whether to repeat RoPE frequencies for keys
            when key sequence is longer than query sequence (cross-attention).
            Defaults to ``False``.
        dropout_p: Float, attention dropout rate. Defaults to ``0.1``.
        **kwargs: Additional keyword arguments passed to the base ``Layer``.
    """

    def __init__(
        self,
        hidden_size=256,
        kv_in_dim=None,
        num_heads=1,
        rope_k_repeat=False,
        dropout_p=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kv_in_dim = kv_in_dim or hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.rope_k_repeat = rope_k_repeat
        self.dropout_p = dropout_p

        self.q_proj = layers.Dense(hidden_size, name="q_proj")
        self.k_proj = layers.Dense(hidden_size, name="k_proj")
        self.v_proj = layers.Dense(hidden_size, name="v_proj")
        self.o_proj = layers.Dense(hidden_size, name="o_proj")
        self.attn_drop = layers.Dropout(dropout_p)

    def call(
        self, query, key, value, rope_cos_sin, num_k_exclude_rope=0, training=None
    ):
        batch = ops.shape(query)[0]
        point_batch = ops.shape(query)[1]
        B = batch * point_batch

        q = ops.reshape(self.q_proj(query), [B, -1, self.num_heads, self.head_dim])
        k = ops.reshape(self.k_proj(key), [B, -1, self.num_heads, self.head_dim])
        v = ops.reshape(self.v_proj(value), [B, -1, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        cos, sin = rope_cos_sin
        cos = ops.expand_dims(ops.expand_dims(cos, 0), 0)
        sin = ops.expand_dims(ops.expand_dims(sin, 0), 0)

        q, k = _apply_rope_2d(
            q,
            k,
            cos,
            sin,
            num_k_exclude_rope=num_k_exclude_rope,
            rope_k_repeat=self.rope_k_repeat,
        )

        q = q * self.scale
        attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
        attn = ops.softmax(attn)
        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch, point_batch, -1, self.hidden_size])

        return self.o_proj(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "kv_in_dim": self.kv_in_dim,
                "num_heads": self.num_heads,
                "rope_k_repeat": self.rope_k_repeat,
                "dropout_p": self.dropout_p,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2VideoMemoryAttentionLayer(layers.Layer):
    """
    Single memory attention layer with self-attention, cross-attention, and FFN.

    Pre-norm architecture: LayerNorm before each sub-layer, with residual
    connections. Self-attention uses RoPE on current frame features.
    Cross-attention attends to memory features (64-dim) with RoPE.

    Args:
        hidden_size: Integer, hidden dimension. Defaults to ``256``.
        kv_in_dim: Integer, memory key/value dimension. Defaults to ``64``.
        num_heads: Integer, number of attention heads. Defaults to ``1``.
        ffn_hidden_size: Integer, FFN hidden dimension. Defaults to ``2048``.
        dropout: Float, dropout rate. Defaults to ``0.1``.
        rope_dropout: Float, RoPE attention dropout. Defaults to ``0.1``.
        **kwargs: Additional keyword arguments passed to the base ``Layer``.
    """

    def __init__(
        self,
        hidden_size=256,
        kv_in_dim=64,
        num_heads=1,
        ffn_hidden_size=2048,
        dropout=0.1,
        rope_dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kv_in_dim = kv_in_dim
        self.num_heads = num_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.dropout_rate = dropout
        self.rope_dropout = rope_dropout

        self.self_attn = Sam2VideoRoPEAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_p=rope_dropout,
            name="self_attn",
        )
        self.cross_attn_image = Sam2VideoRoPEAttention(
            hidden_size=hidden_size,
            kv_in_dim=kv_in_dim,
            num_heads=num_heads,
            rope_k_repeat=True,
            dropout_p=rope_dropout,
            name="cross_attn_image",
        )

        self.linear1 = layers.Dense(ffn_hidden_size, name="linear1")
        self.linear2 = layers.Dense(hidden_size, name="linear2")

        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm1")
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm2")
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm3")

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        self.ffn_dropout = layers.Dropout(dropout)

    def call(
        self,
        queries,
        keys,
        key_pos_embed,
        rope_cos_sin,
        num_k_exclude_rope=0,
        training=None,
    ):
        query = self.layer_norm1(queries)
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            rope_cos_sin=rope_cos_sin,
            training=training,
        )
        queries = queries + self.dropout1(query, training=training)

        query = self.layer_norm2(queries)
        query = self.cross_attn_image(
            query=query,
            key=keys + key_pos_embed,
            value=keys,
            rope_cos_sin=rope_cos_sin,
            num_k_exclude_rope=num_k_exclude_rope,
            training=training,
        )
        queries = queries + self.dropout2(query, training=training)

        query = self.layer_norm3(queries)
        query = self.linear2(
            self.ffn_dropout(ops.relu(self.linear1(query)), training=training)
        )
        queries = queries + self.dropout3(query, training=training)

        return queries

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "kv_in_dim": self.kv_in_dim,
                "num_heads": self.num_heads,
                "ffn_hidden_size": self.ffn_hidden_size,
                "dropout": self.dropout_rate,
                "rope_dropout": self.rope_dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2VideoMemoryAttention(layers.Layer):
    """
    Multi-layer memory attention with 2D axial RoPE.

    Stacks N ``Sam2VideoMemoryAttentionLayer`` blocks followed by a final
    LayerNorm. Position embeddings for queries are added with a 0.1 scale
    factor before the attention stack.

    Args:
        hidden_size: Integer, hidden dimension. Defaults to ``256``.
        kv_in_dim: Integer, memory dimension. Defaults to ``64``.
        num_layers: Integer, number of attention layers. Defaults to ``4``.
        num_heads: Integer, number of attention heads. Defaults to ``1``.
        ffn_hidden_size: Integer, FFN hidden dim. Defaults to ``2048``.
        dropout: Float, dropout rate. Defaults to ``0.1``.
        rope_theta: Float, RoPE frequency base. Defaults to ``10000.0``.
        rope_feat_sizes: List of 2 integers, spatial grid size for RoPE.
            Defaults to ``[64, 64]``.
        **kwargs: Additional keyword arguments passed to the base ``Layer``.
    """

    def __init__(
        self,
        hidden_size=256,
        kv_in_dim=64,
        num_layers=4,
        num_heads=1,
        ffn_hidden_size=2048,
        dropout=0.1,
        rope_theta=10000.0,
        rope_feat_sizes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kv_in_dim = kv_in_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.dropout_rate = dropout
        self.rope_theta = rope_theta
        self.rope_feat_sizes = rope_feat_sizes or [64, 64]

        self.attn_layers = []
        for i in range(num_layers):
            self.attn_layers.append(
                Sam2VideoMemoryAttentionLayer(
                    hidden_size=hidden_size,
                    kv_in_dim=kv_in_dim,
                    num_heads=num_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    dropout=dropout,
                    rope_dropout=dropout,
                    name=f"layers_{i}",
                )
            )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        end_x, end_y = self.rope_feat_sizes
        cos_np, sin_np = build_rope_2d_axial_cache(
            end_x, end_y, hidden_size, theta=rope_theta
        )
        self._rope_cos = cos_np
        self._rope_sin = sin_np

    def call(
        self,
        current_vision_feats,
        memory,
        current_vision_pos_embeds=None,
        memory_pos_embeds=None,
        num_object_pointer_tokens=0,
        training=None,
    ):
        output = current_vision_feats
        if current_vision_pos_embeds is not None:
            output = output + 0.1 * current_vision_pos_embeds

        output = ops.expand_dims(output, 1)
        memory = ops.expand_dims(memory, 1)
        if memory_pos_embeds is not None:
            memory_pos_embeds = ops.expand_dims(memory_pos_embeds, 1)
        else:
            memory_pos_embeds = ops.zeros_like(memory)

        rope_cos = ops.convert_to_tensor(self._rope_cos)
        rope_sin = ops.convert_to_tensor(self._rope_sin)
        rope_cos_sin = (rope_cos, rope_sin)

        for layer in self.attn_layers:
            output = layer(
                queries=output,
                keys=memory,
                key_pos_embed=memory_pos_embeds,
                rope_cos_sin=rope_cos_sin,
                num_k_exclude_rope=num_object_pointer_tokens,
                training=training,
            )

        output = self.layer_norm(output)
        output = ops.squeeze(output, 1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "kv_in_dim": self.kv_in_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ffn_hidden_size": self.ffn_hidden_size,
                "dropout": self.dropout_rate,
                "rope_theta": self.rope_theta,
                "rope_feat_sizes": self.rope_feat_sizes,
            }
        )
        return config
