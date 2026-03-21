import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class DETRExpandQueryEmbedding(layers.Layer):
    """Expands query embeddings to match the batch dimension of the input.

    Takes learned query embeddings of shape ``(num_queries, hidden_dim)``
    and tiles them along a new batch axis to produce
    ``(batch_size, num_queries, hidden_dim)``.

    Args:
        num_queries: Number of object queries.
        hidden_dim: Embedding dimension.
    """

    def __init__(self, num_queries, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.embedding = layers.Embedding(
            num_queries,
            hidden_dim,
            name="embedding",
        )

    def call(self, batch_ref):
        batch_size = ops.shape(batch_ref)[0]
        indices = ops.arange(self.num_queries)
        query_embed = self.embedding(indices)
        query_embed = ops.expand_dims(query_embed, axis=0)
        query_embed = ops.tile(query_embed, [batch_size, 1, 1])
        return query_embed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_queries": self.num_queries,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DETRFlattenFeatures(layers.Layer):
    """Flattens spatial feature maps for transformer input.

    Reshapes ``(B, H, W, C)`` to ``(B, H*W, C)``.

    Args:
        hidden_dim: Channel dimension (used for the reshape target).
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, inputs):
        shape = ops.shape(inputs)
        return ops.reshape(inputs, [shape[0], shape[1] * shape[2], self.hidden_dim])

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DETRPositionEmbeddingSine(layers.Layer):
    """Fixed sinusoidal 2D position embedding used in the DETR encoder.

    Generates sine/cosine positional encodings for spatial feature maps,
    matching the original DETR implementation (facebook/detr).

    Args:
        hidden_dim: Total embedding dimension. Half is used for row
            embeddings, half for column embeddings.
        temperature: Temperature scaling for the sinusoidal frequencies.
        normalize: Whether to normalize position coordinates to [0, 2*pi].
        eps: Small epsilon to avoid division by zero during normalization.
    """

    def __init__(
        self,
        hidden_dim=256,
        temperature=10000,
        normalize=True,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        self.num_pos_feats = hidden_dim // 2

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]

        y_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, h + 1, dtype="float32"), axis=1),
                w,
                axis=1,
            ),
            dtype="float32",
        )
        x_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, w + 1, dtype="float32"), axis=0),
                h,
                axis=0,
            ),
            dtype="float32",
        )

        if self.normalize:
            y_embed = y_embed / (y_embed[-1:, :] + self.eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, -1:] + self.eps) * 2 * math.pi

        dim_t = ops.arange(self.num_pos_feats, dtype="float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        pos_x_sin = ops.sin(pos_x[:, :, 0::2])
        pos_x_cos = ops.cos(pos_x[:, :, 1::2])
        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos_y_sin = ops.sin(pos_y[:, :, 0::2])
        pos_y_cos = ops.cos(pos_y[:, :, 1::2])
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.expand_dims(pos, axis=0)
        pos = ops.broadcast_to(pos, [batch_size, h, w, self.hidden_dim])

        return pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DETRMultiHeadAttention(layers.Layer):
    """Multi-head attention layer for DETR transformer.

    Implements standard scaled dot-product multi-head attention with
    separate Q, K, V projections matching the HuggingFace DETR layout.

    Args:
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for attention weights.
        block_prefix: Name prefix for sub-layers.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        block_prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout_rate
        self.block_prefix = block_prefix

        self.q_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_q_proj",
        )
        self.k_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_k_proj",
        )
        self.v_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_v_proj",
        )
        self.out_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_out_proj",
        )
        self.attn_dropout = layers.Dropout(dropout_rate)

    def call(self, query, key, value, training=None):
        batch_size = ops.shape(query)[0]
        seq_len_q = ops.shape(query)[1]
        seq_len_k = ops.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = ops.reshape(q, [batch_size, seq_len_q, self.num_heads, self.head_dim])
        k = ops.reshape(k, [batch_size, seq_len_k, self.num_heads, self.head_dim])
        v = ops.reshape(v, [batch_size, seq_len_k, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [batch_size, seq_len_q, self.hidden_dim])
        attn_output = self.out_proj(attn_output)

        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config
