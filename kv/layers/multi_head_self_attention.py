from keras import InputSpec, layers, ops


class MultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention module implementation.

    This class implements multi-head self-attention where the input is split into multiple heads,
    each performing scaled dot-product attention independently. The results are then concatenated
    and projected back to the original dimension.

    Args:
        dim (int): Total dimension of the input and output features
        num_heads (int, optional): Number of attention heads. Default: 8
        qkv_bias (bool, optional): If True, adds learnable bias to query, key, value projections. Default: False
        qk_norm (bool, optional): If True, applies layer normalization to query and key. Default: False
        attn_drop (float, optional): Dropout rate for attention matrix. Default: 0.0
        proj_drop (float, optional): Dropout rate for output projection. Default: 0.0
        block_idx (int, optional): Index of the transformer block this attention belongs to. Default: 0
        **kwargs: Additional keyword arguments passed to the parent Layer class
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        block_idx: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_idx = block_idx
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = layers.Dense(
            dim * 3, use_bias=qkv_bias, name=f"blocks_{block_idx}_attn_qkv"
        )

        self.q_norm = (
            layers.LayerNormalization(name=f"blocks_{block_idx}_attn_layernorm_1")
            if qk_norm
            else None
        )
        self.k_norm = (
            layers.LayerNormalization(name=f"blocks_{block_idx}_attn_layernorm_2")
            if qk_norm
            else None
        )

        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, name=f"blocks_{block_idx}_attn_proj")
        self.proj_drop = layers.Dropout(proj_drop)

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))

        if self.input_spec.ndim not in (3, 4):
            raise ValueError(
                f"MultiHeadSelfAttention expects 3D or 4D input tensor, but received shape: {input_shape}"
            )

        feature_dim = input_shape[-1]
        if feature_dim != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.dim}"
            )

        batch_dim = input_shape[0]
        seq_length = input_shape[1]

        qkv_shape = input_shape
        attention_shape = (batch_dim, self.num_heads, seq_length, seq_length)
        head_shape = (batch_dim, self.num_heads, seq_length, self.head_dim)

        self.qkv.build(qkv_shape)
        self.proj.build(input_shape)

        if self.q_norm is not None:
            self.q_norm.build(head_shape)
        if self.k_norm is not None:
            self.k_norm.build(head_shape)

        self.attn_drop.build(attention_shape)
        self.proj_drop.build(input_shape)

        self._attention_head_size = self.head_dim
        self._num_attention_heads = self.num_heads

        self.built = True

    def call(self, x, training=None):
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (0, 3, 2, 1, 4))
        q, k, v = ops.unstack(qkv, 3, axis=2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q * self.scale
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv.use_bias,
                "qk_norm": self.q_norm,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "block_idx": self.block_idx,
            }
        )
        return config
