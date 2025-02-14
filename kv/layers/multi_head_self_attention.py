import keras
from keras import InputSpec, layers, ops

@keras.saving.register_keras_serializable(package="kv")
class MultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention layer implementing scaled dot-product attention.

    This layer implements the standard multi-head self-attention mechanism where input is split
    into multiple attention heads operating in parallel. Each head performs scaled dot-product
    attention independently, after which results are concatenated and projected back to the
    original dimension. This implementation is particularly suitable for sequence processing
    and transformer-based architectures.

    Key Features:
        - Independent parallel attention heads for capturing different relationship patterns
        - Scaled dot-product attention with optional layer normalization
        - Configurable attention and projection dropout
        - Optional bias terms in query/key/value projections
        - Support for both 3D and 4D input tensors

    Args:
        dim (int): Total dimension of the input and output features. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int, optional): Number of parallel attention heads. Each head operates
            on dim/num_heads features. Defaults to 8
        qkv_bias (bool, optional): If True, adds learnable bias terms to the query, key,
            and value projections. Defaults to False
        qk_norm (bool, optional): If True, applies layer normalization to query and key
            tensors before attention computation. Defaults to False
        attn_drop (float, optional): Dropout rate applied to attention weights. Helps
            prevent overfitting. Defaults to 0.0
        proj_drop (float, optional): Dropout rate applied to the output projection.
            Provides additional regularization. Defaults to 0.0
        epsilon (float, optional): Small constant used in normalization operations for
            numerical stability. A higher value reduces precision but increases stability.
            Defaults to 1e-6
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - 3D tensor: (batch_size, sequence_length, feature_dim)
        - 4D tensor: (batch_size, height, width, feature_dim)

    Output shape:
        - Same as input shape

    Notes:
        - The attention dimension (dim) must be divisible by num_heads
        - Layer normalization on query/key can help stabilize training
        - Suitable for both sequence data and vision transformers
        - Implements the standard scaled dot-product attention formula:
          Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        epsilon=1e-6,
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.epsilon = epsilon
        self.data_format = keras.config.image_data_format()
        self.channels_axis = -1 if self.data_format == "channels_last" else 1

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=prefix + "attn_qkv",
        )

        self.q_norm = (
            layers.LayerNormalization(
                axis=self.channels_axis,
                epsilon=self.epsilon,
                dtype=self.dtype_policy,
                name=prefix + "attn_norm1",
            )
            if qk_norm
            else None
        )
        self.k_norm = (
            layers.LayerNormalization(
                axis=self.channels_axis,
                epsilon=self.epsilon,
                dtype=self.dtype_policy,
                name=prefix + "attn_norm2",
            )
            if qk_norm
            else None
        )

        self.attn_drop = layers.Dropout(
            attn_drop,
            dtype=self.dtype_policy,
        )
        self.proj = layers.Dense(
            dim, dtype=self.dtype_policy, name=prefix + "attn_proj"
        )
        self.proj_drop = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

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

        if self.input_spec.ndim == 3:
            seq_length = input_shape[1]
            attention_shape = (batch_dim, self.num_heads, seq_length, seq_length)
            head_shape = (batch_dim, self.num_heads, seq_length, self.head_dim)
        else:  # 4D input
            height, width = input_shape[1], input_shape[2]
            attention_shape = (batch_dim, height, self.num_heads, width, width)
            head_shape = (batch_dim, height, self.num_heads, width, self.head_dim)

        self.qkv.build(input_shape)
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
        batch_size = ops.shape(x)[0]
        ndim = len(x.shape)

        qkv = self.qkv(x)

        if ndim == 3:
            seq_length = ops.shape(x)[1]
            qkv = ops.reshape(
                qkv, (batch_size, seq_length, 3, self.num_heads, self.head_dim)
            )
            qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        else:  # 4D input
            height, width = ops.shape(x)[1], ops.shape(x)[2]
            qkv = ops.reshape(
                qkv, (batch_size, height, width, 3, self.num_heads, self.head_dim)
            )
            qkv = ops.transpose(qkv, (3, 0, 4, 1, 2, 5))
            qkv = ops.reshape(
                qkv, (3, batch_size, self.num_heads, height * width, self.head_dim)
            )

        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q * self.scale

        attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)

        if ndim == 3:
            x = ops.transpose(x, (0, 2, 1, 3))
            x = ops.reshape(x, (batch_size, seq_length, self.dim))
        else:
            x = ops.transpose(x, (0, 2, 1, 3))
            x = ops.reshape(x, (batch_size, height, width, self.dim))

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
                "qk_norm": self.q_norm is not None,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "epsilon": self.epsilon,
                "block_prefix": self.block_prefix,
            }
        )
        return config
