import keras
from keras import InputSpec, layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
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

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=prefix + "attn_qkv",
        )

        self.q_norm = (
            layers.LayerNormalization(
                epsilon=self.epsilon,
                dtype=self.dtype_policy,
                name=prefix + "attn_norm1",
            )
            if qk_norm
            else None
        )
        self.k_norm = (
            layers.LayerNormalization(
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

        self.qkv.build(input_shape)
        self.proj.build(input_shape)

        if self.q_norm is not None:
            norm_shape = (input_shape[-1],)
            self.q_norm.build(norm_shape)
        if self.k_norm is not None:
            norm_shape = (input_shape[-1],)
            self.k_norm.build(norm_shape)

        self.built = True

    def call(self, inputs, training=None):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        ndim = len(inputs.shape)

        qkv = self.qkv(inputs)

        qkv_split = ops.split(qkv, 3, axis=-1)
        q, k, v = qkv_split

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q * self.scale

        if ndim == 3:
            q = ops.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
            k = ops.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
            v = ops.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

            q = ops.transpose(q, [0, 2, 1, 3])
            k = ops.transpose(k, [0, 2, 1, 3])
            v = ops.transpose(v, [0, 2, 1, 3])

            attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
            attn = ops.softmax(attn)
            attn = self.attn_drop(attn, training=training)
            x = ops.matmul(attn, v)

            x = ops.transpose(x, [0, 2, 1, 3])
            x = ops.reshape(x, input_shape)
        else:
            q = ops.reshape(
                q,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )
            k = ops.reshape(
                k,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )
            v = ops.reshape(
                v,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )

            q = ops.transpose(q, [0, 1, 3, 2, 4])
            k = ops.transpose(k, [0, 1, 3, 2, 4])
            v = ops.transpose(v, [0, 1, 3, 2, 4])

            q = ops.reshape(q, [-1, self.num_heads, input_shape[2], self.head_dim])
            k = ops.reshape(k, [-1, self.num_heads, input_shape[2], self.head_dim])
            v = ops.reshape(v, [-1, self.num_heads, input_shape[2], self.head_dim])

            attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
            attn = ops.softmax(attn)
            attn = self.attn_drop(attn, training=training)
            x = ops.matmul(attn, v)

            x = ops.reshape(
                x,
                [
                    batch_size,
                    input_shape[1],
                    self.num_heads,
                    input_shape[2],
                    self.head_dim,
                ],
            )
            x = ops.transpose(x, [0, 1, 3, 2, 4])
            x = ops.reshape(x, input_shape)

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
