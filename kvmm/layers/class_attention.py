import keras
from keras import InputSpec, layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class ClassAttention(layers.Layer):
    """Class Attention layer for transformer architectures.

    This layer implements a specialized attention mechanism where queries are generated
    only from the first token (class token) of the sequence, while keys and values are
    generated from the entire sequence. This approach is particularly useful in vision
    transformers and other architectures where a special class token aggregates information
    from the entire sequence.

    Key Features:
        - Queries are derived only from the class token (first token)
        - Keys and values are derived from the entire sequence
        - Supports multiple attention heads to capture different relationship patterns
        - Configurable attention and projection dropout for regularization
        - Optional bias terms in query/key/value projections
        - Supports both channels_last (NHWC) and channels_first (NCHW) formats

    Args:
        dim (int): Total dimension of the input and output features. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int): Number of parallel attention heads. Each head operates
            on dim/num_heads features
        qkv_bias (bool, optional): If True, adds learnable bias terms to the query, key,
            and value projections. Defaults to True
        attn_drop (float, optional): Dropout rate applied to attention weights. Helps
            prevent overfitting. Defaults to 0.0
        proj_drop (float, optional): Dropout rate applied to the output projection.
            Provides additional regularization. Defaults to 0.0
        data_format (str, optional): Format of the input tensor. Can be either
            'channels_last' (default) or 'channels_first'. For 'channels_last', the input
            shape is (batch_size, sequence_length, feature_dim), while for 'channels_first',
            the input shape is (batch_size, feature_dim, sequence_length).
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - If data_format='channels_last': 3D tensor (batch_size, sequence_length, feature_dim)
        - If data_format='channels_first': 3D tensor (batch_size, feature_dim, sequence_length)

    Output shape:
        - If data_format='channels_last': 3D tensor (batch_size, 1, feature_dim)
        - If data_format='channels_first': 3D tensor (batch_size, feature_dim, 1)

    Notes:
        - The attention dimension (dim) must be divisible by num_heads
        - Only returns attention results for the class token (first position)
        - Commonly used in Vision Transformer (ViT) architectures
        - Implements a modified scaled dot-product attention where queries come only
          from the class token position
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        data_format: str = "channels_last",
        block_prefix: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert data_format in ["channels_last", "channels_first"], (
            "data_format must be either 'channels_last' or 'channels_first'"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.data_format = data_format
        self.block_prefix = block_prefix

        prefix = f"{block_prefix}_" if block_prefix else ""

        self.q = layers.Dense(
            dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{prefix}q" if prefix else None,
        )
        self.k = layers.Dense(
            dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{prefix}k" if prefix else None,
        )
        self.v = layers.Dense(
            dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{prefix}v" if prefix else None,
        )
        self.proj = layers.Dense(
            dim, dtype=self.dtype_policy, name=f"{prefix}proj" if prefix else None
        )

        self.attn_drop = layers.Dropout(
            attn_drop,
            dtype=self.dtype_policy,
        )
        self.proj_drop = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))

        if self.input_spec.ndim != 3:
            raise ValueError(
                f"ClassAttention expects 3D input tensor, but received shape: {input_shape}"
            )

        if self.data_format == "channels_last":
            feature_dim = input_shape[-1]
        else:  # channels_first
            feature_dim = input_shape[1]

        if feature_dim != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.dim}"
            )

        if self.data_format == "channels_last":
            self.q.build((input_shape[0], 1, input_shape[-1]))
            self.k.build(input_shape)
            self.v.build(input_shape)
            self.proj.build((input_shape[0], 1, self.dim))
        else:
            self.q.build((input_shape[0], 1, input_shape[1]))
            self.k.build((input_shape[0], input_shape[2], input_shape[1]))
            self.v.build((input_shape[0], input_shape[2], input_shape[1]))
            self.proj.build((input_shape[0], 1, self.dim))

        self.built = True

    def call(self, x, training=None):
        B = ops.shape(x)[0]

        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 1))

        N = ops.shape(x)[1]

        q = self.q(x[:, 0:1])
        k = self.k(x)
        v = self.v(x)

        q = ops.reshape(q, (B, 1, self.num_heads, self.head_dim))
        k = ops.reshape(k, (B, N, self.num_heads, self.head_dim))
        v = ops.reshape(v, (B, N, self.num_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = ops.matmul(q * self.scale, ops.transpose(k, (0, 1, 3, 2)))
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, 1, self.dim))

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 1))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "qkv_bias": self.q.use_bias,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "data_format": self.data_format,
                "block_prefix": self.block_prefix,
            }
        )
        return config
