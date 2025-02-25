import keras
from keras import InputSpec, layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class TalkingHeadAttention(layers.Layer):
    """Talking-Head Attention layer implementing enhanced attention mechanism.

    This layer implements the Talking-Head Attention mechanism described in the paper
    "Talking-Heads Attention" (https://arxiv.org/abs/2003.02436), which enhances
    multi-head self-attention by applying linear projections across attention heads.
    This allows for information exchange between attention heads, enabling more
    flexible attention patterns and potentially improved performance.

    Key Features:
        - Linear projections across attention heads for enhanced cross-head communication
        - Scaled dot-product attention with additional head-talking projections
        - Configurable attention and projection dropout
        - Optional bias terms in query/key/value projections
        - Returns both output tensor and attention weights for inspection and visualization
        - Support for both channels_last (NHWC) and channels_first (NCHW) formats

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
        data_format (str, optional): Data format, either 'channels_last' (default) or 'channels_first'.
            Determines the order of dimensions in the input tensor
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - If data_format='channels_last': 3D tensor: (batch_size, sequence_length, feature_dim)
        - If data_format='channels_first': 3D tensor: (batch_size, feature_dim, sequence_length)

    Output shape:
        - Output tensor: Same shape as input
        - Attention weights: (batch_size, num_heads, sequence_length, sequence_length)

    Notes:
        - The attention dimension (dim) must be divisible by num_heads
        - Talking-Head Attention extends standard multi-head attention with two additional
          linear projections: one before softmax (proj_l) and one after softmax (proj_w)
        - These projections allow each attention head to "talk" to other heads,
          enabling more expressive attention distributions
        - Suitable for sequence data in transformer-based architectures
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
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.block_prefix = block_prefix
        self.data_format = data_format

        assert data_format in ["channels_last", "channels_first"], (
            "data_format must be either 'channels_last' or 'channels_first'"
        )

        prefix = f"{block_prefix}_" if block_prefix else ""

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{prefix}qkv" if block_prefix else None,
        )
        self.proj = layers.Dense(
            dim, dtype=self.dtype_policy, name=f"{prefix}proj" if block_prefix else None
        )

        self.proj_l = layers.Dense(
            num_heads,
            dtype=self.dtype_policy,
            name=f"{prefix}proj_l" if block_prefix else None,
        )
        self.proj_w = layers.Dense(
            num_heads,
            dtype=self.dtype_policy,
            name=f"{prefix}proj_w" if block_prefix else None,
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
                f"TalkingHeadAttention expects 3D input tensor, but received shape: {input_shape}"
            )

        feature_dim_idx = 1 if self.data_format == "channels_first" else -1
        feature_dim = input_shape[feature_dim_idx]
        if feature_dim != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.dim}"
            )

        if self.data_format == "channels_last":
            self.qkv.build(input_shape)
            self.proj.build((input_shape[0], input_shape[1], self.dim))
            self.proj_l.build(
                (input_shape[0], input_shape[1], input_shape[1], self.num_heads)
            )
            self.proj_w.build(
                (input_shape[0], input_shape[1], input_shape[1], self.num_heads)
            )
        else:  # channels_first
            self.qkv.build((input_shape[0], input_shape[2], self.dim))
            self.proj.build((input_shape[0], input_shape[2], self.dim))
            self.proj_l.build(
                (input_shape[0], input_shape[2], input_shape[2], self.num_heads)
            )
            self.proj_w.build(
                (input_shape[0], input_shape[2], input_shape[2], self.num_heads)
            )

        self.built = True

    def call(self, x, training=False):
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 1))

        input_shape = ops.shape(x)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        qkv = self.qkv(x)
        qkv = ops.reshape(
            qkv, (batch_size, seq_length, 3, self.num_heads, self.head_dim)
        )
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))

        attn = ops.transpose(attn, (0, 2, 3, 1))
        attn = self.proj_l(attn)
        attn = ops.transpose(attn, (0, 3, 1, 2))

        attn = ops.softmax(attn, axis=-1)

        attn = ops.transpose(attn, (0, 2, 3, 1))
        attn = self.proj_w(attn)
        attn = ops.transpose(attn, (0, 3, 1, 2))

        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (batch_size, seq_length, self.dim))

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
                "qkv_bias": self.qkv.use_bias,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "data_format": self.data_format,
                "block_prefix": self.block_prefix,
            }
        )
        return config
