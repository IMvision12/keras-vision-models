import keras
from keras import layers, ops

from .window_partition import WindowPartition
from .window_reverse import WindowReverse


@keras.saving.register_keras_serializable(package="kvmm")
class WindowAttention(layers.Layer):
    """Window-based Multi-Head Self-Attention layer for transformers.

    This layer implements window-based self-attention where the input is divided into
    windows, and attention is computed within each window. It includes relative
    positional embeddings and optional attention masking, making it particularly
    suitable for vision transformer architectures like Swin Transformer.

    Key Features:
        - Window-based partitioning for efficient computation on images
        - Relative positional embeddings for capturing spatial relationships
        - Support for optional attention masking between windows
        - Independent parallel attention heads for capturing different relationship patterns
        - Scaled dot-product attention with configurable scaling factor
        - Configurable attention and projection dropout

    Args:
        dim (int): Total dimension of the input and output features. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int): Number of parallel attention heads. Each head operates
            on dim/num_heads features
        window_size (int): Size of the window for windowed attention (W x W)
        bias_table_window_size (int): Size of the relative position bias table for window-based
            attention. Determines the range of relative positions that can be represented
            in the bias table.
        qkv_bias (bool, optional): If True, adds learnable bias terms to the query, key,
            and value projections. Defaults to True
        qk_scale (float, optional): Scaling factor for the query-key dot product.
            If None, uses head_dim ** -0.5. Defaults to None
        attn_drop (float, optional): Dropout rate applied to attention weights.
            Helps prevent overfitting. Defaults to 0.0
        proj_drop (float, optional): Dropout rate applied to the output projection.
            Provides additional regularization. Defaults to 0.0
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - List of 4 tensors:
            - 4D input tensor: (batch_size, height, width, feature_dim)
            - 0D window size tensor: () containing window size as a scalar
            - 1D relative index tensor: (window_size² x window_size²) containing relative position indices
            - 5D attention mask tensor: (num_windows, 1, num_heads, window_size², window_size²)

    Output shape:
        - 4D tensor: (batch_size, height, width, feature_dim), same as input[0]

    Notes:
        - Primarily designed for vision transformers with 2D spatial data
        - Implements relative positional embeddings for better spatial awareness
        - Can handle shifted window attention with appropriate masking
        - Suitable for hierarchical vision transformer architectures
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        bias_table_window_size: int,
        qkv_bias: bool = True,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        block_prefix: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.bias_table_window_size = int(bias_table_window_size)
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv_bias = qkv_bias

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=prefix + "attn_qkv",
        )

        self.window_partition = WindowPartition(
            window_size=self.window_size, fused=True, num_heads=num_heads, qkv_mult=3
        )
        self.window_reverse = WindowReverse(
            window_size=self.window_size, fused=True, num_heads=num_heads
        )

        self.drop_attn = layers.Dropout(
            attn_drop,
            dtype=self.dtype_policy,
        )

        self.proj = layers.Dense(
            dim, dtype=self.dtype_policy, name=prefix + "attn_proj"
        )

        self.drop_proj = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        if feature_dim is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. Found `None`."
            )

        if feature_dim != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.dim}"
            )

        self.qkv.build(input_shape[0])
        self.proj.build(
            (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.dim)
        )

        prefix = f"{self.block_prefix}_"
        self.relative_bias = self.add_weight(
            name=prefix + "attn_relative_position_bias_table",
            shape=[(2 * self.bias_table_window_size - 1) ** 2, self.num_heads],
            trainable=True,
            dtype=self.dtype,
        )

        self.built = True

    def with_mask(self, attn, mask, length):
        mask_windows = ops.shape(mask)[1]
        attn = ops.reshape(attn, [-1, mask_windows, self.num_heads, length, length])
        attn = attn + mask
        attn = ops.reshape(attn, [-1, self.num_heads, length, length])
        return attn

    def call(self, inputs, training=None):
        inputs, window_size, relative_index, attention_mask = inputs
        height, width = ops.shape(inputs)[1:3]
        length = window_size**2

        qkv = self.qkv(inputs)
        qkv = self.window_partition(qkv, height=height, width=width)

        q, k, v = ops.unstack(qkv, 3)

        q = q * self.scale
        k = ops.swapaxes(k, -2, -1)
        attn = ops.matmul(q, k)

        bias = ops.take(self.relative_bias, relative_index, axis=0)
        bias = ops.reshape(bias, [length, length, -1])
        bias = ops.transpose(bias, [2, 0, 1])
        attn = attn + bias[None]

        if attention_mask is not None:
            attn = self.with_mask(attn, attention_mask, length)

        attn = ops.softmax(attn)
        attn = self.drop_attn(attn, training=training)

        x = ops.matmul(attn, v)
        x = self.window_reverse(x, height=height, width=width)

        x = self.proj(x)
        x = self.drop_proj(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "bias_table_window_size": self.bias_table_window_size,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.scale if self.scale != self.head_dim**-0.5 else None,
                "attn_drop": self.attn_drop_rate,
                "proj_drop": self.proj_drop_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config
