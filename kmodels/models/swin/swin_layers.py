import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class WindowPartition(layers.Layer):
    """Layer for partitioning input tensor into non-overlapping windows.

    This layer divides the input feature map into non-overlapping windows of specified size.
    It can operate in two modes: standard (fused=False) and fused attention mode (fused=True).
    In fused mode, it handles the partitioning while considering multi-headed attention
    requirements and QKV (Query, Key, Value) transformations.

    Args:
        window_size: int
            Size of each window (both height and width).
        fused: bool, optional
            If True, operates in fused attention mode. Default is False.
        num_heads: int, optional
            Number of attention heads. Required when fused=True.
        qkv_mult: int, optional
            Multiplier for QKV transformations. Default is 3 (Query + Key + Value).
        **kwargs: dict
            Additional keyword arguments passed to the parent Layer class.

    Raises:
        ValueError: If fused=True and num_heads is not provided.

    Example:
        ```python
        # Standard mode
        window_partition = WindowPartition(window_size=7)
        windowed_features = window_partition(features, height=28, width=28)

        # Fused attention mode
        window_partition = WindowPartition(window_size=7, fused=True, num_heads=4)
        qkv_windowed_features = window_partition(features, height=28, width=28)
        ```
    """

    def __init__(self, window_size, fused=False, num_heads=None, qkv_mult=3, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.fused = fused
        self.num_heads = num_heads
        self.qkv_mult = qkv_mult

        if self.fused and self.num_heads is None:
            raise ValueError("num_heads must be set when fused=True")

    def call(self, inputs, height=None, width=None):
        if len(inputs.shape) != 4:
            raise ValueError("Expecting inputs rank to be 4.")

        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        windows_height = height // self.window_size
        windows_width = width // self.window_size

        if not self.fused:
            channels = inputs.shape[-1]
            if channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            outputs = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    self.window_size,
                    windows_width,
                    self.window_size,
                    channels,
                ],
            )
            outputs = ops.transpose(outputs, [0, 1, 3, 2, 4, 5])
            outputs = ops.reshape(outputs, [-1, self.window_size**2, channels])

        else:
            full_channels = inputs.shape[-1]
            if full_channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            head_channels = full_channels // (self.qkv_mult * self.num_heads)

            outputs = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    self.window_size,
                    windows_width,
                    self.window_size,
                    self.qkv_mult,
                    self.num_heads,
                    head_channels,
                ],
            )
            outputs = ops.transpose(outputs, [5, 0, 1, 3, 6, 2, 4, 7])
            outputs = ops.reshape(
                outputs,
                [self.qkv_mult, -1, self.num_heads, self.window_size**2, head_channels],
            )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "fused": self.fused,
                "num_heads": self.num_heads,
                "qkv_mult": self.qkv_mult,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class WindowReverse(layers.Layer):
    """Layer for reverting windows back to the original feature map.

    This layer performs the inverse operation of window partitioning, reconstructing
    the original feature map from window segments. It can operate in two modes:
    standard (fused=False) and fused attention mode (fused=True).

    In standard mode, it takes window segments and reconstructs them into a full
    feature map. In fused mode, it handles multi-headed attention outputs by
    considering the number of attention heads when reconstructing the feature map.

    Args:
        window_size: int
            Size of each window (both height and width).
        fused: bool, optional
            If True, operates in fused attention mode. Default is False.
        num_heads: int, optional
            Number of attention heads. Required when fused=True.
        **kwargs: dict
            Additional keyword arguments passed to the parent Layer class.

    Raises:
        ValueError: If fused=True and num_heads is not provided.

    Example:
        ```python
        # Standard mode
        window_reverse = WindowReverse(window_size=7)
        output = window_reverse(windowed_features, height=28, width=28)

        # Fused attention mode
        window_reverse = WindowReverse(window_size=7, fused=True, num_heads=4)
        output = window_reverse(windowed_features, height=28, width=28)
        ```
    """

    def __init__(self, window_size, fused=False, num_heads=None, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.fused = fused
        self.num_heads = num_heads

        if self.fused and self.num_heads is None:
            raise ValueError("num_heads must be set when fused=True")

    def call(self, inputs, height=None, width=None):
        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        windows_height = height // self.window_size
        windows_width = width // self.window_size

        if not self.fused:
            if len(inputs.shape) != 3:
                raise ValueError("Expecting inputs rank to be 3.")

            channels = inputs.shape[-1]
            if channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            outputs = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    windows_width,
                    self.window_size,
                    self.window_size,
                    channels,
                ],
            )
            outputs = ops.transpose(outputs, [0, 1, 3, 2, 4, 5])
            outputs = ops.reshape(outputs, [-1, height, width, channels])

        else:
            if len(inputs.shape) != 4:
                raise ValueError("Expecting inputs rank to be 4.")

            head_channels = inputs.shape[-1]
            if head_channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            full_channels = head_channels * self.num_heads

            outputs = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    windows_width,
                    self.num_heads,
                    self.window_size,
                    self.window_size,
                    head_channels,
                ],
            )
            outputs = ops.transpose(outputs, [0, 1, 4, 2, 5, 3, 6])
            outputs = ops.reshape(outputs, [-1, height, width, full_channels])

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "fused": self.fused,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
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
            - 1D relative index tensor: (window_size^2 x window_size^2) containing relative position indices
            - 5D attention mask tensor: (num_windows, 1, num_heads, window_size^2, window_size^2)

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


class RollLayer(layers.Layer):
    """A layer that performs circular shifting of tensor elements along a specified axis.

    This layer shifts elements of the input tensor by a specified amount along
    the given axis. Elements that are shifted beyond the last position are
    re-introduced at the first position (circular/cyclic behavior).

    Args:
        shift: int or tuple of ints
            Number of positions to shift. If positive, shift to the right/down.
            If negative, shift to the left/up. If tuple, shifts by the specified
            amount for each corresponding axis.
        axis: int or tuple of ints
            Axis or axes along which to shift. If tuple, must have same length as shift.
        **kwargs: dict
            Additional keyword arguments passed to the parent Layer class.

    Example:
        ```python
        # Shift elements 2 positions to the right along axis 1
        roll_layer = RollLayer(shift=2, axis=1)
        output = roll_layer(input_tensor)

        # Shift elements in multiple axes
        roll_layer = RollLayer(shift=(1, -2), axis=(0, 1))
        output = roll_layer(input_tensor)
        ```

    Input Shape:
        Arbitrary. This layer can operate on tensors of any shape.

    Output Shape:
        Same as input shape.
    """

    def __init__(self, shift, axis, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift
        self.axis = axis

    def call(self, inputs):
        return ops.roll(inputs, shift=self.shift, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shift": self.shift,
                "axis": self.axis,
            }
        )
        return config
