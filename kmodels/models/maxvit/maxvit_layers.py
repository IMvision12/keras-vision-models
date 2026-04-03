import keras
from keras import layers, ops


def gelu_tanh(x):
    """GELU activation with tanh approximation, matching timm's GELUTanh."""
    return keras.activations.gelu(x, approximate=True)


@keras.saving.register_keras_serializable(package="kmodels")
class RelPosBiasTf(layers.Layer):
    """Relative position bias (TensorFlow-compatible, log-spaced) for MaxViT attention.

    Stores a learnable bias table of shape (num_heads, 2*window_size-1, 2*window_size-1)
    and uses one-hot lookup tensors to reindex into (num_heads, window_area, window_area).

    Args:
        window_size: Tuple of (height, width) for the attention window.
        num_heads: Number of attention heads.
        **kwargs: Additional keyword arguments passed to the Layer class.
    """

    def __init__(self, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        self.window_area = window_size[0] * window_size[1]
        self.vocab_height = 2 * window_size[0] - 1
        self.vocab_width = 2 * window_size[1] - 1

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(self.num_heads, self.vocab_height, self.vocab_width),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        self.built = True

    def _generate_lookup(self, length, max_rel_pos):
        """Generate one-hot lookup tensor for relative position reindexing."""
        vocab_size = 2 * max_rel_pos + 1
        indices = []
        for i in range(length):
            for x in range(length):
                v = x - i + max_rel_pos
                if abs(x - i) <= max_rel_pos:
                    indices.append([i, x, v])
        indices_array = ops.convert_to_tensor(indices, dtype="int32")
        ones = ops.ones((len(indices),))
        lookup = ops.scatter_update(
            ops.zeros((length, length, vocab_size)),
            indices_array,
            ones,
        )
        return lookup

    def call(self, attn):
        height_lookup = self._generate_lookup(
            self.window_size[0], self.window_size[0] - 1
        )
        width_lookup = self._generate_lookup(
            self.window_size[1], self.window_size[1] - 1
        )

        reindexed = ops.einsum(
            "nhw,ixh->nixw", self.relative_position_bias_table, height_lookup
        )
        reindexed = ops.einsum("nixw,jyw->nijxy", reindexed, width_lookup)
        bias = ops.reshape(
            reindexed, (self.num_heads, self.window_area, self.window_area)
        )
        return attn + bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class MaxViTAttention(layers.Layer):
    """Channels-last multi-head attention with relative position bias for MaxViT.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        window_size: Window size for relative position bias.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for projection output.
        **kwargs: Additional keyword arguments passed to the Layer class.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        attn_drop=0.0,
        proj_drop=0.0,
        prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head**-0.5
        self.window_size = (
            window_size
            if isinstance(window_size, tuple)
            else (window_size, window_size)
        )

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=True,
            name=prefix + "attn_qkv",
        )
        self.rel_pos = RelPosBiasTf(
            window_size=self.window_size,
            num_heads=num_heads,
            name=prefix + "attn_rel_pos",
        )
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(
            dim,
            use_bias=True,
            name=prefix + "attn_proj",
        )
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        B = input_shape[0]

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, [B, -1, 3, self.num_heads, self.dim_head])
        qkv = ops.transpose(qkv, [0, 3, 2, 1, 4])
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q = q * self.scale
        attn = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))
        attn = self.rel_pos(attn)
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, [0, 2, 1, 3])
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
                "window_size": self.window_size,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class MaxViTSEModule(layers.Layer):
    """Squeeze-and-Excitation module for MaxViT MBConv blocks.

    Args:
        in_channels: Number of input channels.
        se_ratio: Reduction ratio for the bottleneck. Defaults to 0.0625 (1/16).
        **kwargs: Additional keyword arguments passed to the Layer class.
    """

    def __init__(self, in_channels, se_ratio=0.0625, prefix="", **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.se_ratio = se_ratio
        self.reduced_channels = max(1, int(in_channels * se_ratio))

        self.fc1 = layers.Conv2D(
            self.reduced_channels,
            1,
            use_bias=True,
            data_format="channels_last",
            name=prefix + "se_fc1",
        )
        self.fc2 = layers.Conv2D(
            in_channels,
            1,
            use_bias=True,
            data_format="channels_last",
            name=prefix + "se_fc2",
        )

    def call(self, x, training=None):
        x_se = ops.mean(x, axis=(1, 2), keepdims=True)
        x_se = self.fc1(x_se)
        x_se = keras.activations.silu(x_se)
        x_se = self.fc2(x_se)
        x_se = keras.activations.sigmoid(x_se)
        return x * x_se

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "se_ratio": self.se_ratio,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class MaxViTMBConv(layers.Layer):
    """Mobile Inverted Bottleneck Convolution block for MaxViT.

    Consists of: pre_norm -> expand 1x1 -> norm+act -> depthwise 3x3 -> norm+act
    -> SE -> project 1x1 -> drop_path + residual.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        expand_ratio: Expansion ratio for the bottleneck. Defaults to 4.
        se_ratio: SE reduction ratio. Defaults to 0.0625.
        stride: Stride for the depthwise convolution. Defaults to 1.
        drop_path_rate: Drop path rate. Defaults to 0.0.
        **kwargs: Additional keyword arguments passed to the Layer class.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio=4,
        se_ratio=0.0625,
        stride=1,
        drop_path_rate=0.0,
        prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.stride = stride
        self.drop_path_rate = drop_path_rate

        expanded_channels = out_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        self.pre_norm = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-3,
            momentum=0.9,
            name=prefix + "conv_pre_norm",
        )

        self.conv1_1x1 = layers.Conv2D(
            expanded_channels,
            1,
            use_bias=False,
            data_format="channels_last",
            name=prefix + "conv_conv1_1x1",
        )
        self.norm1 = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-3,
            momentum=0.9,
            name=prefix + "conv_norm1",
        )

        self.conv2_kxk = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            data_format="channels_last",
            name=prefix + "conv_conv2_kxk",
        )
        self.norm2 = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-3,
            momentum=0.9,
            name=prefix + "conv_norm2",
        )

        self.se = MaxViTSEModule(
            expanded_channels,
            se_ratio=se_ratio,
            prefix=prefix,
        )

        self.conv3_1x1 = layers.Conv2D(
            out_channels,
            1,
            use_bias=True,
            data_format="channels_last",
            name=prefix + "conv_conv3_1x1",
        )

        self.has_shortcut_pool = stride > 1
        self.has_shortcut_expand = in_channels != out_channels
        if self.has_shortcut_pool:
            self.shortcut_pool = layers.AveragePooling2D(
                pool_size=2,
                strides=2,
                padding="same",
                data_format="channels_last",
                name=prefix + "conv_shortcut_pool",
            )
        if self.has_shortcut_expand:
            self.shortcut_expand = layers.Conv2D(
                out_channels,
                1,
                use_bias=True,
                data_format="channels_last",
                name=prefix + "conv_shortcut_expand",
            )

    def call(self, x, training=None):
        shortcut = x
        if self.has_shortcut_pool:
            shortcut = self.shortcut_pool(shortcut)
        if self.has_shortcut_expand:
            shortcut = self.shortcut_expand(shortcut)

        x = self.pre_norm(x, training=training)
        x = self.conv1_1x1(x)
        x = self.norm1(x, training=training)
        x = keras.activations.gelu(x, approximate=True)
        x = self.conv2_kxk(x)
        x = self.norm2(x, training=training)
        x = keras.activations.gelu(x, approximate=True)
        x = self.se(x, training=training)
        x = self.conv3_1x1(x)
        x = x + shortcut
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "expand_ratio": self.expand_ratio,
                "se_ratio": self.se_ratio,
                "stride": self.stride,
                "drop_path_rate": self.drop_path_rate,
            }
        )
        return config
