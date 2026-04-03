import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class MaxViTWindowPartition(layers.Layer):
    """Partition a spatial tensor into non-overlapping local windows.

    Supports both ``channels_last`` and ``channels_first`` inputs. Output is
    always ``channels_last`` ``(B * nH * nW, wh, ww, C)`` so that downstream
    attention layers can operate on a flat sequence.

    Args:
        window_size: Tuple ``(wh, ww)`` or int for the window dimensions.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.
    """

    def __init__(self, window_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.data_format = data_format

    def call(self, x):
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 2, 3, 1])
        wh, ww = self.window_size
        x_shape = ops.shape(x)
        B, H, W, C = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        nH = H // wh
        nW = W // ww
        x = ops.reshape(x, [B, nH, wh, nW, ww, C])
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [B * nH * nW, wh, ww, C])
        return x

    def compute_output_shape(self, input_shape):
        wh, ww = self.window_size
        if self.data_format == "channels_first":
            C = input_shape[1]
        else:
            C = input_shape[-1]
        return (None, wh, ww, C)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"window_size": self.window_size, "data_format": self.data_format}
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class MaxViTWindowReverse(layers.Layer):
    """Reverse window partition back to a spatial tensor.

    Input is always ``channels_last`` ``(B * nH * nW, wh, ww, C)``. Output
    format matches ``data_format``.

    Args:
        window_size: Tuple ``(wh, ww)`` or int for the window dimensions.
        img_size: Tuple ``(H, W)`` of the original spatial dimensions.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.
    """

    def __init__(self, window_size, img_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.img_size = img_size
        self.data_format = data_format

    def call(self, windows):
        wh, ww = self.window_size
        H, W = self.img_size
        C = ops.shape(windows)[-1]
        nH = H // wh
        nW = W // ww
        total_windows = ops.shape(windows)[0]
        B = total_windows // (nH * nW)
        x = ops.reshape(windows, [B, nH, nW, wh, ww, C])
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [B, H, W, C])
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 3, 1, 2])
        return x

    def compute_output_shape(self, input_shape):
        H, W = self.img_size
        C = input_shape[-1]
        if self.data_format == "channels_first":
            return (None, C, H, W)
        return (None, H, W, C)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "img_size": self.img_size,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class MaxViTGridPartition(layers.Layer):
    """Partition a spatial tensor into dilated (grid) windows.

    Supports both data formats. Output is always ``channels_last``.

    Args:
        grid_size: Tuple ``(gh, gw)`` or int for the grid dimensions.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.
    """

    def __init__(self, grid_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size
        self.data_format = data_format

    def call(self, x):
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 2, 3, 1])
        gh, gw = self.grid_size
        x_shape = ops.shape(x)
        B, H, W, C = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        nH = H // gh
        nW = W // gw
        x = ops.reshape(x, [B, gh, nH, gw, nW, C])
        x = ops.transpose(x, [0, 2, 4, 1, 3, 5])
        x = ops.reshape(x, [B * nH * nW, gh, gw, C])
        return x

    def compute_output_shape(self, input_shape):
        gh, gw = self.grid_size
        if self.data_format == "channels_first":
            C = input_shape[1]
        else:
            C = input_shape[-1]
        return (None, gh, gw, C)

    def get_config(self):
        config = super().get_config()
        config.update({"grid_size": self.grid_size, "data_format": self.data_format})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class MaxViTGridReverse(layers.Layer):
    """Reverse grid partition back to a spatial tensor.

    Input is always ``channels_last``. Output format matches ``data_format``.

    Args:
        grid_size: Tuple ``(gh, gw)`` or int for the grid dimensions.
        img_size: Tuple ``(H, W)`` of the original spatial dimensions.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.
    """

    def __init__(self, grid_size, img_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size
        self.img_size = img_size
        self.data_format = data_format

    def call(self, windows):
        gh, gw = self.grid_size
        H, W = self.img_size
        C = ops.shape(windows)[-1]
        nH = H // gh
        nW = W // gw
        total_windows = ops.shape(windows)[0]
        B = total_windows // (nH * nW)
        x = ops.reshape(windows, [B, nH, nW, gh, gw, C])
        x = ops.transpose(x, [0, 3, 1, 4, 2, 5])
        x = ops.reshape(x, [B, H, W, C])
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 3, 1, 2])
        return x

    def compute_output_shape(self, input_shape):
        H, W = self.img_size
        C = input_shape[-1]
        if self.data_format == "channels_first":
            return (None, C, H, W)
        return (None, H, W, C)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_size": self.grid_size,
                "img_size": self.img_size,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class RelPosBiasTf(layers.Layer):
    """Learnable relative position bias for MaxViT attention.

    Maintains a bias table of shape ``(num_heads, 2*wh-1, 2*ww-1)`` and
    reindexes it into ``(num_heads, window_area, window_area)`` using
    one-hot lookup tensors, matching the TensorFlow-style implementation
    from the original MaxViT codebase.

    Args:
        window_size: Tuple ``(wh, ww)`` for the attention window.
        num_heads: Number of attention heads.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.
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
    """Multi-head self-attention with relative position bias for MaxViT.

    Operates on channels-last ``(B, N, C)`` sequences. The partition layers
    handle any necessary format conversion before and after this layer.

    Args:
        dim: Total input / output feature dimension.
        num_heads: Number of parallel attention heads.
        window_size: Spatial window size for the relative position bias table.
        attn_drop: Dropout rate applied to attention weights. Defaults to ``0.0``.
        proj_drop: Dropout rate applied after the output projection. Defaults
            to ``0.0``.
        prefix: String prefix prepended to sub-layer names. Defaults to ``""``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.
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
