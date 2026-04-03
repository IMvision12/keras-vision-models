import math

import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class SwinV2WindowPartition(layers.Layer):
    """Partitions input tensor into non-overlapping windows.

    Divides the input feature map into non-overlapping windows of specified
    size. Supports two modes: standard partitioning (fused=False) and fused
    attention mode (fused=True) that handles multi-headed attention and QKV
    transformations in a single reshape-transpose operation.

    Args:
        window_size: int. Size of each window (both height and width).
        fused: bool. If True, operates in fused attention mode that produces
            output shaped for multi-head attention with QKV split.
            Defaults to False.
        num_heads: int. Number of attention heads. Required when fused=True.
        qkv_mult: int. Multiplier for QKV transformations. Defaults to 3.
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
class SwinV2WindowReverse(layers.Layer):
    """Reverses window partitioning to reconstruct the original feature map.

    Performs the inverse operation of ``SwinV2WindowPartition``, merging
    window segments back into a contiguous spatial feature map. Supports
    both standard and fused attention modes.

    Args:
        window_size: int. Size of each window (both height and width).
        fused: bool. If True, operates in fused attention mode.
            Defaults to False.
        num_heads: int. Number of attention heads. Required when fused=True.
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
class SwinV2Attention(layers.Layer):
    """Window-based multi-head self-attention with cosine similarity (SwinV2).

    Implements the attention mechanism from Swin Transformer V2 with several
    key differences from V1:

    - **Cosine attention**: Q and K are L2-normalized before computing
      attention, with a learnable per-head ``logit_scale``.
    - **Continuous Position Bias (CPB)**: A small MLP maps log-spaced
      relative coordinates to per-head bias values, replacing the learnable
      bias table from V1.
    - **Separate Q/V bias**: Learnable ``q_bias`` and ``v_bias`` are added
      to the QKV projection; K has no bias.

    Args:
        dim: int. Total dimension of input and output features.
        num_heads: int. Number of parallel attention heads.
        window_size: int. Spatial size of the attention window (W x W).
        pretrained_window_size: int. Window size used during pretraining
            for CPB coordinate normalization. Defaults to 0.
        attn_drop: float. Dropout rate for attention weights. Defaults to 0.0.
        proj_drop: float. Dropout rate for output projection. Defaults to 0.0.
        block_prefix: str. Prefix for naming layer components.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        pretrained_window_size=0,
        attn_drop=0.0,
        proj_drop=0.0,
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.pretrained_window_size = int(pretrained_window_size)
        self.head_dim = dim // num_heads

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=False,
            dtype=self.dtype_policy,
            name=prefix + "attn_qkv",
        )

        self.window_partition = SwinV2WindowPartition(
            window_size=self.window_size, fused=True, num_heads=num_heads, qkv_mult=3
        )
        self.window_reverse = SwinV2WindowReverse(
            window_size=self.window_size, fused=True, num_heads=num_heads
        )

        self.cpb_dense1 = layers.Dense(
            512,
            use_bias=True,
            dtype=self.dtype_policy,
            name=prefix + "attn_cpb_mlp_0",
        )
        self.cpb_dense2 = layers.Dense(
            num_heads,
            use_bias=False,
            dtype=self.dtype_policy,
            name=prefix + "attn_cpb_mlp_2",
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

        self.logit_scale = self.add_weight(
            name=prefix + "attn_logit_scale",
            shape=[self.num_heads],
            initializer=keras.initializers.Constant(math.log(10.0)),
            trainable=True,
            dtype=self.dtype,
        )

        self.q_bias = self.add_weight(
            name=prefix + "attn_q_bias",
            shape=[self.dim],
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
        )
        self.v_bias = self.add_weight(
            name=prefix + "attn_v_bias",
            shape=[self.dim],
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
        )

        self.cpb_dense1.build((None, 2))
        self.cpb_dense2.build((None, 512))

        ws = self.window_size
        coords_h = np.arange(-(ws - 1), ws, dtype=np.float32)
        coords_w = np.arange(-(ws - 1), ws, dtype=np.float32)
        coords_table = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)

        if self.pretrained_window_size > 0:
            coords_table[:, :, 0] = coords_table[:, :, 0] / (
                self.pretrained_window_size - 1
            )
            coords_table[:, :, 1] = coords_table[:, :, 1] / (
                self.pretrained_window_size - 1
            )
        else:
            coords_table[:, :, 0] = coords_table[:, :, 0] / (ws - 1)
            coords_table[:, :, 1] = coords_table[:, :, 1] / (ws - 1)

        coords_table = (
            np.sign(coords_table)
            * np.log2(1.0 + np.abs(coords_table) * 8.0)
            / math.log2(8.0)
        )

        self.relative_coords_table = self.add_weight(
            name=prefix + "attn_relative_coords_table",
            shape=coords_table.shape,
            initializer=keras.initializers.Constant(coords_table),
            trainable=False,
            dtype=self.dtype,
        )

        coords = np.arange(ws)
        gx, gy = np.meshgrid(coords, coords, indexing="ij")
        flat_gx = gx.reshape(-1)
        flat_gy = gy.reshape(-1)

        rel_pos_x = flat_gx[:, None] - flat_gx[None, :]
        rel_pos_y = flat_gy[:, None] - flat_gy[None, :]

        rel_pos_x = rel_pos_x + ws - 1
        rel_pos_y = rel_pos_y + ws - 1

        rel_pos_index = rel_pos_x * (2 * ws - 1) + rel_pos_y
        rel_pos_index = rel_pos_index.reshape(-1).astype(np.int32)

        self.relative_position_index = self.add_weight(
            name=prefix + "attn_relative_position_index",
            shape=rel_pos_index.shape,
            initializer=keras.initializers.Constant(rel_pos_index),
            trainable=False,
            dtype="int32",
        )

        self.built = True

    def with_mask(self, attn, mask, length):
        mask_windows = ops.shape(mask)[1]
        attn = ops.reshape(attn, [-1, mask_windows, self.num_heads, length, length])
        attn = attn + mask
        attn = ops.reshape(attn, [-1, self.num_heads, length, length])
        return attn

    def call(self, inputs, training=None):
        inputs, window_size, attention_mask = inputs
        height, width = ops.shape(inputs)[1:3]
        length = window_size**2

        k_bias = ops.zeros_like(self.q_bias)
        qkv_bias = ops.concatenate([self.q_bias, k_bias, self.v_bias])

        qkv = ops.matmul(inputs, self.qkv.kernel)
        qkv = qkv + qkv_bias

        qkv = self.window_partition(qkv, height=height, width=width)
        q, k, v = ops.unstack(qkv, 3)

        q_norm = ops.sqrt(
            ops.maximum(ops.sum(q * q, axis=-1, keepdims=True), ops.cast(1e-6, q.dtype))
        )
        k_norm = ops.sqrt(
            ops.maximum(ops.sum(k * k, axis=-1, keepdims=True), ops.cast(1e-6, k.dtype))
        )
        q = q / q_norm
        k = k / k_norm

        k = ops.swapaxes(k, -2, -1)
        attn = ops.matmul(q, k)

        log_scale = ops.minimum(self.logit_scale, math.log(100.0))
        scale = ops.exp(log_scale)
        attn = attn * scale[None, :, None, None]

        table = ops.reshape(self.relative_coords_table, [-1, 2])
        cpb = self.cpb_dense1(table)
        cpb = ops.relu(cpb)
        cpb = self.cpb_dense2(cpb)

        cpb = 16.0 * ops.sigmoid(cpb)

        bias = ops.take(cpb, self.relative_position_index, axis=0)
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
                "pretrained_window_size": self.pretrained_window_size,
                "attn_drop": self.attn_drop_rate,
                "proj_drop": self.proj_drop_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SwinV2Roll(layers.Layer):
    """Circular shift of tensor elements along spatial axes.

    Wraps ``ops.roll`` as a serializable Keras layer for use in shifted
    window attention. Elements that roll past the last position are
    re-introduced at the first position.

    Args:
        shift: int or tuple of ints. Number of positions to shift.
        axis: int or tuple of ints. Axes along which to shift.
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
