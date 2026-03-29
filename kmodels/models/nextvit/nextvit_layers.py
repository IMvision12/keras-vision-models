import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class EfficientAttention(layers.Layer):
    """Efficient Multi-Head Self-Attention with optional spatial reduction.

    Operates on sequence format (B, N, C). When sr_ratio > 1, applies
    AvgPool1d to reduce the spatial dimension of keys and values.

    Args:
        dim: Input dimension.
        head_dim: Dimension per attention head. Defaults to 32.
        sr_ratio: Spatial reduction ratio. Defaults to 1.
        attn_drop: Dropout rate for attention weights. Defaults to 0.0.
        proj_drop: Dropout rate for projection output. Defaults to 0.0.
        prefix: Name prefix for sublayers.
        **kwargs: Additional keyword arguments passed to the Layer class.
    """

    def __init__(
        self,
        dim,
        head_dim=32,
        sr_ratio=1,
        attn_drop=0.0,
        proj_drop=0.0,
        prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.n_ratio = sr_ratio**2

        self.q = layers.Dense(dim, use_bias=True, name=prefix + "e_mhsa_q")
        self.k = layers.Dense(dim, use_bias=True, name=prefix + "e_mhsa_k")
        self.v = layers.Dense(dim, use_bias=True, name=prefix + "e_mhsa_v")
        self.proj = layers.Dense(dim, use_bias=True, name=prefix + "e_mhsa_proj")
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj_drop = layers.Dropout(proj_drop)

        self.use_sr = sr_ratio > 1
        if self.use_sr:
            self.norm = layers.BatchNormalization(
                axis=1,
                epsilon=1e-5,
                momentum=0.9,
                name=prefix + "e_mhsa_norm",
            )

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        B = input_shape[0]
        N = input_shape[1]

        q = self.q(x)
        q = ops.reshape(q, [B, N, self.num_heads, self.head_dim])
        q = ops.transpose(q, [0, 2, 1, 3])

        if self.use_sr:
            x_sr = ops.transpose(x, [0, 2, 1])
            N_reduced = N // self.n_ratio
            x_sr = ops.reshape(x_sr, [B, self.dim, N_reduced, self.n_ratio])
            x_sr = ops.mean(x_sr, axis=-1)
            x_sr = self.norm(x_sr, training=training)
            x_sr = ops.transpose(x_sr, [0, 2, 1])
        else:
            x_sr = x

        k = self.k(x_sr)
        v = self.v(x_sr)
        N_kv = ops.shape(k)[1]
        k = ops.reshape(k, [B, N_kv, self.num_heads, self.head_dim])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.reshape(v, [B, N_kv, self.num_heads, self.head_dim])
        v = ops.transpose(v, [0, 2, 1, 3])

        q = q * self.scale
        attn = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, [0, 2, 1, 3])
        out = ops.reshape(out, input_shape)

        out = self.proj(out)
        out = self.proj_drop(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "head_dim": self.head_dim,
                "sr_ratio": self.sr_ratio,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class ConvAttention(layers.Layer):
    """Multi-Head Convolutional Attention (MHCA).

    Applies grouped 3x3 convolution followed by BN, ReLU, and 1x1 projection.
    Operates on NHWC spatial tensors.

    Args:
        out_chs: Number of output channels.
        head_dim: Dimension per head (determines number of groups). Defaults to 32.
        prefix: Name prefix for sublayers.
        **kwargs: Additional keyword arguments passed to the Layer class.
    """

    def __init__(self, out_chs, head_dim=32, prefix="", **kwargs):
        super().__init__(**kwargs)
        self.out_chs = out_chs
        self.head_dim = head_dim
        self.num_groups = out_chs // head_dim

        self.group_conv3x3 = layers.Conv2D(
            out_chs,
            3,
            strides=1,
            padding="same",
            groups=self.num_groups,
            use_bias=False,
            data_format="channels_last",
            name=prefix + "mhca_group_conv3x3",
        )
        self.norm = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name=prefix + "mhca_norm",
        )
        self.projection = layers.Conv2D(
            out_chs,
            1,
            use_bias=False,
            data_format="channels_last",
            name=prefix + "mhca_projection",
        )

    def call(self, x, training=None):
        out = self.group_conv3x3(x)
        out = self.norm(out, training=training)
        out = keras.activations.relu(out)
        out = self.projection(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_chs": self.out_chs,
                "head_dim": self.head_dim,
            }
        )
        return config
