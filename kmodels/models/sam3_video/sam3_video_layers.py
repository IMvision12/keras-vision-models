"""Sam3Video layers: Vision Neck (FPN) and sine position embedding."""

import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3SinePositionEmbedding(layers.Layer):
    """Standard sine positional embedding for 2D feature maps.

    Generates position encodings based on spatial coordinates
    (cumulative sums along H and W axes), similar to the Attention
    Is All You Need paper, generalized to images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, **kwargs
    ):
        super().__init__(**kwargs)
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def call(self, x):
        """Generate positional encoding for a feature map.

        Args:
            x: (B, C, H, W) feature map tensor (NCHW).

        Returns:
            (B, 2*num_pos_feats, H, W) positional encoding in NCHW format.
        """
        shape = ops.shape(x)
        batch_size = shape[0]
        h = shape[2]
        w = shape[3]

        y_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, h + 1, dtype="float32"), axis=1),
                w,
                axis=1,
            ),
            "float32",
        )  # (H, W)
        x_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, w + 1, dtype="float32"), axis=0),
                h,
                axis=0,
            ),
            "float32",
        )  # (H, W)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = ops.arange(self.num_pos_feats, dtype="float32")
        dim_t = ops.cast(
            self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats),
            "float32",
        )

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t  # (H, W, D)
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t  # (H, W, D)

        pos_x_sin = ops.sin(pos_x[..., 0::2])
        pos_x_cos = ops.cos(pos_x[..., 1::2])
        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=-1),
            (h, w, self.num_pos_feats),
        )

        pos_y_sin = ops.sin(pos_y[..., 0::2])
        pos_y_cos = ops.cos(pos_y[..., 1::2])
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=-1),
            (h, w, self.num_pos_feats),
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)  # (H, W, 2*D)
        pos = ops.transpose(pos, (2, 0, 1))  # (2*D, H, W)
        pos = ops.expand_dims(pos, axis=0)  # (1, 2*D, H, W)
        pos = ops.broadcast_to(pos, (batch_size,) + ops.shape(pos)[1:])
        return pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3FPNLayer(layers.Layer):
    """Single FPN level with scale-dependent up/downsampling.

    Scale factors:
    - 4.0: ConvTranspose2d(1024→512, k2s2) + GELU + ConvTranspose2d(512→256, k2s2)
    - 2.0: ConvTranspose2d(1024→512, k2s2)
    - 1.0: identity (no scaling)
    - 0.5: MaxPool2d(k2, s2)

    Followed by: Conv2d(proj1, 1x1) + Conv2d(proj2, 3x3)
    """

    def __init__(self, in_channels=1024, fpn_dim=256, scale_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.fpn_dim = fpn_dim
        self.scale_factor = scale_factor

    def build(self, input_shape):
        sf = self.scale_factor
        ic = self.in_channels
        self._scale_layers = []

        if sf == 4.0:
            self._deconv1 = layers.Conv2DTranspose(
                ic // 2,
                kernel_size=2,
                strides=2,
                data_format="channels_first",
                name="scale_layers_0",
            )
            self._deconv1.build((None, ic, None, None))
            self._deconv2 = layers.Conv2DTranspose(
                ic // 4,
                kernel_size=2,
                strides=2,
                data_format="channels_first",
                name="scale_layers_2",
            )
            self._deconv2.build((None, ic // 2, None, None))
            self._scale_layers = [self._deconv1, "gelu", self._deconv2]
            intermediate = ic // 4
        elif sf == 2.0:
            self._deconv1 = layers.Conv2DTranspose(
                ic // 2,
                kernel_size=2,
                strides=2,
                data_format="channels_first",
                name="scale_layers_0",
            )
            self._deconv1.build((None, ic, None, None))
            self._scale_layers = [self._deconv1]
            intermediate = ic // 2
        elif sf == 1.0:
            intermediate = ic
        elif sf == 0.5:
            self._pool = layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                data_format="channels_first",
                name="scale_layers_0",
            )
            self._scale_layers = [self._pool]
            intermediate = ic
        else:
            raise NotImplementedError(f"scale_factor={sf} not supported")

        self.proj1 = layers.Conv2D(
            self.fpn_dim,
            kernel_size=1,
            data_format="channels_first",
            name="proj1",
        )
        self.proj1.build((None, intermediate, None, None))

        self.proj2 = layers.Conv2D(
            self.fpn_dim,
            kernel_size=3,
            padding="same",
            data_format="channels_first",
            name="proj2",
        )
        self.proj2.build((None, self.fpn_dim, None, None))
        self.built = True

    def call(self, hidden_states):
        """
        Args:
            hidden_states: (B, in_channels, H, W)
        Returns:
            (B, fpn_dim, H', W') where H'/W' depend on scale_factor.
        """
        x = hidden_states
        for layer in self._scale_layers:
            if layer == "gelu":
                x = ops.nn.gelu(x)
            else:
                x = layer(x)

        x = self.proj1(x)
        x = self.proj2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "fpn_dim": self.fpn_dim,
                "scale_factor": self.scale_factor,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3VisionNeck(layers.Layer):
    """FPN neck that bridges detector backbone features to tracker format.

    Creates multi-scale feature pyramid from backbone output and generates
    sine positional encodings for each FPN level.

    Architecture:
        backbone_out (B, 1024, H, W)
            → FPN level 0 (4x upsample): (B, 256, 4H, 4W)
            → FPN level 1 (2x upsample): (B, 256, 2H, 2W)
            → FPN level 2 (1x):           (B, 256, H, W)
            → FPN level 3 (0.5x):         (B, 256, H/2, W/2)
    """

    def __init__(
        self,
        backbone_hidden_size=1024,
        fpn_hidden_size=256,
        scale_factors=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_hidden_size = backbone_hidden_size
        self.fpn_hidden_size = fpn_hidden_size
        self.scale_factors = scale_factors or [4.0, 2.0, 1.0, 0.5]

    def build(self, input_shape):
        self.position_encoding = Sam3SinePositionEmbedding(
            num_pos_feats=self.fpn_hidden_size // 2,
            normalize=True,
            name="position_encoding",
        )

        self.fpn_layers = []
        for i, sf in enumerate(self.scale_factors):
            fpn = Sam3FPNLayer(
                in_channels=self.backbone_hidden_size,
                fpn_dim=self.fpn_hidden_size,
                scale_factor=sf,
                name=f"fpn_layers_{i}",
            )
            fpn.build((None, self.backbone_hidden_size, None, None))
            self.fpn_layers.append(fpn)

        self.built = True

    def call(self, hidden_states):
        """
        Args:
            hidden_states: (B, backbone_hidden_size, H, W) backbone output.

        Returns:
            fpn_hidden_states: list of (B, fpn_hidden_size, H', W') at each scale.
            fpn_position_encoding: list of (B, fpn_hidden_size, H', W') sine PE.
        """
        fpn_hidden_states = []
        fpn_position_encoding = []

        for fpn_layer in self.fpn_layers:
            fpn_output = fpn_layer(hidden_states)
            fpn_hidden_states.append(fpn_output)
            pos_enc = self.position_encoding(fpn_output)
            fpn_position_encoding.append(pos_enc)

        return fpn_hidden_states, fpn_position_encoding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_hidden_size": self.backbone_hidden_size,
                "fpn_hidden_size": self.fpn_hidden_size,
                "scale_factors": self.scale_factors,
            }
        )
        return config
