import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.model_registry import register_model
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import MAXVIT_MODEL_CONFIG, MAXVIT_WEIGHTS_CONFIG
from .maxvit_layers import (
    MaxViTAttention,
    MaxViTGridPartition,
    MaxViTGridReverse,
    MaxViTWindowPartition,
    MaxViTWindowReverse,
)


def maxvit_gelu_approximate(x):
    """GELU activation with tanh approximation, matching timm's GELUTanh."""
    return keras.activations.gelu(x, approximate=True)


def maxvit_mbconv_block(
    x,
    in_channels,
    out_channels,
    expand_ratio=4,
    se_ratio=0.0625,
    stride=1,
    data_format="channels_last",
    channels_axis=-1,
    prefix="",
):
    """Mobile Inverted Bottleneck Convolution (MBConv) block for MaxViT.

    Implements a pre-norm MBConv block with Squeeze-and-Excitation. The block
    consists of: BatchNorm -> 1x1 expand -> BN+GELU -> 3x3 depthwise ->
    BN+GELU -> SE -> 1x1 project -> residual add.

    Args:
        x: Input tensor.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        expand_ratio: Channel expansion ratio. Defaults to ``4``.
        se_ratio: Squeeze-and-Excitation reduction ratio. Defaults to ``0.0625``.
        stride: Spatial stride for the depthwise convolution. Defaults to ``1``.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index (``-1`` or ``1``).
        prefix: String prefix for layer names.

    Returns:
        Output tensor.
    """
    expanded = out_channels * expand_ratio
    se_reduced = max(1, int(expanded * se_ratio))

    shortcut = x
    if stride > 1:
        shortcut = layers.AveragePooling2D(
            pool_size=2,
            strides=2,
            padding="same",
            data_format=data_format,
            name=prefix + "conv_shortcut_pool",
        )(shortcut)
    if in_channels != out_channels:
        shortcut = layers.Conv2D(
            out_channels,
            1,
            use_bias=True,
            data_format=data_format,
            name=prefix + "conv_shortcut_expand",
        )(shortcut)

    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.9,
        name=prefix + "conv_pre_norm",
    )(x)
    x = layers.Conv2D(
        expanded,
        1,
        use_bias=False,
        data_format=data_format,
        name=prefix + "conv_conv1_1x1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.9,
        name=prefix + "conv_norm1",
    )(x)
    x = layers.Activation(maxvit_gelu_approximate, name=prefix + "conv_act1")(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=prefix + "conv_conv2_kxk",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.9,
        name=prefix + "conv_norm2",
    )(x)
    x = layers.Activation(maxvit_gelu_approximate, name=prefix + "conv_act2")(x)

    # Squeeze-and-Excitation
    x_se = layers.GlobalAveragePooling2D(
        data_format=data_format,
        keepdims=True,
        name=prefix + "se_pool",
    )(x)
    x_se = layers.Conv2D(
        se_reduced,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "se_fc1",
    )(x_se)
    x_se = layers.Activation("silu", name=prefix + "se_act")(x_se)
    x_se = layers.Conv2D(
        expanded,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "se_fc2",
    )(x_se)
    x_se = layers.Activation("sigmoid", name=prefix + "se_gate")(x_se)
    x = layers.Multiply()([x, x_se])

    x = layers.Conv2D(
        out_channels,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "conv_conv3_1x1",
    )(x)
    x = layers.Add()([x, shortcut])
    return x


def maxvit_partition_attn_block(
    x,
    dim,
    num_heads,
    window_size,
    img_size,
    partition_type="block",
    mlp_ratio=4.0,
    data_format="channels_last",
    prefix="",
):
    """Partition-based attention block with MLP for MaxViT.

    Applies multi-head self-attention within spatial partitions (local windows
    or dilated grids), followed by an MLP branch. Both branches use pre-norm
    residual connections.

    The partition layers convert to channels-last for attention and convert
    back to the original ``data_format`` after.

    Args:
        x: Input tensor.
        dim: Channel dimension.
        num_heads: Number of attention heads.
        window_size: Window / grid size for partitioning.
        img_size: Tuple ``(H, W)`` of the current spatial dimensions.
        partition_type: ``"block"`` or ``"grid"``. Defaults to ``"block"``.
        mlp_ratio: MLP hidden-dimension expansion ratio. Defaults to ``4.0``.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        prefix: String prefix for layer names.

    Returns:
        Output tensor of same shape as input.
    """
    part_size = (
        (window_size, window_size) if isinstance(window_size, int) else window_size
    )

    if partition_type == "block":
        partitioned = MaxViTWindowPartition(
            part_size,
            data_format=data_format,
            name=prefix + "win_part",
        )(x)
    else:
        partitioned = MaxViTGridPartition(
            part_size,
            data_format=data_format,
            name=prefix + "grid_part",
        )(x)

    # Attention operates on channels-last (B*windows, wh, ww, C)
    y = layers.LayerNormalization(epsilon=1e-5, name=prefix + "norm1")(partitioned)
    y = MaxViTAttention(
        dim=dim,
        num_heads=num_heads,
        window_size=part_size,
        prefix=prefix,
    )(y)

    if partition_type == "block":
        y = MaxViTWindowReverse(
            part_size,
            img_size,
            data_format=data_format,
            name=prefix + "win_rev",
        )(y)
    else:
        y = MaxViTGridReverse(
            part_size,
            img_size,
            data_format=data_format,
            name=prefix + "grid_rev",
        )(y)

    x = layers.Add()([x, y])

    residual = x
    if data_format == "channels_first":
        y = layers.Permute((2, 3, 1), name=prefix + "mlp_to_cl")(x)
        y = layers.LayerNormalization(epsilon=1e-5, name=prefix + "norm2")(y)
        y = layers.Dense(int(dim * mlp_ratio), use_bias=True, name=prefix + "mlp_fc1")(
            y
        )
        y = layers.Activation(maxvit_gelu_approximate, name=prefix + "mlp_gelu")(y)
        y = layers.Dense(dim, use_bias=True, name=prefix + "mlp_fc2")(y)
        y = layers.Permute((3, 1, 2), name=prefix + "mlp_to_cf")(y)
    else:
        y = layers.LayerNormalization(epsilon=1e-5, name=prefix + "norm2")(x)
        y = layers.Dense(int(dim * mlp_ratio), use_bias=True, name=prefix + "mlp_fc1")(
            y
        )
        y = layers.Activation(maxvit_gelu_approximate, name=prefix + "mlp_gelu")(y)
        y = layers.Dense(dim, use_bias=True, name=prefix + "mlp_fc2")(y)

    x = layers.Add()([residual, y])
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class MaxViT(keras.Model):
    """MaxViT: Multi-Axis Vision Transformer.

    Combines MBConv blocks with window and grid attention for efficient
    multi-scale feature extraction. Each MaxViT block applies MBConv for
    local features, then window attention for local context, then grid
    attention for global context.

    References:
        - [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)

    Args:
        stem_width: Width of the stem convolution.
        depths: List of block depths for each stage.
        embed_dim: List of embedding dimensions per stage.
        num_heads: List of attention heads per stage.
        window_size: Window size for partitioned attention.
        mlp_ratio: MLP expansion ratio.
        se_ratio: SE reduction ratio.
        expand_ratio: MBConv expansion ratio.
        include_top: Whether to include the classification head.
        include_normalization: Whether to include input normalization.
        normalization_mode: Normalization mode.
        weights: Pre-trained weight identifier or file path.
        input_shape: Input shape tuple.
        input_tensor: Optional input tensor.
        pooling: Pooling mode when include_top=False.
        num_classes: Number of output classes.
        classifier_activation: Activation for the classification head.
        name: Model name.
    """

    def __init__(
        self,
        stem_width=64,
        depths=(2, 2, 5, 2),
        embed_dim=(64, 128, 256, 512),
        num_heads=(2, 4, 8, 16),
        window_size=7,
        mlp_ratio=4.0,
        se_ratio=0.0625,
        expand_ratio=4,
        include_top=True,
        include_normalization=True,
        normalization_mode="imagenet",
        weights=None,
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="MaxViT",
        **kwargs,
    ):
        if include_top and num_classes is None:
            raise ValueError(
                "If `include_top` is True, `num_classes` must be specified."
            )

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
            min_size=32,
            data_format=data_format,
            require_flatten=include_top,
            weights=weights,
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input

        if include_normalization:
            from kmodels.layers import ImageNormalizationLayer

            x = ImageNormalizationLayer(mode=normalization_mode)(inputs)
        else:
            x = inputs

        if data_format == "channels_last":
            H, W = input_shape[0], input_shape[1]
        else:
            H, W = input_shape[1], input_shape[2]

        # Stem
        x = layers.Conv2D(
            stem_width,
            3,
            strides=2,
            padding="same",
            use_bias=True,
            data_format=data_format,
            name="stem_conv1",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-3,
            momentum=0.9,
            name="stem_norm1",
        )(x)
        x = layers.Activation(maxvit_gelu_approximate, name="stem_act1")(x)
        x = layers.Conv2D(
            stem_width,
            3,
            strides=1,
            padding="same",
            use_bias=True,
            data_format=data_format,
            name="stem_conv2",
        )(x)

        cur_H = H // 2
        cur_W = W // 2

        # Stages
        in_ch = stem_width
        for stage_idx in range(len(depths)):
            out_ch = embed_dim[stage_idx]
            for block_idx in range(depths[stage_idx]):
                prefix = f"stages_{stage_idx}_blocks_{block_idx}_"
                stride = 2 if block_idx == 0 else 1
                block_in_ch = in_ch if block_idx == 0 else out_ch

                x = maxvit_mbconv_block(
                    x,
                    in_channels=block_in_ch,
                    out_channels=out_ch,
                    expand_ratio=expand_ratio,
                    se_ratio=se_ratio,
                    stride=stride,
                    data_format=data_format,
                    channels_axis=channels_axis,
                    prefix=prefix,
                )

                if stride == 2:
                    cur_H = cur_H // 2
                    cur_W = cur_W // 2

                x = maxvit_partition_attn_block(
                    x,
                    dim=out_ch,
                    num_heads=num_heads[stage_idx],
                    window_size=window_size,
                    img_size=(cur_H, cur_W),
                    partition_type="block",
                    mlp_ratio=mlp_ratio,
                    data_format=data_format,
                    prefix=prefix + "attn_block_",
                )
                x = maxvit_partition_attn_block(
                    x,
                    dim=out_ch,
                    num_heads=num_heads[stage_idx],
                    window_size=window_size,
                    img_size=(cur_H, cur_W),
                    partition_type="grid",
                    mlp_ratio=mlp_ratio,
                    data_format=data_format,
                    prefix=prefix + "attn_grid_",
                )

            in_ch = out_ch

        # Head
        if include_top:
            x = layers.GlobalAveragePooling2D(
                data_format=data_format,
                name="head_global_pool",
            )(x)
            x = layers.LayerNormalization(
                axis=-1,
                epsilon=1e-5,
                name="head_norm",
            )(x)
            x = layers.Dense(
                embed_dim[-1],
                use_bias=True,
                name="head_pre_logits_fc",
            )(x)
            x = layers.Activation("tanh", name="head_pre_logits_act")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="head_fc",
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(
                    data_format=data_format,
                    name="avg_pool",
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(
                    data_format=data_format,
                    name="max_pool",
                )(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.stem_width = stem_width
        self.depths = list(depths)
        self.embed_dim = list(embed_dim)
        self.num_heads = list(num_heads)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.include_top = include_top
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_width": self.stem_width,
                "depths": self.depths,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "se_ratio": self.se_ratio,
                "expand_ratio": self.expand_ratio,
                "include_top": self.include_top,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "pooling": self.pooling,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_maxvit(
    variant,
    weights,
    include_top,
    include_normalization,
    normalization_mode,
    input_tensor,
    input_shape,
    pooling,
    num_classes,
    classifier_activation,
    name,
    **kwargs,
):
    cfg = {**MAXVIT_MODEL_CONFIG[variant], **kwargs}
    model = MaxViT(
        **cfg,
        include_top=include_top,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
    )

    if weights in get_all_weight_names(MAXVIT_WEIGHTS_CONFIG):
        load_weights_from_config(variant, weights, model, MAXVIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)

    return model


@register_model
def MaxViTTiny(
    include_top=True,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MaxViTTiny",
    **kwargs,
):
    return _create_maxvit(
        "MaxViTTiny",
        weights,
        include_top,
        include_normalization,
        normalization_mode,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )


@register_model
def MaxViTSmall(
    include_top=True,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MaxViTSmall",
    **kwargs,
):
    return _create_maxvit(
        "MaxViTSmall",
        weights,
        include_top,
        include_normalization,
        normalization_mode,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )


@register_model
def MaxViTBase(
    include_top=True,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MaxViTBase",
    **kwargs,
):
    return _create_maxvit(
        "MaxViTBase",
        weights,
        include_top,
        include_normalization,
        normalization_mode,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )


@register_model
def MaxViTLarge(
    include_top=True,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MaxViTLarge",
    **kwargs,
):
    return _create_maxvit(
        "MaxViTLarge",
        weights,
        include_top,
        include_normalization,
        normalization_mode,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )


@register_model
def MaxViTXLarge(
    include_top=True,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in21k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=21843,
    classifier_activation="softmax",
    name="MaxViTXLarge",
    **kwargs,
):
    return _create_maxvit(
        "MaxViTXLarge",
        weights,
        include_top,
        include_normalization,
        normalization_mode,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )
