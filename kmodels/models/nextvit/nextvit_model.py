import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.model_registry import register_model
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import NEXTVIT_MODEL_CONFIG, NEXTVIT_WEIGHTS_CONFIG
from .nextvit_layers import EfficientAttention


def nextvit_conv_attention(x, out_chs, head_dim=32, prefix=""):
    """Multi-Head Convolutional Attention (MHCA).

    Applies grouped 3x3 convolution followed by batch normalization,
    ReLU activation, and 1x1 projection. Operates on NHWC spatial
    tensors.

    Args:
        x: Input tensor of shape ``(B, H, W, C)``.
        out_chs: Integer, number of output channels.
        head_dim: Integer, dimension per head (determines number of
            groups). Defaults to ``32``.
        prefix: String, name prefix for all sub-layers.

    Returns:
        Output tensor of shape ``(B, H, W, out_chs)``.
    """
    num_groups = out_chs // head_dim
    out = layers.Conv2D(
        out_chs,
        3,
        strides=1,
        padding="same",
        groups=num_groups,
        use_bias=False,
        data_format="channels_last",
        name=prefix + "mhca_group_conv3x3",
    )(x)
    out = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "mhca_norm",
    )(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(
        out_chs,
        1,
        use_bias=False,
        data_format="channels_last",
        name=prefix + "mhca_projection",
    )(out)
    return out


def _make_divisible(v, divisor, min_value=None):
    """Round value to be divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _calculate_drop_path_rates(drop_path_rate, depths):
    """Generate stagewise drop path rates."""
    total_depth = sum(depths)
    rates = []
    idx = 0
    for d in depths:
        stage_rates = []
        for i in range(d):
            stage_rates.append(
                drop_path_rate * idx / (total_depth - 1) if total_depth > 1 else 0.0
            )
            idx += 1
        rates.append(stage_rates)
    return rates


def _get_stage_out_chs(depths):
    """Compute per-block output channels for each stage."""
    return [
        [96] * depths[0],
        [192] * (depths[1] - 1) + [256],
        [384, 384, 384, 384, 512] * (depths[2] // 5),
        [768] * (depths[3] - 1) + [1024],
    ]


def _get_stage_block_types(depths):
    """Compute per-block type ('conv' or 'transformer') for each stage."""
    return [
        ["conv"] * depths[0],
        ["conv"] * (depths[1] - 1) + ["transformer"],
        ["conv", "conv", "conv", "conv", "transformer"] * (depths[2] // 5),
        ["conv"] * (depths[3] - 1) + ["transformer"],
    ]


def conv_mlp(x, in_features, hidden_features, out_features=None, prefix=""):
    """ConvMlp: 1x1 Conv -> ReLU -> Dropout -> 1x1 Conv.

    Args:
        x: Input tensor (B, H, W, C) in channels_last format.
        in_features: Number of input channels.
        hidden_features: Number of hidden channels.
        out_features: Number of output channels. Defaults to in_features.
        prefix: Name prefix for layers.

    Returns:
        Output tensor (B, H, W, out_features).
    """
    if out_features is None:
        out_features = in_features
    x = layers.Conv2D(
        hidden_features,
        1,
        use_bias=True,
        data_format="channels_last",
        name=prefix + "mlp_fc1",
    )(x)
    x = layers.Activation("relu", name=prefix + "mlp_act")(x)
    x = layers.Conv2D(
        out_features,
        1,
        use_bias=True,
        data_format="channels_last",
        name=prefix + "mlp_fc2",
    )(x)
    return x


def patch_embed_block(x, in_chs, out_chs, use_pool, prefix=""):
    """PatchEmbed: optional pool -> conv1x1 -> BN, or identity if in_chs == out_chs and no pool.

    Args:
        x: Input tensor (B, H, W, C).
        in_chs: Input channels.
        out_chs: Output channels.
        use_pool: Whether to apply 2x2 average pooling (stride=2).
        prefix: Name prefix for layers.

    Returns:
        Output tensor.
    """
    if use_pool:
        x = layers.AveragePooling2D(
            pool_size=2,
            strides=2,
            padding="valid",
            data_format="channels_last",
            name=prefix + "patch_embed_pool",
        )(x)
    if use_pool or in_chs != out_chs:
        x = layers.Conv2D(
            out_chs,
            1,
            use_bias=False,
            data_format="channels_last",
            name=prefix + "patch_embed_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name=prefix + "patch_embed_norm",
        )(x)
    return x


def next_conv_block(
    x, in_chs, out_chs, stride, drop_path_rate, head_dim, mlp_ratio, prefix=""
):
    """NextConvBlock: patch_embed -> mhca + residual -> norm -> conv_mlp + residual.

    Args:
        x: Input tensor (B, H, W, C).
        in_chs: Input channels.
        out_chs: Output channels.
        stride: Stride for patch embed (1 or 2).
        drop_path_rate: Drop path rate for stochastic depth.
        head_dim: Head dimension for ConvAttention groups.
        mlp_ratio: MLP expansion ratio (3.0 for ConvBlocks).
        prefix: Name prefix for layers.

    Returns:
        Output tensor (B, H', W', out_chs).
    """
    use_pool = stride == 2
    x = patch_embed_block(x, in_chs, out_chs, use_pool, prefix=prefix)

    # MHCA + residual
    mhca_out = nextvit_conv_attention(x, out_chs, head_dim=head_dim, prefix=prefix)
    x = layers.Add()([x, mhca_out])

    # Norm -> MLP + residual
    residual = x
    out = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm",
    )(x)
    out = conv_mlp(out, out_chs, int(out_chs * mlp_ratio), out_chs, prefix=prefix)
    x = layers.Add()([residual, out])
    return x


def next_transformer_block(
    x,
    in_chs,
    out_chs,
    stride,
    drop_path_rate,
    head_dim,
    sr_ratio,
    mix_block_ratio,
    mlp_ratio,
    prefix="",
):
    """NextTransformerBlock: patch_embed -> E-MHSA + residual -> projection -> MHCA -> concat -> norm -> MLP + residual.

    The block splits the output channels between MHSA and MHCA branches using
    mix_block_ratio. The MHSA branch gets mhsa_out_chs = round(out_chs * mix_block_ratio)
    and the MHCA branch gets mhca_out_chs = out_chs - mhsa_out_chs.

    Args:
        x: Input tensor (B, H, W, C).
        in_chs: Input channels.
        out_chs: Output channels.
        stride: Stride for patch embed.
        drop_path_rate: Drop path rate.
        head_dim: Head dimension.
        sr_ratio: Spatial reduction ratio for E-MHSA.
        mix_block_ratio: Ratio for splitting channels between MHSA and MHCA.
        mlp_ratio: MLP expansion ratio (2.0 for TransformerBlocks).
        prefix: Name prefix for layers.

    Returns:
        Output tensor (B, H', W', out_chs).
    """
    mhsa_out_chs = _make_divisible(int(out_chs * mix_block_ratio), 32)
    mhca_out_chs = out_chs - mhsa_out_chs

    use_pool = stride == 2
    x = patch_embed_block(x, in_chs, mhsa_out_chs, use_pool, prefix=prefix)

    # E-MHSA branch: BN -> reshape to (B, N, C) -> attention -> reshape back
    out = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm1",
    )(x)
    # NHWC -> (B, N, C) for attention
    out = layers.Reshape((-1, mhsa_out_chs), name=prefix + "reshape_to_seq")(out)

    out = EfficientAttention(
        mhsa_out_chs,
        head_dim=head_dim,
        sr_ratio=sr_ratio,
        prefix=prefix,
        name=prefix + "e_mhsa",
    )(out)

    # (B, N, C) -> NHWC: we need to get H, W from x
    # Use a Lambda that reshapes using the spatial dims from x
    x_shape = ops.shape(x)
    out = layers.Reshape(
        (x_shape[1], x_shape[2], mhsa_out_chs),
        name=prefix + "reshape_to_spatial",
    )(out)

    x = layers.Add()([x, out])

    # Projection: conv 1x1 from mhsa_out_chs -> mhca_out_chs (with BN)
    proj_out = layers.Conv2D(
        mhca_out_chs,
        1,
        use_bias=False,
        data_format="channels_last",
        name=prefix + "projection_conv",
    )(x)
    proj_out = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "projection_norm",
    )(proj_out)

    # MHCA on projected features + residual
    mhca_out = nextvit_conv_attention(
        proj_out, mhca_out_chs, head_dim=head_dim, prefix=prefix
    )
    proj_out = layers.Add()([proj_out, mhca_out])

    # Concat MHSA and MHCA branches along channel axis
    x = layers.Concatenate(axis=-1)([x, proj_out])

    # Norm -> MLP + residual
    residual = x
    out = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm2",
    )(x)
    out = conv_mlp(out, out_chs, int(out_chs * mlp_ratio), out_chs, prefix=prefix)
    x = layers.Add()([residual, out])
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class NextViT(keras.Model):
    """NextViT: Next Vision Transformer.

    A hybrid CNN-Transformer architecture that combines convolutional attention
    blocks with efficient multi-head self-attention blocks for image classification.

    References:
        - [Next-ViT: Next Generation Vision Transformer for Efficient Deployment
          in Realistic Industrial Scenarios](https://arxiv.org/abs/2207.05501)

    Args:
        depths: List of block depths for each of the 4 stages.
        stem_chs: List of 3 channel dimensions for the stem convolutions.
        head_dim: Dimension per attention head. Defaults to 32.
        mix_block_ratio: Ratio for splitting channels in TransformerBlocks. Defaults to 0.75.
        sr_ratios: List of spatial reduction ratios per stage. Defaults to [8, 4, 2, 1].
        drop_path_rate: Maximum drop path rate. Defaults to 0.1.
        include_top: Whether to include the classification head. Defaults to True.
        as_backbone: Whether to output intermediate feature maps. Defaults to False.
        include_normalization: Whether to include input normalization. Defaults to True.
        normalization_mode: Normalization mode. Defaults to 'imagenet'.
        weights: Path to weights or None.
        input_shape: Input shape tuple.
        input_tensor: Optional input tensor.
        pooling: Pooling mode when include_top=False.
        num_classes: Number of output classes. Defaults to 1000.
        classifier_activation: Activation for the classification head. Defaults to 'softmax'.
        name: Model name. Defaults to 'NextViT'.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        depths=(3, 4, 10, 3),
        stem_chs=(64, 32, 64),
        head_dim=32,
        mix_block_ratio=0.75,
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.1,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights=None,
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="NextViT",
        **kwargs,
    ):
        if include_top and num_classes is None:
            raise ValueError(
                "If `include_top` is True, `num_classes` must be specified. "
                f"Received: {num_classes}"
            )

        if include_top and as_backbone:
            raise ValueError(
                "Cannot use `as_backbone=True` with `include_top=True`. "
                f"Received: as_backbone={as_backbone}, include_top={include_top}"
            )

        if pooling is not None and pooling not in ["avg", "max"]:
            raise ValueError(
                "The `pooling` argument should be one of 'avg', 'max', or None. "
                f"Received: pooling={pooling}"
            )

        data_format = keras.config.image_data_format()

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

        stem_configs = [
            (3, stem_chs[0], 2),
            (stem_chs[0], stem_chs[1], 1),
            (stem_chs[1], stem_chs[2], 1),
            (stem_chs[2], stem_chs[2], 2),
        ]
        for i, (in_c, out_c, stride) in enumerate(stem_configs):
            if stride == 2:
                x = layers.ZeroPadding2D(
                    padding=1,
                    data_format="channels_last",
                    name=f"stem_{i}_pad",
                )(x)
            x = layers.Conv2D(
                out_c,
                3,
                strides=stride,
                padding="valid" if stride == 2 else "same",
                use_bias=False,
                data_format="channels_last",
                name=f"stem_{i}_conv",
            )(x)
            x = layers.BatchNormalization(
                axis=-1,
                epsilon=1e-5,
                momentum=0.9,
                name=f"stem_{i}_norm",
            )(x)
            x = layers.Activation("relu", name=f"stem_{i}_act")(x)

        stage_out_chs = _get_stage_out_chs(depths)
        stage_block_types = _get_stage_block_types(depths)
        dpr = _calculate_drop_path_rates(drop_path_rate, depths)
        strides = [1, 2, 2, 2]

        in_chs = stem_chs[-1]

        for stage_idx in range(4):
            block_chs = stage_out_chs[stage_idx]
            block_types = stage_block_types[stage_idx]

            for block_idx in range(depths[stage_idx]):
                stride = strides[stage_idx] if block_idx == 0 else 1
                out_chs = block_chs[block_idx]
                block_type = block_types[block_idx]
                dp_rate = dpr[stage_idx][block_idx]
                prefix = f"stages_{stage_idx}_blocks_{block_idx}_"

                if block_type == "conv":
                    x = next_conv_block(
                        x,
                        in_chs=in_chs,
                        out_chs=out_chs,
                        stride=stride,
                        drop_path_rate=dp_rate,
                        head_dim=head_dim,
                        mlp_ratio=3.0,
                        prefix=prefix,
                    )
                else:
                    x = next_transformer_block(
                        x,
                        in_chs=in_chs,
                        out_chs=out_chs,
                        stride=stride,
                        drop_path_rate=dp_rate,
                        head_dim=head_dim,
                        sr_ratio=sr_ratios[stage_idx],
                        mix_block_ratio=mix_block_ratio,
                        mlp_ratio=2.0,
                        prefix=prefix,
                    )
                in_chs = out_chs

        x = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name="norm",
        )(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(
                data_format="channels_last",
                name="head_global_pool",
            )(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="head_fc",
            )(x)
        elif as_backbone:
            x = x
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(
                    data_format="channels_last",
                    name="avg_pool",
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(
                    data_format="channels_last",
                    name="max_pool",
                )(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.depths = list(depths)
        self.stem_chs = list(stem_chs)
        self.head_dim = head_dim
        self.mix_block_ratio = mix_block_ratio
        self.sr_ratios = list(sr_ratios)
        self.drop_path_rate = drop_path_rate
        self.include_top = include_top
        self.as_backbone = as_backbone
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
                "depths": self.depths,
                "stem_chs": self.stem_chs,
                "head_dim": self.head_dim,
                "mix_block_ratio": self.mix_block_ratio,
                "sr_ratios": self.sr_ratios,
                "drop_path_rate": self.drop_path_rate,
                "include_top": self.include_top,
                "as_backbone": self.as_backbone,
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


@register_model
def NextViTSmall(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="NextViTSmall",
    **kwargs,
):
    model = NextViT(
        **NEXTVIT_MODEL_CONFIG["NextViTSmall"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(NEXTVIT_WEIGHTS_CONFIG):
        load_weights_from_config("NextViTSmall", weights, model, NEXTVIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def NextViTBase(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="NextViTBase",
    **kwargs,
):
    model = NextViT(
        **NEXTVIT_MODEL_CONFIG["NextViTBase"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(NEXTVIT_WEIGHTS_CONFIG):
        load_weights_from_config("NextViTBase", weights, model, NEXTVIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def NextViTLarge(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="NextViTLarge",
    **kwargs,
):
    model = NextViT(
        **NEXTVIT_MODEL_CONFIG["NextViTLarge"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(NEXTVIT_WEIGHTS_CONFIG):
        load_weights_from_config("NextViTLarge", weights, model, NEXTVIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
