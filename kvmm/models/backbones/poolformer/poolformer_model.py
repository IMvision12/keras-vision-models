import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer, LayerScale, StochasticDepth
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import POOLFORMER_MODEL_CONFIG, POOLFORMER_WEIGHTS_CONFIG


def mlp_block(x, hidden_dim, embed_dim, drop_rate, data_format, name):
    """
    Implements a Multi-Layer Perceptron (MLP) block using 1x1 convolutions for vision models.

    The block consists of two 1x1 convolution layers with GELU activation and dropout
    regularization. This implementation maintains spatial dimensions and uses convolutions
    instead of dense layers, making it suitable for vision transformers and similar architectures.

    Args:
        x: Input tensor to the MLP block.
        hidden_dim: Number of filters in the first (hidden) convolution layer.
        embed_dim: Number of filters in the second (output) convolution layer.
        drop_rate: Dropout rate applied after each convolution layer.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        name: Base name for the layers in this block, used for layer naming.

    Returns:
        Output tensor after passing through the MLP block.
    """
    x = layers.Conv2D(
        filters=hidden_dim,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_1",
    )(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = layers.Dropout(drop_rate, name=f"{name}_drop_1")(x)

    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_2",
    )(x)
    x = layers.Dropout(drop_rate, name=f"{name}_drop_2")(x)

    return x


def poolformer_block(
    x,
    embed_dim,
    mlp_ratio,
    drop_rate,
    drop_path_rate,
    init_scale,
    data_format,
    channels_axis,
    name,
):
    """
    Implements a PoolFormer block that uses average pooling for token mixing instead of self-attention.

    The block consists of two main components:
    1. Token mixing layer using average pooling and subtraction
    2. Multi-Layer Perceptron (MLP) layer

    Both components use group normalization and residual connections with optional layer scaling
    and stochastic depth.

    Args:
        x: Input tensor to the PoolFormer block.
        embed_dim: Dimensionality of the input and output features.
        mlp_ratio: Expansion ratio for the hidden dimension in the MLP layer.
            Hidden layer size will be embed_dim * mlp_ratio.
        drop_rate: Dropout rate used in the MLP layer.
        drop_path_rate: Drop path rate for stochastic depth. If 0, stochastic depth
            is disabled.
        init_scale: Initial scaling factor for the LayerScale layers.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        name: Base name used for all layers in this block.

    Returns:
        Output tensor after passing through the PoolFormer block,
        with the same shape and dimensionality as the input.
    """
    shortcut = x

    x = layers.GroupNormalization(
        groups=1, axis=channels_axis, epsilon=1e-5, name=f"{name}_groupnorm_1"
    )(x)

    x_pool = layers.AveragePooling2D(
        pool_size=3, strides=1, padding="same", data_format=data_format
    )(x)
    x = layers.Subtract(name=f"{name}_token_mixer")([x_pool, x])

    layer_scale_1 = LayerScale(init_scale, name=f"{name}_layerscale_1")(x)

    if drop_path_rate > 0:
        layer_scale_1 = StochasticDepth(drop_path_rate)(layer_scale_1)

    x = layers.Add(name=f"{name}_add_1")([shortcut, layer_scale_1])

    shortcut = x
    x = layers.GroupNormalization(
        groups=1, axis=channels_axis, epsilon=1e-5, name=f"{name}_groupnorm_2"
    )(x)
    x = mlp_block(
        x,
        hidden_dim=int(embed_dim * mlp_ratio),
        embed_dim=embed_dim,
        drop_rate=drop_rate,
        data_format=data_format,
        name=f"{name}_mlp",
    )

    layer_scale_2 = LayerScale(init_scale, name=f"{name}_layerscale_2")(x)

    if drop_path_rate > 0:
        layer_scale_2 = StochasticDepth(drop_path_rate)(layer_scale_2)

    x = layers.Add(name=f"{name}_add_2")([shortcut, layer_scale_2])

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class PoolFormer(keras.Model):
    """Instantiates the PoolFormer architecture that uses pooling for token mixing.

    This implementation follows the PoolFormer design which replaces the self-attention
    mechanism in transformers with a simple pooling-based token mixer, providing an
    efficient alternative for vision tasks.

    Reference:
    - [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)

    Args:
        embed_dims: List of integers, dimensions for each stage's embedding.
        num_blocks: List of integers, number of blocks in each stage.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Defaults to `4.0`.
        drop_rate: Float, dropout rate applied to the MLP layers.
            Defaults to `0.0`.
        drop_path_rate: Float, stochastic depth rate.
            Defaults to `0.0`.
        init_scale: Float, initial scaling factor for LayerScale.
            Defaults to `1e-5`.
        include_top: Boolean, whether to include the classification head.
            Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to `False`.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options in `keras-vision`.
        input_shape: Optional tuple specifying the shape of the input data.
            Should be (height, width, channels).
        input_tensor: Optional Keras tensor as input.
            Useful for connecting the model to other Keras components.
        pooling: Optional pooling mode when `include_top=False`:
            - `None`: output is the last stage's output
            - `"avg"`: global average pooling is applied
            - `"max"`: global max pooling is applied
        num_classes: Integer, the number of output classes for classification.
            Defaults to `1000`.
        classifier_activation: String or callable, activation function for the top layer.
            Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"PoolFormer"`.
        **kwargs: Additional keyword arguments passed to the parent Model class.

    Returns:
        A Keras `Model` instance.

    Example:
        ```python
        # PoolFormer-S12
        model = PoolFormer(
            embed_dims=[64, 128, 320, 512],
            num_blocks=[2, 2, 6, 2],
            mlp_ratio=4.0,
            drop_path_rate=0.1
        )
        ```
    """

    def __init__(
        self,
        embed_dims,
        num_blocks,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_scale=1e-5,
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
        name="PoolFormer",
        **kwargs,
    ):
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
        features = []

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        x = layers.ZeroPadding2D(
            padding=((2, 2), (2, 2)), data_format=data_format, name="stem_pad"
        )(x)
        x = layers.Conv2D(
            filters=embed_dims[0],
            kernel_size=7,
            strides=4,
            use_bias=True,
            padding="valid",
            data_format=data_format,
            name="stem_conv",
        )(x)
        features.append(x)

        total_blocks = sum(num_blocks)
        dpr = [x * drop_path_rate / total_blocks for x in range(total_blocks)]
        cur = 0

        for stage_idx in range(len(num_blocks)):
            for block_idx in range(num_blocks[stage_idx]):
                x = poolformer_block(
                    x,
                    embed_dim=embed_dims[stage_idx],
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur],
                    init_scale=init_scale,
                    data_format=data_format,
                    channels_axis=channels_axis,
                    name=f"stage_{stage_idx}_block_{block_idx}",
                )
                cur += 1

            features.append(x)

            if stage_idx < len(num_blocks) - 1:
                x = layers.ZeroPadding2D(
                    padding=((1, 1), (1, 1)),
                    data_format=data_format,
                    name=f"stage_{stage_idx + 1}_downsample_pad",
                )(x)
                x = layers.Conv2D(
                    filters=embed_dims[stage_idx + 1],
                    kernel_size=3,
                    strides=2,
                    use_bias=True,
                    padding="valid",
                    data_format=data_format,
                    name=f"stage_{stage_idx + 1}_downsample_conv",
                )(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.LayerNormalization(epsilon=1e-6, name="layernorm")(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(x)
        elif as_backbone:
            x = features
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(
                    data_format=data_format, name="avg_pool"
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(data_format=data_format, name="max_pool")(
                    x
                )

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.embed_dims = embed_dims
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_scale = init_scale
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
                "embed_dims": self.embed_dims,
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "init_scale": self.init_scale,
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
def PoolFormerS12(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="PoolFormerS12",
    **kwargs,
):
    model = PoolFormer(
        **POOLFORMER_MODEL_CONFIG["PoolFormerS12"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(POOLFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "PoolFormerS12", weights, model, POOLFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PoolFormerS24(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="PoolFormerS24",
    **kwargs,
):
    model = PoolFormer(
        **POOLFORMER_MODEL_CONFIG["PoolFormerS24"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(POOLFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "PoolFormerS24", weights, model, POOLFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PoolFormerS36(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="PoolFormerS36",
    **kwargs,
):
    model = PoolFormer(
        **POOLFORMER_MODEL_CONFIG["PoolFormerS36"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(POOLFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "PoolFormerS36", weights, model, POOLFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PoolFormerM36(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="PoolFormerM36",
    **kwargs,
):
    model = PoolFormer(
        **POOLFORMER_MODEL_CONFIG["PoolFormerM36"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(POOLFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "PoolFormerM36", weights, model, POOLFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PoolFormerM48(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="PoolFormerM48",
    **kwargs,
):
    model = PoolFormer(
        **POOLFORMER_MODEL_CONFIG["PoolFormerM48"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(POOLFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "PoolFormerM48", weights, model, POOLFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
