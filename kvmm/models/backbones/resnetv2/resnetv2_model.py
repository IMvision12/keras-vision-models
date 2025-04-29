import keras
import numpy as np
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer, StdConv2D, StochasticDepth
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import RESNETV2_MODEL_CONFIG, RESNETV2_WEIGHTS_CONFIG


def make_divisible(v, divisor=8):
    """
    Returns a value that is divisible by the given divisor while staying close to the input value.
    This is typically used for ensuring channel counts are divisible by a specific number,
    which can be important for hardware efficiency.

    Args:
        v: The input value to make divisible.
        divisor: The number that the output should be divisible by (default=8).

    Returns:
        int: A number that is divisible by divisor and close to input v.
           If the divisible number is less than 90% of the input,
           the next divisible number is returned.

    Example:
        >>> make_divisible(24, 8)
        24
        >>> make_divisible(27, 8)
        32
        >>> make_divisible(5, 8)
        8
    """
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_block(
    x,
    filters,
    kernel_size,
    data_format,
    strides=1,
    padding="same",
    use_bias=False,
    name=None,
):
    """Applies a convolution block with option to use standard convolution.

    Args:
        x: Input Keras layer.
        filters: Number of output filters for the convolution.
        kernel_size: Size of the convolution kernel.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride of the convolution (default=1).
        padding: Type of padding to use, either 'same' or 'valid' (default='same').
            Note: When strides > 1, padding is automatically handled with ZeroPadding2D.
        use_bias: Boolean, whether to use bias in the convolution (default=False).
        name: Optional name for the convolution layer.

    Returns:
        Output tensor after applying convolution block.

    Notes:
        - When strides > 1, the function automatically applies zero padding of
          kernel_size // 2 and switches to 'valid' padding for the convolution.
    """
    if strides > 1:
        pad_h = pad_w = kernel_size // 2
        x = layers.ZeroPadding2D(padding=(pad_h, pad_w))(x)
        padding = "valid"

    x = StdConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        data_format=data_format,
        name=name,
    )(x)

    return x


def preact_bottleneck(
    x,
    filters,
    data_format,
    channels_axis,
    strides=1,
    downsample=False,
    drop_path_rate=0.0,
    block_prefix=None,
    bottleneck_ratio=0.25,
):
    """Pre-activation Bottleneck ResNet block with optional BatchNorm/GroupNorm.

    Args:
        x: Input Keras layer.
        filters: Number of output filters for the bottleneck layers.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        strides: int, default 1. Stride for the middle convolution layer.
        downsample: bool, default False. Whether to downsample the input.
        drop_path_rate: float, default 0.0. Drop path rate for stochastic depth.
        block_prefix: Optional string prefix for naming layers in the block.
        bottleneck_ratio: float, default 0.25. Ratio to determine middle channel dimensions.

    Returns:
        Output tensor for the pre-activation bottleneck block.

    The block implements a pre-activation bottleneck architecture with:
    - Optional downsampling of input
    - Three conv layers (1x1, 3x3, 1x1) with normalization and ReLU
    - Stochastic depth option for regularization
    - Residual connection
    """
    shortcut = x
    mid_channels = make_divisible(filters * bottleneck_ratio)

    preact = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_1"
    )(x)
    preact = layers.Activation("relu", name=f"{block_prefix}_relu_1")(preact)

    if downsample:
        shortcut = conv_block(
            preact,
            filters=filters,
            kernel_size=1,
            data_format=data_format,
            strides=strides,
            use_bias=False,
            name=f"{block_prefix}_downsample_conv",
        )

    x = conv_block(
        preact,
        filters=mid_channels,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=f"{block_prefix}_conv_1",
    )
    x = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_2"
    )(x)
    x = layers.Activation("relu", name=f"{block_prefix}_relu_2")(x)

    x = conv_block(
        x,
        filters=mid_channels,
        kernel_size=3,
        data_format=data_format,
        strides=strides,
        use_bias=False,
        name=f"{block_prefix}_conv_2",
    )
    x = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_3"
    )(x)
    x = layers.Activation("relu", name=f"{block_prefix}_relu_3")(x)

    x = conv_block(
        x,
        filters=filters,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=f"{block_prefix}_conv_3",
    )

    if drop_path_rate > 0:
        x = StochasticDepth(drop_path_rate)(x)

    x = layers.Add(name=f"{block_prefix}_add")([shortcut, x])
    return x


@keras.saving.register_keras_serializable(package="kvmm")
class ResNetV2(keras.Model):
    """
    Instantiates the ResNetV2 architecture with pre-activation residual blocks.
    Reference:
    - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

    Args:
        block_repeats: List of integers, number of blocks to repeat at each stage.
            Defaults to (2, 2, 2, 2).
        filters: List of integers, number of filters for each stage.
            Defaults to (256, 512, 1024, 2048).
        width_factor: Integer, scaling factor for the width of the network.
            Defaults to 1.
        stem_width: Integer, number of filters in the initial stem convolution.
            Defaults to 64.
        drop_rate: Float between 0 and 1, dropout rate for the final classifier layer.
            Defaults to 0.0.
        drop_path_rate: Float between 0 and 1, stochastic depth rate for randomly
            dropping residual connections. Defaults to 0.0.
        include_top: Boolean, whether to include the fully-connected classification
            layer at the top. Defaults to True.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to False.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to True.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options in keras-vision.
        input_tensor: Optional Keras tensor to use as the model's input. If not provided,
            a new input tensor is created based on input_shape.
        input_shape: Optional tuple specifying the shape of the input data. If not
            specified, defaults to (224, 224, 3).
        pooling: Optional pooling mode for feature extraction when include_top=False:
            - None (default): the output is the 4D tensor from the last convolutional block.
            - "avg": global average pooling is applied, and the output is a 2D tensor.
            - "max": global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to 1000. Only applicable if include_top=True.
        classifier_activation: String or callable, activation function for the
            classifier layer. Set to None to return logits.
            Defaults to "linear".
        name: String, the name of the model. Defaults to "resnetv2".
        **kwargs: Additional keyword arguments passed to the parent Model class.

    Returns:
        A Keras Model instance.
    """

    def __init__(
        self,
        block_repeats=(2, 2, 2, 2),
        filters=(256, 512, 1024, 2048),
        width_factor=1,
        stem_width=64,
        drop_rate=0.0,
        drop_path_rate=0.0,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResNetV2",
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
        channels_axis = -1 if data_format == "channels_last" else -3

        if weights and "448" in weights:
            default_size = 448
        elif weights and "480" in weights:
            default_size = 480
        else:
            default_size = 224

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_size,
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

        # Stem
        x = conv_block(
            x,
            filters=make_divisible(stem_width * width_factor),
            kernel_size=7,
            data_format=data_format,
            strides=2,
            use_bias=False,
            name="stem_conv",
        )

        x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
        x = layers.MaxPooling2D(
            pool_size=3,
            strides=2,
            data_format=data_format,
            padding="valid",
            name="stem_maxpool",
        )(x)
        features.append(x)

        dpr = list(np.linspace(0.0, drop_path_rate, sum(block_repeats)))
        block_idx = 0

        for stage_idx in range(len(block_repeats)):
            nb_channels_stage = make_divisible(filters[stage_idx] * width_factor)
            for block_idx_in_stage in range(block_repeats[stage_idx]):
                block_prefix = f"stages_{stage_idx}_blocks_{block_idx_in_stage}"
                x = preact_bottleneck(
                    x,
                    filters=nb_channels_stage,
                    data_format=data_format,
                    channels_axis=channels_axis,
                    strides=2 if (stage_idx > 0) and (block_idx_in_stage == 0) else 1,
                    downsample=block_idx_in_stage == 0,
                    drop_path_rate=dpr[block_idx],
                    block_prefix=block_prefix,
                )
                block_idx += 1
            features.append(x)

        x = layers.GroupNormalization(axis=channels_axis, name="groupnorm")(x)
        x = layers.Activation("relu", name="relu")(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            if drop_rate > 0:
                x = layers.Dropout(drop_rate, name="dropout")(x)
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

        self.block_repeats = block_repeats
        self.filters = filters
        self.width_factor = width_factor
        self.stem_width = stem_width
        self.drop_rate = drop_rate
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
                "block_repeats": self.block_repeats,
                "filters": self.filters,
                "width_factor": self.width_factor,
                "stem_width": self.stem_width,
                "drop_rate": self.drop_rate,
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
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def ResNetV2_50x1(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="goog_in21k_ft_in1k_448",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNetV2_50x1",
):
    model = ResNetV2(
        **RESNETV2_MODEL_CONFIG[name],
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
    )
    if weights in get_all_weight_names(RESNETV2_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, RESNETV2_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNetV2_50x3(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="goog_in21k_ft_in1k_448",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNetV2_50x3",
):
    model = ResNetV2(
        **RESNETV2_MODEL_CONFIG[name],
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
    )
    if weights in get_all_weight_names(RESNETV2_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, RESNETV2_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNetV2_101x1(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="goog_in21k_ft_in1k_448",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNetV2_101x1",
):
    model = ResNetV2(
        **RESNETV2_MODEL_CONFIG[name],
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
    )
    if weights in get_all_weight_names(RESNETV2_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, RESNETV2_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNetV2_101x3(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="goog_in21k_ft_in1k_448",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNetV2_101x3",
):
    model = ResNetV2(
        **RESNETV2_MODEL_CONFIG[name],
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
    )
    if weights in get_all_weight_names(RESNETV2_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, RESNETV2_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNetV2_152x2(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="goog_in21k_ft_in1k_448",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNetV2_152x2",
):
    model = ResNetV2(
        **RESNETV2_MODEL_CONFIG[name],
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
    )
    if weights in get_all_weight_names(RESNETV2_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, RESNETV2_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNetV2_152x4(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="goog_in21k_ft_in1k_480",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNetV2_152x4",
):
    model = ResNetV2(
        **RESNETV2_MODEL_CONFIG[name],
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
    )
    if weights in get_all_weight_names(RESNETV2_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, RESNETV2_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
