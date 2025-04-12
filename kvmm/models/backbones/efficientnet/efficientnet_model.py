import copy
import math

import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import (
    CONV_KERNEL_INITIALIZER,
    DEFAULT_BLOCKS_ARGS,
    DENSE_KERNEL_INITIALIZER,
    EFFICIENTNET_MODEL_CONFIG,
    EFFICIENTNET_WEIGHTS_CONFIG,
)


def round_filters(filters, width_coefficient, divisor=8):
    """
    Rounds number of filters based on width coefficient according to EfficientNet scaling.

    This function calculates the scaled number of filters and ensures it is divisible
    by the divisor (default=8) for hardware efficiency. If the rounded value is less than
    90% of the scaled filters, it adds one more divisor unit.

    Args:
        filters (int): The original number of filters/channels
        width_coefficient (float): The coefficient for scaling network width (typically > 1.0)
        divisor (int, optional): Ensures the filters are divisible by this number. Defaults to 8.

    Returns:
        int: The rounded number of filters that is divisible by divisor

    Example:
        >>> round_filters(32, 1.2)  # Scale 32 filters by 1.2x
        40  # Rounded to nearest multiple of 8
    """
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """
    Rounds number of repeats based on depth coefficient according to EfficientNet scaling.

    This function calculates the number of repeated layers in a block after applying
    the depth scaling factor. The result is always rounded up to ensure sufficient
    network depth.

    Args:
        repeats (int): The original number of layer repetitions
        depth_coefficient (float): The coefficient for scaling network depth

    Returns:
        int: The rounded number of repeats after scaling

    Example:
        >>> round_repeats(3, 1.2)  # Scale 3 repeats by 1.2x
        4  # Rounded up from 3.6
    """
    return int(math.ceil(depth_coefficient * repeats))


def efficientnet_block(
    inputs,
    channels_axis,
    data_format,
    drop_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
):
    """
    Implements a mobile inverted residual block with squeeze-and-excitation,
    serving as the core building block of the EfficientNet architecture.
    The block includes expansion, depthwise convolution, optional SE, and projection phases.

    Args:
        inputs: Input tensor to the block.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        drop_rate: Dropout rate applied before the residual connection. Default is 0.0.
        name: Base name for all layers in the block. Default is "".
        filters_in: Number of input channels to the block. Default is 32.
        filters_out: Number of output channels from the block. Default is 16.
        kernel_size: Size of the depthwise convolution kernel. Default is 3.
        strides: Stride size for the depthwise convolution. Default is 1.
        expand_ratio: Channel expansion ratio for the MBConv block. Default is 1.
        se_ratio: Squeeze-and-excitation ratio, determining the bottleneck size. Default is 0.0.
        id_skip: Whether to include a residual connection. Default is True.

    Returns:
        Output tensor for the block.

    """
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=name + "conv2d_1",
        )(inputs)
        x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_1")(x)
        x = layers.Activation("swish")(x)
    else:
        x = inputs

    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            data_format=data_format,
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name=name + "dwconv2d",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_2")(x)
    x = layers.Activation("swish")(x)

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(data_format=data_format)(x)
        if channels_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = layers.Reshape(se_shape)(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation="swish",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=name + "se_conv_expand",
        )(se)
        x = layers.multiply([x, se])

    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name=name + "conv2d_2",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_3")(x)

    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)
        x = layers.add([x, inputs])
    return x


@keras.saving.register_keras_serializable(package="kvmm")
class EfficientNet(keras.Model):
    """
    Instantiates the EfficientNet architecture.

    Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

    Args:
        width_coefficient: Float, scaling coefficient for the network width
            (number of channels).
        depth_coefficient: Float, scaling coefficient for the network depth
            (number of layers).
        dropout_rate: Float, dropout rate used in the final classification layer.
        default_size: Integer, default resolution of input images.
        include_top: Boolean, whether to include the classification head at the
            top of the network. Defaults to `True`.
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
        input_tensor: Optional Keras tensor (output of `layers.Input()`) to use
            as the model's input. If not provided, a new input tensor is created
            based on `input_shape`.
        input_shape: Optional tuple specifying the shape of the input data. If
            not specified, it is derived from `default_size`. Typically defaults
            to `(default_size, default_size, 3)`.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, specifying the number of output classes for classification.
            Defaults to `1000`. Only applicable if `include_top=True`.
        classifier_activation: String or callable, specifying the activation function
            for the classification layer. Set to `None` to return logits. Defaults to `"linear"`.
        name: String, specifying the name of the model. Defaults to `"EfficientNet"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        width_coefficient,
        depth_coefficient,
        dropout_rate,
        default_size,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="ink1",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="EfficientNet",
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

        if weights and "475" in weights:
            default_img_size = 475
        else:
            default_img_size = default_size

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_img_size,
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
            padding=imagenet_utils.correct_pad(inputs, 3), data_format=data_format
        )(x)
        x = layers.Conv2D(
            round_filters(32, width_coefficient=width_coefficient),
            3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name="conv_stem",
        )(x)
        x = layers.BatchNormalization(axis=channels_axis, name="batchnorm_1")(x)
        x = layers.Activation("swish")(x)
        features.append(x)

        b = 0
        blocks = float(
            sum(
                round_repeats(args["repeats"], depth_coefficient=depth_coefficient)
                for args in DEFAULT_BLOCKS_ARGS
            )
        )

        for i, block_args in enumerate(DEFAULT_BLOCKS_ARGS):
            assert block_args["repeats"] > 0
            args = copy.deepcopy(block_args)

            args["filters_in"] = round_filters(
                args["filters_in"], width_coefficient=width_coefficient
            )
            args["filters_out"] = round_filters(
                args["filters_out"], width_coefficient=width_coefficient
            )

            repeats = round_repeats(
                args["repeats"], depth_coefficient=depth_coefficient
            )
            del args["repeats"]

            for j in range(repeats):
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]

                x = efficientnet_block(
                    x,
                    channels_axis,
                    data_format,
                    dropout_rate * b / blocks,
                    name=f"blocks_{i}_{j}_",
                    **args,
                )
                b += 1

            features.append(x)

        x = layers.Conv2D(
            round_filters(1280, width_coefficient=width_coefficient),
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name="conv_head",
        )(x)
        x = layers.BatchNormalization(axis=channels_axis, name="batchnorm_2")(x)
        x = layers.Activation("swish")(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name="dropout")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                name="predictions",
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

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
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
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "dropout_rate": self.dropout_rate,
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
def EfficientNetB0(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB0",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB0"],
        name=name,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB0", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetB1(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB1",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB1"],
        name=name,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB1", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetB2(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB2",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB2"],
        name=name,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB2", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetB3(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB3",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB3"],
        name=name,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB3", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetB4(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB4",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB4"],
        name=name,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB4", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetB5(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB5",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB5"],
        name=name,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB5", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetB6(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB6",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB6"],
        name=name,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB6", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetB7(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB7",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB7"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB7", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetB8(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ap_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetB8",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetB8"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB8", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetL2(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ns_jft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetL2",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_MODEL_CONFIG["EfficientNetL2"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetL2", weights, model, EFFICIENTNET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
