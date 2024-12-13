import copy
import math

import keras
from keras import backend, layers
from keras.src.applications import imagenet_utils

from ...model_registry import register_model
from .config import (
    CONV_KERNEL_INITIALIZER,
    DEFAULT_BLOCKS_ARGS,
    DENSE_KERNEL_INITIALIZER,
    EFFICIENTNET_MODEL_CONFIG,
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
    channels_axis = 3 if backend.image_data_format() == "channels_last" else 1
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "conv2d_1",
        )(inputs)
        x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_1")(x)
        x = layers.Activation("swish")(x)
    else:
        x = inputs

    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
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
        name=name + "dwconv2d",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_2")(x)
    x = layers.Activation("swish")(x)

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D()(x)
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
            name=name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_conv_expand",
        )(se)
        x = layers.multiply([x, se])

    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
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


@keras.saving.register_keras_serializable(package="kv")
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
        weights="ink1",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="EfficientNet",
        **kwargs,
    ):
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_size,
            min_size=32,
            data_format=backend.image_data_format(),
            require_flatten=include_top,
            weights=weights,
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input
        channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

        x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(inputs, 3))(inputs)
        x = layers.Conv2D(
            round_filters(32, width_coefficient=width_coefficient),
            3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="conv_stem",
        )(x)
        x = layers.BatchNormalization(axis=channels_axis, name="batchnorm_1")(x)
        x = layers.Activation("swish")(x)

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
                    dropout_rate * b / blocks,
                    name=f"blocks_{i}_{j}_",
                    **args,
                )
                b += 1

        # Build top
        x = layers.Conv2D(
            round_filters(1280, width_coefficient=width_coefficient),
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="conv_head",
        )(x)
        x = layers.BatchNormalization(axis=channels_axis, name="batchnorm_2")(x)
        x = layers.Activation("swish")(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name="dropout")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                name="predictions",
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "width_coefficient": self.width_coefficient,
            "depth_coefficient": self.depth_coefficient,
            "default_size": self.default_size,
            "dropout_rate": self.dropout_rate,
            "include_top": self.include_top,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def EfficientNetB0(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    return model


@register_model
def EfficientNetB1(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    return model


@register_model
def EfficientNetB2(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    return model


@register_model
def EfficientNetB3(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    return model


@register_model
def EfficientNetB4(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    return model


@register_model
def EfficientNetB5(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    return model


@register_model
def EfficientNetB6(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    return model


@register_model
def EfficientNetB7(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )
    return model


@register_model
def EfficientNetB8(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )
    return model


@register_model
def EfficientNetL2(
    include_top=True,
    weights="ink1",
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
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )
    return model
