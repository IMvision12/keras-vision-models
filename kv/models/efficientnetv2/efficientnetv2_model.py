import copy
import math

import keras
from keras import backend, initializers, layers
from keras.src.applications import imagenet_utils

from ...model_registry import register_model
from .config import (
    CONV_KERNEL_INITIALIZER,
    DENSE_KERNEL_INITIALIZER,
    EFFICIENTNETV2_BLOCK_CONFIG,
    EFFICIENTNETV2_MODEL_CONFIG,
)


def round_filters(filters, width_coefficient, min_depth=8, depth_divisor=8):
    """
    Rounds the number of filters based on the width coefficient, ensuring hardware efficiency.

    This function scales the number of filters according to the width coefficient and then rounds
    it to the nearest multiple of `depth_divisor` (default=8). If the rounded value is less than
    90% of the original scaled filters, it increments the rounded value by `depth_divisor`.

    Args:
        filters (int): The original number of filters (e.g., channels) before scaling.
        width_coefficient (float): A coefficient for scaling the network width (e.g., 1.2).
        min_depth (int, optional): The minimum number of filters allowed after rounding. Defaults to 8.
        depth_divisor (int, optional): The number to which the number of filters should be divisible. Defaults to 8.

    Returns:
        int: The rounded number of filters that is divisible by `depth_divisor`.

    Example:
        >>> round_filters(32, 1.2)  # Scales 32 filters by a width coefficient of 1.2
        40  # Rounded to the nearest multiple of 8
    """
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
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


def mb_conv_block(
    inputs,
    input_filters,
    output_filters,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    survival_probability=0.8,
    block_idx=0,
    layer_idx=0,
):
    """
    A Mobile Inverted Residual Block with optional Squeeze-and-Excitation (SE), depthwise convolution,
    and the use of survival probability for stochastic depth. This block is commonly used in architectures
    like MobileNetV2 for efficient computation.

    Args:
        inputs: The input tensor to the block.
        input_filters: The number of input channels to the block.
        output_filters: The number of output channels from the block.
        expand_ratio: The expansion ratio for the pointwise convolution. Default is 1.
        kernel_size: The size of the depthwise convolution kernel. Default is 3.
        strides: The stride of the depthwise convolution. Default is 1.
        se_ratio: The squeeze-and-excitation ratio for channel attention. Default is 0.0.
        survival_probability: Probability for using the skip connection (stochastic depth).
            Default is 0.8.
        block_idx: Index of the block in the model. Default is 0.
        layer_idx: Index of the layer in the block. Default is 0.

    Returns:
        Output tensor for the block.
    """
    channel_axis = 3 if backend.image_data_format() == "channels_last" else 1
    block_name = f"blocks_{block_idx}_{layer_idx}_"

    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format=backend.image_data_format(),
            use_bias=False,
            name=block_name + "MBconv1",
        )(inputs)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=0.9,
            name=block_name + "batchnorm1",
        )(x)
        x = layers.Activation("swish", name=block_name + "act1")(x)
    else:
        x = inputs

    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format=backend.image_data_format(),
        use_bias=False,
        name=block_name + "MBdwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, momentum=0.9, name=block_name + "batchnorm2"
    )(x)
    x = layers.Activation("swish", name=block_name + "act2")(x)

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D(name=block_name + "se_avgpool")(x)
        if channel_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = layers.Reshape(se_shape, name=block_name + "se_reshape")(se)

        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation="swish",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=block_name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=block_name + "se_conv_expand",
        )(se)

        x = layers.multiply([x, se], name=block_name + "se_excite")

    x = layers.Conv2D(
        filters=output_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format=backend.image_data_format(),
        use_bias=False,
        name=block_name + "MBconv2",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, momentum=0.9, name=block_name + "batchnorm3"
    )(x)

    if strides == 1 and input_filters == output_filters:
        if survival_probability:
            x = layers.Dropout(
                survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=block_name + "dropout",
            )(x)
        x = layers.add([x, inputs], name=block_name + "add")

    return x


def fusedmb_conv_block(
    inputs,
    input_filters,
    output_filters,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    survival_probability=0.8,
    block_idx=0,
    layer_idx=0,
):
    """
    A Fused Mobile Inverted Residual Block that combines pointwise convolution with depthwise
    convolution into a single fused operation. This block is optimized for efficiency and is
    typically used in architectures like MobileNetV3 to improve computation speed and reduce the number
    of parameters. It also includes optional Squeeze-and-Excitation (SE) and survival probability
    for stochastic depth.

    Args:
        inputs: The input tensor to the block.
        input_filters: The number of input channels to the block.
        output_filters: The number of output channels from the block.
        expand_ratio: The expansion ratio for the pointwise convolution. Default is 1.
        kernel_size: The size of the depthwise convolution kernel. Default is 3.
        strides: The stride of the depthwise convolution. Default is 1.
        se_ratio: The squeeze-and-excitation ratio for channel attention. Default is 0.0.
        survival_probability: Probability for using the skip connection (stochastic depth).
            Default is 0.8.
        block_idx: Index of the block in the model. Default is 0.
        layer_idx: Index of the layer in the block. Default is 0.

    Returns:
        Output tensor for the block.

    """
    channel_axis = 3 if backend.image_data_format() == "channels_last" else 1
    block_name = f"blocks_{block_idx}_{layer_idx}_"

    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=backend.image_data_format(),
            padding="same",
            use_bias=False,
            name=block_name + "FMBconv1",
        )(inputs)
        x = layers.BatchNormalization(
            axis=channel_axis, momentum=0.9, name=block_name + "batchnorm1"
        )(x)
        x = layers.Activation(activation="swish", name=block_name + "act1")(x)
    else:
        x = inputs

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D(name=block_name + "se_avgpool")(x)
        if channel_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)

        se = layers.Reshape(se_shape, name=block_name + "se_reshape")(se)

        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation="swish",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=block_name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=block_name + "se_conv_expand",
        )(se)

        x = layers.multiply([x, se], name=block_name + "se_excite")

    x = layers.Conv2D(
        output_filters,
        kernel_size=1 if expand_ratio != 1 else kernel_size,
        strides=1 if expand_ratio != 1 else strides,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        name=block_name + "FMBconv2",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, momentum=0.9, name=block_name + "batchnorm2"
    )(x)
    if expand_ratio == 1:
        x = layers.Activation(activation="swish", name=block_name + "act2")(x)

    if strides == 1 and input_filters == output_filters:
        if survival_probability:
            x = layers.Dropout(
                survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=block_name + "dropout",
            )(x)
        x = layers.add([x, inputs], name=block_name + "add")

    return x


@keras.saving.register_keras_serializable(package="kv")
class EfficientNetV2(keras.Model):
    """
    Instantiates the EfficientNetV2 architecture.

    Reference:
    - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

    Args:
        width_coefficient: Float, scaling coefficient for the network width
            (number of channels).
        depth_coefficient: Float, scaling coefficient for the network depth
            (number of layers).
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
            for the classification layer. Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, specifying the name of the model. Defaults to `"EfficientNetV2"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        width_coefficient,
        depth_coefficient,
        default_size,
        include_top=True,
        weights="imagenet",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="EfficientNetV2",
        **kwargs,
    ):
        if name.startswith("EfficientNetV2B"):
            block_config = EFFICIENTNETV2_BLOCK_CONFIG["EfficientNetV2B"]
        else:
            block_config = EFFICIENTNETV2_BLOCK_CONFIG[name]

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

        channel_axis = 3 if backend.image_data_format() == "channels_last" else 1

        inputs = img_input

        stem_filters = round_filters(
            filters=block_config[0]["input_filters"],
            width_coefficient=width_coefficient,
        )
        x = layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name="conv_stem",
        )(inputs)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=0.9,
            name="batchnorm1",
        )(x)
        x = layers.Activation("swish", name="act1")(x)

        block_config = copy.deepcopy(block_config)
        b = 0
        blocks = float(sum(args["num_repeat"] for args in block_config))

        for i, args in enumerate(block_config):
            assert args["num_repeat"] > 0

            args["input_filters"] = round_filters(
                filters=args["input_filters"],
                width_coefficient=width_coefficient,
            )
            args["output_filters"] = round_filters(
                filters=args["output_filters"],
                width_coefficient=width_coefficient,
            )

            block = {0: mb_conv_block, 1: fusedmb_conv_block}[args.pop("conv_type")]
            repeats = round_repeats(
                repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient
            )
            for j in range(repeats):
                if j > 0:
                    args["strides"] = 1
                    args["input_filters"] = args["output_filters"]

                x = block(
                    x,
                    survival_probability=0.2 * b / blocks,
                    block_idx=i,
                    layer_idx=j,
                    **args,
                )
                b += 1

        head_filters = {"EfficientNetV2B2": 1408, "EfficientNetV2B3": 1536}
        head_filter = head_filters.get(name, 1280)
        x = layers.Conv2D(
            filters=head_filter,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            use_bias=False,
            padding="same",
            name="conv_head",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=0.9,
            name="batchnorm2",
        )(x)
        x = layers.Activation(activation="swish", name="act2")(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dropout(0.2, name="top_dropout")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                bias_initializer=initializers.Constant(0.0),
                name="classifier",
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
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
            "include_top": self.include_top,
            "input_tensor": self.input_tensor,
            "input_shape": self.input_shape[1:],
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
def EfficientNetV2S(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetV2S",
    **kwargs,
):
    return EfficientNetV2(
        **EFFICIENTNETV2_MODEL_CONFIG["EfficientNetV2S"],
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


@register_model
def EfficientNetV2M(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetV2M",
    **kwargs,
):
    return EfficientNetV2(
        **EFFICIENTNETV2_MODEL_CONFIG["EfficientNetV2M"],
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


@register_model
def EfficientNetV2L(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetV2L",
    **kwargs,
):
    return EfficientNetV2(
        **EFFICIENTNETV2_MODEL_CONFIG["EfficientNetV2L"],
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


@register_model
def EfficientNetV2XL(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetV2XL",
    **kwargs,
):
    return EfficientNetV2(
        **EFFICIENTNETV2_MODEL_CONFIG["EfficientNetV2XL"],
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


# B variants
@register_model
def EfficientNetV2B0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetV2B0",
    **kwargs,
):
    return EfficientNetV2(
        **EFFICIENTNETV2_MODEL_CONFIG["EfficientNetV2B0"],
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


@register_model
def EfficientNetV2B1(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetV2B1",
    **kwargs,
):
    return EfficientNetV2(
        **EFFICIENTNETV2_MODEL_CONFIG["EfficientNetV2B1"],
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


@register_model
def EfficientNetV2B2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetV2B2",
    **kwargs,
):
    return EfficientNetV2(
        **EFFICIENTNETV2_MODEL_CONFIG["EfficientNetV2B2"],
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


@register_model
def EfficientNetV2B3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="EfficientNetV2B3",
    **kwargs,
):
    return EfficientNetV2(
        **EFFICIENTNETV2_MODEL_CONFIG["EfficientNetV2B3"],
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
