import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kv.layers import ImageNormalizationLayer
from kv.utils import get_all_weight_names, load_weights_from_config, register_model

from .config import MOBILENETV3_MODEL_CONFIG, MOBILENETV3_WEIGHTS_CONFIG


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """
    Adjusts the given value `v` to be divisible by `divisor`,
        ensuring it meets the specified constraints.

    Args:
        v (int or float): The value to be adjusted.
        divisor (int, optional): The divisor to which `v` should be rounded. Default is 8.
        min_value (int, optional): The minimum allowed value. If None, it defaults to `divisor`.
        round_limit (float, optional): The threshold to increase `new_v` if it is too small.
            Default is 0.9.

    Returns:
        int: The adjusted value that is divisible by `divisor` and meets the
            given constraints.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def depth_separation_block(
    inputs,
    filters,
    kernel_size,
    strides,
    se_ratio,
    activation,
    residual_connection,
    channels_axis,
    data_format,
    block_name,
    minimal=False,
):
    """A building block for MobileNetV3-style architectures using depth-separable convolutions.

    Args:
        inputs: Input tensor.
        filters: Integer, the number of output filters for the pointwise convolution.
        kernel_size: Integer, the size of the depthwise convolution kernel.
        strides: Integer, the stride of the depthwise convolution.
        se_ratio: Float, squeeze and excitation ratio. If 0 or None, no SE block is added.
        activation: String or callable, the activation function to use.
            Can be 'relu', 'h_swish', or a custom activation function.
        residual_connection: Boolean, whether to add a residual connection if input
            and output shapes match.
        channels_axis: Integer, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: String, either 'channels_last' or 'channels_first',
            specifies the input data format.
        block_name: String, unique identifier for the block used in layer names.
        minimal: Boolean, whether to use the minimal version of the block with reduced
            operators. Defaults to False.

    Returns:
        Output tensor for the block.

    The block consists of the following operations:
    1. Depthwise convolution
    2. Batch normalization and activation
    3. Squeeze-and-Excitation module (if se_ratio > 0)
    4. Pointwise convolution (1x1)
    5. Residual connection (if enabled and shapes match)

    When minimal=True, certain optimizations are applied to reduce computation while
    maintaining similar accuracy.
    """
    x = inputs

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_dwconv",
    )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-3, name=f"{block_name}_batchnorm_1"
    )(x)

    if activation is not None:
        x = layers.Activation(activation, name=f"{block_name}_act")(x)

    if se_ratio > 0 and not minimal:
        se_input_channels = x.shape[channels_axis]
        se_channels = make_divisible(se_input_channels * se_ratio, 8)

        se = layers.GlobalAveragePooling2D(
            data_format=data_format, keepdims=True, name=f"{block_name}_se_pool"
        )(x)

        se = layers.Conv2D(
            se_channels,
            1,
            use_bias=True,
            data_format=data_format,
            name=f"{block_name}_se_conv_reduce",
        )(se)

        se = layers.Activation("relu", name=f"{block_name}_se_act_1")(se)

        se = layers.Conv2D(
            se_input_channels,
            1,
            use_bias=True,
            data_format=data_format,
            name=f"{block_name}_se_conv_expand",
        )(se)

        se = layers.Activation("hard_sigmoid", name=f"{block_name}_se_act_2")(se)

        x = layers.Multiply(name=f"{block_name}_multiply")([x, se])

    x = layers.Conv2D(
        filters,
        1,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_conv_pw",
    )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-3, name=f"{block_name}_batchnorm_2"
    )(x)

    if residual_connection:
        x = layers.Add(name=f"{block_name}_add")([x, inputs])

    return x


def inverted_residual_block(
    inputs,
    filters,
    kernel_size,
    strides,
    expansion_ratio,
    se_ratio,
    activation,
    channels_axis,
    data_format,
    block_name,
    minimal=False,
):
    """A building block for MobileNetV3-style architectures using inverted residuals with squeeze-excitation.

    Args:
        inputs: Input tensor.
        filters: Integer, the number of output filters for the pointwise convolution.
        kernel_size: Integer, the size of the depthwise convolution kernel.
        strides: Integer, the stride of the depthwise convolution.
        expansion_ratio: Float, the expansion factor applied to the input channels.
        se_ratio: Float, squeeze and excitation ratio. If 0 or None, no SE block is added.
        activation: String or callable, the activation function to use.
            Can be 'relu', 'h_swish', or a custom activation function.
        channels_axis: Integer, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: String, either 'channels_last' or 'channels_first',
            specifies the input data format.
        block_name: String, unique identifier for the block used in layer names.
        minimal: Boolean, whether to use the minimal version of the block with reduced
            operators. Defaults to False.

    Returns:
        Output tensor for the block.

    The block consists of the following operations:
    1. Expansion convolution (1x1)
    2. Depthwise convolution
    3. Squeeze-and-Excitation module (if se_ratio > 0)
    4. Projection convolution (1x1)

    When minimal=True, certain optimizations are applied to reduce computation while
    maintaining similar accuracy.
    """
    residual_connection = strides == 1 and inputs.shape[channels_axis] == filters
    x = inputs

    x = layers.Conv2D(
        make_divisible(inputs.shape[channels_axis] * expansion_ratio),
        1,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_conv_pw",
    )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-3, name=f"{block_name}_batchnorm_1"
    )(x)

    if activation is not None:
        x = layers.Activation(activation)(x)

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_dwconv",
    )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-3, name=f"{block_name}_batchnorm_2"
    )(x)

    if activation is not None:
        x = layers.Activation(activation, name=f"{block_name}_act")(x)

    if se_ratio > 0 and not minimal:
        se = layers.GlobalAveragePooling2D(
            data_format=data_format, keepdims=True, name=f"{block_name}_se_pool"
        )(x)

        se = layers.Conv2D(
            make_divisible(x.shape[channels_axis] * se_ratio, 8),
            1,
            use_bias=True,
            data_format=data_format,
            name=f"{block_name}_se_conv_reduce",
        )(se)

        se = layers.Activation("relu", name=f"{block_name}_se_act_1")(se)

        se = layers.Conv2D(
            x.shape[channels_axis],
            1,
            use_bias=True,
            data_format=data_format,
            name=f"{block_name}_se_conv_expand",
        )(se)

        se = layers.Activation("hard_sigmoid", name=f"{block_name}_se_act_2")(se)

        x = layers.Multiply(name=f"{block_name}_multiply")([x, se])

    x = layers.Conv2D(
        filters,
        1,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_conv_pwl",
    )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-3, name=f"{block_name}_batchnorm_3"
    )(x)

    if residual_connection:
        x = layers.Add(name=f"{block_name}_add")([x, inputs])

    return x


@keras.saving.register_keras_serializable(package="kv")
class MobileNetV3(keras.Model):
    """Instantiates the MobileNetV3 architecture.

    Reference:
    - [Searching for MobileNetV3](
        https://arxiv.org/abs/1905.02244) (ICCV 2019)

    Args:
        width_multiplier: Float, controls the width of the network by scaling the number
            of filters in each layer. Defaults to 1.0.
        depth_multiplier: Float, controls the depth of the network by scaling the number
            of blocks in each stage. Defaults to 1.0.
        config: String, specifies the model configuration to use. Must be one of:
            'small' (default) or 'large'.
        minimal: Boolean, whether to use the minimal version of the network with reduced
            operators. Defaults to False.
        include_top: Boolean, whether to include the classification head at the top
            of the network. Defaults to True.
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
            available options in keras-vision. Defaults to "in1k".
        input_tensor: Optional Keras tensor (output of layers.Input()) to use as
            the model's input. If not provided, a new input tensor is created based
            on input_shape.
        input_shape: Optional tuple specifying the shape of the input data. If not
            specified, it defaults to (224, 224, 3) when include_top=True.
        pooling: Optional pooling mode for feature extraction when include_top=False:
            - None (default): the output is the 4D tensor from the last convolutional block.
            - "avg": global average pooling is applied, and the output is a 2D tensor.
            - "max": global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to 1000.
        classifier_activation: String or callable, activation function for the top
            layer. Set to None to return logits. Defaults to "softmax".
        name: String, the name of the model. Defaults to "MobileNetV2".

    Returns:
        A Keras Model instance.

    The MobileNetV3 architecture introduces several improvements over MobileNetV2:
    - Network architecture search (NAS) for optimized blocks
    - Squeeze-and-Excitation modules for channel-wise attention
    - New activation functions (h-swish)
    - Platform-aware NAS for optimized inference
    """

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        config="small",
        minimal=False,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="in1k",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="MobileNetV2",
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

        if config == "small":
            default_config = [
                [["depthwise_separable", 1, 3, 2, 1.0, 16, 0.25]],
                [
                    ["inverted_residual", 1, 3, 2, 4.5, 24, 0.0],
                    ["inverted_residual", 1, 3, 1, 3.67, 24, 0.0],
                ],
                [
                    ["inverted_residual", 1, 5, 2, 4.0, 40, 0.25],
                    ["inverted_residual", 2, 5, 1, 6.0, 40, 0.25],
                ],
                [["inverted_residual", 2, 5, 1, 3.0, 48, 0.25]],
                [["inverted_residual", 3, 5, 2, 6.0, 96, 0.25]],
                [["conv_normal", 1, 1, 1, 1.0, 576, 0.0]],
            ]
            head_channels = 1024
        elif config == "large":
            default_config = [
                [["depthwise_separable", 1, 3, 1, 1.0, 16, 0.0]],
                [
                    ["inverted_residual", 1, 3, 2, 4.0, 24, 0.0],
                    ["inverted_residual", 1, 3, 1, 3.0, 24, 0.0],
                ],
                [["inverted_residual", 3, 5, 2, 3.0, 40, 0.25]],
                [
                    ["inverted_residual", 1, 3, 2, 6.0, 80, 0.0],
                    ["inverted_residual", 1, 3, 1, 2.5, 80, 0.0],
                    ["inverted_residual", 2, 3, 1, 2.3, 80, 0.0],
                ],
                [["inverted_residual", 2, 3, 1, 6.0, 112, 0.25]],
                [["inverted_residual", 3, 5, 2, 6.0, 160, 0.25]],
                [["conv_normal", 1, 1, 1, 1.0, 960, 0.0]],
            ]
            head_channels = 1280

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        stem_channels = make_divisible(16 * width_multiplier)
        x = layers.Conv2D(
            filters=stem_channels,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=-1, momentum=0.9, epsilon=1e-3, name="stem_batchnorm"
        )(x)
        x = layers.Activation("hard_swish" if not minimal else "relu", name="stem_act")(
            x
        )
        features.append(x)

        current_stride = 2
        for stage_index, stage_config in enumerate(default_config):
            for block_index, block_config in enumerate(stage_config):
                (
                    block_type,
                    repeats,
                    kernel_size,
                    stride,
                    expansion_ratio,
                    output_channels,
                    squeeze_ratio,
                ) = block_config

                if minimal:
                    activation_type = "relu"
                    if kernel_size > 3:
                        kernel_size = 3
                else:
                    activation_type = "relu" if stage_index < 2 else "hard_swish"
                    if block_type == "conv_normal":
                        activation_type = "hard_swish"

                output_channels = make_divisible(output_channels * width_multiplier)
                if block_index not in (0, len(default_config) - 1):
                    repeats = int(ops.ceil(repeats * depth_multiplier))
                for layer_index in range(repeats):
                    current_stride = stride if layer_index == 0 else 1
                    block_name = f"blocks.{stage_index}.{block_index + layer_index}"

                    if block_type == "depthwise_separable":
                        residual_connection_connection = (
                            x.shape[channels_axis] == output_channels
                            and current_stride == 1
                        )
                        x = depth_separation_block(
                            inputs=x,
                            filters=output_channels,
                            kernel_size=kernel_size,
                            strides=current_stride,
                            se_ratio=squeeze_ratio if not minimal else 0.0,
                            activation=activation_type,
                            residual_connection=residual_connection_connection,
                            channels_axis=channels_axis,
                            data_format=data_format,
                            block_name=block_name,
                            minimal=minimal,
                        )
                    elif block_type == "inverted_residual":
                        x = inverted_residual_block(
                            inputs=x,
                            filters=output_channels,
                            kernel_size=kernel_size,
                            strides=current_stride,
                            expansion_ratio=expansion_ratio,
                            se_ratio=squeeze_ratio if not minimal else 0.0,
                            activation=activation_type,
                            channels_axis=channels_axis,
                            data_format=data_format,
                            block_name=block_name,
                            minimal=minimal,
                        )
                    elif block_type == "conv_normal":
                        x = layers.Conv2D(
                            filters=output_channels,
                            kernel_size=kernel_size,
                            strides=current_stride,
                            padding="same",
                            use_bias=False,
                            data_format=data_format,
                            name=f"{block_name}_conv",
                        )(x)
                        x = layers.BatchNormalization(
                            axis=-1,
                            momentum=0.9,
                            epsilon=1e-3,
                            name=f"{block_name}_batchnorm_1",
                        )(x)
                        x = layers.Activation(activation_type)(x)
                    current_stride *= current_stride
            features.append(x)

        if include_top:
            head_channels = max(
                head_channels, make_divisible(head_channels * width_multiplier)
            )
            head_activation = "relu" if minimal else "hard_swish"
            x = layers.GlobalAveragePooling2D(
                name="avg_pool", keepdims=True, data_format=data_format
            )(x)
            x = layers.Conv2D(
                head_channels,
                1,
                1,
                use_bias=True,
                data_format=data_format,
                name="head_conv",
            )(x)
            x = layers.Activation(head_activation, name="final_act")(x)
            x = layers.Flatten()(x)
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

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.config = config
        self.minimal = minimal
        self.include_top = include_top
        self.as_backbone = as_backbone
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "width_multiplier": self.width_multiplier,
            "depth_multiplier": self.depth_multiplier,
            "config": self.config,
            "minimal": self.minimal,
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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def MobileNetV3Small075(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV3Small075",
    **kwargs,
):
    model = MobileNetV3(
        **MOBILENETV3_MODEL_CONFIG["MobileNetV3Small075"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MOBILENETV3_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV3Small075", weights, model, MOBILENETV3_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV3Small100(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV3Small100",
    **kwargs,
):
    model = MobileNetV3(
        **MOBILENETV3_MODEL_CONFIG["MobileNetV3Small100"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MOBILENETV3_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV3Small100", weights, model, MOBILENETV3_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV3SmallMinimal100(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV3SmallMinimal100",
    **kwargs,
):
    model = MobileNetV3(
        **MOBILENETV3_MODEL_CONFIG["MobileNetV3SmallMinimal100"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MOBILENETV3_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV3SmallMinimal100", weights, model, MOBILENETV3_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV3Large75(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV3Large75",
    **kwargs,
):
    model = MobileNetV3(
        **MOBILENETV3_MODEL_CONFIG["MobileNetV3Large75"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MOBILENETV3_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV3Large75", weights, model, MOBILENETV3_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV3Large100(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV3Large100",
    **kwargs,
):
    model = MobileNetV3(
        **MOBILENETV3_MODEL_CONFIG["MobileNetV3Large100"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MOBILENETV3_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV3Large100", weights, model, MOBILENETV3_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV3LargeMinimal100(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV3LargeMinimal100",
    **kwargs,
):
    model = MobileNetV3(
        **MOBILENETV3_MODEL_CONFIG["MobileNetV3LargeMinimal100"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MOBILENETV3_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV3LargeMinimal100", weights, model, MOBILENETV3_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
