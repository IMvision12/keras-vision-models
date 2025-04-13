import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

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


def inverted_residual_block(
    x,
    expansion_ratio,
    filters,
    kernel_size,
    stride,
    se_ratio,
    activation,
    block_id,
    data_format,
    channels_axis,
):
    """A building block for MobileNetV3-style architectures using inverted residuals with optional squeeze-and-excitation.

    This block implements an inverted residual structure where the input is first expanded
    through a 1x1 convolution, then processed with a depthwise convolution, optionally
    enhanced with squeeze-and-excitation, and finally projected back to a smaller size
    through another 1x1 convolution. If input and output shapes match and stride is 1,
    a residual connection is added.

    Args:
        x: Input tensor.
        expansion_ratio: float, the expansion multiplier for the input channels in the first 1x1 conv.
            If 1, the expansion phase is skipped.
        filters: int, the number of output filters for the final pointwise convolution.
        kernel_size: int, the size of the depthwise convolution kernel.
        stride: int, the stride of the depthwise convolution. Must be either 1 or 2.
        se_ratio: float or None, squeeze-and-excitation ratio. If None, no SE is applied.
            If float, should be between 0 and 1, representing the ratio of channels to use
            in the SE block relative to the expanded filters.
        activation: str or callable, the activation function to use after expansions
            and depthwise convolutions.
        block_id: int, unique identifier for the block used in layer naming.
        data_format: str, either 'channels_first' or 'channels_last', specifies the
            input data format.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').

    Returns:
        Output tensor for the block. If stride=1 and input/output channels match,
        includes a residual connection.

    Notes:
        - The block follows MobileNetV3 architecture improvements including:
          * Optional squeeze-and-excitation
          * Hard sigmoid activation in SE blocks
          * Configurable activation functions
        - All convolution layers use same padding except when stride=2
        - Batch normalization uses epsilon=1e-3 and momentum=0.999
        - The make_divisible function should be imported separately to ensure channel
          counts are efficiently aligned with hardware
    """
    shortcut = x
    prefix = f"ir_block_{block_id}"
    input_filters = x.shape[channels_axis]
    expanded_filters = make_divisible(input_filters * expansion_ratio)

    if expansion_ratio != 1:
        x = layers.Conv2D(
            expanded_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{prefix}_conv_pw",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=f"{prefix}_batchnorm_1",
        )(x)
        x = layers.Activation(activation, name=f"{prefix}_activation_1")(x)

    if stride == 1:
        pad_h = pad_w = kernel_size // 2
        x = layers.ZeroPadding2D(data_format=data_format, padding=(pad_h, pad_w))(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=f"{prefix}_batchnorm_2",
    )(x)
    x = layers.Activation(activation, name=f"{prefix}_activation_2")(x)

    if se_ratio:
        x_se = layers.GlobalAveragePooling2D(
            keepdims=True, data_format=data_format, name=f"{prefix}_se_pool"
        )(x)
        x_se = layers.Conv2D(
            make_divisible(expanded_filters * se_ratio),
            kernel_size=1,
            padding="same",
            data_format=data_format,
            name=f"{prefix}_se_conv_1",
        )(x_se)
        x_se = layers.ReLU(name=f"{prefix}_se_activation_1")(x_se)
        x_se = layers.Conv2D(
            expanded_filters,
            kernel_size=1,
            padding="same",
            data_format=data_format,
            name=f"{prefix}_se_conv_2",
        )(x_se)
        x_se = layers.Activation("hard_sigmoid", name=f"{prefix}_se_activation_2")(x_se)
        x = layers.Multiply(name=f"{prefix}_se_multiply")([x, x_se])

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv_pwl",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=f"{prefix}_batchnorm_3",
    )(x)

    if stride == 1 and input_filters == filters:
        x = layers.Add(name=f"{prefix}_add")([shortcut, x])
    return x


@keras.saving.register_keras_serializable(package="kvmm")
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
            'inception' (default), 'imagenet', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options in keras-vision. Defaults to "imagenet".
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
        config="large",
        minimal=False,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="inception",
        weights="in1k",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        dropout_rate=0.2,
        name="MobileNetV3",
        **kwargs,
    ):
        if include_top and as_backbone:
            raise ValueError(
                "Cannot use `as_backbone=True` with `include_top=True`. "
                f"Received: as_backbone={as_backbone}, include_top={include_top}"
            )

        if config not in ["large", "small"]:
            raise ValueError(
                f"Invalid model type. Expected 'large' or 'small', got {config}"
            )

        if config == "small":
            default_config = [
                # [expansion_ratio, filters, kernel_size, stride, se_ratio, activation]
                [1, 16, 3, 2, 0.25, "relu"],
                [72.0 / 16, 24, 3, 2, None, "relu"],
                [88.0 / 24, 24, 3, 1, None, "relu"],
                [4, 40, 5, 2, 0.25, "hard_swish"],
                [6, 40, 5, 1, 0.25, "hard_swish"],
                [6, 40, 5, 1, 0.25, "hard_swish"],
                [3, 48, 5, 1, 0.25, "hard_swish"],
                [3, 48, 5, 1, 0.25, "hard_swish"],
                [6, 96, 5, 2, 0.25, "hard_swish"],
                [6, 96, 5, 1, 0.25, "hard_swish"],
                [6, 96, 5, 1, 0.25, "hard_swish"],
            ]
            head_channels = 1024
        else:
            default_config = [
                # [expansion_ratio, filters, kernel_size, stride, se_ratio, activation]
                [1, 16, 3, 1, None, "relu"],
                [4, 24, 3, 2, None, "relu"],
                [3, 24, 3, 1, None, "relu"],
                [3, 40, 5, 2, 0.25, "relu"],
                [3, 40, 5, 1, 0.25, "relu"],
                [3, 40, 5, 1, 0.25, "relu"],
                [6, 80, 3, 2, None, "hard_swish"],
                [2.5, 80, 3, 1, None, "hard_swish"],
                [2.3, 80, 3, 1, None, "hard_swish"],
                [2.3, 80, 3, 1, None, "hard_swish"],
                [6, 112, 3, 1, 0.25, "hard_swish"],
                [6, 112, 3, 1, 0.25, "hard_swish"],
                [6, 160, 5, 2, 0.25, "hard_swish"],
                [6, 160, 5, 1, 0.25, "hard_swish"],
                [6, 160, 5, 1, 0.25, "hard_swish"],
            ]
            head_channels = 1280

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

        x = layers.Conv2D(
            16,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format=data_format,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-3,
            momentum=0.999,
            name="stem_batchnorm",
        )(x)
        x = layers.Activation(
            "hard_swish" if not minimal else "relu", name="stem_activation"
        )(x)
        features.append(x)

        for idx, layer_config in enumerate(default_config):
            expansion_ratio, filters, kernel_size, stride, se_ratio, activation = (
                layer_config
            )

            if minimal:
                kernel_size = 3
                activation = "relu"
                se_ratio = None

            x = inverted_residual_block(
                x,
                expansion_ratio=expansion_ratio * depth_multiplier,
                filters=make_divisible(filters * width_multiplier),
                kernel_size=kernel_size,
                stride=stride,
                se_ratio=se_ratio,
                activation=activation,
                block_id=idx,
                data_format=data_format,
                channels_axis=channels_axis,
            )
            features.append(x)

        final_conv_head_channels = make_divisible(x.shape[channels_axis] * 6)

        x = layers.Conv2D(
            final_conv_head_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name="final_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-3,
            momentum=0.999,
            name="final_batchnorm",
        )(x)
        x = layers.Activation(
            "hard_swish" if not minimal else "relu", name="final_activation"
        )(x)
        features.append(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(
                data_format=data_format, keepdims=True, name="head_pool"
            )(x)
            x = layers.Conv2D(
                head_channels,
                kernel_size=1,
                padding="same",
                use_bias=True,
                data_format=data_format,
                name="head_conv",
            )(x)
            x = layers.Activation(activation, name="head_activation")(x)

            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name="head_dropout")(x)
            x = layers.Conv2D(
                num_classes,
                kernel_size=1,
                padding="same",
                data_format=data_format,
                name="predictions",
            )(x)
            x = layers.Flatten()(x)
            x = layers.Activation(
                activation=classifier_activation, name="predictions_act"
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

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

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
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
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
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@register_model
def MobileNetV3Small075(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="inception",
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
    normalization_mode="inception",
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
    normalization_mode="inception",
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
def MobileNetV3Large075(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="inception",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV3Large075",
    **kwargs,
):
    model = MobileNetV3(
        **MOBILENETV3_MODEL_CONFIG["MobileNetV3Large075"],
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
            "MobileNetV3Large075", weights, model, MOBILENETV3_WEIGHTS_CONFIG
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
    normalization_mode="inception",
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
    normalization_mode="inception",
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
