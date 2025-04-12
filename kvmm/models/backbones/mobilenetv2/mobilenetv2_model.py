import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import MOBILENETV2_MODEL_CONFIG, MOBILENETV2_WEIGHTS_CONFIG


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
    filters,
    kernel_size,
    stride,
    expansion_ratio,
    channels_axis,
    data_format,
    block_id,
    sub_block_id,
):
    """A building block for MobileNetV2-style architectures using inverted residuals.

    Args:
        x: input tensor.
        filters: int, the number of output filters for the pointwise convolution.
        kernel_size: int, the size of the depthwise convolution kernel.
        stride: int, the stride of the depthwise convolution.
        expansion_ratio: float, the expansion factor applied to the input channels.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        block_id: int, unique identifier for the block.
        sub_block_id: int, unique identifier for the sub-block within the block.

    Returns:
        Output tensor for the block.
    """
    inputs = x
    block_name = f"blocks_{block_id}_{sub_block_id}"

    if expansion_ratio > 1:
        x = layers.Conv2D(
            make_divisible(x.shape[channels_axis] * expansion_ratio),
            1,
            1,
            use_bias=False,
            data_format=data_format,
            name=f"{block_name}_conv_pw",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name=f"{block_name}_batchnorm_1",
        )(x)
        x = layers.Activation("relu6", name=f"{block_name}_relu1")(x)

    if stride > 1:
        x = layers.ZeroPadding2D(
            padding=((1, 1), (1, 1)),
            data_format=data_format,
            name=f"{block_name}_zeropadding",
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size,
        stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{block_name}_batchnorm_2",
    )(x)
    x = layers.Activation("relu6", name=f"{block_name}_relu2")(x)

    x = layers.Conv2D(
        filters,
        1,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_conv_pwl",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{block_name}_batchnorm_3",
    )(x)

    if stride == 1 and inputs.shape[channels_axis] == filters:
        x = layers.Add(name=f"{block_name}_add")([inputs, x])

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class MobileNetV2(keras.Model):
    """Instantiates the MobileNetV2 architecture.

    Reference:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)

    Args:
        width_multiplier: Float, controls the width of the network by scaling the number
            of filters in each layer. Defaults to 1.0.
        depth_multiplier: Float, controls the depth of the network by scaling the number
            of blocks in each stage. Defaults to 1.0.
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
    """

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        fix_channels=False,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="ra_in1k",
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

        mobilenetv2_default_config = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

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

        initial_dims = 32 if fix_channels else make_divisible(32 * width_multiplier)
        x = layers.ZeroPadding2D(
            padding=((1, 1), (1, 1)), data_format=data_format, name="stem_padding"
        )(x)
        x = layers.Conv2D(
            initial_dims,
            3,
            2,
            padding="valid",
            use_bias=False,
            data_format=data_format,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name="stem_batchnorm",
        )(x)
        x = layers.Activation("relu6", name="relu1")(x)
        features.append(x)

        spatial_reduction = 2
        for layer_idx, layer_config in enumerate(mobilenetv2_default_config):
            expansion_factor, output_channels, num_blocks, initial_stride = layer_config
            scaled_output_channels = make_divisible(output_channels * width_multiplier)

            if layer_idx not in (0, len(mobilenetv2_default_config) - 1):
                num_blocks = int(keras.ops.ceil(num_blocks * depth_multiplier))

            for block_idx in range(num_blocks):
                current_stride = initial_stride if block_idx == 0 else 1

                x = inverted_residual_block(
                    x,
                    filters=scaled_output_channels,
                    kernel_size=3,
                    stride=current_stride,
                    expansion_ratio=expansion_factor,
                    channels_axis=channels_axis,
                    data_format=data_format,
                    block_id=layer_idx,
                    sub_block_id=block_idx,
                )
                spatial_reduction *= current_stride
            features.append(x)

        head_dims = (
            1280
            if fix_channels or width_multiplier <= 1.0
            else make_divisible(1280 * width_multiplier)
        )
        x = layers.Conv2D(
            head_dims,
            1,
            1,
            use_bias=False,
            data_format=data_format,
            name="head_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name="head_batchnorm",
        )(x)
        x = layers.Activation("relu6", name="relu2")(x)
        features.append(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.Dense(
                num_classes,
                use_bias=True,
                activation=classifier_activation,
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

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
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
                "width_multiplier": self.width_multiplier,
                "depth_multiplier": self.depth_multiplier,
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
def MobileNetV2WM50(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="lamb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV2WM50",
    **kwargs,
):
    model = MobileNetV2(
        **MOBILENETV2_MODEL_CONFIG["MobileNetV2WM50"],
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

    if weights in get_all_weight_names(MOBILENETV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV2WM50", weights, model, MOBILENETV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV2WM100(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ra_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV2WM100",
    **kwargs,
):
    model = MobileNetV2(
        **MOBILENETV2_MODEL_CONFIG["MobileNetV2WM100"],
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

    if weights in get_all_weight_names(MOBILENETV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV2WM100", weights, model, MOBILENETV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV2WM110(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ra_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV2WM110",
    **kwargs,
):
    model = MobileNetV2(
        **MOBILENETV2_MODEL_CONFIG["MobileNetV2WM110"],
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

    if weights in get_all_weight_names(MOBILENETV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV2WM110", weights, model, MOBILENETV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV2WM120(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ra_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV2WM120",
    **kwargs,
):
    model = MobileNetV2(
        **MOBILENETV2_MODEL_CONFIG["MobileNetV2WM120"],
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

    if weights in get_all_weight_names(MOBILENETV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV2WM120", weights, model, MOBILENETV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileNetV2WM140(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ra_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileNetV2WM140",
    **kwargs,
):
    model = MobileNetV2(
        **MOBILENETV2_MODEL_CONFIG["MobileNetV2WM140"],
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

    if weights in get_all_weight_names(MOBILENETV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileNetV2WM140", weights, model, MOBILENETV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
