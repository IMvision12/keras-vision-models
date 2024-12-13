from typing import Optional

import keras
from keras import backend, layers
from keras.src.applications import imagenet_utils

from kv.utils import download_weights

from ...model_registry import register_model
from .config import RESNET_MODEL_CONFIG, RESNET_WEIGHTS_CONFIG


def conv_block(
    x: layers.Layer,
    filters: int,
    kernel_size: int,
    channels_axis,
    strides: int = 1,
    use_relu: bool = True,
    groups: int = 1,
    group_width: Optional[int] = None,
    name: Optional[str] = None,
    bn_name: Optional[str] = None,
) -> layers.Layer:
    """Applies a convolution block with optional grouped convolutions.

    Args:
        x: Input Keras layer.
        filters: Number of output filters for the convolution.
        kernel_size: Size of the convolution kernel.
        strides: Stride of the convolution.
        use_relu: Whether to apply ReLU activation after convolution.
        groups: Number of groups for grouped convolution.
        group_width: Width per group (used if groups > 1).
        name: Optional name for the convolution layer.
        bn_name: Optional name for the batch normalization layer.

    Returns:
       Output tensor for the block.
    """
    if isinstance(kernel_size, int):
        pad_h = pad_w = kernel_size // 2
    else:
        pad_h, pad_w = kernel_size[0] // 2, kernel_size[1] // 2

    if strides > 1:
        x = layers.ZeroPadding2D(padding=(pad_h, pad_w))(x)
        padding = "valid"
    else:
        padding = "same"

    if groups > 1:
        assert (
            filters % groups == 0
        ), f"Filters ({filters}) must be divisible by groups ({groups})"
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            groups=groups,
            kernel_initializer="he_normal",
            name=name,
        )(x)
    else:
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer="he_normal",
            name=name,
        )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=bn_name
    )(x)

    if use_relu:
        x = layers.ReLU()(x)
    return x


def squeeze_excitation_block(
    x: layers.Layer, reduction_ratio: int = 16, name: Optional[str] = None
) -> layers.Layer:
    """Applies a Squeeze-and-Excitation block for channel recalibration.

    Args:
        x: Input Keras layer.
        reduction_ratio: Reduction ratio for squeeze operation.
        name: Optional name for layers within the block.

    Returns:
        Output tensor for the block.
    """
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(
        filters // reduction_ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_dense1" if name else None,
    )(se)
    se = layers.Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_dense2" if name else None,
    )(se)
    return layers.Multiply(name=f"{name}_scale" if name else None)([x, se])


def bottleneck_block(
    x: layers.Layer,
    filters: int,
    channels_axis,
    strides: int = 1,
    downsample: bool = False,
    senet: bool = False,
    block_name: Optional[str] = None,
) -> layers.Layer:
    """Bottleneck ResNet block.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the bottleneck layers.
        strides: Stride for the main convolution layer.
        downsample: Whether to downsample the input.
        senet: Whether to apply SE block.
        block_name: Optional name for layers in the block.

    Returns:
        Output tensor for the block.
    """
    residual = x
    expansion = 4

    x = conv_block(
        x,
        filters,
        kernel_size=1,
        strides=1,
        name=f"{block_name}_conv1",
        bn_name=f"{block_name}_batchnorm1",
        channels_axis=channels_axis,
    )
    x = conv_block(
        x,
        filters,
        kernel_size=3,
        strides=strides,
        name=f"{block_name}_conv2",
        bn_name=f"{block_name}_batchnorm2",
        channels_axis=channels_axis,
    )
    x = conv_block(
        x,
        filters * expansion,
        kernel_size=1,
        use_relu=False,
        name=f"{block_name}_conv3",
        bn_name=f"{block_name}_batchnorm3",
        channels_axis=channels_axis,
    )

    if senet:
        x = squeeze_excitation_block(x, name=f"{block_name}.se")

    if downsample or strides != 1 or x.shape[-1] != residual.shape[-1]:
        residual = conv_block(
            residual,
            filters * expansion,
            kernel_size=1,
            strides=strides,
            use_relu=False,
            name=f"{block_name}_downsample_conv",
            bn_name=f"{block_name}_downsample_batchnorm",
            channels_axis=channels_axis,
        )

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    return x


def resnext_block(
    x: layers.Layer,
    filters: int,
    channels_axis,
    strides: int = 1,
    groups: int = 32,
    width_factor: int = 2,
    downsample: bool = False,
    senet: bool = False,
    block_name: Optional[str] = None,
) -> layers.Layer:
    """ResNeXt block with group convolutions.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the block.
        strides: Stride for the main convolution layer.
        groups: Number of groups for grouped convolution.
        width_factor: Factor to determine width for grouped convolution.
        downsample: Whether to downsample the input.
        senet: Whether to apply SE block.
        block_name: Optional name for layers in the block.

    Returns:
        Output tensor for the block.
    """
    residual = x
    expansion = 4
    width = filters * width_factor

    x = conv_block(
        x,
        width,
        kernel_size=1,
        strides=1,
        name=f"{block_name}_conv1",
        bn_name=f"{block_name}_batchnorm1",
        channels_axis=channels_axis,
    )
    group_width = width // groups
    x = conv_block(
        x,
        width,
        kernel_size=3,
        strides=strides,
        groups=groups,
        group_width=group_width,
        name=f"{block_name}_conv2",
        bn_name=f"{block_name}_batchnorm2",
        channels_axis=channels_axis,
    )
    x = conv_block(
        x,
        filters * expansion,
        kernel_size=1,
        use_relu=False,
        name=f"{block_name}_conv3",
        bn_name=f"{block_name}_batchnorm3",
        channels_axis=channels_axis,
    )

    if senet:
        x = squeeze_excitation_block(x, name=f"{block_name}_se")

    if downsample or strides != 1 or x.shape[-1] != residual.shape[-1]:
        residual = conv_block(
            residual,
            filters * expansion,
            kernel_size=1,
            strides=strides,
            use_relu=False,
            name=f"{block_name}_downsample_conv",
            bn_name=f"{block_name}_downsample_batchnorm",
            channels_axis=channels_axis,
        )

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    return x


@keras.saving.register_keras_serializable(package="kv")
class ResNet(keras.Model):
    """
    Instantiates the ResNet architecture with support for ResNeXt and SE-ResNet/SE-ResNeXt configurations.

    Reference:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016)

    Args:
        block_type: String, type of residual block to use. Options are
            `'basic'`, `'bottleneck'`, or `'resnext'`.
        block_repeats: List of integers, number of blocks to repeat at each stage.
        filters: List of integers, number of filters for each stage.
        groups: Integer, number of groups for group convolutions in ResNeXt blocks.
            Default is `32`.
        senet: Boolean, whether to include Squeeze-and-Excitation (SE) blocks for improved feature recalibration.
            Default is `False`.
        width_factor: Integer, scaling factor for the width of ResNeXt blocks.
            Default is `2`.
        include_top: Boolean, whether to include the fully-connected classification
            layer at the top. Defaults to `True`.
        weights: String, path to pretrained weights or one of the available
            options in `keras-vision`. Defaults to `'in1k'`.
        input_tensor: Optional Keras tensor to use as the model's input. If not provided,
            a new input tensor is created based on `input_shape`.
        input_shape: Optional tuple specifying the shape of the input data. If not
            specified, defaults to `(224, 224, 3)`.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification. Defaults to `1000`.
            Only applicable if `include_top=True`.
        classifier_activation: String or callable, activation function for the
            classifier layer. Set to `None` to return logits.
            Defaults to `"linear"`.
        name: String, the name of the model. Defaults to `"ResNet"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        block_type="bottleneck",
        block_repeats=[2, 2, 2, 2],
        filters=[64, 128, 256, 512],
        groups=32,
        senet=False,
        width_factor=2,
        include_top=True,
        weights="in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResNet",
        **kwargs,
    ):
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
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

        if block_type == "bottleneck":

            def block_fn(x, filters, strides=1, downsample=False, block_name=None):
                return bottleneck_block(
                    x,
                    filters,
                    strides=strides,
                    downsample=downsample,
                    senet=senet,
                    block_name=block_name,
                    channels_axis=channels_axis,
                )

        elif block_type == "resnext":

            def block_fn(x, filters, strides=1, downsample=False, block_name=None):
                return resnext_block(
                    x,
                    filters,
                    strides=strides,
                    groups=groups,
                    width_factor=width_factor,
                    downsample=downsample,
                    senet=senet,
                    block_name=block_name,
                    channels_axis=channels_axis,
                )

        else:
            raise ValueError(f"Unsupported block type: {block_type}")

        x = conv_block(
            inputs,
            filters[0],
            kernel_size=7,
            strides=2,
            name="conv1",
            bn_name="batchnorm1",
            channels_axis=channels_axis,
        )
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding="valid")(x)

        for i, num_blocks in enumerate(block_repeats):
            for j in range(num_blocks):
                block_name = f"resnet_layer{i+1}.{j}"
                if j == 0 and i > 0:
                    x = block_fn(
                        x, filters[i], strides=2, downsample=True, block_name=block_name
                    )
                else:
                    x = block_fn(x, filters[i], block_name=block_name)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                kernel_initializer="zeros",
                name="predictions",
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.block_type = block_type
        self.block_repeats = block_repeats
        self.filters = filters
        self.groups = groups
        self.senet = senet
        self.width_factor = width_factor
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "block_type": self.block_type,
            "block_repeats": self.block_repeats,
            "filters": self.filters,
            "groups": self.groups,
            "senet": self.senet,
            "width_factor": self.width_factor,
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


# ResNet Variants
@register_model
def ResNet50(
    include_top=True,
    weights="a1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["resnet50"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
    )
    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["ResNet50"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNet101(
    include_top=True,
    weights="a1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["resnet101"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["ResNet101"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNet152(
    include_top=True,
    weights="a1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["resnet152"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["ResNet152"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


# ResNeXt Variants
@register_model
def ResNeXt50_32x4d(
    include_top=True,
    weights="gluon_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["resnext50_32x4d"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights == "yfcc_ssl_in1k":
        weights = "fb_ssl_yfcc100m_ft_in1k"
    elif weights == "ig_swsl_in1k":
        weights = "fb_swsl_ig1b_ft_in1k"

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["ResNeXt50_32x4d"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNeXt101_32x4d(
    include_top=True,
    weights="gluon_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["resnext101_32x4d"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights == "yfcc_ssl_in1k":
        weights = "fb_ssl_yfcc100m_ft_in1k"
    elif weights == "ig_swsl_in1k":
        weights = "fb_swsl_ig1b_ft_in1k"

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["ResNeXt101_32x4d"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNeXt101_32x8d(
    include_top=True,
    weights="tv_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["resnext101_32x8d"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights == "yfcc_ssl_in1k":
        weights = "fb_ssl_yfcc100m_ft_in1k"
    elif weights == "ig_swsl_in1k":
        weights = "fb_swsl_ig1b_ft_in1k"
    elif weights == "ig_wsl_in1k":
        weights = "fb_wsl_ig1b_ft_in1k"

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["ResNeXt101_32x8d"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNeXt101_32x16d(
    include_top=True,
    weights="fb_wsl_ig1b_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["resnext101_32x16d"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights == "yfcc_ssl_in1k":
        weights = "fb_ssl_yfcc100m_ft_in1k"
    elif weights == "ig_swsl_in1k":
        weights = "fb_swsl_ig1b_ft_in1k"
    elif weights == "ig_wsl_in1k":
        weights = "fb_wsl_ig1b_ft_in1k"

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["ResNeXt101_32x16d"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNeXt101_32x32d(
    include_top=True,
    weights="ig_wsl_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["resnext101_32x32d"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights == "ig_wsl_in1k":
        weights = "fb_wsl_ig1b_ft_in1k"

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["ResNeXt101_32x32d"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


# SE-resent and SE-ResNext
@register_model
def SEResNet50(
    include_top=True,
    weights="a1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["seresnet50"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["SEResNet50"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def SEResNeXt50_32x4d(
    include_top=True,
    weights="gluon_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["seresnext50_32x4d"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["SEResNeXt50_32x4d"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def SEResNeXt101_32x4d(
    include_top=True,
    weights="gluon_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["seresnext101_32x4d"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["SEResNeXt101_32x4d"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model


@register_model
def SEResNeXt101_32x8d(
    include_top=True,
    weights="ah_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = ResNet(
        **RESNET_MODEL_CONFIG["seresnext101_32x8d"],
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights:
        weights_path = download_weights(
            RESNET_WEIGHTS_CONFIG["SEResNeXt101_32x8d"][weights]["url"]
        )
        model.load_weights(weights_path)
    else:
        print("No weights loaded.")

    return model
