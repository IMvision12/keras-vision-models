from typing import Optional

import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import RESNET_MODEL_CONFIG, RESNET_WEIGHTS_CONFIG


def conv_block(
    x: layers.Layer,
    filters: int,
    kernel_size: int,
    channels_axis,
    data_format,
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
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride of the convolution.
        use_relu: Whether to apply ReLU activation after convolution.
        groups: Number of groups for grouped convolution.
        group_width: Width per group (used if groups > 1).
        name: Optional name for the convolution layer.
        bn_name: Optional name for the batch normalization layer.

    Returns:
       Output tensor for the block.
    """
    pad_h = pad_w = kernel_size // 2

    if strides > 1:
        x = layers.ZeroPadding2D(data_format=data_format, padding=(pad_h, pad_w))(x)
        padding = "valid"
    else:
        padding = "same"

    if groups > 1:
        assert filters % groups == 0, (
            f"Filters ({filters}) must be divisible by groups ({groups})"
        )
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            groups=groups,
            kernel_initializer="he_normal",
            data_format=data_format,
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
            data_format=data_format,
            name=name,
        )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=bn_name
    )(x)

    if use_relu:
        x = layers.ReLU()(x)
    return x


def squeeze_excitation_block(
    x: layers.Layer, data_format, reduction_ratio: int = 16, name: Optional[str] = None
) -> layers.Layer:
    """
    Squeeze-and-Excitation block that properly handles both channels_first and channels_last formats.

    Args:
        x: Input tensor
        data_format: String, either 'channels_first' or 'channels_last'
        reduction_ratio: Integer, reduction ratio for the bottleneck
        name: String, optional name prefix for layers

    Returns:
        Tensor with same shape as input after applying SE attention
    """
    if data_format == "channels_first":
        channel_axis = 1
        filters = x.shape[channel_axis]
    else:
        channel_axis = -1
        filters = x.shape[channel_axis]

    se = layers.GlobalAveragePooling2D(data_format=data_format)(x)

    if data_format == "channels_first":
        se = layers.Reshape((filters, 1, 1))(se)
    else:
        se = layers.Reshape((1, 1, filters))(se)

    reduced_filters = max(1, filters // reduction_ratio)
    se = layers.Reshape((filters,))(se)
    se = layers.Dense(
        reduced_filters,
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

    if data_format == "channels_first":
        se = layers.Reshape((filters, 1, 1))(se)
    else:
        se = layers.Reshape((1, 1, filters))(se)

    return layers.Multiply(name=f"{name}_scale" if name else None)([x, se])


def bottleneck_block(
    x: layers.Layer,
    filters: int,
    channels_axis,
    data_format,
    strides: int = 1,
    downsample: bool = False,
    senet: bool = False,
    block_name: Optional[str] = None,
) -> layers.Layer:
    """Bottleneck ResNet block.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the bottleneck layers.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
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
        data_format=data_format,
    )
    x = conv_block(
        x,
        filters,
        kernel_size=3,
        strides=strides,
        name=f"{block_name}_conv2",
        bn_name=f"{block_name}_batchnorm2",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    x = conv_block(
        x,
        filters * expansion,
        kernel_size=1,
        use_relu=False,
        name=f"{block_name}_conv3",
        bn_name=f"{block_name}_batchnorm3",
        channels_axis=channels_axis,
        data_format=data_format,
    )

    if senet:
        x = squeeze_excitation_block(
            x, data_format=data_format, name=f"{block_name}_se"
        )

    if (
        downsample
        or strides != 1
        or x.shape[channels_axis] != residual.shape[channels_axis]
    ):
        residual = conv_block(
            residual,
            filters * expansion,
            kernel_size=1,
            strides=strides,
            use_relu=False,
            name=f"{block_name}_downsample_conv",
            bn_name=f"{block_name}_downsample_batchnorm",
            channels_axis=channels_axis,
            data_format=data_format,
        )

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class ResNet(keras.Model):
    """
    Instantiates the ResNet architecture with support for ResNeXt and SE-ResNet/SE-ResNeXt configurations.

    Reference:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016)

    Args:
        block_fn: Callable, the block function to use for residual blocks. Should accept parameters:
            (x, filters, strides=1, downsample=False, block_name=None)
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
        block_fn=bottleneck_block,
        block_repeats=[2, 2, 2, 2],
        filters=[64, 128, 256, 512],
        groups=32,
        senet=False,
        width_factor=2,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResNet",
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
        x = conv_block(
            x,
            filters[0],
            kernel_size=7,
            strides=2,
            name="conv1",
            bn_name="batchnorm1",
            channels_axis=channels_axis,
            data_format=data_format,
        )
        x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
        x = layers.MaxPooling2D(
            data_format=data_format, pool_size=3, strides=2, padding="valid"
        )(x)
        features.append(x)

        common_args = {
            "channels_axis": channels_axis,
            "data_format": data_format,
            "senet": senet,
        }

        if isinstance(block_fn, dict):
            if block_fn.get("module") == "kvmm.models.resnext.resnext_model":
                common_args.update({"groups": groups, "width_factor": width_factor})
        elif hasattr(block_fn, "__module__") and "resnext" in block_fn.__module__:
            common_args.update({"groups": groups, "width_factor": width_factor})

        for i, num_blocks in enumerate(block_repeats):
            for j in range(num_blocks):
                common_args["block_name"] = f"resnet_layer{i + 1}_{j}"
                if j == 0 and i > 0:
                    x = block_fn(
                        x, filters[i], strides=2, downsample=True, **common_args
                    )
                else:
                    x = block_fn(x, filters[i], **common_args)
            if j == num_blocks - 1:
                features.append(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                kernel_initializer="zeros",
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

        self.block_fn = block_fn
        self.block_repeats = block_repeats
        self.filters = filters
        self.groups = groups
        self.senet = senet
        self.width_factor = width_factor
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

        if hasattr(self.block_fn, "__module__"):
            block_fn_config = {
                "class_name": "function",
                "config": self.block_fn.__name__,
                "module": self.block_fn.__module__,
                "registered_name": "function",
            }
        else:
            block_fn_config = {
                "class_name": "function",
                "config": "bottleneck_block",
                "module": "kv.models.resnet.resnet_model",
                "registered_name": "function",
            }

        config.update(
            {
                "block_fn": block_fn_config,
                "block_repeats": self.block_repeats,
                "filters": self.filters,
                "groups": self.groups,
                "senet": self.senet,
                "width_factor": self.width_factor,
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
        if isinstance(config["block_fn"], dict):
            block_fn_name = config["block_fn"]["config"]
            module_path = config["block_fn"]["module"]

            if module_path == "kvmm.models.backbones.resnet.resnet_model":
                if block_fn_name == "bottleneck_block":
                    config["block_fn"] = bottleneck_block
            elif module_path == "kvmm.models.backbones.resnext.resnext_model":
                from kvmm.models.backbones.resnext.resnext_model import resnext_block

                if block_fn_name == "resnext_block":
                    config["block_fn"] = resnext_block
            else:
                raise ValueError(
                    f"Unknown block function: {block_fn_name} from module {module_path}"
                )

        return cls(**config)


# ResNet Variants
@register_model
def ResNet50(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="a1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNet50",
):
    model = ResNet(
        block_fn=globals()[RESNET_MODEL_CONFIG["ResNet50"]["block_fn"]],
        block_repeats=RESNET_MODEL_CONFIG["ResNet50"]["block_repeats"],
        filters=RESNET_MODEL_CONFIG["ResNet50"]["filters"],
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
    if weights in get_all_weight_names(RESNET_WEIGHTS_CONFIG):
        load_weights_from_config("ResNet50", weights, model, RESNET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNet101(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="a1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNet101",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[RESNET_MODEL_CONFIG["ResNet101"]["block_fn"]],
        block_repeats=RESNET_MODEL_CONFIG["ResNet101"]["block_repeats"],
        filters=RESNET_MODEL_CONFIG["ResNet101"]["filters"],
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

    if weights in get_all_weight_names(RESNET_WEIGHTS_CONFIG):
        load_weights_from_config("ResNet101", weights, model, RESNET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNet152(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="a1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNet152",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[RESNET_MODEL_CONFIG["ResNet152"]["block_fn"]],
        block_repeats=RESNET_MODEL_CONFIG["ResNet152"]["block_repeats"],
        filters=RESNET_MODEL_CONFIG["ResNet152"]["filters"],
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

    if weights in get_all_weight_names(RESNET_WEIGHTS_CONFIG):
        load_weights_from_config("ResNet152", weights, model, RESNET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
