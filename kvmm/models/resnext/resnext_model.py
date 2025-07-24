from typing import Optional

from keras import layers

from kvmm.model_registry import register_model
from kvmm.models.backbones.resnet.resnet_model import (
    ResNet,
    conv_block,
    squeeze_excitation_block,
)
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import RESNEXT_MODEL_CONFIG, RESNEXT_WEIGHTS_CONFIG


def resnext_block(
    x: layers.Layer,
    filters: int,
    channels_axis,
    data_format,
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
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
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
        data_format=data_format,
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


# ResNeXt Variants
@register_model
def ResNeXt50_32x4d(
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
    name="ResNeXt50_32x4d",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[RESNEXT_MODEL_CONFIG["ResNeXt50_32x4d"]["block_fn"]],
        block_repeats=RESNEXT_MODEL_CONFIG["ResNeXt50_32x4d"]["block_repeats"],
        filters=RESNEXT_MODEL_CONFIG["ResNeXt50_32x4d"]["filters"],
        groups=RESNEXT_MODEL_CONFIG["ResNeXt50_32x4d"]["groups"],
        width_factor=RESNEXT_MODEL_CONFIG["ResNeXt50_32x4d"]["width_factor"],
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

    if weights in get_all_weight_names(RESNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResNeXt50_32x4d", weights, model, RESNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNeXt101_32x4d(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="gluon_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNeXt101_32x4d",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[RESNEXT_MODEL_CONFIG["ResNeXt101_32x4d"]["block_fn"]],
        block_repeats=RESNEXT_MODEL_CONFIG["ResNeXt101_32x4d"]["block_repeats"],
        filters=RESNEXT_MODEL_CONFIG["ResNeXt101_32x4d"]["filters"],
        groups=RESNEXT_MODEL_CONFIG["ResNeXt101_32x4d"]["groups"],
        width_factor=RESNEXT_MODEL_CONFIG["ResNeXt101_32x4d"]["width_factor"],
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

    if weights in get_all_weight_names(RESNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResNeXt101_32x4d", weights, model, RESNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNeXt101_32x8d(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="tv_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNeXt101_32x8d",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[RESNEXT_MODEL_CONFIG["ResNeXt101_32x8d"]["block_fn"]],
        block_repeats=RESNEXT_MODEL_CONFIG["ResNeXt101_32x8d"]["block_repeats"],
        filters=RESNEXT_MODEL_CONFIG["ResNeXt101_32x8d"]["filters"],
        groups=RESNEXT_MODEL_CONFIG["ResNeXt101_32x8d"]["groups"],
        width_factor=RESNEXT_MODEL_CONFIG["ResNeXt101_32x8d"]["width_factor"],
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

    if weights in get_all_weight_names(RESNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResNeXt101_32x8d", weights, model, RESNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNeXt101_32x16d(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_wsl_ig1b_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNeXt101_32x16d",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[RESNEXT_MODEL_CONFIG["ResNeXt101_32x16d"]["block_fn"]],
        block_repeats=RESNEXT_MODEL_CONFIG["ResNeXt101_32x16d"]["block_repeats"],
        filters=RESNEXT_MODEL_CONFIG["ResNeXt101_32x16d"]["filters"],
        groups=RESNEXT_MODEL_CONFIG["ResNeXt101_32x16d"]["groups"],
        width_factor=RESNEXT_MODEL_CONFIG["ResNeXt101_32x16d"]["width_factor"],
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

    if weights in get_all_weight_names(RESNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResNeXt101_32x16d", weights, model, RESNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ResNeXt101_32x32d(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_wsl_ig1b_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ResNeXt101_32x16d",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[RESNEXT_MODEL_CONFIG["ResNeXt101_32x32d"]["block_fn"]],
        block_repeats=RESNEXT_MODEL_CONFIG["ResNeXt101_32x32d"]["block_repeats"],
        filters=RESNEXT_MODEL_CONFIG["ResNeXt101_32x32d"]["filters"],
        groups=RESNEXT_MODEL_CONFIG["ResNeXt101_32x32d"]["groups"],
        width_factor=RESNEXT_MODEL_CONFIG["ResNeXt101_32x32d"]["width_factor"],
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
    if weights in get_all_weight_names(RESNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResNeXt101_32x32d", weights, model, RESNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
