from kvmm.model_registry import register_model
from kvmm.models.backbones.resnet.resnet_model import ResNet
from kvmm.models.backbones.resnext.resnext_model import resnext_block
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import SENET_MODEL_CONFIG, SENET_WEIGHTS_CONFIG

__all__ = [
    "resnext_block",
    "SEResNet50",
    "SEResNeXt50_32x4d",
    "SEResNeXt101_32x4d",
    "SEResNeXt101_32x8d",
]


# SE-resent and SE-ResNext
@register_model
def SEResNet50(
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
    name="SEResNet50",
    **kwargs,
):
    model = ResNet(
        block_repeats=SENET_MODEL_CONFIG["SEResNet50"]["block_repeats"],
        filters=SENET_MODEL_CONFIG["SEResNet50"]["filters"],
        senet=SENET_MODEL_CONFIG["SEResNet50"]["senet"],
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

    if weights in get_all_weight_names(SENET_WEIGHTS_CONFIG):
        load_weights_from_config("SEResNet50", weights, model, SENET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SEResNeXt50_32x4d(
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
    name="SEResNeXt50_32x4d",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[SENET_MODEL_CONFIG["SEResNeXt50_32x4d"]["block_fn"]],
        block_repeats=SENET_MODEL_CONFIG["SEResNeXt50_32x4d"]["block_repeats"],
        filters=SENET_MODEL_CONFIG["SEResNeXt50_32x4d"]["filters"],
        groups=SENET_MODEL_CONFIG["SEResNeXt50_32x4d"]["groups"],
        width_factor=SENET_MODEL_CONFIG["SEResNeXt50_32x4d"]["width_factor"],
        senet=SENET_MODEL_CONFIG["SEResNeXt50_32x4d"]["senet"],
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

    if weights in get_all_weight_names(SENET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SEResNeXt50_32x4d", weights, model, SENET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SEResNeXt101_32x4d(
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
    name="SEResNeXt101_32x4d",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[SENET_MODEL_CONFIG["SEResNeXt101_32x4d"]["block_fn"]],
        block_repeats=SENET_MODEL_CONFIG["SEResNeXt101_32x4d"]["block_repeats"],
        filters=SENET_MODEL_CONFIG["SEResNeXt101_32x4d"]["filters"],
        groups=SENET_MODEL_CONFIG["SEResNeXt101_32x4d"]["groups"],
        width_factor=SENET_MODEL_CONFIG["SEResNeXt101_32x4d"]["width_factor"],
        senet=SENET_MODEL_CONFIG["SEResNeXt101_32x4d"]["senet"],
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

    if weights in get_all_weight_names(SENET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SEResNeXt101_32x4d", weights, model, SENET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SEResNeXt101_32x8d(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ah_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="SEResNeXt101_32x8d",
    **kwargs,
):
    model = ResNet(
        block_fn=globals()[SENET_MODEL_CONFIG["SEResNeXt101_32x8d"]["block_fn"]],
        block_repeats=SENET_MODEL_CONFIG["SEResNeXt101_32x8d"]["block_repeats"],
        filters=SENET_MODEL_CONFIG["SEResNeXt101_32x8d"]["filters"],
        groups=SENET_MODEL_CONFIG["SEResNeXt101_32x8d"]["groups"],
        width_factor=SENET_MODEL_CONFIG["SEResNeXt101_32x8d"]["width_factor"],
        senet=SENET_MODEL_CONFIG["SEResNeXt101_32x8d"]["senet"],
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

    if weights in get_all_weight_names(SENET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SEResNeXt101_32x8d", weights, model, SENET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
