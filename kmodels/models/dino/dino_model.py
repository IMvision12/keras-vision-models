from kmodels.model_registry import register_model
from kmodels.models.resnet.resnet_model import ResNet, bottleneck_block
from kmodels.models.vit.vit_model import VisionTransformer
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import (
    DINO_RESNET_MODEL_CONFIG,
    DINO_VIT_MODEL_CONFIG,
    DINO_WEIGHTS_CONFIG,
)


def _build_dino_vit(
    model_name,
    include_top,
    as_backbone,
    include_normalization,
    normalization_mode,
    weights,
    input_tensor,
    input_shape,
    pooling,
    num_classes,
    classifier_activation,
    name,
    **kwargs,
):
    if include_top and num_classes is None:
        num_classes = 1000

    if input_shape is None and input_tensor is None:
        input_shape = (224, 224, 3)

    model = VisionTransformer(
        **DINO_VIT_MODEL_CONFIG[model_name],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=None,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(DINO_WEIGHTS_CONFIG):
        load_weights_from_config(model_name, weights, model, DINO_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


def _build_dino_resnet(
    model_name,
    include_top,
    as_backbone,
    include_normalization,
    normalization_mode,
    weights,
    input_tensor,
    input_shape,
    pooling,
    num_classes,
    classifier_activation,
    name,
    **kwargs,
):
    if include_top and num_classes is None:
        num_classes = 1000

    model = ResNet(
        block_fn=bottleneck_block,
        block_repeats=DINO_RESNET_MODEL_CONFIG[model_name]["block_repeats"],
        filters=DINO_RESNET_MODEL_CONFIG[model_name]["filters"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=None,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(DINO_WEIGHTS_CONFIG):
        load_weights_from_config(model_name, weights, model, DINO_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DinoViTSmall16(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dino",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoViTSmall16",
    **kwargs,
):
    """DINO ViT-S/16 (21 M params backbone, 16x16 patches)."""
    return _build_dino_vit(
        "DinoViTSmall16",
        include_top,
        as_backbone,
        include_normalization,
        normalization_mode,
        weights,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )


@register_model
def DinoViTSmall8(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dino",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoViTSmall8",
    **kwargs,
):
    """DINO ViT-S/8 (21 M params backbone, 8x8 patches)."""
    return _build_dino_vit(
        "DinoViTSmall8",
        include_top,
        as_backbone,
        include_normalization,
        normalization_mode,
        weights,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )


@register_model
def DinoViTBase16(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dino",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoViTBase16",
    **kwargs,
):
    """DINO ViT-B/16 (85 M params backbone, 16x16 patches)."""
    return _build_dino_vit(
        "DinoViTBase16",
        include_top,
        as_backbone,
        include_normalization,
        normalization_mode,
        weights,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )


@register_model
def DinoViTBase8(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dino",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoViTBase8",
    **kwargs,
):
    """DINO ViT-B/8 (85 M params backbone, 8x8 patches)."""
    return _build_dino_vit(
        "DinoViTBase8",
        include_top,
        as_backbone,
        include_normalization,
        normalization_mode,
        weights,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )


@register_model
def DinoResNet50(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dino",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoResNet50",
    **kwargs,
):
    """DINO ResNet-50 (torchvision ResNet-50 trained with DINO)."""
    return _build_dino_resnet(
        "DinoResNet50",
        include_top,
        as_backbone,
        include_normalization,
        normalization_mode,
        weights,
        input_tensor,
        input_shape,
        pooling,
        num_classes,
        classifier_activation,
        name,
        **kwargs,
    )
