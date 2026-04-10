from kmodels.model_registry import register_model
from kmodels.models.resnet.resnet_model import ResNet, bottleneck_block
from kmodels.models.vit.vit_model import VisionTransformer
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import (
    DINO_RESNET_MODEL_CONFIG,
    DINO_VIT_MODEL_CONFIG,
    DINO_WEIGHTS_CONFIG,
)


def DinoViT(
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
    """Instantiates a DINO Vision Transformer backbone.

    Builds a standard ViT pretrained with the DINO self-supervised method.

    Reference:
    - [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)

    Args:
        model_name: String, key into ``DINO_VIT_MODEL_CONFIG`` selecting the
            variant (e.g. ``"DinoViTSmall16"``).
        include_top: Boolean, whether to include a classification head.
            Defaults to ``False``.
        as_backbone: Boolean, whether to return intermediate feature maps.
            When True, returns a list of feature maps at different stages.
            Defaults to ``False``.
        include_normalization: Boolean, whether to include normalization layers
            at the start of the network. When True, input images should be in
            uint8 format with values in [0, 255]. Defaults to ``True``.
        normalization_mode: String, specifying the normalization mode to use.
            Defaults to ``"imagenet"``.
        weights: String, one of ``"dino"`` (pretrained) or a filepath to
            custom weights. Set to ``None`` for random initialization.
        input_tensor: Optional Keras tensor to use as input.
        input_shape: Optional tuple specifying the input shape.
            Defaults to ``(224, 224, 3)``.
        pooling: Optional pooling mode when ``include_top=False``:
            - ``None``: output is the token sequence ``(B, N, dim)``
            - ``"avg"``: global average pooling
            - ``"max"``: global max pooling
        num_classes: Integer, number of output classes when ``include_top=True``.
        classifier_activation: String or callable, activation for the
            classification head. Defaults to ``"softmax"``.
        name: String, the name of the model.

    Returns:
        A Keras ``Model`` instance.
    """
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


def DinoResNet(
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
    """Instantiates a DINO ResNet backbone.

    Builds a ResNet-50 pretrained with the DINO self-supervised method.

    Reference:
    - [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)

    Args:
        model_name: String, key into ``DINO_RESNET_MODEL_CONFIG`` selecting
            the variant (e.g. ``"DinoResNet50"``).
        include_top: Boolean, whether to include a classification head.
            Defaults to ``False``.
        as_backbone: Boolean, whether to return intermediate feature maps.
            When True, returns a list of feature maps at different stages.
            Defaults to ``False``.
        include_normalization: Boolean, whether to include normalization layers
            at the start of the network. When True, input images should be in
            uint8 format with values in [0, 255]. Defaults to ``True``.
        normalization_mode: String, specifying the normalization mode to use.
            Defaults to ``"imagenet"``.
        weights: String, one of ``"dino"`` (pretrained) or a filepath to
            custom weights. Set to ``None`` for random initialization.
        input_tensor: Optional Keras tensor to use as input.
        input_shape: Optional tuple specifying the input shape.
        pooling: Optional pooling mode when ``include_top=False``:
            - ``None``: output is the spatial feature map ``(B, H, W, C)``
            - ``"avg"``: global average pooling ``(B, C)``
            - ``"max"``: global max pooling ``(B, C)``
        num_classes: Integer, number of output classes when ``include_top=True``.
        classifier_activation: String or callable, activation for the
            classification head. Defaults to ``"softmax"``.
        name: String, the name of the model.

    Returns:
        A Keras ``Model`` instance.
    """
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
    return DinoViT(
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
    return DinoViT(
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
    return DinoViT(
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
    return DinoViT(
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
    return DinoResNet(
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
