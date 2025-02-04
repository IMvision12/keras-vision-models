from kv.utils import get_all_weight_names, load_weights_from_config, register_model

from ..vision_transformer import ViT
from .config import FLEXIVIT_MODEL_CONFIG, FLEXIVIT_WEIGHTS_CONFIG


@register_model
def FlexiViTSmall(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="1200ep_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="FlexiViTSmall",
    **kwargs,
):
    model = ViT(
        **FLEXIVIT_MODEL_CONFIG["FlexiViTSmall"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(FLEXIVIT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "FlexiViTSmall", weights, model, FLEXIVIT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def FlexiViTBase(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="1200ep_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="FlexiViTBase",
    **kwargs,
):
    model = ViT(
        **FLEXIVIT_MODEL_CONFIG["FlexiViTBase"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(FLEXIVIT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "FlexiViTBase", weights, model, FLEXIVIT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def FlexiViTLarge(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="1200ep_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="FlexiViTLarge",
    **kwargs,
):
    model = ViT(
        **FLEXIVIT_MODEL_CONFIG["FlexiViTLarge"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(FLEXIVIT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "FlexiViTLarge", weights, model, FLEXIVIT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
