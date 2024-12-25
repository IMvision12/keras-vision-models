from kv.models.vision_transformer.vision_transformer_model import ViT
from kv.utils import get_all_weight_names, load_weights_from_config

from ...model_registry import register_model
from .config import DEIT_MODEL_CONFIG


@register_model
def DEiTTiny16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiTTiny16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiTTiny16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config("DEiTTiny16", weights, model, DEIT_MODEL_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiTSmall16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiTSmall16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiTSmall16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config("DEiTSmall16", weights, model, DEIT_MODEL_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiTBase16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiTBase16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiTBase16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config("DEiTBase16", weights, model, DEIT_MODEL_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiTTinyDistilled16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiTTinyDistilled16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiTTinyDistilled16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config(
            "DEiTTinyDistilled16", weights, model, DEIT_MODEL_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiTSmallDistilled16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiTSmallDistilled16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiTSmallDistilled16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config(
            "DEiTSmallDistilled16", weights, model, DEIT_MODEL_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiTBaseDistilled16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiTBaseDistilled16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiTBaseDistilled16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config(
            "DEiTBaseDistilled16", weights, model, DEIT_MODEL_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiT3Small16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiT3Small16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiT3Small16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config("DEiT3Small16", weights, model, DEIT_MODEL_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiT3Medium16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiT3Medium16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiT3Medium16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config("DEiT3Medium16", weights, model, DEIT_MODEL_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiT3Base16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiT3Base16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiT3Base16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config("DEiT3Base16", weights, model, DEIT_MODEL_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiT3Large16(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiT3Large16",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiT3Large16"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config("DEiT3Large16", weights, model, DEIT_MODEL_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DEiT3Huge14(
    include_top=True,
    weights="fb_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DEiT3Huge14",
    **kwargs,
):
    model = ViT(
        **DEIT_MODEL_CONFIG["DEiT3Huge14"],
        include_top=include_top,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(DEIT_MODEL_CONFIG):
        load_weights_from_config("DEiT3Huge14", weights, model, DEIT_MODEL_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
