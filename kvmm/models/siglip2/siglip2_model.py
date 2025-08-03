from kvmm.model_registry import register_model
from kvmm.models.siglip.siglip_model import SigLIPModel
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import SigLIP2_MODEL_CONFIG, SigLIP2_WEIGHTS_CONFIG


@register_model
def SigLIP2BaseP16(
    weights="google_224",
    input_tensor=None,
    input_shape=None,
    name="SigLIP2BaseP16",
    **kwargs,
):
    model = SigLIPModel(
        **SigLIP2_MODEL_CONFIG["SigLIP2BaseP16"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        weights=weights,
        **kwargs,
    )

    if weights in get_all_weight_names(SigLIP2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SigLIP2BaseP16", weights, model, SigLIP2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SigLIP2BaseP32(
    weights="google_224",
    input_tensor=None,
    input_shape=None,
    name="SigLIP2BaseP32",
    **kwargs,
):
    model = SigLIPModel(
        **SigLIP2_MODEL_CONFIG["SigLIP2BaseP32"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        weights=weights,
        **kwargs,
    )

    if weights in get_all_weight_names(SigLIP2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SigLIP2BaseP32", weights, model, SigLIP2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SigLIP2LargeP16(
    weights="google_224",
    input_tensor=None,
    input_shape=None,
    name="SigLIP2LargeP16",
    **kwargs,
):
    model = SigLIPModel(
        **SigLIP2_MODEL_CONFIG["SigLIP2LargeP16"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        weights=weights,
        **kwargs,
    )

    if weights in get_all_weight_names(SigLIP2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SigLIP2LargeP16", weights, model, SigLIP2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SigLIP2So400mP14(
    weights="google_224",
    input_tensor=None,
    input_shape=None,
    name="SigLIP2So400mP14",
    **kwargs,
):
    model = SigLIPModel(
        **SigLIP2_MODEL_CONFIG["SigLIP2So400mP14"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        weights=weights,
        **kwargs,
    )

    if weights in get_all_weight_names(SigLIP2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SigLIP2So400mP14", weights, model, SigLIP2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SigLIP2So400mP16(
    weights="google_224",
    input_tensor=None,
    input_shape=None,
    name="SigLIP2So400mP16",
    **kwargs,
):
    model = SigLIPModel(
        **SigLIP2_MODEL_CONFIG["SigLIP2So400mP16"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        weights=weights,
        **kwargs,
    )

    if weights in get_all_weight_names(SigLIP2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "SigLIP2So400mP16", weights, model, SigLIP2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
