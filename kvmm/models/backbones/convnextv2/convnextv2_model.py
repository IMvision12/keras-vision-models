from kvmm.model_registry import register_model
from kvmm.models.backbones.convnext.convnext_model import ConvNeXt
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import CONVNEXTV2_MODEL_CONFIG, CONVNEXTV2_WEIGHTS_CONFIG


# ConvNeXt V2
@register_model
def ConvNeXtV2Atto(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fcmae_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtV2Atto",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXTV2_MODEL_CONFIG["atto"],
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        use_grn=True,
        use_conv=True,
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

    if weights in get_all_weight_names(CONVNEXTV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtV2Atto", weights, model, CONVNEXTV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtV2Femto(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fcmae_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtV2Femto",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXTV2_MODEL_CONFIG["femto"],
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        use_grn=True,
        use_conv=True,
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

    if weights in get_all_weight_names(CONVNEXTV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtV2Femto", weights, model, CONVNEXTV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtV2Pico(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fcmae_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtV2Pico",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXTV2_MODEL_CONFIG["pico"],
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        use_grn=True,
        use_conv=True,
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

    if weights in get_all_weight_names(CONVNEXTV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtV2Pico", weights, model, CONVNEXTV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtV2Nano(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fcmae_ft_in22k_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtV2Nano",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXTV2_MODEL_CONFIG["nano"],
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        use_grn=True,
        use_conv=True,
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

    if weights in get_all_weight_names(CONVNEXTV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtV2Nano", weights, model, CONVNEXTV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtV2Tiny(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fcmae_ft_in22k_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtV2Tiny",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXTV2_MODEL_CONFIG["tiny"],
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        use_grn=True,
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

    if weights in get_all_weight_names(CONVNEXTV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtV2Tiny", weights, model, CONVNEXTV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtV2Base(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fcmae_ft_in22k_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtV2Base",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXTV2_MODEL_CONFIG["base"],
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        use_grn=True,
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

    if weights in get_all_weight_names(CONVNEXTV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtV2Base", weights, model, CONVNEXTV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtV2Large(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fcmae_ft_in22k_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtV2Large",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXTV2_MODEL_CONFIG["large"],
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        use_grn=True,
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

    if weights in get_all_weight_names(CONVNEXTV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtV2Large", weights, model, CONVNEXTV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtV2Huge(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fcmae_ft_in22k_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtV2Huge",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXTV2_MODEL_CONFIG["huge"],
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        use_grn=True,
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

    if weights in get_all_weight_names(CONVNEXTV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtV2Huge", weights, model, CONVNEXTV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
