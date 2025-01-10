from kv.models.efficientnet.efficientnet_model import EfficientNet
from kv.utils import get_all_weight_names, load_weights_from_config

from ...model_registry import register_model
from .config import EFFICIENTNET_LITE_MODEL_CONFIG, EFFICIENTNET_LITE_WEIGHTS_CONFIG


@register_model
def EfficientNetLite0(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_LITE_MODEL_CONFIG["EfficientNetLite0"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        use_se=False,
        activation="relu6",
        name="EfficientNetLite0",
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_LITE_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB0", weights, model, EFFICIENTNET_LITE_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetLite1(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_LITE_MODEL_CONFIG["EfficientNetLite1"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        use_se=False,
        activation="relu6",
        name="EfficientNetLite1",
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_LITE_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB0", weights, model, EFFICIENTNET_LITE_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetLite2(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_LITE_MODEL_CONFIG["EfficientNetLite2"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        use_se=False,
        activation="relu6",
        name="EfficientNetLite2",
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_LITE_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB0", weights, model, EFFICIENTNET_LITE_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetLite3(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_LITE_MODEL_CONFIG["EfficientNetLite3"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        use_se=False,
        activation="relu6",
        name="EfficientNetLite3",
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_LITE_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB0", weights, model, EFFICIENTNET_LITE_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientNetLite4(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    model = EfficientNet(
        **EFFICIENTNET_LITE_MODEL_CONFIG["EfficientNetLite4"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        use_se=False,
        activation="relu6",
        name="EfficientNetLite4",
        **kwargs,
    )
    if weights in get_all_weight_names(EFFICIENTNET_LITE_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientNetB0", weights, model, EFFICIENTNET_LITE_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
