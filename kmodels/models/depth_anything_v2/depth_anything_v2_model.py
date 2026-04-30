import keras

from kmodels.model_registry import register_model
from kmodels.models.depth_anything_v1.depth_anything_v1_model import DepthAnythingV1
from kmodels.weight_utils import get_all_weight_names, load_weights_from_config

from .config import DEPTH_ANYTHING_V2_MODEL_CONFIG, DEPTH_ANYTHING_V2_WEIGHTS_CONFIG


def create_depth_anything_v2(variant, input_shape, input_tensor, weights, **kwargs):
    """Factory helper that wires a named V2 variant into ``DepthAnythingV1``.

    Depth Anything V2 reuses the V1 architecture end-to-end — the
    changes in V2 are training-time (synthetic data + larger-capacity
    teacher model), not architectural — so this helper looks up the
    variant entry in ``DEPTH_ANYTHING_V2_MODEL_CONFIG`` and builds a
    ``DepthAnythingV1`` instance with those hyperparameters, including
    the metric-depth variants, which only differ in
    ``depth_estimation_type`` and ``max_depth``. Pretrained weights
    are loaded via the V2 weights config when ``weights`` matches a
    registered preset, or from a local path otherwise.

    Args:
        variant: String, variant name matching a key in
            ``DEPTH_ANYTHING_V2_MODEL_CONFIG`` (e.g.
            ``"DepthAnythingV2Small"``,
            ``"DepthAnythingV2MetricIndoorLarge"``).
        input_shape: Optional input shape forwarded to
            ``DepthAnythingV1``. When ``None``, defaults to
            ``(518, 518, 3)`` or ``(3, 518, 518)`` depending on the
            active data format.
        input_tensor: Optional Keras tensor to use as model input.
        weights: Either ``None`` (random init), a registered preset
            name from ``DEPTH_ANYTHING_V2_WEIGHTS_CONFIG``, or a path
            to a local ``.weights.h5`` file.
        **kwargs: Additional keyword arguments forwarded to
            ``DepthAnythingV1``.

    Returns:
        A built ``DepthAnythingV1`` model configured for the requested
        V2 variant with weights loaded when requested.
    """
    config = DEPTH_ANYTHING_V2_MODEL_CONFIG[variant]

    if input_shape is None:
        df = keras.config.image_data_format()
        if df == "channels_first":
            input_shape = (3, DepthAnythingV1.IMAGE_SIZE, DepthAnythingV1.IMAGE_SIZE)
        else:
            input_shape = (
                DepthAnythingV1.IMAGE_SIZE,
                DepthAnythingV1.IMAGE_SIZE,
                3,
            )

    model = DepthAnythingV1(
        backbone_dim=config["backbone_dim"],
        backbone_depth=config["backbone_depth"],
        backbone_num_heads=config["backbone_num_heads"],
        out_indices=config["out_indices"],
        neck_hidden_sizes=config["neck_hidden_sizes"],
        fusion_hidden_size=config["fusion_hidden_size"],
        reassemble_factors=config["reassemble_factors"],
        depth_estimation_type=config.get("depth_estimation_type", "relative"),
        max_depth=config.get("max_depth", 1.0),
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **kwargs,
    )

    if weights in get_all_weight_names(DEPTH_ANYTHING_V2_WEIGHTS_CONFIG):
        load_weights_from_config(
            variant, weights, model, DEPTH_ANYTHING_V2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DepthAnythingV2Small(
    input_shape=None, input_tensor=None, weights="da_v2", **kwargs
):
    return create_depth_anything_v2(
        "DepthAnythingV2Small", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def DepthAnythingV2Base(input_shape=None, input_tensor=None, weights="da_v2", **kwargs):
    return create_depth_anything_v2(
        "DepthAnythingV2Base", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def DepthAnythingV2Large(
    input_shape=None, input_tensor=None, weights="da_v2", **kwargs
):
    return create_depth_anything_v2(
        "DepthAnythingV2Large", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def DepthAnythingV2MetricIndoorSmall(
    input_shape=None, input_tensor=None, weights="da_v2", **kwargs
):
    return create_depth_anything_v2(
        "DepthAnythingV2MetricIndoorSmall",
        input_shape,
        input_tensor,
        weights,
        **kwargs,
    )


@register_model
def DepthAnythingV2MetricIndoorBase(
    input_shape=None, input_tensor=None, weights="da_v2", **kwargs
):
    return create_depth_anything_v2(
        "DepthAnythingV2MetricIndoorBase",
        input_shape,
        input_tensor,
        weights,
        **kwargs,
    )


@register_model
def DepthAnythingV2MetricIndoorLarge(
    input_shape=None, input_tensor=None, weights="da_v2", **kwargs
):
    return create_depth_anything_v2(
        "DepthAnythingV2MetricIndoorLarge",
        input_shape,
        input_tensor,
        weights,
        **kwargs,
    )


@register_model
def DepthAnythingV2MetricOutdoorSmall(
    input_shape=None, input_tensor=None, weights="da_v2", **kwargs
):
    return create_depth_anything_v2(
        "DepthAnythingV2MetricOutdoorSmall",
        input_shape,
        input_tensor,
        weights,
        **kwargs,
    )


@register_model
def DepthAnythingV2MetricOutdoorBase(
    input_shape=None, input_tensor=None, weights="da_v2", **kwargs
):
    return create_depth_anything_v2(
        "DepthAnythingV2MetricOutdoorBase",
        input_shape,
        input_tensor,
        weights,
        **kwargs,
    )


@register_model
def DepthAnythingV2MetricOutdoorLarge(
    input_shape=None, input_tensor=None, weights="da_v2", **kwargs
):
    return create_depth_anything_v2(
        "DepthAnythingV2MetricOutdoorLarge",
        input_shape,
        input_tensor,
        weights,
        **kwargs,
    )
