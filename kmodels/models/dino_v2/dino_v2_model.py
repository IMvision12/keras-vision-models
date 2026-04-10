"""DINOv2 models in pure Keras 3.

Implements the self-supervised models from
`DINOv2: Learning Robust Visual Features without Supervision
<https://arxiv.org/abs/2304.07193>`_ (Oquab et al., 2024).  Weights are
converted from the official HuggingFace checkpoints ``facebook/dinov2-*``.

DINOv2 is structurally a standard ViT with ``patch_size=14``, per-block
**LayerScale** (``init_values=1.0``), and no classification head.  The
``ViT-G/14`` variant (SwiGLU FFN) is not included.

Available variants:

* ``DinoV2Small14`` -- ViT-S/14 (~22 M params)
* ``DinoV2Base14``  -- ViT-B/14 (~86 M params)
* ``DinoV2Large14`` -- ViT-L/14 (~300 M params)

Example::

    from kmodels.models.dino_v2 import DinoV2Small14

    # Backbone (token sequence)
    model = DinoV2Small14(weights="dinov2")                          # (B, 257, 384)

    # Intermediate feature maps for FPN / segmentation
    feat = DinoV2Small14(weights="dinov2", as_backbone=True)         # list of tensors
"""

from kmodels.model_registry import register_model
from kmodels.models.vit.vit_model import VisionTransformer
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import DINOV2_MODEL_CONFIG, DINOV2_WEIGHTS_CONFIG


def _build_dinov2(
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
        **DINOV2_MODEL_CONFIG[model_name],
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

    if weights in get_all_weight_names(DINOV2_WEIGHTS_CONFIG):
        load_weights_from_config(model_name, weights, model, DINOV2_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DinoV2Small14(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov2",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV2Small14",
    **kwargs,
):
    """DINOv2 ViT-S/14 (~22 M params, 14x14 patches)."""
    return _build_dinov2(
        "DinoV2Small14",
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
def DinoV2Base14(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov2",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV2Base14",
    **kwargs,
):
    """DINOv2 ViT-B/14 (~86 M params, 14x14 patches)."""
    return _build_dinov2(
        "DinoV2Base14",
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
def DinoV2Large14(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov2",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV2Large14",
    **kwargs,
):
    """DINOv2 ViT-L/14 (~300 M params, 14x14 patches)."""
    return _build_dinov2(
        "DinoV2Large14",
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
