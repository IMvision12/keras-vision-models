"""DINOv3 models in pure Keras 3.

Implements the self-supervised models from
`DINOv3: A Rethinking and An Improvement
<https://arxiv.org/abs/2508.10104>`_ (Oquab et al., 2025).

DINOv3 ViT differs from standard ViT in three ways:
- **2D RoPE** (rotary position embeddings, theta=100) instead of learned pos embeds
- **SwiGLU FFN** for the "plus" variants (S+, H+, 7B); standard MLP+GELU for S/B/L
- **Register tokens** (4 extra learnable tokens prepended after CLS)

ConvNeXt variants are ConvNeXt-v2 (GRN) distilled from the DINOv3 ViT-7B teacher.

Available ViT variants::

    DinoV3ViTSmall16     -- ViT-S/16  (~21 M params, GELU MLP)
    DinoV3ViTBase16      -- ViT-B/16  (~86 M params, GELU MLP)
    DinoV3ViTLarge16     -- ViT-L/16  (~300 M params, GELU MLP)

Available ConvNeXt variants::

    DinoV3ConvNeXtTiny   -- ConvNeXt-v2-Tiny  (~29 M params)
    DinoV3ConvNeXtSmall  -- ConvNeXt-v2-Small (~50 M params)
    DinoV3ConvNeXtBase   -- ConvNeXt-v2-Base  (~89 M params)
    DinoV3ConvNeXtLarge  -- ConvNeXt-v2-Large (~198 M params)
"""

import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.layers import ImageNormalizationLayer, LayerScale
from kmodels.model_registry import register_model
from kmodels.models.convnext.convnext_model import ConvNeXt
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import (
    DINOV3_CONVNEXT_MODEL_CONFIG,
    DINOV3_VIT_MODEL_CONFIG,
    DINOV3_WEIGHTS_CONFIG,
)
from .dino_v3_layers import DinoV3Attention, _build_rope_2d_cache

# --------------------------------------------------------------------------- #
# Helpers: SwiGLU & MLP
# --------------------------------------------------------------------------- #


def _swiglu_ffn(x, dim, hidden_dim, block_idx):
    """SwiGLU feed-forward: gate = SiLU(x W1), up = x W2, out = (gate * up) W3."""
    gate = layers.Dense(
        hidden_dim, use_bias=True, name=f"blocks_{block_idx}_swiglu_gate"
    )(x)
    gate = layers.Activation("silu")(gate)
    up = layers.Dense(hidden_dim, use_bias=True, name=f"blocks_{block_idx}_swiglu_up")(
        x
    )
    x = layers.Multiply()([gate, up])
    x = layers.Dense(dim, use_bias=True, name=f"blocks_{block_idx}_swiglu_down")(x)
    return x


def _mlp_block(x, dim, hidden_dim, block_idx):
    """Standard GELU MLP."""
    x = layers.Dense(hidden_dim, use_bias=True, name=f"blocks_{block_idx}_dense_1")(x)
    x = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu")(x)
    x = layers.Dense(dim, use_bias=True, name=f"blocks_{block_idx}_dense_2")(x)
    return x


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #


def _dinov3_block(
    inputs,
    dim,
    num_heads,
    mlp_hidden_dim,
    num_prefix_tokens,
    rope_theta,
    use_swiglu,
    init_values,
    block_idx,
    rope_cos,
    rope_sin,
):
    # Attention
    x = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=f"blocks_{block_idx}_layernorm_1"
    )(inputs)
    attn = DinoV3Attention(
        dim=dim,
        num_heads=num_heads,
        num_prefix_tokens=num_prefix_tokens,
        rope_theta=rope_theta,
        block_prefix=f"blocks_{block_idx}",
    )
    attn.set_rope_cache(rope_cos, rope_sin)
    x = attn(x)
    if init_values is not None:
        x = LayerScale(
            init_values=init_values, name=f"blocks_{block_idx}_layerscale_1"
        )(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([x, inputs])

    # FFN
    y = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=f"blocks_{block_idx}_layernorm_2"
    )(x)
    if use_swiglu:
        y = _swiglu_ffn(y, dim, mlp_hidden_dim, block_idx)
    else:
        y = _mlp_block(y, dim, mlp_hidden_dim, block_idx)
    if init_values is not None:
        y = LayerScale(
            init_values=init_values, name=f"blocks_{block_idx}_layerscale_2"
        )(y)
    out = layers.Add(name=f"blocks_{block_idx}_add_2")([x, y])
    return out


# --------------------------------------------------------------------------- #
# DinoV3ViT
# --------------------------------------------------------------------------- #


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV3ViT(keras.Model):
    """DINOv3 Vision Transformer with 2D RoPE and register tokens."""

    def __init__(
        self,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        use_swiglu=False,
        num_register_tokens=4,
        init_values=1.0,
        rope_theta=100.0,
        include_top=False,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights=None,
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=None,
        classifier_activation="softmax",
        name="DinoV3ViT",
        **kwargs,
    ):
        if include_top and num_classes is None:
            num_classes = 1000

        if include_top and as_backbone:
            raise ValueError(
                "Cannot use `as_backbone=True` with `include_top=True`. "
                f"Received: as_backbone={as_backbone}, include_top={include_top}"
            )

        data_format = keras.config.image_data_format()

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
            min_size=32,
            data_format=data_format,
            require_flatten=include_top,
            weights=weights,
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input
        features = []

        if data_format == "channels_first":
            _, height, width = (
                input_shape if len(input_shape) == 3 else (None, *input_shape[1:])
            )
        else:
            height, width = input_shape[0], input_shape[1]

        grid_h = height // patch_size
        grid_w = width // patch_size
        num_prefix_tokens = 1 + num_register_tokens  # CLS + registers

        # Build RoPE cache
        rope_cos_np, rope_sin_np = _build_rope_2d_cache(
            grid_h,
            grid_w,
            dim // num_heads,
            theta=rope_theta,
        )
        rope_cos = ops.convert_to_tensor(rope_cos_np)
        rope_sin = ops.convert_to_tensor(rope_sin_np)

        # SwiGLU hidden dim: 8/3 * dim (param-matched to 4x MLP)
        if use_swiglu:
            mlp_hidden_dim = int(8 * dim / 3)
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)

        # ---------- Patch embedding ----------
        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )
        x = layers.Conv2D(
            filters=dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            name="patch_embed",
        )(x)
        x = layers.Reshape((-1, dim))(x)  # (B, num_patches, dim)

        # ---------- CLS token ----------
        cls_token = layers.Layer(name="cls_token")
        cls_token.cls = cls_token.add_weight(
            name="cls_token",
            shape=(1, 1, dim),
            initializer="zeros",
            trainable=True,
        )

        cls_broadcast = layers.Lambda(
            lambda v: ops.broadcast_to(cls_token.cls, [ops.shape(v)[0], 1, dim]),
            name="broadcast_cls",
        )(x)
        x = layers.Concatenate(axis=1, name="prepend_cls")([cls_broadcast, x])

        # ---------- Register tokens ----------
        if num_register_tokens > 0:
            reg_layer = layers.Layer(name="register_tokens")
            reg_layer.reg = reg_layer.add_weight(
                name="register_tokens",
                shape=(1, num_register_tokens, dim),
                initializer="zeros",
                trainable=True,
            )
            reg_broadcast = layers.Lambda(
                lambda v: ops.broadcast_to(
                    reg_layer.reg, [ops.shape(v)[0], num_register_tokens, dim]
                ),
                name="broadcast_reg",
            )(x)
            # Insert registers after CLS: [CLS, reg_1..reg_R, patch_1..patch_N]
            cls_part = layers.Lambda(lambda v: v[:, :1, :], name="split_cls")(x)
            patch_part = layers.Lambda(lambda v: v[:, 1:, :], name="split_patches")(x)
            x = layers.Concatenate(axis=1, name="assemble_tokens")(
                [cls_part, reg_broadcast, patch_part]
            )

        features.append(x)

        # ---------- Transformer blocks ----------
        for i in range(depth):
            x = _dinov3_block(
                x,
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                num_prefix_tokens=num_prefix_tokens,
                rope_theta=rope_theta,
                use_swiglu=use_swiglu,
                init_values=init_values,
                block_idx=i,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            features.append(x)

        x = layers.LayerNormalization(epsilon=1e-6, axis=-1, name="final_layernorm")(x)

        # ---------- Output ----------
        if include_top:
            x = layers.Lambda(lambda v: v[:, 0], name="extract_cls")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        elif as_backbone:
            x = features
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling1D(
                    data_format=data_format, name="avg_pool"
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling1D(data_format=data_format, name="max_pool")(
                    x
                )

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_swiglu = use_swiglu
        self.num_register_tokens = num_register_tokens
        self.init_values = init_values
        self.rope_theta = rope_theta
        self.include_top = include_top
        self.as_backbone = as_backbone
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "use_swiglu": self.use_swiglu,
                "num_register_tokens": self.num_register_tokens,
                "init_values": self.init_values,
                "rope_theta": self.rope_theta,
                "include_top": self.include_top,
                "as_backbone": self.as_backbone,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "pooling": self.pooling,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# --------------------------------------------------------------------------- #
# Builder helpers
# --------------------------------------------------------------------------- #


def _build_dinov3_vit(
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

    model = DinoV3ViT(
        **DINOV3_VIT_MODEL_CONFIG[model_name],
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

    if weights in get_all_weight_names(DINOV3_WEIGHTS_CONFIG):
        load_weights_from_config(model_name, weights, model, DINOV3_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


def _build_dinov3_convnext(
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

    model = ConvNeXt(
        **DINOV3_CONVNEXT_MODEL_CONFIG[model_name],
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

    if weights in get_all_weight_names(DINOV3_WEIGHTS_CONFIG):
        load_weights_from_config(model_name, weights, model, DINOV3_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


# --------------------------------------------------------------------------- #
# ViT variants
# --------------------------------------------------------------------------- #


@register_model
def DinoV3ViTSmall16(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov3",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV3ViTSmall16",
    **kwargs,
):
    """DINOv3 ViT-S/16 (~21 M params, GELU MLP, 16x16 patches)."""
    return _build_dinov3_vit(
        "DinoV3ViTSmall16",
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
def DinoV3ViTBase16(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov3",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV3ViTBase16",
    **kwargs,
):
    """DINOv3 ViT-B/16 (~86 M params, GELU MLP, 16x16 patches)."""
    return _build_dinov3_vit(
        "DinoV3ViTBase16",
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
def DinoV3ViTLarge16(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov3",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV3ViTLarge16",
    **kwargs,
):
    """DINOv3 ViT-L/16 (~300 M params, GELU MLP, 16x16 patches)."""
    return _build_dinov3_vit(
        "DinoV3ViTLarge16",
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


# --------------------------------------------------------------------------- #
# ConvNeXt variants
# --------------------------------------------------------------------------- #


@register_model
def DinoV3ConvNeXtTiny(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov3",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV3ConvNeXtTiny",
    **kwargs,
):
    """DINOv3 ConvNeXt-v2-Tiny (~29 M params, distilled from ViT-7B)."""
    return _build_dinov3_convnext(
        "DinoV3ConvNeXtTiny",
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
def DinoV3ConvNeXtSmall(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov3",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV3ConvNeXtSmall",
    **kwargs,
):
    """DINOv3 ConvNeXt-v2-Small (~50 M params, distilled from ViT-7B)."""
    return _build_dinov3_convnext(
        "DinoV3ConvNeXtSmall",
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
def DinoV3ConvNeXtBase(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov3",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV3ConvNeXtBase",
    **kwargs,
):
    """DINOv3 ConvNeXt-v2-Base (~89 M params, distilled from ViT-7B)."""
    return _build_dinov3_convnext(
        "DinoV3ConvNeXtBase",
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
def DinoV3ConvNeXtLarge(
    include_top=False,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="dinov3",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="DinoV3ConvNeXtLarge",
    **kwargs,
):
    """DINOv3 ConvNeXt-v2-Large (~198 M params, distilled from ViT-7B)."""
    return _build_dinov3_convnext(
        "DinoV3ConvNeXtLarge",
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
