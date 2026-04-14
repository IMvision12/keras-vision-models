import keras
from keras import layers, ops

from kmodels.layers import LayerScale
from kmodels.model_registry import register_model
from kmodels.models.vit.vit_layers import (
    AddPositionEmbs,
    ClassDistToken,
    MultiHeadSelfAttention,
)
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import DEPTH_ANYTHING_V1_MODEL_CONFIG, DEPTH_ANYTHING_V1_WEIGHTS_CONFIG


def depth_anything_v1_aligned_bilinear_resize(x, target_h, target_w, data_format):
    """Pure-Keras aligned-corners bilinear resize.

    Matches ``torch.nn.functional.interpolate(..., mode='bilinear',
    align_corners=True)`` via explicit gather + lerp. ``keras.ops.image.resize``
    only supports half-pixel alignment, so we implement the align-corners
    coordinate mapping manually: the ``target_h`` output rows are sampled
    at evenly spaced source coordinates ``i * (H - 1) / (target_h - 1)``
    (and analogously for width).

    Args:
        x: Input tensor. ``(B, H, W, C)`` for ``channels_last``,
            ``(B, C, H, W)`` for ``channels_first``.
        target_h: Target output height.
        target_w: Target output width.
        data_format: Either ``"channels_last"`` or ``"channels_first"``.
    """
    shape = ops.shape(x)
    if data_format == "channels_first":
        h_axis, w_axis = 2, 3
        h = shape[2]
        w = shape[3]
    else:
        h_axis, w_axis = 1, 2
        h = shape[1]
        w = shape[2]

    h_f = ops.cast(h, "float32")
    w_f = ops.cast(w, "float32")

    if target_h > 1:
        y_coords = (
            ops.arange(target_h, dtype="float32") * (h_f - 1.0) / float(target_h - 1)
        )
    else:
        y_coords = ops.zeros((1,), dtype="float32")

    if target_w > 1:
        x_coords = (
            ops.arange(target_w, dtype="float32") * (w_f - 1.0) / float(target_w - 1)
        )
    else:
        x_coords = ops.zeros((1,), dtype="float32")

    y0 = ops.cast(ops.floor(y_coords), "int32")
    x0 = ops.cast(ops.floor(x_coords), "int32")
    y1 = ops.minimum(y0 + 1, h - 1)
    x1 = ops.minimum(x0 + 1, w - 1)
    y0 = ops.minimum(y0, h - 1)
    x0 = ops.minimum(x0, w - 1)

    dy = y_coords - ops.cast(y0, "float32")
    dx = x_coords - ops.cast(x0, "float32")

    top = ops.take(x, y0, axis=h_axis)
    bot = ops.take(x, y1, axis=h_axis)
    tl = ops.take(top, x0, axis=w_axis)
    tr = ops.take(top, x1, axis=w_axis)
    bl = ops.take(bot, x0, axis=w_axis)
    br = ops.take(bot, x1, axis=w_axis)

    if data_format == "channels_first":
        dx_r = ops.reshape(dx, (1, 1, 1, target_w))
        dy_r = ops.reshape(dy, (1, 1, target_h, 1))
    else:
        dx_r = ops.reshape(dx, (1, 1, target_w, 1))
        dy_r = ops.reshape(dy, (1, target_h, 1, 1))

    top_lerp = tl * (1.0 - dx_r) + tr * dx_r
    bot_lerp = bl * (1.0 - dx_r) + br * dx_r
    return top_lerp * (1.0 - dy_r) + bot_lerp * dy_r


def depth_anything_v1_pre_act_residual(x, channels, name, data_format):
    residual = x
    x = layers.Activation("relu", name=f"{name}_act1")(x)
    x = layers.Conv2D(
        channels, 3, padding="same", data_format=data_format, name=f"{name}_conv1"
    )(x)
    x = layers.Activation("relu", name=f"{name}_act2")(x)
    x = layers.Conv2D(
        channels, 3, padding="same", data_format=data_format, name=f"{name}_conv2"
    )(x)
    return layers.Add(name=f"{name}_add")([x, residual])


def depth_anything_v1_fusion_block(
    hidden_state,
    residual,
    target_h,
    target_w,
    fusion_hidden_size,
    data_format,
    name,
):
    if residual is not None:
        hidden_state = layers.Add(name=f"{name}_add_residual")(
            [
                hidden_state,
                depth_anything_v1_pre_act_residual(
                    residual, fusion_hidden_size, f"{name}_res1", data_format
                ),
            ]
        )

    hidden_state = depth_anything_v1_pre_act_residual(
        hidden_state, fusion_hidden_size, f"{name}_res2", data_format
    )

    hidden_state = layers.Lambda(
        lambda x: depth_anything_v1_aligned_bilinear_resize(
            x, target_h, target_w, data_format
        ),
        name=f"{name}_upsample",
    )(hidden_state)

    hidden_state = layers.Conv2D(
        fusion_hidden_size,
        1,
        padding="valid",
        data_format=data_format,
        name=f"{name}_projection",
    )(hidden_state)

    return hidden_state


def depth_anything_v1_dino_backbone(
    pixel_values,
    backbone_dim,
    backbone_depth,
    backbone_num_heads,
    out_indices,
    patch_size,
    patch_h,
    patch_w,
    mlp_ratio,
    data_format,
    name="backbone",
):
    """Functional DINOv2 ViT backbone that returns intermediate features.

    Runs the standard DINOv2 stack (patch embed → CLS token → pos embed →
    transformer blocks → final LayerNorm), extracts the outputs of the
    blocks listed in ``out_indices``, strips the CLS token, and returns
    each feature as a spatial tensor in the requested ``data_format``.
    """
    x = layers.Conv2D(
        backbone_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name=f"{name}_patch_embed",
    )(pixel_values)
    x = layers.Reshape((-1, backbone_dim))(x)
    x = ClassDistToken(use_distillation=False, name=f"{name}_cls_token")(x)
    x = AddPositionEmbs(
        name=f"{name}_pos_embed",
        no_embed_class=False,
        use_distillation=False,
        grid_h=patch_h,
        grid_w=patch_w,
    )(x)

    all_features = []
    for i in range(backbone_depth):
        x_norm = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name=f"{name}_block_{i}_ln1"
        )(x)
        x_attn = MultiHeadSelfAttention(
            dim=backbone_dim,
            num_heads=backbone_num_heads,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.0,
            proj_drop=0.0,
            block_prefix=f"{name}_block_{i}",
            name=f"{name}_block_{i}_attn",
        )(x_norm)
        x_attn = LayerScale(init_values=1.0, name=f"{name}_block_{i}_ls1")(x_attn)
        x = layers.Add(name=f"{name}_block_{i}_add1")([x, x_attn])

        y_norm = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name=f"{name}_block_{i}_ln2"
        )(x)
        y_mlp = layers.Dense(
            int(backbone_dim * mlp_ratio), name=f"{name}_block_{i}_mlp_fc1"
        )(y_norm)
        y_mlp = layers.Activation("gelu", name=f"{name}_block_{i}_gelu")(y_mlp)
        y_mlp = layers.Dense(backbone_dim, name=f"{name}_block_{i}_mlp_fc2")(y_mlp)
        y_mlp = LayerScale(init_values=1.0, name=f"{name}_block_{i}_ls2")(y_mlp)
        x = layers.Add(name=f"{name}_block_{i}_add2")([x, y_mlp])

        if (i + 1) in out_indices:
            all_features.append(x)

    backbone_ln = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=f"{name}_layernorm"
    )
    backbone_features = []
    for i, feat in enumerate(all_features):
        feat = backbone_ln(feat)
        feat = layers.Lambda(lambda v: v[:, 1:], name=f"{name}_strip_cls_{i}")(feat)
        feat = layers.Reshape(
            (patch_h, patch_w, backbone_dim), name=f"{name}_reshape_{i}"
        )(feat)
        if data_format == "channels_first":
            feat = layers.Permute((3, 1, 2), name=f"{name}_permute_{i}")(feat)
        backbone_features.append(feat)
    return backbone_features


def depth_anything_v1_neck(
    backbone_features,
    reassemble_factors,
    neck_hidden_sizes,
    fusion_hidden_size,
    patch_h,
    patch_w,
    data_format,
    name="neck",
):
    """Functional DPT-style neck: reassemble + project + bottom-up fusion.

    Turns the 4 intermediate DINOv2 feature maps into a single fused
    tensor at 2x the coarsest reassembled resolution. Reassemble uses a
    1x1 projection followed by per-factor up/down sampling, projection
    applies a 3x3 conv to ``fusion_hidden_size``, and fusion walks the
    feature pyramid bottom-up through ``depth_anything_v1_fusion_block``.
    """
    reassembled = []
    for i, (feat, factor, out_ch) in enumerate(
        zip(backbone_features, reassemble_factors, neck_hidden_sizes)
    ):
        feat = layers.Conv2D(
            out_ch,
            1,
            padding="valid",
            data_format=data_format,
            name=f"{name}_reassemble_{i}_projection",
        )(feat)
        if factor > 1:
            feat = layers.Conv2DTranspose(
                out_ch,
                kernel_size=int(factor),
                strides=int(factor),
                padding="valid",
                data_format=data_format,
                name=f"{name}_reassemble_{i}_resize",
            )(feat)
        elif factor < 1:
            stride = int(1 / factor)
            feat = layers.Conv2D(
                out_ch,
                kernel_size=3,
                strides=stride,
                padding="same",
                data_format=data_format,
                name=f"{name}_reassemble_{i}_resize",
            )(feat)
        reassembled.append(feat)

    projected = []
    for i, feat in enumerate(reassembled):
        feat = layers.Conv2D(
            fusion_hidden_size,
            3,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{name}_conv_{i}",
        )(feat)
        projected.append(feat)

    reassembled_sizes = []
    for factor in reassemble_factors:
        if factor > 1:
            reassembled_sizes.append((patch_h * int(factor), patch_w * int(factor)))
        elif factor < 1:
            stride = int(1 / factor)
            reassembled_sizes.append((-(-patch_h // stride), -(-patch_w // stride)))
        else:
            reassembled_sizes.append((patch_h, patch_w))
    reversed_sizes = reassembled_sizes[::-1]
    features_reversed = projected[::-1]

    fused = None
    for idx in range(4):
        if idx < 3:
            target_h, target_w = reversed_sizes[idx + 1]
        else:
            target_h = reversed_sizes[idx][0] * 2
            target_w = reversed_sizes[idx][1] * 2

        fused = depth_anything_v1_fusion_block(
            features_reversed[idx] if fused is None else fused,
            None if fused is None else features_reversed[idx],
            target_h,
            target_w,
            fusion_hidden_size,
            data_format,
            name=f"{name}_fusion_{idx}",
        )
    return fused


def depth_anything_v1_head(
    fused,
    height,
    width,
    fusion_hidden_size,
    head_hidden_size,
    depth_estimation_type,
    max_depth,
    data_format,
    name="head",
):
    """Functional depth-estimation head.

    Three convolutions with an aligned-corners bilinear upsample to the
    input resolution between the first and second conv. ``relative``
    models end in a ReLU; ``metric`` models end in a sigmoid scaled by
    ``max_depth``.
    """
    x = layers.Conv2D(
        fusion_hidden_size // 2,
        3,
        padding="same",
        data_format=data_format,
        name=f"{name}_conv1",
    )(fused)
    x = layers.Lambda(
        lambda z: depth_anything_v1_aligned_bilinear_resize(
            z, height, width, data_format
        ),
        name=f"{name}_upsample",
    )(x)
    x = layers.Conv2D(
        head_hidden_size,
        3,
        padding="same",
        data_format=data_format,
        name=f"{name}_conv2",
    )(x)
    x = layers.Activation("relu", name=f"{name}_act1")(x)
    x = layers.Conv2D(
        1, 1, padding="valid", data_format=data_format, name=f"{name}_conv3"
    )(x)

    if depth_estimation_type == "metric":
        x = layers.Activation("sigmoid", name=f"{name}_act2")(x)
        x = layers.Lambda(lambda z: z * max_depth, name=f"{name}_scale_depth")(x)
    else:
        x = layers.Activation("relu", name=f"{name}_act2")(x)
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class DepthAnythingV1(keras.Model):
    PATCH_SIZE = 14
    IMAGE_SIZE = 518
    MLP_RATIO = 4.0
    HEAD_HIDDEN_SIZE = 32

    def __init__(
        self,
        backbone_dim=384,
        backbone_depth=12,
        backbone_num_heads=6,
        out_indices=None,
        neck_hidden_sizes=None,
        fusion_hidden_size=64,
        reassemble_factors=None,
        depth_estimation_type="relative",
        max_depth=1.0,
        input_shape=None,
        input_tensor=None,
        name="DepthAnythingV1",
        **kwargs,
    ):
        if out_indices is None:
            out_indices = [9, 10, 11, 12]
        if neck_hidden_sizes is None:
            neck_hidden_sizes = [48, 96, 192, 384]
        if reassemble_factors is None:
            reassemble_factors = [4, 2, 1, 0.5]

        data_format = keras.config.image_data_format()

        if input_shape is None:
            if data_format == "channels_first":
                input_shape = (3, self.IMAGE_SIZE, self.IMAGE_SIZE)
            else:
                input_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)

        if input_tensor is not None:
            if not keras.utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=input_shape, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=input_shape, name="pixel_values")

        if data_format == "channels_first":
            height, width = input_shape[1], input_shape[2]
        else:
            height, width = input_shape[0], input_shape[1]

        patch_h = height // self.PATCH_SIZE
        patch_w = width // self.PATCH_SIZE

        backbone_features = depth_anything_v1_dino_backbone(
            pixel_values,
            backbone_dim=backbone_dim,
            backbone_depth=backbone_depth,
            backbone_num_heads=backbone_num_heads,
            out_indices=out_indices,
            patch_size=self.PATCH_SIZE,
            patch_h=patch_h,
            patch_w=patch_w,
            mlp_ratio=self.MLP_RATIO,
            data_format=data_format,
            name="backbone",
        )

        fused = depth_anything_v1_neck(
            backbone_features,
            reassemble_factors=reassemble_factors,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            patch_h=patch_h,
            patch_w=patch_w,
            data_format=data_format,
            name="neck",
        )

        predicted_depth = depth_anything_v1_head(
            fused,
            height=height,
            width=width,
            fusion_hidden_size=fusion_hidden_size,
            head_hidden_size=self.HEAD_HIDDEN_SIZE,
            depth_estimation_type=depth_estimation_type,
            max_depth=max_depth,
            data_format=data_format,
            name="head",
        )

        super().__init__(
            inputs=pixel_values,
            outputs=predicted_depth,
            name=name,
            **kwargs,
        )

        self.backbone_dim = backbone_dim
        self.backbone_depth = backbone_depth
        self.backbone_num_heads = backbone_num_heads
        self.out_indices = list(out_indices)
        self.neck_hidden_sizes = list(neck_hidden_sizes)
        self.fusion_hidden_size = fusion_hidden_size
        self.reassemble_factors = list(reassemble_factors)
        self.depth_estimation_type = depth_estimation_type
        self.max_depth = max_depth
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_dim": self.backbone_dim,
                "backbone_depth": self.backbone_depth,
                "backbone_num_heads": self.backbone_num_heads,
                "out_indices": self.out_indices,
                "neck_hidden_sizes": self.neck_hidden_sizes,
                "fusion_hidden_size": self.fusion_hidden_size,
                "reassemble_factors": self.reassemble_factors,
                "depth_estimation_type": self.depth_estimation_type,
                "max_depth": self.max_depth,
                "input_shape": self._input_shape_val,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_depth_anything_v1(variant, input_shape, input_tensor, weights, **kwargs):
    config = DEPTH_ANYTHING_V1_MODEL_CONFIG[variant]

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

    if weights in get_all_weight_names(DEPTH_ANYTHING_V1_WEIGHTS_CONFIG):
        load_weights_from_config(
            variant, weights, model, DEPTH_ANYTHING_V1_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DepthAnythingV1Small(input_shape=None, input_tensor=None, weights=None, **kwargs):
    return _create_depth_anything_v1(
        "DepthAnythingV1Small", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def DepthAnythingV1Base(input_shape=None, input_tensor=None, weights=None, **kwargs):
    return _create_depth_anything_v1(
        "DepthAnythingV1Base", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def DepthAnythingV1Large(input_shape=None, input_tensor=None, weights=None, **kwargs):
    return _create_depth_anything_v1(
        "DepthAnythingV1Large", input_shape, input_tensor, weights, **kwargs
    )
