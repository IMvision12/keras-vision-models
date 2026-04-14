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


def _aligned_bilinear_resize(x, target_h, target_w):
    """Bilinear resize matching PyTorch align_corners=True."""
    backend_name = keras.backend.backend()
    if backend_name == "torch":
        import torch.nn.functional as F

        x = ops.transpose(x, [0, 3, 1, 2])
        x = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=True
        )
        return ops.transpose(x, [0, 2, 3, 1])
    return ops.image.resize(x, size=(target_h, target_w), method="bilinear")


def depth_anything_v1_pre_act_residual(x, channels, name):
    residual = x
    x = layers.Activation("relu", name=f"{name}_act1")(x)
    x = layers.Conv2D(
        channels, 3, padding="same", data_format="channels_last", name=f"{name}_conv1"
    )(x)
    x = layers.Activation("relu", name=f"{name}_act2")(x)
    x = layers.Conv2D(
        channels, 3, padding="same", data_format="channels_last", name=f"{name}_conv2"
    )(x)
    return layers.Add(name=f"{name}_add")([x, residual])


def depth_anything_v1_fusion_block(
    hidden_state, residual, target_h, target_w, fusion_hidden_size, idx
):
    name = f"neck_fusion_{idx}"

    if residual is not None:
        hidden_state = layers.Add(name=f"{name}_add_residual")(
            [
                hidden_state,
                depth_anything_v1_pre_act_residual(
                    residual, fusion_hidden_size, f"{name}_res1"
                ),
            ]
        )

    hidden_state = depth_anything_v1_pre_act_residual(
        hidden_state, fusion_hidden_size, f"{name}_res2"
    )

    hidden_state = layers.Lambda(
        lambda x: _aligned_bilinear_resize(x, target_h, target_w),
        name=f"{name}_upsample",
    )(hidden_state)

    hidden_state = layers.Conv2D(
        fusion_hidden_size,
        1,
        padding="valid",
        data_format="channels_last",
        name=f"{name}_projection",
    )(hidden_state)

    return hidden_state


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

        # --- DINOv2 Backbone ---
        if data_format == "channels_first":
            height, width = input_shape[1], input_shape[2]
        else:
            height, width = input_shape[0], input_shape[1]

        patch_h = height // self.PATCH_SIZE
        patch_w = width // self.PATCH_SIZE

        x = layers.Conv2D(
            backbone_dim,
            kernel_size=self.PATCH_SIZE,
            strides=self.PATCH_SIZE,
            padding="valid",
            data_format=data_format,
            name="backbone_patch_embed",
        )(pixel_values)
        x = layers.Reshape((-1, backbone_dim))(x)
        x = ClassDistToken(use_distillation=False, name="backbone_cls_token")(x)
        x = AddPositionEmbs(
            name="backbone_pos_embed",
            no_embed_class=False,
            use_distillation=False,
            grid_h=patch_h,
            grid_w=patch_w,
        )(x)

        all_features = []
        for i in range(backbone_depth):
            x_norm = layers.LayerNormalization(
                epsilon=1e-6, axis=-1, name=f"backbone_block_{i}_ln1"
            )(x)
            x_attn = MultiHeadSelfAttention(
                dim=backbone_dim,
                num_heads=backbone_num_heads,
                qkv_bias=True,
                qk_norm=False,
                attn_drop=0.0,
                proj_drop=0.0,
                block_prefix=f"backbone_block_{i}",
                name=f"backbone_block_{i}_attn",
            )(x_norm)
            x_attn = LayerScale(init_values=1.0, name=f"backbone_block_{i}_ls1")(x_attn)
            x = layers.Add(name=f"backbone_block_{i}_add1")([x, x_attn])

            y_norm = layers.LayerNormalization(
                epsilon=1e-6, axis=-1, name=f"backbone_block_{i}_ln2"
            )(x)
            y_mlp = layers.Dense(
                int(backbone_dim * self.MLP_RATIO),
                name=f"backbone_block_{i}_mlp_fc1",
            )(y_norm)
            y_mlp = layers.Activation("gelu", name=f"backbone_block_{i}_gelu")(y_mlp)
            y_mlp = layers.Dense(backbone_dim, name=f"backbone_block_{i}_mlp_fc2")(
                y_mlp
            )
            y_mlp = LayerScale(init_values=1.0, name=f"backbone_block_{i}_ls2")(y_mlp)
            x = layers.Add(name=f"backbone_block_{i}_add2")([x, y_mlp])

            block_num = i + 1
            if block_num in out_indices:
                all_features.append(x)

        # Apply shared LayerNorm to each extracted feature
        backbone_ln = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name="backbone_layernorm"
        )
        backbone_features = []
        for i, feat in enumerate(all_features):
            feat = backbone_ln(feat)
            # Strip CLS token: (B, 1+H*W, C) -> (B, H*W, C)
            feat = layers.Lambda(lambda v: v[:, 1:], name=f"backbone_strip_cls_{i}")(
                feat
            )
            # Reshape to spatial: (B, H*W, C) -> (B, H, W, C)
            feat = layers.Reshape(
                (patch_h, patch_w, backbone_dim), name=f"backbone_reshape_{i}"
            )(feat)
            backbone_features.append(feat)

        # --- Neck: Reassemble ---
        reassembled = []
        for i, (feat, factor, out_ch) in enumerate(
            zip(backbone_features, reassemble_factors, neck_hidden_sizes)
        ):
            # 1x1 projection
            feat = layers.Conv2D(
                out_ch,
                1,
                padding="valid",
                data_format="channels_last",
                name=f"neck_reassemble_{i}_projection",
            )(feat)

            # Resize
            if factor > 1:
                feat = layers.Conv2DTranspose(
                    out_ch,
                    kernel_size=int(factor),
                    strides=int(factor),
                    padding="valid",
                    data_format="channels_last",
                    name=f"neck_reassemble_{i}_resize",
                )(feat)
            elif factor < 1:
                stride = int(1 / factor)
                feat = layers.Conv2D(
                    out_ch,
                    kernel_size=3,
                    strides=stride,
                    padding="same",
                    data_format="channels_last",
                    name=f"neck_reassemble_{i}_resize",
                )(feat)
            # factor == 1: Identity, no resize

            reassembled.append(feat)

        # --- Neck: Project to fusion_hidden_size ---
        projected = []
        for i, feat in enumerate(reassembled):
            feat = layers.Conv2D(
                fusion_hidden_size,
                3,
                padding="same",
                use_bias=False,
                data_format="channels_last",
                name=f"neck_conv_{i}",
            )(feat)
            projected.append(feat)

        # --- Neck: Fusion (reverse order, bottom-up) ---
        # Compute static spatial sizes for reassembled features
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

            if fused is None:
                fused = depth_anything_v1_fusion_block(
                    features_reversed[idx],
                    None,
                    target_h,
                    target_w,
                    fusion_hidden_size,
                    idx,
                )
            else:
                fused = depth_anything_v1_fusion_block(
                    fused,
                    features_reversed[idx],
                    target_h,
                    target_w,
                    fusion_hidden_size,
                    idx,
                )

        # --- Head ---
        head_in = fused
        head_in = layers.Conv2D(
            fusion_hidden_size // 2,
            3,
            padding="same",
            data_format="channels_last",
            name="head_conv1",
        )(head_in)

        head_in = layers.Lambda(
            lambda x: _aligned_bilinear_resize(x, height, width),
            name="head_upsample",
        )(head_in)

        head_in = layers.Conv2D(
            self.HEAD_HIDDEN_SIZE,
            3,
            padding="same",
            data_format="channels_last",
            name="head_conv2",
        )(head_in)
        head_in = layers.Activation("relu", name="head_act1")(head_in)

        head_in = layers.Conv2D(
            1,
            1,
            padding="valid",
            data_format="channels_last",
            name="head_conv3",
        )(head_in)

        if depth_estimation_type == "metric":
            head_in = layers.Activation("sigmoid", name="head_act2")(head_in)
            predicted_depth = layers.Lambda(
                lambda x: x * max_depth, name="head_scale_depth"
            )(head_in)
        else:
            predicted_depth = layers.Activation("relu", name="head_act2")(head_in)

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
