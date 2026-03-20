import keras
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import RF_DETR_MODEL_CONFIG, RF_DETR_WEIGHTS_CONFIG
from .rf_detr_layers import (
    C2f,
    ChannelLayerNorm,
    DinoV2Embeddings,
    PositionEmbeddingSine,
    RFDETRDecoderLayer,
    SinePositionEmbeddingForRefPoints,
    WindowedDinoV2Encoder,
)


@keras.saving.register_keras_serializable(package="kmodels")
class UnwindowFeatures(layers.Layer):
    """Remove register/CLS tokens, undo windowing, reshape to spatial (B, H, W, C)."""

    def __init__(
        self, num_h, num_w, num_windows, hidden_size, num_register_tokens, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_h = num_h
        self.num_w = num_w
        self.num_windows = num_windows
        self.hidden_size = hidden_size
        self.num_register_tokens = num_register_tokens

    def call(self, hidden_state):
        hidden_state = hidden_state[:, self.num_register_tokens + 1 :, :]

        if self.num_windows > 1:
            nw2 = self.num_windows**2
            shape = ops.shape(hidden_state)
            HW_win = shape[1]
            C = shape[2]

            hidden_state = ops.reshape(hidden_state, [-1, nw2 * HW_win, C])
            h_pw = self.num_h // self.num_windows
            w_pw = self.num_w // self.num_windows
            hidden_state = ops.reshape(
                hidden_state, [-1, self.num_windows, h_pw, w_pw, C]
            )
            hidden_state = ops.transpose(hidden_state, [0, 2, 1, 3, 4])

        hidden_state = ops.reshape(
            hidden_state, [-1, self.num_h, self.num_w, self.hidden_size]
        )
        return hidden_state

    def compute_output_spec(self, input_spec, **kwargs):
        return keras.KerasTensor(
            shape=(None, self.num_h, self.num_w, self.hidden_size),
            dtype=input_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_h": self.num_h,
                "num_w": self.num_w,
                "num_windows": self.num_windows,
                "hidden_size": self.hidden_size,
                "num_register_tokens": self.num_register_tokens,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EncoderOutputProposals(layers.Layer):
    """Generate encoder output proposals for two-stage detection."""

    def __init__(self, spatial_shapes, bbox_reparam=True, **kwargs):
        super().__init__(**kwargs)
        self.spatial_shapes = spatial_shapes
        self.bbox_reparam = bbox_reparam

    def call(self, memory):
        proposals = []
        for lvl, (H_, W_) in enumerate(self.spatial_shapes):
            y_range = ops.cast(ops.arange(H_), "float32")
            x_range = ops.cast(ops.arange(W_), "float32")
            grid_y, grid_x = ops.meshgrid(y_range, x_range, indexing="ij")
            grid = ops.stack([grid_x, grid_y], axis=-1)
            grid = ops.reshape(grid, [1, H_ * W_, 2])

            scale = ops.convert_to_tensor([[W_, H_]], dtype="float32")
            scale = ops.reshape(scale, [1, 1, 2])
            grid = (grid + 0.5) / scale

            wh = ops.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = ops.concatenate([grid, wh], axis=-1)
            proposals.append(proposal)

        output_proposals = ops.concatenate(proposals, axis=1)
        batch = ops.shape(memory)[0]
        output_proposals = ops.broadcast_to(
            output_proposals, [batch, ops.shape(output_proposals)[1], 4]
        )

        valid = ops.all(
            (output_proposals > 0.01) & (output_proposals < 0.99),
            axis=-1,
            keepdims=True,
        )

        if self.bbox_reparam:
            output_proposals = ops.where(
                valid, output_proposals, ops.zeros_like(output_proposals)
            )
        else:
            eps = 1e-7
            clamped = ops.clip(output_proposals, eps, 1.0 - eps)
            unsig = ops.log(clamped / (1.0 - clamped))
            inf_val = ops.full_like(unsig, 1e8)
            output_proposals = ops.where(valid, unsig, inf_val)

        output_memory = ops.where(valid, memory, ops.zeros_like(memory))
        return output_memory, output_proposals

    def compute_output_spec(self, memory_spec, **kwargs):
        total = sum(h * w for h, w in self.spatial_shapes)
        return (
            keras.KerasTensor(shape=memory_spec.shape, dtype=memory_spec.dtype),
            keras.KerasTensor(
                shape=(memory_spec.shape[0], total, 4), dtype=memory_spec.dtype
            ),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spatial_shapes": self.spatial_shapes,
                "bbox_reparam": self.bbox_reparam,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class LearnedEmbedding(layers.Layer):
    """A layer that holds a learned embedding and broadcasts it to batch size."""

    def __init__(self, num_embeddings, embedding_dim, initializer="zeros", **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.initializer = initializer

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=self.initializer,
        )
        self.built = True

    def call(self, x):
        batch_size = ops.shape(x)[0]
        return ops.repeat(ops.expand_dims(self.weight, axis=0), batch_size, axis=0)

    def compute_output_spec(self, input_spec, **kwargs):
        return keras.KerasTensor(
            shape=(None, self.num_embeddings, self.embedding_dim),
            dtype=input_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "initializer": self.initializer,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class TwoStageRefineRefpoints(layers.Layer):
    """Refine encoder proposals with learned refpoint deltas (bbox_reparam)."""

    def __init__(self, bbox_reparam=True, num_queries=300, **kwargs):
        super().__init__(**kwargs)
        self.bbox_reparam = bbox_reparam
        self.num_queries = num_queries

    def call(self, inputs):
        refpoint_embed, refpoint_embed_ts = inputs
        refpoint_embed_subset = refpoint_embed[:, : self.num_queries, :]
        if self.bbox_reparam:
            ref_cxcy = (
                refpoint_embed_subset[..., :2] * refpoint_embed_ts[..., 2:]
                + refpoint_embed_ts[..., :2]
            )
            ref_wh = (
                ops.exp(refpoint_embed_subset[..., 2:]) * refpoint_embed_ts[..., 2:]
            )
            return ops.concatenate([ref_cxcy, ref_wh], axis=-1)
        else:
            return refpoint_embed_subset + refpoint_embed_ts

    def compute_output_spec(self, input_spec, **kwargs):
        return keras.KerasTensor(
            shape=(None, self.num_queries, 4),
            dtype=input_spec[0].dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bbox_reparam": self.bbox_reparam,
                "num_queries": self.num_queries,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class RFDETR(keras.Model):
    """RF-DETR: Real-Time Detection Transformer.

    A real-time object detection model based on DINOv2 backbone with windowed
    attention, a simple projector, and a deformable DETR decoder. Uses two-stage
    query initialization and iterative bounding box refinement.

    Reference:
        - RF-DETR (Roboflow, ICLR 2026)
        - https://github.com/roboflow/rf-detr

    Args:
        hidden_dim: Transformer decoder hidden dimension.
        backbone_hidden_size: DINOv2 backbone hidden size.
        backbone_num_heads: Number of attention heads in backbone.
        backbone_num_layers: Number of transformer layers in backbone.
        backbone_mlp_ratio: MLP expansion ratio in backbone.
        backbone_use_swiglu: Whether backbone uses SwiGLU FFN.
        num_register_tokens: Number of register tokens in DINOv2.
        out_feature_indexes: Backbone layer indices to extract features from.
        patch_size: Patch size for DINOv2 patch embeddings.
        num_windows: Number of windows for windowed attention.
        positional_encoding_size: Size of positional encoding grid.
        resolution: Input image resolution.
        dec_layers: Number of decoder layers.
        sa_nheads: Number of self-attention heads in decoder.
        ca_nheads: Number of cross-attention heads in decoder.
        dec_n_points: Number of sampling points in deformable attention.
        num_queries: Number of object queries.
        num_classes: Number of object classes (COCO: 91).
        two_stage: Whether to use two-stage query initialization.
        bbox_reparam: Whether to use bbox reparameterization.
        lite_refpoint_refine: Whether to use lite reference point refinement.
        group_detr: Number of DETR groups (training only, inference uses 1).
        dim_feedforward: FFN dimension in decoder.
        weights: Pre-trained weight identifier or file path.
        input_shape: Input image shape as (H, W, C).
        input_tensor: Optional input Keras tensor.
        name: Model name.
    """

    def __init__(
        self,
        hidden_dim=256,
        backbone_hidden_size=384,
        backbone_num_heads=6,
        backbone_num_layers=12,
        backbone_mlp_ratio=4,
        backbone_use_swiglu=True,
        num_register_tokens=4,
        out_feature_indexes=None,
        patch_size=14,
        num_windows=4,
        positional_encoding_size=37,
        resolution=560,
        dec_layers=3,
        sa_nheads=8,
        ca_nheads=16,
        dec_n_points=2,
        num_queries=300,
        num_classes=91,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        group_detr=13,
        dim_feedforward=2048,
        weights="coco",
        input_shape=None,
        input_tensor=None,
        name="RFDETR",
        **kwargs,
    ):
        if out_feature_indexes is None:
            out_feature_indexes = [2, 5, 8, 11]

        if input_shape is None:
            input_shape = (resolution, resolution, 3)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input

        # --- DINOv2 Backbone ---
        embeddings = DinoV2Embeddings(
            hidden_size=backbone_hidden_size,
            patch_size=patch_size,
            num_channels=3,
            num_register_tokens=num_register_tokens,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            name="backbone_embeddings",
        )(inputs)

        window_block_indexes = sorted(
            set(range(max(out_feature_indexes) + 1)) - set(out_feature_indexes)
        )

        encoder_features = WindowedDinoV2Encoder(
            hidden_size=backbone_hidden_size,
            num_heads=backbone_num_heads,
            num_layers=backbone_num_layers,
            mlp_ratio=backbone_mlp_ratio,
            use_swiglu=backbone_use_swiglu,
            num_windows=num_windows,
            out_feature_indexes=out_feature_indexes,
            window_block_indexes=window_block_indexes,
            name="backbone_encoder",
        )(embeddings)

        num_h = input_shape[0] // patch_size
        num_w = input_shape[1] // patch_size

        unwindowed_features = []
        for i, feat in enumerate(encoder_features):
            uw = UnwindowFeatures(
                num_h,
                num_w,
                num_windows,
                backbone_hidden_size,
                num_register_tokens,
                name=f"unwindow_{i}",
            )(feat)
            unwindowed_features.append(uw)

        # --- Projector: concatenate all multi-scale features + C2f + LayerNorm ---
        concat_feat = layers.Concatenate(axis=-1, name="concat_features")(
            unwindowed_features
        )
        projected = C2f(
            hidden_dim,
            num_blocks=3,
            shortcut=False,
            expansion=0.5,
            activation="silu",
            use_layer_norm=True,
            name="projector_c2f",
        )(concat_feat)
        projected = ChannelLayerNorm(name="projector_ln")(projected)

        # --- Position Encoding ---
        pos_embed = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2,
            normalize=True,
            name="position_embedding",
        )(projected)

        proj_shape = (input_shape[0] // patch_size, input_shape[1] // patch_size)
        num_feature_levels = 1
        spatial_shapes = [proj_shape]
        level_start_index = [0]

        # Flatten to (B, H*W, C)
        src_flat = ops.reshape(
            projected, [-1, proj_shape[0] * proj_shape[1], hidden_dim]
        )
        pos_flat = ops.reshape(
            pos_embed, [1, proj_shape[0] * proj_shape[1], hidden_dim]
        )
        pos_flat = ops.broadcast_to(pos_flat, ops.shape(src_flat))

        memory = src_flat

        # --- Learned query embeddings (used in both two-stage and single-stage) ---
        tgt = LearnedEmbedding(
            num_queries,
            hidden_dim,
            initializer="glorot_uniform",
            name="query_feat_embed",
        )(memory)
        refpoint_embed = LearnedEmbedding(
            num_queries,
            4,
            initializer="zeros",
            name="refpoint_embed_layer",
        )(memory)

        # --- Two-Stage Query Initialization ---
        if two_stage:
            output_memory_filtered, output_proposals = EncoderOutputProposals(
                spatial_shapes=spatial_shapes,
                bbox_reparam=bbox_reparam,
                name="encoder_proposals",
            )(memory)

            enc_output_proj = layers.Dense(hidden_dim, name="enc_output_0")(
                output_memory_filtered
            )
            enc_output_norm = layers.LayerNormalization(
                epsilon=1e-5, name="enc_output_norm_0"
            )
            output_memory_proj = enc_output_norm(enc_output_proj)

            enc_cls = layers.Dense(num_classes, name="enc_out_class_embed_0")(
                output_memory_proj
            )

            bbox_embed_enc_0 = layers.Dense(
                hidden_dim, activation="relu", name="enc_bbox_0"
            )
            bbox_embed_enc_1 = layers.Dense(
                hidden_dim, activation="relu", name="enc_bbox_1"
            )
            bbox_embed_enc_2 = layers.Dense(4, name="enc_bbox_2")

            enc_bbox_delta = bbox_embed_enc_2(
                bbox_embed_enc_1(bbox_embed_enc_0(output_memory_proj))
            )

            if bbox_reparam:
                enc_coord_cxcy = (
                    enc_bbox_delta[..., :2] * output_proposals[..., 2:]
                    + output_proposals[..., :2]
                )
                enc_coord_wh = (
                    ops.exp(enc_bbox_delta[..., 2:]) * output_proposals[..., 2:]
                )
                enc_coords = ops.concatenate([enc_coord_cxcy, enc_coord_wh], axis=-1)
            else:
                enc_coords = enc_bbox_delta + output_proposals

            enc_cls_max = ops.max(enc_cls, axis=-1)
            topk_indices = ops.top_k(
                enc_cls_max, k=min(num_queries, proj_shape[0] * proj_shape[1])
            )[1]
            topk_idx_4 = ops.repeat(ops.expand_dims(topk_indices, axis=-1), 4, axis=-1)
            refpoint_embed_ts = ops.take_along_axis(enc_coords, topk_idx_4, axis=1)
            refpoint_embed_ts = ops.stop_gradient(refpoint_embed_ts)

            refpoints_unsigmoid = TwoStageRefineRefpoints(
                bbox_reparam=bbox_reparam,
                num_queries=num_queries,
                name="refine_refpoints",
            )([refpoint_embed, refpoint_embed_ts])
        else:
            refpoints_unsigmoid = refpoint_embed

        # --- Ref point head ---
        ref_point_head_0 = layers.Dense(
            hidden_dim, activation="relu", name="ref_point_head_0"
        )
        ref_point_head_1 = layers.Dense(hidden_dim, name="ref_point_head_1")

        # --- Decoder ---
        decoder_layers_list = []
        for i in range(dec_layers):
            decoder_layers_list.append(
                RFDETRDecoderLayer(
                    d_model=hidden_dim,
                    sa_nhead=sa_nheads,
                    ca_nhead=ca_nheads,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    num_feature_levels=num_feature_levels,
                    dec_n_points=dec_n_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    name=f"decoder_layer_{i}",
                )
            )

        decoder_norm = layers.LayerNormalization(epsilon=1e-5, name="decoder_norm")

        # Bbox embed (shared across layers for iterative refinement)
        bbox_embed_0 = layers.Dense(hidden_dim, activation="relu", name="bbox_embed_0")
        bbox_embed_1 = layers.Dense(hidden_dim, activation="relu", name="bbox_embed_1")
        bbox_embed_2 = layers.Dense(4, name="bbox_embed_2")

        level_start_index = [0]

        sine_embed_layer = SinePositionEmbeddingForRefPoints(
            dim=hidden_dim // 2, name="sine_pos_embed"
        )

        if bbox_reparam:
            ref_for_query = refpoints_unsigmoid
        else:
            ref_for_query = ops.sigmoid(refpoints_unsigmoid)

        if lite_refpoint_refine:
            query_sine = sine_embed_layer(ref_for_query[..., :4])
            query_pos = ref_point_head_1(ref_point_head_0(query_sine))

        output = tgt
        for layer_id, dec_layer in enumerate(decoder_layers_list):
            if not lite_refpoint_refine:
                if bbox_reparam:
                    ref_for_query = refpoints_unsigmoid
                else:
                    ref_for_query = ops.sigmoid(refpoints_unsigmoid)
                query_sine = sine_embed_layer(ref_for_query[..., :4])
                query_pos = ref_point_head_1(ref_point_head_0(query_sine))

            if bbox_reparam:
                ref_input = refpoints_unsigmoid[:, :, None, :]
            else:
                sig_ref = ops.sigmoid(refpoints_unsigmoid)
                ref_input = sig_ref[:, :, None, :]

            output = dec_layer(
                output,
                memory,
                query_pos,
                ref_input,
                training=False,
            )

            if not lite_refpoint_refine:
                new_delta = bbox_embed_2(bbox_embed_1(bbox_embed_0(output)))
                if bbox_reparam:
                    new_cxcy = (
                        new_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                        + refpoints_unsigmoid[..., :2]
                    )
                    new_wh = ops.exp(new_delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
                    refpoints_unsigmoid = ops.concatenate([new_cxcy, new_wh], axis=-1)
                else:
                    refpoints_unsigmoid = refpoints_unsigmoid + new_delta
                refpoints_unsigmoid = ops.stop_gradient(refpoints_unsigmoid)

        output = decoder_norm(output)

        # --- Prediction Heads ---
        class_embed = layers.Dense(num_classes, name="class_embed")
        pred_logits = class_embed(output)

        pred_bbox_delta = bbox_embed_2(bbox_embed_1(bbox_embed_0(output)))
        if bbox_reparam:
            pred_cxcy = (
                pred_bbox_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                + refpoints_unsigmoid[..., :2]
            )
            pred_wh = ops.exp(pred_bbox_delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
            pred_boxes = ops.concatenate([pred_cxcy, pred_wh], axis=-1)
        else:
            pred_boxes = ops.sigmoid(pred_bbox_delta + refpoints_unsigmoid)

        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_num_heads = backbone_num_heads
        self.backbone_num_layers = backbone_num_layers
        self.backbone_mlp_ratio = backbone_mlp_ratio
        self.backbone_use_swiglu = backbone_use_swiglu
        self.num_register_tokens = num_register_tokens
        self.out_feature_indexes = out_feature_indexes
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.positional_encoding_size = positional_encoding_size
        self.resolution = resolution
        self.dec_layers = dec_layers
        self.sa_nheads = sa_nheads
        self.ca_nheads = ca_nheads
        self.dec_n_points = dec_n_points
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.two_stage = two_stage
        self.bbox_reparam = bbox_reparam
        self.lite_refpoint_refine = lite_refpoint_refine
        self.group_detr = group_detr
        self.dim_feedforward = dim_feedforward
        self._input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "backbone_hidden_size": self.backbone_hidden_size,
                "backbone_num_heads": self.backbone_num_heads,
                "backbone_num_layers": self.backbone_num_layers,
                "backbone_mlp_ratio": self.backbone_mlp_ratio,
                "backbone_use_swiglu": self.backbone_use_swiglu,
                "num_register_tokens": self.num_register_tokens,
                "out_feature_indexes": self.out_feature_indexes,
                "patch_size": self.patch_size,
                "num_windows": self.num_windows,
                "positional_encoding_size": self.positional_encoding_size,
                "resolution": self.resolution,
                "dec_layers": self.dec_layers,
                "sa_nheads": self.sa_nheads,
                "ca_nheads": self.ca_nheads,
                "dec_n_points": self.dec_n_points,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "two_stage": self.two_stage,
                "bbox_reparam": self.bbox_reparam,
                "lite_refpoint_refine": self.lite_refpoint_refine,
                "group_detr": self.group_detr,
                "dim_feedforward": self.dim_feedforward,
                "input_shape": self.input_shape[1:],
                "input_tensor": self._input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ---------------------------------------------------------------------------
# Factory + Registration
# ---------------------------------------------------------------------------


def _create_rf_detr_model(
    variant,
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name=None,
    **kwargs,
):
    config = RF_DETR_MODEL_CONFIG[variant]

    if input_shape is None:
        res = config["resolution"]
        input_shape = (res, res, 3)

    model = RFDETR(
        hidden_dim=config["hidden_dim"],
        backbone_hidden_size=config["backbone_hidden_size"],
        backbone_num_heads=config["backbone_num_heads"],
        backbone_num_layers=config["backbone_num_layers"],
        backbone_mlp_ratio=config["backbone_mlp_ratio"],
        backbone_use_swiglu=config["backbone_use_swiglu"],
        num_register_tokens=config["num_register_tokens"],
        out_feature_indexes=config["out_feature_indexes"],
        patch_size=config["patch_size"],
        num_windows=config["num_windows"],
        positional_encoding_size=config["positional_encoding_size"],
        resolution=config["resolution"],
        dec_layers=config["dec_layers"],
        sa_nheads=config["sa_nheads"],
        ca_nheads=config["ca_nheads"],
        dec_n_points=config["dec_n_points"],
        num_queries=num_queries,
        num_classes=num_classes,
        two_stage=config["two_stage"],
        bbox_reparam=config["bbox_reparam"],
        lite_refpoint_refine=config["lite_refpoint_refine"],
        group_detr=config["group_detr"],
        dim_feedforward=config["dim_feedforward"],
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name or variant,
        **kwargs,
    )

    if weights in RF_DETR_WEIGHTS_CONFIG.get(variant, {}):
        load_weights_from_config(variant, weights, model, RF_DETR_WEIGHTS_CONFIG)
    elif weights is not None and weights != "coco":
        model.load_weights(weights)

    return model


@register_model
def RFDETRNano(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRNano",
    **kwargs,
):
    """RF-DETR Nano -- smallest variant.

    DINOv2-Small backbone, 2 decoder layers, 384px resolution.
    COCO AP: 48.0

    Args:
        num_queries: Number of object queries. Default 300.
        num_classes: Number of object classes (COCO: 91). Default 91.
        weights: Pre-trained weights identifier or file path. Default "coco".
        input_shape: Input shape as (H, W, C). Default (384, 384, 3).
        input_tensor: Optional input tensor.
        name: Model name.

    Returns:
        RF-DETR Nano model.

    Example:
        ```python
        model = kmodels.models.rf_detr.RFDETRNano(weights=None)
        output = model(np.random.rand(1, 384, 384, 3).astype("float32"))
        print(output["pred_logits"].shape)   # (1, 300, 92)
        print(output["pred_boxes"].shape)    # (1, 300, 4)
        ```
    """
    return _create_rf_detr_model(
        "RFDETRNano",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def RFDETRSmall(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRSmall",
    **kwargs,
):
    """RF-DETR Small.

    DINOv2-Small backbone, 3 decoder layers, 512px resolution.

    Args:
        num_queries: Number of object queries. Default 300.
        num_classes: Number of object classes (COCO: 91). Default 91.
        weights: Pre-trained weights identifier or file path. Default "coco".
        input_shape: Input shape as (H, W, C). Default (512, 512, 3).
        input_tensor: Optional input tensor.
        name: Model name.

    Returns:
        RF-DETR Small model.
    """
    return _create_rf_detr_model(
        "RFDETRSmall",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def RFDETRMedium(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRMedium",
    **kwargs,
):
    """RF-DETR Medium.

    DINOv2-Small backbone, 4 decoder layers, 576px resolution.

    Args:
        num_queries: Number of object queries. Default 300.
        num_classes: Number of object classes (COCO: 91). Default 91.
        weights: Pre-trained weights identifier or file path. Default "coco".
        input_shape: Input shape as (H, W, C). Default (576, 576, 3).
        input_tensor: Optional input tensor.
        name: Model name.

    Returns:
        RF-DETR Medium model.
    """
    return _create_rf_detr_model(
        "RFDETRMedium",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def RFDETRBase(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRBase",
    **kwargs,
):
    """RF-DETR Base.

    DINOv2-Small backbone, 3 decoder layers, 560px resolution, patch_size=14.

    Args:
        num_queries: Number of object queries. Default 300.
        num_classes: Number of object classes (COCO: 91). Default 91.
        weights: Pre-trained weights identifier or file path. Default "coco".
        input_shape: Input shape as (H, W, C). Default (560, 560, 3).
        input_tensor: Optional input tensor.
        name: Model name.

    Returns:
        RF-DETR Base model.
    """
    return _create_rf_detr_model(
        "RFDETRBase",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def RFDETRLarge(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRLarge",
    **kwargs,
):
    """RF-DETR Large.

    DINOv2-Small backbone, 4 decoder layers, 704px resolution.
    First real-time detector to surpass 60 AP on COCO.

    Args:
        num_queries: Number of object queries. Default 300.
        num_classes: Number of object classes (COCO: 91). Default 91.
        weights: Pre-trained weights identifier or file path. Default "coco".
        input_shape: Input shape as (H, W, C). Default (704, 704, 3).
        input_tensor: Optional input tensor.
        name: Model name.

    Returns:
        RF-DETR Large model.
    """
    return _create_rf_detr_model(
        "RFDETRLarge",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )
