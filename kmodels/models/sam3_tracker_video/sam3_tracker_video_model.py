"""Sam3TrackerVideo model: extends Sam3Tracker with temporal memory.

Adds memory attention, memory encoder, object pointer projection,
and temporal positional encoding for multi-frame video tracking.
"""

import keras
import numpy as np
from keras import layers, ops

from kmodels.model_registry import register_model
from kmodels.models.sam3_tracker.sam3_tracker_layers import (
    Sam3TrackerFeedForward,
    Sam3TrackerMaskDecoder,
    Sam3TrackerPositionalEmbedding,
    Sam3TrackerPromptEncoder,
)
from kmodels.models.sam3_video.sam3_video_layers import Sam3VisionNeck

from .config import SAM3_TRACKER_VIDEO_MODEL_CONFIG, SAM3_TRACKER_VIDEO_WEIGHTS_CONFIG
from .sam3_tracker_video_layers import (
    Sam3TrackerVideoMemoryAttention,
    Sam3TrackerVideoMemoryEncoder,
)


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerVideoModel(keras.Model):
    """Sam3TrackerVideo: multi-frame video tracker with temporal memory.

    Extends the single-frame Sam3Tracker with:
    - Memory attention (4 layers with RoPE cross-attention to past frames)
    - Memory encoder (encodes masks into memory for future conditioning)
    - Object pointer projection (projects SAM output tokens to pointers)
    - Temporal positional encoding (sine PE for frame offsets)

    The vision encoder is shared with the SAM3 detector — both load
    from the same weight file. When used inside Sam3Video, the vision
    encoder is bypassed (features come through the tracker_neck FPN).
    """

    def __init__(
        self,
        sam3_model=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        memory_attention_config=None,
        memory_encoder_config=None,
        hidden_dim=256,
        mem_dim=64,
        num_maskmem=7,
        max_object_pointers_in_encoder=16,
        enable_temporal_pos_encoding=True,
        enable_occlusion_spatial_embedding=True,
        **kwargs,
    ):
        super().__init__(name=kwargs.pop("name", "Sam3TrackerVideo"), **kwargs)

        cfg = SAM3_TRACKER_VIDEO_MODEL_CONFIG["Sam3TrackerVideo"]
        if prompt_encoder_config is None:
            prompt_encoder_config = cfg["prompt_encoder"]
        if mask_decoder_config is None:
            mask_decoder_config = cfg["mask_decoder"]
        if memory_attention_config is None:
            memory_attention_config = cfg["memory_attention"]
        if memory_encoder_config is None:
            memory_encoder_config = cfg["memory_encoder"]

        self._prompt_encoder_config = prompt_encoder_config
        self._mask_decoder_config = mask_decoder_config
        self._memory_attention_config = memory_attention_config
        self._memory_encoder_config = memory_encoder_config
        self._hidden_dim = hidden_dim
        self._mem_dim = mem_dim
        self._num_maskmem = num_maskmem
        self._max_object_pointers = max_object_pointers_in_encoder
        self._enable_temporal_pos = enable_temporal_pos_encoding
        self._enable_occlusion = enable_occlusion_spatial_embedding

        # Store sam3_model as non-tracked reference (avoid weight duplication).
        # Keras tracks Layer/Model attributes for save_weights; we bypass this
        # so Sam3VideoModel can own sam3_model and save weights without duplicates.
        object.__setattr__(self, "_sam3_ref", sam3_model)

        # ── Shared positional embedding ──
        self.shared_image_embedding = Sam3TrackerPositionalEmbedding(
            hidden_size=prompt_encoder_config["hidden_size"],
            name="shared_image_embedding",
        )
        self.shared_image_embedding.build(None)

        # ── Prompt encoder ──
        pe_keys = {"hidden_size", "image_size", "patch_size", "num_point_embeddings"}
        pe_kwargs = {k: v for k, v in prompt_encoder_config.items() if k in pe_keys}
        self.prompt_encoder = Sam3TrackerPromptEncoder(
            name="prompt_encoder", **pe_kwargs
        )
        self.prompt_encoder.build(None)

        # ── Mask decoder ──
        self.mask_decoder = Sam3TrackerMaskDecoder(
            name="mask_decoder", **mask_decoder_config
        )
        self.mask_decoder.build(None)

        # ── Vision neck (tracker's own FPN, separate from detector FPN) ──
        self.vision_neck = Sam3VisionNeck(
            backbone_hidden_size=1024,
            fpn_hidden_size=hidden_dim,
            scale_factors=[4.0, 2.0, 1.0, 0.5],
            name="vision_neck",
        )
        self.vision_neck.build((None, 1024, None, None))

        # ── Memory attention ──
        ma_cfg = memory_attention_config
        self.memory_attention = Sam3TrackerVideoMemoryAttention(
            hidden_size=ma_cfg["hidden_size"],
            num_layers=ma_cfg["num_layers"],
            num_attention_heads=ma_cfg["num_attention_heads"],
            downsample_rate=ma_cfg["downsample_rate"],
            feed_forward_hidden_size=ma_cfg["feed_forward_hidden_size"],
            feed_forward_act=ma_cfg["feed_forward_hidden_act"],
            mem_dim=mem_dim,
            rope_theta=ma_cfg["rope_theta"],
            rope_feat_sizes=ma_cfg["rope_feat_sizes"],
            name="memory_attention",
        )
        self.memory_attention.build(None)

        # ── Memory encoder ──
        me_cfg = memory_encoder_config
        self.memory_encoder = Sam3TrackerVideoMemoryEncoder(
            hidden_size=me_cfg["hidden_size"],
            output_channels=me_cfg["output_channels"],
            mask_downsampler_config={
                "embed_dim": me_cfg["mask_downsampler_embed_dim"],
                "kernel_size": me_cfg["mask_downsampler_kernel_size"],
                "stride": me_cfg["mask_downsampler_stride"],
                "padding": me_cfg["mask_downsampler_padding"],
                "total_stride": me_cfg["mask_downsampler_total_stride"],
            },
            memory_fuser_config={
                "num_layers": me_cfg["memory_fuser_num_layers"],
                "embed_dim": me_cfg["memory_fuser_embed_dim"],
                "intermediate_dim": me_cfg["memory_fuser_intermediate_dim"],
                "kernel_size": me_cfg["memory_fuser_kernel_size"],
                "padding": me_cfg["memory_fuser_padding"],
                "layer_scale_init_value": me_cfg["memory_fuser_layer_scale_init_value"],
            },
            name="memory_encoder",
        )
        self.memory_encoder.build(None)

        # ── Object pointer projection (3-layer MLP) ──
        self.object_pointer_proj = Sam3TrackerFeedForward(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=3,
            name="object_pointer_proj",
        )
        self.object_pointer_proj.build((None, hidden_dim))

        # ── Mask downsample (for mask prompts: 4x4 stride-4 conv) ──
        self.mask_downsample = layers.Conv2D(
            1,
            kernel_size=4,
            strides=4,
            data_format="channels_first",
            name="mask_downsample",
        )
        self.mask_downsample.build((None, 1, None, None))

        # ── Learnable embeddings ──
        self.no_memory_embedding = self.add_weight(
            name="no_memory_embedding",
            shape=(1, 1, hidden_dim),
            initializer="zeros",
        )
        self.no_memory_positional_encoding = self.add_weight(
            name="no_memory_positional_encoding",
            shape=(1, 1, hidden_dim),
            initializer="zeros",
        )
        self.no_object_pointer = self.add_weight(
            name="no_object_pointer",
            shape=(1, hidden_dim),
            initializer="zeros",
        )
        self.memory_temporal_positional_encoding = self.add_weight(
            name="memory_temporal_positional_encoding",
            shape=(num_maskmem, 1, 1, mem_dim),
            initializer="zeros",
        )

        if enable_temporal_pos_encoding:
            self.temporal_positional_encoding_projection_layer = layers.Dense(
                mem_dim, name="temporal_positional_encoding_projection_layer"
            )
            self.temporal_positional_encoding_projection_layer.build((None, hidden_dim))

        if enable_occlusion_spatial_embedding:
            self.occlusion_spatial_embedding_parameter = self.add_weight(
                name="occlusion_spatial_embedding_parameter",
                shape=(1, mem_dim),
                initializer="zeros",
            )

        pe_cfg = prompt_encoder_config
        self._image_embedding_size = pe_cfg["image_size"] // pe_cfg["patch_size"]
        self.built = True

    def get_image_wide_positional_embeddings(self):
        """Create positional embeddings for the full image grid.

        Returns:
            (1, 256, H, W) positional embeddings.
        """
        size = self._image_embedding_size
        grid = ops.ones((size, size), dtype="float32")
        y = ops.cumsum(grid, axis=0) - 0.5
        x = ops.cumsum(grid, axis=1) - 0.5
        y = y / size
        x = x / size
        coords = ops.stack([x, y], axis=-1)
        pe = self.shared_image_embedding(coords)
        pe = ops.transpose(pe, (2, 0, 1))
        return ops.expand_dims(pe, 0)

    def get_image_features(self, pixel_values):
        """Extract vision features using shared backbone + tracker's own FPN.

        Uses the SAM3 detector's backbone for the ViT features, then passes
        them through the tracker's own vision_neck FPN (separate weights from
        the detector FPN). This matches HF's Sam3TrackerVideoModel which has
        its own vision_encoder.neck.

        Returns:
            list of 3 feature maps: [feat_s0, feat_s1, fpn_2_with_no_mem]
        """
        if self._sam3_ref is None:
            raise ValueError(
                "sam3_model must be set before calling get_image_features."
            )

        from kmodels.models.sam3.sam3_processor import _SUBMODEL_CACHE

        # Extract backbone output (before detector's FPN)
        det = self._sam3_ref.detector
        bb_key = f"{id(det)}_backbone_nchw"
        if bb_key not in _SUBMODEL_CACHE:
            backbone_layer = det.get_layer("backbone_to_nchw")
            _SUBMODEL_CACHE[bb_key] = keras.Model(
                inputs=det.input,
                outputs=backbone_layer.output,
                name="backbone_nchw",
            )

        backbone_sub = _SUBMODEL_CACHE[bb_key]
        dummy_text = np.zeros((1, 1, 1024), dtype=np.float32)
        dummy_mask = np.ones((1, 1), dtype=np.float32)

        backbone_nchw = backbone_sub.predict(
            {
                "pixel_values": pixel_values,
                "text_features": dummy_text,
                "text_attention_mask": dummy_mask,
            },
            verbose=0,
        )
        backbone_nchw = ops.convert_to_tensor(backbone_nchw)

        # Run through tracker's OWN FPN neck (not detector's FPN)
        fpn_hidden_states, _ = self.vision_neck(backbone_nchw)

        # Use first 3 FPN levels: 4x, 2x, 1x (discard 0.5x)
        feat_s0 = self.mask_decoder.conv_s0(fpn_hidden_states[0])
        feat_s1 = self.mask_decoder.conv_s1(fpn_hidden_states[1])
        fpn_2 = fpn_hidden_states[2]

        no_mem = ops.reshape(self.no_memory_embedding, (1, -1, 1, 1))
        fpn_2 = fpn_2 + no_mem

        return [feat_s0, feat_s1, fpn_2]

    def get_image_features_from_neck(self, fpn_hidden_states):
        """Process pre-computed FPN features from tracker_neck.

        Used by Sam3Video where vision features come through the neck FPN
        instead of the detector's own FPN.

        Args:
            fpn_hidden_states: list of 4 FPN levels from Sam3VisionNeck.
                [0]: (B, 256, 4H, 4W), [1]: (B, 256, 2H, 2W),
                [2]: (B, 256, H, W), [3]: (B, 256, H/2, W/2)

        Returns:
            list of 3 feature maps: [feat_s0, feat_s1, fpn_2_with_no_mem]
        """
        feat_s0 = self.mask_decoder.conv_s0(fpn_hidden_states[0])
        feat_s1 = self.mask_decoder.conv_s1(fpn_hidden_states[1])
        fpn_2 = fpn_hidden_states[2]

        no_mem = ops.reshape(self.no_memory_embedding, (1, -1, 1, 1))
        fpn_2 = fpn_2 + no_mem

        return [feat_s0, feat_s1, fpn_2]

    def encode_memory(self, vision_features, masks):
        """Encode predicted masks into memory features.

        Args:
            vision_features: (B, C, H, W) FPN level 2 features.
            masks: (B, 1, H_mask, W_mask) high-res mask logits.

        Returns:
            maskmem_features: (B, mem_dim, H', W')
            maskmem_pos_enc: (B, mem_dim, H', W')
        """
        return self.memory_encoder(vision_features, masks)

    def project_object_pointer(self, sam_output_token, is_obj_appearing):
        """Project SAM output token to object pointer for memory.

        Args:
            sam_output_token: (B, hidden_dim) from mask decoder.
            is_obj_appearing: (B, 1) binary flag.

        Returns:
            (B, hidden_dim) object pointer embedding.
        """
        pointer = self.object_pointer_proj(sam_output_token)
        lam = ops.cast(is_obj_appearing, pointer.dtype)
        pointer = lam * pointer + (1 - lam) * self.no_object_pointer
        return pointer

    def call(
        self,
        pixel_values=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        input_masks=None,
        image_embeddings=None,
        multimask_output=True,
    ):
        """Single-frame inference (no memory conditioning).

        For video inference with memory, use Sam3Video.forward()
        which orchestrates detection + memory propagation.
        """
        batch_size = (
            ops.shape(pixel_values)[0]
            if pixel_values is not None
            else ops.shape(image_embeddings[-1])[0]
        )

        if image_embeddings is None:
            image_embeddings = self.get_image_features(pixel_values)

        image_pe = self.get_image_wide_positional_embeddings()
        image_pe = ops.broadcast_to(image_pe, (batch_size,) + ops.shape(image_pe)[1:])

        if input_points is None and input_boxes is None:
            input_points = ops.zeros((batch_size, 1, 1, 2), dtype="float32")
            input_labels = -ops.ones((batch_size, 1, 1), dtype="int32")

        if input_points is not None and input_labels is None:
            input_labels = ops.ones_like(input_points[..., 0], dtype="int32")

        sparse_emb, dense_emb = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        masks, iou_pred, sam_tokens, object_score = self.mask_decoder(
            image_embeddings=image_embeddings[-1],
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
            high_resolution_features=image_embeddings[:2],
        )

        return {
            "pred_masks": masks,
            "iou_scores": iou_pred,
            "object_score_logits": object_score,
            "sam_tokens": sam_tokens,
        }

    def predict_masks(
        self,
        image,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        input_masks=None,
        multimask_output=True,
        image_embeddings=None,
    ):
        """High-level predict for a single image."""
        from kmodels.models.sam3.sam3_processor import preprocess_image

        if image_embeddings is None:
            pixel_values, _ = preprocess_image(image)
        else:
            pixel_values = None

        if input_points is not None:
            input_points = np.asarray(input_points, dtype=np.float32)
            if input_points.ndim == 2:
                input_points = input_points[np.newaxis, np.newaxis]
            elif input_points.ndim == 3:
                input_points = input_points[np.newaxis]

        if input_labels is not None:
            input_labels = np.asarray(input_labels, dtype=np.int32)
            if input_labels.ndim == 1:
                input_labels = input_labels[np.newaxis, np.newaxis]
            elif input_labels.ndim == 2:
                input_labels = input_labels[np.newaxis]

        if input_boxes is not None:
            input_boxes = np.asarray(input_boxes, dtype=np.float32)
            if input_boxes.ndim == 2:
                input_boxes = input_boxes[np.newaxis]

        return self(
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
            image_embeddings=image_embeddings,
            multimask_output=multimask_output,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "prompt_encoder_config": self._prompt_encoder_config,
                "mask_decoder_config": self._mask_decoder_config,
                "memory_attention_config": self._memory_attention_config,
                "memory_encoder_config": self._memory_encoder_config,
                "hidden_dim": self._hidden_dim,
                "mem_dim": self._mem_dim,
                "num_maskmem": self._num_maskmem,
                "max_object_pointers_in_encoder": self._max_object_pointers,
                "enable_temporal_pos_encoding": self._enable_temporal_pos,
                "enable_occlusion_spatial_embedding": self._enable_occlusion,
            }
        )
        return config


def _create_sam3_tracker_video(variant, sam3_model=None, weights=None, **kwargs):
    config = SAM3_TRACKER_VIDEO_MODEL_CONFIG[variant]
    model = Sam3TrackerVideoModel(
        sam3_model=sam3_model,
        prompt_encoder_config=config["prompt_encoder"],
        mask_decoder_config=config["mask_decoder"],
        memory_attention_config=config["memory_attention"],
        memory_encoder_config=config["memory_encoder"],
        hidden_dim=config["hidden_dim"],
        mem_dim=config["mem_dim"],
        num_maskmem=config["num_maskmem"],
        max_object_pointers_in_encoder=config["max_object_pointers_in_encoder"],
        enable_temporal_pos_encoding=config[
            "enable_temporal_pos_encoding_for_object_pointers"
        ],
        enable_occlusion_spatial_embedding=config["enable_occlusion_spatial_embedding"],
        **kwargs,
    )

    valid_weights = list(SAM3_TRACKER_VIDEO_WEIGHTS_CONFIG.get(variant, {}).keys())
    if weights in valid_weights:
        from kmodels.models.sam3.weights_config import load_unified_weights

        load_unified_weights(
            sam3_model=sam3_model, tracker_video_model=model, weights=weights
        )
    elif weights is not None:
        model.load_weights(weights, skip_mismatch=True)
    else:
        print("No tracker video weights loaded.")

    return model


@register_model
def Sam3TrackerVideo(sam3_model=None, weights=None, **kwargs):
    """Create a Sam3TrackerVideo model.

    Args:
        sam3_model: a trained SAM3Model (from Sam3(weights="pcs")).
        weights: weight variant name or file path.

    Usage:
        from kmodels.models.sam3 import Sam3
        from kmodels.models.sam3_tracker_video import Sam3TrackerVideo

        sam3 = Sam3(weights="pcs")
        tracker_video = Sam3TrackerVideo(sam3_model=sam3, weights="pcs")
    """
    return _create_sam3_tracker_video(
        "Sam3TrackerVideo", sam3_model=sam3_model, weights=weights, **kwargs
    )
