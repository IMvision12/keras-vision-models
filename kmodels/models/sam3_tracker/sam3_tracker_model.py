"""Sam3Tracker model: single-frame prompt-based segmentation.

Reuses the SAM3 vision encoder (backbone + FPN) and adds a prompt
encoder + two-way transformer mask decoder for point/box/mask prompts.
"""

import keras
import numpy as np
from keras import ops

from kmodels.model_registry import register_model

from .config import SAM3_TRACKER_MODEL_CONFIG, SAM3_TRACKER_WEIGHTS_CONFIG
from .sam3_tracker_layers import (
    Sam3TrackerMaskDecoder,
    Sam3TrackerPositionalEmbedding,
    Sam3TrackerPromptEncoder,
)


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerModel(keras.Model):
    """Sam3Tracker: single-frame segmentation with point/box/mask prompts.

    Uses the SAM3 vision encoder for image features and a two-way
    transformer mask decoder for prompt-conditioned segmentation.

    The vision encoder is shared with the SAM3 detector — both load
    from the same weight file.
    """

    def __init__(
        self,
        sam3_model=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        **kwargs,
    ):
        """
        Args:
            sam3_model: a trained SAM3Model (from Sam3()). The vision encoder
                is reused. If None, must be set later before inference.
            prompt_encoder_config: dict with prompt encoder params.
            mask_decoder_config: dict with mask decoder params.
        """
        super().__init__(name=kwargs.pop("name", "Sam3Tracker"), **kwargs)

        if prompt_encoder_config is None:
            prompt_encoder_config = SAM3_TRACKER_MODEL_CONFIG["Sam3Tracker"][
                "prompt_encoder"
            ]
        if mask_decoder_config is None:
            mask_decoder_config = SAM3_TRACKER_MODEL_CONFIG["Sam3Tracker"][
                "mask_decoder"
            ]

        self._prompt_encoder_config = prompt_encoder_config
        self._mask_decoder_config = mask_decoder_config

        self.sam3_model = sam3_model

        # Shared positional embedding
        self.shared_image_embedding = Sam3TrackerPositionalEmbedding(
            hidden_size=prompt_encoder_config["hidden_size"],
            name="shared_image_embedding",
        )
        self.shared_image_embedding.build(None)

        # Prompt encoder (filter config to accepted params)
        pe_keys = {"hidden_size", "image_size", "patch_size", "num_point_embeddings"}
        pe_kwargs = {k: v for k, v in prompt_encoder_config.items() if k in pe_keys}
        self.prompt_encoder = Sam3TrackerPromptEncoder(
            name="prompt_encoder", **pe_kwargs
        )
        self.prompt_encoder.build(None)

        # Mask decoder
        self.mask_decoder = Sam3TrackerMaskDecoder(
            name="mask_decoder", **mask_decoder_config
        )
        self.mask_decoder.build(None)

        # No-memory embedding (for compatibility with video tracker)
        self.no_memory_embedding = self.add_weight(
            name="no_memory_embedding",
            shape=(1, 1, prompt_encoder_config["hidden_size"]),
            initializer="zeros",
        )

        pe_cfg = prompt_encoder_config
        self._image_embedding_size = pe_cfg["image_size"] // pe_cfg["patch_size"]

    def get_image_wide_positional_embeddings(self):
        """Create positional embeddings for the full image grid.

        Returns:
            (1, 256, H, W) positional embeddings.
        """
        size = self._image_embedding_size  # 72
        grid = ops.ones((size, size), dtype="float32")
        y = ops.cumsum(grid, axis=0) - 0.5
        x = ops.cumsum(grid, axis=1) - 0.5
        y = y / size
        x = x / size

        coords = ops.stack([x, y], axis=-1)  # (72, 72, 2)
        pe = self.shared_image_embedding(coords)  # (72, 72, 256)
        pe = ops.transpose(pe, (2, 0, 1))  # (256, 72, 72)
        return ops.expand_dims(pe, 0)  # (1, 256, 72, 72)

    def get_image_features(self, pixel_values):
        """Extract vision features from pixel values.

        Uses the SAM3 detector's vision backbone + FPN.

        Args:
            pixel_values: (B, H, W, 3) NHWC image.

        Returns:
            list of 3 feature maps at different scales, pre-projected
            for the mask decoder:
            - [0]: (B, 32, 288, 288) — 4x, projected by conv_s0
            - [1]: (B, 64, 144, 144) — 2x, projected by conv_s1
            - [2]: (B, 256, 72, 72) — 1x + no_memory_embedding
        """
        if self.sam3_model is None:
            raise ValueError(
                "sam3_model must be set before calling get_image_features."
            )

        from kmodels.models.sam3.sam3_processor import _SUBMODEL_CACHE

        det = self.sam3_model.detector
        vis_key = f"{id(det)}_tracker_vision"
        if vis_key not in _SUBMODEL_CACHE:
            outputs = {}
            for i in range(3):
                layer = det.get_layer(f"fpn_level_{i}_proj2")
                outputs[f"fpn_{i}"] = layer.output
            _SUBMODEL_CACHE[vis_key] = keras.Model(
                inputs=det.input, outputs=outputs, name="tracker_vision"
            )

        vision_sub = _SUBMODEL_CACHE[vis_key]
        dummy_text = np.zeros((1, 1, 1024), dtype=np.float32)
        dummy_mask = np.ones((1, 1), dtype=np.float32)

        vis_out = vision_sub.predict(
            {
                "pixel_values": pixel_values,
                "text_features": dummy_text,
                "text_attention_mask": dummy_mask,
            },
            verbose=0,
        )

        # Pre-project high-res features for mask decoder
        fpn_0 = ops.convert_to_tensor(vis_out["fpn_0"])  # (B, 256, 288, 288)
        fpn_1 = ops.convert_to_tensor(vis_out["fpn_1"])  # (B, 256, 144, 144)
        fpn_2 = ops.convert_to_tensor(vis_out["fpn_2"])  # (B, 256, 72, 72)

        feat_s0 = self.mask_decoder.conv_s0(fpn_0)  # (B, 32, 288, 288)
        feat_s1 = self.mask_decoder.conv_s1(fpn_1)  # (B, 64, 144, 144)

        # Add no-memory embedding: (1,1,256) → (1,256,1,1) for NCHW
        no_mem = ops.reshape(self.no_memory_embedding, (1, -1, 1, 1))
        fpn_2 = fpn_2 + no_mem

        return [feat_s0, feat_s1, fpn_2]

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
        """
        Args:
            pixel_values: (B, H, W, 3) or None if image_embeddings provided.
            input_points: (B, pb, N, 2) point prompts in pixel coords.
            input_labels: (B, pb, N) point labels (1=fg, 0=bg, -1=pad).
            input_boxes: (B, nb, 4) box prompts [x1,y1,x2,y2] pixel coords.
            input_masks: (B, 1, 288, 288) mask prompts or None.
            image_embeddings: pre-computed from get_image_features().
            multimask_output: if True return 3 masks, else 1.

        Returns:
            dict with:
            - pred_masks: (B, pb, num_masks, 288, 288) mask logits
            - iou_scores: (B, pb, num_masks)
            - object_score_logits: (B, pb, 1)
        """
        batch_size = (
            ops.shape(pixel_values)[0]
            if pixel_values is not None
            else ops.shape(image_embeddings[-1])[0]
        )

        # Get image features
        if image_embeddings is None:
            image_embeddings = self.get_image_features(pixel_values)

        # Image positional embeddings
        image_pe = self.get_image_wide_positional_embeddings()
        image_pe = ops.broadcast_to(
            image_pe,
            (batch_size,) + ops.shape(image_pe)[1:],
        )

        # Default prompts if none provided
        if input_points is None and input_boxes is None:
            input_points = ops.zeros((batch_size, 1, 1, 2), dtype="float32")
            input_labels = -ops.ones((batch_size, 1, 1), dtype="int32")

        if input_points is not None and input_labels is None:
            input_labels = ops.ones_like(input_points[..., 0], dtype="int32")

        # Encode prompts
        sparse_emb, dense_emb = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        # Decode masks (high-res features are the first two elements)
        masks, iou_pred, _, object_score = self.mask_decoder(
            image_embeddings=image_embeddings[-1],  # (B, 256, 72, 72)
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
        """High-level predict for a single image.

        Args:
            image: PIL Image, numpy array, or file path.
            input_points: list of [x, y] or (1, pb, N, 2) array.
            input_labels: list of labels or (1, pb, N) array.
            input_boxes: list of [x1,y1,x2,y2] or (1, nb, 4) array.
            input_masks: (1, 1, 288, 288) mask or None.
            multimask_output: bool.
            image_embeddings: pre-computed features.

        Returns:
            dict with pred_masks, iou_scores, object_score_logits.
        """
        from kmodels.models.sam3.sam3_processor import preprocess_image

        if image_embeddings is None:
            pixel_values, _ = preprocess_image(image)
        else:
            pixel_values = None

        # Normalize inputs to expected shapes
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
            }
        )
        return config


def _create_sam3_tracker(variant, sam3_model=None, weights=None, **kwargs):
    config = SAM3_TRACKER_MODEL_CONFIG[variant]
    model = Sam3TrackerModel(
        sam3_model=sam3_model,
        prompt_encoder_config=config["prompt_encoder"],
        mask_decoder_config=config["mask_decoder"],
        **kwargs,
    )

    valid_weights = list(SAM3_TRACKER_WEIGHTS_CONFIG.get(variant, {}).keys())
    if weights in valid_weights:
        url = SAM3_TRACKER_WEIGHTS_CONFIG[variant][weights].get("url", "")
        if url:
            from kmodels.utils import load_weights_from_config

            load_weights_from_config(
                variant, weights, model, SAM3_TRACKER_WEIGHTS_CONFIG
            )
        else:
            print(f"Weight URL for '{weights}' not available.")
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No tracker weights loaded.")

    return model


@register_model
def Sam3Tracker(sam3_model=None, weights=None, **kwargs):
    """Create a Sam3Tracker model.

    Args:
        sam3_model: a trained SAM3Model (from Sam3(weights="pcs")).
            Required for vision encoder features.
        weights: weight variant name or file path.

    Usage:
        from kmodels.models.sam3 import Sam3
        from kmodels.models.sam3_tracker import Sam3Tracker

        sam3 = Sam3(weights="pcs")
        tracker = Sam3Tracker(sam3_model=sam3, weights="pcs")

        results = tracker.predict_masks(image, input_points=[[100, 200]], input_labels=[1])
    """
    return _create_sam3_tracker(
        "Sam3Tracker", sam3_model=sam3_model, weights=weights, **kwargs
    )
