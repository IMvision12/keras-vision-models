"""Sam3Tracker model: single-frame prompt-based segmentation.

Reuses the SAM3 vision encoder (backbone + FPN) and adds a prompt
encoder + two-way transformer mask decoder for point/box/mask prompts.

Uses the functional-subclassed pattern (like SAM2): the entire forward
graph is built in __init__ with keras.Input, and super().__init__(inputs, outputs)
makes it a functional model. No custom call() needed.
"""

import keras
import numpy as np
from keras import layers, ops

from kmodels.model_registry import register_model

from .config import SAM3_TRACKER_MODEL_CONFIG, SAM3_TRACKER_WEIGHTS_CONFIG
from .sam3_tracker_layers import (
    Sam3TrackerMaskDecoder,
    Sam3TrackerPositionalEmbedding,
    Sam3TrackerPromptEncoder,
)


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerImagePE(layers.Layer):
    """Computes image-wide positional embeddings for the tracker.

    Takes image embeddings as input (for batch size), computes a fixed
    coordinate grid, passes through shared_image_embedding, and returns
    the positional encoding broadcast to batch size.
    """

    def __init__(self, image_embedding_size, shared_image_embedding, **kwargs):
        super().__init__(**kwargs)
        self.image_embedding_size = image_embedding_size
        self._shared_embedding = shared_image_embedding

    def call(self, image_embeddings):
        batch_size = ops.shape(image_embeddings)[0]
        size = self.image_embedding_size
        grid = ops.ones((size, size), dtype="float32")
        y = ops.cumsum(grid, axis=0) - 0.5
        x = ops.cumsum(grid, axis=1) - 0.5
        y = y / size
        x = x / size
        coords = ops.stack([x, y], axis=-1)
        pe = self._shared_embedding(coords)
        pe = ops.transpose(pe, (2, 0, 1))
        pe = ops.expand_dims(pe, 0)
        return ops.broadcast_to(pe, (batch_size,) + ops.shape(pe)[1:])

    def get_config(self):
        config = super().get_config()
        config.update({"image_embedding_size": self.image_embedding_size})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerModel(keras.Model):
    """Sam3Tracker: single-frame segmentation with point/box/mask prompts.

    Functional-subclassed model (like SAM2): the forward graph is built
    in __init__ using keras.Input and layer calls. Inputs are pre-computed
    image embeddings + point prompts. Vision feature extraction stays as
    a separate method (get_image_features).
    """

    IMAGE_EMBEDDING_SIZE = 72

    def __init__(
        self,
        sam3_model=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        **kwargs,
    ):
        if prompt_encoder_config is None:
            prompt_encoder_config = SAM3_TRACKER_MODEL_CONFIG["Sam3Tracker"][
                "prompt_encoder"
            ]
        if mask_decoder_config is None:
            mask_decoder_config = SAM3_TRACKER_MODEL_CONFIG["Sam3Tracker"][
                "mask_decoder"
            ]

        pe_cfg = prompt_encoder_config
        image_embedding_size = pe_cfg["image_size"] // pe_cfg["patch_size"]

        s = image_embedding_size
        image_embeddings = layers.Input(shape=(256, s, s), name="image_embeddings")
        image_embeddings_s0 = layers.Input(
            shape=(32, 4 * s, 4 * s), name="image_embeddings_s0"
        )
        image_embeddings_s1 = layers.Input(
            shape=(64, 2 * s, 2 * s), name="image_embeddings_s1"
        )
        input_points = layers.Input(
            shape=(None, None, 2), name="input_points", dtype="float32"
        )
        input_labels = layers.Input(
            shape=(None, None), name="input_labels", dtype="int32"
        )

        shared_image_embedding = Sam3TrackerPositionalEmbedding(
            hidden_size=pe_cfg["hidden_size"],
            name="shared_image_embedding",
        )

        image_pe = Sam3TrackerImagePE(
            image_embedding_size=image_embedding_size,
            shared_image_embedding=shared_image_embedding,
            name="image_pe",
        )(image_embeddings)

        pe_keys = {"hidden_size", "image_size", "patch_size", "num_point_embeddings"}
        pe_kwargs = {k: v for k, v in pe_cfg.items() if k in pe_keys}
        prompt_result = Sam3TrackerPromptEncoder(name="prompt_encoder", **pe_kwargs)(
            [input_points, input_labels]
        )

        sparse_embeddings = prompt_result["sparse_embeddings"]
        dense_embeddings = prompt_result["dense_embeddings"]

        decoder_output = Sam3TrackerMaskDecoder(
            name="mask_decoder", **mask_decoder_config
        )(
            [
                image_embeddings,
                image_pe,
                sparse_embeddings,
                dense_embeddings,
                image_embeddings_s0,
                image_embeddings_s1,
            ]
        )

        pred_masks = decoder_output["pred_masks"][:, :, 1:, :, :]
        iou_scores = decoder_output["iou_scores"][:, :, 1:]
        object_score_logits = decoder_output["object_score_logits"]

        super().__init__(
            inputs={
                "image_embeddings": image_embeddings,
                "image_embeddings_s0": image_embeddings_s0,
                "image_embeddings_s1": image_embeddings_s1,
                "input_points": input_points,
                "input_labels": input_labels,
            },
            outputs={
                "pred_masks": pred_masks,
                "iou_scores": iou_scores,
                "object_score_logits": object_score_logits,
            },
            name=kwargs.pop("name", "Sam3Tracker"),
            **kwargs,
        )

        self._prompt_encoder_config = prompt_encoder_config
        self._mask_decoder_config = mask_decoder_config
        self._image_embedding_size = image_embedding_size

        object.__setattr__(self, "_sam3_ref", sam3_model)

        self.no_memory_embedding = self.add_weight(
            name="no_memory_embedding",
            shape=(1, 1, pe_cfg["hidden_size"]),
            initializer="zeros",
        )

    def get_image_wide_positional_embeddings(self):
        """Create positional embeddings for the full image grid."""
        shared = self.get_layer("image_pe")._shared_embedding
        size = self._image_embedding_size
        grid = ops.ones((size, size), dtype="float32")
        y = ops.cumsum(grid, axis=0) - 0.5
        x = ops.cumsum(grid, axis=1) - 0.5
        y = y / size
        x = x / size
        coords = ops.stack([x, y], axis=-1)
        pe = shared(coords)
        pe = ops.transpose(pe, (2, 0, 1))
        return ops.expand_dims(pe, 0)

    def get_image_features(self, pixel_values):
        """Extract vision features from pixel values.

        Uses the SAM3 detector's vision backbone + FPN.
        Returns list of 3 feature maps: [feat_s0, feat_s1, fpn_2].
        """
        if self._sam3_ref is None:
            raise ValueError(
                "sam3_model must be set before calling get_image_features."
            )

        from kmodels.models.sam3.sam3_processor import _SUBMODEL_CACHE

        det = self._sam3_ref.detector
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

        fpn_0 = ops.convert_to_tensor(vis_out["fpn_0"])
        fpn_1 = ops.convert_to_tensor(vis_out["fpn_1"])
        fpn_2 = ops.convert_to_tensor(vis_out["fpn_2"])

        mask_decoder = self.get_layer("mask_decoder")
        feat_s0 = mask_decoder.conv_s0(fpn_0)
        feat_s1 = mask_decoder.conv_s1(fpn_1)

        no_mem = ops.reshape(self.no_memory_embedding, (1, -1, 1, 1))
        fpn_2 = fpn_2 + no_mem

        return [feat_s0, feat_s1, fpn_2]

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
            image_embeddings = self.get_image_features(pixel_values)

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

        if input_points is None:
            batch_size = ops.shape(image_embeddings[-1])[0]
            input_points = np.zeros((batch_size, 1, 1, 2), dtype=np.float32)
            input_labels = -np.ones((batch_size, 1, 1), dtype=np.int32)

        if input_labels is None:
            input_labels = np.ones_like(input_points[..., 0], dtype=np.int32)

        return self(
            {
                "image_embeddings": image_embeddings[-1],
                "image_embeddings_s0": image_embeddings[0],
                "image_embeddings_s1": image_embeddings[1],
                "input_points": input_points,
                "input_labels": input_labels,
            }
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
        from kmodels.models.sam3.weights_config import load_unified_weights

        load_unified_weights(
            sam3_model=sam3_model, tracker_video_model=model, weights=weights
        )
    elif weights is not None:
        model.load_weights(weights, skip_mismatch=True)
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
