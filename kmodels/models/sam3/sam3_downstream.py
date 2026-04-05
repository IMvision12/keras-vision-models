"""SAM3 downstream task models.

Provides clean Model classes for each downstream task:
- SAM3ObjectDetection: bounding box detection with scores
- SAM3InstanceSegmentation: per-object masks + boxes + scores
- SAM3SemanticSegmentation: scene-level binary mask for a category

Each wraps the base SAM3 model, text encoder, and optional geometry
encoder into a single object with a unified predict() interface.

Usage:
    # Object detection
    detector = SAM3ObjectDetection.from_pretrained(
        model_weights="sam3.weights.h5",
        text_encoder_weights="sam3_text_encoder.weights.h5",
    )
    results = detector.predict(image, text="cat")
    # [{"scores": array, "boxes": array}]

    # Instance segmentation with box prompts
    segmenter = SAM3InstanceSegmentation.from_pretrained(
        model_weights="sam3.weights.h5",
        text_encoder_weights="sam3_text_encoder.weights.h5",
        geometry_encoder_weights="sam3_geometry_encoder.weights.h5",
    )
    results = segmenter.predict(image, text="cat", input_boxes=[[x1,y1,x2,y2]])
    # [{"scores": array, "boxes": array, "masks": array}]
"""

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from .sam3_processor import (
    _SUBMODEL_CACHE,
    _compute_scores,
    _get_vision_submodel,
    _run_decoder_with_projected,
    _scale_boxes,
    _sigmoid,
    preprocess_boxes,
    preprocess_image,
    preprocess_text_with_encoder,
)
from .sam3_utils import compute_sine_pos_encoding

# ═══════════════════════════════════════════════════════════════════
#  Downstream utilities
# ═══════════════════════════════════════════════════════════════════


def _resize_mask(mask, target_h, target_w):
    """Resize a single 2D mask to target size using PIL bilinear."""
    if Image is None:
        raise ImportError("PIL required for mask resizing")
    pil_m = Image.fromarray((mask * 255).astype(np.uint8))
    pil_m = pil_m.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil_m, dtype=np.float32) / 255.0


def _resize_masks_batch(masks, target_h, target_w):
    """Resize a batch of masks (N, H, W) to (N, target_h, target_w)."""
    if len(masks) == 0:
        return masks
    return np.stack([_resize_mask(m, target_h, target_w) for m in masks])


# ═══════════════════════════════════════════════════════════════════
#  Base pipeline
# ═══════════════════════════════════════════════════════════════════


class _SAM3Base:
    """Base class for SAM3 downstream tasks.

    Handles model loading, preprocessing, and the three prompt modes:
    - Mode A: text-only
    - Mode B: box-only (text defaults to "visual")
    - Mode C: text + boxes (hybrid)
    """

    def __init__(
        self,
        sam3_model,
        text_encoder_model,
        tokenizer=None,
        geometry_encoder=None,
    ):
        self.model = sam3_model
        self.text_encoder = text_encoder_model
        self.tokenizer = tokenizer
        self.geometry_encoder = geometry_encoder

    @classmethod
    def from_pretrained(
        cls,
        model_weights,
        text_encoder_weights,
        geometry_encoder_weights=None,
        input_shape=(1008, 1008, 3),
    ):
        """Load all components from weight files.

        Args:
            model_weights: path to sam3.weights.h5
            text_encoder_weights: path to sam3_text_encoder.weights.h5
            geometry_encoder_weights: path to sam3_geometry_encoder.weights.h5
                (required for box prompt modes B/C)
            input_shape: model input shape (H, W, 3)
        """
        import keras

        from .sam3_clip import SAM3CLIPTokenizer, build_text_encoder
        from .sam3_model import Sam3

        sam3_model = Sam3(input_shape=input_shape, weights=None)
        sam3_model.load_weights(model_weights)

        text_encoder_model = build_text_encoder(weights_path=text_encoder_weights)
        tokenizer = SAM3CLIPTokenizer()

        geometry_encoder = None
        if geometry_encoder_weights is not None:
            from .sam3_layers import SAM3GeometryEncoder

            geometry_encoder = SAM3GeometryEncoder(name="geometry_encoder")
            geometry_encoder.build((None, None, 4))
            # Wrap in model for weight loading
            boxes_in = keras.Input(shape=(None, 4), dtype="float32")
            labels_in = keras.Input(shape=(None,), dtype="int32")
            mask_in = keras.Input(shape=(None,), dtype="float32")
            vis_in = keras.Input(shape=(None, 256), dtype="float32")
            pos_in = keras.Input(shape=(None, 256), dtype="float32")
            geo_out, geo_mask = geometry_encoder(
                boxes_in, labels_in, mask_in, vis_in, pos_in
            )
            geo_model = keras.Model(
                inputs=[boxes_in, labels_in, mask_in, vis_in, pos_in],
                outputs={"features": geo_out, "mask": geo_mask},
            )
            geo_model.load_weights(geometry_encoder_weights)

        return cls(
            sam3_model=sam3_model,
            text_encoder_model=text_encoder_model,
            tokenizer=tokenizer,
            geometry_encoder=geometry_encoder,
        )

    def get_vision_features(self, image):
        """Pre-compute vision features for an image.

        Runs the vision backbone + FPN once. Pass the result as
        vision_embeds to predict() to avoid recomputing for each prompt.

        Args:
            image: PIL Image, numpy array (H,W,3), or file path.

        Returns:
            dict with pre-computed features to pass as vision_embeds.
        """
        from .sam3_model import build_sam3_decoder_model, build_sam3_vision_model

        pixel_values, original_size = preprocess_image(image)

        model_id = id(self.model)
        vis_key = f"{model_id}_vision"
        dec_key = f"{model_id}_decoder"

        if vis_key not in _SUBMODEL_CACHE:
            _SUBMODEL_CACHE[vis_key] = build_sam3_vision_model(self.model)
        if dec_key not in _SUBMODEL_CACHE:
            _SUBMODEL_CACHE[dec_key] = build_sam3_decoder_model(self.model)

        vision_model = _SUBMODEL_CACHE[vis_key]
        dummy_text = np.zeros((1, 1, 1024), dtype=np.float32)
        dummy_mask = np.ones((1, 1), dtype=np.float32)
        vis_out = vision_model.predict(
            {
                "pixel_values": pixel_values,
                "text_features": dummy_text,
                "text_attention_mask": dummy_mask,
            },
            verbose=0,
        )
        return {
            "fpn_0": vis_out["fpn_0"],
            "fpn_1": vis_out["fpn_1"],
            "fpn_2": vis_out["fpn_2"],
            "fpn_3": vis_out["fpn_3"],
            "text_projected": vis_out["text_projected"],
            "original_size": original_size,
        }

    def get_text_features(self, text):
        """Pre-compute text features for a prompt.

        Use this to avoid redundant text encoding when running the same
        prompt on multiple images.

        Args:
            text: str or list of str.

        Returns:
            dict with pre-computed text features to pass as text_embeds.
        """
        text_features, text_attention_mask = preprocess_text_with_encoder(
            text, self.text_encoder, self.tokenizer
        )
        return {
            "text_features": text_features,
            "text_attention_mask": text_attention_mask,
        }

    def _run_model(
        self,
        images,
        texts=None,
        input_boxes=None,
        input_boxes_labels=None,
        vision_embeds=None,
        text_embeds=None,
    ):
        """Run the base SAM3 model and return raw outputs.

        Supports single or batched inputs. Handles all prompt modes.

        Args:
            images: single image or list of images (PIL, numpy, or path).
                Ignored if vision_embeds is provided.
            texts: str, list of str, or None.
            input_boxes: list of [x1,y1,x2,y2] for single image, or
                list of list of boxes per image for batch.
            input_boxes_labels: matching labels (see predict methods).
            vision_embeds: pre-computed vision features from get_vision_features().
                Skips backbone recomputation when provided.

        Returns:
            raw: dict of model outputs
            original_sizes: list of (H, W) tuples
        """
        from keras import ops

        # Use pre-computed vision features if provided
        if vision_embeds is not None:
            pixel_values = None  # not needed — decoder model uses FPN directly
            original_sizes = [vision_embeds["original_size"]]
            batch_size = 1
        else:
            # Normalize to lists
            single = not isinstance(images, (list, tuple))
            if single:
                images = [images]

            if texts is not None and isinstance(texts, str):
                texts = [texts] * len(images)

            # Preprocess all images
            all_pixels = []
            original_sizes = []
            for img in images:
                pv, orig = preprocess_image(img)
                all_pixels.append(pv[0])  # (H, W, 3)
                original_sizes.append(orig)
            pixel_values = np.stack(all_pixels)  # (B, H, W, 3)
            batch_size = len(images)

        if texts is not None and isinstance(texts, str):
            texts = [texts] * batch_size

        # Normalize input_boxes/labels to per-image lists
        if input_boxes is not None and not isinstance(input_boxes, list):
            input_boxes = [input_boxes]
        if input_boxes_labels is not None and not isinstance(input_boxes_labels, list):
            input_boxes_labels = [input_boxes_labels]

        # Resolve text prompts per-image: None → "visual" if that image has boxes
        if texts is None and text_embeds is None:
            if input_boxes is not None:
                texts = []
                for i in range(batch_size):
                    bi = input_boxes[i] if i < len(input_boxes) else None
                    texts.append("visual" if bi is not None else None)
            else:
                raise ValueError("Provide text, text_embeds, or input_boxes.")
        elif texts is not None:
            # Fill None entries with "visual" if that image has boxes
            texts = list(texts)
            for i in range(batch_size):
                if texts[i] is None:
                    bi = (
                        input_boxes[i] if input_boxes and i < len(input_boxes) else None
                    )
                    texts[i] = "visual" if bi is not None else None
            if any(t is None for t in texts):
                raise ValueError(
                    "Each image must have either a text prompt or input_boxes."
                )

        if text_embeds is not None:
            text_features = text_embeds["text_features"]
            text_attention_mask = text_embeds["text_attention_mask"]
            # Repeat for batch if single text applied to multiple images
            if text_features.shape[0] == 1 and batch_size > 1:
                text_features = np.tile(text_features, (batch_size, 1, 1))
                text_attention_mask = np.tile(text_attention_mask, (batch_size, 1))
        else:
            text_features, text_attention_mask = preprocess_text_with_encoder(
                texts if batch_size > 1 else texts[0],
                self.text_encoder,
                self.tokenizer,
            )

        # Check if ANY image has box prompts
        has_any_boxes = input_boxes is not None and any(
            b is not None and len(b) > 0 for b in input_boxes if b is not None
        )

        # Mode A: text-only (no boxes at all)
        if not has_any_boxes:
            if vision_embeds is None:
                # Full model from scratch
                raw = self.model.predict(
                    {
                        "pixel_values": pixel_values,
                        "text_features": text_features,
                        "text_attention_mask": text_attention_mask,
                    },
                    verbose=0,
                )
            else:
                # Use cached vision — project text and run decoder model
                from keras import ops

                tp_layer = self.model.get_layer("text_projection")
                text_proj = ops.convert_to_numpy(
                    tp_layer(ops.convert_to_tensor(text_features))
                )
                dec_key = f"{id(self.model)}_decoder"
                if dec_key not in _SUBMODEL_CACHE:
                    from .sam3_model import build_sam3_decoder_model

                    _SUBMODEL_CACHE[dec_key] = build_sam3_decoder_model(self.model)
                decoder_model = _SUBMODEL_CACHE[dec_key]
                raw = decoder_model.predict(
                    {
                        "fpn_0": vision_embeds["fpn_0"],
                        "fpn_1": vision_embeds["fpn_1"],
                        "fpn_2": vision_embeds["fpn_2"],
                        "fpn_3": vision_embeds["fpn_3"],
                        "text_projected": text_proj,
                        "text_attention_mask": text_attention_mask,
                    },
                    verbose=0,
                )
                # Remove passthrough output
                raw.pop("fpn_05x", None)
            return raw, original_sizes

        # Mixed/box mode: process per-image through geometry encoder
        if self.geometry_encoder is None:
            raise ValueError(
                "geometry_encoder required for box prompts. "
                "Pass geometry_encoder_weights to from_pretrained()."
            )

        all_combined = []
        all_combined_mask = []
        vision_sub = _get_vision_submodel(self.model)

        for i in range(batch_size):
            pv_i = pixel_values[i : i + 1]
            tf_i = text_features[i : i + 1]
            tm_i = text_attention_mask[i : i + 1]

            # Get boxes for this image
            boxes_i = input_boxes[i] if input_boxes and i < len(input_boxes) else None
            labels_i = (
                input_boxes_labels[i]
                if input_boxes_labels and i < len(input_boxes_labels)
                else None
            )

            fpn_out = vision_sub.predict(
                {
                    "pixel_values": pv_i,
                    "text_features": tf_i,
                    "text_attention_mask": tm_i,
                },
                verbose=0,
            )
            text_projected = fpn_out["text_projected"]

            if boxes_i is None or (isinstance(boxes_i, list) and len(boxes_i) == 0):
                # No boxes — text-only projected features
                all_combined.append(text_projected)
                all_combined_mask.append(tm_i)
                continue

            if not isinstance(boxes_i[0], (list, tuple)):
                boxes_i = [boxes_i]
            if labels_i is None:
                labels_i = [1] * len(boxes_i)
            if not isinstance(labels_i, (list, tuple)):
                labels_i = [labels_i]

            boxes_cxcywh, box_labels, box_mask = preprocess_boxes(
                [boxes_i], [labels_i], [original_sizes[i]]
            )

            fpn_1x_nchw = fpn_out["fpn_1x"]
            enc_h = fpn_1x_nchw.shape[2]
            vision_flat = np.transpose(fpn_1x_nchw, (0, 2, 3, 1)).reshape(
                1, enc_h * enc_h, -1
            )
            pos_np = compute_sine_pos_encoding(enc_h, enc_h, 128, normalize=True)
            pos_flat = pos_np.transpose(0, 2, 3, 1).reshape(1, enc_h * enc_h, -1)

            geo_features, geo_mask = self.geometry_encoder(
                boxes_cxcywh,
                box_labels,
                box_mask,
                vision_flat,
                pos_flat,
                vision_features_nchw=fpn_1x_nchw,
            )
            geo_features = ops.convert_to_numpy(ops.stop_gradient(geo_features))
            geo_mask = ops.convert_to_numpy(ops.stop_gradient(geo_mask))

            combined = np.concatenate([text_projected, geo_features], axis=1)
            cmask = np.concatenate([tm_i, geo_mask], axis=1)
            all_combined.append(combined)
            all_combined_mask.append(cmask)

        # Pad combined features to same sequence length across batch
        max_seq = max(c.shape[1] for c in all_combined)
        feat_dim = all_combined[0].shape[2]
        padded_feats = np.zeros((batch_size, max_seq, feat_dim), dtype=np.float32)
        padded_masks = np.zeros((batch_size, max_seq), dtype=np.float32)
        for i, (feat, mask) in enumerate(zip(all_combined, all_combined_mask)):
            seq_len = feat.shape[1]
            padded_feats[i, :seq_len, :] = feat[0]
            padded_masks[i, :seq_len] = mask[0]

        raw = _run_decoder_with_projected(
            self.model, pixel_values, padded_feats, padded_masks
        )
        return raw, original_sizes


# ═══════════════════════════════════════════════════════════════════
#  Object Detection
# ═══════════════════════════════════════════════════════════════════


class SAM3ObjectDetection(_SAM3Base):
    """SAM3 for object detection.

    Returns bounding boxes and confidence scores for detected objects.

    Example:
        detector = SAM3ObjectDetection.from_pretrained(
            model_weights="sam3.weights.h5",
            text_encoder_weights="sam3_text_encoder.weights.h5",
        )

        # Text-only detection
        results = detector.predict(image, text="cat")

        # Box-guided detection
        results = detector.predict(image, input_boxes=[[x1,y1,x2,y2]])

        # Hybrid detection
        results = detector.predict(image, text="cat", input_boxes=[[x1,y1,x2,y2]])

        for det in results:
            print(det["scores"], det["boxes"])
    """

    def predict(
        self,
        images=None,
        text=None,
        input_boxes=None,
        input_boxes_labels=None,
        threshold=0.3,
        vision_embeds=None,
        text_embeds=None,
    ):
        """Run object detection.

        Args:
            images: single image or list of images (PIL, numpy, or path).
                Ignored if vision_embeds is provided.
            text: str or list of str. Auto-set to "visual" if only boxes given.
            input_boxes: boxes for single image, or list of boxes per image.
            input_boxes_labels: labels for single image, or list per image.
            threshold: minimum confidence score to keep.
            vision_embeds: pre-computed features from get_vision_features().
            text_embeds: pre-computed features from get_text_features().

        Returns:
            list of dict per image, each with:
                "scores": (N,) float array of confidence scores
                "boxes": (N, 4) float array in [x1, y1, x2, y2] pixel coords
        """
        raw, original_sizes = self._run_model(
            images,
            text,
            input_boxes,
            input_boxes_labels,
            vision_embeds=vision_embeds,
            text_embeds=text_embeds,
        )

        pred_logits = np.asarray(raw["pred_logits"])
        pred_boxes = np.asarray(raw["pred_boxes"])
        presence = raw.get("presence_logits")
        batch_scores = _compute_scores(pred_logits, presence)

        results = []
        for idx in range(pred_logits.shape[0]):
            scores = batch_scores[idx]
            boxes = _scale_boxes(pred_boxes[idx], original_sizes[idx])
            keep = scores > threshold
            results.append({"scores": scores[keep], "boxes": boxes[keep]})
        return results


# ═══════════════════════════════════════════════════════════════════
#  Instance Segmentation
# ═══════════════════════════════════════════════════════════════════


class SAM3InstanceSegmentation(_SAM3Base):
    """SAM3 for instance segmentation.

    Returns per-object binary masks, bounding boxes, and confidence scores.

    Example:
        segmenter = SAM3InstanceSegmentation.from_pretrained(
            model_weights="sam3.weights.h5",
            text_encoder_weights="sam3_text_encoder.weights.h5",
        )
        results = segmenter.predict(image, text="cat")

        for inst in results:
            for score, box, mask in zip(inst["scores"], inst["boxes"], inst["masks"]):
                print(f"score={score:.2f}, box={box}, mask_shape={mask.shape}")
    """

    def predict(
        self,
        images=None,
        text=None,
        input_boxes=None,
        input_boxes_labels=None,
        threshold=0.3,
        mask_threshold=0.5,
        vision_embeds=None,
        text_embeds=None,
    ):
        """Run instance segmentation.

        Args:
            images: single image or list of images (PIL, numpy, or path).
                Ignored if vision_embeds is provided.
            text: str or list of str. Auto-set to "visual" if only boxes given.
            input_boxes: boxes for single image, or list of boxes per image.
            input_boxes_labels: labels for single image, or list per image.
            threshold: minimum confidence score to keep.
            mask_threshold: threshold for binarizing mask probabilities.
            vision_embeds: pre-computed features from get_vision_features().
            text_embeds: pre-computed features from get_text_features().

        Returns:
            list of dict per image, each with:
                "scores": (N,) float array of confidence scores
                "boxes": (N, 4) float array in [x1, y1, x2, y2] pixel coords
                "masks": (N, H, W) int32 binary masks at original image size
        """
        raw, original_sizes = self._run_model(
            images,
            text,
            input_boxes,
            input_boxes_labels,
            vision_embeds=vision_embeds,
            text_embeds=text_embeds,
        )

        pred_logits = np.asarray(raw["pred_logits"])
        pred_boxes = np.asarray(raw["pred_boxes"])
        pred_masks = np.asarray(raw["pred_masks"])
        presence = raw.get("presence_logits")
        batch_scores = _compute_scores(pred_logits, presence)
        batch_masks = _sigmoid(pred_masks)

        results = []
        for idx in range(pred_logits.shape[0]):
            h, w = original_sizes[idx]
            scores = batch_scores[idx]
            boxes = _scale_boxes(pred_boxes[idx], original_sizes[idx])
            masks = batch_masks[idx]

            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            masks = masks[keep]

            if len(masks) > 0:
                masks = _resize_masks_batch(masks, h, w)
            masks = (masks > mask_threshold).astype(np.int32)

            results.append({"scores": scores, "boxes": boxes, "masks": masks})
        return results


# ═══════════════════════════════════════════════════════════════════
#  Semantic Segmentation
# ═══════════════════════════════════════════════════════════════════


class SAM3SemanticSegmentation(_SAM3Base):
    """SAM3 for semantic segmentation.

    Returns a single binary mask for the queried category across the whole scene.

    Example:
        segmenter = SAM3SemanticSegmentation.from_pretrained(
            model_weights="sam3.weights.h5",
            text_encoder_weights="sam3_text_encoder.weights.h5",
        )
        masks = segmenter.predict(image, text="cat")
        # masks[0] is a (H, W) binary mask at original image size
    """

    def predict(
        self,
        images=None,
        text=None,
        input_boxes=None,
        input_boxes_labels=None,
        threshold=0.5,
        vision_embeds=None,
        text_embeds=None,
    ):
        """Run semantic segmentation.

        Args:
            images: single image or list of images (PIL, numpy, or path).
                Ignored if vision_embeds is provided.
            text: str or list of str. Auto-set to "visual" if only boxes given.
            input_boxes: boxes for single image, or list of boxes per image.
            input_boxes_labels: labels for single image, or list per image.
            threshold: threshold for binarizing the semantic mask.
            vision_embeds: pre-computed features from get_vision_features().
            text_embeds: pre-computed features from get_text_features().

        Returns:
            list of (H, W) int32 binary masks, one per image.
        """
        raw, original_sizes = self._run_model(
            images,
            text,
            input_boxes,
            input_boxes_labels,
            vision_embeds=vision_embeds,
            text_embeds=text_embeds,
        )

        semantic = np.asarray(raw["semantic_seg"])
        probs = _sigmoid(semantic)

        results = []
        for idx in range(semantic.shape[0]):
            h, w = original_sizes[idx]
            mask = probs[idx, 0]
            mask = _resize_mask(mask, h, w)
            mask = (mask > threshold).astype(np.int32)
            results.append(mask)
        return results


# ═══════════════════════════════════════════════════════════════════
#  Visualization utilities
# ═══════════════════════════════════════════════════════════════════

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


def draw_detections(image, results, title=""):
    """Draw detection boxes and scores on image.

    Args:
        image: PIL Image.
        results: dict with "scores" and "boxes" (from SAM3ObjectDetection).
        title: optional title text.

    Returns:
        PIL Image with drawn detections.
    """
    from PIL import ImageDraw

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for i, (score, box) in enumerate(zip(results["scores"], results["boxes"])):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 12), f"{score:.2f}", fill=color)
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))
    return vis


def draw_instance_masks(image, results, title=""):
    """Draw instance masks, boxes, and scores on image.

    Args:
        image: PIL Image.
        results: dict with "scores", "boxes", "masks" (from SAM3InstanceSegmentation).
        title: optional title text.

    Returns:
        PIL Image with drawn masks and boxes.
    """
    from PIL import ImageDraw

    vis = np.array(image, dtype=np.float32)
    for i in range(len(results["scores"])):
        mask = np.asarray(results["masks"][i])
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        vis[mask > 0] = vis[mask > 0] * 0.5 + color * 0.5

    vis = Image.fromarray(vis.astype(np.uint8))
    draw = ImageDraw.Draw(vis)
    for i, (score, box) in enumerate(zip(results["scores"], results["boxes"])):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 12), f"{score:.2f}", fill=color)
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))
    return vis


def draw_semantic_mask(image, mask, title="", color=(0, 200, 255)):
    """Draw semantic segmentation mask overlay on image.

    Args:
        image: PIL Image.
        mask: (H, W) binary mask (from SAM3SemanticSegmentation).
        title: optional title text.
        color: RGB tuple for the mask overlay.

    Returns:
        PIL Image with mask overlay.
    """
    from PIL import ImageDraw

    vis = np.array(image, dtype=np.float32)
    mask = np.asarray(mask)
    vis[mask > 0] = vis[mask > 0] * 0.4 + np.array(color, dtype=np.float32) * 0.6

    vis = Image.fromarray(vis.astype(np.uint8))
    draw = ImageDraw.Draw(vis)
    pct = 100 * mask.sum() / mask.size
    label = f"{title} ({pct:.1f}%)" if title else f"{pct:.1f}%"
    draw.text((10, 10), label, fill=(255, 255, 255))
    return vis
