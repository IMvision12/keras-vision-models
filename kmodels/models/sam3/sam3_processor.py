"""SAM3 Processor: end-to-end preprocessing, inference, and post-processing.

Matches HF Sam3Processor functionality using pure Keras 3 ops.
Supports text prompts and box prompts for detection + segmentation.
"""

import keras
import numpy as np
from keras import ops
from PIL import Image

from .sam3_clip_tokenizer import SAM3CLIPTokenizer
from .sam3_model import SAM3Main
from .sam3_utils import (
    box_xyxy_to_cxcywh,
    compute_scores,
    compute_sine_pos_encoding,
    scale_boxes,
    sigmoid,
)


IMAGE_SIZE = 1008
RESCALE_FACTOR = 1.0 / 255.0
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess_image(image, target_size=IMAGE_SIZE):
    """Preprocess image for SAM3: resize, rescale, normalize.

    Uses keras.ops.image.resize for backend-native bilinear interpolation
    (matches torchvision on torch backend, tf.image on TF backend).
    Rescales via float64 intermediate to match HF's precision.

    Args:
        image: PIL Image, numpy array (H,W,3), or file path.
        target_size: int, target square size.

    Returns:
        pixel_values: (1, H, W, 3) float32 normalized.
        original_size: (height, width) tuple.
    """
    if isinstance(image, str):
        if Image is None:
            raise ImportError("PIL required for loading images from paths")
        image = Image.open(image).convert("RGB")

    if Image is not None and isinstance(image, Image.Image):
        original_size = (image.height, image.width)
        image = np.array(image)
    else:
        image = np.asarray(image)
        original_size = (image.shape[0], image.shape[1])
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

    image_t = ops.convert_to_tensor(image.astype(np.float32) / 256.0)
    image_4d = ops.expand_dims(image_t, 0)
    resized = ops.image.resize(
        image_4d, (target_size, target_size), interpolation="bilinear"
    )
    resized = resized * 256.0
    resized = ops.clip(resized, 0, 255)
    resized = ops.round(resized)
    resized = ops.convert_to_numpy(resized)[0]

    image = (resized.astype(np.float64) * RESCALE_FACTOR).astype(np.float32)
    image = (image - IMAGE_MEAN) / IMAGE_STD
    return image[np.newaxis], original_size


def preprocess_text_with_encoder(text, text_encoder_model, tokenizer=None):
    """Tokenize and encode text using the CLIP text encoder.

    Args:
        text: str or list of str.
        text_encoder_model: Keras SAM3 text encoder model.
        tokenizer: SAM3CLIPTokenizer instance. If None, creates one automatically.

    Returns:
        text_features: (batch, 32, 1024) numpy array.
        attention_mask: (batch, 32) float32 numpy array.
    """
    if tokenizer is None:
        tokenizer = SAM3CLIPTokenizer()

    input_ids, attention_mask = tokenizer.encode(text)

    text_features = text_encoder_model.predict(
        {"input_ids": input_ids, "attention_mask": attention_mask.astype(np.int32)},
        verbose=0,
    )
    return text_features, attention_mask


POINT_PAD_VALUE = -10


def preprocess_boxes(input_boxes, input_boxes_labels, original_sizes):
    """Normalize and convert box prompts to model input format.

    Args:
        input_boxes: list of list of [x1,y1,x2,y2] boxes per image.
            Shape: [batch, num_boxes, 4] in absolute pixel coordinates.
        input_boxes_labels: list of list of int labels (0 or 1) per image.
            Shape: [batch, num_boxes].
        original_sizes: list of (H, W) tuples per image.

    Returns:
        boxes_cxcywh: (batch, max_boxes, 4) float32, normalized cxcywh.
        box_labels: (batch, max_boxes) int32.
        box_mask: (batch, max_boxes) float32, 1=valid 0=padding.
    """
    if isinstance(input_boxes, np.ndarray):
        input_boxes = input_boxes.tolist()
    if isinstance(input_boxes_labels, np.ndarray):
        input_boxes_labels = input_boxes_labels.tolist()

    batch_size = len(input_boxes)

    max_boxes = max(len(boxes) for boxes in input_boxes)

    all_boxes = []
    all_labels = []
    all_masks = []

    for img_idx in range(batch_size):
        boxes = input_boxes[img_idx]
        labels = input_boxes_labels[img_idx] if input_boxes_labels else [1] * len(boxes)
        h, w = original_sizes[img_idx]

        normalized = []
        for box in boxes:
            x1, y1, x2, y2 = box
            normalized.append([x1 / w, y1 / h, x2 / w, y2 / h])

        if normalized:
            norm_arr = np.array(normalized, dtype=np.float32)
            cxcywh = box_xyxy_to_cxcywh(norm_arr)
        else:
            cxcywh = np.zeros((0, 4), dtype=np.float32)

        num_boxes = len(boxes)
        pad_count = max_boxes - num_boxes
        if pad_count > 0:
            pad_boxes = np.full((pad_count, 4), POINT_PAD_VALUE, dtype=np.float32)
            cxcywh = np.concatenate([cxcywh, pad_boxes], axis=0)
            labels = list(labels) + [0] * pad_count

        mask = np.array([1.0] * num_boxes + [0.0] * pad_count, dtype=np.float32)

        all_boxes.append(cxcywh)
        all_labels.append(np.array(labels, dtype=np.int32))
        all_masks.append(mask)

    return (
        np.stack(all_boxes),
        np.stack(all_labels),
        np.stack(all_masks),
    )



def post_process_object_detection(outputs, threshold=0.3, target_sizes=None):
    """Convert raw outputs to detection results.

    Returns: list of dict with 'scores' and 'boxes'.
    """
    pred_logits = np.asarray(outputs["pred_logits"])
    pred_boxes = np.asarray(outputs["pred_boxes"])
    presence = outputs.get("presence_logits")
    batch_scores = compute_scores(pred_logits, presence)

    results = []
    for idx in range(pred_logits.shape[0]):
        scores = batch_scores[idx]
        boxes = pred_boxes[idx].copy()
        if target_sizes is not None:
            boxes = scale_boxes(boxes, target_sizes[idx])
        keep = scores > threshold
        results.append({"scores": scores[keep], "boxes": boxes[keep]})
    return results


def post_process_instance_segmentation(
    outputs, threshold=0.3, mask_threshold=0.5, target_sizes=None
):
    """Convert raw outputs to instance segmentation results.

    Returns: list of dict with 'scores', 'boxes', 'masks'.
    """
    pred_logits = np.asarray(outputs["pred_logits"])
    pred_boxes = np.asarray(outputs["pred_boxes"])
    pred_masks = np.asarray(outputs["pred_masks"])
    presence = outputs.get("presence_logits")
    batch_scores = compute_scores(pred_logits, presence)
    batch_masks = sigmoid(pred_masks)

    results = []
    for idx in range(pred_logits.shape[0]):
        scores = batch_scores[idx]
        boxes = pred_boxes[idx].copy()
        masks = batch_masks[idx]

        if target_sizes is not None:
            boxes = scale_boxes(boxes, target_sizes[idx])

        keep = scores > threshold
        scores = scores[keep]
        boxes = boxes[keep]
        masks = masks[keep]

        if target_sizes is not None and len(masks) > 0 and Image is not None:
            th, tw = target_sizes[idx]
            resized = []
            for m in masks:
                pil_m = Image.fromarray((m * 255).astype(np.uint8))
                pil_m = pil_m.resize((tw, th), Image.BILINEAR)
                resized.append(np.array(pil_m, dtype=np.float32) / 255.0)
            masks = np.stack(resized)

        masks = (masks > mask_threshold).astype(np.int32)
        results.append({"scores": scores, "boxes": boxes, "masks": masks})
    return results


def post_process_semantic_segmentation(outputs, target_sizes=None, threshold=0.5):
    """Convert raw outputs to semantic segmentation maps.

    Returns: list of binary masks (H, W).
    """
    semantic = np.asarray(outputs["semantic_seg"])
    probs = sigmoid(semantic)

    results = []
    for idx in range(semantic.shape[0]):
        mask = probs[idx, 0]
        if target_sizes is not None and Image is not None:
            th, tw = target_sizes[idx]
            pil_m = Image.fromarray((mask * 255).astype(np.uint8))
            pil_m = pil_m.resize((tw, th), Image.BILINEAR)
            mask = np.array(pil_m, dtype=np.float32) / 255.0
        mask = (mask > threshold).astype(np.int32)
        results.append(mask)
    return results


_SUBMODEL_CACHE = {}


def _get_vision_submodel(model):
    """Sub-model: full inputs → FPN 1x (NCHW) + projected text (256d)."""
    model_id = id(model)
    cache_key = f"{model_id}_vision"
    if cache_key in _SUBMODEL_CACHE:
        return _SUBMODEL_CACHE[cache_key]

    det = model
    fpn_1x = det.get_layer("fpn_level_2_proj2")
    text_proj = det.get_layer("text_projection")
    sub = keras.Model(
        inputs=det.input,
        outputs={"fpn_1x": fpn_1x.output, "text_projected": text_proj.output},
    )
    _SUBMODEL_CACHE[cache_key] = sub
    return sub


def _get_decoder_model(model):
    """Build a model that takes pixel_values + pre-projected features (256d).

    Shares all weights with the original SAM3 model but replaces the
    text_features(1024d) input with a projected_features(256d) input
    that bypasses text_projection.
    """

    model_id = id(model)
    cache_key = f"{model_id}_decoder"
    if cache_key in _SUBMODEL_CACHE:
        return _SUBMODEL_CACHE[cache_key]

    det = model
    cfg = det.get_config()
    decoder_model = SAM3Main(
        input_shape=det._input_shape_val,
        text_hidden_size=cfg["detr_encoder_hidden_size"],
        **{
            k: v
            for k, v in cfg.items()
            if k
            not in [
                "input_shape",
                "text_hidden_size",
                "name",
                "input_tensor",
            ]
        },
        name="SAM3_decoder",
    )

    orig_weights = {w.path: w.numpy() for w in det.weights}
    for w in decoder_model.weights:
        path = w.path.replace("SAM3_decoder/", "SAM3Main/")
        if path in orig_weights:
            if w.shape == orig_weights[path].shape:
                w.assign(orig_weights[path])

    tp = decoder_model.get_layer("text_projection")
    tp.kernel.assign(np.eye(256, dtype=np.float32))
    tp.bias.assign(np.zeros(256, dtype=np.float32))

    _SUBMODEL_CACHE[cache_key] = decoder_model
    return decoder_model


def _run_decoder_with_projected(model, pixel_values, combined_features, combined_mask):
    """Run SAM3 with pre-projected 256d features via decoder model."""
    decoder_model = _get_decoder_model(model)
    return decoder_model.predict(
        {
            "pixel_values": pixel_values,
            "text_features": combined_features,
            "text_attention_mask": combined_mask,
        },
        verbose=0,
    )


def predict(
    model,
    image,
    text=None,
    tokenizer=None,
    text_encoder_model=None,
    text_features=None,
    text_attention_mask=None,
    input_boxes=None,
    input_boxes_labels=None,
    geometry_encoder=None,
    threshold=0.3,
    mask_threshold=0.5,
    return_raw=False,
):
    """End-to-end SAM3 prediction pipeline.

    Supports three modes:
    - Mode A (text-only): text prompt → detection + segmentation.
    - Mode B (box-guided): box prompts (text defaults to "visual").
    - Mode C (hybrid): text + box prompts combined.

    Args:
        model: Keras SAM3 segmentation model.
        image: PIL Image, numpy array, or file path.
        text: Text prompt string (requires tokenizer + text_encoder_model).
        tokenizer: CLIP tokenizer.
        text_encoder_model: Keras CLIP text encoder model.
        text_features: Pre-computed (1, seq, 1024) features.
        text_attention_mask: (1, seq) mask.
        input_boxes: list of [x1,y1,x2,y2] boxes per image (pixel coords).
            Shape: [[box1, box2, ...]] for single image.
        input_boxes_labels: list of int labels per image (1=positive, 0=negative).
            Shape: [[label1, label2, ...]] for single image. Defaults to all 1s.
        geometry_encoder: SAM3GeometryEncoder layer with loaded weights.
            Required when input_boxes is provided.
        threshold: Detection score threshold.
        mask_threshold: Mask binarization threshold.
        return_raw: Return raw model outputs.

    Returns:
        dict with 'detection', 'instance_segmentation', 'semantic_segmentation'.
    """
    pixel_values, original_size = preprocess_image(image)

    if text is None and text_features is None and input_boxes is not None:
        text = "visual"

    if text_features is None:
        if text is None:
            raise ValueError("Provide either text, text_features, or input_boxes.")
        if text_encoder_model is None:
            raise ValueError("text_encoder_model required for text input.")
        text_features, text_attention_mask = preprocess_text_with_encoder(
            text, text_encoder_model, tokenizer
        )
    else:
        text_features = np.asarray(text_features)
        if text_attention_mask is None:
            text_attention_mask = np.ones(text_features.shape[:2], dtype=np.float32)
        else:
            text_attention_mask = np.asarray(text_attention_mask, dtype=np.float32)

    if input_boxes is not None:
        if geometry_encoder is None:
            raise ValueError("geometry_encoder required when input_boxes is provided.")

        if not isinstance(input_boxes[0][0], (list, tuple)):
            input_boxes = [input_boxes]
        if input_boxes_labels is None:
            input_boxes_labels = [[1] * len(boxes) for boxes in input_boxes]
        elif not isinstance(input_boxes_labels[0], (list, tuple)):
            input_boxes_labels = [input_boxes_labels]

        boxes_cxcywh, box_labels, box_mask = preprocess_boxes(
            input_boxes, input_boxes_labels, [original_size]
        )

        fpn_submodel = _get_vision_submodel(model)
        fpn_inputs = {
            "pixel_values": pixel_values,
            "text_features": text_features,
            "text_attention_mask": text_attention_mask,
        }
        fpn_out = fpn_submodel.predict(fpn_inputs, verbose=0)
        df = keras.config.image_data_format()
        fpn_1x = fpn_out["fpn_1x"]
        text_projected = fpn_out["text_projected"]

        if df == "channels_first":
            fpn_1x_nhwc = np.transpose(fpn_1x, (0, 2, 3, 1))
            enc_h = fpn_1x.shape[2]
        else:
            fpn_1x_nhwc = fpn_1x
            enc_h = fpn_1x.shape[1]
        vision_flat = fpn_1x_nhwc.reshape(1, enc_h * enc_h, -1)

        pos = compute_sine_pos_encoding(enc_h, enc_h, 128, normalize=True)
        pos_flat = ops.convert_to_numpy(pos).reshape(1, enc_h * enc_h, -1)

        geo_features, geo_mask = geometry_encoder(
            boxes_cxcywh,
            box_labels,
            box_mask,
            vision_flat,
            pos_flat,
            vision_features_spatial=fpn_1x,
            data_format=df,
        )
        geo_features = ops.convert_to_numpy(ops.stop_gradient(geo_features))
        geo_mask = ops.convert_to_numpy(ops.stop_gradient(geo_mask))

        combined_features = np.concatenate([text_projected, geo_features], axis=1)
        combined_mask = np.concatenate([text_attention_mask, geo_mask], axis=1)

        raw = _run_decoder_with_projected(
            model, pixel_values, combined_features, combined_mask
        )
    else:
        raw = model.predict(
            {
                "pixel_values": pixel_values,
                "text_features": text_features,
                "text_attention_mask": text_attention_mask,
            },
            verbose=0,
        )

    if return_raw:
        return raw

    target_sizes = [original_size]
    return {
        "detection": post_process_object_detection(raw, threshold, target_sizes),
        "instance_segmentation": post_process_instance_segmentation(
            raw, threshold, mask_threshold, target_sizes
        ),
        "semantic_segmentation": post_process_semantic_segmentation(
            raw, target_sizes, mask_threshold
        ),
    }
