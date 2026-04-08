"""SAM3 Processor: end-to-end preprocessing, inference, and post-processing.

Matches HF Sam3Processor functionality using pure Keras 3 ops.
Supports text prompts and box prompts for detection + segmentation.
"""

import keras
import numpy as np
from keras import ops
from PIL import Image

from .sam3_clip_tokenizer import SAM3CLIPTokenizer
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
    """Preprocess an image for SAM3 inference.

    Resizes to a square target size using backend-native bilinear
    interpolation, applies rescaling via float64 intermediate to
    match HF precision, and normalizes with ImageNet-style mean/std.

    Args:
        image: PIL Image, numpy array ``(H, W, 3)``, or file path.
        target_size (int): Target square size. Defaults to ``1008``.

    Returns:
        Tuple of ``(pixel_values, original_size)`` where
        ``pixel_values`` is ``(1, H, W, 3)`` float32 and
        ``original_size`` is ``(height, width)``.
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
        text: String or list of strings to encode.
        text_encoder_model: Keras CLIP text encoder model (functional).
        tokenizer: ``SAM3CLIPTokenizer`` instance. If ``None``, creates
            one automatically.

    Returns:
        Tuple of ``(text_features, attention_mask)`` where
        ``text_features`` is ``(batch, 32, 1024)`` and
        ``attention_mask`` is ``(batch, 32)`` float32.
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

    Converts absolute pixel coordinates to normalized ``(cx, cy, w, h)``
    format and pads to the maximum number of boxes in the batch.

    Args:
        input_boxes: List of list of ``[x1, y1, x2, y2]`` boxes per
            image in absolute pixel coordinates.
        input_boxes_labels: List of list of int labels (0 or 1) per
            image, or ``None`` for all-positive.
        original_sizes: List of ``(H, W)`` tuples per image.

    Returns:
        Tuple of ``(boxes_cxcywh, box_labels, box_mask)`` where
        shapes are ``(batch, max_boxes, 4)``, ``(batch, max_boxes)``,
        and ``(batch, max_boxes)`` respectively.
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
    """Convert raw model outputs to detection results.

    Applies sigmoid scoring with optional presence logits, scales
    boxes to target image sizes, and filters by confidence threshold.

    Args:
        outputs: Dict with ``"pred_logits"`` ``(B, Q)`` and
            ``"pred_boxes"`` ``(B, Q, 4)`` in normalized cxcywh.
        threshold (float): Minimum score to keep. Defaults to ``0.3``.
        target_sizes: List of ``(H, W)`` tuples for box scaling,
            or ``None``.

    Returns:
        List of dicts, each with ``"scores"`` and ``"boxes"`` arrays.
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
    """Convert raw model outputs to instance segmentation results.

    Applies sigmoid to masks, resizes to target sizes using PIL
    bilinear interpolation, and binarizes with ``mask_threshold``.

    Args:
        outputs: Dict with ``"pred_logits"``, ``"pred_boxes"``, and
            ``"pred_masks"`` ``(B, Q, H, W)``.
        threshold (float): Minimum score to keep. Defaults to ``0.3``.
        mask_threshold (float): Binarization threshold for masks.
            Defaults to ``0.5``.
        target_sizes: List of ``(H, W)`` tuples, or ``None``.

    Returns:
        List of dicts, each with ``"scores"``, ``"boxes"``, and
        ``"masks"`` arrays.
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
    """Convert raw model outputs to semantic segmentation maps.

    Applies sigmoid to the single-channel semantic output, resizes
    to target sizes, and binarizes with ``threshold``.

    Args:
        outputs: Dict with ``"semantic_seg"`` ``(B, 1, H, W)`` or
            ``(B, H, W, 1)``.
        target_sizes: List of ``(H, W)`` tuples, or ``None``.
        threshold (float): Binarization threshold. Defaults to ``0.5``.

    Returns:
        List of ``(H, W)`` int32 binary mask arrays.
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
    """Build a vision-only sub-model for FPN feature extraction.

    Creates a ``keras.Model`` that outputs the 1x FPN feature map
    and projected text features from the full SAM3 model. Results
    are cached per model instance.

    Args:
        model: ``SAM3Main`` instance.

    Returns:
        ``keras.Model`` with outputs ``{"fpn_1x", "text_projected"}``.
    """
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
    """Build a decoder model that bypasses text projection.

    Creates a copy of the SAM3 model where the text projection layer
    is replaced with an identity mapping, allowing pre-projected 256d
    features (from text + geometry) to be passed directly.

    Args:
        model: ``SAM3Main`` instance.

    Returns:
        ``SAM3Main`` instance with identity text projection.
    """
    model_id = id(model)
    cache_key = f"{model_id}_decoder"
    if cache_key in _SUBMODEL_CACHE:
        return _SUBMODEL_CACHE[cache_key]

    from .sam3_model import SAM3Main

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
    """Run SAM3 inference with pre-projected 256d features.

    Uses the decoder model (with identity text projection) to run
    inference using combined text + geometry features that have
    already been projected to the encoder hidden dimension.

    Args:
        model: ``SAM3Main`` instance.
        pixel_values: ``(1, H, W, 3)`` preprocessed image.
        combined_features: ``(1, seq, 256)`` projected features.
        combined_mask: ``(1, seq)`` attention mask.

    Returns:
        Dict of raw model outputs.
    """
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

    Supports three prompt modes:

    - **Mode A** (text-only): Text prompt for detection + segmentation.
    - **Mode B** (box-guided): Box prompts with implicit "visual" text.
    - **Mode C** (hybrid): Text + box prompts combined.

    When box prompts are provided, the geometry encoder fuses box
    features with vision features, and the combined prompt is passed
    through a decoder model with identity text projection.

    Args:
        model: ``SAM3Main`` instance.
        image: PIL Image, numpy array, or file path.
        text (str): Text prompt. Required unless ``text_features``
            or ``input_boxes`` is provided.
        tokenizer: ``SAM3CLIPTokenizer`` instance. Created
            automatically if ``None``.
        text_encoder_model: Keras CLIP text encoder model. Required
            when ``text`` is provided.
        text_features: Pre-computed ``(1, seq, 1024)`` text features.
        text_attention_mask: ``(1, seq)`` mask for pre-computed
            features.
        input_boxes: List of ``[x1, y1, x2, y2]`` boxes in absolute
            pixel coordinates.
        input_boxes_labels: List of int labels per box (1=positive,
            0=negative). Defaults to all positive.
        geometry_encoder: ``SAM3GeometryEncoder`` layer. Required
            when ``input_boxes`` is provided.
        threshold (float): Detection score threshold.
            Defaults to ``0.3``.
        mask_threshold (float): Mask binarization threshold.
            Defaults to ``0.5``.
        return_raw (bool): Return raw model outputs instead of
            post-processed results. Defaults to ``False``.

    Returns:
        Dict with ``"detection"``, ``"instance_segmentation"``, and
        ``"semantic_segmentation"`` keys, each containing a list of
        per-image result dicts. If ``return_raw=True``, returns the
        raw model output dict instead.
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
