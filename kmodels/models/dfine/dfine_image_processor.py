"""Preprocessing and postprocessing for D-FINE object detection."""

from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def DFineImageProcessor(
    image: Union[str, np.ndarray, "Image.Image"],
    target_size: Tuple[int, int] = (640, 640),
    rescale_factor: float = 1.0 / 255.0,
):
    """Preprocess an image for D-FINE inference.

    Loads the image (if needed), resizes to ``target_size`` using bilinear
    interpolation, and rescales pixel values to ``[0, 1]``. D-FINE does
    **not** apply ImageNet normalisation.

    Args:
        image: Input image as a file path, numpy array ``(H, W, 3)`` with
            values in ``[0, 255]``, or a PIL ``Image``.
        target_size: ``(height, width)`` to resize to.
            Defaults to ``(640, 640)``.
        rescale_factor: Multiplicative rescale factor.
            Defaults to ``1/255``.

    Returns:
        A Keras tensor of shape ``(1, height, width, 3)`` with float32
        values in ``[0, 1]``.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
        pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        image = np.array(pil_img, dtype=np.float32)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.float32)
        if image.ndim == 4:
            image = image[0]
        if image.shape[:2] != target_size:
            pil_tmp = Image.fromarray(image.astype(np.uint8))
            pil_tmp = pil_tmp.resize((target_size[1], target_size[0]), Image.BILINEAR)
            image = np.array(pil_tmp, dtype=np.float32)
    else:
        raise TypeError("image must be a file path (str), numpy array, or PIL Image.")

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

    image = ops.convert_to_tensor(image, dtype="float32")
    image = ops.expand_dims(image, axis=0)
    image = image * rescale_factor
    return image


def DFinePostProcessor(
    outputs: Dict[str, "keras.KerasTensor"],
    threshold: float = 0.5,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    num_top_queries: int = 300,
):
    """Post-process raw D-FINE outputs into detection results.

    Applies sigmoid to logits, selects the top scoring ``(query, class)``
    pairs, converts boxes from ``cxcywh`` normalised format to ``xyxy``
    pixel coordinates, and filters by confidence threshold. All operations
    use ``keras.ops`` for backend portability.

    Args:
        outputs: Raw model output dict with ``"logits"`` of shape
            ``(B, Q, C)`` and ``"pred_boxes"`` of shape ``(B, Q, 4)``
            in normalised ``cxcywh`` format.
        threshold: Minimum confidence score to keep a detection.
            Defaults to ``0.5``.
        target_sizes: Optional list of ``(height, width)`` tuples, one
            per image in the batch, used to scale boxes to pixel
            coordinates. If ``None`` boxes stay normalised.
        num_top_queries: Number of top ``(query, class)`` pairs to
            consider before thresholding. Defaults to ``300``.

    Returns:
        List of dicts (one per image) with keys:

        - ``"scores"``: 1-D tensor of confidence scores.
        - ``"labels"``: 1-D integer tensor of class indices.
        - ``"label_names"``: List of class name strings.
        - ``"boxes"``: Tensor of shape ``(N, 4)`` in ``xyxy`` format.
    """
    logits = outputs["logits"]
    pred_boxes = outputs["pred_boxes"]

    batch_size = ops.shape(logits)[0]
    num_classes = ops.shape(logits)[2]

    scores = ops.sigmoid(logits)

    cx = pred_boxes[..., 0]
    cy = pred_boxes[..., 1]
    w = pred_boxes[..., 2]
    h = pred_boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_xyxy = ops.stack([x1, y1, x2, y2], axis=-1)

    results = []
    batch_size_int = int(ops.convert_to_numpy(batch_size))
    num_classes_int = int(ops.convert_to_numpy(num_classes))

    for i in range(batch_size_int):
        scores_i = scores[i]  # (Q, C)
        boxes_i = boxes_xyxy[i]  # (Q, 4)

        flat_scores = ops.reshape(scores_i, [-1])

        num_select = min(num_top_queries, int(ops.shape(flat_scores)[0]))
        topk_scores, topk_indices = ops.top_k(flat_scores, k=num_select)

        topk_labels = topk_indices % num_classes_int
        topk_box_idx = ops.cast(topk_indices // num_classes_int, "int32")

        topk_boxes = ops.take(boxes_i, topk_box_idx, axis=0)

        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            scale = ops.convert_to_tensor([img_w, img_h, img_w, img_h], dtype="float32")
            topk_boxes = topk_boxes * scale

        keep = topk_scores > threshold
        kept_scores = topk_scores[keep]
        kept_labels = topk_labels[keep]
        kept_boxes = topk_boxes[keep]

        labels_np = ops.convert_to_numpy(kept_labels).astype(int)
        label_names = [
            COCO_CLASSES[l] if l < len(COCO_CLASSES) else f"class_{l}"
            for l in labels_np
        ]

        results.append(
            {
                "scores": kept_scores,
                "labels": kept_labels,
                "label_names": label_names,
                "boxes": kept_boxes,
            }
        )

    return results
