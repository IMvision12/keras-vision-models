from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
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
    image: Union[str, np.ndarray, Image.Image],
    size: Optional[Dict[str, int]] = None,
    resample: str = "bilinear",
    do_rescale: bool = True,
    rescale_factor: float = 1 / 255,
    return_tensor: bool = True,
) -> Union[keras.KerasTensor, np.ndarray]:
    """Preprocess an image for D-FINE inference.

    Handles loading, resizing, and rescaling to match the preprocessing
    used during D-FINE training. Like RT-DETR, D-FINE does **not** apply
    ImageNet normalization; only rescaling to ``[0, 1]`` is needed.

    Args:
        image: Input image as a file path, numpy array, or PIL Image.
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 640, "width": 640}``.
        resample: Interpolation method.
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        return_tensor: If True return a Keras tensor, otherwise numpy.

    Returns:
        Preprocessed image with shape ``(1, H, W, 3)``.
    """
    if size is None:
        size = {"height": 640, "width": 640}

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        image = np.array(image, dtype=np.float32)
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.float32)
        if image.ndim == 4:
            image = image[0]
    else:
        raise TypeError("Input must be a file path (str), numpy array, or PIL Image.")

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

    image = keras.ops.convert_to_tensor(image, dtype="float32")
    image = keras.ops.expand_dims(image, axis=0)

    target_size = (size["height"], size["width"])
    image = keras.ops.image.resize(
        image,
        size=target_size,
        interpolation=resample,
    )

    if do_rescale:
        image = image * rescale_factor

    if not return_tensor:
        image = keras.ops.convert_to_numpy(image)

    return image


def DFinePostProcessor(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.5,
    num_top_queries: int = 300,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw D-FINE outputs into usable detections.

    Args:
        outputs: Raw model output dict with keys ``"logits"`` and
            ``"pred_boxes"``.
        threshold: Minimum confidence score to keep.
        num_top_queries: Top ``(query, class)`` pairs to consider.
        target_sizes: List of ``(height, width)`` tuples for scaling.

    Returns:
        List of dicts per image with ``"scores"``, ``"labels"``,
        ``"label_names"``, and ``"boxes"`` in xyxy format.
    """
    logits = keras.ops.convert_to_numpy(outputs["logits"])
    boxes = keras.ops.convert_to_numpy(outputs["pred_boxes"])

    batch_size = logits.shape[0]
    num_classes = logits.shape[2]

    probs = _sigmoid(logits)

    results = []
    for i in range(batch_size):
        prob_i = probs[i]
        boxes_i = boxes[i]

        flat_scores = prob_i.reshape(-1)
        num_select = min(num_top_queries, flat_scores.shape[0])
        topk_indices = np.argpartition(flat_scores, -num_select)[-num_select:]
        topk_indices = topk_indices[np.argsort(-flat_scores[topk_indices])]

        topk_scores = flat_scores[topk_indices]
        topk_box_indices = topk_indices // num_classes
        topk_labels = topk_indices % num_classes

        topk_boxes = boxes_i[topk_box_indices]

        keep = topk_scores > threshold
        scores = topk_scores[keep]
        labels = topk_labels[keep]
        kept_boxes = topk_boxes[keep]

        cx, cy, w, h = (
            kept_boxes[:, 0],
            kept_boxes[:, 1],
            kept_boxes[:, 2],
            kept_boxes[:, 3],
        )
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        xyxy_boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)

        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            scale = np.array(
                [img_w, img_h, img_w, img_h],
                dtype=np.float32,
            )
            xyxy_boxes = xyxy_boxes * scale

        label_names = [
            COCO_CLASSES[l] if l < len(COCO_CLASSES) else f"class_{l}" for l in labels
        ]

        results.append(
            {
                "scores": scores,
                "labels": labels,
                "label_names": label_names,
                "boxes": xyxy_boxes,
            }
        )

    return results


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
