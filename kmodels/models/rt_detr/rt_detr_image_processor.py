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


def RTDETRImageProcessor(
    image: Union[str, np.ndarray, Image.Image],
    size: Optional[Dict[str, int]] = None,
    resample: str = "bilinear",
    do_rescale: bool = True,
    rescale_factor: float = 1 / 255,
    return_tensor: bool = True,
) -> Union[keras.KerasTensor, np.ndarray]:
    """Preprocess an image for RT-DETR inference.

    Handles loading, resizing, and rescaling to match the preprocessing
    used during RT-DETR training. Unlike DETR, RT-DETR does **not**
    apply ImageNet normalization; only rescaling to ``[0, 1]`` is needed.

    Args:
        image: Input image as a file path, numpy array, or PIL Image.
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 640, "width": 640}``.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``).
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        return_tensor: If True return a Keras tensor, otherwise numpy
            array.

    Returns:
        Preprocessed image with shape ``(1, H, W, 3)`` ready for model
        input.

    Example:
        ```python
        from kmodels.models.rt_detr import RTDETRImageProcessor, RTDETRResNet50

        model = RTDETRResNet50(weights="coco")
        img = RTDETRImageProcessor("photo.jpg")
        output = model(img, training=False)
        ```
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


def RTDETRPostProcessor(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.5,
    num_top_queries: int = 300,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw RT-DETR outputs into usable detections.

    RT-DETR uses sigmoid activation (not softmax) and has no dedicated
    background class. This post-processor applies sigmoid to raw logits,
    selects top-K scoring ``(query, class)`` pairs, converts boxes from
    normalised ``(cx, cy, w, h)`` to ``(x_min, y_min, x_max, y_max)``
    pixel coordinates, and filters by score threshold.

    Args:
        outputs: Raw model output dict with keys ``"logits"`` of shape
            ``(B, num_queries, num_classes)`` and ``"pred_boxes"`` of
            shape ``(B, num_queries, 4)`` in normalised
            ``(cx, cy, w, h)`` format.
        threshold: Minimum confidence score to keep a detection.
        num_top_queries: Number of top ``(query, class)`` pairs to
            consider before threshold filtering. Default ``300``.
        target_sizes: List of ``(height, width)`` tuples for each image
            in the batch. Used to convert normalised boxes to pixel
            coordinates. If ``None``, boxes stay in ``[0, 1]``
            coordinates.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to COCO class names. Provide this
            when using a model fine-tuned on a custom dataset.

    Returns:
        List of dicts (one per image in the batch), each containing:
            - ``"scores"``: Confidence scores ``(num_detections,)``.
            - ``"labels"``: Integer class IDs ``(num_detections,)``.
            - ``"label_names"``: Human-readable COCO class names.
            - ``"boxes"``: ``[x_min, y_min, x_max, y_max]``
              ``(num_detections, 4)``.

    Example:
        ```python
        from kmodels.models.rt_detr import (
            RTDETRResNet50, RTDETRImageProcessor, RTDETRPostProcessor,
        )

        model = RTDETRResNet50(weights="coco")
        img = RTDETRImageProcessor("photo.jpg")
        output = model(img, training=False)
        results = RTDETRPostProcessor(
            output, threshold=0.5, target_sizes=[(orig_h, orig_w)],
        )
        for r in results:
            for name, score in zip(r["label_names"], r["scores"]):
                print(f"{name}: {score:.2f}")
        ```
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

        _names = label_names if label_names is not None else COCO_CLASSES
        mapped_names = [_names[l] if l < len(_names) else f"class_{l}" for l in labels]

        results.append(
            {
                "scores": scores,
                "labels": labels,
                "label_names": mapped_names,
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
