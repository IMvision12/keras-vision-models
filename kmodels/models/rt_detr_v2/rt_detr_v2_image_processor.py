from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from kmodels.utils.image import preprocess_image

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


def RTDETRV2ImageProcessor(
    image: Union[str, np.ndarray, Image.Image],
    size: Optional[Dict[str, int]] = None,
    resample: str = "bilinear",
    do_rescale: bool = True,
    rescale_factor: float = 1 / 255,
    return_tensor: bool = True,
) -> Union[keras.KerasTensor, np.ndarray]:
    """Preprocess an image for RT-DETRv2 inference.

    Handles loading, resizing, and rescaling to match the preprocessing
    used during RT-DETRv2 training. Unlike DETR, RT-DETRv2 does **not**
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
        from kmodels.models.rt_detr_v2 import (
            RTDETRV2ImageProcessor, RTDETRV2ResNet50,
        )

        model = RTDETRV2ResNet50(weights="coco")
        img = RTDETRV2ImageProcessor("photo.jpg")
        output = model(img, training=False)
        ```
    """
    if size is None:
        size = {"height": 640, "width": 640}

    image, _, _, _ = preprocess_image(
        image,
        target_size=(size["height"], size["width"]),
        image_mean=None,
        image_std=None,
        rescale=do_rescale,
        interpolation=resample,
        antialias=False,
    )
    if do_rescale and rescale_factor != 1 / 255:
        image = image * (rescale_factor * 255)

    if not return_tensor:
        image = keras.ops.convert_to_numpy(image)

    return image


def RTDETRV2PostProcessor(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.5,
    num_top_queries: int = 300,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw RT-DETRv2 outputs into usable detections.

    RT-DETRv2 uses sigmoid activation (not softmax) and has no dedicated
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
        from kmodels.models.rt_detr_v2 import (
            RTDETRV2ResNet50, RTDETRV2ImageProcessor, RTDETRV2PostProcessor,
        )

        model = RTDETRV2ResNet50(weights="coco")
        img = RTDETRV2ImageProcessor("photo.jpg")
        output = model(img, training=False)
        results = RTDETRV2PostProcessor(
            output, threshold=0.5, target_sizes=[(orig_h, orig_w)],
        )
        for r in results:
            for name, score in zip(r["label_names"], r["scores"]):
                print(f"{name}: {score:.2f}")
        ```
    """
    ops = keras.ops
    logits = ops.convert_to_tensor(outputs["logits"])
    boxes = ops.convert_to_tensor(outputs["pred_boxes"])

    batch_size = int(logits.shape[0])
    num_queries = int(logits.shape[1])
    num_classes = int(logits.shape[2])

    probs = ops.sigmoid(logits)

    flat_probs = ops.reshape(probs, (batch_size, num_queries * num_classes))
    num_select = min(num_top_queries, num_queries * num_classes)
    topk_scores, topk_indices = ops.top_k(flat_probs, k=num_select)

    topk_box_indices = topk_indices // num_classes
    topk_labels = topk_indices % num_classes

    gather_idx = ops.expand_dims(topk_box_indices, axis=-1)
    gather_idx = ops.repeat(gather_idx, 4, axis=-1)
    topk_boxes = ops.take_along_axis(boxes, gather_idx, axis=1)

    cx = topk_boxes[..., 0]
    cy = topk_boxes[..., 1]
    w = topk_boxes[..., 2]
    h = topk_boxes[..., 3]
    xyxy_boxes = ops.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    if target_sizes is not None:
        scales = ops.convert_to_tensor(
            [[w_i, h_i, w_i, h_i] for (h_i, w_i) in target_sizes],
            dtype="float32",
        )
        xyxy_boxes = xyxy_boxes * ops.expand_dims(scales, axis=1)

    topk_scores_np = ops.convert_to_numpy(topk_scores)
    topk_labels_np = ops.convert_to_numpy(topk_labels)
    xyxy_boxes_np = ops.convert_to_numpy(xyxy_boxes)

    _names = label_names if label_names is not None else COCO_CLASSES

    results = []
    for i in range(batch_size):
        keep = topk_scores_np[i] > threshold
        scores = topk_scores_np[i][keep]
        labels = topk_labels_np[i][keep]
        kept_boxes = xyxy_boxes_np[i][keep]

        mapped_names = [_names[l] if l < len(_names) else f"class_{l}" for l in labels]

        results.append(
            {
                "scores": scores,
                "labels": labels,
                "label_names": mapped_names,
                "boxes": kept_boxes,
            }
        )

    return results
