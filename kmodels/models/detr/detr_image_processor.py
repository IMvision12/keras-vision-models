from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from kmodels.utils.image import preprocess_image

# COCO 2017 class labels (91 categories). Index 0..90 are object classes,
# the model's last logit (index 91) is the "no object" class.
COCO_CLASSES = [
    "N/A",
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
    "N/A",
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
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
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
    "N/A",
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
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
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
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def DETRImageProcessor(
    image: Union[str, np.ndarray, Image.Image],
    size: Optional[Dict[str, int]] = None,
    resample: str = "bilinear",
    do_rescale: bool = True,
    rescale_factor: float = 1 / 255,
    do_normalize: bool = True,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    return_tensor: bool = True,
    data_format: Optional[str] = None,
) -> Union[keras.KerasTensor, np.ndarray]:
    """Preprocess an image for DETR inference.

    Handles loading, resizing, rescaling, and ImageNet normalization to match
    the preprocessing used during DETR training.

    Use this when the model is created with ``include_normalization=False``.
    If ``include_normalization=True``, only resizing is needed and you can
    skip rescale/normalize (or simply pass uint8 images directly to the model).

    Args:
        image: Input image as a file path, numpy array, or PIL Image.
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 800, "width": 800}``.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``).
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_normalize: Whether to apply ImageNet normalization.
        image_mean: Per-channel mean for normalization.
            Default: ``(0.485, 0.456, 0.406)``.
        image_std: Per-channel std for normalization.
            Default: ``(0.229, 0.224, 0.225)``.
        return_tensor: If True return a Keras tensor, otherwise numpy array.

    Returns:
        Preprocessed image with shape ``(1, H, W, 3)`` ready for model input.

    Example:
        ```python
        from kmodels.models.detr import DETRImageProcessor, DETRResNet50

        model = DETRResNet50(weights="detr.weights.h5", include_normalization=False)
        img = DETRImageProcessor("photo.jpg")
        output = model(img, training=False)
        ```
    """
    if size is None:
        size = {"height": 800, "width": 800}
    if image_mean is None:
        image_mean = (0.485, 0.456, 0.406)
    if image_std is None:
        image_std = (0.229, 0.224, 0.225)

    image, _, _, _ = preprocess_image(
        image,
        target_size=(size["height"], size["width"]),
        image_mean=image_mean if do_normalize else None,
        image_std=image_std if do_normalize else None,
        rescale=do_rescale,
        interpolation=resample,
        antialias=False,
        data_format=data_format,
    )
    if do_rescale and rescale_factor != 1 / 255:
        # preprocess_image always divides by 255; re-apply any custom ratio.
        image = image * (rescale_factor * 255)

    if not return_tensor:
        image = keras.ops.convert_to_numpy(image)

    return image


def DETRPostProcessor(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.7,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw DETR outputs into usable detections.

    Converts raw model outputs (logits + normalized boxes) into filtered
    detections with class labels, confidence scores, and bounding boxes
    in ``[x_min, y_min, x_max, y_max]`` pixel coordinates.

    Args:
        outputs: Raw model output dict with keys ``"logits"`` of shape
            ``(B, num_queries, num_classes)`` and ``"pred_boxes"`` of shape
            ``(B, num_queries, 4)`` in normalized ``(cx, cy, w, h)`` format.
        threshold: Minimum confidence score to keep a detection.
        target_sizes: List of ``(height, width)`` tuples for each image in
            the batch. Used to convert normalized boxes to pixel coordinates.
            If None, boxes are returned in normalized ``[0, 1]`` coordinates.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to COCO class names. Provide this
            when using a model fine-tuned on a custom dataset.

    Returns:
        List of dicts (one per image in the batch), each containing:
            - ``"scores"``: Confidence scores, shape ``(num_detections,)``.
            - ``"labels"``: Integer class IDs, shape ``(num_detections,)``.
            - ``"label_names"``: Human-readable COCO class names.
            - ``"boxes"``: Bounding boxes as ``[x_min, y_min, x_max, y_max]``,
              shape ``(num_detections, 4)``.

    Example:
        ```python
        from kmodels.models.detr import DETRResNet50, DETRPostProcessor

        model = DETRResNet50(weights="detr.weights.h5")
        output = model(image, training=False)
        results = DETRPostProcessor(output, threshold=0.7,
                                    target_sizes=[(800, 800)])
        for det in results[0]["label_names"]:
            print(det)
        ```
    """
    logits = keras.ops.convert_to_numpy(outputs["logits"])
    boxes = keras.ops.convert_to_numpy(outputs["pred_boxes"])

    batch_size = logits.shape[0]

    # Softmax over classes; last class is "no object"
    probs = _softmax(logits)

    results = []
    for i in range(batch_size):
        # Scores and labels from object classes only (exclude no-object)
        obj_probs = probs[i, :, :-1]  # (num_queries, num_classes - 1)
        scores = np.max(obj_probs, axis=-1)  # (num_queries,)
        labels = np.argmax(obj_probs, axis=-1)  # (num_queries,)

        # Filter by threshold
        keep = scores > threshold
        scores = scores[keep]
        labels = labels[keep]
        kept_boxes = boxes[i][keep]  # (num_kept, 4) in (cx, cy, w, h)

        # Convert (cx, cy, w, h) -> (x_min, y_min, x_max, y_max)
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

        # Scale to pixel coordinates if target_sizes provided
        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
            xyxy_boxes = xyxy_boxes * scale

        # Map label indices to class names
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


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
