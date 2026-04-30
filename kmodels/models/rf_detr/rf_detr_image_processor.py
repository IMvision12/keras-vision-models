from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from kmodels.base import BaseImageProcessor
from kmodels.utils.image import preprocess_image
from kmodels.utils.labels import COCO_91_CLASSES


class RFDETRImageProcessor(BaseImageProcessor):
    """Preprocess images for RF-DETR inference.

    Args:
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 560, "width": 560}`` (RFDETRBase
            resolution). Use the model's resolution: 384 (Nano),
            512 (Small), 576 (Medium), 560 (Base), or 704 (Large).
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``).
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_normalize: Whether to apply ImageNet normalization.
        image_mean: Per-channel mean. Default: ``(0.485, 0.456, 0.406)``.
        image_std: Per-channel std. Default: ``(0.229, 0.224, 0.225)``.
        return_tensor: If True return a Keras tensor, otherwise numpy.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.
    """

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        resample: str = "bilinear",
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size if size is not None else {"height": 560, "width": 560}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean if image_mean is not None else (0.485, 0.456, 0.406)
        )
        self.image_std = image_std if image_std is not None else (0.229, 0.224, 0.225)
        self.return_tensor = return_tensor
        self.data_format = data_format

    def __call__(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> Union[keras.KerasTensor, np.ndarray]:
        return self.call(image)

    def call(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> Union[keras.KerasTensor, np.ndarray]:
        image, _, _, _ = preprocess_image(
            image,
            target_size=(self.size["height"], self.size["width"]),
            image_mean=self.image_mean if self.do_normalize else None,
            image_std=self.image_std if self.do_normalize else None,
            rescale=self.do_rescale,
            interpolation=self.resample,
            antialias=True,
            data_format=self.data_format,
        )
        if self.do_rescale and self.rescale_factor != 1 / 255:
            image = image * (self.rescale_factor * 255)

        if not self.return_tensor:
            image = keras.ops.convert_to_numpy(image)

        return {"pixel_values": image}

    def post_process_object_detection(
        self,
        outputs,
        threshold=0.5,
        num_top_queries=300,
        target_sizes=None,
        label_names=None,
    ):
        return rf_detr_post_process_object_detection(
            outputs,
            threshold=threshold,
            num_top_queries=num_top_queries,
            target_sizes=target_sizes,
            label_names=label_names,
        )


def rf_detr_post_process_object_detection(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.5,
    num_top_queries: int = 300,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw RF-DETR outputs into usable detections.

    RF-DETR uses sigmoid activation (not softmax) and does not have a
    dedicated background class. This post-processor applies sigmoid to raw
    logits, selects top-K scoring (query, class) pairs, converts boxes from
    normalized cxcywh to xyxy pixel coordinates, and filters by score threshold.

    Args:
        outputs: Raw model output dict with keys ``"pred_logits"`` of shape
            ``(B, num_queries, num_classes)`` and ``"pred_boxes"`` of shape
            ``(B, num_queries, 4)`` in normalized ``(cx, cy, w, h)`` format.
        threshold: Minimum confidence score to keep a detection.
        num_top_queries: Number of top (query, class) pairs to consider
            before threshold filtering. Default 300.
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
        from kmodels.models.rf_detr import RFDETRBase, RFDETRImageProcessor, rf_detr_post_process_object_detection

        model = RFDETRBase(weights="coco")
        img = RFDETRImageProcessor("photo.jpg", size={"height": 560, "width": 560})
        output = model(img, training=False)
        results = rf_detr_post_process_object_detection(output, threshold=0.5,
                                      target_sizes=[(orig_h, orig_w)])
        for r in results:
            for name, score in zip(r["label_names"], r["scores"]):
                print(f"{name}: {score:.2f}")
        ```
    """
    logits = keras.ops.convert_to_numpy(outputs["pred_logits"])
    boxes = keras.ops.convert_to_numpy(outputs["pred_boxes"])

    batch_size = logits.shape[0]
    num_classes = logits.shape[2]

    probs = _sigmoid(logits)

    results = []
    for i in range(batch_size):
        prob_i = probs[i]  # (num_queries, num_classes)
        boxes_i = boxes[i]  # (num_queries, 4)

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
            scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
            xyxy_boxes = xyxy_boxes * scale

        _names = label_names if label_names is not None else COCO_91_CLASSES
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
