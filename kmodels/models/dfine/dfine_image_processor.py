"""Preprocessing and postprocessing for D-FINE object detection."""

from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from kmodels.base import BaseImageProcessor
from kmodels.utils.image import get_data_format, load_image

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


class DFineImageProcessor(BaseImageProcessor):
    """Preprocess images for D-FINE inference.

    Loads the image (if needed), resizes to ``target_size`` using
    bilinear interpolation (PIL-based to match HF's DFineImageProcessor
    exactly), and rescales pixel values to ``[0, 1]``. D-FINE does
    **not** apply ImageNet normalisation.

    Args:
        target_size: ``(height, width)`` to resize to.
            Defaults to ``(640, 640)``.
        rescale_factor: Multiplicative rescale factor.
            Defaults to ``1/255``.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        rescale_factor: float = 1.0 / 255.0,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_size = target_size
        self.rescale_factor = rescale_factor
        self.data_format = data_format

    def __call__(self, image: Union[str, np.ndarray, "Image.Image"]):
        return self.call(image)

    def call(self, image: Union[str, np.ndarray, "Image.Image"]):
        if isinstance(image, np.ndarray) and image.ndim == 4:
            image = image[0]
        arr = load_image(image)
        pil_img = Image.fromarray(arr)
        if pil_img.size != (self.target_size[1], self.target_size[0]):
            pil_img = pil_img.resize(
                (self.target_size[1], self.target_size[0]), Image.BILINEAR
            )
        image = np.array(pil_img, dtype=np.float32)

        image = ops.convert_to_tensor(image, dtype="float32")
        image = ops.expand_dims(image, axis=0)
        image = image * self.rescale_factor
        if get_data_format(self.data_format) == "channels_first":
            image = ops.transpose(image, (0, 3, 1, 2))
        return image

    def post_process_object_detection(
        self,
        outputs,
        threshold=0.5,
        target_sizes=None,
        num_top_queries=300,
        label_names=None,
    ):
        return dfine_post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            num_top_queries=num_top_queries,
            label_names=label_names,
        )


def dfine_post_process_object_detection(
    outputs: Dict[str, "keras.KerasTensor"],
    threshold: float = 0.5,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    num_top_queries: int = 300,
    label_names: Optional[List[str]] = None,
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
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to COCO class names. Provide this
            when using a model fine-tuned on a custom dataset.

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
        _names = label_names if label_names is not None else COCO_CLASSES
        mapped_names = [
            _names[l] if l < len(_names) else f"class_{l}" for l in labels_np
        ]

        results.append(
            {
                "scores": kept_scores,
                "labels": kept_labels,
                "label_names": mapped_names,
                "boxes": kept_boxes,
            }
        )

    return results
