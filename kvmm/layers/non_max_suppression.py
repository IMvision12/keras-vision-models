import keras
import numpy as np
from keras import ops
from keras.layers import Layer


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes.

    This function calculates the IoU metric, which measures the overlap between two
    bounding boxes. IoU is defined as the area of intersection divided by the area
    of union of the two boxes. It returns a value between 0 and 1, where 0 indicates
    no overlap and 1 indicates perfect overlap. A small epsilon (1e-7) is added to
    the denominator to prevent division by zero.

    Args:
        box1 (array-like): First bounding box in xyxy format [x1, y1, x2, y2].
            Coordinates represent top-left (x1, y1) and bottom-right (x2, y2) corners.
        box2 (array-like): Second bounding box in xyxy format [x1, y1, x2, y2].
            Coordinates represent top-left (x1, y1) and bottom-right (x2, y2) corners.

    Returns:
        float: IoU value between 0.0 and 1.0, where:
            - 0.0 indicates no overlap between boxes
            - 1.0 indicates perfect overlap (identical boxes)

    Example:
        ```python
        box1 = [0, 0, 10, 10]  # 10x10 box at origin
        box2 = [5, 5, 15, 15]  # 10x10 box with 25% overlap
        iou = compute_iou(box1, box2)  # Returns ~0.14

        identical_boxes = [0, 0, 10, 10]
        iou_perfect = compute_iou(identical_boxes, identical_boxes)  # Returns 1.0

        no_overlap = [0, 0, 5, 5]
        far_box = [10, 10, 15, 15]
        iou_none = compute_iou(no_overlap, far_box)  # Returns 0.0
        ```
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-7)

    return iou


@keras.saving.register_keras_serializable(package="kvmm")
class NonMaxSuppression(Layer):
    """A Keras layer that performs Non-Maximum Suppression on object detection predictions.

    This layer applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
    from object detection model predictions. It processes predictions in batches, converting
    boxes from XYWH to XYXY format, filtering by confidence threshold, and suppressing
    overlapping detections based on IoU threshold. The layer is optimized for Keras 3
    operations and supports vectorized batch processing.

    The NMS algorithm works by:
    1. Filtering predictions below the confidence threshold
    2. Sorting remaining detections by confidence score
    3. Iteratively selecting the highest-confidence detection
    4. Suppressing overlapping detections (IoU > threshold) with lower confidence
    5. Repeating until max_detections is reached or no more detections remain

    Args:
        conf_threshold (float): Minimum confidence score for detections to be considered.
            Detections with confidence below this threshold are filtered out.
            Default: 0.25
        iou_threshold (float): IoU threshold for suppression. Detections with IoU
            greater than this threshold with a higher-confidence detection are suppressed.
            Default: 0.7
        max_detections (int): Maximum number of detections to keep per image after NMS.
            Default: 300
        num_classes (int): Number of object classes in the prediction tensor.
            Default: 80
        **kwargs: Additional layer arguments.

    Input shape:
        3D tensor with shape: `(batch_size, num_predictions, 4 + num_classes)`
        where each prediction contains:
        - First 4 values: bounding box in XYWH format [center_x, center_y, width, height]
        - Next num_classes values: class confidence scores

    Output shape:
        List of 2D tensors, one per batch item, each with shape: `(num_detections, 6)`
        where each detection contains: [x1, y1, x2, y2, confidence, class_id]
        - x1, y1: top-left corner coordinates
        - x2, y2: bottom-right corner coordinates
        - confidence: detection confidence score
        - class_id: predicted class index (as float)

    Example:
        ```python
        # Initialize NMS layer
        nms_layer = NonMaxSuppression(
            conf_threshold=0.5,
            iou_threshold=0.6,
            max_detections=100,
            num_classes=80
        )

        # Input predictions: batch_size=2, 8400 predictions, 84 values each (4 + 80 classes)
        predictions = tf.random.normal((2, 8400, 84))

        # Apply NMS
        detections = nms_layer(predictions)
        # Returns list of 2 tensors, each with shape (num_kept_detections, 6)

        # Access detections for first image
        img_detections = detections[0]  # Shape: (N, 6) where N <= 100
        boxes = img_detections[:, :4]   # Bounding boxes in XYXY format
        scores = img_detections[:, 4]   # Confidence scores
        classes = img_detections[:, 5]  # Class predictions
        ```

    Note:
        This implementation uses NumPy operations within the NMS core algorithm for
        efficiency, while maintaining Keras tensor compatibility for inputs and outputs.
        Empty results (no detections) are returned as zero-shaped tensors.
    """

    def __init__(
        self,
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=300,
        num_classes=80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.num_classes = num_classes

    def call(self, predictions):
        batch_size = ops.shape(predictions)[0]
        results = []

        for batch_idx in range(batch_size):
            img_pred = predictions[batch_idx]
            boxes_xywh = img_pred[:, :4]
            class_scores = img_pred[:, 4 : 4 + self.num_classes]

            xy = boxes_xywh[:, :2]
            wh = boxes_xywh[:, 2:4] / 2.0
            x1y1 = xy - wh
            x2y2 = xy + wh
            boxes_xyxy = ops.concatenate([x1y1, x2y2], axis=-1)

            max_scores = ops.max(class_scores, axis=-1)
            best_classes = ops.argmax(class_scores, axis=-1)

            conf_mask = max_scores > self.conf_threshold

            if not ops.any(conf_mask):
                results.append(ops.zeros((0, 6), dtype="float32"))
                continue

            if not ops.any(conf_mask):
                results.append(ops.zeros((0, 6), dtype="float32"))
                continue

            valid_boxes = boxes_xyxy[conf_mask]
            valid_scores = max_scores[conf_mask]
            valid_classes = best_classes[conf_mask]

            keep_indices = self._perform_nms(valid_boxes, valid_scores)

            if ops.shape(keep_indices)[0] == 0:
                results.append(ops.zeros((0, 6), dtype="float32"))
                continue

            final_boxes = ops.take(valid_boxes, keep_indices, axis=0)
            final_scores = ops.take(valid_scores, keep_indices, axis=0)
            final_classes = ops.take(valid_classes, keep_indices, axis=0)

            detections = ops.concatenate(
                [
                    final_boxes,
                    ops.expand_dims(final_scores, axis=-1),
                    ops.expand_dims(ops.cast(final_classes, "float32"), axis=-1),
                ],
                axis=-1,
            )

            results.append(detections)

        return results

    def _perform_nms(self, boxes, scores):
        num_boxes = ops.shape(boxes)[0]

        if num_boxes == 0:
            return ops.zeros((0,), dtype="int32")

        boxes_np = ops.convert_to_numpy(boxes)
        scores_np = ops.convert_to_numpy(scores)

        sorted_indices = np.argsort(-scores_np)

        keep_indices = []
        suppressed = np.zeros(num_boxes, dtype=bool)

        for i in range(min(num_boxes, self.max_detections * 2)):
            idx = sorted_indices[i]

            if suppressed[idx]:
                continue

            keep_indices.append(idx)

            if len(keep_indices) >= self.max_detections:
                break

            current_box = boxes_np[idx]

            for j in range(i + 1, num_boxes):
                other_idx = sorted_indices[j]

                if suppressed[other_idx]:
                    continue

                other_box = boxes_np[other_idx]
                iou = compute_iou(current_box, other_box)

                if iou > self.iou_threshold:
                    suppressed[other_idx] = True

        if len(keep_indices) == 0:
            return ops.zeros((0,), dtype="int32")

        return ops.convert_to_tensor(np.array(keep_indices, dtype=np.int32))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_detections": self.max_detections,
                "num_classes": self.num_classes,
            }
        )
        return config
