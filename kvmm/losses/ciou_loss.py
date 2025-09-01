"""Complete Intersection over Union (CIoU) Loss implementation."""

import keras
import numpy as np
from keras import ops


class CIoULoss(keras.losses.Loss):
    """Complete Intersection over Union (CIoU) Loss.

    CIoU loss considers overlap area, central point distance, and aspect ratio.
    Paper: https://arxiv.org/abs/1911.08287
    """

    def __init__(self, eps=1e-7, name="ciou_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth boxes [batch_size, max_boxes, 4] in format [x1, y1, x2, y2]
            y_pred: Predicted boxes [batch_size, max_boxes, 4] in format [x1, y1, x2, y2]

        Returns:
            CIoU loss value
        """
        # Calculate intersection
        inter_x1 = ops.maximum(y_true[..., 0], y_pred[..., 0])
        inter_y1 = ops.maximum(y_true[..., 1], y_pred[..., 1])
        inter_x2 = ops.minimum(y_true[..., 2], y_pred[..., 2])
        inter_y2 = ops.minimum(y_true[..., 3], y_pred[..., 3])

        inter_w = ops.maximum(0.0, inter_x2 - inter_x1)
        inter_h = ops.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Calculate union
        true_area = (y_true[..., 2] - y_true[..., 0]) * (
            y_true[..., 3] - y_true[..., 1]
        )
        pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (
            y_pred[..., 3] - y_pred[..., 1]
        )
        union_area = true_area + pred_area - inter_area

        # IoU
        iou = inter_area / (union_area + self.eps)

        # Calculate center points
        true_cx = (y_true[..., 0] + y_true[..., 2]) / 2
        true_cy = (y_true[..., 1] + y_true[..., 3]) / 2
        pred_cx = (y_pred[..., 0] + y_pred[..., 2]) / 2
        pred_cy = (y_pred[..., 1] + y_pred[..., 3]) / 2

        # Distance between centers
        center_distance_sq = (true_cx - pred_cx) ** 2 + (true_cy - pred_cy) ** 2

        # Diagonal distance of enclosing box
        enclose_x1 = ops.minimum(y_true[..., 0], y_pred[..., 0])
        enclose_y1 = ops.minimum(y_true[..., 1], y_pred[..., 1])
        enclose_x2 = ops.maximum(y_true[..., 2], y_pred[..., 2])
        enclose_y2 = ops.maximum(y_true[..., 3], y_pred[..., 3])

        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        diagonal_sq = enclose_w**2 + enclose_h**2 + self.eps

        # Aspect ratio penalty
        true_w = y_true[..., 2] - y_true[..., 0]
        true_h = y_true[..., 3] - y_true[..., 1]
        pred_w = y_pred[..., 2] - y_pred[..., 0]
        pred_h = y_pred[..., 3] - y_pred[..., 1]

        v = (4 / (np.pi**2)) * ops.square(
            ops.arctan(true_w / (true_h + self.eps))
            - ops.arctan(pred_w / (pred_h + self.eps))
        )

        alpha = v / (1 - iou + v + self.eps)

        # CIoU loss
        ciou = iou - center_distance_sq / diagonal_sq - alpha * v
        loss = 1 - ciou

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config
