"""Distribution Focal Loss (DFL) implementation."""

import keras
from keras import ops


class DFLLoss(keras.losses.Loss):
    """Distribution Focal Loss (DFL) for bounding box regression.

    DFL treats bounding box regression as a classification problem over
    a discrete set of possible values.
    Paper: https://arxiv.org/abs/2006.04388
    """

    def __init__(self, reg_max=16, name="dfl_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.reg_max = reg_max

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth distribution [batch_size, max_boxes, 4, reg_max+1]
            y_pred: Predicted distribution [batch_size, max_boxes, 4, reg_max+1]

        Returns:
            DFL loss value
        """
        # Apply softmax to predictions to get probability distribution
        y_pred_softmax = ops.softmax(y_pred, axis=-1)

        # Calculate cross-entropy loss
        # Clip predictions to avoid log(0)
        y_pred_softmax = ops.clip(y_pred_softmax, 1e-7, 1.0)

        # Cross-entropy: -sum(y_true * log(y_pred))
        loss = -ops.sum(y_true * ops.log(y_pred_softmax), axis=-1)

        # Average over the 4 coordinates (left, top, right, bottom)
        loss = ops.mean(loss, axis=-1)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({"reg_max": self.reg_max})
        return config
