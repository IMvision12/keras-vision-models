"""Binary Cross-Entropy Loss implementation."""

import keras
from keras import ops


class BCELoss(keras.losses.Loss):
    """Binary Cross-Entropy Loss with optional label smoothing.

    This is a wrapper around Keras BCE that provides consistent interface
    with other YOLO losses and adds label smoothing support.
    """

    def __init__(self, label_smoothing=0.0, pos_weight=1.0, name="bce_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

        self.bce = keras.losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=label_smoothing, reduction="none"
        )

    def call(self, y_true, y_pred):
        loss = self.bce(y_true, y_pred)

        if self.pos_weight != 1.0:
            pos_weight_tensor = ops.where(y_true > 0.5, self.pos_weight, 1.0)
            loss = loss * pos_weight_tensor

        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {"label_smoothing": self.label_smoothing, "pos_weight": self.pos_weight}
        )
        return config
