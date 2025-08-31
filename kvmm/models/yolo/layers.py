import keras
from keras import ops


class DFL(keras.layers.Layer):
    """
    Distribution Focal Loss (DFL) layer for object detection.

    DFL is a technique used in modern object detection models (like YOLOv8) to convert
    distribution predictions into precise localization values. Instead of directly predicting
    bounding box coordinates, the model predicts a probability distribution over a range
    of possible values, and DFL computes the expected value of this distribution.

    This approach helps improve localization accuracy by allowing the model to express
    uncertainty in its predictions and learn more nuanced representations of object boundaries.

    The layer takes distribution predictions and converts them to scalar values by:
    1. Reshaping input to separate the 4 bounding box coordinates
    2. Applying softmax to get probability distributions
    3. Computing weighted sum (expected value) using learned weights

    Attributes:
        c1 (int): Number of channels in the distribution (default: 16).
                 This represents how many discrete values the distribution spans.
        conv_weight (Tensor): Learnable weight tensor of shape (c1,) containing
                             values [0, 1, 2, ..., c1-1] used as bin centers
                             for computing the expected value.

    Args:
        c1 (int, optional): Number of channels/bins in the distribution. Default is 16.
                           Higher values allow for more fine-grained distributions
                           but increase computational cost.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input Shape:
        (batch_size, 4*c1, num_anchors): Where 4*c1 represents distributions for
        4 bounding box coordinates (left, top, right, bottom), each with c1 bins.

    Output Shape:
        (batch_size, 4, num_anchors): Converted scalar values for the 4 bounding
        box coordinate predictions.

    Example:
        >>> # Create DFL layer with 16 distribution bins
        >>> dfl = DFL(c1=16)
        >>>
        >>> # Input: batch of distribution predictions
        >>> # Shape: (batch=2, channels=64, anchors=8400) where 64 = 4*16
        >>> x = keras.random.normal((2, 64, 8400))
        >>>
        >>> # Forward pass
        >>> output = dfl(x)
        >>> print(output.shape)  # (2, 4, 8400)
        >>>
        >>> # Each output represents expected values of the distributions
        >>> # output[:, 0, :] = left distances
        >>> # output[:, 1, :] = top distances
        >>> # output[:, 2, :] = right distances
        >>> # output[:, 3, :] = bottom distances

    Note:
        This layer is typically used in the head of object detection models
        after the backbone and neck have produced feature maps. The distributions
        are usually learned to span a range like [0, c1-1], representing possible
        distance values in the detection task.
    """

    def __init__(self, c1=16, **kwargs):
        super().__init__(**kwargs)
        self.c1 = c1

    def build(self, input_shape):
        super().build(input_shape)
        self.conv_weight = ops.arange(self.c1, dtype="float32")

    def call(self, x):
        b, _, a = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        x_reshaped = ops.reshape(x, (b, 4, self.c1, a))
        x_transposed = ops.transpose(x_reshaped, (0, 3, 1, 2))
        x_softmax = ops.softmax(x_transposed, axis=3)
        result = ops.sum(x_softmax * self.conv_weight[None, None, None, :], axis=3)
        result = ops.transpose(result, (0, 2, 1))
        return result
