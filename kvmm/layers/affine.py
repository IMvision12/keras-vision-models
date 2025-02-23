import keras
from keras import layers


@keras.saving.register_keras_serializable(package="kvmm")
class Affine(layers.Layer):
    """Affine transformation layer that applies learnable scale and shift parameters to input tensors.

    This layer performs an affine transformation on the input tensor of the form:
    output = alpha * input + beta
    where alpha (scale) and beta (shift) are learnable parameters broadcast to match
    the input shape.

    The weight shapes are designed to match the PyTorch implementation for compatibility
    and easier model conversion between frameworks.

    Args:
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        N-D tensor with shape: (batch_size, ..., channels)
        The last axis should contain the feature channels

    Output shape:
        Same shape as input

    Attributes:
        dim (int): Number of input features/channels, inferred from input shape
        alpha (Weight): Learnable scale parameter with shape (1, 1, channels)
        beta (Weight): Learnable shift parameter with shape (1, 1, channels)

    Example:
        ```python
        # Apply affine transformation to 16-dim feature vectors
        x = tf.random.normal((32, 10, 16))  # (batch_size, seq_len, channels)
        affine = Affine()
        output = affine(x)  # shape = (32, 10, 16)
        ```
    """

    def __init__(self, dim=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.alpha = None
        self.beta = None

    def build(self, input_shape):
        if self.dim is None:
            self.dim = input_shape[-1]

        self.broadcast_shape = (1,) * (len(input_shape) - 1) + (self.dim,)

        self.alpha = self.add_weight(
            shape=self.broadcast_shape, initializer="ones", trainable=True, name="alpha"
        )
        self.beta = self.add_weight(
            shape=self.broadcast_shape, initializer="zeros", trainable=True, name="beta"
        )

    def call(self, x):
        return self.alpha * x + self.beta

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
            }
        )
        return config
