from keras import layers, ops


class ClassToken(layers.Layer):
    """
    Implements a learnable class token that is prepended to the input sequence.
    This is commonly used in Vision Transformer (ViT) architectures where a special
    token is added to aggregate sequence information for classification tasks.

    Args:
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Methods:
        build(input_shape):
            Creates the learnable class token with shape matching the hidden dimension
            of the input. The hidden dimension is inferred from the last dimension
            of the input shape.
        call(inputs):
            Broadcasts the class token to match the batch size and prepends it to
            the input sequence.
        get_config():
            Returns the layer configuration.

    Example:
        >>> layer = ClassToken()
        >>> sequence_length = 196  # For 14x14 patches from 224x224 image
        >>> hidden_dim = 768
        >>> x = tf.random.normal((batch_size, sequence_length, hidden_dim))
        >>> output = layer(x)  # Shape: (batch_size, sequence_length + 1, hidden_dim)
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        cls_broadcasted = ops.broadcast_to(self.cls, [batch_size, 1, self.hidden_size])
        return ops.concatenate([cls_broadcasted, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        return config
