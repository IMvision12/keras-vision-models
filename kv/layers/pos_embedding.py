
from keras import layers


class AddPositionEmbs(layers.Layer):
    """
    Implements learnable position embeddings that are added to the input sequence.
    These embeddings provide positional information to the otherwise position-invariant
    self-attention mechanism in transformer architectures.

    Args:
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Methods:
        build(input_shape):
            Creates the learnable position embeddings with shape matching the input
            sequence length and hidden dimension. Expects input shape of
            (batch_size, sequence_length, hidden_dim).
        call(inputs):
            Adds the position embeddings to the input tensor.
        get_config():
            Returns the layer configuration.

    Raises:
        ValueError: If the input tensor does not have exactly 3 dimensions
            (batch_size, sequence_length, hidden_dim).

    Example:
        >>> layer = AddPositionEmbs()
        >>> sequence_length = 196
        >>> hidden_dim = 768
        >>> x = tf.random.normal((batch_size, sequence_length, hidden_dim))
        >>> output = layer(x)  # Shape matches input: (batch_size, sequence_length, hidden_dim)
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"Number of dimensions should be 3, got {len(input_shape)}"
            )
        self.pe = self.add_weight(
            name="pos_embed",
            shape=(1, input_shape[1], input_shape[2]),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.pe

    def get_config(self):
        config = super().get_config()
        return config
