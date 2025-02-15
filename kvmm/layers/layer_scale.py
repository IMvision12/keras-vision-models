import keras
from keras import initializers, layers


@keras.saving.register_keras_serializable(package="kvmm")
class LayerScale(layers.Layer):
    """
    Implements LayerScale, a learnable scaling layer that multiplies the input by a
    trainable scale factor. It is often used in modern architectures to add stability
    to the training process by scaling the output of certain layers.

    Args:
        init_values (float): Initial value for the scaling factor `gamma`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Methods:
        build(input_shape):
            Creates the trainable scaling factor `gamma`, initialized to the `init_values`
            and with the shape automatically determined from the input shape.
        call(x):
            Multiplies the input `x` by the scaling factor `gamma`.
        get_config():
            Returns a dictionary containing the configuration of the layer.

    Example:
        >>> layer = LayerScale(init_values=0.1)
        >>> output = layer(input_tensor)
    """

    def __init__(self, init_values, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
            }
        )
        return config
