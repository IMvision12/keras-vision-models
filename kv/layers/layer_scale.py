from keras import initializers, layers


class LayerScale(layers.Layer):
    """
    Implements LayerScale, a learnable scaling layer that multiplies the input by a
    trainable scale factor. It is often used in modern architectures to add stability
    to the training process by scaling the output of certain layers.

    Args:
        init_values (float): Initial value for the scaling factor `gamma`.
        projection_dim (int): Dimensionality of the input projection.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Methods:
        build(_):
            Creates the trainable scaling factor `gamma`, initialized to the `init_values`
            and with the shape matching the projection dimension.

        call(x):
            Multiplies the input `x` by the scaling factor `gamma`.

        get_config():
            Returns a dictionary containing the configuration of the layer, including the
            `init_values` and `projection_dim`.

    Example:
        >>> layer = LayerScale(init_values=0.1, projection_dim=768)
        >>> output = layer(input_tensor)
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
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
                "projection_dim": self.projection_dim,
            }
        )
        return config
