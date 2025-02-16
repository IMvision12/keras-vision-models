import keras
from keras import layers, ops, random


@keras.saving.register_keras_serializable(package="kvmm")
class StochasticDepth(layers.Layer):
    """
    Implements the Stochastic Depth regularization layer, which randomly drops
    paths (connections) during training to prevent overfitting. This is typically
    used in deep networks to make them more robust and reduce overfitting.

    Args:
        drop_path_rate (float): Probability of dropping a path. Should be between 0 and 1.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Methods:
        call(x, training=None):
            Applies the stochastic depth mechanism during training by randomly dropping
            some of the connections with a probability defined by `drop_path_rate`. During
            inference, the input is returned unchanged.

        get_config():
            Returns a dictionary containing the configuration of the layer, including the
            `drop_path_rate`.

    Example:
        >>> layer = StochasticDepth(drop_path_rate=0.1)
        >>> output = layer(input_tensor, training=True)
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate
        if not 0 <= drop_path_rate <= 1:
            raise ValueError(
                f"drop_path_rate should be between 0 and 1, got {drop_path_rate}"
            )

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
            random_tensor = keep_prob + random.uniform(shape, 0, 1)
            random_tensor = ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config
