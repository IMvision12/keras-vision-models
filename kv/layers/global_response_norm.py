import keras
from keras import layers, ops


class GlobalResponseNorm(layers.Layer):
    """
    GRN (Global Response Normalization) layer.

    This layer applies a global normalization to the input tensor, using the square root of the
    sum of squares of the input along the spatial dimensions (height and width) as a global feature.
    The normalized global feature is then used to scale and shift the input tensor.

    Parameters:
    dim (int): The number of channels (depth) of the input tensor.

    Attributes:
    weight (np.ndarray): The weight tensor used for scaling the normalized global feature.
    bias (np.ndarray): The bias tensor used for shifting the input tensor.
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape=None):
        self.weight = self.add_weight(
            name="weight",
            shape=(1, 1, 1, self.dim),
            initializer=keras.initializers.Zeros(),
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(1, 1, 1, self.dim),
            initializer=keras.initializers.Zeros(),
        )
        return super().build(input_shape)

    def call(self, hidden_states):
        global_features = ops.sqrt(
            ops.sum(ops.square(hidden_states), axis=(1, 2), keepdims=True)
        )
        norm_features = global_features / (
            ops.mean(global_features, axis=-1, keepdims=True) + 1e-6
        )
        hidden_states = (
            self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        )
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
