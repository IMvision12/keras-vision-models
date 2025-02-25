import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class BiasLayer(layers.Layer):
    """
    A standalone Keras 3 layer that adds bias terms to inputs.
    Supports both channels-last and channels-first data formats.

    This layer adds a bias vector to each position of the input. The bias vector
    length matches the number of channels in the input. For channels_first data format,
    bias is added to dimension 1. For channels_last data format, bias is added
    to the last dimension.

    Arguments:
        trainable: Boolean, whether the bias weights are trainable.
        initializer: Initializer for the bias vector.
        data_format: String, either 'channels_first' or 'channels_last'.
            Defaults to the image data format set in Keras config.

    Input shape:
        N-D tensor with shape: (batch_size, ...) if data_format='channels_last'
        or (batch_size, channels, ...) if data_format='channels_first'.

    Output shape:
        Same as input shape.

    Example:
        ```python
        # Add trainable bias to a Conv2D layer output
        x = layers.Conv2D(32, 3, use_bias=False)(inputs)
        x = BiasLayer()(x)
        ```
    """

    def __init__(self, trainable=True, initializer="zeros", data_format=None, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        self.trainable_bias = trainable
        self.initializer = keras.initializers.get(initializer)
        self.data_format = data_format or keras.backend.image_data_format()

    def build(self, input_shape):
        if self.data_format == "channels_first":
            bias_shape = (input_shape[1],)
        else:
            bias_shape = (input_shape[-1],)

        self.bias = self.add_weight(
            shape=bias_shape,
            initializer=self.initializer,
            trainable=self.trainable_bias,
            name="bias",
        )
        super(BiasLayer, self).build(input_shape)

    def call(self, inputs):
        if self.data_format == "channels_first":
            ndim = len(inputs.shape)
            broadcast_shape = [1, inputs.shape[1]] + [1] * (ndim - 2)
            reshaped_bias = ops.reshape(self.bias, broadcast_shape)
        else:
            reshaped_bias = self.bias

        return inputs + reshaped_bias

    def get_config(self):
        config = super(BiasLayer, self).get_config()
        config.update(
            {
                "trainable": self.trainable_bias,
                "initializer": keras.initializers.serialize(self.initializer),
                "data_format": self.data_format,
            }
        )
        return config
