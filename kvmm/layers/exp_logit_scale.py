import math

import keras
from keras import initializers, layers


@keras.saving.register_keras_serializable(package="kvmm")
class ExpLogitScale(layers.Layer):
    """
    A custom Keras layer that applies an exponential scaling factor to attention logits.

    This layer scales the input tensor (typically attention logits) by an exponential
    learned parameter. Each attention head has its own scaling parameter, initialized
    to `init_value` and capped at `max_value` to prevent numerical instability.

    Args:
        init_value (float): Initial value for the log scale parameter.
            Defaults to math.log(10.0), which means initial scaling is 10.0.
        max_value (float): Maximum value for the log scale parameter to prevent
            numerical overflow. Defaults to math.log(100.0), which caps scaling at 100.0.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        4D tensor with shape: `(batch_size, num_heads, height*width, height*width)`
        Typically represents attention logits in a multi-head attention mechanism.

    Output shape:
        Same as the input shape, with values scaled by exp(scale) where scale is
        the learned parameter per attention head.

    Example:
        ```python
        # Apply exponential scaling to attention logits
        attention_logits = tf.random.normal((1, 8, 64, 64))  # [batch, heads, tokens, tokens]
        scaling_layer = ExpLogitScale(init_value=math.log(5.0))
        scaled_logits = scaling_layer(attention_logits)
        ```
    """

    def __init__(self, init_value=math.log(10.0), max_value=math.log(100.0), **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.max_value = max_value

    def build(self, input_shape):
        num_heads = input_shape[1]  # For [batch, num_heads, hh*ww, hh*ww]

        self.scale = self.add_weight(
            name="gamma",
            shape=(num_heads,),
            initializer=initializers.constant(self.init_value),
            trainable=True,
        )
        self.__max_value__ = float(self.max_value)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        scale_reshape = keras.ops.reshape(self.scale, [1, inputs.shape[1], 1, 1])
        return inputs * keras.ops.exp(
            keras.ops.minimum(scale_reshape, self.__max_value__)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value, "max_value": self.max_value})
        return config
