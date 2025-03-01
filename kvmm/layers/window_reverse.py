import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class WindowReverse(layers.Layer):
    """Layer for reverting windows back to the original feature map."""

    def __init__(self, window_size, fused=False, num_heads=None, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.fused = fused
        self.num_heads = num_heads

        if self.fused and self.num_heads is None:
            raise ValueError("num_heads must be set when fused=True")

    def call(self, inputs, height=None, width=None):
        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        windows_height = height // self.window_size
        windows_width = width // self.window_size

        if not self.fused:
            if inputs.shape.rank != 3:
                raise ValueError("Expecting inputs rank to be 3.")

            channels = inputs.shape[-1]
            if channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            outputs = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    windows_width,
                    self.window_size,
                    self.window_size,
                    channels,
                ],
            )
            outputs = ops.transpose(outputs, [0, 1, 3, 2, 4, 5])
            outputs = ops.reshape(outputs, [-1, height, width, channels])

        else:
            if inputs.shape.rank != 4:
                raise ValueError("Expecting inputs rank to be 4.")

            head_channels = inputs.shape[-1]
            if head_channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            full_channels = head_channels * self.num_heads

            outputs = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    windows_width,
                    self.num_heads,
                    self.window_size,
                    self.window_size,
                    head_channels,
                ],
            )
            outputs = ops.transpose(outputs, [0, 1, 4, 2, 5, 3, 6])
            outputs = ops.reshape(outputs, [-1, height, width, full_channels])

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "fused": self.fused,
                "num_heads": self.num_heads,
            }
        )
        return config
