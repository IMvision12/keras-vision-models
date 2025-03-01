import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class WindowPartition(layers.Layer):
    """Layer for partitioning input tensor into non-overlapping windows."""

    def __init__(self, window_size, fused=False, num_heads=None, qkv_mult=3, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.fused = fused
        self.num_heads = num_heads
        self.qkv_mult = qkv_mult

        if self.fused and self.num_heads is None:
            raise ValueError("num_heads must be set when fused=True")

    def call(self, inputs, height=None, width=None):
        if inputs.shape.rank != 4:
            raise ValueError("Expecting inputs rank to be 4.")

        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        windows_height = height // self.window_size
        windows_width = width // self.window_size

        if not self.fused:
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
                    self.window_size,
                    windows_width,
                    self.window_size,
                    channels,
                ],
            )
            outputs = ops.transpose(outputs, [0, 1, 3, 2, 4, 5])
            outputs = ops.reshape(outputs, [-1, self.window_size**2, channels])

        else:
            full_channels = inputs.shape[-1]
            if full_channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            head_channels = full_channels // (self.qkv_mult * self.num_heads)

            outputs = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    self.window_size,
                    windows_width,
                    self.window_size,
                    self.qkv_mult,
                    self.num_heads,
                    head_channels,
                ],
            )
            outputs = ops.transpose(outputs, [5, 0, 1, 3, 6, 2, 4, 7])
            outputs = ops.reshape(
                outputs,
                [self.qkv_mult, -1, self.num_heads, self.window_size**2, head_channels],
            )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "fused": self.fused,
                "num_heads": self.num_heads,
                "qkv_mult": self.qkv_mult,
            }
        )
        return config
