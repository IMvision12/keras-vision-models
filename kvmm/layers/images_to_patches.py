import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class ImageToPatchesLayer(layers.Layer):
    """Layer that unfolds a feature map into patches.

    This layer transforms an input tensor into patches, handling both
    'channels_first' and 'channels_last' formats automatically.

    For 'channels_last': [B, H, W, C] -> [B, P, N, C]
    For 'channels_first': [B, C, H, W] -> [B, P, N, C]
    where:
    - B is the batch size
    - P is patch_size * patch_size
    - N is the number of patches
    - C is the number of channels
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.resize = False
        self.data_format = keras.config.image_data_format()

    def call(self, inputs):
        x = inputs

        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 2, 3, 1])

        h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]

        new_h = math.ceil(h / self.patch_size) * self.patch_size
        new_w = math.ceil(w / self.patch_size) * self.patch_size
        num_patches_h = new_h // self.patch_size
        num_patches_w = new_w // self.patch_size

        self.resize = False
        if new_h != h or new_w != w:
            x = ops.image.resize(x, size=(new_h, new_w), data_format="channels_last")
            self.resize = True

        x = ops.reshape(
            x, [-1, num_patches_h, self.patch_size, num_patches_w, self.patch_size, c]
        )
        x = ops.transpose(x, [0, 2, 4, 1, 3, 5])
        x = ops.reshape(
            x, [-1, self.patch_size * self.patch_size, num_patches_h * num_patches_w, c]
        )

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
            }
        )
        return config
