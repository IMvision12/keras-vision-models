import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class PatchesToImageLayer(layers.Layer):
    """Layer that folds patches back into a feature map.

    This layer transforms patches back into the original feature map shape,
    handling both 'channels_first' and 'channels_last' formats automatically.

    For 'channels_last': [B, P, N, C] -> [B, H, W, C]
    For 'channels_first': [B, P, N, C] -> [B, C, H, W]
    where:
    - B is the batch size
    - P is patch_size * patch_size
    - N is the number of patches
    - C is the number of channels
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.data_format = keras.config.image_data_format()

    def build(self, input_shape):
        self.h = None
        self.w = None
        self.c = input_shape[-1]
        super().build(input_shape)

    def call(self, inputs, original_size=None, resize=False):
        x = inputs

        if original_size is not None:
            self.h, self.w = original_size

        if self.h is None or self.w is None:
            num_patches = inputs.shape[2]
            patch_area = self.patch_size * self.patch_size
            side_patches = int(math.sqrt(num_patches))
            self.h = self.w = side_patches * self.patch_size

        new_h = math.ceil(self.h / self.patch_size) * self.patch_size
        new_w = math.ceil(self.w / self.patch_size) * self.patch_size
        num_patches_h = new_h // self.patch_size
        num_patches_w = new_w // self.patch_size

        x = ops.reshape(
            x,
            [
                -1,
                self.patch_size,
                self.patch_size,
                num_patches_h,
                num_patches_w,
                self.c,
            ],
        )
        x = ops.transpose(x, [0, 3, 1, 4, 2, 5])
        x = ops.reshape(x, [-1, new_h, new_w, self.c])

        if resize:
            x = ops.image.resize(x, size=(self.h, self.w), data_format="channels_last")

        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 3, 1, 2])

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
            }
        )
        return config
