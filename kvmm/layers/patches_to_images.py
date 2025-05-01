import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class PatchesToImageLayer(layers.Layer):
    """A Keras layer that reconstructs images from patches.

    This layer takes a sequence of image patches and reconstructs the original image by
    placing the patches back in their original positions. It can handle both cases where
    the original image dimensions are known or unknown, and can optionally resize the
    output to match the original image dimensions.

    Args:
        patch_size (int): The size of each square patch (both height and width).
        **kwargs: Additional layer arguments.

    Input shape:
        4D tensor with shape:
        - If data_format='channels_last': `(batch_size, patch_size*patch_size, num_patches, channels)`
        - If data_format='channels_first': `(batch_size, channels, patch_size*patch_size, num_patches)`

    Output shape:
        4D tensor with shape:
        - If data_format='channels_last': `(batch_size, height, width, channels)`
        - If data_format='channels_first': `(batch_size, channels, height, width)`
        where height and width are either determined from the original_size parameter,
        or calculated as sqrt(num_patches) * patch_size.

    Example:
        ```python
        layer = PatchesToImageLayer(patch_size=16)
        patches = tf.random.normal((1, 256, 196, 3))  # 196 patches of size 16x16
        image = layer(patches)  # Shape: (1, 224, 224, 3)
        # Or with original size specified:
        image = layer(patches, original_size=(220, 220), resize=True)
        ```

    Args:
        inputs: Input tensor of patches.
        original_size (tuple, optional): Tuple of (height, width) of the original image.
            If not provided, assumes the image is square with dimensions determined by
            the number of patches.
        resize (bool, optional): If True and original_size is provided, resizes the
            output to match the original dimensions. Defaults to False.
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.data_format = keras.config.image_data_format()

    def build(self, input_shape):
        self.h = None
        self.w = None
        self.c = (
            input_shape[-1] if self.data_format == "channels_last" else input_shape[1]
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        c = input_shape[-1] if self.data_format == "channels_last" else input_shape[1]
        num_patches = (
            input_shape[2] if self.data_format == "channels_last" else input_shape[-1]
        )
        side_patches = int(num_patches**0.5)
        h = w = side_patches * self.patch_size

        if self.data_format == "channels_last":
            return input_shape[0], h, w, c
        else:
            return input_shape[0], c, h, w

    def compute_output_spec(self, inputs, original_size=None, resize=False):
        input_spec = keras.KerasTensor(inputs.shape, dtype=inputs.dtype)
        batch_size = input_spec.shape[0]
        c = (
            input_spec.shape[-1]
            if self.data_format == "channels_last"
            else input_spec.shape[1]
        )

        if original_size is None:
            num_patches = (
                input_spec.shape[2]
                if self.data_format == "channels_last"
                else input_spec.shape[-1]
            )
            side_patches = int(num_patches**0.5)
            h = w = side_patches * self.patch_size
        else:
            h, w = original_size

            h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
            w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size

            if resize:
                h, w = original_size

        if self.data_format == "channels_last":
            return keras.KerasTensor((batch_size, h, w, c), dtype=inputs.dtype)
        else:
            return keras.KerasTensor((batch_size, c, h, w), dtype=inputs.dtype)

    def call(self, inputs, original_size=None, resize=False):
        x = inputs

        if original_size is not None:
            self.h, self.w = original_size

        if self.h is None or self.w is None:
            num_patches = (
                inputs.shape[2]
                if self.data_format == "channels_last"
                else inputs.shape[-1]
            )
            side_patches = int(num_patches**0.5)
            self.h = self.w = side_patches * self.patch_size

        new_h = ((self.h + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_w = ((self.w + self.patch_size - 1) // self.patch_size) * self.patch_size
        num_patches_h = new_h // self.patch_size
        num_patches_w = new_w // self.patch_size

        if self.data_format == "channels_last":
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
        else:
            x = ops.reshape(
                x,
                [
                    -1,
                    self.c,
                    self.patch_size,
                    self.patch_size,
                    num_patches_h,
                    num_patches_w,
                ],
            )
            x = ops.transpose(x, [0, 1, 4, 2, 5, 3])
            x = ops.reshape(x, [-1, self.c, new_h, new_w])

        if resize:
            x = ops.image.resize(x, size=(self.h, self.w), data_format=self.data_format)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
            }
        )
        return config
