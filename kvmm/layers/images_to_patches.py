import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class ImageToPatchesLayer(layers.Layer):
    """A Keras layer that converts images into patches.

    This layer takes a batch of images and converts each image into a sequence of patches.
    The patches are created by dividing the image into non-overlapping squares of size
    `patch_size` x `patch_size`. If the image dimensions are not perfectly divisible by
    the patch size, the image is resized to the nearest larger size that is divisible.

    Args:
        patch_size (int): The size of each square patch (both height and width).
        **kwargs: Additional layer arguments.

    Input shape:
        4D tensor with shape:
        - If data_format='channels_last': `(batch_size, height, width, channels)`
        - If data_format='channels_first': `(batch_size, channels, height, width)`

    Output shape:
        4D tensor with shape:
        - If data_format='channels_last': `(batch_size, patch_size*patch_size, num_patches, channels)`
        - If data_format='channels_first': `(batch_size, channels, patch_size*patch_size, num_patches)`
        where num_patches = ceil(height/patch_size) * ceil(width/patch_size)

    Example:
        ```python
        layer = ImageToPatchesLayer(patch_size=16)
        image = tf.random.normal((1, 224, 224, 3))  # Single RGB image
        patches = layer(image)  # Shape: (1, 256, 196, 3)
        ```
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.resize = False
        self.data_format = keras.config.image_data_format()

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch_size, height, width, channels) or "
                f"(batch_size, channels, height, width), got {len(input_shape)}"
            )
        super().build(input_shape)

    def call(self, inputs):
        x = inputs

        if self.data_format == "channels_last":
            h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]
        else:
            c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]

        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        num_patches_h = new_h // self.patch_size
        num_patches_w = new_w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        self.resize = False
        if new_h != h or new_w != w:
            x = ops.image.resize(x, size=(new_h, new_w), data_format=self.data_format)
            self.resize = True

        if self.data_format == "channels_last":
            x = ops.reshape(
                x,
                [-1, num_patches_h, self.patch_size, num_patches_w, self.patch_size, c],
            )
            x = ops.transpose(x, [0, 2, 4, 1, 3, 5])
            x = ops.reshape(x, [-1, self.patch_size * self.patch_size, num_patches, c])
        else:
            x = ops.reshape(
                x,
                [-1, c, num_patches_h, self.patch_size, num_patches_w, self.patch_size],
            )
            x = ops.transpose(x, [0, 1, 3, 5, 2, 4])
            x = ops.reshape(x, [-1, c, self.patch_size * self.patch_size, num_patches])

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
            }
        )
        return config
