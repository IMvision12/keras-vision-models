import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class ImageToPatchesLayer(layers.Layer):
    """Unfolds a feature map into patches of specified size.

    This layer transforms an input tensor into patches by dividing the input image
    into equal-sized squares and restructuring them into a sequence. It automatically
    handles both 'channels_first' and 'channels_last' data formats, and can resize
    input dimensions that aren't perfectly divisible by the patch size.

    The transformation follows these steps:
    1. Converts input to channels_last format if necessary
    2. Resizes input if dimensions aren't divisible by patch_size
    3. Reshapes the input into patches
    4. Reorganizes patches into the desired output format

    Args:
        patch_size (int): The size of each square patch (both height and width)
        **kwargs: Additional layer arguments inherited from keras.layers.Layer

    Input shape:
        4D tensor with shape:
        - If channels_last: `(batch_size, height, width, channels)`
        - If channels_first: `(batch_size, channels, height, width)`

    Output shape:
        4D tensor with shape:
        `(batch_size, patch_size * patch_size, num_patches, channels)`
        where num_patches = (ceil(height/patch_size) * ceil(width/patch_size))

    Examples:
        ```python
        # Create layer with 8x8 patches
        layer = ImageToPatchesLayer(patch_size=8)

        # Process a 32x32 RGB image
        input_shape = (1, 32, 32, 3)  # channels_last
        output = layer(tf.random.normal(input_shape))
        # output shape: (1, 64, 16, 3)
        ```

    Notes:
        - If input dimensions are not perfectly divisible by patch_size,
          the input will be resized to the nearest larger size that is divisible
        - The layer maintains the original channel order and batch dimension
        - The layer is serializable and can be saved as part of a Keras model

    Attributes:
        patch_size (int): Size of each square patch
        resize (bool): Indicates if the last input required resizing
        data_format (str): The input data format ('channels_first' or 'channels_last')
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
