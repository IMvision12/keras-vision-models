import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class PatchesToImageLayer(layers.Layer):
    """Folds patches back into their original feature map shape.

    This layer reconstructs an image or feature map from its patch representation by
    combining patches into their original spatial arrangement. It automatically handles
    both 'channels_first' and 'channels_last' data formats, and can handle resizing
    to original dimensions when needed.

    The transformation follows these steps:
    1. Determines output dimensions (from original_size or patch arrangement)
    2. Reshapes patches into their spatial arrangement
    3. Resizes to original dimensions if requested
    4. Converts to channels_first format if necessary

    Args:
        patch_size (int): The size of each square patch (both height and width)
        **kwargs: Additional layer arguments inherited from keras.layers.Layer

    Input shape:
        4D tensor with shape:
        `(batch_size, patch_size * patch_size, num_patches, channels)`
        where num_patches should be a perfect square

    Output shape:
        4D tensor with shape:
        - If channels_last: `(batch_size, height, width, channels)`
        - If channels_first: `(batch_size, channels, height, width)`
        where height and width are either:
        - Specified by original_size parameter
        - Computed as sqrt(num_patches) * patch_size

    Examples:
        ```python
        # Create layer with 8x8 patches
        layer = PatchesToImageLayer(patch_size=8)

        # Process patches back to image
        patches_shape = (1, 64, 16, 3)  # From 32x32 image
        output = layer(tf.random.normal(patches_shape),
                      original_size=(32, 32))
        # output shape: (1, 32, 32, 3)  # channels_last
        ```

    Notes:
        - If original_size is not provided, output dimensions are inferred from
          the number of patches, assuming they form a square arrangement
        - The layer can handle non-square original images when original_size
          is provided
        - Resizing is optional and controlled by the resize parameter
        - The layer maintains the original channel order and batch dimension
        - The layer is serializable and can be saved as part of a Keras model

    Attributes:
        patch_size (int): Size of each square patch
        data_format (str): The output data format ('channels_first' or 'channels_last')
        h (int): Output height (set during build or call)
        w (int): Output width (set during build or call)
        c (int): Number of channels (set during build)
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
