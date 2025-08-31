import keras
from keras import ops


class Letterbox(keras.layers.Layer):
    """Letterbox layer for resizing and padding images while maintaining aspect ratio.

    This layer implements image resizing with letterboxing, a technique commonly used in
    computer vision tasks where the aspect ratio of the original image needs to be preserved.
    The image is resized to fit within the target dimensions and padded with a specified
    color to reach the exact target size.

    Key Features:
        - Preserves aspect ratio during resizing
        - Configurable padding color
        - Optional automatic padding adjustment based on stride
        - Support for both single images and batches
        - Returns resize ratios and padding information for downstream tasks
        - Flexible scaling options with scaleup and scaleFill modes

    Args:
        new_shape (tuple or int, optional): Target size as (height, width) or single int
            for square output. Defaults to (640, 640)
        color (tuple, optional): RGB values for padding color, each in range [0, 255].
            Defaults to (114, 114, 114)
        auto (bool, optional): If True, adjusts padding to be divisible by stride.
            Helps maintain compatibility with certain network architectures.
            Defaults to True
        scaleFill (bool, optional): If True, stretches image to new_shape instead
            of padding, ignoring aspect ratio. Defaults to False
        scaleup (bool, optional): If True, allows scaling up the image. If False,
            only scales down. Defaults to True
        stride (int, optional): Stride value for automatic padding adjustment when
            auto=True. Defaults to 32

    Input shape:
        - Single image: 3D tensor (height, width, channels)
        - Batch of images: 4D tensor (batch_size, height, width, channels)

    Output shape:
        Tuple of:
        - Resized and padded images: Same rank as input with dimensions matching new_shape
        - Scale ratios: (batch_size, 2) tensor of [width_ratio, height_ratio]
        - Padding values: (batch_size, 2) tensor of [width_padding, height_padding]

    Notes:
        - The layer automatically handles both single images and batches
        - Padding is applied symmetrically on all sides
        - Scale ratios and padding information are useful for converting
          predictions back to original image coordinates
        - RGB images are expected as input, with pixel values in [0, 1]
    """

    def __init__(
        self,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        stride=32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        self.new_shape = new_shape
        self.color = color
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

        self.color_norm = ops.convert_to_tensor(
            [c / 255.0 for c in color], dtype="float32"
        )

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype="float32")
        if len(inputs.shape) == 3:
            inputs = ops.expand_dims(inputs, axis=0)
            was_single_image = True
        else:
            was_single_image = False

        batch_size = ops.shape(inputs)[0]
        current_h = ops.shape(inputs)[1]
        current_w = ops.shape(inputs)[2]

        target_h, target_w = self.new_shape[0], self.new_shape[1]

        r_h = ops.cast(target_h, "float32") / ops.cast(current_h, "float32")
        r_w = ops.cast(target_w, "float32") / ops.cast(current_w, "float32")
        r = ops.minimum(r_h, r_w)

        if not self.scaleup:
            r = ops.minimum(r, 1.0)

        new_unpad_w = ops.cast(ops.round(ops.cast(current_w, "float32") * r), "int32")
        new_unpad_h = ops.cast(ops.round(ops.cast(current_h, "float32") * r), "int32")

        dw = target_w - new_unpad_w
        dh = target_h - new_unpad_h

        if self.auto:
            dw = dw % self.stride
            dh = dh % self.stride
        elif self.scaleFill:
            dw, dh = 0, 0
            new_unpad_w = target_w
            new_unpad_h = target_h

        dw_half = ops.cast(dw, "float32") / 2.0
        dh_half = ops.cast(dh, "float32") / 2.0

        need_resize = ops.logical_or(
            ops.not_equal(current_h, new_unpad_h), ops.not_equal(current_w, new_unpad_w)
        )

        if need_resize:
            im_resized = ops.image.resize(
                inputs,
                size=[new_unpad_h, new_unpad_w],
                interpolation="bilinear",
                antialias=False,
            )
        else:
            im_resized = inputs

        top = ops.cast(ops.round(dh_half - 0.1), "int32")
        bottom = ops.cast(ops.round(dh_half + 0.1), "int32")
        left = ops.cast(ops.round(dw_half - 0.1), "int32")
        right = ops.cast(ops.round(dw_half + 0.1), "int32")

        top = ops.maximum(top, 0)
        bottom = ops.maximum(bottom, 0)
        left = ops.maximum(left, 0)
        right = ops.maximum(right, 0)

        paddings = ops.convert_to_tensor(
            [[0, 0], [top, bottom], [left, right], [0, 0]], dtype="int32"
        )
        im_padded = ops.pad(im_resized, paddings, mode="constant", constant_values=0.0)

        im_final = self._apply_letterbox_color(im_padded, top, bottom, left, right)
        im_final = ops.flip(im_final, axis=-1)

        ratios = ops.stack([r, r], axis=0)
        padding_values = ops.stack([dw_half, dh_half], axis=0)

        ratios = ops.broadcast_to(ops.expand_dims(ratios, 0), [batch_size, 2])
        padding_values = ops.broadcast_to(
            ops.expand_dims(padding_values, 0), [batch_size, 2]
        )

        if was_single_image:
            im_final = ops.squeeze(im_final, axis=0)
            ratios = ops.squeeze(ratios, axis=0)
            padding_values = ops.squeeze(padding_values, axis=0)

        return im_final, ratios, padding_values

    def _apply_letterbox_color(self, im_padded, top, bottom, left, right):
        batch_size, final_h, final_w, channels = ops.shape(im_padded)

        y_coords = ops.arange(final_h, dtype="int32")
        x_coords = ops.arange(final_w, dtype="int32")

        top_mask = y_coords < top
        bottom_mask = y_coords >= (final_h - bottom)
        left_mask = x_coords < left
        right_mask = x_coords >= (final_w - right)

        top_mask_2d = ops.expand_dims(top_mask, 1)
        bottom_mask_2d = ops.expand_dims(bottom_mask, 1)
        left_mask_2d = ops.expand_dims(left_mask, 0)
        right_mask_2d = ops.expand_dims(right_mask, 0)

        border_mask_h = ops.logical_or(top_mask_2d, bottom_mask_2d)
        border_mask_w = ops.logical_or(left_mask_2d, right_mask_2d)
        border_mask_2d = ops.logical_or(border_mask_h, border_mask_w)

        border_mask = ops.expand_dims(ops.expand_dims(border_mask_2d, 0), -1)
        border_mask = ops.broadcast_to(
            border_mask, [batch_size, final_h, final_w, channels]
        )

        color_broadcast = ops.broadcast_to(
            ops.reshape(self.color_norm, [1, 1, 1, 3]),
            [batch_size, final_h, final_w, 3],
        )

        im_final = ops.where(border_mask, color_broadcast, im_padded)

        return im_final

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            return [
                (input_shape[0], self.new_shape[0], self.new_shape[1], input_shape[-1]),
                (input_shape[0], 2),
                (input_shape[0], 2),
            ]
        else:
            return [
                (self.new_shape[0], self.new_shape[1], input_shape[-1]),
                (2,),
                (2,),
            ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "new_shape": self.new_shape,
                "color": self.color,
                "auto": self.auto,
                "scaleFill": self.scaleFill,
                "scaleup": self.scaleup,
                "stride": self.stride,
            }
        )
        return config
