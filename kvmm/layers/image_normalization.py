import typing

import keras
from keras import layers, ops

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (0.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@keras.saving.register_keras_serializable(package="kvmm")
class ImageNormalizationLayer(layers.Layer):
    """
    Implements image normalization  operations commonly used in computer vision models.
    This layer handles pixel value normalization and standardization using predefined
    constants for popular model architectures.

    Args:
        mode (str): Normalization  mode to use. Must be one of:
            - 'imagenet': Standard ImageNet normalization (default)
            - 'inception': Inception-style normalization
            - 'dpn': DPN model normalization
            - 'clip': OpenAI CLIP model normalization
            - 'zero_to_one': Scales pixels to [0, 1] range
            - 'minus_one_to_one': Scales pixels to [-1, 1] range
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Methods:
        call(inputs):
            Applies the specified normalization to the input images.
            Input should be in uint8 format with values in [0, 255].
        compute_output_shape(input_shape):
            Returns the shape of the output tensor.
        get_config():
            Returns a dictionary containing the configuration of the layer.

    Example:
        >>> layer = ImageNormalizationLayer(mode='imagenet')
        >>> normalized_images = layer(raw_images)
    """

    def __init__(
        self,
        mode: typing.Literal[
            "imagenet", "inception", "dpn", "clip", "zero_to_one", "minus_one_to_one"
        ] = "imagenet",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode

        if mode == "imagenet":
            self.mean = ops.convert_to_tensor(IMAGENET_DEFAULT_MEAN)
            self.std = ops.convert_to_tensor(IMAGENET_DEFAULT_STD)
        elif mode == "inception":
            self.mean = ops.convert_to_tensor(IMAGENET_INCEPTION_MEAN)
            self.std = ops.convert_to_tensor(IMAGENET_INCEPTION_STD)
        elif mode == "dpn":
            self.mean = ops.convert_to_tensor(IMAGENET_DPN_MEAN)
            self.std = ops.convert_to_tensor(IMAGENET_DPN_STD)
        elif mode == "clip":
            self.mean = ops.convert_to_tensor(OPENAI_CLIP_MEAN)
            self.std = ops.convert_to_tensor(OPENAI_CLIP_STD)
        elif mode in ["zero_to_one", "minus_one_to_one"]:
            self.mean = None
            self.std = None
        else:
            raise ValueError(
                f"Mode '{mode}' not recognized. Must be one of: "
                "'imagenet', 'inception', 'dpn', 'clip', 'zero_to_one', 'minus_one_to_one'"
            )

    def call(self, inputs):
        x = ops.cast(inputs, dtype="float32")

        x = x / 255.0

        if self.mode == "minus_one_to_one":
            return x * 2.0 - 1.0
        elif self.mode == "zero_to_one":
            return x
        else:
            if keras.config.image_data_format() == "channels_first":
                mean = ops.reshape(self.mean, (-1, 1, 1))
                std = ops.reshape(self.std, (-1, 1, 1))
            else:
                mean = ops.reshape(self.mean, (1, 1, -1))
                std = ops.reshape(self.std, (1, 1, -1))

            x = (x - mean) / std
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config
