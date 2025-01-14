import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kv.layers import ImagePreprocessingLayer
from kv.utils import get_all_weight_names, load_weights_from_config

from ...model_registry import register_model
from .config import CONVMIXER_MODEL_CONFIG, CONVMIXER_WEIGHTS_CONFIG


def convmixer_block(
    x, filters, kernel_size, act_layer, channels_axis, data_format, name
):
    """A building block for the ConvMixer architecture.

    Args:
        x: input tensor.
        filters: int, the number of output filters for the convolution layers.
        kernel_size: int, the size of the convolution kernel.
        act_layer: string, activation function to apply after each convolution.
        channels_axis: axis along which the channels are defined in the input tensor.
        name: string, block name.

    Returns:
        Output tensor for the block.
    """
    inputs = x
    x = layers.DepthwiseConv2D(
        kernel_size,
        1,
        padding="same",
        use_bias=True,
        activation=act_layer,
        data_format=data_format,
        name=f"{name}_depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_1"
    )(x)

    x = layers.Add(name=f"{name}_add")([inputs, x])

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        activation=act_layer,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv2d",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_2"
    )(x)

    return x


@keras.saving.register_keras_serializable(package="kv")
class ConvMixer(keras.Model):
    """Instantiates the ConvMixer architecture.

    Reference:
    - [Patches Are All You Need?](
        https://arxiv.org/abs/2201.09792) (OpenReview 2022)

    Args:
        dim: Integer, the dimensionality of the feature maps in the ConvMixer blocks.
        depth: Integer, the number of ConvMixer blocks to stack.
        kernel_size: Integer or tuple, specifying the kernel size for depthwise
            convolutions in ConvMixer blocks.
        patch_size: Integer or tuple, specifying the patch size for the initial
            convolutional layer.
        act_layer: String, activation function to use throughout the model. Defaults to `"gelu"`.
        include_top: Boolean, whether to include the classification head at the top
            of the network. Defaults to `True`.
        include_preprocessing: Boolean, whether to include preprocessing layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        preprocessing_mode: String, specifying the preprocessing mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_preprocessing=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options in `keras-vision`.
        input_tensor: Optional Keras tensor (output of `layers.Input()`) to use as
            the model's input. If not provided, a new input tensor is created based
            on `input_shape`.
        input_shape: Optional tuple specifying the shape of the input data. If not
            specified, it defaults to `(224, 224, 3)` when `include_top=True`.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to `1000`.
        classifier_activation: String or callable, activation function for the top
            layer. Set to `None` to return logits. Defaults to `"linear"`.
        name: String, the name of the model. Defaults to `"ConvMixer"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        dim,
        depth,
        kernel_size,
        patch_size,
        act_layer="gelu",
        include_top=True,
        include_preprocessing=True,
        preprocessing_mode="imagenet",
        weights="ink1",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="ConvMixer",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
            min_size=32,
            data_format=data_format,
            require_flatten=include_top,
            weights=weights,
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input

        x = (
            ImagePreprocessingLayer(mode=preprocessing_mode)(inputs)
            if include_preprocessing
            else inputs
        )

        # Stem layer
        x = layers.Conv2D(
            dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=True,
            activation=act_layer,
            data_format=data_format,
            name="stem_conv2d",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_batchnorm"
        )(x)

        # ConvMixer Blocks
        for i in range(depth):
            x = convmixer_block(
                x,
                dim,
                kernel_size,
                act_layer,
                channels_axis,
                data_format,
                f"mixer_block_{i}",
            )

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)

        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(
                    data_format=data_format, name="avg_pool"
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(data_format=data_format, name="max_pool")(
                    x
                )

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.act_layer = act_layer
        self.include_top = include_top
        self.include_preprocessing = include_preprocessing
        self.preprocessing_mode = preprocessing_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "dim": self.dim,
            "depth": self.depth,
            "patch_size": self.patch_size,
            "kernel_size": self.kernel_size,
            "act_layer": self.act_layer,
            "include_top": self.include_top,
            "include_preprocessing": self.include_preprocessing,
            "preprocessing_mode": self.preprocessing_mode,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def ConvMixer_1536_20(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="ink1",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvMixer_1536_20",
    **kwargs,
):
    model = ConvMixer(
        **CONVMIXER_MODEL_CONFIG["ConvMixer_1536_20"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVMIXER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvMixer_1536_20", weights, model, CONVMIXER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvMixer_768_32(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="ink1",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvMixer_768_32",
    **kwargs,
):
    model = ConvMixer(
        **CONVMIXER_MODEL_CONFIG["ConvMixer_768_32"],
        act_layer="relu",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(CONVMIXER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvMixer_768_32", weights, model, CONVMIXER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvMixer_1024_20(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="ink1",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvMixer_1024_20",
    **kwargs,
):
    model = ConvMixer(
        **CONVMIXER_MODEL_CONFIG["ConvMixer_1024_20"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVMIXER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvMixer_1024_20", weights, model, CONVMIXER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
