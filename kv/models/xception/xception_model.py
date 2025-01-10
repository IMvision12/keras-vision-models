import keras
from keras import backend, layers
from keras.src.applications import imagenet_utils

from kv.layers import ImagePreprocessingLayer
from kv.utils import get_all_weight_names, load_weights_from_config

from ...model_registry import register_model
from .config import XCEPTION_WEIGHTS_CONFIG


def conv_block(
    x,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    separable=False,
    use_activation=True,
    use_preactivation=False,
    use_bias=False,
):
    """
    Applies a convolutional block with batch normalization and optional activation.

    Args:
        x: Input tensor
        filters: Number of output filters
        kernel_size: Size of the convolution kernel
        strides: Stride dimensions for the convolution
        padding: Padding mode ('same' or 'valid')
        separable: Whether to use separable convolution
        use_activation: Whether to apply ReLU activation after convolution
        use_preactivation: Whether to apply ReLU activation before convolution
        use_bias: Whether to include a bias vector (True by default)

    Returns:
        Processed tensor
    """
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    x = layers.Activation("relu")(x) if use_preactivation else x

    conv_layer = layers.SeparableConv2D if separable else layers.Conv2D
    x = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )(x)

    x = layers.BatchNormalization(axis=channels_axis)(x)
    x = layers.Activation("relu")(x) if use_activation else x
    return x


def entry_flow(x):
    """
    Entry flow of the Xception architecture
    Contains blocks 1-4
    """
    # Block 1
    x = conv_block(x, 32, (3, 3), strides=(2, 2), padding="valid")
    x = conv_block(x, 64, (3, 3), padding="valid")

    residual = conv_block(x, 128, (1, 1), strides=(2, 2), use_activation=False)

    # Block 2
    x = conv_block(x, 128, (3, 3), separable=True)
    x = conv_block(x, 128, (3, 3), separable=True, use_activation=False)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # Block 3
    residual = conv_block(
        x, 256, (1, 1), strides=(2, 2), use_bias=False, use_activation=False
    )
    x = conv_block(x, 256, (3, 3), use_preactivation=True, separable=True)
    x = conv_block(x, 256, (3, 3), use_activation=False, separable=True)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # Block 4
    residual = conv_block(x, 728, (1, 1), strides=(2, 2), use_activation=False)
    x = conv_block(x, 728, (3, 3), separable=True, use_preactivation=True)
    x = conv_block(x, 728, (3, 3), separable=True, use_activation=False)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    return x


def middle_flow(x):
    """
    Middle flow of the Xception architecture
    Contains blocks 5-12 (8 repeated blocks)
    """
    for i in range(8):
        residual = x
        x = conv_block(x, 728, (3, 3), separable=True, use_preactivation=True)
        x = conv_block(x, 728, (3, 3), separable=True)
        x = conv_block(x, 728, (3, 3), separable=True, use_activation=False)
        x = layers.add([x, residual])

    return x


def exit_flow(x):
    """
    Exit flow of the Xception architecture
    Contains blocks 13-14 and the final classification layers
    """
    residual = conv_block(x, 1024, (1, 1), strides=(2, 2), use_activation=False)

    # Block 13
    x = conv_block(x, 728, (3, 3), separable=True, use_preactivation=True)
    x = conv_block(x, 1024, (3, 3), separable=True, use_activation=False)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # Block 14
    x = conv_block(x, 1536, (3, 3), separable=True)
    x = conv_block(x, 2048, (3, 3), separable=True)

    return x


@keras.saving.register_keras_serializable(package="kv")
class XceptionMain(keras.Model):
    """
    Instantiates the Xception architecture.

    Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (CVPR 2017)

    Args:
        include_top: Boolean, whether to include the fully-connected classification layer at the top of the model.
            Defaults to `True`.
        include_preprocessing: Boolean, whether to include preprocessing layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        preprocessing_mode: String, specifying the preprocessing mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_preprocessing=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options in `keras-vision`.
        input_tensor: Optional Keras tensor (output of `layers.Input()`) to use as the model's input.
            If not provided, a new input tensor is created based on `input_shape`.
        input_shape: Optional tuple specifying the shape of the input data. If not specified, defaults to `(299, 299, 3)`.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification. Defaults to `1000`.
            Only applicable if `include_top=True`.
        classifier_activation: String or callable, activation function for the classification layer.
            Set to `None` to return logits. Defaults to `"linear"`.
        name: String, name of the model. Defaults to `"Xception"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        include_top=True,
        include_preprocessing=True,
        preprocessing_mode="imagenet",
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="xception",
        **kwargs,
    ):
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=299,
            min_size=32,
            data_format=backend.image_data_format(),
            require_flatten=include_top,
            weights=weights,
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input
        x = (
            ImagePreprocessingLayer(mode=preprocessing_mode)(inputs)
            if include_preprocessing
            else inputs
        )
        x = entry_flow(x)
        x = middle_flow(x)
        x = exit_flow(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        # Store configuration
        self.include_top = include_top
        self.include_preprocessing = include_preprocessing
        self.preprocessing_mode = preprocessing_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self) -> dict:
        return {
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
def Xception(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="inception",
    num_classes=1000,
    weights="imagenet",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="Xception",
    **kwargs,
):
    model = XceptionMain(
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

    if weights in get_all_weight_names(XCEPTION_WEIGHTS_CONFIG):
        load_weights_from_config("Xception", weights, model, XCEPTION_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
