import keras
from keras import backend, layers
from keras.src.applications import imagenet_utils
from keras.src.utils.argument_validation import standardize_tuple

from kv.utils import download_weights

from ...model_registry import register_model
from .config import INCEPTIONV3_WEIGHTS_CONFIG


def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    strides=1,
    bn_momentum=0.9,
    bn_epsilon=1e-3,
    padding="valid",
    name="conv2d_block",
):
    """
    Creates a convolutional block with batch normalization and ReLU activation.

    Args:
        inputs: Input tensor
        filters: Number of output filters
        kernel_size: Size of the convolution kernel
        strides: Stride length of the convolution
        bn_momentum: Momentum for batch normalization
        bn_epsilon: Epsilon value for batch normalization
        padding: Padding type ("valid", "same", or None)
        name: Name prefix for the layers

    Returns:
        Output tensor for the block.
    """
    kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    x = inputs
    if padding is None:
        padding = "same"
        if strides > 1:
            padding = "valid"
            x = layers.ZeroPadding2D(
                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                name=f"{name}_padding",
            )(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=f"{name}_conv2d",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        name=f"{name}_batchnorm",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def inception_blocka(inputs, pool_channels, name="inception_block_a"):
    """
    Implements Inception block type A.

    Args:
        inputs: Input tensor
        pool_channels: Number of filters for the pooling branch
        name: Name prefix for the layers

    Returns:
        Output tensor for the block.
    """
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    branch1x1 = conv_block(inputs, 64, 1, name=f"{name}_branch1x1")

    branch5x5 = conv_block(inputs, 48, 1, name=f"{name}_branch5x5_1")
    branch5x5 = conv_block(branch5x5, 64, 5, padding=None, name=f"{name}_branch5x5_2")

    branch3x3dbl = conv_block(inputs, 64, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_3"
    )

    branch_pool = layers.ZeroPadding2D(padding=1)(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3,
        strides=1,
        data_format=backend.image_data_format(),
    )(branch_pool)
    branch_pool = conv_block(
        branch_pool,
        pool_channels,
        name=f"{name}_branch_pool",
    )

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool]
    )


def inception_blockb(inputs, name="inception_block_b"):
    """
    Implements Inception block type B with dimensionality reduction.

    Args:
        inputs: Input tensor
        name: Name prefix for the layers

    Returns:
        Output tensor for the block.
    """
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    branch3x3 = conv_block(inputs, 384, 3, 2, name=f"{name}_branch3x3")

    branch3x3dbl = conv_block(inputs, 64, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, strides=2, name=f"{name}_branch3x3dbl_3"
    )

    branch_pool = layers.MaxPooling2D(
        pool_size=3, strides=2, name=f"{name}_branch_pool"
    )(inputs)

    return layers.Concatenate(axis=channels_axis)(
        [branch3x3, branch3x3dbl, branch_pool]
    )


def inception_blockc(inputs, branch7x7_channels, name="inception_block_c"):
    """
    Implements Inception block type C with factorized 7x7 convolutions.

    Args:
        inputs: Input tensor
        branch7x7_channels: Number of filters for 7x7 convolution branches
        name: Name prefix for the layers

    Returns:
        Output tensor for the block.
    """
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
    c7 = branch7x7_channels

    branch1x1 = conv_block(inputs, 192, 1, name=f"{name}_branch1x1")

    branch7x7 = conv_block(inputs, c7, 1, name=f"{name}_branch7x7_1")
    branch7x7 = conv_block(
        branch7x7, c7, (1, 7), padding=None, name=f"{name}_branch7x7_2"
    )
    branch7x7 = conv_block(
        branch7x7, 192, (7, 1), padding=None, name=f"{name}_branch7x7_3"
    )

    branch7x7dbl = conv_block(inputs, c7, 1, name=f"{name}_branch7x7dbl_1")
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (7, 1), padding=None, name=f"{name}_branch7x7dbl_2"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (1, 7), padding=None, name=f"{name}_branch7x7dbl_3"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (7, 1), padding=None, name=f"{name}_branch7x7dbl_4"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, 192, (1, 7), padding=None, name=f"{name}_branch7x7dbl_5"
    )

    branch_pool = layers.ZeroPadding2D(padding=1)(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3, strides=1, data_format=backend.image_data_format()
    )(branch_pool)
    branch_pool = conv_block(branch_pool, 192, 1, name=f"{name}_branch_pool")

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool]
    )


def inception_blockd(inputs, name="inception_block_d"):
    """
    Implements Inception block type D with strided convolutions.

    Args:
        inputs: Input tensor
        name: Name prefix for the layers

    Returns:
        Output tensor for the block.
    """
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    branch3x3 = conv_block(inputs, 192, 1, name=f"{name}_branch3x3_1")
    branch3x3 = conv_block(branch3x3, 320, 3, strides=2, name=f"{name}_branch3x3_2")

    branch7x7x3 = conv_block(inputs, 192, 1, name=f"{name}_branch7x7x3_1")
    branch7x7x3 = conv_block(
        branch7x7x3, 192, (1, 7), padding=None, name=f"{name}_branch7x7x3_2"
    )
    branch7x7x3 = conv_block(
        branch7x7x3, 192, (7, 1), padding=None, name=f"{name}_branch7x7x3_3"
    )
    branch7x7x3 = conv_block(
        branch7x7x3, 192, 3, strides=2, name=f"{name}_branch7x7x3_4"
    )

    branch_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inputs)

    return layers.Concatenate(axis=channels_axis)([branch3x3, branch7x7x3, branch_pool])


def inception_blocke(inputs, name="inception_block_e"):
    """
    Implements Inception block type E with parallel factorized convolutions.

    Args:
        inputs: Input tensor
        name: Name prefix for the layers

    Returns:
        Output tensor for the block.
    """
    channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

    branch1x1 = conv_block(inputs, 320, 1, name=f"{name}_branch1x1")

    branch3x3 = conv_block(inputs, 384, 1, name=f"{name}_branch3x3_1")
    branch3x3_a = conv_block(
        branch3x3,
        filters=384,
        kernel_size=(1, 3),
        padding=None,
        name=f"{name}_branch3x3_2a",
    )
    branch3x3_b = conv_block(
        branch3x3,
        filters=384,
        kernel_size=(3, 1),
        padding=None,
        name=f"{name}_branch3x3_2b",
    )
    branch3x3 = layers.Concatenate(axis=channels_axis)([branch3x3_a, branch3x3_b])

    branch3x3dbl = conv_block(inputs, 448, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 384, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl_a = conv_block(
        branch3x3dbl,
        filters=384,
        kernel_size=(1, 3),
        padding=None,
        name=f"{name}_branch3x3dbl_3a",
    )
    branch3x3dbl_b = conv_block(
        branch3x3dbl,
        filters=384,
        kernel_size=(3, 1),
        padding=None,
        name=f"{name}_branch3x3dbl_3b",
    )
    branch3x3dbl = layers.Concatenate(axis=channels_axis)(
        [branch3x3dbl_a, branch3x3dbl_b]
    )

    branch_pool = layers.ZeroPadding2D(padding=1)(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3, strides=1, data_format=backend.image_data_format()
    )(branch_pool)
    branch_pool = conv_block(branch_pool, 192, 1, name=f"{name}_branch_pool")

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    )


@keras.saving.register_keras_serializable(package="kv")
class InceptionV3Main(keras.Model):
    """
    Instantiates the InceptionV3 architecture.

    Reference:
    - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) (CVPR 2016)

    Args:
        include_top: Boolean, whether to include the fully-connected classification layer at the top of the model.
            Defaults to `True`.
        weights: String, path to pretrained weights or one of the available options in `keras-vision`.
            Options include 'in1k', 'in22k', or `None` to initialize the model with random weights.
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
        name: String, name of the model. Defaults to `"InceptionV3"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        include_top=True,
        weights="ink1",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionV3",
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

        # Stem block
        x = conv_block(inputs, 32, 3, strides=2, name="Conv2d_1a_3x3")
        x = conv_block(x, 32, 3, name="Conv2d_2a_3x3")
        x = conv_block(x, 64, 3, padding=None, name="Conv2d_2b_3x3")

        # Blocks
        x = layers.MaxPooling2D(3, 2, name="Pool1")(x)
        x = conv_block(x, 80, 1, name="Conv2d_3b_1x1")
        x = conv_block(x, 192, 3, name="Conv2d_4a_3x3")
        x = layers.MaxPooling2D(3, 2, name="Pool2")(x)
        x = inception_blocka(x, 32, "Mixed_5b")
        x = inception_blocka(x, 64, "Mixed_5c")
        x = inception_blocka(x, 64, "Mixed_5d")

        x = inception_blockb(x, "Mixed_6a")

        x = inception_blockc(x, 128, "Mixed_6b")
        x = inception_blockc(x, 160, "Mixed_6c")
        x = inception_blockc(x, 160, "Mixed_6d")
        x = inception_blockc(x, 192, "Mixed_6e")

        x = inception_blockd(x, "Mixed_7a")
        x = inception_blocke(x, "Mixed_7b")
        x = inception_blocke(x, "Mixed_7c")

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="classifier",
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        # Store configuration
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self) -> dict:
        return {
            "include_top": self.include_top,
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
def InceptionV3(
    include_top=True,
    num_classes=1000,
    weights="gluon_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="InceptionV3",
    **kwargs,
):
    model = InceptionV3Main(
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights:
        weights_path = download_weights(
            INCEPTIONV3_WEIGHTS_CONFIG["InceptionV3"][weights]["url"]
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
