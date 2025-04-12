import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils
from keras.src.utils.argument_validation import standardize_tuple

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import INCEPTIONV4_WEIGHTS_CONFIG


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
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    x = inputs
    if padding is None:

        def calculate_padding(kernel_dim):
            pad_total = kernel_dim - 1
            pad_size = pad_total // 2
            pad_extra = (kernel_dim - 1) % 2
            return pad_size, pad_extra

        pad_h, extra_h = calculate_padding(kernel_size[0])
        pad_w, extra_w = calculate_padding(kernel_size[1])

        if strides > 1:
            padding_config = ((pad_h + extra_h, pad_h), (pad_w + extra_w, pad_w))
        else:
            padding_config = ((pad_h, pad_h), (pad_w, pad_w))

        x = layers.ZeroPadding2D(padding=padding_config, name=f"{name}_padding")(x)
        padding = "valid"

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=keras.config.image_data_format(),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        name=f"{name}_bn",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def stem_blocks(x, conv_block):
    """
    Implements the stem block of InceptionV4 which processes the input before the main inception blocks.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block

    Returns:
        Output tensor for the stem block
    """
    x = conv_block(x, 32, kernel_size=3, strides=2, name="features_0")
    x = conv_block(x, 32, kernel_size=3, name="features_1")
    x = conv_block(x, 64, kernel_size=3, padding=None, name="features_2")
    return x


def mixed3a(x, conv_block, name="features_3"):
    """
    Implements the Mixed3a block which combines max pooling and convolution operations.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block
        name: Name prefix for the layers

    Returns:
        Output tensor for the block
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    maxpool = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)
    conv = conv_block(x, 96, kernel_size=3, strides=2, name=f"{name}_conv")
    return layers.Concatenate(axis=channels_axis, name=name)([maxpool, conv])


def mixed4a(x, conv_block, name="features_4"):
    """
    Implements the Mixed4a block with parallel paths including 1x7 and 7x1 factorized convolutions.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block
        name: Name prefix for the layers

    Returns:
        Output tensor for the block
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 96, kernel_size=3, strides=1, name=f"{name}_branch0_1"
    )

    branch1 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        64,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        64,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )
    branch1 = conv_block(
        branch1, 96, kernel_size=3, strides=1, name=f"{name}_branch1_3"
    )

    return layers.Concatenate(axis=channels_axis, name=name)([branch0, branch1])


def mixed5a(x, conv_block, name="features_5"):
    """
    Implements the Mixed5a block which combines strided convolution with max pooling.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block
        name: Name prefix for the layers

    Returns:
        Output tensor for the block
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    conv = conv_block(x, 192, kernel_size=3, strides=2, name=f"{name}_conv")
    maxpool = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)
    return layers.Concatenate(axis=channels_axis, name=name)([conv, maxpool])


def inception_a(x, conv_block, block_idx):
    """
    Implements Inception-A block with four parallel paths.

    The block includes 1x1 convolutions, 3x3 convolutions, double 3x3 convolutions,
    and a pooling path, all concatenated at the end.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block
        block_idx: Integer index for naming the block

    Returns:
        Output tensor for the Inception-A block
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 96, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch1_1"
    )

    branch2 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch2_1"
    )
    branch2 = conv_block(
        branch2, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch2_2"
    )

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 96, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


def reduction_a(x, conv_block, name="features_10"):
    """
    Implements Reduction-A block which reduces the spatial dimensions of the input.

    The block includes three parallel paths: a strided 3x3 convolution,
    a 1x1 -> 3x3 -> 3x3 convolution path with stride 2, and a max pooling path.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block
        name: Name prefix for the layers

    Returns:
        Output tensor for the Reduction-A block
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 384, kernel_size=3, strides=2, name=f"{name}_branch0")

    branch1 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 224, kernel_size=3, strides=1, padding=None, name=f"{name}_branch1_1"
    )
    branch1 = conv_block(
        branch1, 256, kernel_size=3, strides=2, name=f"{name}_branch1_2"
    )

    branch2 = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2]
    )


def inception_b(x, conv_block, block_idx):
    """
    Implements Inception-B block with four parallel paths including 1x7 and 7x1
    factorized convolutions.

    The block includes a 1x1 convolution path, two paths with factorized convolutions,
    and an average pooling path.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block
        block_idx: Integer index for naming the block

    Returns:
        Output tensor for the Inception-B block
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        224,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        256,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )

    branch2 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2,
        192,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_1",
    )
    branch2 = conv_block(
        branch2,
        224,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch2_2",
    )
    branch2 = conv_block(
        branch2,
        224,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3",
    )
    branch2 = conv_block(
        branch2,
        256,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch2_4",
    )

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 128, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


def reduction_b(x, conv_block, name="features_18"):
    """
    Implements Reduction-B block which reduces the spatial dimensions of the input.

    The block includes three parallel paths: two paths with strided convolutions
    (one with additional factorized convolutions) and a max pooling path.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block
        name: Name prefix for the layers

    Returns:
        Output tensor for the Reduction-B block
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 192, kernel_size=3, strides=2, name=f"{name}_branch0_1"
    )

    branch1 = conv_block(x, 256, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        256,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        320,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )
    branch1 = conv_block(
        branch1, 320, kernel_size=3, strides=2, name=f"{name}_branch1_3"
    )

    branch2 = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2]
    )


def inception_c(x, conv_block, block_idx):
    """
    Implements Inception-C block with four parallel paths and split convolutions.

    The block includes a 1x1 convolution path, two paths with split convolutions
    into parallel 1x3 and 3x1 operations, and an average pooling path.

    Args:
        x: Input tensor
        conv_block: Function that creates a convolutional block
        block_idx: Integer index for naming the block

    Returns:
        Output tensor for the Inception-C block
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 256, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1_1a = conv_block(
        branch1,
        256,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1a",
    )
    branch1_1b = conv_block(
        branch1,
        256,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1b",
    )
    branch1 = layers.Concatenate(axis=channels_axis)([branch1_1a, branch1_1b])

    branch2 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2,
        448,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_1",
    )
    branch2 = conv_block(
        branch2,
        512,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch2_2",
    )
    branch2_3a = conv_block(
        branch2,
        256,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3a",
    )
    branch2_3b = conv_block(
        branch2,
        256,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3b",
    )
    branch2 = layers.Concatenate(axis=channels_axis)([branch2_3a, branch2_3b])

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 256, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


@keras.saving.register_keras_serializable(package="kvmm")
class InceptionV4Main(keras.Model):
    """
    Instantiates the InceptionV4 architecture.

    Reference:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning]
            (https://arxiv.org/abs/1602.07261) (AAAI 2017)

    Args:
        include_top: Boolean, whether to include the fully-connected classification layer at the top of the model.
            Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to False.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
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
            Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, name of the model. Defaults to `"InceptionV4"`.

    Returns:
        A Keras `Model` instance.

    """

    def __init__(
        self,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="inception",
        weights="tf_in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="InceptionV4",
        **kwargs,
    ):
        if include_top and as_backbone:
            raise ValueError(
                "Cannot use `as_backbone=True` with `include_top=True`. "
                f"Received: as_backbone={as_backbone}, include_top={include_top}"
            )

        if pooling is not None and pooling not in ["avg", "max"]:
            raise ValueError(
                "The `pooling` argument should be one of 'avg', 'max', or None. "
                f"Received: pooling={pooling}"
            )
        data_format = keras.config.image_data_format()

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=299,
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
        features = []

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        x = stem_blocks(x, conv_block)
        features.append(x)

        x = mixed3a(x, conv_block)
        features.append(x)

        x = mixed4a(x, conv_block)
        x = mixed5a(x, conv_block)
        features.append(x)

        for i in range(4):
            x = inception_a(x, conv_block, block_idx=6 + i)
        features.append(x)

        x = reduction_a(x, conv_block)

        for i in range(7):
            x = inception_b(x, conv_block, block_idx=11 + i)
        features.append(x)

        x = reduction_b(x, conv_block)

        for i in range(3):
            x = inception_c(x, conv_block, block_idx=19 + i)
        features.append(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        elif as_backbone:
            x = features
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

        self.include_top = include_top
        self.as_backbone = as_backbone
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_top": self.include_top,
                "as_backbone": self.as_backbone,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "pooling": self.pooling,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def InceptionV4(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="inception",
    num_classes=1000,
    weights="tf_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="InceptionV4",
    **kwargs,
):
    model = InceptionV4Main(
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(INCEPTIONV4_WEIGHTS_CONFIG):
        load_weights_from_config(
            "InceptionV4", weights, model, INCEPTIONV4_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
