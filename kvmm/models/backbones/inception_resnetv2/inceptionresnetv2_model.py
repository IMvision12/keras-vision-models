import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils
from keras.src.utils.argument_validation import standardize_tuple

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import INCEPTIONRESNETV2_WEIGHTS_CONFIG


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
        name=f"{name}_batchnorm",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_5b_block(inputs, name="mixed_5b"):
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 96, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 48, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 64, 5, padding="same", name=f"{name}_branch1_1")

    branch2 = conv_block(inputs, 64, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 96, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(branch2, 96, 3, padding="same", name=f"{name}_branch2_2")

    branch_pool = layers.AveragePooling2D(
        pool_size=3,
        strides=1,
        padding="same",
        data_format=keras.config.image_data_format(),
    )(inputs)
    branch_pool = conv_block(branch_pool, 64, name=f"{name}_branch3_1")

    return layers.Concatenate(axis=channels_axis)(
        [branch0, branch1, branch2, branch_pool]
    )


def block35(inputs, scale=1.0, name="repeat_0"):
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 32, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 32, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 32, 3, padding="same", name=f"{name}_branch1_1")

    branch2 = conv_block(inputs, 32, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 48, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(branch2, 64, 3, padding="same", name=f"{name}_branch2_2")

    branches = [branch0, branch1, branch2]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(320, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_6a_block(inputs, name="mixed_6a"):
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(
        inputs, 384, 3, strides=2, padding="valid", name=f"{name}_branch0"
    )

    branch1 = conv_block(inputs, 256, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 256, 3, padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(
        branch1, 384, 3, strides=2, padding="valid", name=f"{name}_branch1_2"
    )

    branch_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inputs)

    return layers.Concatenate(axis=channels_axis)([branch0, branch1, branch_pool])


def block17(inputs, scale=1.0, name="repeat_1_0"):
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 192, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 128, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 160, (1, 7), padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(branch1, 192, (7, 1), padding="same", name=f"{name}_branch1_2")

    branches = [branch0, branch1]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(1088, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_7a_block(inputs, name="mixed_7a"):
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 256, 1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 384, 3, strides=2, padding="valid", name=f"{name}_branch0_1"
    )

    branch1 = conv_block(inputs, 256, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 288, 3, strides=2, padding="valid", name=f"{name}_branch1_1"
    )

    branch2 = conv_block(inputs, 256, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 288, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(
        branch2, 320, 3, strides=2, padding="valid", name=f"{name}_branch2_2"
    )

    branch_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inputs)

    return layers.Concatenate(axis=channels_axis)(
        [branch0, branch1, branch2, branch_pool]
    )


def block8(inputs, scale=1.0, activation=True, name="repeat_2_0"):
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 192, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 192, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 224, (1, 3), padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(branch1, 256, (3, 1), padding="same", name=f"{name}_branch1_2")

    branches = [branch0, branch1]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(2080, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    if activation:
        x = layers.Activation("relu", name=name)(x)
    return x


@keras.saving.register_keras_serializable(package="kvmm")
class InceptionResNetV2Main(keras.Model):
    """
    Instantiates the Inception-ResNet-v2 architecture.

    Reference:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)

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
        name: String, name of the model. Defaults to `"InceptionResNetV2"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="inception",
        weights="tf_ens_adv_in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="InceptionResNetV2",
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

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=299,
            min_size=75,
            data_format=keras.config.image_data_format(),
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

        x = conv_block(x, 32, 3, strides=2, padding="valid", name="conv2d_1a")
        x = conv_block(x, 32, 3, padding="valid", name="conv2d_2a")
        x = conv_block(x, 64, 3, padding="same", name="conv2d_2b")
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = conv_block(x, 80, 1, name="conv2d_3b")
        x = conv_block(x, 192, 3, padding="valid", name="conv2d_4a")
        x = layers.MaxPooling2D(3, strides=2)(x)
        features.append(x)

        x = mixed_5b_block(x, name="mixed_5b")

        for i in range(10):
            x = block35(x, scale=0.17, name=f"repeat_{i}")
        features.append(x)

        x = mixed_6a_block(x, name="mixed_6a")

        for i in range(20):
            x = block17(x, scale=0.10, name=f"repeats_1_{i}")
        features.append(x)

        x = mixed_7a_block(x, name="mixed_7a")

        for i in range(9):
            x = block8(x, scale=0.20, name=f"repeats_2_{i}")

        x = block8(x, activation=False, name="block8")

        x = conv_block(x, 1536, 1, name="conv2d_7b")
        features.append(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        elif as_backbone:
            x = features
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.include_top = include_top
        self.as_backbone = as_backbone
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self) -> dict:
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
def InceptionResNetV2(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="inception",
    num_classes=1000,
    weights="tf_ens_adv_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="InceptionResNetV2",
    **kwargs,
):
    model = InceptionResNetV2Main(
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
    if weights in get_all_weight_names(INCEPTIONRESNETV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "InceptionResNetV2", weights, model, INCEPTIONRESNETV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
