import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import DENSENET_MODEL_CONFIG, DENSENET_WEIGHTS_CONFIG


def conv_block(
    x,
    growth_rate,
    expansion_ratio,
    channels_axis,
    data_format,
    name,
):
    """Creates a convolution block for DenseNet.

    Args:
        x: Input tensor.
        growth_rate: Number of output filters in the convolution.
        expansion_ratio: Expansion ratio for the bottleneck layer.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        name: Name prefix for the layers.

    Returns:
        Output tensor for the block.
    """
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        int(growth_rate * expansion_ratio),
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2d_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_2"
    )(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Conv2D(
        growth_rate,
        3,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2d_2",
    )(x)
    return x


def densenet_block(
    x,
    num_layers,
    growth_rate,
    channels_axis,
    data_format,
    name,
):
    """Creates a dense block containing multiple convolution blocks.

    Args:
        x: Input tensor.
        num_layers: Number of convolution blocks in the dense block.
        growth_rate: Growth rate for the convolution blocks.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        name: Name prefix for the layers.

    Returns:
        Output tensor for the block.
    """
    output = x

    for i in range(num_layers):
        layer_output = conv_block(
            output,
            growth_rate,
            expansion_ratio=4.0,
            channels_axis=channels_axis,
            data_format=data_format,
            name=f"{name}_denselayer{i + 1}",
        )
        output = layers.Concatenate(axis=channels_axis)([output, layer_output])

    return output


def transition_block(x, reduction, channels_axis, data_format, name):
    """Creates a transition block that reduces the number of channels and spatial dimensions.

    Args:
        x: Input tensor.
        reduction: Factor by which to reduce the number of channels.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        name: Name prefix for the layers.

    Returns:
        Output tensor for the block.
    """
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_transition_batchnorm",
    )(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Conv2D(
        int(x.shape[channels_axis] * reduction),
        1,
        1,
        "same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_transition_conv2d",
    )(x)
    x = layers.AveragePooling2D(
        2, 2, data_format=data_format, name=f"{name}_transition_pool"
    )(x)
    return x


@keras.saving.register_keras_serializable(package="kvmm")
class DenseNet(keras.Model):
    """Instantiates the DenseNet architecture.

    Reference:
    - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

    Args:
        num_blocks: List of integers, specifying the number of layers in each dense block.
        growth_rate: Integer, the growth rate for the dense blocks, controlling the number
            of filters added per layer.
        initial_filter: Integer, the number of filters in the initial convolutional layer.
            Defaults to `64`.
        include_top: Boolean, whether to include the fully-connected classification head
            at the top of the network. Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to `False`.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, specifying the path to pretrained weights or one of the available
            options in `keras-vision`.
        input_tensor: Optional Keras tensor (output of `layers.Input()`) to use as the
            model's input. If not provided, a new input tensor is created based on `input_shape`.
        input_shape: Optional tuple specifying the shape of the input data. If not specified,
            it defaults to `(None, None, 3)`, which means dynamic spatial dimensions with
            three color channels.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, specifying the number of output classes for classification.
            Defaults to `1000`. Only applicable if `include_top=True`.
        classifier_activation: String or callable, specifying the activation function for
            the classification layer. Set to `None` to return logits. Defaults to `"linear"`.
        name: String, specifying the name of the model. Defaults to `"DenseNet"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        num_blocks,
        growth_rate,
        initial_filter=64,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="tv_in1k",
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="DenseNet",
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
        channels_axis = -1 if data_format == "channels_last" else 1

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
        features = []

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), data_format=data_format)(x)
        x = layers.Conv2D(
            initial_filter,
            7,
            2,
            use_bias=False,
            data_format=data_format,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_norm"
        )(x)
        x = layers.ReLU(name="stem_relu")(x)
        x = layers.ZeroPadding2D(1, data_format=data_format)(x)
        x = layers.MaxPooling2D(3, 2, data_format=data_format, name="stem_pool")(x)
        features.append(x)

        for i, num_layers in enumerate(num_blocks):
            x = densenet_block(
                x,
                num_layers,
                growth_rate,
                channels_axis,
                data_format,
                name=f"dense_block{i + 1}",
            )

            if i != len(num_blocks) - 1:
                x = transition_block(
                    x,
                    0.5,
                    channels_axis,
                    data_format,
                    name=f"transition_block{i + 1}",
                )
            features.append(x)

        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name="final_batchnorm",
        )(x)
        x = layers.ReLU(name="final_relu")(x)

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

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.initial_filter = initial_filter
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
                "num_blocks": self.num_blocks,
                "growth_rate": self.growth_rate,
                "initial_filter": self.initial_filter,
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
def DenseNet121(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="tv_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DenseNet121",
    **kwargs,
):
    model = DenseNet(
        **DENSENET_MODEL_CONFIG["DenseNet121"],
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

    if weights in get_all_weight_names(DENSENET_WEIGHTS_CONFIG):
        load_weights_from_config("DenseNet121", weights, model, DENSENET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DenseNet161(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="tv_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DenseNet161",
    **kwargs,
):
    model = DenseNet(
        **DENSENET_MODEL_CONFIG["DenseNet161"],
        initial_filter=96,
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

    if weights in get_all_weight_names(DENSENET_WEIGHTS_CONFIG):
        load_weights_from_config("DenseNet161", weights, model, DENSENET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DenseNet169(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="tv_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DenseNet169",
    **kwargs,
):
    model = DenseNet(
        **DENSENET_MODEL_CONFIG["DenseNet169"],
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

    if weights in get_all_weight_names(DENSENET_WEIGHTS_CONFIG):
        load_weights_from_config("DenseNet169", weights, model, DENSENET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def DenseNet201(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="tv_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="DenseNet201",
    **kwargs,
):
    model = DenseNet(
        **DENSENET_MODEL_CONFIG["DenseNet201"],
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

    if weights in get_all_weight_names(DENSENET_WEIGHTS_CONFIG):
        load_weights_from_config("DenseNet201", weights, model, DENSENET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
