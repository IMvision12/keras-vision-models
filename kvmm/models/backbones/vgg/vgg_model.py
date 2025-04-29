import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import VGG_MODEL_CONFIG, VGG_WEIGHTS_CONFIG


def vgg_block(
    inputs,
    num_filters,
    channels_axis,
    data_format,
    batch_norm=False,
):
    """

    Args:
        inputs: Input tensor or layer.
        num_filters: List of filter specifications. Integer values
            specify the number of filters in Conv2D layers, while "M" indicates a MaxPooling2D
            layer should be inserted.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        batch_norm: Whether to include batch normalization layers
            after each convolution. Defaults to False.

    Returns:
        Tuple of (output tensor, list of intermediate feature tensors).
    """
    x = inputs
    layer_idx = 0
    features = []

    for v in num_filters:
        if v == "M":
            features.append(x)
            x = layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                data_format=data_format,
                name=f"Max_Pool_{layer_idx}",
            )(x)
            layer_idx += 1
        else:
            x = layers.Conv2D(
                v,
                3,
                padding="same",
                data_format=data_format,
                name=f"conv2d_{layer_idx}",
            )(x)
            layer_idx += 1

            if batch_norm:
                x = layers.BatchNormalization(
                    axis=channels_axis,
                    momentum=0.9,
                    epsilon=1e-5,
                    name=f"batchnorm_{layer_idx}",
                )(x)
                layer_idx += 1

            x = layers.ReLU(name=f"relu_{layer_idx}")(x)
            layer_idx += 1

    return x, features


@keras.saving.register_keras_serializable(package="kvmm")
class VGG(keras.Model):
    """Instantiates the VGG architecture with optional batch normalization.

    Reference:
        - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)

    Args:
        num_filters: List of integers specifying the number of filters for each convolutional block.
        batch_norm: Boolean, whether to include batch normalization after each convolutional layer.
            Defaults to `False`.
        include_top: Boolean, whether to include the fully-connected classification layers at the top of the network.
            Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to `False`.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, path to pretrained weights or one of the available options in `keras-vision`.
        input_tensor: Optional Keras tensor to use as the input to the model. If not provided, a new input tensor is created
            based on `input_shape`.
        input_shape: Optional tuple specifying the shape of the input data. Only required if `include_top=False`. Defaults to `None`.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification. Defaults to `1000`.
            Only applicable if `include_top=True`.
        classifier_activation: String or callable, activation function for the classifier layer. Set to `None` to return logits.
            Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"VGG"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        num_filters,
        batch_norm=False,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="tv_in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="VGG",
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

        x, features_extracted = vgg_block(
            x,
            num_filters,
            batch_norm=batch_norm,
            channels_axis=channels_axis,
            data_format=data_format,
        )
        features = features_extracted

        # Pre-logit layers
        x = layers.Conv2D(4096, 7, data_format=data_format, name="conv_fc1")(x)
        x = layers.ReLU(name="relu_fc1")(x)
        x = layers.Dropout(0.5, name="dropout_fc1")(x)
        x = layers.Conv2D(4096, 1, data_format=data_format, name="conv_fc2")(x)
        x = layers.ReLU(name="relu_fc2")(x)
        x = layers.Dropout(0.5, name="dropout_fc2")(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.Dropout(rate=0, name="dropout")(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
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

        self.num_filters = num_filters
        self.batch_norm = batch_norm
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
                "num_filters": self.num_filters,
                "batch_norm": self.batch_norm,
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
def VGG16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    num_classes=1000,
    weights="tv_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="VGG16",
    **kwargs,
):
    model = VGG(
        num_filters=VGG_MODEL_CONFIG["VGG16"],
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

    if weights in get_all_weight_names(VGG_WEIGHTS_CONFIG):
        load_weights_from_config("VGG16", weights, model, VGG_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def VGG19(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    num_classes=1000,
    weights="tv_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="VGG19",
    **kwargs,
):
    model = VGG(
        num_filters=VGG_MODEL_CONFIG["VGG19"],
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

    if weights in get_all_weight_names(VGG_WEIGHTS_CONFIG):
        load_weights_from_config("VGG19", weights, model, VGG_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def VGG16_BN(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    num_classes=1000,
    weights="tv_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="VGG16_BN",
    **kwargs,
):
    model = VGG(
        num_filters=VGG_MODEL_CONFIG["VGG16"],
        batch_norm=True,
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

    if weights in get_all_weight_names(VGG_WEIGHTS_CONFIG):
        load_weights_from_config("VGG16_BN", weights, model, VGG_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def VGG19_BN(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    num_classes=1000,
    weights="tv_in1k",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="VGG19_BN",
    **kwargs,
):
    model = VGG(
        num_filters=VGG_MODEL_CONFIG["VGG19"],
        batch_norm=True,
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

    if weights in get_all_weight_names(VGG_WEIGHTS_CONFIG):
        load_weights_from_config("VGG19_BN", weights, model, VGG_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
