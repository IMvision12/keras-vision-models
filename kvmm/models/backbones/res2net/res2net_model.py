import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import RES2NET_MODEL_CONFIG, RES2NET_WEIGHTS_CONFIG


def conv_block(
    x,
    filters,
    kernel_size,
    channels_axis,
    data_format,
    strides=1,
    use_relu=True,
    groups=1,
    name=None,
    bn_name=None,
):
    """Applies a convolution block with optional grouped convolutions.

    Args:
        x: Input Keras layer.
        filters: Number of output filters for the convolution.
        kernel_size: Size of the convolution kernel.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride of the convolution.
        use_relu: Whether to apply ReLU activation after convolution.
        groups: Number of groups for grouped convolution.
        name: Optional name for the convolution layer.
        bn_name: Optional name for the batch normalization layer.

    Returns:
       Output tensor for the block.
    """
    if strides > 1:
        pad_h = pad_w = kernel_size // 2
        x = layers.ZeroPadding2D(padding=(pad_h, pad_w), data_format=data_format)(x)
        padding_mode = "valid"
    else:
        padding_mode = "same"

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding_mode,
        use_bias=False,
        groups=groups,
        data_format=data_format,
        name=name,
    )(x)

    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        name=bn_name,
    )(x)

    if use_relu:
        x = layers.ReLU()(x)
    return x


def bottle2neck_block(
    x,
    filters,
    block_name,
    data_format,
    stride=1,
    downsample=False,
    cardinality=1,
    base_width=26,
    scale=4,
):
    """Res2Net/ResNeSt Bottle2neck block with multi-scale features.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the bottleneck layers.
        block_name: Name prefix for layers in the block.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        stride: Stride for the 3x3 convolution layers.
        downsample: Whether to downsample the input.
        cardinality: Number of groups for grouped convolutions.
        base_width: Base width of the block, controls channel scaling.
        scale: Scale factor that determines number of feature scales.

    Returns:
        Output tensor for the block.

    Notes:
        The block implements multi-scale feature processing by:
        1. Initial 1x1 conv to expand channels
        2. Split features into multiple scales
        3. Hierarchical residual-like connections between scales
        4. Optional average pooling for the last scale
        5. Concatenate all scales and reduce with 1x1 conv

        The expansion factor is fixed at 4, similar to standard ResNet bottleneck blocks.
    """
    channels_axis = -1 if data_format == "channels_last" else 1
    expansion = 4
    is_first = stride > 1 or downsample
    width = int(filters * (base_width / 64.0)) * cardinality
    outplanes = filters * expansion

    identity = x

    x = conv_block(
        x,
        width * scale,
        kernel_size=1,
        channels_axis=channels_axis,
        data_format=data_format,
        name=f"{block_name}_conv_1",
        bn_name=f"{block_name}_batchnorm_1",
    )

    x_splits = ops.split(x, scale, axis=channels_axis)
    spouts = []

    for i in range(scale - 1):
        if i == 0 or is_first:
            sp = x_splits[i]
        else:
            sp = layers.Add()([spouts[-1], x_splits[i]])

        sp = conv_block(
            sp,
            width,
            kernel_size=3,
            channels_axis=channels_axis,
            data_format=data_format,
            strides=stride if is_first else 1,
            groups=cardinality,
            name=f"{block_name}_conv_s_{i}",
            bn_name=f"{block_name}_batchnorm_s_{i}",
        )
        spouts.append(sp)

    if scale > 1:
        if is_first:
            last = layers.ZeroPadding2D(
                padding=((1, 1), (1, 1)), data_format=data_format
            )(x_splits[-1])
            last = layers.AveragePooling2D(
                pool_size=3,
                strides=stride,
                padding="valid",
                data_format=data_format,
            )(last)
        else:
            last = x_splits[-1]
        spouts.append(last)

    out = layers.Concatenate(axis=channels_axis)(spouts)

    out = conv_block(
        out,
        outplanes,
        kernel_size=1,
        channels_axis=channels_axis,
        data_format=data_format,
        use_relu=False,
        name=f"{block_name}_conv_3",
        bn_name=f"{block_name}_batchnorm_3",
    )

    if downsample:
        identity = conv_block(
            identity,
            outplanes,
            kernel_size=1,
            channels_axis=channels_axis,
            data_format=data_format,
            strides=stride,
            use_relu=False,
            name=f"{block_name}_downsample_0",
            bn_name=f"{block_name}_downsample_1",
        )

    out = layers.Add()([identity, out])
    out = layers.ReLU()(out)

    return out


@keras.saving.register_keras_serializable(package="kvmm")
class Res2Net(keras.Model):
    """
    Instantiates the Res2Net architecture, which introduces a novel building block for
    CNNs that constructs hierarchical residual-like connections within a single residual block.

    Reference:
    - [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169) (TPAMI 2019)

    Args:
        depth: List of integers, number of blocks to include in each stage of the network.
        base_width: Integer, the base width of the Res2Net block. Controls the number of
            channels in each scale. Defaults to `26`.
        scale: Integer, the number of scales in each Res2Net block. Higher values create
            more hierarchical feature representations. Defaults to `4`.
        cardinality: Integer, the size of the set of transformations in each block.
            Similar to ResNeXt's cardinality parameter. Defaults to `1`.
        include_top: Boolean, whether to include the fully-connected classification
            layer at the top. Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to `False`.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options in `keras-vision`.
        input_tensor: Optional Keras tensor to use as the model's input. If not provided,
            a new input tensor is created based on `input_shape`.
        input_shape: Optional tuple specifying the shape of the input data. If not
            specified, defaults to `(224, 224, 3)`.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification. Defaults to `1000`.
            Only applicable if `include_top=True`.
        classifier_activation: String or callable, activation function for the
            classifier layer. Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"Res2Net"`.

    Returns:
        A Keras `Model` instance.

    The Res2Net architecture enhances multi-scale feature learning by introducing a hierarchical
    residual-like connection within each residual block. This design allows the network to
    represent features at multiple scales, making it particularly effective for tasks that
    require understanding both fine and coarse patterns in images.

    Key features of Res2Net:
    - Hierarchical residual connections that enable multi-scale feature learning
    - Flexible scale parameter that controls the granularity of feature hierarchies
    - Compatible with various CNN architectures as a drop-in replacement for ResNet blocks
    - Maintains computational efficiency while increasing feature expressiveness
    """

    def __init__(
        self,
        depth,
        base_width=26,
        scale=4,
        cardinality=1,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="in1k",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="Res2Net",
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

        x = layers.ZeroPadding2D(padding=3, data_format=data_format)(x)
        x = layers.Conv2D(
            64,
            kernel_size=7,
            strides=2,
            padding="valid",
            use_bias=False,
            data_format=data_format,
            name="conv1",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.1,
            name="bn1",
        )(x)
        x = layers.ReLU()(x)
        x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
        x = layers.MaxPooling2D(
            pool_size=3,
            strides=2,
            padding="valid",
            data_format=data_format,
        )(x)
        features.append(x)

        filters = [64, 128, 256, 512]
        for i, (blocks, filter_size) in enumerate(zip(depth, filters)):
            stride = 1 if i == 0 else 2
            x = bottle2neck_block(
                x,
                filter_size,
                f"layer{i + 1}_0",
                stride=stride,
                downsample=True,
                base_width=base_width,
                cardinality=cardinality,
                scale=scale,
                data_format=data_format,
            )

            for j in range(1, blocks):
                x = bottle2neck_block(
                    x,
                    filter_size,
                    f"layer{i + 1}_{j}",
                    base_width=base_width,
                    cardinality=cardinality,
                    scale=scale,
                    data_format=data_format,
                )
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
                    data_format=data_format,
                    name="avg_pool",
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(
                    data_format=data_format,
                    name="max_pool",
                )(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.depth = depth
        self.base_width = base_width
        self.scale = scale
        self.cardinality = cardinality
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
                "depth": self.depth,
                "base_width": self.base_width,
                "scale": self.scale,
                "cardinality": self.cardinality,
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
def Res2Net50_26w_4s(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="Res2Net50_26w_4s",
    **kwargs,
):
    model = Res2Net(
        **RES2NET_MODEL_CONFIG["Res2Net50_26w_4s"],
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

    if weights in get_all_weight_names(RES2NET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "Res2Net50_26w_4s", weights, model, RES2NET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Res2Net101_26w_4s(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="Res2Net101_26w_4s",
    **kwargs,
):
    model = Res2Net(
        **RES2NET_MODEL_CONFIG["Res2Net101_26w_4s"],
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

    if weights in get_all_weight_names(RES2NET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "Res2Net101_26w_4s", weights, model, RES2NET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Res2Net50_26w_6s(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="Res2Net50_26w_6s",
    **kwargs,
):
    model = Res2Net(
        **RES2NET_MODEL_CONFIG["Res2Net50_26w_6s"],
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

    if weights in get_all_weight_names(RES2NET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "Res2Net50_26w_6s", weights, model, RES2NET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Res2Net50_26w_8s(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="Res2Net50_26w_8s",
    **kwargs,
):
    model = Res2Net(
        **RES2NET_MODEL_CONFIG["Res2Net50_26w_8s"],
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

    if weights in get_all_weight_names(RES2NET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "Res2Net50_26w_8s", weights, model, RES2NET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Res2Net50_48w_2s(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="Res2Net50_48w_2s",
    **kwargs,
):
    model = Res2Net(
        **RES2NET_MODEL_CONFIG["Res2Net50_48w_2s"],
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

    if weights in get_all_weight_names(RES2NET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "Res2Net50_48w_2s", weights, model, RES2NET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Res2Net50_14w_8s(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="Res2Net50_14w_8s",
    **kwargs,
):
    model = Res2Net(
        **RES2NET_MODEL_CONFIG["Res2Net50_14w_8s"],
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

    if weights in get_all_weight_names(RES2NET_WEIGHTS_CONFIG):
        load_weights_from_config(
            "Res2Net50_14w_8s", weights, model, RES2NET_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Res2Next50(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="Res2Next50",
    **kwargs,
):
    model = Res2Net(
        **RES2NET_MODEL_CONFIG["Res2Next50"],
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

    if weights in get_all_weight_names(RES2NET_WEIGHTS_CONFIG):
        load_weights_from_config("Res2Next50", weights, model, RES2NET_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
