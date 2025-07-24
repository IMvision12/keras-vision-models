import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer, LayerScale
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import INCEPTION_NEXT_MODEL_CONFIG, INCEPTION_NEXT_WEIGHTS_CONFIG


def inception_dwconv2d(
    x,
    square_kernel_size=3,
    band_kernel_size=11,
    branch_ratio=0.125,
    data_format=None,
    channels_axis=None,
    name="token_mixer",
):
    """Implements the InceptionNeXt token mixer with parallel convolutional pathways.

    Args:
        x: Input tensor.
        square_kernel_size: Integer, size of the square kernel for local spatial mixing.
            Defaults to 3.
        band_kernel_size: Integer, size of the band kernels (horizontal and vertical)
            for capturing wide-range spatial dependencies. Defaults to 11.
        branch_ratio: Float, ratio determining the number of channels allocated to each
            specialized convolution branch. Defaults to 0.125.
        data_format: String, either 'channels_last' or 'channels_first'.
            Specifies the input data format.
        channels_axis: Integer, axis along which the channels are defined.
            (-1 for 'channels_last', 1 for 'channels_first')
        name: String, prefix for naming the layers. Defaults to "token_mixer".

    Returns:
        Tensor with same shape as input after applying parallel convolutions
        and concatenating results.
    """
    input_channels = x.shape[channels_axis]
    branch_channels = int(input_channels * branch_ratio)
    split_sizes = [input_channels - 3 * branch_channels] + [branch_channels] * 3
    split_indices = [sum(split_sizes[: i + 1]) for i in range(len(split_sizes) - 1)]

    def calculate_padding(kernel_size):
        return (kernel_size - 1) // 2

    square_padding, band_padding = (
        calculate_padding(square_kernel_size),
        calculate_padding(band_kernel_size),
    )

    x_splits = keras.ops.split(x, split_indices, axis=channels_axis)
    x_id, *x_branches = x_splits

    conv_configs = [
        (square_kernel_size, square_padding, f"{name}_dwconv_hw"),
        ((1, band_kernel_size), (0, band_padding), f"{name}_dwconv_w"),
        ((band_kernel_size, 1), (band_padding, 0), f"{name}_dwconv_h"),
    ]

    x = [
        layers.DepthwiseConv2D(
            kernel, use_bias=True, data_format=data_format, name=name
        )(layers.ZeroPadding2D(padding)(x))
        for (kernel, padding, name), x in zip(conv_configs, x_branches)
    ]

    return layers.Concatenate(axis=channels_axis)([x_id, *x])


def inception_next_block(
    x,
    num_filter,
    mlp_ratio=4.0,
    dropout_rate=0.0,
    layer_scale_init_value=1e-6,
    band_kernel_size=11,
    branch_ratio=0.125,
    data_format=None,
    channels_axis=None,
    name="blocks",
):
    """Applies a complete InceptionNeXt block combining token mixing and channel mixing.

    Args:
        x: Input tensor.
        num_filter: Integer, number of output filters for the block.
        mlp_ratio: Float, expansion ratio for the MLP hidden dimension.
            Defaults to 4.0.
        dropout_rate: Float between 0 and 1, dropout rate applied after
            each dense layer. Defaults to 0.0.
        layer_scale_init_value: Float, initial value for the layer scale
            parameter. Helps stabilize training of deep networks.
            Defaults to 1e-6.
        data_format: String, either 'channels_last' or 'channels_first'.
            Specifies the input data format.
        channels_axis: Integer, axis along which the channels are defined.
            (-1 for 'channels_last', 1 for 'channels_first')
        name: String, prefix for naming the layers. Defaults to "blocks".

    Returns:
        Output tensor after applying the complete InceptionNeXt block.
        Shape is same as input tensor.

    """
    x_input = x

    x = inception_dwconv2d(
        x,
        band_kernel_size=band_kernel_size,
        branch_ratio=branch_ratio,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_token_mixer",
    )

    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_batchnorm"
    )(x)

    x = layers.Conv2D(
        int(num_filter * mlp_ratio),
        1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv1",
    )(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(
        num_filter, 1, use_bias=True, data_format=data_format, name=f"{name}_conv2"
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = LayerScale(layer_scale_init_value, name=f"{name}_gamma")(x)
    x = layers.Add()([x, x_input])

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class InceptionNeXt(keras.Model):
    """Instantiates the InceptionNeXt architecture.

    Reference:
    - [InceptionNeXt: When Inception Meets ConvNeXt](
        https://arxiv.org/abs/2303.16900)

    Args:
        depths: List of integers, specifying the number of blocks in each stage.
            Defaults to [3, 3, 9, 3].
        num_filters: List of integers, specifying the number of filters for each stage.
            Defaults to [96, 192, 384, 768].
        mlp_ratios: List of integers, specifying the MLP ratio for each stage.
            Defaults to [4, 4, 4, 3].
        include_top: Boolean, whether to include the classification head at the top
            of the network. Defaults to `True`.
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
            layer. Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"InceptionNeXt"`.

    Returns:
        A Keras `Model` instance.

    The InceptionNeXt architecture combines design principles from the Inception family
    and ConvNeXt models. It introduces an efficient token mixer that processes spatial
    information through parallel pathways with different receptive fields, enabling
    better feature extraction at multiple scales. The model maintains computational
    efficiency while achieving strong performance on image classification tasks.
    """

    def __init__(
        self,
        depths=[3, 3, 9, 3],
        num_filters=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 3],
        band_kernel_size=11,
        branch_ratio=0.125,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="inceptioon",
        weights="sail_in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="InceptionNeXt",
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

        # Stem
        x = layers.Conv2D(
            num_filters[0],
            4,
            4,
            use_bias=True,
            data_format=data_format,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_batchnorm"
        )(x)
        features.append(x)

        current_stride = 4
        for i in range(len(depths)):
            strides = 2 if i > 0 else 1

            if strides > 1:
                x = layers.BatchNormalization(
                    axis=channels_axis,
                    momentum=0.9,
                    epsilon=1e-5,
                    name=f"stages_{i}_downsample_batchnorm",
                )(x)
                x = layers.Conv2D(
                    num_filters[i],
                    2,
                    strides,
                    use_bias=True,
                    data_format=data_format,
                    name=f"stages_{i}_downsample_conv",
                )(x)

            for j in range(depths[i]):
                x = inception_next_block(
                    x,
                    num_filter=num_filters[i],
                    mlp_ratio=mlp_ratios[i],
                    band_kernel_size=band_kernel_size,
                    branch_ratio=branch_ratio,
                    data_format=data_format,
                    channels_axis=channels_axis,
                    name=f"stages_{i}_blocks_{j}",
                )

            current_stride *= strides
            features.append(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.Dense(int(num_filters[-1] * 3.0), use_bias=True, name="head_fc")(
                x
            )
            x = layers.Activation("gelu")(x)
            x = layers.LayerNormalization(epsilon=1e-6, name="head_batchnorm")(x)
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

        self.depths = depths
        self.num_filters = num_filters
        self.mlp_ratios = mlp_ratios
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
                "depths": self.depths,
                "num_filters": self.num_filters,
                "mlp_ratios": self.mlp_ratios,
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
def InceptionNeXtAtto(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="inception",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="InceptionNeXtAtto",
    **kwargs,
):
    model = InceptionNeXt(
        **INCEPTION_NEXT_MODEL_CONFIG["InceptionNeXtAtto"],
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

    if weights in get_all_weight_names(INCEPTION_NEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "InceptionNeXtAtto", weights, model, INCEPTION_NEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def InceptionNeXtTiny(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="inception",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="InceptionNeXtTiny",
    **kwargs,
):
    model = InceptionNeXt(
        **INCEPTION_NEXT_MODEL_CONFIG["InceptionNeXtTiny"],
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

    if weights in get_all_weight_names(INCEPTION_NEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "InceptionNeXtTiny", weights, model, INCEPTION_NEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def InceptionNeXtSmall(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="inception",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="InceptionNeXtSmall",
    **kwargs,
):
    model = InceptionNeXt(
        **INCEPTION_NEXT_MODEL_CONFIG["InceptionNeXtSmall"],
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

    if weights in get_all_weight_names(INCEPTION_NEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "InceptionNeXtSmall", weights, model, INCEPTION_NEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def InceptionNeXtBase(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="inception",
    weights="sail_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="InceptionNeXtBase",
    **kwargs,
):
    model = InceptionNeXt(
        **INCEPTION_NEXT_MODEL_CONFIG["InceptionNeXtBase"],
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

    if weights in get_all_weight_names(INCEPTION_NEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "InceptionNeXtBase", weights, model, INCEPTION_NEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
