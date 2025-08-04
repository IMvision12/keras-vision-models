import keras
import numpy as np
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.layers.global_response_norm import GlobalResponseNorm
from kvmm.layers.layer_scale import LayerScale
from kvmm.layers.stochastic_depth import StochasticDepth
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import CONVNEXT_MODEL_CONFIG, CONVNEXT_WEIGHTS_CONFIG


def convnext_block(
    inputs,
    projection_dim,
    channels_axis,
    data_format,
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    name=None,
    use_grn=False,
    use_conv=False,
):
    """
    Implements a ConvNeXt block, consisting of depthwise convolution, layer normalization,
    pointwise convolutions, GELU activation, optional LayerScale, and Stochastic Depth.

    Args:
        inputs: Input tensor or layer to the block.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        projection_dim: Number of filters for the convolutions (also known as the projection dimension).
        channel_axis: axis along which the channels are defined in the input tensor.
        drop_path_rate: Drop path rate for Stochastic Depth regularization. Default is 0.0.
        layer_scale_init_value: Initial value for LayerScale scaling factor.
            If None, LayerScale is not applied. Default is 1e-6.
        name: Base name for all layers in the block. Default is None.
        use_grn: Whether to use Global Response Normalization (GRN). Default is False.
        use_conv: Whether to apply convolution in the block. Default is False.

    Returns:
        Output tensor for the block.
    """
    x = layers.DepthwiseConv2D(
        kernel_size=7,
        padding="same",
        use_bias=True,
        data_format=data_format,
        name=name + "_depthwise_conv",
    )(inputs)
    x = layers.LayerNormalization(
        axis=channels_axis, epsilon=1e-6, name=name + "_layernorm"
    )(x)
    if use_conv:
        x = layers.Conv2D(
            projection_dim * 4, 1, data_format=data_format, name=name + "_conv_1"
        )(x)
    else:
        x = layers.Dense(4 * projection_dim, name=name + "_dense_1")(x)
    x = layers.Activation("gelu", name=name + "_gelu")(x)
    if use_grn:
        x = GlobalResponseNorm(name=name + "_grn")(x)
    if use_conv:
        x = layers.Conv2D(
            projection_dim, 1, data_format=data_format, name=name + "_conv_2"
        )(x)
    else:
        x = layers.Dense(projection_dim, name=name + "_dense_2")(x)

    if layer_scale_init_value is not None:
        x = LayerScale(layer_scale_init_value, name=name + "_layer_scale")(x)

    if drop_path_rate:
        x = StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")(x)

    return layers.Add(name=name + "_add")([inputs, x])


@keras.saving.register_keras_serializable(package="kvmm")
class ConvNeXt(keras.Model):
    """Instantiates the ConvNeXt architecture.

    Reference:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (CVPR 2022)

    Args:
        depths: List of integers, specifying the number of blocks in each stage
            of the model.
        projection_dims: List of integers, specifying the number of output channels
            for each stage of the model.
        drop_path_rate: Float, the drop path rate for stochastic depth regularization.
            Defaults to `0.0`.
        layer_scale_init_value: Float, initial value for the layer scale parameter
            to stabilize training. Defaults to `1e-6`.
        use_conv: Boolean, whether to replace fully-connected layers with convolutional layers.
            Defaults to `False`.
        use_grn: Boolean, whether to use Global Response Normalization (GRN) in the model.
            Defaults to `False`.
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
        input_shape: Optional tuple specifying the shape of the input data. If not
            specified, it defaults to `(224, 224, 3)` when `include_top=True`.
        input_tensor: Optional Keras tensor (output of `layers.Input()`) to use as
            the model's input. If not provided, a new input tensor is created based
            on `input_shape`.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to `1000`.
        classifier_activation: String or callable, activation function for the top
            layer. Set to `None` to return logits. Defaults to `"linear"`.
        name: String, the name of the model. Defaults to `"ConvNeXt"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        depths,
        projection_dims,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        use_conv=False,
        use_grn=False,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="ConvNeXt",
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

        if (
            include_top
            and weights is not None
            and weights == "fb_in22k"
            and num_classes != 21841
        ):
            raise ValueError(
                f"When using 'fb_in22k' weights, num_classes must be 21841. "
                f"Received num_classes: {num_classes}"
            )

        if (
            include_top
            and weights is not None
            and weights.endswith(("in1k", "ft_in1k"))
            and num_classes != 1000
        ):
            raise ValueError(
                f"When using {weights}, num_classes must be 1000. "
                f"Received num_classes: {num_classes}"
            )

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        default_input_shape = (
            384
            if weights and "384" in weights
            else 512
            if weights and "512" in weights
            else 224
        )

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_input_shape,
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
        # Stem block
        x = layers.Conv2D(
            projection_dims[0],
            kernel_size=4,
            strides=4,
            data_format=data_format,
            name="stem_conv",
        )(x)
        x = layers.LayerNormalization(
            axis=channels_axis, epsilon=1e-6, name="stem_layernorm"
        )(x)
        features.append(x)

        depth_drop_rates = np.linspace(0.0, drop_path_rate, sum(depths))
        cur = 0
        for i in range(len(depths)):
            if i > 0:
                x = layers.LayerNormalization(
                    axis=channels_axis,
                    epsilon=1e-6,
                    name=f"stages_{i}_downsampling_layernorm",
                )(x)
                x = layers.Conv2D(
                    projection_dims[i],
                    kernel_size=2,
                    strides=2,
                    data_format=data_format,
                    name=f"stages_{i}_downsampling_conv",
                )(x)
            for j in range(depths[i]):
                x = convnext_block(
                    x,
                    projection_dim=projection_dims[i],
                    drop_path_rate=depth_drop_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    use_conv=use_conv,
                    channels_axis=channels_axis,
                    data_format=data_format,
                    name=f"stages_{i}_blocks_{j}",
                )
                if j == depths[i] - 1:
                    features.append(x)
            cur += depths[i]

        # Head
        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.LayerNormalization(
                axis=channels_axis, epsilon=1e-6, name="final_layernorm"
            )(x)
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

        self.depths = depths
        self.projection_dims = projection_dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.use_conv = use_conv
        self.use_grn = use_grn
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
                "projection_dims": self.projection_dims,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
                "use_conv": self.use_conv,
                "use_grn": self.use_grn,
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
def ConvNeXtAtto(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="d2_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtAtto",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["atto"],
        use_conv=True,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtAtto", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtFemto(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="d1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtFemto",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["femto"],
        use_conv=True,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtFemto", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtPico(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="d1_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtPico",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["pico"],
        use_conv=True,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtPico", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtNano(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="d1h_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtNano",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["nano"],
        use_conv=True,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtNano", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtTiny(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtTiny",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["tiny"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtTiny", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtSmall(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtSmall",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["small"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtSmall", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtBase(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtBase",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["base"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtBase", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtLarge(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtLarge",
    **kwargs,
):
    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["large"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtLarge", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ConvNeXtXLarge(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ConvNeXtXLarge",
    **kwargs,
):
    if weights == "in1k":
        raise ValueError(
            "The 'in1k' weight variant is not available for ConvNeXtLarge."
        )

    model = ConvNeXt(
        **CONVNEXT_MODEL_CONFIG["xlarge"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(CONVNEXT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ConvNeXtXLarge", weights, model, CONVNEXT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
