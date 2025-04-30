import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import (
    ImageNormalizationLayer,
    ImageToPatchesLayer,
    PatchesToImageLayer,
)
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import MOBILEVITV2_MODEL_CONFIG, MOBILEVITV2_WEIGHTS_CONFIG


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """
    Adjusts the given value `v` to be divisible by `divisor`,
        ensuring it meets the specified constraints.

    Args:
        v (int or float): The value to be adjusted.
        divisor (int, optional): The divisor to which `v` should be rounded. Default is 8.
        min_value (int, optional): The minimum allowed value. If None, it defaults to `divisor`.
        round_limit (float, optional): The threshold to increase `new_v` if it is too small.
            Default is 0.9.

    Returns:
        int: The adjusted value that is divisible by `divisor` and meets the
            given constraints.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def inverted_residual_block(
    inputs,
    filters,
    channels_axis,
    data_format,
    strides=1,
    expansion_ratio=2.0,
    name="inverted_residual_block",
):
    """Creates an inverted residual block as used in MobileNetV2 and MobileViT architectures.

    This block consists of an expansion 1x1 conv, a depthwise 3x3 conv, and a projection
    1x1 conv. If input/output dimensions match and stride is 1, a residual connection is added.

    Args:
        inputs: Input tensor.
        filters: Number of output filters.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: str, either 'channels_first' or 'channels_last', specifies the
            input data format.
        strides: Integer, stride size for the depthwise convolution. Defaults to 1.
        expansion_ratio: Float, expansion ratio for the first 1x1 convolution. Defaults to 2.0.
        name: String, prefix for layer names in the block. Defaults to "inverted_residual_block".

    Returns:
        Output tensor after applying the inverted residual block operations.
    """
    residual_connection = (strides == 1) and (inputs.shape[channels_axis] == filters)

    x = layers.Conv2D(
        make_divisible(inputs.shape[channels_axis] * expansion_ratio),
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_conv_1",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_1",
    )(x)
    x = layers.Activation("swish", name=f"{name}_ir_act_1")(x)

    if strides > 1:
        x = layers.ZeroPadding2D(
            padding=1,
            data_format=data_format,
            name=f"{name}_ir_zeropadding",
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_2",
    )(x)
    x = layers.Activation("swish", name=f"{name}_ir_act_2")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_conv_2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_3",
    )(x)

    if residual_connection:
        x = layers.Add(name=f"{name}_ir_add")([x, inputs])

    return x


def linear_self_attention(
    inputs, dim, data_format, use_bias=True, name="linear_self_attention"
):
    """Creates a linear self-attention block.

    This block applies self-attention mechanisms to the input tensor, using a combination
    of convolutional layers, softmax, and element-wise multiplication.

    Args:
        inputs: Input tensor.
        dim: Integer, dimension of the key and value tensors.
        data_format: String, either 'channels_first' or 'channels_last', specifies the
            input data format.
        use_bias: Boolean, whether to use bias in the convolutional layers. Defaults to True.
        name: String, prefix for layer names in the block. Defaults to "linear_self_attention".

    Returns:
        Output tensor after applying the linear self-attention operations.
    """
    num_patch_axis = -2 if data_format == "channels_last" else -1

    x = layers.Conv2D(1 + (2 * dim), 1, use_bias=use_bias, name=f"{name}_attn_conv_1")(
        inputs
    )

    if data_format == "channels_last":
        query = x[..., :1]
        key = x[..., 1 : dim + 1]
        value = x[..., dim + 1 :]
    else:
        query = x[:, :1]
        key = x[:, 1 : dim + 1]
        value = x[:, dim + 1 :]

    context_scores = layers.Softmax(axis=num_patch_axis, name=f"{name}_attn_softmax")(
        query
    )
    context_vector = layers.Multiply(name=f"{name}_attn_multiply_1")(
        [key, context_scores]
    )
    context_vector = keras.ops.sum(context_vector, axis=num_patch_axis, keepdims=True)

    out = layers.ReLU(name=f"{name}_attn_relu")(value)
    out = layers.Multiply(name=f"{name}_attn_multiply_2")([out, context_vector])
    out = layers.Conv2D(dim, 1, use_bias=use_bias, name=f"{name}_attn_conv_2")(out)

    return out


def mobilevitv2_block(
    inputs,
    block_dims,
    channels_axis,
    data_format,
    kernel_size=3,
    expansion_ratio=2.0,
    transformer_dim=None,
    transformer_depth=2,
    patch_size=2,
    name="mobilevitv2_block",
):
    """Creates a MobileViTv2 block, which combines convolutions with transformer layers.

    This block consists of depthwise convolution, pointwise convolution, unfolding into patches,
    multiple transformer layers, and folding back to the original dimensions. It helps in capturing
    both local and global features efficiently.

    Args:
        inputs: Input tensor.
        block_dims: Number of output filters for the final convolution.
        channels_axis: Integer, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: String, either 'channels_first' or 'channels_last', specifies the
            input data format.
        kernel_size: Integer, size of the depthwise convolution kernel. Defaults to 3.
        expansion_ratio: Float, expansion ratio for the first 1x1 convolution. Defaults to 2.0.
        transformer_dim: Integer, dimension of the transformer layers. If None, it is calculated
            based on the input dimensions and expansion ratio. Defaults to None.
        transformer_depth: Integer, number of transformer layers to apply. Defaults to 2.
        patch_size: Integer, size of the patches for transformer layers. Defaults to 2.
        name: String, prefix for layer names in the block. Defaults to "mobilevitv2_block".

    Returns:
        Output tensor after applying the MobileViTv2 block operations.
    """
    transformer_dim = transformer_dim or make_divisible(
        inputs.shape[channels_axis] * expansion_ratio
    )

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_dwconv",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv2_batchnorm_1",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mc2_act_1")(x)

    x = layers.Conv2D(
        transformer_dim,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_conv_1",
    )(x)

    if data_format == "channels_first":
        h, w = x.shape[-2], x.shape[-1]
    else:
        h, w = x.shape[-3], x.shape[-2]

    unfold_layer = ImageToPatchesLayer(patch_size)
    x = unfold_layer(x)
    resize = unfold_layer.resize

    for i in range(transformer_depth):
        residual = x
        x = layers.GroupNormalization(
            1,
            axis=channels_axis,
            epsilon=1e-5,
            name=f"{name}_transformer_{i}_groupnorm_1",
        )(x)
        x = linear_self_attention(
            x,
            transformer_dim,
            data_format,
            use_bias=True,
            name=f"{name}_transformer_{i}",
        )
        x = layers.Add(name=f"{name}_transformer_{i}_add_1")([residual, x])

        residual = x
        x = layers.GroupNormalization(
            1,
            axis=channels_axis,
            epsilon=1e-5,
            name=f"{name}_transformer_{i}_groupnorm_2",
        )(x)
        mlp_hidden_dim = int(transformer_dim * 2.0)

        x = layers.Conv2D(
            mlp_hidden_dim,
            1,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_conv_1",
        )(x)
        x = layers.Activation("swish", name=f"{name}_transformer_{i}_mlp_act")(x)
        x = layers.Conv2D(
            transformer_dim,
            1,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_conv_2",
        )(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_2")([residual, x])

    x = layers.GroupNormalization(
        1,
        axis=channels_axis,
        epsilon=1e-5,
        name=f"{name}_groupnorm",
    )(x)

    fold_layer = PatchesToImageLayer(patch_size)
    x = fold_layer(x, original_size=(h, w), resize=resize)

    x = layers.Conv2D(
        block_dims,
        1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_proj_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv2_proj_batchnorm",
    )(x)

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class MobileViTV2(keras.Model):
    """Instantiates the MobileViTV2 architecture.

    MobileViTV2 is a lightweight vision transformer architecture designed for mobile and
    edge devices. It combines the efficiency of convolutional neural networks (CNNs) with
    the capability of transformers to capture long-range dependencies.

    Reference:
    - [MobileViTV2: Light-weight, General-purpose, and Mobile-friendly Vision Transformer]
      (https://arxiv.org/abs/2208.01563)

    Args:
        multiplier: Float, a multiplier for the number of filters in each layer.
            Defaults to 1.0.
        include_top: Boolean, whether to include the classification head at the top
            of the network. Defaults to True.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to False.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to True.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options. Defaults to "cvnets_in1k".
        input_shape: Optional tuple specifying the shape of the input data.
        input_tensor: Optional Keras tensor (output of layers.Input()) to use as
            the model's input. If not provided, a new input tensor is created based
            on input_shape.
        pooling: Optional pooling mode for feature extraction when include_top=False:
            - None (default): the output is the 4D tensor from the last convolutional block.
            - "avg": global average pooling is applied, and the output is a 2D tensor.
            - "max": global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to 1000.
        classifier_activation: String or callable, activation function for the top
            layer. Set to None to return logits. Defaults to "softmax".
        name: String, the name of the model. Defaults to "MobileViTV2".

    Returns:
        A Keras Model instance.

    The MobileViTV2 architecture introduces several key innovations:
    - Efficient transformer blocks for capturing global dependencies
    - Mobile-friendly design with efficient convolutions
    - Hybrid approach combining CNNs and Transformers
    - Lightweight architecture suitable for mobile and edge devices
    - Patching mechanism to process feature maps with transformers
    """

    def __init__(
        self,
        multiplier=1.0,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="zero_to_one",
        weights="cvnets_in1k_256",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="MobileViTV2",
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
        channels_axis = -1 if data_format == "channels_last" else -3

        if weights and "384" in weights:
            default_img_size = 384
        else:
            default_img_size = 256

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_img_size,
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

        x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
        x = layers.Conv2D(
            int(32 * multiplier),
            3,
            strides=2,
            padding="valid",
            use_bias=False,
            data_format=data_format,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name="stem_batchnorm",
        )(x)
        x = layers.Activation("swish", name="stem_act")(x)
        features.append(x)

        for stage in range(5):
            channels = int(([64, 128, 256, 384, 512][stage]) * multiplier)
            stride = 1 if stage == 0 else 2

            x = inverted_residual_block(
                x,
                channels,
                channels_axis,
                data_format,
                strides=stride,
                expansion_ratio=2.0,
                name=f"stages_{stage}_0",
            )

            if stage <= 1:
                if stage == 1:
                    x = inverted_residual_block(
                        x,
                        channels,
                        channels_axis,
                        data_format,
                        strides=1,
                        expansion_ratio=2.0,
                        name=f"stages_{stage}_1",
                    )
            else:
                x = mobilevitv2_block(
                    x,
                    channels,
                    channels_axis,
                    data_format,
                    kernel_size=3,
                    expansion_ratio=0.5,
                    transformer_depth=[2, 4, 3][stage - 2],
                    patch_size=2,
                    name=f"stages_{stage}_1",
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

        self.multiplier = multiplier
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
                "multiplier": self.multiplier,
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
def MobileViTV2M050(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k_256",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTV2M050",
    **kwargs,
):
    model = MobileViTV2(
        **MOBILEVITV2_MODEL_CONFIG["MobileViTV2M050"],
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
    if weights in get_all_weight_names(MOBILEVITV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTV2M050", weights, model, MOBILEVITV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileViTV2M075(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k_256",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTV2M075",
    **kwargs,
):
    model = MobileViTV2(
        **MOBILEVITV2_MODEL_CONFIG["MobileViTV2M075"],
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
    if weights in get_all_weight_names(MOBILEVITV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTV2M075", weights, model, MOBILEVITV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileViTV2M100(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k_256",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTV2M100",
    **kwargs,
):
    model = MobileViTV2(
        **MOBILEVITV2_MODEL_CONFIG["MobileViTV2M100"],
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
    if weights in get_all_weight_names(MOBILEVITV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTV2M100", weights, model, MOBILEVITV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileViTV2M125(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k_256",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTV2M125",
    **kwargs,
):
    model = MobileViTV2(
        **MOBILEVITV2_MODEL_CONFIG["MobileViTV2M125"],
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
    if weights in get_all_weight_names(MOBILEVITV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTV2M125", weights, model, MOBILEVITV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileViTV2M150(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k_256",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTV2M150",
    **kwargs,
):
    model = MobileViTV2(
        **MOBILEVITV2_MODEL_CONFIG["MobileViTV2M150"],
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
    if weights in get_all_weight_names(MOBILEVITV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTV2M150", weights, model, MOBILEVITV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileViTV2M175(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k_256",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTV2M175",
    **kwargs,
):
    model = MobileViTV2(
        **MOBILEVITV2_MODEL_CONFIG["MobileViTV2M175"],
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
    if weights in get_all_weight_names(MOBILEVITV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTV2M175", weights, model, MOBILEVITV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileViTV2M200(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k_256",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTV2M200",
    **kwargs,
):
    model = MobileViTV2(
        **MOBILEVITV2_MODEL_CONFIG["MobileViTV2M200"],
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
    if weights in get_all_weight_names(MOBILEVITV2_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTV2M200", weights, model, MOBILEVITV2_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
