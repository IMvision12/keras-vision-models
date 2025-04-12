import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import (
    ImageNormalizationLayer,
    ImageToPatchesLayer,
    MultiHeadSelfAttention,
    PatchesToImageLayer,
)
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import MOBILEVIT_MODEL_CONFIG, MOBILEVIT_WEIGHTS_CONFIG


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
    expansion_ratio=1.0,
    name: str = "inverted_residual_block",
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
        expansion_ratio: Float, expansion ratio for the first 1x1 convolution. Defaults to 1.0.
        name: String, prefix for layer names in the block. Defaults to "inverted_residual_block".

    Returns:
        Output tensor after applying the inverted residual block operations.
    """
    residual_connection = (strides == 1) and (inputs.shape[channels_axis] == filters)

    x = layers.Conv2D(
        filters=make_divisible(inputs.shape[channels_axis] * expansion_ratio),
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
            padding=(1, 1),
            data_format=data_format,
            name=f"{name}_zeropadding",
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
        x = layers.Add(name=f"{name}_add")([x, inputs])

    return x


def mobilevit_block(
    inputs,
    block_dims,
    channels_axis,
    data_format,
    attention_dims=None,
    num_attention_blocks=2,
    patch_size=8,
    name="mobilevit_transformer_block",
):
    """Creates a MobileViT block that combines convolution and transformer operations.

    This block first applies convolutional layers, then converts the feature map into patches,
    processes them through transformer blocks, converts back to image space, and finally
    applies fusion convolutions. The block enables both local and global feature processing.

    Args:
        inputs: Input tensor.
        block_dims: Integer, number of output channels for the block.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: str, either 'channels_first' or 'channels_last', specifies the
            input data format.
        attention_dims: Integer, dimension of attention layers. If None, uses input channels.
        num_attention_blocks: Integer, number of transformer blocks to stack. Defaults to 2.
        patch_size: Integer, size of patches for converting image to sequences. Defaults to 8.
        name: String, prefix for layer names in the block.
            Defaults to "mobilevit_transformer_block".

    Returns:
        Output tensor after applying the MobileViT block operations.

    Notes:
        The block follows these main steps:
        1. Initial convolutional processing
        2. Image to patches conversion
        3. Transformer blocks processing (with self-attention and MLP)
        4. Patches to image conversion
        5. Final fusion with input via concatenation and convolution
    """
    if attention_dims is None:
        attention_dims = make_divisible(inputs.shape[channels_axis])

    x = inputs

    x = layers.Conv2D(
        inputs.shape[channels_axis],
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_mv_batchnorm_1"
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_1")(x)

    x = layers.Conv2D(attention_dims, 1, use_bias=False, name=f"{name}_mv_conv_2")(x)

    if data_format == "channels_first":
        h, w = x.shape[-2], x.shape[-1]
    else:
        h, w = x.shape[-3], x.shape[-2]

    unfold_layer = ImageToPatchesLayer(patch_size)
    x = unfold_layer(x)
    resize = unfold_layer.resize

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)

    for i in range(num_attention_blocks):
        residual_1 = x
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_transformer_{i}_layernorm_1"
        )(x)
        x = MultiHeadSelfAttention(
            attention_dims,
            num_heads=4,
            qkv_bias=True,
            block_prefix=f"{name}_transformer_{i}",
        )(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_1")([residual_1, x])

        residual_2 = x
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_transformer_{i}_layernorm_2"
        )(x)
        mlp_hidden_dim = int(attention_dims * 2.0)
        x = layers.Dense(
            mlp_hidden_dim,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_fc1",
        )(x)
        x = layers.Activation("swish", name=f"{name}_transformer_{i}_mlp_act")(x)
        x = layers.Dropout(0.0, name=f"{name}_transformer_{i}_mlp_drop_1")(x)
        x = layers.Dense(
            attention_dims,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_fc2",
        )(x)
        x = layers.Dropout(0.0, name=f"{name}_transformer_{i}_mlp_drop_2")(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_2")([residual_2, x])

    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name=f"{name}_layernorm")(x)

    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2))(x)

    fold_layer = PatchesToImageLayer(patch_size)
    x = fold_layer(x, original_size=(h, w), resize=resize)

    x = layers.Conv2D(
        block_dims,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_3",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv_batchnorm_2",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_2")(x)

    x = layers.Concatenate(axis=channels_axis, name=f"{name}_concat")([inputs, x])

    # Fusion convolution
    x = layers.Conv2D(
        block_dims,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_4",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv_batchnorm_3",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_3")(x)

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class MobileViT(keras.Model):
    """Instantiates the MobileViT architecture.

    MobileViT combines the benefits of CNNs and Transformers by introducing a lightweight
    transformer architecture for mobile vision tasks. It uses transformer blocks to capture
    global dependencies while maintaining the computational efficiency of CNNs.

    Reference:
    - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer]
      (https://arxiv.org/abs/2110.02178)

    Args:
        initial_dims: Integer, number of filters in the first convolutional layer.
            Defaults to 16.
        head_dims: Integer, number of filters in the final layers before classification.
            Defaults to 640.
        block_dims: List of integers, specifying the number of filters for each block.
            Defaults to [32, 64, 96, 128, 160].
        expansion_ratio: List of floats, controlling the expansion ratio in inverted
            residual blocks for each stage. Defaults to [4.0, 4.0, 4.0, 4.0, 4.0].
        attention_dims: List of integers or None, specifying the dimension of attention
            layers in transformer blocks. None means using the same dimension as input.
            Defaults to [None, None, 144, 192, 240].
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
        input_tensor: Optional Keras tensor (output of layers.Input()) to use as
            the model's input. If not provided, a new input tensor is created based
            on input_shape.
        input_shape: Optional tuple specifying the shape of the input data.
        pooling: Optional pooling mode for feature extraction when include_top=False:
            - None (default): the output is the 4D tensor from the last convolutional block.
            - "avg": global average pooling is applied, and the output is a 2D tensor.
            - "max": global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to 1000.
        classifier_activation: String or callable, activation function for the top
            layer. Set to None to return logits. Defaults to "softmax".
        name: String, the name of the model. Defaults to "MobileViT".

    Returns:
        A Keras Model instance.

    The MobileViT architecture introduces several key innovations:
    - Transformer blocks for capturing global dependencies
    - Mobile-friendly design with efficient convolutions
    - Hybrid approach combining CNNs and Transformers
    - Lightweight architecture suitable for mobile devices
    - Decomposition of feature maps into patches for transformer processing
    """

    def __init__(
        self,
        initial_dims: int = 16,
        head_dims: int = 640,
        block_dims: list = [32, 64, 96, 128, 160],
        expansion_ratio: list = [4.0, 4.0, 4.0, 4.0, 4.0],
        attention_dims: list = [None, None, 144, 192, 240],
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="cvnets_in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="MobileViT",
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

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=256,
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

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), data_format=data_format)(x)
        x = layers.Conv2D(
            initial_dims,
            kernel_size=3,
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

        for i in range(5):
            x = inverted_residual_block(
                x,
                filters=block_dims[i],
                channels_axis=channels_axis,
                data_format=data_format,
                strides=2 if i > 0 else 1,
                expansion_ratio=expansion_ratio[i],
                name=f"stages_{i}_0",
            )

            if i == 1:
                x = inverted_residual_block(
                    x,
                    filters=block_dims[i],
                    channels_axis=channels_axis,
                    data_format=data_format,
                    strides=1,
                    expansion_ratio=expansion_ratio[i],
                    name=f"stages_{i}_1",
                )
                x = inverted_residual_block(
                    x,
                    filters=block_dims[i],
                    channels_axis=channels_axis,
                    data_format=data_format,
                    strides=1,
                    expansion_ratio=expansion_ratio[i],
                    name=f"stages_{i}_2",
                )

            if i >= 2:
                x = mobilevit_block(
                    x,
                    block_dims=block_dims[i],
                    channels_axis=channels_axis,
                    data_format=data_format,
                    attention_dims=attention_dims[i],
                    num_attention_blocks=2 if i == 2 else 4 if i == 3 else 3,
                    patch_size=2,
                    name=f"stages_{i}_1",
                )

            if as_backbone:
                features.append(x)

        x = layers.Conv2D(
            head_dims,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="final_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name="final_batchnorm",
        )(x)
        x = layers.Activation("swish", name="final_act")(x)

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

        self.initial_dims = initial_dims
        self.head_dims = head_dims
        self.block_dims = block_dims
        self.expansion_ratio = expansion_ratio
        self.attention_dims = attention_dims
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
                "initial_dims": self.initial_dims,
                "head_dims": self.head_dims,
                "block_dims": self.block_dims,
                "expansion_ratio": self.expansion_ratio,
                "attention_dims": self.attention_dims,
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
def MobileViTXXS(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTXXS",
    **kwargs,
):
    model = MobileViT(
        **MOBILEVIT_MODEL_CONFIG["MobileViTXXS"],
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
    if weights in get_all_weight_names(MOBILEVIT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTXXS", weights, model, MOBILEVIT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileViTXS(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTXS",
    **kwargs,
):
    model = MobileViT(
        **MOBILEVIT_MODEL_CONFIG["MobileViTXS"],
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
    if weights in get_all_weight_names(MOBILEVIT_WEIGHTS_CONFIG):
        load_weights_from_config(
            "MobileViTXS", weights, model, MOBILEVIT_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MobileViTS(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="zero_to_one",
    weights="cvnets_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MobileViTS",
    **kwargs,
):
    model = MobileViT(
        **MOBILEVIT_MODEL_CONFIG["MobileViTS"],
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
    if weights in get_all_weight_names(MOBILEVIT_WEIGHTS_CONFIG):
        load_weights_from_config("MobileViTS", weights, model, MOBILEVIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
