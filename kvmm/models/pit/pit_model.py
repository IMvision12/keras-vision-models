import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import (
    AddPositionEmbs,
    ClassDistToken,
    ImageNormalizationLayer,
    MultiHeadSelfAttention,
)
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import PIT_MODEL_CONFIG, PIT_WEIGHTS_CONFIG


def mlp_block(inputs, hidden_features, out_features=None, drop=0.0, block_prefix=None):
    """
    Implements a Multi-Layer Perceptron (MLP) block typically used in transformer architectures.

    The block consists of two fully connected (dense) layers with GELU activation,
    dropout regularization, and optional feature dimension specification.

    Args:
        inputs: Input tensor to the MLP block.
        hidden_features: Number of neurons in the first (hidden) dense layer.
        out_features: Number of neurons in the second (output) dense layer.
            If None, uses the same number of features as the input.
        drop: Dropout rate applied after each dense layer. Default is 0.0.
        block_prefix: String prefix used for naming layers. Default is None.

    Returns:
        Output tensor after passing through the MLP block.
    """
    x = layers.Dense(hidden_features, use_bias=True, name=block_prefix + "_dense_1")(
        inputs
    )
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(out_features, use_bias=True, name=block_prefix + "_dense_2")(x)
    x = layers.Dropout(drop)(x)
    return x


def transformer_block(
    inputs,
    dim,
    num_heads,
    mlp_ratio,
    channels_axis,
    block_prefix=None,
):
    """
    Implements a standard Transformer block with self-attention and MLP layers.

    The block consists of two main components:
    1. Multi-Head Self-Attention layer with layer normalization and residual connection
    2. Multi-Layer Perceptron (MLP) layer with layer normalization and residual connection

    Args:
        inputs: Input tensor to the transformer block.
        dim: Dimensionality of the input and output features.
        num_heads: Number of attention heads in the multi-head attention mechanism.
        mlp_ratio: Expansion ratio for the hidden dimension in the MLP layer.
            Hidden layer size will be `dim * mlp_ratio`.
        channels_axis: Axis along which the channels are defined.
        block_prefix: String prefix used for naming layers. Default is None.

    Returns:
        Output tensor after passing through the transformer block,
        with the same shape and dimensionality as the input.
    """
    x = layers.LayerNormalization(
        epsilon=1e-6, axis=channels_axis, name=block_prefix + "_layernorm_1"
    )(inputs)

    x = MultiHeadSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=block_prefix.replace("pit", "transformers"),
    )(x)

    x = layers.Add()([inputs, x])

    y = layers.LayerNormalization(
        epsilon=1e-6, axis=channels_axis, name=block_prefix + "_layernorm_2"
    )(x)

    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        block_prefix=block_prefix,
    )

    outputs = layers.Add()([x, y])
    return outputs


def conv_pooling(
    x,
    nb_tokens,
    in_channels,
    out_channels,
    stride,
    data_format,
    block_prefix,
):
    """
    Implements a convolutional pooling operation for transforming token representations.

    This function performs pooling on both token and spatial representations separately,
    then combines them. It includes:
    1. Separate processing for class/distillation tokens and spatial tokens
    2. Convolution-based downsampling of spatial representations
    3. Linear projection of class/distillation tokens
    4. Concatenation of processed tokens

    Args:
        x: Tuple of (input_tensor, (height, width)) where input_tensor contains
            the token representations and (height, width) are spatial dimensions.
        nb_tokens: Number of special tokens (e.g., class token, distillation token)
            at the beginning of the sequence.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride value for the convolution operation, determines
            the downsampling factor.
        data_format: String, either 'channels_first' or 'channels_last'.
        block_prefix: String prefix used for naming layers.

    Returns:
        Tuple of (output, (new_height, new_width)) where output is the processed
        and concatenated tokens, and (new_height, new_width) are the new spatial
        dimensions after pooling.
    """

    input_tensor, (height, width) = x
    tokens = input_tensor[:, :nb_tokens]
    spatial = input_tensor[:, nb_tokens:]

    new_height = (height + stride - 1) // stride
    new_width = (width + stride - 1) // stride

    if data_format == "channels_first":
        spatial = layers.Reshape((in_channels, height, width))(spatial)
    else:
        spatial = layers.Reshape((height, width, in_channels))(spatial)
    spatial = layers.ZeroPadding2D(data_format=data_format, padding=stride // 2)(
        spatial
    )
    spatial = layers.Conv2D(
        filters=out_channels,
        kernel_size=stride + 1,
        strides=stride,
        groups=in_channels,
        data_format=data_format,
        name=block_prefix + "_conv",
    )(spatial)

    tokens = layers.Dense(units=out_channels, name=block_prefix + "_dense")(tokens)
    if data_format == "channels_first":
        spatial = layers.Reshape((out_channels, new_height * new_width))(spatial)
        spatial = layers.Permute((2, 1))(spatial)
    else:
        spatial = layers.Reshape((new_height * new_width, out_channels))(spatial)
    output = layers.Concatenate(axis=1)([tokens, spatial])

    return output, (new_height, new_width)


@keras.saving.register_keras_serializable(package="kvmm")
class PoolingVisionTransformer(keras.Model):
    """Instantiates the Pooling Vision Transformer architecture.

    This implementation provides a hierarchical vision transformer that uses pooling
    to progressively reduce spatial dimensions while increasing the channel dimension
    through multiple stages.

    Args:
        patch_size: Integer, size of the patches to extract from the image.
            Defaults to `16`.
        stride: Integer, stride size for patch extraction.
            Defaults to `8`.
        embed_dim: Tuple of integers, embedding dimensions for each transformer stage.
            Defaults to `(64, 128, 256)`.
        depth: Tuple of integers, number of transformer blocks in each stage.
            Defaults to `(2, 6, 4)`.
        heads: Tuple of integers, number of attention heads in each stage.
            Defaults to `(2, 4, 8)`.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Defaults to `4.0`.
        distilled: Boolean, whether to use distillation tokens.
            Defaults to `False`.
        drop_rate: Float, dropout rate applied to the model.
            Defaults to `0.0`.
        include_top: Boolean, whether to include the classification head.
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
        weights: String, specifying the path to pretrained weights or one of the
            available options in `keras-vision`.
        input_tensor: Optional Keras tensor as input.
            Useful for connecting the model to other Keras components.
        input_shape: Optional tuple specifying the shape of the input data.
        pooling: Optional pooling mode when `include_top=False`:
            - `None`: output is the last transformer block's output
            - `"avg"`: global average pooling is applied
            - `"max"`: global max pooling is applied
        num_classes: Integer, the number of output classes for classification.
            Defaults to `1000`.
        classifier_activation: String or callable, activation function for the top layer.
            Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"PoolingVisionTransformer"`.

    Returns:
        A Keras `Model` instance.

    Example:
        ```python
        # Create a basic pooling vision transformer
        model = PoolingVisionTransformer(
            patch_size=16,
            stride=8,
            embed_dim=(64, 128, 256),
            depth=(2, 6, 4),
            heads=(2, 4, 8)
        )

        # Create a pooling vision transformer with distillation
        model = PoolingVisionTransformer(
            patch_size=16,
            stride=8,
            embed_dim=(64, 128, 256),
            depth=(2, 6, 4),
            heads=(2, 4, 8),
            distilled=True
        )
        ```
    """

    def __init__(
        self,
        patch_size=16,
        stride=8,
        embed_dim=(64, 128, 256),
        depth=(2, 6, 4),
        heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=False,
        drop_rate=0.0,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="PoolingVisionTransformer",
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

        if data_format == "channels_first":
            if len(input_shape) == 3:
                _, height, width = input_shape
            else:
                height, width = input_shape[1:]
        else:  # channels_last
            if len(input_shape) == 3:
                height, width, _ = input_shape
            else:
                height, width = input_shape[:2]

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        x = layers.Conv2D(
            filters=embed_dim[0],
            kernel_size=patch_size,
            strides=stride,
            data_format=data_format,
            name="patch_embed_conv",
        )(x)

        grid_h = (height - patch_size) // stride + 1
        grid_w = (width - patch_size) // stride + 1
        input_size = (grid_h, grid_w)

        x = layers.Reshape(
            (grid_h * grid_w, embed_dim[0]), name="patch_tokens_reshape"
        )(x)

        x = AddPositionEmbs(
            grid_h=grid_h,
            grid_w=grid_w,
            no_embed_class=True,
            use_distillation=distilled,
            name="pos_embed",
        )(x)

        x = ClassDistToken(
            use_distillation=distilled,
            combine_tokens=True,
            name="class_dist_token",
        )(x)

        features.append(x)

        x = layers.Dropout(drop_rate, name="pos_drop")(x)

        for stage_idx in range(len(depth)):
            for block_idx in range(depth[stage_idx]):
                x = transformer_block(
                    x,
                    dim=embed_dim[stage_idx],
                    num_heads=heads[stage_idx],
                    mlp_ratio=mlp_ratio,
                    channels_axis=channels_axis,
                    block_prefix=f"pit_{stage_idx}_blocks_{block_idx}",
                )

            if stage_idx < len(depth) - 1:
                x, input_size = conv_pooling(
                    (x, input_size),
                    nb_tokens=2 if distilled else 1,
                    in_channels=embed_dim[stage_idx],
                    out_channels=embed_dim[stage_idx + 1],
                    stride=2,
                    data_format=data_format,
                    block_prefix=f"pit_{stage_idx + 1}_pool",
                )

            features.append(x)

        x = x[:, : 2 if distilled else 1]
        x = layers.LayerNormalization(epsilon=1e-6, axis=channels_axis, name="norm")(x)

        if include_top:
            if distilled:
                cls_token = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(x)
                dist_token = layers.Lambda(lambda v: v[:, 1], name="ExtractDistToken")(
                    x
                )
                cls_token = layers.Dropout(drop_rate)(cls_token)
                dist_token = layers.Dropout(drop_rate)(dist_token)

                cls_head = layers.Dense(num_classes, name="predictions")(cls_token)
                dist_head = layers.Dense(num_classes, name="predictions_dist")(
                    dist_token
                )
                x = layers.Average()([cls_head, dist_head])
                if classifier_activation is not None:
                    x = layers.Activation(
                        classifier_activation, name="predictions_activation"
                    )(x)
            else:
                x = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
                x = layers.Dropout(drop_rate)(x)
                x = layers.Dense(
                    num_classes, activation=classifier_activation, name="predictions"
                )(x)
        elif as_backbone:
            x = features
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling1D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        # Save configuration
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled
        self.drop_rate = drop_rate
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
                "patch_size": self.patch_size,
                "stride": self.stride,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "heads": self.heads,
                "mlp_ratio": self.mlp_ratio,
                "distilled": self.distilled,
                "drop_rate": self.drop_rate,
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
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Model variants
@register_model
def PiT_XS(
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
    name="PiT_XS",
    **kwargs,
):
    model = PoolingVisionTransformer(
        **PIT_MODEL_CONFIG["PiT_XS"],
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

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config("PiT_XS", weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_XS_Distilled(
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
    name="PiT_XS_Distilled",
    **kwargs,
):
    model = PoolingVisionTransformer(
        **PIT_MODEL_CONFIG["PiT_XS_Distilled"],
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

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config("PiT_XS_Distilled", weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_Ti(
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
    name="PiT_Ti",
    **kwargs,
):
    model = PoolingVisionTransformer(
        **PIT_MODEL_CONFIG["PiT_Ti"],
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

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config("PiT_Ti", weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_Ti_Distilled(
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
    name="PiT_Ti_Distilled",
    **kwargs,
):
    model = PoolingVisionTransformer(
        **PIT_MODEL_CONFIG["PiT_Ti_Distilled"],
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

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config("PiT_Ti_Distilled", weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_S(
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
    name="PiT_S",
    **kwargs,
):
    model = PoolingVisionTransformer(
        **PIT_MODEL_CONFIG["PiT_S"],
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

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config("PiT_S", weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_S_Distilled(
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
    name="PiT_S_Distilled",
    **kwargs,
):
    model = PoolingVisionTransformer(
        **PIT_MODEL_CONFIG["PiT_S_Distilled"],
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

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config("PiT_S_Distilled", weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_B(
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
    name="PiT_B",
    **kwargs,
):
    model = PoolingVisionTransformer(
        **PIT_MODEL_CONFIG["PiT_B"],
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

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config("PiT_B", weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_B_Distilled(
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
    name="PiT_B_Distilled",
    **kwargs,
):
    model = PoolingVisionTransformer(
        **PIT_MODEL_CONFIG["PiT_B_Distilled"],
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

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config("PiT_B_Distilled", weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
