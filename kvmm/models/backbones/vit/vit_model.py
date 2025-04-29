import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import (
    AddPositionEmbs,
    ClassDistToken,
    ImageNormalizationLayer,
    LayerScale,
    MultiHeadSelfAttention,
)
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import VIT_MODEL_CONFIG, VIT_WEIGHTS_CONFIG


def mlp_block(inputs, hidden_features, out_features=None, drop=0.0, block_idx=0):
    """
    Implements a Multi-Layer Perceptron (MLP) block typically used in transformer architectures.

    The block consists of two fully connected (dense) layers with GELU activation,
    dropout regularization, and optional feature dimension specification.

    Args:
        inputs: Input tensor to the MLP block.
        hidden_features: Number of neurons in the first (hidden) dense layer.
        out_features: Number of neurons in the second (output) dense layer.
            If None, uses the same number of features as the input. Default is None.
        drop: Dropout rate applied after each dense layer. Default is 0.
        block_idx: Index of the block, used for naming layers. Default is 0.

    Returns:
        Output tensor after passing through the MLP block.
    """
    x = layers.Dense(
        hidden_features, use_bias=True, name=f"blocks_{block_idx}_dense_1"
    )(inputs)
    x = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_1")(x)
    x = layers.Dense(out_features, use_bias=True, name=f"blocks_{block_idx}_dense_2")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_2")(x)
    return x


def transformer_block(
    inputs,
    dim: int,
    num_heads: int,
    channels_axis,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    proj_drop: float = 0.0,
    attn_drop: float = 0.0,
    block_idx: int = 0,
    init_values: float = None,
):
    """
    Implements a standard Transformer block with self-attention and MLP layers.

    The block consists of two main components:
    1. Multi-Head Self-Attention layer with optional normalization
    2. Multi-Layer Perceptron (MLP) layer

    Both components use layer normalization and residual connections.

    Args:
        inputs: Input tensor to the transformer block.
        dim: Dimensionality of the input and output features.
        num_heads: Number of attention heads in the multi-head attention mechanism.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        mlp_ratio: Expansion ratio for the hidden dimension in the MLP layer.
            Hidden layer size will be `dim * mlp_ratio`. Default is 4.0.
        qkv_bias: Whether to use bias in the query, key, and value projections.
            Default is False.
        qk_norm: Whether to apply normalization to query and key before attention.
            Default is False.
        proj_drop: Dropout rate for the projection layers. Default is 0.
        attn_drop: Dropout rate for the attention probabilities. Default is 0.
        block_idx: Index of the block, used for naming layers. Default is 0.
        use_layer_scale: Whether to use LayerScale for scaling residual connections.
        init_valuess: Initial values for LayerScale weights if enabled. Default is None.

    Returns:
        Output tensor after passing through the transformer block,
        with the same shape and dimensionality as the input.
    """

    # Attention branch
    x = layers.LayerNormalization(
        epsilon=1e-6, axis=channels_axis, name=f"blocks_{block_idx}_layernorm_1"
    )(inputs)
    x = MultiHeadSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        block_prefix=f"blocks_{block_idx}",
    )(x)
    if init_values:
        x = LayerScale(
            init_values=init_values,
            name=f"blocks_{block_idx}_layerscale_1",
        )(x)
    x = keras.layers.Add(name=f"blocks_{block_idx}_add_1")([x, inputs])

    # MLP branch
    y = layers.LayerNormalization(
        epsilon=1e-6,
        axis=channels_axis,
        name=f"blocks_{block_idx}_layernorm_2",
    )(x)
    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=proj_drop,
        block_idx=block_idx,
    )
    if init_values:
        y = LayerScale(
            init_values=init_values,
            name=f"blocks_{block_idx}_layerscale_2",
        )(y)
    outputs = keras.layers.Add(name=f"blocks_{block_idx}_add_2")([x, y])
    return outputs


@keras.saving.register_keras_serializable(package="kvmm")
class VisionTransformer(keras.Model):
    """Instantiates the Vision Transformer (ViT) architecture with optional FlexiViT support.

    This implementation supports both the original ViT architecture and FlexiViT modifications,
    allowing for flexible patch sizes and dynamic position embeddings.

    References:
    - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
    - [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013)

    Args:
        patch_size: Integer, size of the patches to extract from the image.
            Defaults to `16`.
        dim: Integer, the embedding dimension for the transformer.
            Defaults to `768`.
        depth: Integer, number of transformer blocks.
            Defaults to `12`.
        num_heads: Integer, number of attention heads in each block.
            Defaults to `12`.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Defaults to `4.0`.
        qkv_bias: Boolean, whether to include bias for query, key, and value projections.
            Defaults to `True`.
        qk_norm: Boolean, whether to apply layer normalization to query and key.
            Defaults to `False`.
        drop_rate: Float, dropout rate applied to the model.
            Defaults to `0.1`.
        attn_drop_rate: Float, dropout rate applied to attention weights.
            Defaults to `0.0`.
        no_embed_class: Boolean, enables FlexiViT mode for position embeddings.
            If True, position embeddings are created only for patches (not class token),
            enabling flexible patch sizes. If False, uses standard ViT embeddings.
            Defaults to `False`.
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
        input_shape: Optional tuple specifying the shape of the input data.
            When using FlexiViT (no_embed_class=True), input shape can be flexible.
            For standard ViT, defaults to (224, 224, 3).
        input_tensor: Optional Keras tensor as input.
            Useful for connecting the model to other Keras components.
        pooling: Optional pooling mode when `include_top=False`:
            - `None`: output is the last transformer block's output
            - `"avg"`: global average pooling is applied
            - `"max"`: global max pooling is applied
        num_classes: Integer, the number of output classes for classification.
            Defaults to `1000`.
        classifier_activation: String or callable, activation function for the top layer.
            Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"ViT"`.

    Returns:
        A Keras `Model` instance.

    Example:
        ```python
        # Standard ViT
        model = ViT(
            patch_size=16,
            dim=768,
            no_embed_class=False,
            input_shape=(224, 224, 3)
        )

        # FlexiViT with dynamic patch sizes
        model = ViT(
            patch_size=16,
            dim=768,
            no_embed_class=True,
            input_shape=(240, 240, 3)  # Can use different input sizes
        )
        ```
    """

    def __init__(
        self,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        no_embed_class=False,
        use_distillation=False,
        init_values=None,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="augreg_in21k_ft_in1k_224",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=None,
        classifier_activation="softmax",
        name="ViT",
        **kwargs,
    ):
        if include_top and num_classes is None:
            raise ValueError(
                f"If `include_top` is True, `num_classes` must be specified. "
                f"Received: {num_classes}"
            )

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
            and (weights == "augreg_in21k" or weights.endswith("in21k"))
            and num_classes != 21843
        ):
            raise ValueError(
                f"When using {weights} weights, num_classes must be 21843. "
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

        if weights and "384" in weights:
            default_size = 384
        else:
            default_size = 224

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_size,
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

        grid_h = height // patch_size
        grid_w = width // patch_size

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        x = layers.Conv2D(
            filters=dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            name="conv1",
        )(x)

        x = layers.Reshape((-1, dim))(x)
        x = ClassDistToken(use_distillation=use_distillation, name="cls_token")(x)

        x = AddPositionEmbs(
            name="pos_embed",
            no_embed_class=no_embed_class,
            use_distillation=use_distillation,
            grid_h=grid_h,
            grid_w=grid_w,
        )(x)
        features.append(x)

        x = layers.Dropout(drop_rate)(x)

        for i in range(depth):
            x = transformer_block(
                x,
                dim=dim,
                num_heads=num_heads,
                channels_axis=channels_axis,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                init_values=init_values,
                block_idx=i,
            )
            features.append(x)

        x = layers.LayerNormalization(
            epsilon=1e-6, axis=channels_axis, name="final_layernorm"
        )(x)

        if include_top:
            if use_distillation:
                cls_token = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(x)
                dist_token = layers.Lambda(lambda v: v[:, 1], name="ExtractDistToken")(
                    x
                )

                cls_token = layers.Dropout(drop_rate)(cls_token)
                dist_token = layers.Dropout(drop_rate)(dist_token)

                cls_head = layers.Dense(
                    num_classes, activation=classifier_activation, name="predictions"
                )(cls_token)

                dist_head = layers.Dense(
                    num_classes,
                    activation=classifier_activation,
                    name="predictions_dist",
                )(dist_token)

                x = (cls_head + dist_head) / 2

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
                x = layers.GlobalAveragePooling1D(
                    data_format=data_format, name="avg_pool"
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling1D(data_format=data_format, name="max_pool")(
                    x
                )

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.init_values = init_values
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
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_norm": self.qk_norm,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "init_values": self.init_values,
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
def ViTTiny16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTTiny16",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTTiny16"],
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
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTTiny16", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTSmall16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTSmall16",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTSmall16"],
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
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTSmall16", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTSmall32(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTSmall32",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTSmall32"],
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
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTSmall32", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTBase16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTBase16",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTBase16"],
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
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTBase16", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTBase32(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTBase32",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTBase32"],
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
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTBase32", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTLarge16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTLarge16",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTLarge16"],
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
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTLarge16", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTLarge32(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="orig_in21k_ft_in1k_384",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTLarge32",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTLarge32"],
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
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTLarge32", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
