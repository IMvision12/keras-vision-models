import keras
from keras import backend, layers
from keras.src.applications import imagenet_utils

from kv.layers import AddPositionEmbs, ClassToken, MultiHeadSelfAttention
from kv.utils import get_all_weight_names, load_weights_from_config

from ...model_registry import register_model
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
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    proj_drop: float = 0.0,
    attn_drop: float = 0.0,
    block_idx: int = 0,
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
        mlp_ratio: Expansion ratio for the hidden dimension in the MLP layer.
            Hidden layer size will be `dim * mlp_ratio`. Default is 4.0.
        qkv_bias: Whether to use bias in the query, key, and value projections.
            Default is False.
        qk_norm: Whether to apply normalization to query and key before attention.
            Default is False.
        proj_drop: Dropout rate for the projection layers. Default is 0.
        attn_drop: Dropout rate for the attention probabilities. Default is 0.
        block_idx: Index of the block, used for naming layers. Default is 0.

    Returns:
        Output tensor after passing through the transformer block,
        with the same shape and dimensionality as the input.
    """

    # Attention branch
    x = layers.LayerNormalization(epsilon=1e-6, name=f"blocks_{block_idx}_layernorm_1")(
        inputs
    )
    x = MultiHeadSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        block_idx=block_idx,
    )(x)
    x = keras.layers.Add(name=f"blocks_{block_idx}_add_1")([x, inputs])

    # MLP branch
    y = layers.LayerNormalization(epsilon=1e-6, name=f"blocks_{block_idx}_layernorm_2")(
        x
    )
    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=proj_drop,
        block_idx=block_idx,
    )
    outputs = keras.layers.Add(name=f"blocks_{block_idx}_add_2")([x, y])
    return outputs


@keras.saving.register_keras_serializable(package="kv")
class ViT(keras.Model):
    """Instantiates the Vision Transformer (ViT) architecture.

    Reference:
    - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

    Args:
        input_shape: Tuple specifying the shape of the input data.
            Defaults to `(224, 224, 3)`.
        patch_size: Integer, size of the patches to extract from the image.
            Defaults to `16`.
        num_classes: Integer, the number of output classes for classification.
            Defaults to `1000`.
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
        representation_size: Optional integer for an intermediate dense layer before classification.
            Defaults to `None`.
        weights: String, specifying the path to pretrained weights or available options.
        include_top: Boolean, whether to include the classification head.
            Defaults to `True`.
        pooling: Optional pooling mode when `include_top=False`:
            - `None`: output is the last transformer block's output
            - `"avg"`: global average pooling is applied
            - `"max"`: global max pooling is applied
        classifier_activation: String or callable, activation function for the top layer.
            Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"ViT"`.

    Returns:
        A Keras `Model` instance.
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
        include_top=True,
        weights="imagenet",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="ViT",
        **kwargs,
    ):
        if no_embed_class:
            input_shape = imagenet_utils.obtain_input_shape(
                input_shape,
                default_size=240,
                min_size=32,
                data_format=backend.image_data_format(),
                require_flatten=include_top,
                weights=weights,
            )

        else:
            input_shape = imagenet_utils.obtain_input_shape(
                input_shape,
                default_size=224,
                min_size=32,
                data_format=backend.image_data_format(),
                require_flatten=include_top,
                weights=weights,
            )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input

        x = layers.Conv2D(
            filters=dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="conv1",
        )(img_input)

        x = layers.Reshape((-1, dim))(x)
        x = ClassToken(name="cls_token")(x)
        x = AddPositionEmbs(
            name="pos_embed",
            no_embed_class=no_embed_class,
        )(x)
        x = layers.Dropout(drop_rate)(x)

        for i in range(depth):
            x = transformer_block(
                x,
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                block_idx=i,
            )

        x = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")(x)

        # Head
        if include_top:
            x = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
            x = layers.Dropout(drop_rate)(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling1D(name="max_pool")(x)

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
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
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
            "include_top": self.include_top,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def ViTTiny16(
    include_top=True,
    weights="augreg_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTTiny16",
    **kwargs,
):
    if include_top and weights == "augreg_in21k" and num_classes != 21843:
        raise ValueError(
            f"When using 'augreg_in21k' weights, num_classes must be 21843. "
            f"Received num_classes: {num_classes}"
        )

    model = ViT(
        **VIT_MODEL_CONFIG["vit_tiny_patch16"],
        include_top=include_top,
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
    weights="augreg_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTSmall16",
    **kwargs,
):
    if include_top and weights == "augreg_in21k" and num_classes != 21843:
        raise ValueError(
            f"When using 'augreg_in21k' weights, num_classes must be 21843. "
            f"Received num_classes: {num_classes}"
        )
    model = ViT(
        **VIT_MODEL_CONFIG["vit_small_patch16"],
        include_top=include_top,
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
    weights="augreg_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTSmall32",
    **kwargs,
):
    if include_top and weights == "augreg_in21k" and num_classes != 21843:
        raise ValueError(
            f"When using 'augreg_in21k' weights, num_classes must be 21843. "
            f"Received num_classes: {num_classes}"
        )
    model = ViT(
        **VIT_MODEL_CONFIG["vit_small_patch32"],
        include_top=include_top,
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
    weights="augreg_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTBase16",
    **kwargs,
):
    if include_top and weights == "augreg_in21k" and num_classes != 21843:
        raise ValueError(
            f"When using 'augreg_in21k' weights, num_classes must be 21843. "
            f"Received num_classes: {num_classes}"
        )
    model = ViT(
        **VIT_MODEL_CONFIG["vit_base_patch16"],
        include_top=include_top,
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
    weights="augreg_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTBase32",
    **kwargs,
):
    if include_top and weights == "augreg_in21k" and num_classes != 21843:
        raise ValueError(
            f"When using 'augreg_in21k' weights, num_classes must be 21843. "
            f"Received num_classes: {num_classes}"
        )
    model = ViT(
        **VIT_MODEL_CONFIG["vit_base_patch32"],
        include_top=include_top,
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
    weights="augreg_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTLarge16",
    **kwargs,
):
    if include_top and weights == "augreg_in21k" and num_classes != 21843:
        raise ValueError(
            f"When using 'augreg_in21k' weights, num_classes must be 21843. "
            f"Received num_classes: {num_classes}"
        )
    model = ViT(
        **VIT_MODEL_CONFIG["vit_large_patch16"],
        include_top=include_top,
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
    weights="orig_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTLarge32",
    **kwargs,
):
    model = ViT(
        **VIT_MODEL_CONFIG["vit_large_patch32"],
        include_top=include_top,
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
