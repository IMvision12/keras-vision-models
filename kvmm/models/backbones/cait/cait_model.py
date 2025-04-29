import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import (
    AddPositionEmbs,
    ClassAttention,
    ClassDistToken,
    ImageNormalizationLayer,
    LayerScale,
    StochasticDepth,
    TalkingHeadAttention,
)
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import CAIT_MODEL_CONFIG, CAIT_WEIGHTS_CONFIG


def mlp_block(x, hidden_dim, out_dim, drop_rate=0.0, block_prefix=None):
    """A multilayer perceptron block with two dense layers.

    Args:
        x: input tensor.
        hidden_dim: int, the number of units in the first dense layer.
        out_dim: int, the number of units in the second dense layer.
        drop_rate: float, dropout rate to be applied after each dense layer.
            Default is 0.0 (no dropout).
        block_prefix: string or None, optional prefix for layer names.
            If None, no name prefix is used.

    Returns:
        Output tensor for the block.
    """
    x = layers.Dense(
        hidden_dim,
        activation="gelu",
        name=f"{block_prefix}_dense_1" if block_prefix else None,
    )(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(out_dim, name=f"{block_prefix}_dense_2" if block_prefix else None)(
        x
    )
    x = layers.Dropout(drop_rate)(x)
    return x


def LayerScaleBlockTalkingHeadAttn(
    x,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    drop_rate=0.0,
    init_values=1e-5,
    block_prefix="block",
):
    """A transformer block with layer scaling and talking head attention.

    This block implements a transformer architecture with layer normalization,
    talking head attention, MLP, layer scaling, and residual connections with
    optional stochastic depth for regularization.

    Args:
        x: input tensor.
        embed_dim: int, the embedding dimension for the block.
        num_heads: int, the number of attention heads to use.
        mlp_ratio: float, ratio that determines the hidden dimension size of the MLP
            relative to the embedding dimension. Default is 4.0.
        drop_rate: float, dropout rate for stochastic depth. If 0, no stochastic
            depth is applied. Default is 0.0.
        init_values: float, initial value for the layer scale parameters.
            Default is 1e-5.
        block_prefix: string, prefix used for all layer names within this block.
            Default is "block".

    Returns:
        Output tensor for the block.
    """
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_1")(x)

    attn_output = TalkingHeadAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=f"{block_prefix}_attn",
    )(y)

    attn_output = LayerScale(
        init_values=init_values, name=f"{block_prefix}_layerscale_1"
    )(attn_output)

    attn_output = (
        StochasticDepth(drop_rate)(attn_output) if drop_rate > 0 else attn_output
    )
    x = layers.Add(name=f"{block_prefix}_add_1")([x, attn_output])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_2")(x)

    mlp_output = mlp_block(
        y,
        hidden_dim=int(embed_dim * mlp_ratio),
        out_dim=embed_dim,
        block_prefix=f"{block_prefix}_mlp",
    )

    mlp_output = LayerScale(
        init_values=init_values, name=f"{block_prefix}_layerscale_2"
    )(mlp_output)

    mlp_output = StochasticDepth(drop_rate)(mlp_output) if drop_rate > 0 else mlp_output
    x = layers.Add(name=f"{block_prefix}_add_2")([x, mlp_output])

    return x


def LayerScaleBlockClassAttn(
    cls_token,
    x,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    init_values=1e-5,
    block_prefix="block_token_only",
):
    """A transformer block with layer scaling and class attention for the CLS token.

    This block implements a transformer architecture specifically designed for
    class token processing, with layer normalization, class attention, MLP,
    layer scaling, and residual connections. It processes the class token while
    attending to all tokens (class token + patch tokens).

    Args:
        cls_token: tensor, the class token input with shape [batch_size, 1, embed_dim].
        x: tensor, the patch tokens with shape [batch_size, num_patches, embed_dim].
        embed_dim: int, the embedding dimension for the block.
        num_heads: int, the number of attention heads to use.
        mlp_ratio: float, ratio that determines the hidden dimension size of the MLP
            relative to the embedding dimension. Default is 4.0.
        init_values: float, initial value for the layer scale parameters.
            Default is 1e-5.
        block_prefix: string, prefix used for all layer names within this block.
            Default is "block_token_only".

    Returns:
        Updated class token tensor after processing through the block.
    """
    concat_features = layers.Concatenate(axis=1)([cls_token, x])
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_1")(
        concat_features
    )

    cls_output = ClassAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=f"{block_prefix}_attn",
    )(y)

    cls_output = LayerScale(
        init_values=init_values, name=f"{block_prefix}_layerscale_1"
    )(cls_output)

    cls_token = layers.Add(name=f"{block_prefix}_add_1")([cls_token, cls_output])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_2")(
        cls_token
    )

    mlp_output = mlp_block(
        y,
        hidden_dim=int(embed_dim * mlp_ratio),
        out_dim=embed_dim,
        block_prefix=f"{block_prefix}_mlp",
    )

    mlp_output = LayerScale(
        init_values=init_values, name=f"{block_prefix}_layerscale_2"
    )(mlp_output)

    cls_token = layers.Add(name=f"{block_prefix}_add_2")([cls_token, mlp_output])

    return cls_token


@keras.saving.register_keras_serializable(package="kvmm")
class CaiT(keras.Model):
    """Instantiates the CaiT (Class-Attention in Image Transformers) architecture.

    Reference:
    - [Going deeper with Image Transformers](
        https://arxiv.org/abs/2103.17239) (ICCV 2021)

    Args:
        patch_size: Integer or tuple, specifying the size of image patches to extract.
            Defaults to `16`.
        embed_dim: Integer, the dimensionality of token embeddings. Defaults to `192`.
        depth: Integer, the number of transformer blocks to stack. Defaults to `24`.
        num_heads: Integer, the number of attention heads in each transformer block.
            Defaults to `4`.
        drop_path_rate: Float, stochastic depth rate. Defaults to `0.0`.
        include_top: Boolean, whether to include the classification head at the top
            of the network. Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to `False`.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `False`.
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
            - `None` (default): the output is the 4D tensor from the last transformer block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to `1000`.
        classifier_activation: String or callable, activation function for the top
            layer. Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"CaiT"`.

    Returns:
        A Keras `Model` instance.
    """

    def __init__(
        self,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        drop_path_rate=0.0,
        include_top=True,
        as_backbone=False,
        include_normalization=False,
        normalization_mode="imagenet",
        weights=None,
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="CaiT",
        **kwargs,
    ):
        if weights is not None:
            print(
                "NOTICE: The current pretrained weights are provisional and subject to change. "
                "They are experimental and will be updated in future releases."
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

        data_format = keras.config.image_data_format()

        default_input_shape = (
            (384, 384, 3)
            if weights and "384" in weights
            else (512, 512, 3)
            if weights and "448" in weights
            else (224, 224, 3)
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

        depth_token_only = 2

        inputs = img_input
        features = []

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        x = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            name="stem_conv",
        )(x)

        grid_h = input_shape[0] // patch_size
        grid_w = input_shape[1] // patch_size

        x = layers.Reshape((-1, embed_dim))(x)

        x = AddPositionEmbs(
            grid_h=grid_h, grid_w=grid_w, no_embed_class=True, name="pos_embed"
        )(x)
        features.append(x)

        dpr = list(ops.linspace(0.0, drop_path_rate, depth))

        for i in range(depth):
            x = LayerScaleBlockTalkingHeadAttn(
                x,
                embed_dim=embed_dim,
                num_heads=num_heads,
                drop_rate=dpr[i],
                init_values=1e-5,
                block_prefix=f"blocks_{i}",
            )
            if (i + 1) % 4 == 0 or i == depth - 1:
                features.append(x)

        cls_token = ClassDistToken(name="cls_token")(x)

        for i in range(depth_token_only):
            cls_token = LayerScaleBlockClassAttn(
                cls_token,
                x,
                embed_dim=embed_dim,
                num_heads=num_heads,
                init_values=1e-5,
                block_prefix=f"blocks_token_only_{i}",
            )
            if i == depth_token_only - 1:
                features.append(cls_token)

        x = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")(cls_token)

        if include_top:
            x = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(x[:, 0])
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

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
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
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "drop_path_rate": self.drop_path_rate,
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
def CaiTXXS24(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_dist_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="CaiTXXS24",
    **kwargs,
):
    model = CaiT(
        **CAIT_MODEL_CONFIG["CaiTXXS24"],
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

    if weights in get_all_weight_names(CAIT_WEIGHTS_CONFIG):
        load_weights_from_config("CaiTXXS24", weights, model, CAIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def CaiTXXS36(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_dist_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="CaiTXXS36",
    **kwargs,
):
    model = CaiT(
        **CAIT_MODEL_CONFIG["CaiTXXS36"],
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

    if weights in get_all_weight_names(CAIT_WEIGHTS_CONFIG):
        load_weights_from_config("CaiTXXS36", weights, model, CAIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def CaiTXS24(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_dist_in1k_384",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="CaiTXS24",
    **kwargs,
):
    model = CaiT(
        **CAIT_MODEL_CONFIG["CaiTXS24"],
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

    if weights in get_all_weight_names(CAIT_WEIGHTS_CONFIG):
        load_weights_from_config("CaiTXS24", weights, model, CAIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def CaiTS24(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_dist_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="CaiTS24",
    **kwargs,
):
    model = CaiT(
        **CAIT_MODEL_CONFIG["CaiTS24"],
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

    if weights in get_all_weight_names(CAIT_WEIGHTS_CONFIG):
        load_weights_from_config("CaiTS24", weights, model, CAIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def CaiTS36(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_dist_in1k_384",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="CaiTS36",
    **kwargs,
):
    model = CaiT(
        **CAIT_MODEL_CONFIG["CaiTS36"],
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

    if weights in get_all_weight_names(CAIT_WEIGHTS_CONFIG):
        load_weights_from_config("CaiTS36", weights, model, CAIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def CaiTM36(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_dist_in1k_384",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="CaiTM36",
    **kwargs,
):
    model = CaiT(
        **CAIT_MODEL_CONFIG["CaiTM36"],
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

    if weights in get_all_weight_names(CAIT_WEIGHTS_CONFIG):
        load_weights_from_config("CaiTM36", weights, model, CAIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def CaiTM48(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_dist_in1k_448",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="CaiTM48",
    **kwargs,
):
    model = CaiT(
        **CAIT_MODEL_CONFIG["CaiTM48"],
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

    if weights in get_all_weight_names(CAIT_WEIGHTS_CONFIG):
        load_weights_from_config("CaiTM48", weights, model, CAIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
