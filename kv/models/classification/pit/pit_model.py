import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kv.layers import (
    AddPositionEmbs,
    ClassDistToken,
    ImagePreprocessingLayer,
    MultiHeadSelfAttention,
)
from kv.utils import get_all_weight_names, load_weights_from_config, register_model

from .config import PIT_MODEL_CONFIG, PIT_WEIGHTS_CONFIG


def mlp_block(inputs, hidden_features, out_features=None, drop=0.0, block_prefix=None):
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
    input_tensor, (height, width) = x

    tokens = input_tensor[:, :nb_tokens]
    spatial = input_tensor[:, nb_tokens:]

    new_height = (height + stride - 1) // stride
    new_width = (width + stride - 1) // stride

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

    spatial = layers.Reshape((new_height * new_width, out_channels))(spatial)

    output = layers.Concatenate(axis=1)([tokens, spatial])

    return output, (new_height, new_width)


@keras.saving.register_keras_serializable(package="kv")
class PoolingVisionTransformer(keras.Model):
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
        include_preprocessing=True,
        preprocessing_mode="imagenet",
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

        x = img_input
        features = []

        if include_preprocessing:
            x = ImagePreprocessingLayer(mode=preprocessing_mode)(x)

        x = layers.Conv2D(
            filters=embed_dim[0],
            kernel_size=patch_size,
            strides=stride,
            data_format=data_format,
            name="patch_embed_conv",
        )(x)

        height = (input_shape[0] - patch_size) // stride + 1
        width = (input_shape[1] - patch_size) // stride + 1
        input_size = (height, width)

        x = layers.Reshape((height * width, embed_dim[0]))(x)

        x = ClassDistToken(
            use_distillation=distilled,
            combine_tokens=True,
            name="class_dist_token",
        )(x)

        x = AddPositionEmbs(
            grid_h=height,
            grid_w=width,
            no_embed_class=False,
            use_distillation=distilled,
            name="pos_embed",
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

                cls_head = layers.Dense(
                    num_classes, activation=classifier_activation, name="predictions"
                )(cls_token)
                dist_head = layers.Dense(
                    num_classes,
                    activation=classifier_activation,
                    name="predictions_dist",
                )(dist_token)

                x = layers.Average()([cls_head, dist_head])
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

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

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
        self.include_preprocessing = include_preprocessing
        self.preprocessing_mode = preprocessing_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = {
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
            "include_preprocessing": self.include_preprocessing,
            "preprocessing_mode": self.preprocessing_mode,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "name": self.name,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Model variants
@register_model
def PiT_XS(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights=None,
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
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_XS_Distilled(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights=None,
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
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_Ti(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights=None,
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
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_Ti_Distilled(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights=None,
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
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_S(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights=None,
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
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_S_Distilled(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights=None,
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
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_B(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights=None,
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
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def PiT_B_Distilled(
    include_top=True,
    as_backbone=False,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights=None,
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
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )

    if weights in get_all_weight_names(PIT_WEIGHTS_CONFIG):
        load_weights_from_config(name, weights, model, PIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
