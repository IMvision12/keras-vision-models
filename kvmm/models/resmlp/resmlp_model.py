import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer, Affine, LayerScale
from kvmm.utils import get_all_weight_names, load_weights_from_config, register_model

from .config import RESMLP_MODEL_CONFIG, RESMLP_WEIGHTS_CONFIG

def resmlp_block(
    x,
    dim,
    seq_len,
    mlp_ratio=4,
    init_values=1e-4,
    drop_rate=0.0,
    block_idx=None,
):
    inputs = x

    x = Affine(name=f"blocks_{block_idx}_affine_1")(inputs)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_1")(x)
    x_t = layers.Dense(
        seq_len,
        name=f"blocks_{block_idx}_dense_1",
        kernel_initializer="glorot_uniform",
    )(x_t)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_2")(x_t)
    if drop_rate > 0:
        x_t = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_1")(x_t)
    x_t = LayerScale(
        init_values,
        name=f"blocks_{block_idx}_scale_1"
    )(x_t)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([inputs, x_t])

    inputs = x
    x = Affine(name=f"blocks_{block_idx}_affine_2")(x)
    x = layers.Dense(
        dim * mlp_ratio,
        activation="gelu",
        name=f"blocks_{block_idx}_dense_2",
    )(x)
    x = layers.Dense(
        dim,
        name=f"blocks_{block_idx}_dense_3",
    )(x)
    if drop_rate > 0:
        x = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_2")(x)
    x = LayerScale(
        init_values,
        name=f"blocks_{block_idx}_scale_2"
    )(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_2")([inputs, x])

    return x

@keras.saving.register_keras_serializable(package="kvmm")
class ResMLP(keras.Model):

    def __init__(
        self,
        patch_size,
        embed_dim,
        depth,
        mlp_ratio=4,
        init_values=1e-4,
        drop_rate=0.0,
        drop_path_rate=0.0,
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
        name="ResMLP",
        **kwargs,
    ):
        if include_top and as_backbone:
            raise ValueError(
                "Cannot use `as_backbone=True` with `include_top=True`. "
                f"Received: as_backbone={as_backbone}, include_top={include_top}"
            )

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
            min_size=32,
            data_format=keras.config.image_data_format(),
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

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(img_input)
            if include_normalization
            else img_input
        )

        x = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            name="stem_conv",
        )(x)

        height, width = input_shape[:2]
        num_patches = (height // patch_size) * (width // patch_size)
        x = layers.Reshape((num_patches, embed_dim))(x)

        features = [x]
        features_at = [
            depth // 4,
            depth // 2,
            3 * depth // 4,
            depth - 1,
        ]

        for i in range(depth):
            drop_path = drop_path_rate * (i / depth)
            x = resmlp_block(
                x,
                embed_dim,
                num_patches,
                mlp_ratio,
                init_values,
                drop_path,
                block_idx=i,
            )
            if i in features_at:
                features.append(x)

        x = Affine(name="Final_affine")(x)
        if include_top:
            x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        elif as_backbone:
            x = features
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling1D(name="max_pool")(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.init_values = init_values
        self.drop_rate = drop_rate
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
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "mlp_ratio": self.mlp_ratio,
            "init_values": self.init_values,
            "drop_rate": self.drop_rate,
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
        })
        return config

@register_model
def ResMLP12(
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
    name="ResMLP12",
    **kwargs,
):
    model = ResMLP(
        **RESMLP_MODEL_CONFIG["ResMLP12"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(RESMLP_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResMLP12", weights, model, RESMLP_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model

@register_model
def ResMLP24(
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
    name="ResMLP24",
    **kwargs,
):
    model = ResMLP(
        **RESMLP_MODEL_CONFIG["ResMLP24"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(RESMLP_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResMLP24", weights, model, RESMLP_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model

@register_model
def ResMLP36(
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
    name="ResMLP36",
    **kwargs,
):
    model = ResMLP(
        **RESMLP_MODEL_CONFIG["ResMLP36"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(RESMLP_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResMLP36", weights, model, RESMLP_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model

@register_model
def ResMLPBig24(
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
    name="ResMLPBig24",
    **kwargs,
):
    model = ResMLP(
        **RESMLP_MODEL_CONFIG["ResMLPBig24"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        name=name,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(RESMLP_WEIGHTS_CONFIG):
        load_weights_from_config(
            "ResMLPBig24", weights, model, RESMLP_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model