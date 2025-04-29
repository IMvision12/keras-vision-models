import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import Affine, ImageNormalizationLayer, LayerScale
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

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
    """A building block for the ResMLP architecture.

    Args:
        x: input tensor.
        dim: int, dimension of the input features.
        seq_len: int, length of the input sequence for cross-patch mixing.
        mlp_ratio: float, ratio of the hidden dimension in the MLP to the input
            dimension (default: 4).
        init_values: float, initial value for layer scale parameters
            (default: 1e-4).
        drop_rate: float, dropout rate to apply after dense layers (default: 0.0).
        block_idx: int or None, index of the block for naming layers (default: None).

    Returns:
        Output tensor for the block.
    """
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
    x_t = LayerScale(init_values, name=f"blocks_{block_idx}_scale_1")(x_t)
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
    x = LayerScale(init_values, name=f"blocks_{block_idx}_scale_2")(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_2")([inputs, x])

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class ResMLP(keras.Model):
    """Instantiates the ResMLP architecture.

    Reference:
    - [ResMLP: Feedforward networks for image classification with data-efficient training](
        https://arxiv.org/abs/2105.03404) (CVPR 2021)

    Args:
        patch_size: Integer or tuple, size of patches to be extracted from the input image.
        embed_dim: Integer, the embedding dimension for the token mixing and channel mixing MLPs.
        depth: Integer, the number of ResMLP blocks to stack.
        mlp_ratio: Float, scaling factor for the MLP hidden dimension relative to embed_dim.
            Defaults to 4.0.
        init_values: Float, initial value for the layer scale parameters.
            Defaults to 1e-4.
        drop_rate: Float, dropout rate for the MLPs. Defaults to 0.0.
        drop_path_rate: Float, stochastic depth rate for the blocks. Defaults to 0.0.
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
            available options in keras-vision.
        input_tensor: Optional Keras tensor (output of `layers.Input()`) to use as
            the model's input. If not provided, a new input tensor is created based
            on input_shape.
        input_shape: Optional tuple specifying the shape of the input data.
        pooling: Optional pooling mode for feature extraction when include_top=False:
            - None (default): the output is the 4D tensor from the last ResMLP block.
            - "avg": global average pooling is applied, and the output is a 2D tensor.
            - "max": global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to 1000.
        classifier_activation: String or callable, activation function for the top
            layer. Set to None to return logits. Defaults to "softmax".
        name: String, the name of the model. Defaults to "ResMLP".

    Returns:
        A Keras Model instance.
    """

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

        if pooling is not None and pooling not in ["avg", "max"]:
            raise ValueError(
                "The `pooling` argument should be one of 'avg', 'max', or None. "
                f"Received: pooling={pooling}"
            )

        data_format = keras.config.image_data_format()

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
            data_format=data_format,
            name="stem_conv",
        )(x)

        if data_format == "channels_first":
            if len(input_shape) == 3:
                _, height, width = input_shape
            else:
                height, width = input_shape[1:]
        else:
            if len(input_shape) == 3:
                height, width, _ = input_shape
            else:
                height, width = input_shape[:2]

        num_patches = (height // patch_size) * (width // patch_size)

        if data_format == "channels_first":
            x = layers.Permute((2, 3, 1))(x)
            x = layers.Reshape((num_patches, embed_dim))(x)
        else:
            x = layers.Reshape((num_patches, embed_dim))(x)

        features.append(x)

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
        config.update(
            {
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
            }
        )
        return config


@register_model
def ResMLP12(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="fb_in1k",
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
        load_weights_from_config("ResMLP12", weights, model, RESMLP_WEIGHTS_CONFIG)
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
    weights="fb_in1k",
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
        load_weights_from_config("ResMLP24", weights, model, RESMLP_WEIGHTS_CONFIG)
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
    weights="fb_in1k",
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
        load_weights_from_config("ResMLP36", weights, model, RESMLP_WEIGHTS_CONFIG)
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
    weights="fb_in1k",
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
        load_weights_from_config("ResMLPBig24", weights, model, RESMLP_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
