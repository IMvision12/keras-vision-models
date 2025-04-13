import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import ImageNormalizationLayer
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import MLPMIXER_MODEL_CONFIG, MLPMIXER_WEIGHTS_CONFIG


def mixer_block(
    x,
    patches,
    filters,
    token_mlp_dim,
    channel_mlp_dim,
    channels_axis,
    drop_rate=0.0,
    block_idx=None,
):
    """A building block for the MLP-Mixer architecture.

    Args:
        x: input tensor.
        patches: int, the number of patches (sequence length) for token mixing.
        filters: int, the number of output filters for channel mixing.
        token_mlp_dim: int, hidden dimension for token mixing MLP.
        channel_mlp_dim: int, hidden dimension for channel mixing MLP.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        drop_rate: float, dropout rate to apply after dense layers (default: 0.0).
        block_idx: int or None, index of the block for naming layers (default: None).

    Returns:
        Output tensor for the block.
    """

    inputs = x

    x = layers.LayerNormalization(
        axis=channels_axis, epsilon=1e-6, name=f"blocks_{block_idx}_layernorm_1"
    )(x)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_1")(x)
    x_t = layers.Dense(
        token_mlp_dim,
        name=f"blocks_{block_idx}_dense_1",
        kernel_initializer="glorot_uniform",
    )(x_t)
    x_t = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu_1")(x_t)
    if drop_rate > 0:
        x_t = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_1")(x_t)
    x_t = layers.Dense(
        patches, name=f"blocks_{block_idx}_dense_2", kernel_initializer="glorot_uniform"
    )(x_t)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_2")(x_t)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([inputs, x_t])

    inputs = x
    x = layers.LayerNormalization(
        axis=channels_axis, epsilon=1e-6, name=f"blocks_{block_idx}_layernorm_2"
    )(x)
    x = layers.Dense(
        channel_mlp_dim,
        name=f"blocks_{block_idx}_dense_3",
        kernel_initializer="glorot_uniform",
    )(x)
    x = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu_2")(x)
    if drop_rate > 0:
        x = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_2")(x)
    x = layers.Dense(
        filters, name=f"blocks_{block_idx}_dense_4", kernel_initializer="glorot_uniform"
    )(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_2")([inputs, x])

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class MLPMixer(keras.Model):
    """Instantiates the MLP-Mixer architecture.

    Reference:
    - [MLP-Mixer: An all-MLP Architecture for Vision](
        https://arxiv.org/abs/2105.01601) (NIPS 2021)

    Args:
        patch_size: Integer or tuple, size of patches to be extracted from the input image.
        embed_dim: Integer, the embedding dimension for the token mixing and channel mixing MLPs.
        num_blocks: Integer, the number of MLP-Mixer blocks to stack.
        mlp_ratio: Tuple of two floats, scaling factors for (token_mixing_mlp, channel_mixing_mlp)
            hidden dimensions relative to embed_dim. Defaults to (0.5, 4.0).
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
            - None (default): the output is the 4D tensor from the last mixer block.
            - "avg": global average pooling is applied, and the output is a 2D tensor.
            - "max": global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification.
            Defaults to 1000.
        classifier_activation: String or callable, activation function for the top
            layer. Set to None to return logits. Defaults to "softmax".
        name: String, the name of the model. Defaults to "MLPMixer".

    Returns:
        A Keras Model instance.
    """

    def __init__(
        self,
        patch_size,
        embed_dim,
        num_blocks,
        mlp_ratio=(0.5, 4.0),
        drop_rate=0.0,
        drop_path_rate=0.0,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="goog_in21k_ft_in1k",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="MLPMixer",
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
            and weights.endswith("in21k")
            and num_classes != 21843
        ):
            raise ValueError(
                f"When using 'in21k' weights, num_classes must be 21843. "
                f"Received num_classes: {num_classes}"
            )

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

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

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        # Patch embedding
        x = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=data_format,
            name="stem_conv",
        )(x)

        if data_format == "channels_first":
            height, width = input_shape[1], input_shape[2]
            x = layers.Permute((2, 3, 1))(x)
        else:
            height, width = input_shape[0], input_shape[1]

        num_patches = (height // patch_size) * (width // patch_size)
        x = layers.Reshape((num_patches, embed_dim))(x)
        features.append(x)

        token_mlp_dim = int(embed_dim * mlp_ratio[0])
        channel_mlp_dim = int(embed_dim * mlp_ratio[1])

        features_at = [
            num_blocks // 4,
            num_blocks // 2,
            3 * num_blocks // 4,
            num_blocks - 1,
        ]
        for i in range(num_blocks):
            drop_path = drop_path_rate * (i / num_blocks)

            x = mixer_block(
                x,
                num_patches,
                embed_dim,
                token_mlp_dim,
                channel_mlp_dim,
                channels_axis,
                drop_rate=drop_path,
                block_idx=i,
            )
            if i in features_at:
                features.append(x)

        x = layers.LayerNormalization(
            axis=channels_axis, epsilon=1e-6, name="final_layernomr"
        )(x)

        if include_top:
            x = layers.GlobalAveragePooling1D(data_format=data_format, name="avg_pool")(
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
                x = layers.GlobalAveragePooling1D(
                    data_format=data_format, name="avg_pool"
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling1D(data_format=data_format, name="max_pool")(
                    x
                )

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
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
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def MLPMixerB16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="goog_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MLPMixerB16",
    **kwargs,
):
    model = MLPMixer(
        **MLPMIXER_MODEL_CONFIG["MLPMixerB16"],
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

    if weights in get_all_weight_names(MLPMIXER_WEIGHTS_CONFIG):
        load_weights_from_config("MLPMixerB16", weights, model, MLPMIXER_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MLPMixerL16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="goog_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MLPMixerL16",
    **kwargs,
):
    model = MLPMixer(
        **MLPMIXER_MODEL_CONFIG["MLPMixerL16"],
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

    if weights in get_all_weight_names(MLPMIXER_WEIGHTS_CONFIG):
        load_weights_from_config("MLPMixerL16", weights, model, MLPMIXER_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
