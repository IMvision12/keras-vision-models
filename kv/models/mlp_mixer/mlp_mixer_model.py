import keras
from keras import backend, layers
from keras.src.applications import imagenet_utils

from kv.utils import get_all_weight_names, load_weights_from_config

from ...model_registry import register_model
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
        channels_axis: axis along which the channels are defined in the input tensor.
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


@keras.saving.register_keras_serializable(package="kv")
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
            of the network. Defaults to True.
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
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="MLPMixer",
        **kwargs,
    ):
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
        channels_axis = -1 if backend.image_data_format() == "channels_last" else -3

        # Patch embedding
        x = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            name="stem_conv",
        )(inputs)

        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        x = layers.Reshape((num_patches, embed_dim))(x)

        token_mlp_dim = int(embed_dim * mlp_ratio[0])
        channel_mlp_dim = int(embed_dim * mlp_ratio[1])

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

        x = layers.LayerNormalization(
            axis=channels_axis, epsilon=1e-6, name="final_layernomr"
        )(x)

        if include_top:
            x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling1D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "drop_path_rate": self.drop_path_rate,
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
def MLPMixer_B16(
    include_top=True,
    weights="goog_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MLPMixer_B16",
    **kwargs,
):
    model = MLPMixer(
        **MLPMIXER_MODEL_CONFIG["MLPMixer_B16"],
        include_top=include_top,
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
        load_weights_from_config(
            "EfficientNetV2L", weights, model, MLPMIXER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MLPMixer_L16(
    include_top=True,
    weights="goog_in21k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MLPMixer_L16",
    **kwargs,
):
    model = MLPMixer(
        **MLPMIXER_MODEL_CONFIG["MLPMixer_L16"],
        include_top=include_top,
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
        load_weights_from_config(
            "EfficientNetV2L", weights, model, MLPMIXER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
