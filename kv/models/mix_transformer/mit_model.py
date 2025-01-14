import keras
import numpy as np
from keras import backend, layers, ops
from keras.src.applications import imagenet_utils

from kv.layers import (
    EfficientMultiheadSelfAttention,
    ImagePreprocessingLayer,
    StochasticDepth,
)
from kv.utils import get_all_weight_names, load_weights_from_config

from ...model_registry import register_model
from .config import MIT_MODEL_CONFIG, MIT_WEIGHTS_CONFIG


def mlp_block(x, H, W, channels, mid_channels, data_format, name_prefix):
    """Creates a MLP block with spatial mixing using depthwise convolution.

    Args:
        x: Input tensor of shape (batch_size, H*W, channels)
        H: Height of the feature map for spatial operations
        W: Width of the feature map for spatial operations
        channels: Number of output channels
        mid_channels: Number of channels in the expanded intermediate representation
        name_prefix: String prefix used for naming the layers

    Returns:
        Tensor of shape (batch_size, H*W, channels) containing the processed features

    """
    x = layers.Dense(mid_channels, name=f"{name_prefix}_dense_1")(x)
    input_shape = ops.shape(x)
    x = layers.Reshape((H, W, input_shape[-1]))(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_dwconv",
    )(x)
    x = layers.Reshape((H * W, input_shape[-1]))(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dense(channels, name=f"{name_prefix}_dense_2")(x)
    return x


def overlap_patch_embedding_block(
    x,
    channels_axis,
    data_format,
    out_channels=32,
    patch_size=7,
    stride=4,
    stage_idx=1,
):
    """Creates overlapping patches from the input and embeds them into a lower-dimensional space.

    Args:
        x: Input tensor of shape (batch_size, height, width, channels)
        out_channels: Number of output channels for the embedding
        patch_size: Size of the patch window for extracting overlapping patches
        stride: Stride length between patches
        stage_idx: Index used for naming the layers in multi-stage architectures

    Returns:
        Tuple containing:
        - Embedded patches tensor of shape (batch_size, H*W, out_channels)
        - H: Height of the feature map after patching
        - W: Width of the feature map after patching

    """
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=patch_size,
        strides=stride,
        padding="same",
        data_format=data_format,
        name=f"overlap_patch_embed{stage_idx}_conv",
    )(x)
    shape = ops.shape(x)
    H, W = shape[1], shape[2]
    x = layers.Reshape((-1, out_channels))(x)
    x = layers.LayerNormalization(
        axis=channels_axis,
        name=f"overlap_patch_embed{stage_idx}_layernorm",
        epsilon=1e-6,
    )(x)
    return x, H, W


def hierarchical_transformer_encoder_block(
    x,
    H,
    W,
    project_dim,
    num_heads,
    stage_idx,
    block_idx,
    channels_axis,
    data_format,
    qkv_bias=False,
    sr_ratio=1,
    drop_prob=0.0,
):
    """Creates a Hierarchical Transformer Encoder block with efficient self-attention and MLP layers.

    Args:
        x: Input tensor of shape (batch_size, H*W, project_dim)
        H: Height of the feature map for spatial operations
        W: Width of the feature map for spatial operations
        project_dim: Dimension of the projection space for attention and output
        num_heads: Number of attention heads
        stage_idx: Index of the current stage in the network
        block_idx: Index of the current block within the stage
        qkv_bias: Boolean indicating whether to use bias in query, key, value projections
        sr_ratio: Spatial reduction ratio for efficient attention
        drop_prob: Probability for stochastic depth dropout

    Returns:
        Tensor of shape (batch_size, H*W, project_dim) containing the processed features
        after self-attention and MLP operations with residual connections
    """

    block_prefix = f"block{stage_idx}_{block_idx}"

    attn_layer = EfficientMultiheadSelfAttention(
        project_dim, sr_ratio, block_prefix, qkv_bias, num_heads
    )
    drop_path_layer = StochasticDepth(drop_prob)

    norm1 = layers.LayerNormalization(
        epsilon=1e-6, axis=channels_axis, name=f"{block_prefix}_layernorm1"
    )(x)
    attn_out = attn_layer(norm1, H=H, W=W)
    attn_out = drop_path_layer(attn_out)
    add1 = layers.Add()([x, attn_out])

    norm2 = layers.LayerNormalization(
        epsilon=1e-6, axis=channels_axis, name=f"{block_prefix}_layernorm2"
    )(add1)
    mlp_out = mlp_block(
        norm2,
        H,
        W,
        channels=project_dim,
        mid_channels=int(project_dim * 4),
        data_format=data_format,
        name_prefix=f"{block_prefix}_mlp",
    )
    mlp_out = drop_path_layer(mlp_out)
    out = layers.Add()([add1, mlp_out])

    return out


@keras.saving.register_keras_serializable(package="kv")
class MixTransformer(keras.Model):
    """ "Instantiates the Mix Transformer (MiT) architecture from the SegFormer paper.

    The Mix Transformer (MiT) serves as the backbone of the SegFormer architecture,
    featuring hierarchical transformer blocks with efficient local attention and
    progressive reduction of sequence length.

    References:
    - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers]
        (https://arxiv.org/abs/2105.15203)

    Args:
        embed_dims: List of integers, specifying the embedding dimensions for each stage
            of the network. For example, [32, 64, 160, 256] creates a hierarchical
            structure with increasing channel dimensions.
        depths: List of integers, specifying the number of transformer blocks in each
            stage. Must have the same length as embed_dims. For example, [2, 2, 2, 2]
            creates 2 transformer blocks per stage.
        include_top: Boolean, whether to include the classification head at the top
            of the network. Defaults to `True`.
        include_preprocessing: Boolean, whether to include preprocessing layers at the
            start of the network. When True, input images should be in uint8 format
            with values in [0, 255]. Defaults to `True`.
        preprocessing_mode: String, specifying the preprocessing mode to use. Must be
            one of: 'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one',
            or 'minus_one_to_one'. Only used when include_preprocessing=True.
        weights: String or None, specifying the path to pretrained weights or one of
            the available options. Defaults to None.
        input_shape: Optional tuple specifying the shape of the input data.
            Should be (height, width, channels). If None, defaults to (224, 224, 3).
        input_tensor: Optional Keras tensor to use as model input. Useful for
            connecting the model to other Keras components.
        pooling: Optional pooling mode when `include_top=False`:
            - `None`: Return the sequence of feature maps from each stage
            - `"avg"`: Apply global average pooling to each feature map
            - `"max"`: Apply global max pooling to each feature map
        num_classes: Integer, number of classes for classification when
            include_top=True. Defaults to 1000.
        classifier_activation: String or callable, the activation function to use
            for the classification head. Set to None to return logits.
            Defaults to "softmax".
        name: String, name of the model. Defaults to "MixTransformer".

    Returns:
        A Keras Model instance.

    Example:
        ```python
        # Create a typical MiT-B0 backbone
        model = MixTransformer(
            embed_dims=[32, 64, 160, 256],
            depths=[2, 2, 2, 2],
            include_top=True,
            input_shape=(224, 224, 3)
        )

        # Create a deeper MiT-B2 backbone
        model = MixTransformer(
            embed_dims=[64, 128, 320, 512],
            depths=[3, 4, 6, 3],
            include_top=False,
            pooling="avg"
        )
        ```

    The MixTransformer architecture includes several key features:
    1. Hierarchical structure with progressively increasing channel dimensions
    2. Efficient local attention mechanism
    3. Overlapped patch embedding
    4. Mix-FFN for better feature representation
    5. Progressive reduction of sequence length for computational efficiency
    """

    def __init__(
        self,
        embed_dims,
        depths,
        include_top=True,
        include_preprocessing=True,
        preprocessing_mode="imagenet",
        weights="in1k",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="MixTransformer",
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

        channels_axis = -1 if backend.image_data_format() == "channels_last" else -3
        data_format = keras.config.image_data_format()

        self.drop_path_rate = 0.1
        self.num_stages = 4
        self.blockwise_num_heads = [1, 2, 5, 8]
        self.blockwise_sr_ratios = [8, 4, 2, 1]

        total_blocks = sum(depths)
        dpr = [x.item() for x in np.linspace(0.0, self.drop_path_rate, total_blocks)]

        x = img_input
        cur_block = 0

        x = (
            ImagePreprocessingLayer(mode=preprocessing_mode)(x)
            if include_preprocessing
            else x
        )

        for i in range(self.num_stages):
            x, H, W = overlap_patch_embedding_block(
                x,
                embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                stage_idx=i + 1,
                data_format=data_format,
                channels_axis=channels_axis,
            )

            for j in range(depths[i]):
                x = hierarchical_transformer_encoder_block(
                    x,
                    H,
                    W,
                    project_dim=embed_dims[i],
                    num_heads=self.blockwise_num_heads[i],
                    stage_idx=i + 1,
                    block_idx=j,
                    sr_ratio=self.blockwise_sr_ratios[i],
                    drop_prob=dpr[cur_block],
                    qkv_bias=True,
                    channels_axis=channels_axis,
                    data_format=data_format,
                )
                cur_block += 1

            x = layers.LayerNormalization(
                name=f"layernorm{i + 1}", axis=channels_axis, epsilon=1e-6
            )(x)
            x = layers.Reshape((H, W, embed_dims[i]))(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
                x
            )
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(
                    data_format=data_format, name="avg_pool"
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(data_format=data_format, name="max_pool")(
                    x
                )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.embed_dims = embed_dims
        self.depths = depths
        self.include_top = include_top
        self.include_preprocessing = include_preprocessing
        self.preprocessing_mode = preprocessing_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "embed_dims": self.embed_dims,
            "depths": self.depths,
            "include_top": self.include_top,
            "include_preprocessing": self.include_preprocessing,
            "preprocessing_mode": self.preprocessing_mode,
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
def MiT_B0(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MiT_B0",
    **kwargs,
):
    model = MixTransformer(
        **MIT_MODEL_CONFIG["MiT_B0"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MIT_WEIGHTS_CONFIG):
        load_weights_from_config("MiT_B0", weights, model, MIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MiT_B1(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MiT_B1",
    **kwargs,
):
    model = MixTransformer(
        **MIT_MODEL_CONFIG["MiT_B1"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MIT_WEIGHTS_CONFIG):
        load_weights_from_config("MiT_B1", weights, model, MIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MiT_B2(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MiT_B2",
    **kwargs,
):
    model = MixTransformer(
        **MIT_MODEL_CONFIG["MiT_B2"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MIT_WEIGHTS_CONFIG):
        load_weights_from_config("MiT_B2", weights, model, MIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MiT_B3(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MiT_B3",
    **kwargs,
):
    model = MixTransformer(
        **MIT_MODEL_CONFIG["MiT_B3"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MIT_WEIGHTS_CONFIG):
        load_weights_from_config("MiT_B3", weights, model, MIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MiT_B4(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MiT_B4",
    **kwargs,
):
    model = MixTransformer(
        **MIT_MODEL_CONFIG["MiT_B4"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MIT_WEIGHTS_CONFIG):
        load_weights_from_config("MiT_B4", weights, model, MIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def MiT_B5(
    include_top=True,
    include_preprocessing=True,
    preprocessing_mode="imagenet",
    weights="in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="MiT_B5",
    **kwargs,
):
    model = MixTransformer(
        **MIT_MODEL_CONFIG["MiT_B5"],
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        preprocessing_mode=preprocessing_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    if weights in get_all_weight_names(MIT_WEIGHTS_CONFIG):
        load_weights_from_config("MiT_B5", weights, model, MIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
