import keras
import numpy as np
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import (
    EfficientMultiheadSelfAttention,
    ImageNormalizationLayer,
    StochasticDepth,
)
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import MIT_MODEL_CONFIG, MIT_WEIGHTS_CONFIG


def mlp_block(x, H, W, channels, mid_channels, data_format, name_prefix):
    """
    Implements an MLP block with a spatial depth-wise convolution in between.

    This function creates a block that processes the input tensor through a dense layer,
    reshapes it to apply a depth-wise convolution to capture spatial information,
    applies GELU activation, and projects it back through another dense layer.

    Args:
        x: Input tensor of shape [batch_size, H*W, input_channels]
        H: Height of the feature map
        W: Width of the feature map
        channels: Number of output channels for the final projection
        mid_channels: Number of channels for the intermediate dense layer
        data_format: Data format for the convolution ('channels_last' or 'channels_first')
        name_prefix: Prefix string for naming the layers

    Returns:
        Processed tensor of shape [batch_size, H*W, channels]

    Note:
        The function assumes input in a sequence format (H*W, C) and internally
        converts to spatial format (H, W, C) for the depth-wise convolution.
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
    """
    Creates an overlapping patch embedding block for vision transformers/MLP-mixers.

    This function implements the initial patch embedding stage commonly used in vision
    transformer architectures. It extracts overlapping patches from the input image,
    projects them to the desired dimension, and reshapes the output for subsequent
    transformer/MLP blocks.

    Args:
        x: Input tensor, typically an image with shape [batch_size, H, W, C]
        channels_axis: Axis index for the channels dimension for normalization
        data_format: Data format for the convolution ('channels_last' or 'channels_first')
        out_channels: Number of output channels for the projection (default: 32)
        patch_size: Size of the patch for convolution kernel (default: 7)
        stride: Stride of the convolution (default: 4)
        stage_idx: Index of the stage in the network, used for naming (default: 1)

    Returns:
        tuple: (
            reshaped tensor of shape [batch_size, H*W, out_channels],
            output feature map height H,
            output feature map width W
        )

    Note:
        The function uses PyTorch-style 0-indexed naming convention internally
        while maintaining a 1-indexed interface.
    """
    pytorch_stage_idx = stage_idx - 1

    x = keras.layers.ZeroPadding2D(padding=(patch_size // 2, patch_size // 2))(x)
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=patch_size,
        strides=stride,
        padding="valid",
        data_format=data_format,
        name=f"patch_embed_{pytorch_stage_idx}_conv_proj",
    )(x)
    shape = ops.shape(x)
    H, W = shape[1], shape[2]
    x = layers.Reshape((-1, out_channels))(x)
    x = layers.LayerNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        name=f"patch_embed_{pytorch_stage_idx}_layernorm",
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
    """
    Implements a hierarchical transformer encoder block with efficient self-attention.

    This function creates a transformer encoder block that combines multi-head self-attention
    with an MLP block containing spatial depth-wise convolution. It follows the typical
    transformer architecture with residual connections, normalization, and optional
    stochastic depth for regularization.

    Args:
        x: Input tensor of shape [batch_size, H*W, project_dim]
        H: Height of the feature map
        W: Width of the feature map
        project_dim: Dimension of the token embeddings
        num_heads: Number of attention heads
        stage_idx: Index of the hierarchical stage (1-indexed)
        block_idx: Index of the block within the stage
        channels_axis: Axis index for the channels dimension for normalization
        data_format: Data format for the convolution operations ('channels_last' or 'channels_first')
        qkv_bias: Whether to include bias in query, key, value projections (default: False)
        sr_ratio: Spatial reduction ratio for efficient attention (default: 1)
        drop_prob: Drop path probability for stochastic depth regularization (default: 0.0)

    Returns:
        Processed tensor of shape [batch_size, H*W, project_dim]

    Note:
        The function uses PyTorch-style 0-indexed naming convention internally
        while maintaining a 1-indexed interface for stage_idx.
    """
    pytorch_stage_idx = stage_idx - 1
    drop_path_layer = StochasticDepth(drop_prob)

    norm1 = layers.LayerNormalization(
        axis=channels_axis,
        epsilon=1e-6,
        name=f"block_{pytorch_stage_idx}_{block_idx}_layernorm_1",
    )(x)

    attn_layer = EfficientMultiheadSelfAttention(
        project_dim,
        sr_ratio,
        block_prefix=f"block_{pytorch_stage_idx}_{block_idx}",
        qkv_bias=qkv_bias,
        num_heads=num_heads,
    )

    attn_out = attn_layer(norm1)
    attn_out = drop_path_layer(attn_out)
    add1 = layers.Add()([x, attn_out])

    norm2 = layers.LayerNormalization(
        axis=channels_axis,
        epsilon=1e-6,
        name=f"block_{pytorch_stage_idx}_{block_idx}_layernorm_2",
    )(add1)

    mlp_out = mlp_block(
        norm2,
        H,
        W,
        channels=project_dim,
        mid_channels=int(project_dim * 4),
        data_format=data_format,
        name_prefix=f"block_{pytorch_stage_idx}_{block_idx}_mlp",
    )

    mlp_out = drop_path_layer(mlp_out)
    out = layers.Add()([add1, mlp_out])

    return out


@keras.saving.register_keras_serializable(package="kvmm")
class MixTransformer(keras.Model):
    """Instantiates the Mix Transformer (MiT) architecture from the SegFormer paper.

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
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to `False`.
        include_normalization: Boolean, whether to include normalization layers at the
            start of the network. When True, input images should be in uint8 format
            with values in [0, 255]. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to use. Must be
            one of: 'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one',
            or 'minus_one_to_one'. Only used when include_normalization=True.
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
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="in1k",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="MixTransformer",
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

        drop_path_rate = 0.1
        num_stages = 4
        blockwise_num_heads = [1, 2, 5, 8]
        blockwise_sr_ratios = [8, 4, 2, 1]

        total_blocks = sum(depths)
        dpr = [x.item() for x in np.linspace(0.0, drop_path_rate, total_blocks)]

        x = img_input
        features = []

        cur_block = 0

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(x)
            if include_normalization
            else x
        )

        for i in range(num_stages):
            x, H, W = overlap_patch_embedding_block(
                x,
                out_channels=embed_dims[i],
                channels_axis=channels_axis,
                data_format=data_format,
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                stage_idx=i + 1,
            )

            for j in range(depths[i]):
                x = hierarchical_transformer_encoder_block(
                    x,
                    H,
                    W,
                    project_dim=embed_dims[i],
                    num_heads=blockwise_num_heads[i],
                    stage_idx=i + 1,
                    block_idx=j,
                    sr_ratio=blockwise_sr_ratios[i],
                    drop_prob=dpr[cur_block],
                    qkv_bias=True,
                    channels_axis=channels_axis,
                    data_format=data_format,
                )
                cur_block += 1

            x = layers.LayerNormalization(
                name=f"final_layernorm_{i}", axis=channels_axis, epsilon=1e-5
            )(x)
            x = layers.Reshape((H, W, embed_dims[i]))(x)
            features.append(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
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
                "embed_dims": self.embed_dims,
                "depths": self.depths,
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
def MiT_B0(
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
    name="MiT_B0",
    **kwargs,
):
    model = MixTransformer(
        **MIT_MODEL_CONFIG["MiT_B0"],
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
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
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
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
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
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
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
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
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
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
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

    if weights in get_all_weight_names(MIT_WEIGHTS_CONFIG):
        load_weights_from_config("MiT_B5", weights, model, MIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
