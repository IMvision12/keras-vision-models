import keras
import numpy as np
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kvmm.layers import (
    ImageNormalizationLayer,
    StochasticDepth,
    WindowAttention,
    WindowPartition,
)
from kvmm.model_registry import register_model
from kvmm.utils import get_all_weight_names, load_weights_from_config

from .config import SWIN_MODEL_CONFIG, SWIN_WEIGHTS_CONFIG


class RollLayer(layers.Layer):
    """A layer that performs circular shifting of tensor elements along a specified axis.

    This layer shifts elements of the input tensor by a specified amount along
    the given axis. Elements that are shifted beyond the last position are
    re-introduced at the first position (circular/cyclic behavior).

    Args:
        shift: int or tuple of ints
            Number of positions to shift. If positive, shift to the right/down.
            If negative, shift to the left/up. If tuple, shifts by the specified
            amount for each corresponding axis.
        axis: int or tuple of ints
            Axis or axes along which to shift. If tuple, must have same length as shift.
        **kwargs: dict
            Additional keyword arguments passed to the parent Layer class.

    Example:
        ```python
        # Shift elements 2 positions to the right along axis 1
        roll_layer = RollLayer(shift=2, axis=1)
        output = roll_layer(input_tensor)

        # Shift elements in multiple axes
        roll_layer = RollLayer(shift=(1, -2), axis=(0, 1))
        output = roll_layer(input_tensor)
        ```

    Input Shape:
        Arbitrary. This layer can operate on tensors of any shape.

    Output Shape:
        Same as input shape.
    """

    def __init__(self, shift, axis, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift
        self.axis = axis

    def call(self, inputs):
        return ops.roll(inputs, shift=self.shift, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shift": self.shift,
                "axis": self.axis,
            }
        )
        return config


def swin_block(
    inputs,
    shift_size,
    window_size,
    relative_index,
    attention_mask,
    num_heads,
    bias_table_window_size,
    channels_axis,
    dropout_rate=0.0,
    drop_path_rate=0.0,
    name="swin_block",
):
    """
    Implements a Swin Transformer block with shifted window-based self-attention.

    The block consists of two main components:
    1. Shifted Window-based Multi-Head Self-Attention (SW-MSA) with LayerNorm
    2. Multi-Layer Perceptron (MLP) layer with LayerNorm

    Both components use layer normalization and residual connections with optional
    stochastic depth for regularization.

    Args:
        inputs: Input tensor to the Swin Transformer block.
        shift_size: Size of the shift for the shifted window-based attention.
            Usually set to 0 or window_size // 2.
        window_size: Size of the local window for computing self-attention.
        relative_index: Tensor containing relative position indices for computing
            positional embeddings in window-based attention.
        attention_mask: Mask tensor for window-based attention to handle padding
            and shifted windows.
        num_heads: Number of attention heads in the multi-head attention mechanism.
        bias_table_window_size: Size of the relative position bias table for window-based
            attention. Determines the range of relative positions that can be represented
            in the bias table.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        dropout_rate: Dropout rate for the attention and MLP layers. Default is 0.
        drop_path_rate: Drop path rate for stochastic depth regularization.
            Default is 0.
        name: String prefix used for naming layers in the block. Default is 'swin_block'.

    Returns:
        Output tensor after passing through the Swin Transformer block,
        with the same shape and dimensionality as the input.
    """
    feature_dim = ops.shape(inputs)[-1]
    img_height, img_width = ops.shape(inputs)[1], ops.shape(inputs)[2]
    x = layers.LayerNormalization(
        epsilon=1.001e-5,
        gamma_initializer="ones",
        axis=channels_axis,
        name=f"{name}_layernorm_1",
    )(inputs)

    height_padding = int((window_size - img_height % window_size) % window_size)
    width_padding = int((window_size - img_width % window_size) % window_size)
    if height_padding > 0 or width_padding > 0:
        x = layers.ZeroPadding2D(
            padding=((0, height_padding), (0, width_padding)),
            data_format=keras.config.image_data_format(),
        )(x)
    padded_x = x

    orig_height, orig_width = ops.shape(inputs)[1], ops.shape(inputs)[2]
    shifted_x = RollLayer(shift=[-shift_size, -shift_size], axis=[1, 2])(padded_x)

    attention_layer = WindowAttention(
        dim=feature_dim,
        num_heads=num_heads,
        window_size=window_size,
        bias_table_window_size=bias_table_window_size,
        proj_drop=dropout_rate,
        block_prefix=name,
    )

    attended_x = attention_layer(
        [shifted_x, window_size, relative_index, attention_mask]
    )
    unshifted_x = RollLayer(shift=[shift_size, shift_size], axis=[1, 2])(attended_x)
    trimmed_x = unshifted_x[:, :orig_height, :orig_width]

    dropout_layer = StochasticDepth(drop_path_rate=drop_path_rate)
    skip_x1 = inputs + dropout_layer(trimmed_x)

    norm_layer2 = layers.LayerNormalization(
        epsilon=1.001e-5,
        gamma_initializer="ones",
        axis=channels_axis,
        name=f"{name}_layernorm_2",
    )
    normalized_x = norm_layer2(skip_x1)
    mlp_x = mlp_block(inputs=normalized_x, dropout=dropout_rate, name=f"{name}_mlp")
    skip_x2 = skip_x1 + dropout_layer(mlp_x)

    return skip_x2


def mlp_block(inputs, dropout=0.0, name="mlp"):
    """
    Implements a Multi-Layer Perceptron (MLP) block with GELU activation and dropout.

    The block consists of two dense layers with the following structure:
    1. Dense layer with expansion ratio of 4x followed by GELU activation and dropout
    2. Dense layer projecting back to input dimension followed by dropout

    This is commonly used as the feed-forward network in transformer architectures.

    Args:
        inputs: Input tensor to the MLP block.
        dropout: Dropout rate for regularization between dense layers.
            Applied after both dense layers. Default is 0.
        name: String prefix used for naming layers in the block.
            Default is 'mlp'.

    Returns:
        Output tensor after passing through the MLP block,
        with the same number of channels as the input.
    """
    channels = inputs.shape[-1]

    x = layers.Dense(int(channels * 4.0), name=f"{name}_dense_1")(inputs)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout_1")(x)

    x = layers.Dense(channels, name=f"{name}_dense_2")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout_2")(x)

    return x


def patch_merging(inputs, channels_axis, name="patch_merging"):
    """
    Implements a patch merging layer to reduce spatial dimensions and increase channels.

    This layer performs the following operations:
    1. Handles odd spatial dimensions with padding
    2. Merges 2x2 neighboring patches into one
    3. Rearranges the channels with a specific permutation
    4. Applies layer normalization
    5. Projects to the final dimension using a dense layer

    The spatial dimensions are reduced by a factor of 2, while the channel
    dimension is increased by a factor of 2.

    Args:
        inputs: Input tensor with shape (batch_size, height, width, channels).
            Height and width should be divisible by 2 after padding.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        name: String prefix used for naming layers in the block.
            Default is 'patch_merging'.

    Returns:
        Output tensor with shape (batch_size, height//2, width//2, channels*2).
        The spatial dimensions are halved while the channel dimension is doubled.
    """
    channels = inputs.shape[-1]

    height, width = ops.shape(inputs)[1:3]
    hpad, wpad = height % 2, width % 2
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    x = ops.pad(inputs, paddings)

    h = ops.shape(x)[1] // 2
    w = ops.shape(x)[2] // 2

    x = ops.reshape(x, (-1, h, 2, w, 2, channels))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, h, w, 4 * channels))

    perm = np.arange(channels * 4).reshape((4, -1))
    perm[[1, 2]] = perm[[2, 1]]
    perm = perm.ravel()

    x_reshaped = ops.reshape(x, (-1, 4 * channels))
    perm_matrix = np.zeros((4 * channels, 4 * channels), dtype=np.float32)
    for i, j in enumerate(perm):
        perm_matrix[i, j] = 1
    x = ops.matmul(x_reshaped, ops.convert_to_tensor(perm_matrix))
    x = ops.reshape(x, (-1, h, w, 4 * channels))

    x = layers.LayerNormalization(
        epsilon=1.001e-5,
        name=f"{name}_pm_layernorm",
        dtype=inputs.dtype,
        axis=channels_axis,
    )(x)
    x = layers.Dense(
        channels * 2, use_bias=False, name=f"{name}_pm_dense", dtype=inputs.dtype
    )(x)

    return x


def swin_stage(
    inputs,
    depth,
    num_heads,
    window_size,
    bias_table_window_size,
    channels_axis,
    dropout_rate=0.0,
    drop_path_rate=0.0,
    name="swin_stage",
):
    """
    Implements a stage in the Swin Transformer architecture consisting of multiple Swin blocks.

    This stage performs the following operations:
    1. Computes window sizes and shift sizes based on input dimensions
    2. Generates relative position indices and attention masks for window attention
    3. Creates a sequence of Swin Transformer blocks with alternating regular and
       shifted window attention

    The stage maintains the input resolution while processing the features through
    multiple Swin Transformer blocks with window-based self-attention.

    Args:
        inputs: Input tensor with shape (batch_size, height, width, channels).
        depth: Number of Swin Transformer blocks in this stage.
        num_heads: Number of attention heads in each Swin block.
        window_size: Size of the local window for computing self-attention.
            Will be automatically adjusted if larger than input dimensions.
        bias_table_window_size: Size of the relative position bias table for window-based
            attention. Determines the range of relative positions that can be represented
            in the bias table.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        dropout_rate: Dropout rate for attention and MLP layers in Swin blocks.
            Default is 0.
        drop_path_rate: Drop path rate or list of rates for stochastic depth.
            If float, same rate applied to all blocks.
            If list, should have length equal to depth.
            Default is 0.
        name: String prefix used for naming layers in the stage.
            Default is 'swin_stage'.

    Returns:
        Output tensor after passing through the Swin Transformer stage,
        with the same shape as the input tensor.
    """
    h, w = ops.shape(inputs)[1:3]
    min_dim = ops.minimum(h, w)
    win_size = ops.minimum(window_size, min_dim)

    shift_size = window_size // 2
    shift_sz = 0
    if min_dim > window_size:
        shift_sz = shift_size

    pad_h = ((h - 1) // win_size + 1) * win_size
    pad_w = ((w - 1) // win_size + 1) * win_size

    coords = ops.arange(win_size)
    gx, gy = ops.meshgrid(coords, coords, indexing="ij")
    flat_gx = ops.reshape(gx, [-1])
    flat_gy = ops.reshape(gy, [-1])

    rel_pos_x = flat_gx[:, None] - flat_gx[None, :]
    rel_pos_y = flat_gy[:, None] - flat_gy[None, :]

    relative_index = (ops.reshape(rel_pos_x, [-1]) + win_size - 1) * (
        2 * win_size - 1
    ) + (ops.reshape(rel_pos_y, [-1]) + win_size - 1)

    dtype = keras.backend.floatx()
    partitioner = WindowPartition(window_size=win_size, fused=False)

    ones = ops.ones((1, h, w, 1), dtype="int32")
    pad_mask = ops.pad(ones, [[0, 0], [0, pad_h - h], [0, pad_w - w], [0, 0]])

    mask_wins = ops.squeeze(partitioner(pad_mask, height=pad_h, width=pad_w), axis=-1)
    win_diffs = mask_wins[:, None] - mask_wins[:, :, None]

    id_mask = ops.where(
        win_diffs == 0,
        ops.zeros_like(win_diffs, dtype=dtype),
        ops.full_like(win_diffs, -100.0, dtype=dtype),
    )[None, :, None]

    if shift_sz > 0:
        pattern = ops.convert_to_tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype="int32"
        )

        expanded_h = ops.concatenate(
            [
                ops.tile(pattern[0:1, :], [pad_h - win_size, 1]),
                ops.tile(pattern[1:2, :], [win_size - shift_sz, 1]),
                ops.tile(pattern[2:3, :], [shift_sz, 1]),
            ],
            axis=0,
        )

        # Then expand horizontally
        shift_base = ops.concatenate(
            [
                ops.tile(expanded_h[:, 0:1], [1, pad_w - win_size]),
                ops.tile(expanded_h[:, 1:2], [1, win_size - shift_sz]),
                ops.tile(expanded_h[:, 2:3], [1, shift_sz]),
            ],
            axis=1,
        )

        shift_wins = ops.squeeze(
            partitioner(shift_base[None, ..., None], height=pad_h, width=pad_w), axis=-1
        )

        shift_diffs = shift_wins[:, None] - shift_wins[:, :, None]
        shift_mask = ops.where(
            (shift_diffs == 0) & (win_diffs == 0),
            ops.zeros_like(win_diffs, dtype=dtype),
            ops.full_like(win_diffs, -100.0, dtype=dtype),
        )[None, :, None]
    else:
        shift_mask = id_mask

    masks = [id_mask, shift_mask]

    if not isinstance(drop_path_rate, (list, tuple)):
        drop_rates = [drop_path_rate] * depth
    else:
        drop_rates = list(drop_path_rate)

    x = inputs
    for i in range(depth):
        is_odd = i % 2
        current_shift = shift_sz if is_odd else 0
        x = swin_block(
            x,
            current_shift,
            win_size,
            relative_index,
            masks[is_odd],
            num_heads=num_heads,
            bias_table_window_size=bias_table_window_size,
            channels_axis=channels_axis,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_rates[i],
            name=f"{name}_blocks_{i}",
        )

    return x


@keras.saving.register_keras_serializable(package="kvmm")
class SwinTransformer(keras.Model):
    """Instantiates the Swin Transformer architecture for vision tasks.

    This implementation provides the hierarchical vision transformer that uses shifted
    windows for self-attention computation. It supports various configurations through
    different depths and embedding dimensions.

    References:
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

    Args:
        pretrain_size: Integer, the input image size used during pretraining.
            This is used to properly initialize the position embedding.
        window_size: Integer, size of the window for local self-attention computation.
            Defines the region where self-attention is computed locally.
        embed_dim: Integer, the initial embedding dimension for the transformer.
            This dimension is progressively increased through the network stages.
        depths: List of integers, specifying the number of transformer blocks in each stage.
            For example, [2, 2, 6, 2] creates 4 stages with respective numbers of blocks.
        num_heads: List of integers, number of attention heads in each stage.
            Should match the length of depths, e.g., [3, 6, 12, 24].
        dropout_rate: Float, dropout rate applied to attention and MLP layers.
            Defaults to `0.0`.
        drop_path_rate: Float, stochastic depth rate applied to blocks.
            Helps prevent overfitting. Defaults to `0.1`.
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
            Should be (height, width, channels). Height and width should be divisible
            by the patch size (4 by default).
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
        name: String, the name of the model. Defaults to `"SwinTransformer"`.

    Returns:
        A Keras `Model` instance.

    Example:
        ```python
        # Basic Swin-T (tiny) configuration
        model = SwinTransformer(
            pretrain_size=224,
            window_size=7,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            input_shape=(224, 224, 3)
        )

        # Swin-B (base) configuration
        model = SwinTransformer(
            pretrain_size=224,
            window_size=7,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            input_shape=(224, 224, 3)
        )
        ```
    """

    def __init__(
        self,
        pretrain_size,
        window_size,
        embed_dim,
        depths,
        num_heads,
        dropout_rate=0.0,
        drop_path_rate=0.1,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="ms_in22k_ft_in1k",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="SwinTransformer",
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
            and weights == "ms_in22k"
            and num_classes != 21841
        ):
            raise ValueError(
                f"When using 'ms_in22k' weights, num_classes must be 21841. "
                f"Received num_classes: {num_classes}"
            )

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=pretrain_size,
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

        x = layers.Conv2D(
            embed_dim,
            kernel_size=4,
            strides=4,
            padding="same",
            data_format=data_format,
            name="stem_conv",
        )(x)
        x = layers.LayerNormalization(
            epsilon=1.001e-5, axis=channels_axis, name="stem_norm"
        )(x)
        x = layers.Dropout(dropout_rate, name="stem_dropout")(x)
        features.append(x)

        path_drops = ops.convert_to_numpy(
            ops.linspace(0.0, drop_path_rate, sum(depths))
        )
        scale_factors = 2 ** ops.arange(2, 6)  # [4, 8, 16, 32]
        pretrain_windows = pretrain_size // scale_factors
        bias_table_window_size = ops.minimum(window_size, pretrain_windows)
        for i in range(len(depths)):
            start_idx = sum(depths[:i])
            end_idx = sum(depths[: i + 1])
            path_drop_values = path_drops[start_idx:end_idx].tolist()
            not_last = i != len(depths) - 1

            x = swin_stage(
                x,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                bias_table_window_size=bias_table_window_size[i],
                channels_axis=channels_axis,
                dropout_rate=dropout_rate,
                drop_path_rate=path_drop_values,
                name=f"layers_{i}",
            )
            if not_last:
                x = patch_merging(
                    x, channels_axis=channels_axis, name=f"layers_{i + 1}_downsample"
                )
            features.append(x)

        x = layers.LayerNormalization(
            epsilon=1.001e-5, axis=channels_axis, name="final_norm"
        )(x)

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

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.pretrain_size = pretrain_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
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
                "pretrain_size": self.pretrain_size,
                "window_size": self.window_size,
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
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
def SwinTinyP4W7(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ms_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="SwinTinyP4W7",
    **kwargs,
):
    model = SwinTransformer(
        **SWIN_MODEL_CONFIG["SwinTinyP4W7"],
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
    if weights in get_all_weight_names(SWIN_WEIGHTS_CONFIG):
        load_weights_from_config("SwinTinyP4W7", weights, model, SWIN_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


@register_model
def SwinSmallP4W7(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ms_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="SwinSmallP4W7",
    **kwargs,
):
    model = SwinTransformer(
        **SWIN_MODEL_CONFIG["SwinSmallP4W7"],
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
    if weights in get_all_weight_names(SWIN_WEIGHTS_CONFIG):
        load_weights_from_config("SwinSmallP4W7", weights, model, SWIN_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


@register_model
def SwinBaseP4W7(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ms_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="SwinBaseP4W7",
    **kwargs,
):
    model = SwinTransformer(
        **SWIN_MODEL_CONFIG["SwinBaseP4W7"],
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
    if weights in get_all_weight_names(SWIN_WEIGHTS_CONFIG):
        load_weights_from_config("SwinBaseP4W7", weights, model, SWIN_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


@register_model
def SwinBaseP4W12(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ms_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="SwinBaseP4W12",
    **kwargs,
):
    model = SwinTransformer(
        **SWIN_MODEL_CONFIG["SwinBaseP4W12"],
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
    if weights in get_all_weight_names(SWIN_WEIGHTS_CONFIG):
        load_weights_from_config("SwinBaseP4W12", weights, model, SWIN_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


@register_model
def SwinLargeP4W7(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ms_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="SwinLargeP4W7",
    **kwargs,
):
    model = SwinTransformer(
        **SWIN_MODEL_CONFIG["SwinLargeP4W7"],
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
    if weights in get_all_weight_names(SWIN_WEIGHTS_CONFIG):
        load_weights_from_config("SwinLargeP4W7", weights, model, SWIN_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


@register_model
def SwinLargeP4W12(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="ms_in22k_ft_in1k",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="SwinLargeP4W12",
    **kwargs,
):
    model = SwinTransformer(
        **SWIN_MODEL_CONFIG["SwinLargeP4W12"],
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
    if weights in get_all_weight_names(SWIN_WEIGHTS_CONFIG):
        load_weights_from_config("SwinLargeP4W12", weights, model, SWIN_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model
