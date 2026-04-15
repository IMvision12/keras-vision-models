import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.layers import (
    ImageNormalizationLayer,
    StochasticDepth,
)
from kmodels.model_registry import register_model
from kmodels.models.swin.swin_layers import (
    RollLayer,
    WindowAttention,
    WindowPartition,
)
from kmodels.weight_utils import get_all_weight_names, load_weights_from_config

from .config import SWIN_MODEL_CONFIG, SWIN_WEIGHTS_CONFIG


def _spatial_layer_norm(x, data_format, epsilon=1.001e-5, name=None):
    """LayerNorm over channels for spatial feature maps.

    For channels_first, permutes to NHWC, normalizes, then permutes back.
    """
    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name=f"{name}_to_cl" if name else None)(x)
    x = layers.LayerNormalization(axis=-1, epsilon=epsilon, name=name)(x)
    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2), name=f"{name}_to_cf" if name else None)(x)
    return x


def swin_block(
    inputs,
    shift_size,
    window_size,
    relative_index,
    attention_mask,
    num_heads,
    bias_table_window_size,
    channels_axis,
    data_format,
    dropout_rate=0.0,
    drop_path_rate=0.0,
    name="swin_block",
):
    """Swin Transformer block with shifted window self-attention.

    Args:
        inputs: Input tensor.
        shift_size: int, shift size for shifted window attention.
        window_size: int, local window size.
        relative_index: Tensor, relative position indices.
        attention_mask: Tensor, attention mask for shifted windows.
        num_heads: int, number of attention heads.
        bias_table_window_size: int, relative position bias table size.
        channels_axis: int, channel axis index.
        data_format: string, image data format.
        dropout_rate: float, dropout rate. Default 0.0.
        drop_path_rate: float, stochastic depth rate. Default 0.0.
        name: string, layer name prefix. Default ``"swin_block"``.

    Returns:
        Output tensor.
    """
    cf = data_format == "channels_first"
    h_ax, w_ax = (2, 3) if cf else (1, 2)
    feature_dim = ops.shape(inputs)[1] if cf else ops.shape(inputs)[-1]
    img_height = ops.shape(inputs)[h_ax]
    img_width = ops.shape(inputs)[w_ax]

    x = _spatial_layer_norm(
        inputs,
        data_format,
        epsilon=1.001e-5,
        name=f"{name}_layernorm_1",
    )

    height_padding = int((window_size - img_height % window_size) % window_size)
    width_padding = int((window_size - img_width % window_size) % window_size)
    if height_padding > 0 or width_padding > 0:
        x = layers.ZeroPadding2D(
            padding=((0, height_padding), (0, width_padding)),
            data_format=data_format,
        )(x)

    padded_x = x
    shifted_x = RollLayer(shift=[-shift_size, -shift_size], axis=[h_ax, w_ax])(padded_x)

    attention_layer = WindowAttention(
        dim=feature_dim,
        num_heads=num_heads,
        window_size=window_size,
        bias_table_window_size=bias_table_window_size,
        proj_drop=dropout_rate,
        data_format=data_format,
        block_prefix=name,
    )
    attended_x = attention_layer(
        [shifted_x, window_size, relative_index, attention_mask]
    )
    unshifted_x = RollLayer(shift=[shift_size, shift_size], axis=[h_ax, w_ax])(
        attended_x
    )

    if cf:
        trimmed_x = unshifted_x[:, :, :img_height, :img_width]
    else:
        trimmed_x = unshifted_x[:, :img_height, :img_width]

    dropout_layer = StochasticDepth(drop_path_rate=drop_path_rate)
    skip_x1 = inputs + dropout_layer(trimmed_x)

    normalized_x = _spatial_layer_norm(
        skip_x1,
        data_format,
        epsilon=1.001e-5,
        name=f"{name}_layernorm_2",
    )

    if cf:
        mlp_in = ops.transpose(normalized_x, [0, 2, 3, 1])
    else:
        mlp_in = normalized_x
    mlp_x = mlp_block(inputs=mlp_in, dropout=dropout_rate, name=f"{name}_mlp")
    if cf:
        mlp_x = ops.transpose(mlp_x, [0, 3, 1, 2])

    skip_x2 = skip_x1 + dropout_layer(mlp_x)
    return skip_x2


def mlp_block(inputs, dropout=0.0, name="mlp"):
    """MLP block with two Dense layers and GELU activation.

    Operates on the last dimension (channels_last layout expected).

    Args:
        inputs: Input tensor of shape ``(B, H, W, C)``.
        dropout: float, dropout rate. Default 0.0.
        name: string, layer name prefix. Default ``"mlp"``.

    Returns:
        Output tensor.
    """
    channels = inputs.shape[-1]
    x = layers.Dense(int(channels * 4.0), name=f"{name}_dense_1")(inputs)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout_1")(x)
    x = layers.Dense(channels, name=f"{name}_dense_2")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout_2")(x)
    return x


def patch_merging(inputs, channels_axis, data_format, name="patch_merging"):
    """Patch merging layer that halves spatial dims and doubles channels.

    Args:
        inputs: Input tensor.
        channels_axis: int, channel axis index.
        data_format: string, image data format.
        name: string, layer name prefix. Default ``"patch_merging"``.

    Returns:
        Output tensor with halved spatial dimensions and doubled channels.
    """
    cf = data_format == "channels_first"
    channels = inputs.shape[1] if cf else inputs.shape[-1]
    h_ax, w_ax = (2, 3) if cf else (1, 2)

    height = ops.shape(inputs)[h_ax]
    width = ops.shape(inputs)[w_ax]
    hpad, wpad = height % 2, width % 2

    if cf:
        paddings = [[0, 0], [0, 0], [0, hpad], [0, wpad]]
    else:
        paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    x = ops.pad(inputs, paddings)

    h = ops.shape(x)[h_ax] // 2
    w = ops.shape(x)[w_ax] // 2

    if cf:
        x = ops.reshape(x, (-1, channels, h, 2, w, 2))
        x = ops.transpose(x, (0, 1, 2, 4, 3, 5))
        x = ops.reshape(x, (-1, 4 * channels, h, w))
    else:
        x = ops.reshape(x, (-1, h, 2, w, 2, channels))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (-1, h, w, 4 * channels))

    perm = ops.reshape(ops.arange(channels * 4), (4, -1))
    perm = ops.convert_to_numpy(perm)
    perm[[1, 2]] = perm[[2, 1]]
    perm = perm.ravel()

    if cf:
        x = ops.transpose(x, (0, 2, 3, 1))
    x_reshaped = ops.reshape(x, (-1, 4 * channels))
    perm_matrix = ops.zeros((4 * channels, 4 * channels), dtype="float32")
    perm_matrix = ops.convert_to_numpy(perm_matrix)
    for i, j in enumerate(perm):
        perm_matrix[i, j] = 1
    x = ops.matmul(x_reshaped, ops.convert_to_tensor(perm_matrix))
    x = ops.reshape(x, (-1, h, w, 4 * channels))

    x = layers.LayerNormalization(
        epsilon=1.001e-5,
        name=f"{name}_pm_layernorm",
        dtype=inputs.dtype,
        axis=-1,
    )(x)
    x = layers.Dense(
        channels * 2, use_bias=False, name=f"{name}_pm_dense", dtype=inputs.dtype
    )(x)

    if cf:
        x = ops.transpose(x, (0, 3, 1, 2))

    return x


def swin_stage(
    inputs,
    depth,
    num_heads,
    window_size,
    bias_table_window_size,
    channels_axis,
    data_format,
    dropout_rate=0.0,
    drop_path_rate=0.0,
    name="swin_stage",
):
    """Swin Transformer stage with multiple blocks.

    Args:
        inputs: Input tensor.
        depth: int, number of Swin blocks.
        num_heads: int, number of attention heads.
        window_size: int, local window size.
        bias_table_window_size: int, relative position bias table size.
        channels_axis: int, channel axis index.
        data_format: string, image data format.
        dropout_rate: float, dropout rate. Default 0.0.
        drop_path_rate: float or list, stochastic depth rate. Default 0.0.
        name: string, layer name prefix. Default ``"swin_stage"``.

    Returns:
        Output tensor.
    """
    cf = data_format == "channels_first"
    h_ax, w_ax = (2, 3) if cf else (1, 2)

    h = ops.shape(inputs)[h_ax]
    w = ops.shape(inputs)[w_ax]
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
    partitioner = WindowPartition(
        window_size=win_size,
        fused=False,
        data_format="channels_last",
    )

    ones = ops.ones((1, h, w, 1), dtype="int32")
    pad_mask = ops.pad(ones, [[0, 0], [0, pad_h - h], [0, pad_w - w], [0, 0]])
    mask_wins = ops.squeeze(
        partitioner(pad_mask, height=pad_h, width=pad_w),
        axis=-1,
    )
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
        shift_base = ops.concatenate(
            [
                ops.tile(expanded_h[:, 0:1], [1, pad_w - win_size]),
                ops.tile(expanded_h[:, 1:2], [1, win_size - shift_sz]),
                ops.tile(expanded_h[:, 2:3], [1, shift_sz]),
            ],
            axis=1,
        )
        shift_wins = ops.squeeze(
            partitioner(shift_base[None, ..., None], height=pad_h, width=pad_w),
            axis=-1,
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
            data_format=data_format,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_rates[i],
            name=f"{name}_blocks_{i}",
        )

    return x


@keras.saving.register_keras_serializable(package="kmodels")
class SwinTransformer(keras.Model):
    """Instantiates the Swin Transformer architecture.

    Reference:
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted
      Windows](https://arxiv.org/abs/2103.14030)

    Args:
        pretrain_size: int, input image size used during pretraining.
        window_size: int, local window size for self-attention.
        embed_dim: int, initial embedding dimension.
        depths: List of integers, number of blocks per stage.
        num_heads: List of integers, attention heads per stage.
        dropout_rate: float, dropout rate. Defaults to ``0.0``.
        drop_path_rate: float, stochastic depth rate. Defaults to ``0.1``.
        include_top: bool, whether to include the classification head.
            Defaults to ``True``.
        as_backbone: bool, whether to output intermediate feature maps.
            Defaults to ``False``.
        include_normalization: bool, whether to include input normalization.
            Defaults to ``True``.
        normalization_mode: string, normalization mode. Defaults to ``"imagenet"``.
        weights: string, path to pretrained weights or weight identifier.
        input_shape: Optional tuple, input shape.
        input_tensor: Optional Keras tensor as model input.
        pooling: Optional string, pooling mode when ``include_top=False``.
        num_classes: int, number of output classes. Defaults to ``1000``.
        classifier_activation: string or callable, activation for the head.
            Defaults to ``"softmax"``.
        name: string, model name. Defaults to ``"SwinTransformer"``.

    Returns:
        A Keras ``Model`` instance.
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
        weights="ms_in1k",
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

        if weights and "in22k" in weights and "ft" not in weights:
            if num_classes != 21841:
                raise ValueError(
                    "When using 'ms_in22k' weights, num_classes must be 21841. "
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
        x = _spatial_layer_norm(
            x,
            data_format,
            epsilon=1.001e-5,
            name="stem_norm",
        )
        x = layers.Dropout(dropout_rate, name="stem_dropout")(x)
        features.append(x)

        path_drops = ops.convert_to_numpy(
            ops.linspace(0.0, drop_path_rate, sum(depths))
        )
        scale_factors = 2 ** ops.arange(2, 6)
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
                data_format=data_format,
                dropout_rate=dropout_rate,
                drop_path_rate=path_drop_values,
                name=f"layers_{i}",
            )
            if not_last:
                x = patch_merging(
                    x,
                    channels_axis=channels_axis,
                    data_format=data_format,
                    name=f"layers_{i + 1}_downsample",
                )
            features.append(x)

        x = _spatial_layer_norm(
            x,
            data_format,
            epsilon=1.001e-5,
            name="final_norm",
        )

        if include_top:
            x = layers.GlobalAveragePooling2D(
                data_format=data_format,
                name="avg_pool",
            )(x)
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
                    data_format=data_format,
                    name="avg_pool",
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(
                    data_format=data_format,
                    name="max_pool",
                )(x)

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
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
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
    weights="ms_in1k",
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
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
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
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
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
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
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
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
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
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
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
