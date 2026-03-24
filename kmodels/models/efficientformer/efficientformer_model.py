import keras
import numpy as np
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.layers import ImageNormalizationLayer, LayerScale, StochasticDepth
from kmodels.model_registry import register_model
from kmodels.models.efficientformer.efficientformer_layers import Attention4D
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import EFFICIENTFORMER_MODEL_CONFIG, EFFICIENTFORMER_WEIGHTS_CONFIG


def conv_mlp_block(
    inputs,
    hidden_features,
    out_features,
    drop=0.0,
    data_format="channels_last",
    name=None,
):
    """MLP block with 1x1 convolutions for 2D spatial feature maps.

    Applies two sequential 1x1 convolutions with batch normalization,
    GELU activation, and dropout. Used in the convolutional (2D) stages
    of EfficientFormer.

    Args:
        inputs: Input tensor of shape `(batch_size, height, width, channels)`.
        hidden_features: Integer, number of filters in the first convolution.
        out_features: Integer, number of filters in the second convolution.
        drop: Float, dropout rate applied after each convolution.
            Defaults to `0.0`.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            Defaults to `"channels_last"`.
        name: String, name prefix for all layers in this block.

    Returns:
        Output tensor of shape `(batch_size, height, width, out_features)`.
    """
    channels_axis = -1 if data_format == "channels_last" else 1

    x = layers.Conv2D(
        hidden_features,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_1",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_norm_1"
    )(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_1")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_1")(x)

    x = layers.Conv2D(
        out_features,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_norm_2"
    )(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_2")(x)
    return x


def mlp_block(inputs, hidden_features, out_features, drop=0.0, name=None):
    """Standard MLP block for 1D token sequences.

    Applies two fully connected layers with GELU activation and dropout.
    Used in the transformer (1D) stages of EfficientFormer.

    Args:
        inputs: Input tensor of shape `(batch_size, seq_len, channels)`.
        hidden_features: Integer, number of units in the hidden dense layer.
        out_features: Integer, number of units in the output dense layer.
        drop: Float, dropout rate applied after each dense layer.
            Defaults to `0.0`.
        name: String, name prefix for all layers in this block.

    Returns:
        Output tensor of shape `(batch_size, seq_len, out_features)`.
    """
    x = layers.Dense(hidden_features, use_bias=True, name=f"{name}_dense_1")(inputs)
    x = layers.Activation("gelu", name=f"{name}_gelu")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_1")(x)
    x = layers.Dense(out_features, use_bias=True, name=f"{name}_dense_2")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_2")(x)
    return x


def meta_block_2d(
    inputs,
    dim,
    pool_size=3,
    mlp_ratio=4.0,
    drop=0.0,
    drop_path=0.0,
    layer_scale_init_value=1e-5,
    data_format="channels_last",
    name=None,
):
    """2D MetaBlock with pooling token mixer for convolutional stages.

    Applies a pooling-based token mixer followed by a convolutional MLP,
    each with residual connections, layer scaling, and optional stochastic
    depth. Used in the early (2D spatial) stages of EfficientFormer.

    Reference:
    - [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

    Args:
        inputs: Input tensor of shape
            `(batch_size, height, width, channels)`.
        dim: Integer, feature dimension (number of channels).
        pool_size: Integer, kernel size for the average pooling token
            mixer. Defaults to `3`.
        mlp_ratio: Float, expansion ratio for the hidden dimension in
            the MLP. Defaults to `4.0`.
        drop: Float, dropout rate applied in the MLP.
            Defaults to `0.0`.
        drop_path: Float, stochastic depth rate for dropping the
            residual branch. Defaults to `0.0`.
        layer_scale_init_value: Float, initial value for the learnable
            layer scale parameters. Defaults to `1e-5`.
        data_format: String, either `"channels_last"` or
            `"channels_first"`. Defaults to `"channels_last"`.
        name: String, name prefix for all layers in this block.

    Returns:
        Output tensor of shape `(batch_size, height, width, channels)`.
    """
    # Token mixer (pooling)
    pooled = layers.AveragePooling2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool_pool"
    )(inputs)
    x = layers.Subtract(name=f"{name}_pool_sub")([pooled, inputs])
    x = LayerScale(layer_scale_init_value, name=f"{name}_ls1")(x)
    if drop_path > 0.0:
        x = StochasticDepth(drop_path, name=f"{name}_drop_path1")(x)
    x = layers.Add(name=f"{name}_add1")([inputs, x])

    # MLP
    y = conv_mlp_block(
        x,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=drop,
        data_format=data_format,
        name=f"{name}_mlp",
    )
    y = LayerScale(layer_scale_init_value, name=f"{name}_ls2")(y)
    if drop_path > 0.0:
        y = StochasticDepth(drop_path, name=f"{name}_drop_path2")(y)
    outputs = layers.Add(name=f"{name}_add2")([x, y])
    return outputs


def meta_block_1d(
    inputs,
    dim,
    mlp_ratio=4.0,
    drop=0.0,
    drop_path=0.0,
    layer_scale_init_value=1e-5,
    resolution=7,
    name=None,
):
    """1D MetaBlock with self-attention token mixer for transformer stages.

    Applies layer-normalized multi-head self-attention followed by a
    dense MLP, each with residual connections, layer scaling, and
    optional stochastic depth. Used in the final (1D sequence) stage
    of EfficientFormer.

    Reference:
    - [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

    Args:
        inputs: Input tensor of shape
            `(batch_size, seq_len, channels)`.
        dim: Integer, feature dimension (number of channels).
        mlp_ratio: Float, expansion ratio for the hidden dimension in
            the MLP. Defaults to `4.0`.
        drop: Float, dropout rate applied in the MLP.
            Defaults to `0.0`.
        drop_path: Float, stochastic depth rate for dropping the
            residual branch. Defaults to `0.0`.
        layer_scale_init_value: Float, initial value for the learnable
            layer scale parameters. Defaults to `1e-5`.
        resolution: Integer, spatial resolution of the feature map.
            Used to compute the relative position bias in the
            attention layer. Defaults to `7`.
        name: String, name prefix for all layers in this block.

    Returns:
        Output tensor of shape `(batch_size, seq_len, channels)`.
    """
    channels_axis = -1

    # Attention
    y = layers.LayerNormalization(
        epsilon=1e-6, axis=channels_axis, name=f"{name}_norm1"
    )(inputs)
    y = Attention4D(dim=dim, resolution=resolution, name=f"{name}_attn")(y)
    y = LayerScale(layer_scale_init_value, name=f"{name}_ls1")(y)
    if drop_path > 0.0:
        y = StochasticDepth(drop_path, name=f"{name}_drop_path1")(y)
    x = layers.Add(name=f"{name}_add1")([inputs, y])

    # MLP
    y = layers.LayerNormalization(
        epsilon=1e-6, axis=channels_axis, name=f"{name}_norm2"
    )(x)
    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=drop,
        name=f"{name}_mlp",
    )
    y = LayerScale(layer_scale_init_value, name=f"{name}_ls2")(y)
    if drop_path > 0.0:
        y = StochasticDepth(drop_path, name=f"{name}_drop_path2")(y)
    outputs = layers.Add(name=f"{name}_add2")([x, y])
    return outputs


@keras.saving.register_keras_serializable(package="kmodels")
class EfficientFormer(keras.Model):
    """Instantiates the EfficientFormer architecture.

    EfficientFormer is a vision transformer that achieves MobileNet-level
    speed while maintaining high accuracy. It uses a dimension-consistent
    hybrid design: early stages apply 2D convolutional MetaBlocks with
    pooling-based token mixing, while the final stage flattens features
    into a 1D sequence and applies transformer MetaBlocks with multi-head
    self-attention. A knowledge-distillation head is averaged with the
    standard classification head at inference.

    Reference:
    - [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

    Args:
        depths: List of integers, number of MetaBlocks in each of the
            four stages.
        embed_dims: List of integers, channel dimensions for each
            stage.
        num_vit: Integer, number of transformer (1D) blocks at the end
            of the last stage. Defaults to `1`.
        mlp_ratio: Float, expansion ratio for hidden dimensions in MLP
            layers. Defaults to `4.0`.
        pool_size: Integer, kernel size for the average pooling token
            mixer in 2D MetaBlocks. Defaults to `3`.
        drop_rate: Float, dropout rate applied in MLP layers and before
            the classification head. Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic depth rate. Linearly
            increases from 0 to this value across all blocks.
            Defaults to `0.0`.
        layer_scale_init_value: Float, initial value for learnable
            layer scale parameters. Defaults to `1e-5`.
        include_top: Boolean, whether to include the fully-connected
            classification head. Defaults to `True`.
        as_backbone: Boolean, whether to return a list of intermediate
            feature maps (one per stage). Defaults to `False`.
        include_normalization: Boolean, whether to prepend an input
            normalization layer. Defaults to `True`.
        normalization_mode: String, normalization mode passed to
            `ImageNormalizationLayer` (e.g., `"imagenet"`).
            Defaults to `"imagenet"`.
        weights: String, one of `None` (random initialization), a
            weight identifier from the config, or a path to a weights
            file to load.
        input_shape: Optional tuple of integers specifying the input
            shape (excluding batch size), e.g., `(224, 224, 3)`.
        input_tensor: Optional Keras tensor to use as the model input.
        pooling: Optional pooling mode when `include_top=False`.
            One of `"avg"`, `"max"`, or `None`.
        num_classes: Integer, number of output classes.
            Defaults to `1000`.
        classifier_activation: String or callable, activation function
            for the classification layer. Defaults to `"softmax"`.
        name: String, the name of the model.
            Defaults to `"EfficientFormer"`.

    Returns:
        A `keras.Model` instance.
    """

    def __init__(
        self,
        depths,
        embed_dims,
        num_vit=1,
        mlp_ratio=4.0,
        pool_size=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights=None,
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=1000,
        classifier_activation="softmax",
        name="EfficientFormer",
        **kwargs,
    ):
        if include_top and num_classes is None:
            raise ValueError(
                "If `include_top` is True, `num_classes` must be specified. "
                f"Received: {num_classes}"
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

        # Stem: 4x downsampling
        # Use explicit padding to match PyTorch behavior (padding=1 for 3x3 conv stride 2)
        x = layers.ZeroPadding2D(padding=1, data_format=data_format, name="stem_pad1")(
            x
        )
        x = layers.Conv2D(
            embed_dims[0] // 2,
            kernel_size=3,
            strides=2,
            padding="valid",
            data_format=data_format,
            name="stem_conv1",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis, epsilon=1e-5, name="stem_norm1"
        )(x)
        x = layers.Activation("relu", name="stem_act1")(x)

        x = layers.ZeroPadding2D(padding=1, data_format=data_format, name="stem_pad2")(
            x
        )
        x = layers.Conv2D(
            embed_dims[0],
            kernel_size=3,
            strides=2,
            padding="valid",
            data_format=data_format,
            name="stem_conv2",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis, epsilon=1e-5, name="stem_norm2"
        )(x)
        x = layers.Activation("relu", name="stem_act2")(x)

        # Calculate drop path rates
        num_stages = len(depths)
        dpr = np.linspace(0.0, drop_path_rate, sum(depths))
        cur = 0

        # Build stages
        for i in range(num_stages):
            # Downsampling (except first stage)
            if i > 0:
                x = layers.ZeroPadding2D(
                    padding=1,
                    data_format=data_format,
                    name=f"stages_{i}_downsample_pad",
                )(x)
                x = layers.Conv2D(
                    embed_dims[i],
                    kernel_size=3,
                    strides=2,
                    padding="valid",
                    data_format=data_format,
                    name=f"stages_{i}_downsample_conv",
                )(x)
                x = layers.BatchNormalization(
                    axis=channels_axis, epsilon=1e-5, name=f"stages_{i}_downsample_norm"
                )(x)

            # Determine if this stage uses transformers
            is_last_stage = i == num_stages - 1
            use_transformer = is_last_stage and num_vit > 0

            # Add blocks
            for j in range(depths[i]):
                remain_idx = depths[i] - j - 1

                # Flatten before transformer blocks
                if (
                    use_transformer
                    and num_vit > remain_idx
                    and j == depths[i] - num_vit
                ):
                    x = layers.Reshape((-1, x.shape[-1]), name=f"stages_{i}_flat")(x)
                    # Calculate resolution for attention from current feature map
                    if data_format == "channels_last":
                        # Get spatial dimensions from the tensor shape
                        resolution = 224 // (4 * (2**i))  # Default calculation
                    else:
                        resolution = 224 // (4 * (2**i))

                # Choose block type
                if use_transformer and num_vit > remain_idx:
                    x = meta_block_1d(
                        x,
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        resolution=resolution,
                        name=f"stages_{i}_blocks_{j}",
                    )
                else:
                    x = meta_block_2d(
                        x,
                        dim=embed_dims[i],
                        pool_size=pool_size,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        data_format=data_format,
                        name=f"stages_{i}_blocks_{j}",
                    )

            features.append(x)
            cur += depths[i]

        # Head
        if include_top:
            x = layers.LayerNormalization(
                epsilon=1e-6, axis=channels_axis, name="final_norm"
            )(x)
            x = layers.Lambda(lambda v: keras.ops.mean(v, axis=1), name="global_pool")(
                x
            )
            x = layers.Dropout(drop_rate, name="head_drop")(x)

            # Distillation heads (EfficientFormer uses distillation)
            x_cls = layers.Dense(
                num_classes, activation=None, name="head", use_bias=True
            )(x)
            x_dist = layers.Dense(
                num_classes, activation=None, name="head_dist", use_bias=True
            )(x)

            # Average predictions
            x = layers.Average(name="avg_predictions")([x_cls, x_dist])
            if classifier_activation:
                x = layers.Activation(classifier_activation, name="predictions")(x)

        elif as_backbone:
            x = features
        else:
            if pooling == "avg":
                if len(x.shape) == 3:  # Already flattened
                    x = layers.Lambda(
                        lambda v: keras.ops.mean(v, axis=1), name="avg_pool"
                    )(x)
                else:
                    x = layers.GlobalAveragePooling2D(
                        data_format=data_format, name="avg_pool"
                    )(x)
            elif pooling == "max":
                if len(x.shape) == 3:
                    x = layers.Lambda(
                        lambda v: keras.ops.max(v, axis=1), name="max_pool"
                    )(x)
                else:
                    x = layers.GlobalMaxPooling2D(
                        data_format=data_format, name="max_pool"
                    )(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.depths = depths
        self.embed_dims = embed_dims
        self.num_vit = num_vit
        self.mlp_ratio = mlp_ratio
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
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
                "depths": self.depths,
                "embed_dims": self.embed_dims,
                "num_vit": self.num_vit,
                "mlp_ratio": self.mlp_ratio,
                "pool_size": self.pool_size,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
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
def EfficientFormerL1(
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
    name="EfficientFormerL1",
    **kwargs,
):
    model = EfficientFormer(
        **EFFICIENTFORMER_MODEL_CONFIG["l1"],
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

    if weights in get_all_weight_names(EFFICIENTFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientFormerL1", weights, model, EFFICIENTFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientFormerL3(
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
    name="EfficientFormerL3",
    **kwargs,
):
    model = EfficientFormer(
        **EFFICIENTFORMER_MODEL_CONFIG["l3"],
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

    if weights in get_all_weight_names(EFFICIENTFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientFormerL3", weights, model, EFFICIENTFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EfficientFormerL7(
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
    name="EfficientFormerL7",
    **kwargs,
):
    model = EfficientFormer(
        **EFFICIENTFORMER_MODEL_CONFIG["l7"],
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

    if weights in get_all_weight_names(EFFICIENTFORMER_WEIGHTS_CONFIG):
        load_weights_from_config(
            "EfficientFormerL7", weights, model, EFFICIENTFORMER_WEIGHTS_CONFIG
        )
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
