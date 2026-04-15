import keras
from keras import layers, ops, utils

from kmodels.layers import ImageNormalizationLayer
from kmodels.model_registry import register_model
from kmodels.models.detr.detr_layers import (
    DETRExpandQueryEmbedding,
    DETRFlattenFeatures,
    DETRMultiHeadAttention,
    DETRPositionEmbeddingSine,
)
from kmodels.weight_utils import load_weights_from_config

from .config import DETR_MODEL_CONFIG, DETR_WEIGHTS_CONFIG


def detr_encoder_layer(
    x,
    pos_embed,
    hidden_dim,
    num_heads,
    dim_feedforward,
    dropout_rate=0.1,
    block_prefix="encoder_layers_0",
):
    """Single DETR transformer encoder layer.

    Applies self-attention followed by a two-layer feedforward network,
    each with a residual connection, dropout, and post-norm layer
    normalization. Positional embeddings are added to the query and key
    inputs of the self-attention but not to the values, following the
    original DETR design.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        x: Input tensor of shape
            `(batch_size, seq_len, hidden_dim)`.
        pos_embed: Positional embedding tensor of shape
            `(batch_size, seq_len, hidden_dim)`, added to the query
            and key inputs of self-attention.
        hidden_dim: Integer, model dimension.
        num_heads: Integer, number of attention heads.
        dim_feedforward: Integer, intermediate dimension of the
            feedforward network.
        dropout_rate: Float, dropout rate applied after attention and
            each feedforward layer. Defaults to `0.1`.
        block_prefix: String, name prefix for all sub-layers in this
            block. Defaults to `"encoder_layers_0"`.

    Returns:
        Output tensor of shape `(batch_size, seq_len, hidden_dim)`.
    """
    self_attn = DETRMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_self_attn",
        name=f"{block_prefix}_self_attn",
    )

    q = k = layers.Add(name=f"{block_prefix}_sa_qk_add")([x, pos_embed])
    attn_output = self_attn(q, k, x)
    attn_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_sa_drop")(
        attn_output
    )
    x = layers.Add(name=f"{block_prefix}_sa_residual")([x, attn_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_self_attn_layer_norm",
    )(x)

    ff_output = layers.Dense(
        dim_feedforward,
        activation="relu",
        name=f"{block_prefix}_fc1",
    )(x)
    ff_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ff_drop")(ff_output)
    ff_output = layers.Dense(
        hidden_dim,
        name=f"{block_prefix}_fc2",
    )(ff_output)
    x = layers.Add(name=f"{block_prefix}_ff_residual")([x, ff_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_final_layer_norm",
    )(x)

    return x


def detr_decoder_layer(
    x,
    memory,
    pos_embed,
    query_pos,
    hidden_dim,
    num_heads,
    dim_feedforward,
    dropout_rate=0.1,
    block_prefix="decoder_layers_0",
):
    """Single DETR transformer decoder layer.

    Applies masked self-attention on object queries, cross-attention
    between object queries and the encoder memory, and a two-layer
    feedforward network. Each sub-block uses a residual connection,
    dropout, and post-norm layer normalization. Positional embeddings
    are added to the query/key inputs of both attention operations.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        x: Decoder input tensor of shape
            `(batch_size, num_queries, hidden_dim)`.
        memory: Encoder output tensor of shape
            `(batch_size, seq_len, hidden_dim)`.
        pos_embed: Encoder positional embedding tensor of shape
            `(batch_size, seq_len, hidden_dim)`, added to the key
            in cross-attention.
        query_pos: Object query positional embedding tensor of shape
            `(batch_size, num_queries, hidden_dim)`, added to the
            query in both self-attention and cross-attention.
        hidden_dim: Integer, model dimension.
        num_heads: Integer, number of attention heads.
        dim_feedforward: Integer, intermediate dimension of the
            feedforward network.
        dropout_rate: Float, dropout rate applied after attention and
            each feedforward layer. Defaults to `0.1`.
        block_prefix: String, name prefix for all sub-layers in this
            block. Defaults to `"decoder_layers_0"`.

    Returns:
        Output tensor of shape
        `(batch_size, num_queries, hidden_dim)`.
    """
    self_attn = DETRMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_self_attn",
        name=f"{block_prefix}_self_attn",
    )

    q = k = layers.Add(name=f"{block_prefix}_sa_qk_add")([x, query_pos])
    attn_output = self_attn(q, k, x)
    attn_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_sa_drop")(
        attn_output
    )
    x = layers.Add(name=f"{block_prefix}_sa_residual")([x, attn_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_self_attn_layer_norm",
    )(x)

    cross_attn = DETRMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_encoder_attn",
        name=f"{block_prefix}_encoder_attn",
    )

    q_cross = layers.Add(name=f"{block_prefix}_ca_q_add")([x, query_pos])
    k_cross = layers.Add(name=f"{block_prefix}_ca_k_add")([memory, pos_embed])
    cross_output = cross_attn(q_cross, k_cross, memory)
    cross_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ca_drop")(
        cross_output
    )
    x = layers.Add(name=f"{block_prefix}_ca_residual")([x, cross_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_encoder_attn_layer_norm",
    )(x)

    ff_output = layers.Dense(
        dim_feedforward,
        activation="relu",
        name=f"{block_prefix}_fc1",
    )(x)
    ff_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ff_drop")(ff_output)
    ff_output = layers.Dense(
        hidden_dim,
        name=f"{block_prefix}_fc2",
    )(ff_output)
    x = layers.Add(name=f"{block_prefix}_ff_residual")([x, ff_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_final_layer_norm",
    )(x)

    return x


def build_detr_backbone(
    input_tensor,
    backbone_variant,
    include_normalization,
    normalization_mode,
    data_format="channels_last",
    channels_axis=-1,
):
    """Build a ResNet backbone for DETR feature extraction.

    Constructs a standard ResNet-50 or ResNet-101 bottleneck backbone
    from scratch with layer naming that mirrors the HuggingFace DETR
    backbone structure, enabling direct weight transfer from pretrained
    HuggingFace checkpoints without name remapping for the backbone
    layers.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    Args:
        input_tensor: Keras input tensor of shape
            `(batch_size, height, width, 3)`.
        backbone_variant: String, one of `"ResNet50"` or
            `"ResNet101"`. Determines the number of bottleneck blocks
            per stage.
        include_normalization: Boolean, whether to prepend an
            `ImageNormalizationLayer` for input preprocessing.
        normalization_mode: String, normalization mode passed to
            `ImageNormalizationLayer` (e.g., `"imagenet"`).
        data_format: String, image data format. Defaults to
            `"channels_last"`.
        channels_axis: Integer, channel axis index. Defaults to ``-1``.

    Returns:
        Feature tensor from the last ResNet stage (C5).
    """

    block_repeats = {
        "ResNet50": [3, 4, 6, 3],
        "ResNet101": [3, 4, 23, 3],
    }[backbone_variant]

    x = (
        ImageNormalizationLayer(mode=normalization_mode)(input_tensor)
        if include_normalization
        else input_tensor
    )

    # Stem
    x = layers.ZeroPadding2D(padding=3, data_format=data_format)(x)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name="backbone_bn1",
    )(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="valid",
        data_format=data_format,
    )(x)

    filters_list = [64, 128, 256, 512]

    for stage_idx, num_blocks in enumerate(block_repeats):
        filters = filters_list[stage_idx]
        for block_idx in range(num_blocks):
            prefix = f"backbone_layer{stage_idx + 1}_{block_idx}"
            strides = 2 if block_idx == 0 and stage_idx > 0 else 1
            residual = x

            # Conv 1x1 reduce
            x = layers.Conv2D(
                filters,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv1",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn1",
            )(x)
            x = layers.ReLU()(x)

            # Conv 3x3
            if strides > 1:
                x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=strides,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv2",
                )(x)
            else:
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv2",
                )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn2",
            )(x)
            x = layers.ReLU()(x)

            # Conv 1x1 expand
            x = layers.Conv2D(
                filters * 4,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv3",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn3",
            )(x)

            # Downsample residual if needed
            in_channels = residual.shape[channels_axis]
            out_channels = filters * 4
            if strides != 1 or in_channels != out_channels:
                if strides > 1:
                    residual = layers.ZeroPadding2D(padding=0, data_format=data_format)(
                        residual
                    )
                residual = layers.Conv2D(
                    out_channels,
                    1,
                    strides=strides,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_downsample_conv",
                )(residual)
                residual = layers.BatchNormalization(
                    axis=channels_axis,
                    epsilon=1e-5,
                    momentum=0.1,
                    name=f"{prefix}_downsample_bn",
                )(residual)

            x = layers.Add()([x, residual])
            x = layers.ReLU()(x)

    return x


@keras.saving.register_keras_serializable(package="kmodels")
class DETR(keras.Model):
    """DEtection TRansformer (DETR) for end-to-end object detection.

    DETR treats object detection as a direct set prediction problem
    using a transformer encoder-decoder architecture with bipartite
    matching loss. It eliminates the need for hand-designed components
    like non-maximum suppression, anchor generation, or region proposal
    networks. The architecture consists of a CNN backbone (ResNet) for
    feature extraction, a 1x1 projection, fixed sinusoidal positional
    embeddings, a transformer encoder that processes the flattened
    feature map, and a transformer decoder that attends to encoder
    output using learned object queries. Two prediction heads output
    class logits and normalized bounding box coordinates for each
    query.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        hidden_dim: Integer, transformer hidden dimension and the
            dimension of all internal projections. Defaults to `256`.
        num_heads: Integer, number of attention heads per transformer
            layer. Defaults to `8`.
        num_encoder_layers: Integer, number of transformer encoder
            layers. Defaults to `6`.
        num_decoder_layers: Integer, number of transformer decoder
            layers. Defaults to `6`.
        dim_feedforward: Integer, intermediate dimension of the
            feedforward networks in each transformer layer.
            Defaults to `2048`.
        dropout_rate: Float, dropout rate applied in all transformer
            layers. Defaults to `0.1`.
        num_queries: Integer, number of learned object queries
            (maximum detections per image). Defaults to `100`.
        num_classes: Integer, number of object classes including the
            no-object class. Defaults to `92` (COCO: 91 + 1).
        backbone_variant: String, ResNet variant for the backbone.
            One of `"ResNet50"` or `"ResNet101"`.
            Defaults to `"ResNet50"`.
        include_normalization: Boolean, whether to prepend an input
            normalization layer. Defaults to `True`.
        normalization_mode: String, normalization mode passed to
            `ImageNormalizationLayer` (e.g., `"imagenet"`).
            Defaults to `"imagenet"`.
        weights: String, one of `None` (random initialization), a
            weight identifier from the config (e.g., `"coco"`), or
            a path to a weights file to load.
            Defaults to `"coco"`.
        input_shape: Optional tuple of integers specifying the input
            shape (excluding batch size), e.g., `(800, 800, 3)`.
        input_tensor: Optional Keras tensor to use as the model input.
        name: String, the name of the model.
            Defaults to `"DETR"`.
        **kwargs: Additional keyword arguments passed to the
            `keras.Model` class.

    Returns:
        A `keras.Model` instance with dict outputs:
        - `"logits"`: `(batch_size, num_queries, num_classes)`
        - `"pred_boxes"`: `(batch_size, num_queries, 4)` in
          `[cx, cy, w, h]` format, normalized to `[0, 1]`.
    """

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout_rate=0.1,
        num_queries=100,
        num_classes=92,
        backbone_variant="ResNet50",
        include_normalization=True,
        normalization_mode="imagenet",
        weights="coco",
        input_shape=None,
        input_tensor=None,
        name="DETR",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (800, 800, 3)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        backbone_features = build_detr_backbone(
            inputs,
            backbone_variant=backbone_variant,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        projected = layers.Conv2D(
            hidden_dim,
            1,
            padding="valid",
            data_format=data_format,
            name="input_projection",
        )(backbone_features)

        pos_embed = DETRPositionEmbeddingSine(
            hidden_dim=hidden_dim,
            name="position_embedding",
        )(projected)

        src = DETRFlattenFeatures(hidden_dim, name="flatten_src")(projected)
        pos = DETRFlattenFeatures(hidden_dim, name="flatten_pos")(pos_embed)

        encoder_output = src
        for i in range(num_encoder_layers):
            encoder_output = detr_encoder_layer(
                encoder_output,
                pos,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout_rate=dropout_rate,
                block_prefix=f"encoder_layers_{i}",
            )

        query_embed_layer = DETRExpandQueryEmbedding(
            num_queries,
            hidden_dim,
            name="query_position_embeddings",
        )
        query_embed = query_embed_layer(encoder_output)

        decoder_input = ops.zeros_like(query_embed)

        decoder_output = decoder_input
        for i in range(num_decoder_layers):
            decoder_output = detr_decoder_layer(
                decoder_output,
                encoder_output,
                pos,
                query_embed,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout_rate=dropout_rate,
                block_prefix=f"decoder_layers_{i}",
            )

        decoder_output = layers.LayerNormalization(
            epsilon=1e-5,
            name="decoder_layernorm",
        )(decoder_output)

        logits = layers.Dense(
            num_classes,
            name="class_labels_classifier",
        )(decoder_output)

        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_0")(
            decoder_output
        )
        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_1")(
            bbox
        )
        bbox = layers.Dense(4, name="bbox_predictor_2")(bbox)
        bbox = layers.Activation("sigmoid", name="bbox_sigmoid")(bbox)

        outputs = {"logits": logits, "pred_boxes": bbox}

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.backbone_variant = backbone_variant
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "backbone_variant": self.backbone_variant,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_detr_model(
    variant,
    num_queries=100,
    num_classes=92,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name=None,
    **kwargs,
):
    """Factory function for creating DETR model variants.

    Looks up the architecture configuration for the given variant
    name, instantiates a `DETR` model, and optionally loads pretrained
    weights from the configured URL or a local file path.

    Args:
        variant: String, model variant name (e.g., `"DETRResNet50"`).
        num_queries: Integer, number of object queries. Defaults to
            `100`.
        num_classes: Integer, number of object classes (COCO default:
            91 + 1 no-object). Defaults to `92`.
        include_normalization: Boolean, whether to include an input
            normalization layer. Defaults to `True`.
        normalization_mode: String, normalization mode.
            Defaults to `"imagenet"`.
        weights: String, one of `None`, a weight identifier from the
            config (e.g., `"coco"`), or a path to a weights file.
            Defaults to `"coco"`.
        input_shape: Optional tuple of integers specifying the input
            shape. Defaults to `(800, 800, 3)`.
        input_tensor: Optional Keras tensor to use as the model input.
        name: String, the name of the model.
        **kwargs: Additional keyword arguments passed to `DETR`.

    Returns:
        A configured `DETR` model instance.
    """
    config = DETR_MODEL_CONFIG[variant]

    if input_shape is None:
        input_shape = (800, 800, 3)

    model = DETR(
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout_rate=config["dropout_rate"],
        num_queries=num_queries,
        num_classes=num_classes,
        backbone_variant=config["backbone_variant"],
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name or variant,
        **kwargs,
    )

    if weights in DETR_WEIGHTS_CONFIG.get(variant, {}):
        url = DETR_WEIGHTS_CONFIG[variant][weights].get("url", "")
        if url:
            load_weights_from_config(variant, weights, model, DETR_WEIGHTS_CONFIG)
        else:
            print(
                f"Weight URL for '{weights}' is not yet available. "
                "Use the conversion script to generate weights."
            )
    elif weights is not None and weights != "coco":
        model.load_weights(weights)
    else:
        if weights == "coco":
            print(
                "COCO weights URL not yet configured. "
                "Run convert_detr_torch_to_keras.py to generate weights, "
                "then pass the .weights.h5 file path."
            )

    return model


@register_model
def DETRResNet50(
    num_queries=100,
    num_classes=92,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="DETRResNet50",
    **kwargs,
):
    return _create_detr_model(
        "DETRResNet50",
        num_queries=num_queries,
        num_classes=num_classes,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def DETRResNet101(
    num_queries=100,
    num_classes=92,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="DETRResNet101",
    **kwargs,
):
    return _create_detr_model(
        "DETRResNet101",
        num_queries=num_queries,
        num_classes=num_classes,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )
