import math

import keras
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import DETR_MODEL_CONFIG, DETR_WEIGHTS_CONFIG


@keras.saving.register_keras_serializable(package="kmodels")
class DETRExpandQueryEmbedding(layers.Layer):
    """Expands query embeddings to match the batch dimension of the input.

    Takes learned query embeddings of shape ``(num_queries, hidden_dim)``
    and tiles them along a new batch axis to produce
    ``(batch_size, num_queries, hidden_dim)``.

    Args:
        num_queries: Number of object queries.
        hidden_dim: Embedding dimension.
    """

    def __init__(self, num_queries, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.embedding = layers.Embedding(
            num_queries,
            hidden_dim,
            name="embedding",
        )

    def call(self, batch_ref):
        batch_size = ops.shape(batch_ref)[0]
        indices = ops.arange(self.num_queries)
        query_embed = self.embedding(indices)
        query_embed = ops.expand_dims(query_embed, axis=0)
        query_embed = ops.tile(query_embed, [batch_size, 1, 1])
        return query_embed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_queries": self.num_queries,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DETRFlattenFeatures(layers.Layer):
    """Flattens spatial feature maps for transformer input.

    Reshapes ``(B, H, W, C)`` to ``(B, H*W, C)``.

    Args:
        hidden_dim: Channel dimension (used for the reshape target).
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, inputs):
        shape = ops.shape(inputs)
        return ops.reshape(inputs, [shape[0], shape[1] * shape[2], self.hidden_dim])

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DETRPositionEmbeddingSine(layers.Layer):
    """Fixed sinusoidal 2D position embedding used in the DETR encoder.

    Generates sine/cosine positional encodings for spatial feature maps,
    matching the original DETR implementation (facebook/detr).

    Args:
        hidden_dim: Total embedding dimension. Half is used for row
            embeddings, half for column embeddings.
        temperature: Temperature scaling for the sinusoidal frequencies.
        normalize: Whether to normalize position coordinates to [0, 2*pi].
        eps: Small epsilon to avoid division by zero during normalization.
    """

    def __init__(
        self,
        hidden_dim=256,
        temperature=10000,
        normalize=True,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        self.num_pos_feats = hidden_dim // 2

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]

        y_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, h + 1, dtype="float32"), axis=1),
                w,
                axis=1,
            ),
            dtype="float32",
        )
        x_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, w + 1, dtype="float32"), axis=0),
                h,
                axis=0,
            ),
            dtype="float32",
        )

        if self.normalize:
            y_embed = y_embed / (y_embed[-1:, :] + self.eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, -1:] + self.eps) * 2 * math.pi

        dim_t = ops.arange(self.num_pos_feats, dtype="float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        pos_x_sin = ops.sin(pos_x[:, :, 0::2])
        pos_x_cos = ops.cos(pos_x[:, :, 1::2])
        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos_y_sin = ops.sin(pos_y[:, :, 0::2])
        pos_y_cos = ops.cos(pos_y[:, :, 1::2])
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.expand_dims(pos, axis=0)
        pos = ops.broadcast_to(pos, [batch_size, h, w, self.hidden_dim])

        return pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DETRMultiHeadAttention(layers.Layer):
    """Multi-head attention layer for DETR transformer.

    Implements standard scaled dot-product multi-head attention with
    separate Q, K, V projections matching the HuggingFace DETR layout.

    Args:
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for attention weights.
        block_prefix: Name prefix for sub-layers.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        block_prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout_rate
        self.block_prefix = block_prefix

        self.q_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_q_proj",
        )
        self.k_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_k_proj",
        )
        self.v_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_v_proj",
        )
        self.out_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_out_proj",
        )
        self.attn_dropout = layers.Dropout(dropout_rate)

    def call(self, query, key, value, training=None):
        batch_size = ops.shape(query)[0]
        seq_len_q = ops.shape(query)[1]
        seq_len_k = ops.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = ops.reshape(q, [batch_size, seq_len_q, self.num_heads, self.head_dim])
        k = ops.reshape(k, [batch_size, seq_len_k, self.num_heads, self.head_dim])
        v = ops.reshape(v, [batch_size, seq_len_k, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [batch_size, seq_len_q, self.hidden_dim])
        attn_output = self.out_proj(attn_output)

        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DETREncoderLayer(layers.Layer):
    """Single DETR transformer encoder layer.

    Implements pre-norm self-attention + feedforward with residual connections,
    matching the HuggingFace DETR encoder layer structure.

    Args:
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        dim_feedforward: FFN intermediate dimension.
        dropout_rate: Dropout rate.
        block_prefix: Name prefix for sub-layers.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.1,
        block_prefix="encoder_layer_0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.block_prefix = block_prefix

        self.self_attn = DETRMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            block_prefix=f"{block_prefix}_self_attn",
            name=f"{block_prefix}_self_attn",
        )

        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{block_prefix}_self_attn_layer_norm",
        )

        self.fc1 = layers.Dense(
            dim_feedforward,
            activation="relu",
            name=f"{block_prefix}_fc1",
        )
        self.fc2 = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_fc2",
        )
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{block_prefix}_final_layer_norm",
        )
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, pos_embed, training=None):
        q = k = x + pos_embed
        attn_output = self.self_attn(q, k, x, training=training)
        x = x + self.dropout1(attn_output, training=training)
        x = self.self_attn_layer_norm(x)

        ff_output = self.fc1(x)
        ff_output = self.dropout2(ff_output, training=training)
        ff_output = self.fc2(ff_output)
        x = x + ff_output
        x = self.final_layer_norm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DETRDecoderLayer(layers.Layer):
    """Single DETR transformer decoder layer.

    Implements self-attention on object queries, cross-attention with encoder
    output, and a feedforward block, each with residual connections and
    layer normalization.

    Args:
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        dim_feedforward: FFN intermediate dimension.
        dropout_rate: Dropout rate.
        block_prefix: Name prefix for sub-layers.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.1,
        block_prefix="decoder_layer_0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.block_prefix = block_prefix

        self.self_attn = DETRMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            block_prefix=f"{block_prefix}_self_attn",
            name=f"{block_prefix}_self_attn",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{block_prefix}_self_attn_layer_norm",
        )

        self.cross_attn = DETRMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            block_prefix=f"{block_prefix}_encoder_attn",
            name=f"{block_prefix}_encoder_attn",
        )
        self.cross_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{block_prefix}_encoder_attn_layer_norm",
        )

        self.fc1 = layers.Dense(
            dim_feedforward,
            activation="relu",
            name=f"{block_prefix}_fc1",
        )
        self.fc2 = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_fc2",
        )
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{block_prefix}_final_layer_norm",
        )

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, memory, pos_embed, query_pos, training=None):
        # Self attention on object queries
        q = k = x + query_pos
        attn_output = self.self_attn(q, k, x, training=training)
        x = x + self.dropout1(attn_output, training=training)
        x = self.self_attn_layer_norm(x)

        # Cross attention: queries attend to encoder memory
        q = x + query_pos
        k = memory + pos_embed
        cross_output = self.cross_attn(q, k, memory, training=training)
        x = x + self.dropout2(cross_output, training=training)
        x = self.cross_attn_layer_norm(x)

        # Feedforward
        ff_output = self.fc1(x)
        ff_output = self.dropout3(ff_output, training=training)
        ff_output = self.fc2(ff_output)
        x = x + ff_output
        x = self.final_layer_norm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config


def build_detr_backbone(
    input_tensor,
    backbone_variant,
    include_normalization,
    normalization_mode,
):
    """Build a frozen ResNet backbone for DETR.

    Instead of reusing kmodels.models.resnet (which would add weight-loading
    complexity during conversion), this builds a standard ResNet backbone
    from scratch with naming that mirrors the HuggingFace DETR backbone,
    making weight transfer straightforward.

    Args:
        input_tensor: Input Keras tensor.
        backbone_variant: One of "ResNet50" or "ResNet101".
        include_normalization: Whether to add ImageNormalizationLayer.
        normalization_mode: Normalization mode string.

    Returns:
        Feature tensor from the last ResNet stage (C5).
    """
    from kmodels.layers import ImageNormalizationLayer

    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

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

    DETR treats object detection as a direct set prediction problem using a
    transformer encoder-decoder architecture. It eliminates the need for
    hand-designed components like non-maximum suppression or anchor generation.

    Architecture:
        1. A CNN backbone (ResNet) extracts spatial features from the input image.
        2. A 1x1 convolution projects backbone features to the transformer dimension.
        3. Fixed sinusoidal positional embeddings encode spatial information.
        4. A transformer encoder processes the flattened feature map.
        5. A transformer decoder attends to encoder output using learned object queries.
        6. Prediction heads output class labels and bounding boxes for each query.

    Reference:
        - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
          (Carion et al., ECCV 2020)

    Args:
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads per layer.
        num_encoder_layers: Number of transformer encoder layers.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: FFN intermediate dimension.
        dropout_rate: Dropout rate for transformer layers.
        num_queries: Number of object queries (max detections per image).
        num_classes: Number of object classes (including background/no-object).
        backbone_variant: ResNet variant for the backbone ("ResNet50" or "ResNet101").
        include_normalization: Whether to include an image normalization layer.
        normalization_mode: Normalization mode if ``include_normalization`` is True.
        weights: Pre-trained weight identifier, file path, or None.
        input_shape: Input image shape as ``(H, W, C)``.
        input_tensor: Optional existing Keras tensor for the model input.
        name: Model name.
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
        data_format = keras.config.image_data_format()

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

        # --- Backbone ---
        backbone_features = build_detr_backbone(
            inputs,
            backbone_variant=backbone_variant,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
        )

        # --- Input projection ---
        projected = layers.Conv2D(
            hidden_dim,
            1,
            padding="valid",
            data_format=data_format,
            name="input_projection",
        )(backbone_features)

        # --- Position embedding ---
        pos_embed = DETRPositionEmbeddingSine(
            hidden_dim=hidden_dim,
            name="position_embedding",
        )(projected)

        # Flatten spatial dims: (B, H, W, C) -> (B, H*W, C)
        src = DETRFlattenFeatures(hidden_dim, name="flatten_src")(projected)
        pos = DETRFlattenFeatures(hidden_dim, name="flatten_pos")(pos_embed)

        # --- Transformer Encoder ---
        encoder_output = src
        for i in range(num_encoder_layers):
            encoder_output = DETREncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout_rate=dropout_rate,
                block_prefix=f"encoder_layers_{i}",
                name=f"encoder_layers_{i}",
            )(encoder_output, pos)

        # --- Object queries (learned embeddings) ---
        query_embed_layer = DETRExpandQueryEmbedding(
            num_queries,
            hidden_dim,
            name="query_position_embeddings",
        )
        query_embed = query_embed_layer(encoder_output)

        # Decoder input: zeros
        decoder_input = ops.zeros_like(query_embed)

        # --- Transformer Decoder ---
        decoder_output = decoder_input
        for i in range(num_decoder_layers):
            decoder_output = DETRDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout_rate=dropout_rate,
                block_prefix=f"decoder_layers_{i}",
                name=f"decoder_layers_{i}",
            )(decoder_output, encoder_output, pos, query_embed)

        decoder_output = layers.LayerNormalization(
            epsilon=1e-5,
            name="decoder_layernorm",
        )(decoder_output)

        # --- Prediction heads ---
        # Class prediction: (B, num_queries, num_classes)
        # For COCO: num_classes=92 (91 object categories + 1 "no object")
        logits = layers.Dense(
            num_classes,
            name="class_labels_classifier",
        )(decoder_output)

        # Bbox prediction: (B, num_queries, 4) with sigmoid
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

        # Store config
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

    Args:
        variant: Model variant name (e.g., "DETRResNet50").
        num_queries: Number of object queries.
        num_classes: Number of object classes (COCO default: 91 + 1 no-object).
        include_normalization: Whether to include image normalization layer.
        normalization_mode: Normalization mode string.
        weights: Pre-trained weights identifier, file path, or None.
        input_shape: Input image shape.
        input_tensor: Optional input tensor.
        name: Model name.
        **kwargs: Additional keyword arguments.

    Returns:
        A configured DETR model instance.
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
    """DETR with ResNet-50 backbone.

    Pre-trained on COCO 2017 object detection (91 classes + 1 no-object).

    Args:
        num_queries: Number of object queries. Default 100.
        num_classes: Number of object classes (COCO: 91 + 1 background). Default 92.
        include_normalization: Whether to normalize input images. Default True.
        normalization_mode: Normalization mode. Default "imagenet".
        weights: Pre-trained weights identifier or file path. Default "coco".
        input_shape: Input shape. Default (800, 800, 3).
        input_tensor: Optional input tensor.
        name: Model name.

    Returns:
        A DETR model with ResNet-50 backbone.

    Example:
        ```python
        model = kmodels.models.detr.DETRResNet50(weights=None)
        output = model(np.random.rand(1, 800, 800, 3).astype("float32"))
        print(output["logits"].shape)      # (1, 100, 92)
        print(output["pred_boxes"].shape)   # (1, 100, 4)
        ```
    """
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
    """DETR with ResNet-101 backbone.

    Pre-trained on COCO 2017 object detection (91 classes + 1 no-object).

    Args:
        num_queries: Number of object queries. Default 100.
        num_classes: Number of object classes (COCO: 91 + 1 background). Default 92.
        include_normalization: Whether to normalize input images. Default True.
        normalization_mode: Normalization mode. Default "imagenet".
        weights: Pre-trained weights identifier or file path. Default "coco".
        input_shape: Input shape. Default (800, 800, 3).
        input_tensor: Optional input tensor.
        name: Model name.

    Returns:
        A DETR model with ResNet-101 backbone.
    """
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
