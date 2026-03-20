import keras
from keras import layers, ops, utils

from kmodels.layers import StochasticDepth
from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import EOMT_MODEL_CONFIG, EOMT_WEIGHTS_CONFIG


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTLayerScale(layers.Layer):
    """Learnable per-channel scaling factor for residual connections.

    Args:
        init_value: Initial value for the scale parameters.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, init_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="lambda1",
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTPatchEmbeddings(layers.Layer):
    """Converts pixel values to patch embeddings using a Conv2D projection.

    Args:
        hidden_size: Embedding dimension.
        patch_size: Size of each patch.
        num_channels: Number of input channels.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, patch_size=16, num_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection = layers.Conv2D(
            hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=True,
            name="projection",
        )

    def call(self, pixel_values):
        # pixel_values: (B, H, W, C) in channels_last
        x = self.projection(pixel_values)  # (B, H/P, W/P, hidden_size)
        shape = ops.shape(x)
        x = ops.reshape(x, (shape[0], shape[1] * shape[2], shape[3]))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTEmbeddings(layers.Layer):
    """Constructs CLS token, register tokens, position embeddings and patch embeddings.

    Args:
        hidden_size: Embedding dimension.
        patch_size: Size of each patch.
        image_size: Input image resolution.
        num_register_tokens: Number of register tokens.
        num_channels: Number of input channels.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size,
        patch_size=16,
        image_size=640,
        num_register_tokens=4,
        num_channels=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_register_tokens = num_register_tokens
        self.num_channels = num_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.num_prefix_tokens = 1 + num_register_tokens  # CLS + register tokens

        self.patch_embeddings = EoMTPatchEmbeddings(
            hidden_size, patch_size, num_channels, name="patch_embeddings"
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.hidden_size),
            initializer="random_normal",
            trainable=True,
        )
        self.register_tokens = self.add_weight(
            name="register_tokens",
            shape=(1, self.num_register_tokens, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, pixel_values):
        batch_size = ops.shape(pixel_values)[0]

        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings + self.position_embeddings

        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.hidden_size))
        register_tokens = ops.broadcast_to(
            self.register_tokens,
            (batch_size, self.num_register_tokens, self.hidden_size),
        )

        embeddings = ops.concatenate([cls_tokens, register_tokens, embeddings], axis=1)
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "num_register_tokens": self.num_register_tokens,
                "num_channels": self.num_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTAttention(layers.Layer):
    """Multi-head self-attention.

    Args:
        hidden_size: Total dimension.
        num_heads: Number of attention heads.
        attention_dropout: Dropout rate for attention weights.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, num_heads, attention_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.attention_dropout = attention_dropout

        self.q_proj = layers.Dense(hidden_size, name="q_proj")
        self.k_proj = layers.Dense(hidden_size, name="k_proj")
        self.v_proj = layers.Dense(hidden_size, name="v_proj")
        self.out_proj = layers.Dense(hidden_size, name="out_proj")

    def call(self, hidden_states, training=False):
        batch_size = ops.shape(hidden_states)[0]
        seq_length = ops.shape(hidden_states)[1]

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = ops.reshape(
            queries, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        queries = ops.transpose(queries, (0, 2, 1, 3))

        keys = ops.reshape(
            keys, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        keys = ops.transpose(keys, (0, 2, 1, 3))

        values = ops.reshape(
            values, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        values = ops.transpose(values, (0, 2, 1, 3))

        attn_weights = ops.matmul(queries, ops.transpose(keys, (0, 1, 3, 2)))
        attn_weights = attn_weights * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)

        if training and self.attention_dropout > 0:
            attn_weights = layers.Dropout(self.attention_dropout)(
                attn_weights, training=training
            )

        attn_output = ops.matmul(attn_weights, values)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_length, self.hidden_size)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTMLP(layers.Layer):
    """Standard MLP with GELU activation.

    Args:
        hidden_size: Input/output dimension.
        mlp_ratio: Ratio for hidden dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        hidden_features = int(hidden_size * mlp_ratio)

        self.fc1 = layers.Dense(hidden_features, name="fc1")
        self.fc2 = layers.Dense(hidden_size, name="fc2")

    def call(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = ops.gelu(hidden_states, approximate=False)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTSwiGLUFFN(layers.Layer):
    """SwiGLU feed-forward network used in DINOv2-large.

    Args:
        hidden_size: Input/output dimension.
        mlp_ratio: Ratio for computing hidden dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        hidden_features = int(hidden_size * mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = layers.Dense(2 * hidden_features, name="weights_in")
        self.weights_out = layers.Dense(hidden_size, name="weights_out")
        self._hidden_features = hidden_features

    def call(self, hidden_states):
        hidden_states = self.weights_in(hidden_states)
        x1 = hidden_states[..., : self._hidden_features]
        x2 = hidden_states[..., self._hidden_features :]
        hidden = ops.silu(x1) * x2
        return self.weights_out(hidden)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTLayer(layers.Layer):
    """Single transformer encoder layer with LayerScale and DropPath.

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        layerscale_value: Initial LayerScale value.
        drop_path_rate: Stochastic depth rate.
        attention_dropout: Attention dropout rate.
        use_swiglu_ffn: Whether to use SwiGLU FFN.
        layer_norm_eps: Epsilon for LayerNorm.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        use_swiglu_ffn=False,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.use_swiglu_ffn = use_swiglu_ffn
        self.layer_norm_eps = layer_norm_eps

        self.norm1 = layers.LayerNormalization(epsilon=layer_norm_eps, name="norm1")
        self.attention = EoMTAttention(
            hidden_size, num_heads, attention_dropout, name="attention"
        )
        self.layer_scale1 = EoMTLayerScale(
            init_value=layerscale_value, name="layer_scale1"
        )
        self.drop_path = (
            StochasticDepth(drop_path_rate)
            if drop_path_rate > 0.0
            else layers.Identity()
        )

        self.norm2 = layers.LayerNormalization(epsilon=layer_norm_eps, name="norm2")

        if use_swiglu_ffn:
            self.mlp = EoMTSwiGLUFFN(hidden_size, mlp_ratio, name="mlp")
        else:
            self.mlp = EoMTMLP(hidden_size, mlp_ratio, name="mlp")

        self.layer_scale2 = EoMTLayerScale(
            init_value=layerscale_value, name="layer_scale2"
        )

    def call(self, hidden_states, training=False):
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, training=training)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states, training=training) + residual

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states, training=training) + residual

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "layerscale_value": self.layerscale_value,
                "drop_path_rate": self.drop_path_rate,
                "attention_dropout": self.attention_dropout,
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTQueryInjection(layers.Layer):
    """Injects learned query tokens by concatenating them with hidden states.

    Args:
        num_queries: Number of query tokens.
        hidden_size: Hidden dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, num_queries, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.query_weight = self.add_weight(
            name="weight",
            shape=(self.num_queries, self.hidden_size),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, hidden_states):
        batch_size = ops.shape(hidden_states)[0]
        query_tokens = ops.broadcast_to(
            ops.expand_dims(self.query_weight, axis=0),
            (batch_size, self.num_queries, self.hidden_size),
        )
        return ops.concatenate([query_tokens, hidden_states], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_queries": self.num_queries,
                "hidden_size": self.hidden_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTLayerNorm2d(layers.Layer):
    """LayerNorm applied to 2D feature maps (channels-last: B,H,W,C).

    Args:
        num_channels: Number of channels to normalize.
        epsilon: Small constant for numerical stability.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, num_channels, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.norm = layers.LayerNormalization(epsilon=epsilon, name="norm")

    def call(self, hidden_states):
        # hidden_states: (B, H, W, C) in channels_last
        return self.norm(hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "epsilon": self.epsilon,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTScaleLayer(layers.Layer):
    """Upscaling layer: ConvTranspose2d(2x) + GELU + DepthwiseConv2d + LayerNorm2d.

    Args:
        hidden_size: Number of channels.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

        self.conv1 = layers.Conv2DTranspose(
            hidden_size,
            kernel_size=2,
            strides=2,
            padding="valid",
            use_bias=True,
            name="conv1",
        )
        self.conv2 = layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="conv2",
        )
        self.layernorm2d = EoMTLayerNorm2d(hidden_size, name="layernorm2d")

    def call(self, hidden_states):
        # hidden_states: (B, H, W, C)
        hidden_states = self.conv1(hidden_states)
        hidden_states = ops.gelu(hidden_states, approximate=False)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layernorm2d(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTScaleBlock(layers.Layer):
    """Stack of upscaling layers.

    Args:
        hidden_size: Number of channels.
        num_upscale_blocks: Number of upscaling layers.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, num_upscale_blocks=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_upscale_blocks = num_upscale_blocks
        self.blocks = [
            EoMTScaleLayer(hidden_size, name=f"block_{i}")
            for i in range(num_upscale_blocks)
        ]

    def call(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_upscale_blocks": self.num_upscale_blocks,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTMaskHead(layers.Layer):
    """Mask prediction head: 3 Dense layers with GELU activations.

    Transforms query tokens into mask embedding vectors that are used
    to predict per-pixel masks via dot product with upscaled features.

    Args:
        hidden_size: Hidden dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.fc1 = layers.Dense(hidden_size, name="fc1")
        self.fc2 = layers.Dense(hidden_size, name="fc2")
        self.fc3 = layers.Dense(hidden_size, name="fc3")

    def call(self, hidden_states):
        hidden_states = ops.gelu(self.fc1(hidden_states), approximate=False)
        hidden_states = ops.gelu(self.fc2(hidden_states), approximate=False)
        hidden_states = self.fc3(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class EoMT(keras.Model):
    """Encoder-only Mask Transformer (EoMT) for universal image segmentation.

    EoMT repurposes a plain Vision Transformer for image segmentation without
    task-specific decoder components. Learned queries are injected into the final
    encoder blocks, enabling joint attention between image patches and object queries.

    Architecture:
        1. DINOv2-style ViT encoder with CLS + register tokens
        2. Learned object queries injected at layer (num_hidden_layers - num_blocks)
        3. Mask prediction via bilinear product of query embeddings and upscaled features
        4. Class prediction via linear projection of query tokens

    Reference:
        - [Your ViT is Secretly an Image Segmentation Model]
          (https://arxiv.org/abs/2503.19108) (Kerssies et al., CVPR 2025)

    Args:
        hidden_size: Transformer hidden dimension.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        patch_size: Patch size for embedding.
        num_register_tokens: Number of register tokens.
        num_blocks: Number of final blocks with query injection.
        num_upscale_blocks: Number of upscaling layers in mask predictor.
        num_queries: Number of learned object queries.
        num_labels: Number of segmentation classes.
        layerscale_value: Initial LayerScale value.
        drop_path_rate: Stochastic depth rate.
        attention_dropout: Attention dropout rate.
        use_swiglu_ffn: Whether to use SwiGLU FFN.
        layer_norm_eps: LayerNorm epsilon.
        input_shape: Input shape (H, W, C).
        input_tensor: Optional input tensor.
        name: Model name.
        **kwargs: Additional arguments.

    Example:
        ```python
        model = EoMT(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_queries=200,
            num_labels=133,
            input_shape=(640, 640, 3),
        )
        ```
    """

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4,
        patch_size=16,
        num_register_tokens=4,
        num_blocks=4,
        num_upscale_blocks=2,
        num_queries=200,
        num_labels=133,
        layerscale_value=1e-5,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        use_swiglu_ffn=False,
        layer_norm_eps=1e-6,
        input_shape=None,
        input_tensor=None,
        name="EoMT",
        **kwargs,
    ):
        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        else:
            img_input = layers.Input(shape=input_shape)

        image_size = input_shape[0]
        grid_h = image_size // patch_size
        grid_w = image_size // patch_size
        num_prefix_tokens = 1 + num_register_tokens

        # Embeddings
        embeddings_layer = EoMTEmbeddings(
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_size=image_size,
            num_register_tokens=num_register_tokens,
            name="embeddings",
        )
        hidden_states = embeddings_layer(img_input)

        # Query injection layer
        query_injection = EoMTQueryInjection(num_queries, hidden_size, name="query")

        # Transformer layers
        query_injection_idx = num_hidden_layers - num_blocks

        for i in range(num_hidden_layers):
            if i == query_injection_idx:
                hidden_states = query_injection(hidden_states)

            hidden_states = EoMTLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                layerscale_value=layerscale_value,
                drop_path_rate=drop_path_rate,
                attention_dropout=attention_dropout,
                use_swiglu_ffn=use_swiglu_ffn,
                layer_norm_eps=layer_norm_eps,
                name=f"layers_{i}",
            )(hidden_states)

        # Final layer norm
        layernorm = layers.LayerNormalization(epsilon=layer_norm_eps, name="layernorm")
        sequence_output = layernorm(hidden_states)

        # Predict masks and classes
        # Extract query tokens and patch tokens
        query_output = sequence_output[:, :num_queries, :]
        patch_output = sequence_output[:, num_queries + num_prefix_tokens :, :]

        # Class prediction
        class_predictor = layers.Dense(num_labels + 1, name="class_predictor")
        class_logits = class_predictor(query_output)

        # Mask prediction
        mask_head = EoMTMaskHead(hidden_size, name="mask_head")
        query_mask_tokens = mask_head(query_output)

        # Reshape patch tokens to spatial grid
        patch_spatial = ops.reshape(patch_output, (-1, grid_h, grid_w, hidden_size))

        # Upscale
        upscale_block = EoMTScaleBlock(
            hidden_size, num_upscale_blocks, name="upscale_block"
        )
        upscaled_features = upscale_block(patch_spatial)

        # Mask logits via einsum: (B, Q, C) x (B, H, W, C) -> (B, Q, H, W)
        mask_logits = ops.einsum("bqc,bhwc->bqhw", query_mask_tokens, upscaled_features)

        super().__init__(
            inputs=img_input,
            outputs={"class_logits": class_logits, "mask_logits": mask_logits},
            name=name,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.num_blocks = num_blocks
        self.num_upscale_blocks = num_upscale_blocks
        self.num_queries = num_queries
        self.num_labels = num_labels
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.use_swiglu_ffn = use_swiglu_ffn
        self.layer_norm_eps = layer_norm_eps
        self._input_shape = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "mlp_ratio": self.mlp_ratio,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "num_blocks": self.num_blocks,
                "num_upscale_blocks": self.num_upscale_blocks,
                "num_queries": self.num_queries,
                "num_labels": self.num_labels,
                "layerscale_value": self.layerscale_value,
                "drop_path_rate": self.drop_path_rate,
                "attention_dropout": self.attention_dropout,
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "layer_norm_eps": self.layer_norm_eps,
                "input_shape": self._input_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_eomt_model(
    variant,
    num_queries=200,
    num_labels=None,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    """Creates an EoMT model with the specified variant.

    Args:
        variant: Model variant name (e.g., "EoMT_Large").
        num_queries: Number of object queries.
        num_labels: Number of segmentation classes.
        input_shape: Input shape (H, W, C).
        input_tensor: Optional input tensor.
        weights: Pretrained weights identifier or file path.
        **kwargs: Additional arguments.

    Returns:
        Configured EoMT model.
    """
    config = EOMT_MODEL_CONFIG[variant]

    valid_model_weights = []
    if variant in EOMT_WEIGHTS_CONFIG:
        valid_model_weights = list(EOMT_WEIGHTS_CONFIG[variant].keys())

    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights for {variant} are "
            f"{', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )

    DATASET_CLASSES = {
        "coco_panoptic": 133,
        "coco_instance": 80,
        "ade20k_semantic": 150,
    }

    if num_labels is None:
        if weights is not None and isinstance(weights, str):
            for dataset_key, n_classes in DATASET_CLASSES.items():
                if dataset_key in weights:
                    num_labels = n_classes
                    print(f"Using {num_labels} classes for {dataset_key}.")
                    break
        if num_labels is None:
            raise ValueError(
                "num_labels must be specified when not using dataset-specific weights."
            )

    if input_shape is None:
        if weights is not None and isinstance(weights, str):
            if "512" in weights:
                input_shape = (512, 512, 3)
            elif "1280" in weights:
                input_shape = (1280, 1280, 3)
            else:
                input_shape = (640, 640, 3)
        else:
            input_shape = (640, 640, 3)
        print(f"Using default input shape {input_shape}.")

    model = EoMT(
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        mlp_ratio=config["mlp_ratio"],
        patch_size=config["patch_size"],
        num_register_tokens=config["num_register_tokens"],
        num_blocks=config["num_blocks"],
        num_upscale_blocks=config["num_upscale_blocks"],
        num_queries=num_queries,
        num_labels=num_labels,
        layerscale_value=config["layerscale_value"],
        use_swiglu_ffn=config["use_swiglu_ffn"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **kwargs,
    )

    if weights in valid_model_weights:
        print(f"Loading {weights} weights for {variant}.")
        load_weights_from_config(variant, weights, model, EOMT_WEIGHTS_CONFIG)
    elif weights is not None and isinstance(weights, str):
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def EoMT_Small(
    num_queries=200,
    num_labels=None,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_eomt_model(
        "EoMT_Small",
        num_queries=num_queries,
        num_labels=num_labels,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def EoMT_Base(
    num_queries=200,
    num_labels=None,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_eomt_model(
        "EoMT_Base",
        num_queries=num_queries,
        num_labels=num_labels,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def EoMT_Large(
    num_queries=200,
    num_labels=None,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_eomt_model(
        "EoMT_Large",
        num_queries=num_queries,
        num_labels=num_labels,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )
