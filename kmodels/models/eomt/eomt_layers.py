import keras
from keras import layers, ops

from kmodels.layers import StochasticDepth


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
