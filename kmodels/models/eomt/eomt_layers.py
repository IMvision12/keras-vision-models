import keras
from keras import layers, ops


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
