import math

import keras
from keras import initializers, layers, ops

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _get_activation(name):
    if name == "silu":
        return layers.Activation("silu")
    if name == "relu":
        return layers.Activation("relu")
    if name == "gelu":
        return layers.Activation("gelu")
    raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# Channel-last LayerNorm for 4D tensors (B, H, W, C)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="kmodels")
class ChannelLayerNorm(layers.Layer):
    """LayerNorm applied over the channel dimension of (B, H, W, C) tensors."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer="ones",
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(dim,),
            initializer="zeros",
        )

    def call(self, x):
        mean = ops.mean(x, axis=-1, keepdims=True)
        variance = ops.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / ops.sqrt(variance + self.epsilon)
        return x_norm * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


# ---------------------------------------------------------------------------
# ConvBN: Conv2D + Norm + Activation  (channels-last)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="kmodels")
class ConvBN(layers.Layer):
    """Conv2D + LayerNorm/BatchNorm + Activation (channels-last)."""

    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        groups=1,
        activation="relu",
        use_layer_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.strides = strides
        self.groups = groups
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm

    def build(self, input_shape):
        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.pad = layers.ZeroPadding2D(padding=padding)
        self.conv = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding="valid",
            groups=self.groups,
            use_bias=False,
            data_format="channels_last",
            name="conv",
        )
        if self.use_layer_norm:
            self.norm = ChannelLayerNorm(name="ln")
        else:
            self.norm = layers.BatchNormalization(
                axis=-1, epsilon=1e-5, momentum=0.1, name="bn"
            )
        self.act = _get_activation(self.activation_name)
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.pad(x)
        x = self.conv(x)
        if self.use_layer_norm:
            x = self.norm(x)
        else:
            x = self.norm(x, training=training)
        x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "groups": self.groups,
                "activation": self.activation_name,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Bottleneck + C2f (CSP Bottleneck with 2 convolutions)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="kmodels")
class Bottleneck(layers.Layer):
    def __init__(
        self,
        out_channels,
        shortcut=True,
        expansion=1.0,
        activation="silu",
        use_layer_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.shortcut = shortcut
        self.expansion = expansion
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm

    def build(self, input_shape):
        hidden = int(self.out_channels * self.expansion)
        in_channels = input_shape[-1]
        self.cv1 = ConvBN(
            hidden,
            3,
            activation=self.activation_name,
            use_layer_norm=self.use_layer_norm,
            name="cv1",
        )
        self.cv2 = ConvBN(
            self.out_channels,
            3,
            activation=self.activation_name,
            use_layer_norm=self.use_layer_norm,
            name="cv2",
        )
        self.add_shortcut = self.shortcut and in_channels == self.out_channels
        super().build(input_shape)

    def call(self, x, training=None):
        out = self.cv1(x, training=training)
        out = self.cv2(out, training=training)
        if self.add_shortcut:
            out = out + x
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "shortcut": self.shortcut,
                "expansion": self.expansion,
                "activation": self.activation_name,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class C2f(layers.Layer):
    """CSP Bottleneck with 2 convolutions (faster implementation)."""

    def __init__(
        self,
        out_channels,
        num_blocks=1,
        shortcut=False,
        expansion=0.5,
        activation="silu",
        use_layer_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.shortcut = shortcut
        self.expansion = expansion
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.c = int(self.out_channels * self.expansion)
        self.cv1 = ConvBN(
            2 * self.c,
            1,
            activation=self.activation_name,
            use_layer_norm=self.use_layer_norm,
            name="cv1",
        )
        self.cv2 = ConvBN(
            self.out_channels,
            1,
            activation=self.activation_name,
            use_layer_norm=self.use_layer_norm,
            name="cv2",
        )
        self.bottlenecks = []
        for i in range(self.num_blocks):
            self.bottlenecks.append(
                Bottleneck(
                    self.c,
                    shortcut=self.shortcut,
                    expansion=1.0,
                    activation=self.activation_name,
                    use_layer_norm=self.use_layer_norm,
                    name=f"bottleneck_{i}",
                )
            )
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.cv1(x, training=training)
        chunks = ops.split(x, 2, axis=-1)
        y = [chunks[0], chunks[1]]
        for m in self.bottlenecks:
            y.append(m(y[-1], training=training))
        return self.cv2(ops.concatenate(y, axis=-1), training=training)

    def compute_output_spec(self, input_spec, **kwargs):
        return keras.KerasTensor(
            shape=(
                input_spec.shape[0],
                input_spec.shape[1],
                input_spec.shape[2],
                self.out_channels,
            ),
            dtype=input_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "num_blocks": self.num_blocks,
                "shortcut": self.shortcut,
                "expansion": self.expansion,
                "activation": self.activation_name,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


# ---------------------------------------------------------------------------
# SimpleProjector (used by all open-source RF-DETR variants)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="kmodels")
class SimpleProjector(layers.Layer):
    """Two ConvBN blocks + LayerNorm projector for single-scale features."""

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.convx1 = ConvBN(
            in_dim * 2, 3, activation="silu", use_layer_norm=True, name="convx1"
        )
        self.convx2 = ConvBN(
            self.out_channels, 3, activation="silu", use_layer_norm=True, name="convx2"
        )
        self.ln = ChannelLayerNorm(name="ln")
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.convx1(x, training=training)
        x = self.convx2(x, training=training)
        x = self.ln(x)
        return x

    def compute_output_spec(self, input_spec, **kwargs):
        return keras.KerasTensor(
            shape=(
                input_spec.shape[0],
                input_spec.shape[1],
                input_spec.shape[2],
                self.out_channels,
            ),
            dtype=input_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels})
        return config


# ---------------------------------------------------------------------------
# DINOv2 with Windowed Attention  (backbone)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2PatchEmbeddings(layers.Layer):
    """Convert pixel values to patch embeddings via a convolution."""

    def __init__(self, hidden_size, patch_size, num_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels

    def build(self, input_shape):
        self.projection = layers.Conv2D(
            self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            name="projection",
        )
        super().build(input_shape)

    def call(self, pixel_values):
        x = self.projection(pixel_values)
        shape = ops.shape(x)
        x = ops.reshape(x, [shape[0], shape[1] * shape[2], self.hidden_size])
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
class DinoV2Embeddings(layers.Layer):
    """CLS token, register tokens, patch embeddings, position embeddings, windowing."""

    def __init__(
        self,
        hidden_size,
        patch_size,
        num_channels=3,
        num_register_tokens=4,
        num_windows=1,
        positional_encoding_size=37,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_register_tokens = num_register_tokens
        self.num_windows = num_windows
        self.positional_encoding_size = positional_encoding_size
        self.num_patches = positional_encoding_size * positional_encoding_size

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.hidden_size),
            initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches + 1, self.hidden_size),
            initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        if self.num_register_tokens > 0:
            self.register_tokens = self.add_weight(
                name="register_tokens",
                shape=(1, self.num_register_tokens, self.hidden_size),
                initializer="zeros",
            )
        self.patch_embeddings = DinoV2PatchEmbeddings(
            self.hidden_size,
            self.patch_size,
            self.num_channels,
            name="patch_embeddings",
        )
        super().build(input_shape)

    def _interpolate_pos_encoding(self, embeddings, height, width):
        num_patches = ops.shape(embeddings)[1] - 1
        num_positions = self.num_patches

        class_pos_embed = self.position_embeddings[:, :1, :]
        patch_pos_embed = self.position_embeddings[:, 1:, :]

        h = height // self.patch_size
        w = width // self.patch_size
        sqrt_n = int(math.sqrt(num_positions))

        patch_pos_embed = ops.reshape(
            patch_pos_embed, [1, sqrt_n, sqrt_n, self.hidden_size]
        )
        patch_pos_embed = ops.cast(patch_pos_embed, "float32")
        patch_pos_embed = ops.image.resize(
            patch_pos_embed, (h, w), interpolation="bicubic", antialias=True
        )
        patch_pos_embed = ops.cast(patch_pos_embed, embeddings.dtype)
        patch_pos_embed = ops.reshape(patch_pos_embed, [1, h * w, self.hidden_size])
        return ops.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def call(self, pixel_values):
        batch_size = ops.shape(pixel_values)[0]
        height = ops.shape(pixel_values)[1]
        width = ops.shape(pixel_values)[2]

        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.hidden_size))
        embeddings = ops.concatenate([cls_tokens, embeddings], axis=1)

        pos_embed = self._interpolate_pos_encoding(embeddings, height, width)
        embeddings = embeddings + pos_embed

        if self.num_windows > 1:
            num_h = height // self.patch_size
            num_w = width // self.patch_size
            cls_tok = embeddings[:, :1, :]
            patch_tok = embeddings[:, 1:, :]
            patch_tok = ops.reshape(
                patch_tok, [batch_size, num_h, num_w, self.hidden_size]
            )
            nw = self.num_windows
            h_pw = num_h // nw
            w_pw = num_w // nw
            patch_tok = ops.reshape(
                patch_tok, [batch_size * nw, h_pw, nw, w_pw, self.hidden_size]
            )
            patch_tok = ops.transpose(patch_tok, [0, 2, 1, 3, 4])
            patch_tok = ops.reshape(
                patch_tok, [batch_size * nw * nw, h_pw * w_pw, self.hidden_size]
            )
            cls_tok = ops.tile(cls_tok, [nw * nw, 1, 1])
            embeddings = ops.concatenate([cls_tok, patch_tok], axis=1)

        if self.num_register_tokens > 0:
            reg = ops.broadcast_to(
                self.register_tokens,
                (ops.shape(embeddings)[0], self.num_register_tokens, self.hidden_size),
            )
            embeddings = ops.concatenate(
                [embeddings[:, :1, :], reg, embeddings[:, 1:, :]], axis=1
            )

        return embeddings

    def compute_output_spec(self, input_spec, **kwargs):
        h = input_spec.shape[1] // self.patch_size
        w = input_spec.shape[2] // self.patch_size
        tokens_per_window = (
            (h // self.num_windows) * (w // self.num_windows)
            if self.num_windows > 1
            else h * w
        )
        seq_len = tokens_per_window + 1 + self.num_register_tokens
        batch = input_spec.shape[0]
        if self.num_windows > 1:
            batch = None
        return keras.KerasTensor(
            shape=(batch, seq_len, self.hidden_size),
            dtype=input_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
                "num_register_tokens": self.num_register_tokens,
                "num_windows": self.num_windows,
                "positional_encoding_size": self.positional_encoding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2SwiGLUFFN(layers.Layer):
    def __init__(self, hidden_size, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio

    def build(self, input_shape):
        hidden_features = int(self.hidden_size * self.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.weights_in = layers.Dense(2 * hidden_features, name="weights_in")
        self.weights_out = layers.Dense(self.hidden_size, name="weights_out")
        super().build(input_shape)

    def call(self, x):
        x = self.weights_in(x)
        x1, x2 = ops.split(x, 2, axis=-1)
        hidden = ops.silu(x1) * x2
        return self.weights_out(hidden)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "mlp_ratio": self.mlp_ratio})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2MLP(layers.Layer):
    def __init__(self, hidden_size, mlp_ratio=4, hidden_act="gelu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act

    def build(self, input_shape):
        hidden_features = int(self.hidden_size * self.mlp_ratio)
        self.fc1 = layers.Dense(hidden_features, name="fc1")
        self.fc2 = layers.Dense(self.hidden_size, name="fc2")
        super().build(input_shape)

    def call(self, x):
        x = self.fc1(x)
        if self.hidden_act == "gelu":
            x = ops.gelu(x, approximate=False)
        else:
            x = ops.relu(x)
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "mlp_ratio": self.mlp_ratio,
                "hidden_act": self.hidden_act,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2Attention(layers.Layer):
    """Multi-head self-attention with separate Q, K, V projections."""

    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def build(self, input_shape):
        self.query = layers.Dense(self.hidden_size, name="query")
        self.key = layers.Dense(self.hidden_size, name="key")
        self.value = layers.Dense(self.hidden_size, name="value")
        self.out_proj = layers.Dense(self.hidden_size, name="out_proj")
        super().build(input_shape)

    def call(self, x):
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = ops.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = ops.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = ops.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        scale = ops.cast(self.head_dim, q.dtype) ** -0.5
        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * scale
        attn_weights = ops.softmax(attn_weights, axis=-1)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [batch_size, seq_len, self.hidden_size])

        return self.out_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2LayerScale(layers.Layer):
    def __init__(self, hidden_size, init_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.init_value = init_value

    def build(self, input_shape):
        self.lambda1 = self.add_weight(
            name="lambda1",
            shape=(self.hidden_size,),
            initializer=initializers.Constant(self.init_value),
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.lambda1

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = ops.random.uniform(shape, dtype=x.dtype)
        random_tensor = ops.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class WindowedDinoV2Block(layers.Layer):
    """Single DINOv2 transformer block with windowed attention support."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4,
        use_swiglu=False,
        drop_path_rate=0.0,
        layer_scale_init=1.0,
        num_windows=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_swiglu = use_swiglu
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init = layer_scale_init
        self.num_windows = num_windows

    def build(self, input_shape):
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attention = DinoV2Attention(
            self.hidden_size, self.num_heads, name="attention"
        )
        self.layer_scale1 = DinoV2LayerScale(
            self.hidden_size, self.layer_scale_init, name="layer_scale1"
        )
        self.drop_path = DinoV2DropPath(self.drop_path_rate, name="drop_path")

        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        if self.use_swiglu:
            self.mlp = DinoV2SwiGLUFFN(self.hidden_size, self.mlp_ratio, name="mlp")
        else:
            self.mlp = DinoV2MLP(self.hidden_size, self.mlp_ratio, name="mlp")
        self.layer_scale2 = DinoV2LayerScale(
            self.hidden_size, self.layer_scale_init, name="layer_scale2"
        )
        super().build(input_shape)

    def call(self, hidden_states, run_full_attention=False, training=None):
        shortcut = hidden_states

        if run_full_attention and self.num_windows > 1:
            nw2 = self.num_windows**2
            shape = ops.shape(hidden_states)
            hidden_states = ops.reshape(hidden_states, [-1, nw2 * shape[1], shape[2]])

        attn_out = self.attention(self.norm1(hidden_states))

        if run_full_attention and self.num_windows > 1:
            full_shape = ops.shape(attn_out)
            attn_out = ops.reshape(attn_out, [-1, full_shape[1] // nw2, full_shape[2]])

        attn_out = self.layer_scale1(attn_out)
        hidden_states = self.drop_path(attn_out, training=training) + shortcut

        layer_output = self.mlp(self.norm2(hidden_states))
        layer_output = self.layer_scale2(layer_output)
        layer_output = self.drop_path(layer_output, training=training) + hidden_states

        return layer_output

    def compute_output_spec(self, input_spec, **kwargs):
        return keras.KerasTensor(
            shape=input_spec.shape,
            dtype=input_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "use_swiglu": self.use_swiglu,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init": self.layer_scale_init,
                "num_windows": self.num_windows,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class WindowedDinoV2Encoder(layers.Layer):
    """Stack of windowed DINOv2 blocks with multi-scale feature extraction."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_layers,
        mlp_ratio=4,
        use_swiglu=False,
        drop_path_rate=0.0,
        layer_scale_init=1.0,
        num_windows=1,
        out_feature_indexes=None,
        window_block_indexes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.use_swiglu = use_swiglu
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init = layer_scale_init
        self.num_windows = num_windows
        self.out_feature_indexes = out_feature_indexes or []
        if window_block_indexes is None:
            all_indexes = set(
                range(
                    max(self.out_feature_indexes) + 1
                    if self.out_feature_indexes
                    else num_layers
                )
            )
            all_indexes.difference_update(set(self.out_feature_indexes))
            self.window_block_indexes = sorted(all_indexes)
        else:
            self.window_block_indexes = window_block_indexes

    def build(self, input_shape):
        max_layer = (
            max(self.out_feature_indexes) + 1
            if self.out_feature_indexes
            else self.num_layers
        )
        self.blocks = []
        for i in range(max_layer):
            block = WindowedDinoV2Block(
                self.hidden_size,
                self.num_heads,
                self.mlp_ratio,
                use_swiglu=self.use_swiglu,
                drop_path_rate=self.drop_path_rate,
                layer_scale_init=self.layer_scale_init,
                num_windows=self.num_windows,
                name=f"layer_{i}",
            )
            self.blocks.append(block)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6, name="layernorm")
        super().build(input_shape)

    def call(self, hidden_states, training=None):
        features = []
        for i, block in enumerate(self.blocks):
            run_full = i not in self.window_block_indexes
            hidden_states = block(
                hidden_states, run_full_attention=run_full, training=training
            )
            if (i + 1) in self.out_feature_indexes:
                features.append(self.layernorm(hidden_states))
        return features

    def compute_output_spec(self, input_spec, **kwargs):
        seq_len = input_spec.shape[1]
        output_specs = []
        for _ in self.out_feature_indexes:
            output_specs.append(
                keras.KerasTensor(
                    shape=(input_spec.shape[0], seq_len, self.hidden_size),
                    dtype=input_spec.dtype,
                )
            )
        return output_specs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "mlp_ratio": self.mlp_ratio,
                "use_swiglu": self.use_swiglu,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init": self.layer_scale_init,
                "num_windows": self.num_windows,
                "out_feature_indexes": self.out_feature_indexes,
                "window_block_indexes": self.window_block_indexes,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Sinusoidal 2D Position Encoding (for decoder feature maps)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="kmodels")
class PositionEmbeddingSine(layers.Layer):
    """2D sinusoidal position embedding for feature maps."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2.0 * math.pi

    def call(self, x):
        shape = ops.shape(x)
        h, w = shape[1], shape[2]

        y_range = ops.cast(ops.arange(h), "float32")
        x_range = ops.cast(ops.arange(w), "float32")

        if self.normalize:
            eps = 1e-6
            y_range = y_range / (ops.cast(h, "float32") - 1 + eps) * self.scale
            x_range = x_range / (ops.cast(w, "float32") - 1 + eps) * self.scale
        else:
            y_range = (y_range + 1) * self.scale
            x_range = (x_range + 1) * self.scale

        dim_t = ops.cast(ops.arange(self.num_pos_feats), "float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_range[:, None] / dim_t[None, :]
        pos_y = y_range[:, None] / dim_t[None, :]

        pos_x_sin = ops.sin(pos_x[:, 0::2])
        pos_x_cos = ops.cos(pos_x[:, 1::2])
        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=-1), [w, self.num_pos_feats]
        )

        pos_y_sin = ops.sin(pos_y[:, 0::2])
        pos_y_cos = ops.cos(pos_y[:, 1::2])
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=-1), [h, self.num_pos_feats]
        )

        pos_y = ops.expand_dims(pos_y, axis=1)
        pos_y = ops.tile(pos_y, [1, w, 1])
        pos_x = ops.expand_dims(pos_x, axis=0)
        pos_x = ops.tile(pos_x, [h, 1, 1])
        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.expand_dims(pos, axis=0)
        return pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "temperature": self.temperature,
                "normalize": self.normalize,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention (pure Keras implementation)
# ---------------------------------------------------------------------------


def _ms_deform_attn_core(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pure implementation of multi-scale deformable attention core.

    Args:
        value: (B, n_heads, head_dim, N_total)
        value_spatial_shapes: list of (H, W) tuples
        sampling_locations: (B, Len_q, n_heads, L, P, 2) in [0, 1]
        attention_weights: (B, Len_q, n_heads, L*P) after softmax
    """
    B = ops.shape(value)[0]
    n_heads = ops.shape(value)[1]
    head_dim = ops.shape(value)[2]
    Len_q = ops.shape(sampling_locations)[1]
    L = len(value_spatial_shapes)
    P = ops.shape(sampling_locations)[4]

    sampling_grids = 2 * sampling_locations - 1

    splits = [h * w for h, w in value_spatial_shapes]
    value_list = ops.split(value, splits, axis=3)

    sampling_value_list = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        value_l = ops.reshape(value_list[lid], [B * n_heads, head_dim, H, W])
        value_l = ops.transpose(value_l, [0, 2, 3, 1])
        val_flat = ops.reshape(value_l, [B * n_heads, H * W, head_dim])

        grid_l = sampling_grids[:, :, :, lid, :, :]
        grid_l = ops.transpose(grid_l, [0, 2, 1, 3, 4])
        grid_l = ops.reshape(grid_l, [B * n_heads, Len_q, P, 2])

        grid_x = grid_l[..., 0]
        grid_y = grid_l[..., 1]

        W_f = ops.cast(W, grid_x.dtype)
        H_f = ops.cast(H, grid_y.dtype)
        ix = ((grid_x + 1) * W_f - 1) / 2.0
        iy = ((grid_y + 1) * H_f - 1) / 2.0

        ix0 = ops.cast(ops.floor(ix), "int32")
        iy0 = ops.cast(ops.floor(iy), "int32")
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        fx = ix - ops.cast(ix0, ix.dtype)
        fy = iy - ops.cast(iy0, iy.dtype)

        valid_00 = ops.cast((ix0 >= 0) & (ix0 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype)
        valid_01 = ops.cast((ix1 >= 0) & (ix1 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype)
        valid_10 = ops.cast((ix0 >= 0) & (ix0 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype)
        valid_11 = ops.cast((ix1 >= 0) & (ix1 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype)

        ix0_c = ops.clip(ix0, 0, W - 1)
        ix1_c = ops.clip(ix1, 0, W - 1)
        iy0_c = ops.clip(iy0, 0, H - 1)
        iy1_c = ops.clip(iy1, 0, H - 1)

        def _gather(val_flat, iy, ix, BN, Len_q, P, H, W, head_dim):
            idx = iy * W + ix
            idx_flat = ops.reshape(idx, [BN, Len_q * P])
            idx_flat = ops.expand_dims(idx_flat, axis=-1)
            idx_flat = ops.repeat(idx_flat, head_dim, axis=-1)
            gathered = ops.take_along_axis(val_flat, idx_flat, axis=1)
            return ops.reshape(gathered, [BN, Len_q, P, head_dim])

        BN = B * n_heads
        v00 = _gather(val_flat, iy0_c, ix0_c, BN, Len_q, P, H, W, head_dim)
        v01 = _gather(val_flat, iy0_c, ix1_c, BN, Len_q, P, H, W, head_dim)
        v10 = _gather(val_flat, iy1_c, ix0_c, BN, Len_q, P, H, W, head_dim)
        v11 = _gather(val_flat, iy1_c, ix1_c, BN, Len_q, P, H, W, head_dim)

        v00 = v00 * ops.expand_dims(valid_00, axis=-1)
        v01 = v01 * ops.expand_dims(valid_01, axis=-1)
        v10 = v10 * ops.expand_dims(valid_10, axis=-1)
        v11 = v11 * ops.expand_dims(valid_11, axis=-1)

        fx = ops.expand_dims(fx, axis=-1)
        fy = ops.expand_dims(fy, axis=-1)

        w00 = (1 - fx) * (1 - fy)
        w01 = fx * (1 - fy)
        w10 = (1 - fx) * fy
        w11 = fx * fy

        sampled = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
        sampled = ops.transpose(sampled, [0, 3, 1, 2])
        sampling_value_list.append(sampled)

    sampling_values = ops.stack(sampling_value_list, axis=-2)
    sampling_values = ops.reshape(
        sampling_values, [B * n_heads, head_dim, Len_q, L * P]
    )

    attn = ops.transpose(attention_weights, [0, 2, 1, 3])
    attn = ops.reshape(attn, [B * n_heads, 1, Len_q, L * P])

    output = ops.sum(sampling_values * attn, axis=-1)
    output = ops.reshape(output, [B, n_heads * head_dim, Len_q])
    output = ops.transpose(output, [0, 2, 1])
    return output


@keras.saving.register_keras_serializable(package="kmodels")
class MSDeformableAttention(layers.Layer):
    """Multi-Scale Deformable Attention Module."""

    def __init__(
        self,
        d_model=256,
        n_levels=1,
        n_heads=8,
        n_points=4,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.spatial_shapes = spatial_shapes or []
        self.level_start_index = level_start_index or [0]

    def build(self, input_shape):
        self.sampling_offsets = layers.Dense(
            self.n_heads * self.n_levels * self.n_points * 2,
            name="sampling_offsets",
        )
        self.attention_weights_proj = layers.Dense(
            self.n_heads * self.n_levels * self.n_points,
            name="attention_weights",
        )
        self.value_proj = layers.Dense(self.d_model, name="value_proj")
        self.output_proj = layers.Dense(self.d_model, name="output_proj")
        super().build(input_shape)

    def call(self, query, reference_points, input_flatten, input_padding_mask=None):
        input_spatial_shapes = self.spatial_shapes
        N = ops.shape(query)[0]
        Len_q = ops.shape(query)[1]
        Len_in = ops.shape(input_flatten)[1]

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            mask = ops.expand_dims(ops.cast(input_padding_mask, value.dtype), axis=-1)
            value = value * (1.0 - mask)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = ops.reshape(
            sampling_offsets, [N, Len_q, self.n_heads, self.n_levels, self.n_points, 2]
        )

        attention_weights = self.attention_weights_proj(query)
        attention_weights = ops.reshape(
            attention_weights, [N, Len_q, self.n_heads, self.n_levels * self.n_points]
        )
        attention_weights = ops.softmax(attention_weights, axis=-1)

        if reference_points.shape[-1] == 2:
            spatial_shapes_wh = [[w, h] for h, w in input_spatial_shapes]
            offset_normalizer = ops.cast(
                ops.convert_to_tensor(spatial_shapes_wh, dtype="float32"),
                sampling_offsets.dtype,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )

        value = ops.transpose(value, [0, 2, 1])
        value = ops.reshape(
            value, [N, self.n_heads, self.d_model // self.n_heads, Len_in]
        )

        output = _ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)

    def compute_output_spec(self, query_spec, *args, **kwargs):
        return keras.KerasTensor(
            shape=(query_spec.shape[0], query_spec.shape[1], self.d_model),
            dtype=query_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "n_levels": self.n_levels,
                "n_heads": self.n_heads,
                "n_points": self.n_points,
                "spatial_shapes": self.spatial_shapes,
                "level_start_index": self.level_start_index,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Decoder Layer (Self-Attn + MS Deformable Cross-Attn + FFN)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="kmodels")
class RFDETRDecoderLayer(layers.Layer):
    def __init__(
        self,
        d_model,
        sa_nhead,
        ca_nhead,
        dim_feedforward=2048,
        dropout=0.0,
        num_feature_levels=1,
        dec_n_points=4,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.sa_nhead = sa_nhead
        self.ca_nhead = ca_nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.num_feature_levels = num_feature_levels
        self.dec_n_points = dec_n_points
        self.spatial_shapes = spatial_shapes or []
        self.level_start_index = level_start_index or [0]

    def build(self, input_shape):
        self.self_attn_q = layers.Dense(self.d_model, name="self_attn_q_proj")
        self.self_attn_k = layers.Dense(self.d_model, name="self_attn_k_proj")
        self.self_attn_v = layers.Dense(self.d_model, name="self_attn_v_proj")
        self.self_attn_out = layers.Dense(self.d_model, name="self_attn_out_proj")
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.dropout1 = layers.Dropout(self.dropout_rate)

        self.cross_attn = MSDeformableAttention(
            d_model=self.d_model,
            n_levels=self.num_feature_levels,
            n_heads=self.ca_nhead,
            n_points=self.dec_n_points,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
            name="cross_attn",
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.dropout2 = layers.Dropout(self.dropout_rate)

        self.linear1 = layers.Dense(self.dim_feedforward, name="linear1")
        self.linear2 = layers.Dense(self.d_model, name="linear2")
        self.norm3 = layers.LayerNormalization(epsilon=1e-5, name="norm3")
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.dropout_ffn = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def _self_attention(self, q, k, v, training=None):
        batch = ops.shape(q)[0]
        seq_len = ops.shape(q)[1]
        head_dim = self.d_model // self.sa_nhead

        q = self.self_attn_q(q)
        k = self.self_attn_k(k)
        v = self.self_attn_v(v)

        q = ops.reshape(q, [batch, seq_len, self.sa_nhead, head_dim])
        k = ops.reshape(k, [batch, seq_len, self.sa_nhead, head_dim])
        v = ops.reshape(v, [batch, seq_len, self.sa_nhead, head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        scale = ops.cast(head_dim, q.dtype) ** -0.5
        attn = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * scale
        attn = ops.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, [0, 2, 1, 3])
        out = ops.reshape(out, [batch, seq_len, self.d_model])
        return self.self_attn_out(out)

    def call(
        self,
        tgt,
        memory,
        query_pos,
        reference_points,
        memory_key_padding_mask=None,
        training=None,
    ):
        q = k = tgt + query_pos
        v = tgt

        tgt2 = self._self_attention(q, k, v, training=training)
        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            tgt + query_pos,
            reference_points,
            memory,
            input_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)

        tgt2 = self.linear1(tgt)
        tgt2 = ops.relu(tgt2)
        tgt2 = self.dropout_ffn(tgt2, training=training)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2, training=training)
        tgt = self.norm3(tgt)
        return tgt

    def compute_output_spec(self, tgt_spec, *args, **kwargs):
        return keras.KerasTensor(
            shape=tgt_spec.shape,
            dtype=tgt_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "sa_nhead": self.sa_nhead,
                "ca_nhead": self.ca_nhead,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_rate,
                "num_feature_levels": self.num_feature_levels,
                "dec_n_points": self.dec_n_points,
                "spatial_shapes": self.spatial_shapes,
                "level_start_index": self.level_start_index,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Sine embedding for reference points (used in decoder)
# ---------------------------------------------------------------------------


def _sincos_interleave(x):
    """Apply sin to even indices and cos to odd indices, then interleave."""
    sin_part = ops.sin(x[..., 0::2])
    cos_part = ops.cos(x[..., 1::2])
    stacked = ops.stack([sin_part, cos_part], axis=-1)
    out_shape = ops.shape(x)
    return ops.reshape(stacked, out_shape)


@keras.saving.register_keras_serializable(package="kmodels")
class SinePositionEmbeddingForRefPoints(layers.Layer):
    """Generate sinusoidal embeddings for reference point positions."""

    def __init__(self, dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, pos_tensor):
        scale = 2 * math.pi
        dim_t = ops.cast(ops.arange(self.dim), "float32")
        dim_t = 10000.0 ** (2 * (dim_t // 2) / self.dim)

        x_embed = pos_tensor[..., 0:1] * scale
        y_embed = pos_tensor[..., 1:2] * scale

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_x = _sincos_interleave(pos_x)
        pos_y = _sincos_interleave(pos_y)

        if pos_tensor.shape[-1] == 2:
            return ops.concatenate([pos_y, pos_x], axis=-1)
        elif pos_tensor.shape[-1] == 4:
            w_embed = pos_tensor[..., 2:3] * scale
            h_embed = pos_tensor[..., 3:4] * scale
            pos_w = _sincos_interleave(w_embed / dim_t)
            pos_h = _sincos_interleave(h_embed / dim_t)
            return ops.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)
        else:
            raise ValueError(
                f"pos_tensor last dim must be 2 or 4, got {pos_tensor.shape[-1]}"
            )

    def compute_output_spec(self, input_spec, **kwargs):
        d = input_spec.shape[-1]
        out_dim = self.dim * d
        new_shape = input_spec.shape[:-1] + (out_dim,)
        return keras.KerasTensor(shape=new_shape, dtype=input_spec.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


def gen_sineembed_for_position(pos_tensor, dim=128):
    """Functional wrapper for backward compat (not used in model graph)."""
    layer = SinePositionEmbeddingForRefPoints(dim=dim)
    return layer(pos_tensor)
