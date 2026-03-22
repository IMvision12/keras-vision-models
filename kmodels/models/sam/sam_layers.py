import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class SAMPatchEmbeddings(layers.Layer):
    """Converts pixel values to patch embeddings via Conv2D projection.

    Args:
        hidden_size: Embedding dimension.
        patch_size: Size of each image patch.
        num_channels: Number of input image channels.
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
        return self.projection(pixel_values)

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
class SAMVisionAttention(layers.Layer):
    """Multi-head attention with decomposed relative position embeddings.

    Supports both windowed and global attention modes. When ``window_size > 0``,
    relative position parameters have shape ``(2*window_size - 1, head_dim)``; for
    global attention layers, the input_size is derived from the full feature map.

    Args:
        hidden_size: Total hidden dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projection.
        use_rel_pos: Whether to add decomposed relative position bias.
        input_size: Spatial resolution ``(H, W)`` used for relative position tables.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        qkv_bias=True,
        use_rel_pos=True,
        input_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        self.qkv = layers.Dense(hidden_size * 3, use_bias=qkv_bias, name="qkv")
        self.proj = layers.Dense(hidden_size, name="proj")

    def build(self, input_shape):
        if self.use_rel_pos and self.input_size is not None:
            self.rel_pos_h = self.add_weight(
                name="rel_pos_h",
                shape=(2 * self.input_size[0] - 1, self.head_dim),
                initializer="zeros",
                trainable=True,
            )
            self.rel_pos_w = self.add_weight(
                name="rel_pos_w",
                shape=(2 * self.input_size[1] - 1, self.head_dim),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def _get_rel_pos(self, q_size, k_size, rel_pos):
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        rel_pos_shape = ops.shape(rel_pos)
        if rel_pos_shape[0] != max_rel_dist:
            rel_pos_resized = ops.transpose(
                ops.reshape(rel_pos, (1, rel_pos_shape[0], -1)), (0, 2, 1)
            )
            rel_pos_resized = ops.image.resize(
                ops.expand_dims(rel_pos_resized, axis=-1),
                (ops.shape(rel_pos_resized)[1], max_rel_dist),
                interpolation="bilinear",
            )
            rel_pos_resized = ops.squeeze(rel_pos_resized, axis=-1)
            rel_pos_resized = ops.transpose(
                ops.reshape(rel_pos_resized, (-1, max_rel_dist)), (1, 0)
            )
        else:
            rel_pos_resized = rel_pos

        q_coords = ops.cast(
            ops.expand_dims(ops.arange(q_size), axis=1), dtype="float32"
        ) * max(k_size / q_size, 1.0)
        k_coords = ops.cast(
            ops.expand_dims(ops.arange(k_size), axis=0), dtype="float32"
        ) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
            q_size / k_size, 1.0
        )
        relative_coords = ops.cast(relative_coords, dtype="int32")
        return ops.take(rel_pos_resized, relative_coords, axis=0)

    def _get_decomposed_rel_pos(self, query, q_size, k_size):
        query_height, query_width = q_size
        key_height, key_width = k_size
        rel_pos_h = self._get_rel_pos(query_height, key_height, self.rel_pos_h)
        rel_pos_w = self._get_rel_pos(query_width, key_width, self.rel_pos_w)

        batch_size = ops.shape(query)[0]
        dim = ops.shape(query)[2]
        reshaped_query = ops.reshape(
            query, (batch_size, query_height, query_width, dim)
        )
        rel_h = ops.einsum("bhwc,hkc->bhwk", reshaped_query, rel_pos_h)
        rel_w = ops.einsum("bhwc,wkc->bhwk", reshaped_query, rel_pos_w)
        return ops.expand_dims(rel_h, axis=-1) + ops.expand_dims(rel_w, axis=-2)

    def call(self, hidden_states):
        batch_size = ops.shape(hidden_states)[0]
        height = ops.shape(hidden_states)[1]
        width = ops.shape(hidden_states)[2]

        qkv = self.qkv(hidden_states)
        qkv = ops.reshape(qkv, (batch_size, height * width, 3, self.num_heads, -1))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        qkv = ops.reshape(qkv, (3, batch_size * self.num_heads, height * width, -1))
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_weights = ops.matmul(query * self.scale, ops.transpose(key, (0, 2, 1)))

        if self.use_rel_pos:
            decomposed_rel_pos = self._get_decomposed_rel_pos(
                query, (height, width), (height, width)
            )
            decomposed_rel_pos = ops.reshape(
                decomposed_rel_pos, ops.shape(attn_weights)
            )
            attn_weights = attn_weights + decomposed_rel_pos

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, value)
        attn_output = ops.reshape(
            attn_output, (batch_size, self.num_heads, height, width, -1)
        )
        attn_output = ops.transpose(attn_output, (0, 2, 3, 1, 4))
        attn_output = ops.reshape(attn_output, (batch_size, height, width, -1))
        attn_output = self.proj(attn_output)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "use_rel_pos": self.use_rel_pos,
                "input_size": self.input_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMVisionMLP(layers.Layer):
    """Two-layer MLP with GELU activation used in the vision encoder.

    Args:
        hidden_size: Input/output dimension.
        mlp_dim: Hidden dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.lin1 = layers.Dense(mlp_dim, name="lin1")
        self.lin2 = layers.Dense(hidden_size, name="lin2")

    def call(self, hidden_states):
        hidden_states = self.lin1(hidden_states)
        hidden_states = ops.nn.gelu(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMVisionLayer(layers.Layer):
    """Single transformer block in the SAM vision encoder.

    Implements windowed or global attention with optional relative position bias,
    followed by a two-layer MLP.  When ``window_size > 0``, the input is
    partitioned into non-overlapping windows before attention.

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension.
        qkv_bias: QKV bias flag.
        use_rel_pos: Relative position flag.
        window_size: Window size (0 = global attention).
        image_size: Full image patch grid size.
        layer_norm_eps: LayerNorm epsilon.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_dim,
        qkv_bias=True,
        use_rel_pos=True,
        window_size=0,
        image_size=64,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps

        if window_size == 0:
            input_size = (image_size, image_size)
        else:
            input_size = (window_size, window_size)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm1"
        )
        self.attn = SAMVisionAttention(
            hidden_size,
            num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size,
            name="attn",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm2"
        )
        self.mlp = SAMVisionMLP(hidden_size, mlp_dim, name="mlp")

    def _window_partition(self, hidden_states, window_size):
        batch_size = ops.shape(hidden_states)[0]
        height = ops.shape(hidden_states)[1]
        width = ops.shape(hidden_states)[2]
        channel = ops.shape(hidden_states)[3]

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            hidden_states = ops.pad(
                hidden_states, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
            )
        pad_height = height + pad_h
        pad_width = width + pad_w

        hidden_states = ops.reshape(
            hidden_states,
            (
                batch_size,
                pad_height // window_size,
                window_size,
                pad_width // window_size,
                window_size,
                channel,
            ),
        )
        windows = ops.reshape(
            ops.transpose(hidden_states, (0, 1, 3, 2, 4, 5)),
            (-1, window_size, window_size, channel),
        )
        return windows, (pad_height, pad_width)

    def _window_unpartition(self, windows, window_size, padding_shape, original_shape):
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = ops.shape(windows)[0] // (
            pad_height * pad_width // window_size // window_size
        )
        hidden_states = ops.reshape(
            windows,
            (
                batch_size,
                pad_height // window_size,
                pad_width // window_size,
                window_size,
                window_size,
                -1,
            ),
        )
        hidden_states = ops.reshape(
            ops.transpose(hidden_states, (0, 1, 3, 2, 4, 5)),
            (batch_size, pad_height, pad_width, -1),
        )
        return hidden_states[:, :height, :width, :]

    def call(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        if self.window_size > 0:
            height = ops.shape(hidden_states)[1]
            width = ops.shape(hidden_states)[2]
            hidden_states, padding_shape = self._window_partition(
                hidden_states, self.window_size
            )

        hidden_states = self.attn(hidden_states)

        if self.window_size > 0:
            hidden_states = self._window_unpartition(
                hidden_states, self.window_size, padding_shape, (height, width)
            )

        hidden_states = residual + hidden_states
        ln_out = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(ln_out)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "qkv_bias": self.qkv_bias,
                "use_rel_pos": self.use_rel_pos,
                "window_size": self.window_size,
                "image_size": self.image_size,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMVisionNeck(layers.Layer):
    """Neck that projects vision encoder output to the mask decoder dimension.

    Two Conv2D layers (1x1 then 3x3) with LayerNorm between, converting from
    ``hidden_size`` to ``output_channels``.

    Args:
        hidden_size: Vision encoder hidden dimension.
        output_channels: Output channel dimension (mask decoder hidden size).
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.conv1 = layers.Conv2D(
            output_channels, kernel_size=1, use_bias=False, name="conv1"
        )
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6, name="layer_norm1")
        self.conv2 = layers.Conv2D(
            output_channels,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="conv2",
        )
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6, name="layer_norm2")

    def call(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "output_channels": self.output_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMPositionalEmbedding(layers.Layer):
    """Random Fourier feature positional encoding used in prompt and mask encoders.

    Encodes 2-D coordinates normalized to ``[0, 1]`` into a fixed-dimensional
    feature vector via ``sin``/``cos`` of a learned random projection.

    Args:
        num_pos_feats: Half of the output dimension (full output = ``2 * num_pos_feats``).
        scale: Standard deviation of the initial random projection matrix.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, num_pos_feats=128, scale=None, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.scale = scale if scale is not None else 1.0

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(2, self.num_pos_feats),
            initializer=keras.initializers.RandomNormal(stddev=self.scale),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, coordinates):
        """Encode pre-normalized coordinates in [-1, 1].

        Args:
            coordinates: Tensor with last dim = 2, already normalized to [0, 1].
        """
        coordinates = 2.0 * coordinates - 1.0
        coordinates = ops.cast(coordinates, dtype=self.positional_embedding.dtype)
        coordinates = ops.matmul(coordinates, self.positional_embedding)
        coordinates = 2.0 * np.pi * coordinates
        return ops.concatenate([ops.sin(coordinates), ops.cos(coordinates)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"num_pos_feats": self.num_pos_feats, "scale": self.scale})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMMaskEmbedding(layers.Layer):
    """Embeds dense mask prompts through a small CNN.

    Three Conv2D layers downsample the mask by 4x total, mapping a single-channel
    mask to ``hidden_size`` channels at the image-embedding resolution.

    Args:
        hidden_size: Output embedding dimension.
        mask_input_channels: Intermediate channel count after the second conv.
        layer_norm_eps: Epsilon for layer normalization.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self, hidden_size=256, mask_input_channels=16, layer_norm_eps=1e-6, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mask_input_channels = mask_input_channels
        self.layer_norm_eps = layer_norm_eps
        inner_channels = mask_input_channels // 4

        self.conv1 = layers.Conv2D(
            inner_channels, kernel_size=2, strides=2, name="conv1"
        )
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm1"
        )
        self.conv2 = layers.Conv2D(
            mask_input_channels, kernel_size=2, strides=2, name="conv2"
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm2"
        )
        self.conv3 = layers.Conv2D(hidden_size, kernel_size=1, name="conv3")

    def call(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = ops.nn.gelu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = ops.nn.gelu(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "mask_input_channels": self.mask_input_channels,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMTwoWayAttention(layers.Layer):
    """Attention layer used in the mask decoder's two-way transformer.

    Supports an optional ``downsample_rate`` that reduces the internal dimension
    of Q/K/V projections for efficiency.

    Args:
        hidden_size: Input hidden dimension.
        num_heads: Number of attention heads.
        downsample_rate: Factor to reduce internal dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, num_heads, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = hidden_size // downsample_rate
        self.head_dim = self.internal_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = layers.Dense(self.internal_dim, name="q_proj")
        self.k_proj = layers.Dense(self.internal_dim, name="k_proj")
        self.v_proj = layers.Dense(self.internal_dim, name="v_proj")
        self.out_proj = layers.Dense(hidden_size, name="out_proj")

    def call(self, query, key, value, attention_similarity=None):
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self._separate_heads(query)
        key = self._separate_heads(key)
        value = self._separate_heads(value)

        attn_weights = ops.matmul(query, ops.transpose(key, (0, 1, 2, 4, 3)))
        attn_weights = attn_weights * self.scale

        if attention_similarity is not None:
            attn_weights = attn_weights + attention_similarity

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, value)
        attn_output = self._recombine_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output

    def _separate_heads(self, x):
        batch = ops.shape(x)[0]
        point_batch = ops.shape(x)[1]
        n_tokens = ops.shape(x)[2]
        x = ops.reshape(
            x, (batch, point_batch, n_tokens, self.num_heads, self.head_dim)
        )
        return ops.transpose(x, (0, 1, 3, 2, 4))

    def _recombine_heads(self, x):
        batch = ops.shape(x)[0]
        point_batch = ops.shape(x)[1]
        n_tokens = ops.shape(x)[3]
        x = ops.transpose(x, (0, 1, 3, 2, 4))
        return ops.reshape(x, (batch, point_batch, n_tokens, self.internal_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMTwoWayAttentionBlock(layers.Layer):
    """A single block of the mask decoder's two-way transformer.

    Consists of four sub-layers:
    1. Self-attention on sparse (query) tokens
    2. Cross-attention from queries to image embeddings
    3. MLP on queries
    4. Cross-attention from image embeddings to queries

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension.
        attention_downsample_rate: Downsample factor for cross-attention.
        skip_first_layer_pe: Skip PE addition in the first layer's self-attention.
        layer_norm_eps: LayerNorm epsilon.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_dim=2048,
        attention_downsample_rate=2,
        skip_first_layer_pe=False,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.skip_first_layer_pe = skip_first_layer_pe
        self.layer_norm_eps = layer_norm_eps

        self.self_attn = SAMTwoWayAttention(
            hidden_size, num_heads, downsample_rate=1, name="self_attn"
        )
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm1"
        )
        self.cross_attn_token_to_image = SAMTwoWayAttention(
            hidden_size,
            num_heads,
            downsample_rate=attention_downsample_rate,
            name="cross_attn_token_to_image",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm2"
        )
        self.mlp_lin1 = layers.Dense(mlp_dim, name="mlp_lin1")
        self.mlp_lin2 = layers.Dense(hidden_size, name="mlp_lin2")
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm3"
        )
        self.layer_norm4 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm4"
        )
        self.cross_attn_image_to_token = SAMTwoWayAttention(
            hidden_size,
            num_heads,
            downsample_rate=attention_downsample_rate,
            name="cross_attn_image_to_token",
        )

    def call(
        self,
        queries,
        keys,
        query_point_embedding,
        key_point_embedding,
        attention_similarity=None,
    ):
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out
        queries = self.layer_norm2(queries)

        mlp_out = self.mlp_lin1(queries)
        mlp_out = ops.nn.relu(mlp_out)
        mlp_out = self.mlp_lin2(mlp_out)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out
        keys = self.layer_norm4(keys)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "attention_downsample_rate": self.attention_downsample_rate,
                "skip_first_layer_pe": self.skip_first_layer_pe,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMFeedForward(layers.Layer):
    """Multi-layer perceptron used in the mask decoder for iou/mask heads.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        num_layers: Total number of linear layers.
        sigmoid_output: Apply sigmoid to the final output.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        sigmoid_output=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output

        self.proj_in = layers.Dense(hidden_dim, name="proj_in")
        self.proj_out = layers.Dense(output_dim, name="proj_out")
        self.hidden_layers = []
        for i in range(num_layers - 2):
            self.hidden_layers.append(layers.Dense(hidden_dim, name=f"layers_{i}"))

    def call(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = ops.nn.relu(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = ops.nn.relu(layer(hidden_states))
        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = ops.sigmoid(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "sigmoid_output": self.sigmoid_output,
            }
        )
        return config
