"""Sam3TrackerVideo layers: memory attention, memory encoder, RoPE.

These extend the base Sam3Tracker with temporal memory propagation
for multi-frame video tracking.
"""

import math

import keras
from keras import layers, ops


def compute_vision_rope(dim, end_x, end_y, theta=10000.0):
    """Pre-compute 2D axial rotary embeddings.

    Args:
        dim: head dimension (must be divisible by 4).
        end_x, end_y: spatial dimensions of the feature map.
        theta: base frequency.

    Returns:
        (cos, sin) each of shape (1, 1, end_x * end_y, dim).
    """
    freqs = 1.0 / (theta ** (ops.cast(ops.arange(0, dim, 4), "float32") / dim))

    flat_idx = ops.arange(end_x * end_y)
    x_pos = ops.cast(flat_idx % end_x, "float32")
    y_pos = ops.cast(flat_idx // end_x, "float32")

    freqs_x = ops.outer(x_pos, freqs)
    freqs_y = ops.outer(y_pos, freqs)
    inv_freq = ops.concatenate([freqs_x, freqs_y], axis=-1)

    inv_freq = ops.repeat(ops.expand_dims(inv_freq, axis=-1), 2, axis=-1)
    inv_freq = ops.reshape(inv_freq, (end_x * end_y, dim))

    cos_emb = ops.cos(inv_freq)
    sin_emb = ops.sin(inv_freq)
    cos_emb = ops.expand_dims(ops.expand_dims(cos_emb, 0), 0)
    sin_emb = ops.expand_dims(ops.expand_dims(sin_emb, 0), 0)
    return cos_emb, sin_emb


def _rotate_pairwise(x):
    """Pairwise rotation: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]."""
    shape = ops.shape(x)
    x = ops.reshape(x, shape[:-1] + (shape[-1] // 2, 2))
    x1 = x[..., 0]
    x2 = x[..., 1]
    rotated = ops.stack([-x2, x1], axis=-1)
    return ops.reshape(rotated, shape)


def apply_rotary_pos_emb_2d(q, k, cos, sin, num_k_exclude_rope=0, repeat_freqs_k=False):
    """Apply 2D rotary position embedding to query and key tensors."""
    seq_k = ops.shape(k)[2]
    if num_k_exclude_rope > 0:
        k_rot = k[:, :, : seq_k - num_k_exclude_rope, :]
        k_pass = k[:, :, seq_k - num_k_exclude_rope :, :]
    else:
        k_rot = k
        k_pass = None

    q_f = ops.cast(q, "float32")
    q_embed = q_f * cos + _rotate_pairwise(q_f) * sin

    if ops.shape(k_rot)[2] == 0:
        k_embed = k_rot
        if k_pass is not None:
            k_embed = ops.concatenate([k_embed, k_pass], axis=2)
        return ops.cast(q_embed, q.dtype), k_embed

    if repeat_freqs_k and ops.shape(k_rot)[2] != ops.shape(q)[2]:
        repeat_factor = ops.shape(k_rot)[2] // ops.shape(q)[2]
        cos_k = ops.repeat(cos, repeat_factor, axis=2)
        sin_k = ops.repeat(sin, repeat_factor, axis=2)
    else:
        cos_k = cos
        sin_k = sin

    k_f = ops.cast(k_rot, "float32")
    k_embed = k_f * cos_k + _rotate_pairwise(k_f) * sin_k
    k_embed = ops.cast(k_embed, k.dtype)

    if k_pass is not None:
        k_embed = ops.concatenate([k_embed, k_pass], axis=2)

    return ops.cast(q_embed, q.dtype), k_embed


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerVideoRoPEAttention(layers.Layer):
    """Multi-head attention with rotary position encoding."""

    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=1,
        downsample_rate=1,
        kv_in_dim=None,
        rope_k_repeat=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = hidden_size // downsample_rate
        self.head_dim = self.internal_dim // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else hidden_size
        self.rope_k_repeat = rope_k_repeat

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.internal_dim, name="q_proj")
        self.q_proj.build((None, None, None, self.hidden_size))
        self.k_proj = layers.Dense(self.internal_dim, name="k_proj")
        self.k_proj.build((None, None, None, self.kv_in_dim))
        self.v_proj = layers.Dense(self.internal_dim, name="v_proj")
        self.v_proj.build((None, None, None, self.kv_in_dim))
        self.o_proj = layers.Dense(self.hidden_size, name="o_proj")
        self.o_proj.build((None, None, None, self.internal_dim))
        self.built = True

    def call(self, query, key, value, position_embeddings, num_k_exclude_rope=0):
        shape_q = ops.shape(query)
        batch_size, point_batch_size = shape_q[0], shape_q[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        def reshape_heads(x):
            s = ops.shape(x)
            x = ops.reshape(
                x, (s[0] * s[1], s[2], self.num_attention_heads, self.head_dim)
            )
            return ops.transpose(x, (0, 2, 1, 3))

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_2d(
            q,
            k,
            cos,
            sin,
            repeat_freqs_k=self.rope_k_repeat,
            num_k_exclude_rope=num_k_exclude_rope,
        )

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        attn = ops.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(
            out,
            (
                batch_size,
                point_batch_size,
                -1,
                self.num_attention_heads * self.head_dim,
            ),
        )
        return self.o_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "downsample_rate": self.downsample_rate,
                "kv_in_dim": self.kv_in_dim,
                "rope_k_repeat": self.rope_k_repeat,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerVideoMemoryAttentionLayer(layers.Layer):
    """Single memory attention block: self-attn -> cross-attn -> FFN."""

    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=1,
        downsample_rate=1,
        feed_forward_hidden_size=2048,
        feed_forward_act="relu",
        mem_dim=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.downsample_rate = downsample_rate
        self.feed_forward_hidden_size = feed_forward_hidden_size
        self.feed_forward_act = feed_forward_act
        self.mem_dim = mem_dim

    def build(self, input_shape):
        hs = self.hidden_size

        self.self_attn = Sam3TrackerVideoRoPEAttention(
            hidden_size=hs,
            num_attention_heads=self.num_attention_heads,
            downsample_rate=self.downsample_rate,
            name="self_attn",
        )
        self.self_attn.build(None)

        self.cross_attn_image = Sam3TrackerVideoRoPEAttention(
            hidden_size=hs,
            num_attention_heads=self.num_attention_heads,
            downsample_rate=self.downsample_rate,
            kv_in_dim=self.mem_dim,
            rope_k_repeat=True,
            name="cross_attn_image",
        )
        self.cross_attn_image.build(None)

        self.linear1 = layers.Dense(self.feed_forward_hidden_size, name="linear1")
        self.linear1.build((None, None, None, hs))
        self.linear2 = layers.Dense(hs, name="linear2")
        self.linear2.build((None, None, None, self.feed_forward_hidden_size))

        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm1")
        self.layer_norm1.build((None, None, None, hs))
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm2")
        self.layer_norm2.build((None, None, None, hs))
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm3")
        self.layer_norm3.build((None, None, None, hs))
        self.built = True

    def call(
        self,
        queries,
        keys,
        key_point_embedding,
        rope_position_embeddings,
        num_k_exclude_rope=0,
    ):
        q = self.layer_norm1(queries)
        q = self.self_attn(
            query=q,
            key=q,
            value=q,
            position_embeddings=rope_position_embeddings,
        )
        queries = queries + q

        q = self.layer_norm2(queries)
        q = self.cross_attn_image(
            query=q,
            key=keys + key_point_embedding,
            value=keys,
            position_embeddings=rope_position_embeddings,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        queries = queries + q

        q = self.layer_norm3(queries)
        act_fn = ops.nn.relu if self.feed_forward_act == "relu" else ops.nn.gelu
        q = self.linear2(act_fn(self.linear1(q)))
        queries = queries + q

        return queries

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "downsample_rate": self.downsample_rate,
                "feed_forward_hidden_size": self.feed_forward_hidden_size,
                "feed_forward_act": self.feed_forward_act,
                "mem_dim": self.mem_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerVideoMemoryAttention(layers.Layer):
    """Memory attention: 4 layers of self-attn + cross-attn with RoPE."""

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=1,
        downsample_rate=1,
        feed_forward_hidden_size=2048,
        feed_forward_act="relu",
        mem_dim=64,
        rope_theta=10000,
        rope_feat_sizes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.downsample_rate = downsample_rate
        self.feed_forward_hidden_size = feed_forward_hidden_size
        self.feed_forward_act = feed_forward_act
        self.mem_dim = mem_dim
        self.rope_theta = rope_theta
        self.rope_feat_sizes = rope_feat_sizes or [72, 72]

    def build(self, input_shape):
        self.attention_layers = []
        for i in range(self.num_layers):
            layer = Sam3TrackerVideoMemoryAttentionLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                downsample_rate=self.downsample_rate,
                feed_forward_hidden_size=self.feed_forward_hidden_size,
                feed_forward_act=self.feed_forward_act,
                mem_dim=self.mem_dim,
                name=f"layers_{i}",
            )
            layer.build(None)
            self.attention_layers.append(layer)

        self.layer_norm = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
        self.layer_norm.build((None, None, None, self.hidden_size))

        head_dim = self.hidden_size // (self.downsample_rate * self.num_attention_heads)
        self._rope_cos, self._rope_sin = compute_vision_rope(
            dim=head_dim,
            end_x=self.rope_feat_sizes[0],
            end_y=self.rope_feat_sizes[1],
            theta=self.rope_theta,
        )
        self.built = True

    def call(
        self,
        current_vision_features,
        memory,
        current_vision_position_embeddings=None,
        memory_position_embeddings=None,
        num_object_pointer_tokens=0,
    ):
        output = current_vision_features
        if current_vision_position_embeddings is not None:
            output = output + 0.1 * current_vision_position_embeddings

        output = ops.transpose(output, (1, 0, 2))
        memory = ops.transpose(memory, (1, 0, 2))
        memory_position_embeddings = ops.transpose(
            memory_position_embeddings, (1, 0, 2)
        )

        memory = ops.expand_dims(memory, 1)
        memory_position_embeddings = ops.expand_dims(memory_position_embeddings, 1)

        rope_pos = (self._rope_cos, self._rope_sin)

        for layer in self.attention_layers:
            if output.ndim == 3:
                output = ops.expand_dims(output, 1)
            output = layer(
                queries=output,
                keys=memory,
                key_point_embedding=memory_position_embeddings,
                rope_position_embeddings=rope_pos,
                num_k_exclude_rope=num_object_pointer_tokens,
            )

        normed_output = self.layer_norm(output)

        if normed_output.ndim == 4:
            normed_output = normed_output[:, 0, :, :]
        normed_output = ops.transpose(normed_output, (1, 0, 2))

        return normed_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
                "downsample_rate": self.downsample_rate,
                "feed_forward_hidden_size": self.feed_forward_hidden_size,
                "feed_forward_act": self.feed_forward_act,
                "mem_dim": self.mem_dim,
                "rope_theta": self.rope_theta,
                "rope_feat_sizes": self.rope_feat_sizes,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerVideoMemoryFuserCXBlock(layers.Layer):
    """ConvNeXt-style fusion block for memory encoder.

    Architecture:
        depthwise_conv(7x7) -> LayerNorm -> pointwise_conv1 -> GELU -> pointwise_conv2
        -> scale -> residual add
    """

    def __init__(
        self,
        embed_dim=256,
        intermediate_dim=1024,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.layer_scale_init_value = layer_scale_init_value

    def build(self, input_shape):
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            padding="same",
            data_format="channels_first",
            name="depthwise_conv",
        )
        self.depthwise_conv.build((None, self.embed_dim, None, None))

        from kmodels.models.sam3_tracker.sam3_tracker_layers import (
            ChannelsFirstLayerNorm,
        )

        self.layer_norm = ChannelsFirstLayerNorm(self.embed_dim, name="layer_norm")
        self.layer_norm.build((None, self.embed_dim, None, None))

        self.pointwise_conv1 = layers.Dense(
            self.intermediate_dim, name="pointwise_conv1"
        )
        self.pointwise_conv1.build((None, None, None, self.embed_dim))

        self.pointwise_conv2 = layers.Dense(self.embed_dim, name="pointwise_conv2")
        self.pointwise_conv2.build((None, None, None, self.intermediate_dim))

        self.scale = self.add_weight(
            name="scale",
            shape=(self.embed_dim,),
            initializer=keras.initializers.Constant(self.layer_scale_init_value),
        )
        self.built = True

    def call(self, hidden_states):
        residual = hidden_states
        x = self.depthwise_conv(hidden_states)
        x = self.layer_norm(x)
        x = ops.transpose(x, (0, 2, 3, 1))
        x = ops.nn.gelu(self.pointwise_conv1(x))
        x = self.pointwise_conv2(x)
        x = self.scale * x
        x = ops.transpose(x, (0, 3, 1, 2))
        return residual + x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "intermediate_dim": self.intermediate_dim,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "layer_scale_init_value": self.layer_scale_init_value,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerVideoMaskDownSamplerLayer(layers.Layer):
    """Single mask downsampling layer: Conv2D + LayerNorm + activation."""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def build(self, input_shape):
        from kmodels.models.sam3_tracker.sam3_tracker_layers import (
            ChannelsFirstLayerNorm,
        )

        self.conv = layers.Conv2D(
            self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="valid",
            data_format="channels_first",
            name="conv",
        )
        self.conv.build((None, self.in_channels, None, None))
        self.layer_norm = ChannelsFirstLayerNorm(self.out_channels, name="layer_norm")
        self.layer_norm.build((None, self.out_channels, None, None))
        self.built = True

    def call(self, x):
        p = self.padding
        x = ops.pad(x, [[0, 0], [0, 0], [p, p], [p, p]])
        return ops.nn.gelu(self.layer_norm(self.conv(x)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerVideoMaskDownSampler(layers.Layer):
    """Progressively downsamples a mask by total_stride."""

    def __init__(
        self,
        embed_dim=256,
        kernel_size=3,
        stride=2,
        padding=1,
        total_stride=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.total_stride = total_stride

    def build(self, input_shape):
        num_layers = int(math.log2(self.total_stride) // math.log2(self.stride))
        self.downsample_layers = []
        mask_in_chans = 1
        for i in range(num_layers):
            mask_out_chans = mask_in_chans * (self.stride**2)
            layer = Sam3TrackerVideoMaskDownSamplerLayer(
                in_channels=mask_in_chans,
                out_channels=mask_out_chans,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                name=f"layers_{i}",
            )
            layer.build((None, mask_in_chans, None, None))
            self.downsample_layers.append(layer)
            mask_in_chans = mask_out_chans

        self.final_conv = layers.Conv2D(
            self.embed_dim,
            kernel_size=1,
            data_format="channels_first",
            name="final_conv",
        )
        self.final_conv.build((None, mask_out_chans, None, None))
        self.built = True

    def call(self, x):
        for layer in self.downsample_layers:
            x = layer(x)
        return self.final_conv(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "total_stride": self.total_stride,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerVideoMemoryEncoder(layers.Layer):
    """Encodes predicted masks into memory for future frame conditioning."""

    def __init__(
        self,
        hidden_size=256,
        output_channels=64,
        mask_downsampler_config=None,
        memory_fuser_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self._mask_ds_config = mask_downsampler_config or {}
        self._memory_fuser_config = memory_fuser_config or {}

    def build(self, input_shape):
        from kmodels.models.sam3_video.sam3_video_layers import (
            Sam3SinePositionEmbedding,
        )

        self.mask_downsampler = Sam3TrackerVideoMaskDownSampler(
            embed_dim=self._mask_ds_config.get("embed_dim", 256),
            kernel_size=self._mask_ds_config.get("kernel_size", 3),
            stride=self._mask_ds_config.get("stride", 2),
            padding=self._mask_ds_config.get("padding", 1),
            total_stride=self._mask_ds_config.get("total_stride", 16),
            name="mask_downsampler",
        )
        self.mask_downsampler.build((None, 1, None, None))

        self.feature_projection = layers.Conv2D(
            self.hidden_size,
            kernel_size=1,
            data_format="channels_first",
            name="feature_projection",
        )
        self.feature_projection.build((None, self.hidden_size, None, None))

        fuser_cfg = self._memory_fuser_config
        num_fuser_layers = fuser_cfg.get("num_layers", 2)
        self.fuser_layers = []
        for i in range(num_fuser_layers):
            block = Sam3TrackerVideoMemoryFuserCXBlock(
                embed_dim=fuser_cfg.get("embed_dim", 256),
                intermediate_dim=fuser_cfg.get("intermediate_dim", 1024),
                kernel_size=fuser_cfg.get("kernel_size", 7),
                padding=fuser_cfg.get("padding", 3),
                layer_scale_init_value=fuser_cfg.get("layer_scale_init_value", 1e-6),
                name=f"memory_fuser_layers_{i}",
            )
            block.build((None, self.hidden_size, None, None))
            self.fuser_layers.append(block)

        self.position_encoding = Sam3SinePositionEmbedding(
            num_pos_feats=self.output_channels // 2,
            normalize=True,
            name="position_encoding",
        )

        self.projection = layers.Conv2D(
            self.output_channels,
            kernel_size=1,
            data_format="channels_first",
            name="projection",
        )
        self.projection.build((None, self.hidden_size, None, None))
        self.built = True

    def call(self, vision_features, masks):
        masks = self.mask_downsampler(masks)
        vision_features = self.feature_projection(vision_features)
        vision_features = vision_features + masks
        for layer in self.fuser_layers:
            vision_features = layer(vision_features)
        vision_features = self.projection(vision_features)
        maskmem_pos_enc = self.position_encoding(vision_features)
        return vision_features, maskmem_pos_enc

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "output_channels": self.output_channels,
                "mask_downsampler_config": self._mask_ds_config,
                "memory_fuser_config": self._memory_fuser_config,
            }
        )
        return config


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """1D sine positional encoding for temporal offsets."""
    pe_dim = dim // 2
    dim_t = ops.arange(pe_dim, dtype="float32")
    dim_t = ops.cast(temperature, "float32") ** (
        2 * (dim_t // 2) / ops.cast(pe_dim, "float32")
    )
    pos_embed = ops.expand_dims(pos_inds, axis=-1) / dim_t
    pos_embed = ops.concatenate([ops.sin(pos_embed), ops.cos(pos_embed)], axis=-1)
    return pos_embed
