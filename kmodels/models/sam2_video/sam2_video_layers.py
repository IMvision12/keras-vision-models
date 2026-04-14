import keras
import numpy as np
from keras import layers, ops


def build_rope_2d_axial_cache(end_x, end_y, dim, theta=10000.0):
    """Precompute 2D axial rotary position embedding cosine/sine tables.

    Splits the embedding dimension in half and applies independent 1D
    rotary frequencies along the ``x`` and ``y`` axes. Each pair of
    consecutive dimensions shares the same frequency, and the resulting
    angles are repeat-interleaved by 2 so they align with the pairwise
    rotation used in :func:`_apply_rope_2d`.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        end_x: Integer, width of the spatial grid.
        end_y: Integer, height of the spatial grid.
        dim: Integer, total embedding dimension. Must be divisible by 4.
        theta: Float, base frequency for the rotary encoding.
            Defaults to ``10000.0``.

    Returns:
        Tuple ``(cos, sin)`` of NumPy arrays each shaped
        ``(end_x * end_y, dim)`` and cast to ``float32``.
    """
    i = np.arange(0, dim, 4, dtype=np.float32)[: dim // 4]
    freqs = 1.0 / (theta ** (i / dim))

    flat = np.arange(end_x * end_y, dtype=np.float32)
    x_pos = flat % end_x
    y_pos = flat // end_x

    freqs_x = np.outer(x_pos, freqs)
    freqs_y = np.outer(y_pos, freqs)

    inv_freq = np.concatenate([freqs_x, freqs_y], axis=-1)
    inv_freq = np.repeat(inv_freq, 2, axis=-1)

    return np.cos(inv_freq).astype(np.float32), np.sin(inv_freq).astype(np.float32)


def _rotate_pairwise(x):
    """Pairwise 90-degree rotation used by rotary position embeddings.

    Reshapes the last dimension into pairs and maps ``(a, b)`` to
    ``(-b, a)``, which corresponds to multiplying the complex pair
    ``a + b * i`` by ``i``.

    Args:
        x: Tensor whose last dimension is even.

    Returns:
        Tensor of the same shape as ``x`` with pairs rotated.
    """
    shape = ops.shape(x)
    x = ops.reshape(x, [*shape[:-1], -1, 2])
    x1 = x[..., 0]
    x2 = x[..., 1]
    x = ops.stack([-x2, x1], axis=-1)
    return ops.reshape(x, shape)


def _apply_rope_2d(q, k, cos, sin, num_k_exclude_rope=0, rope_k_repeat=False):
    """Apply 2D axial rotary position embedding to query and key tensors.

    Rotates query and key along the precomputed ``cos``/``sin`` tables.
    The trailing ``num_k_exclude_rope`` key tokens bypass the rotation so
    object pointer tokens concatenated after spatial memory tokens are
    left untouched. When ``rope_k_repeat`` is True and the key sequence is
    longer than the query sequence (multi-frame memory attention), the
    ``cos``/``sin`` tables are tiled so each memory frame gets its own
    0..L-1 RoPE cycle.

    Args:
        q: Query tensor of shape ``(B, H, N_q, D)``.
        k: Key tensor of shape ``(B, H, N_k, D)``.
        cos: Cosine table broadcastable to ``(1, 1, N_q, D)``.
        sin: Sine table broadcastable to ``(1, 1, N_q, D)``.
        num_k_exclude_rope: Integer, number of trailing key tokens to
            exclude from the rotation. Defaults to ``0``.
        rope_k_repeat: Boolean, whether to tile the ``cos``/``sin`` tables
            when the key sequence is longer than the query sequence.
            Defaults to ``False``.

    Returns:
        Tuple ``(q_embed, k_embed)`` of rotary-embedded tensors.
    """
    if num_k_exclude_rope > 0:
        k_rot = k[..., : ops.shape(k)[-2] - num_k_exclude_rope, :]
        k_pass = k[..., ops.shape(k)[-2] - num_k_exclude_rope :, :]
    else:
        k_rot = k
        k_pass = None

    q_embed = q * cos + _rotate_pairwise(q) * sin

    if rope_k_repeat and k_rot.shape[-2] != q.shape[-2]:
        repeat_factor = k_rot.shape[-2] // q.shape[-2]
        cos_k = ops.tile(cos, [1, 1, repeat_factor, 1])
        sin_k = ops.tile(sin, [1, 1, repeat_factor, 1])
    else:
        cos_k = cos
        sin_k = sin

    k_embed = k_rot * cos_k + _rotate_pairwise(k_rot) * sin_k

    if k_pass is not None:
        k_embed = ops.concatenate([k_embed, k_pass], axis=-2)

    return q_embed, k_embed


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2VideoRoPEAttention(layers.Layer):
    """Multi-head attention with 2D axial rotary position embedding.

    Used as the self-attention and cross-attention block inside the
    memory attention stack. Separate Q/K/V projections are supported so
    that cross-attention can consume lower-dimensional memory features
    (``kv_in_dim != hidden_size``). When running under the Keras torch
    backend, the attention matmul is delegated to
    :func:`torch.nn.functional.scaled_dot_product_attention` for memory
    efficiency on long memory sequences.

    Args:
        hidden_size (int): Query/output dimension. Defaults to ``256``.
        kv_in_dim (int): Key/value input dimension. When ``None`` the
            same value as ``hidden_size`` is used. Defaults to ``None``.
        num_heads (int): Number of attention heads. Defaults to ``1``.
        rope_k_repeat (bool): Whether to tile the RoPE tables across
            multiple memory frames in cross-attention. Defaults to
            ``False``.
        dropout_p (float): Attention dropout probability used by the
            non-torch fallback path. Defaults to ``0.1``.
        **kwargs: Additional keyword arguments passed to the base
            ``Layer`` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        hidden_size=256,
        kv_in_dim=None,
        num_heads=1,
        rope_k_repeat=False,
        dropout_p=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kv_in_dim = kv_in_dim or hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.rope_k_repeat = rope_k_repeat
        self.dropout_p = dropout_p

        self.q_proj = layers.Dense(hidden_size, name="q_proj")
        self.k_proj = layers.Dense(hidden_size, name="k_proj")
        self.v_proj = layers.Dense(hidden_size, name="v_proj")
        self.o_proj = layers.Dense(hidden_size, name="o_proj")
        self.attn_drop = layers.Dropout(dropout_p)

    def call(
        self, query, key, value, rope_cos_sin, num_k_exclude_rope=0, training=None
    ):
        batch = ops.shape(query)[0]
        point_batch = ops.shape(query)[1]
        B = batch * point_batch

        q = ops.reshape(self.q_proj(query), [B, -1, self.num_heads, self.head_dim])
        k = ops.reshape(self.k_proj(key), [B, -1, self.num_heads, self.head_dim])
        v = ops.reshape(self.v_proj(value), [B, -1, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        cos, sin = rope_cos_sin
        cos = ops.expand_dims(ops.expand_dims(cos, 0), 0)
        sin = ops.expand_dims(ops.expand_dims(sin, 0), 0)

        q, k = _apply_rope_2d(
            q,
            k,
            cos,
            sin,
            num_k_exclude_rope=num_k_exclude_rope,
            rope_k_repeat=self.rope_k_repeat,
        )

        if keras.backend.backend() == "torch":
            import torch.nn.functional as F

            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        else:
            q = q * self.scale
            attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
            attn = ops.softmax(attn)
            attn = self.attn_drop(attn, training=training)
            x = ops.matmul(attn, v)

        x = ops.transpose(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch, point_batch, -1, self.hidden_size])

        return self.o_proj(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "kv_in_dim": self.kv_in_dim,
                "num_heads": self.num_heads,
                "rope_k_repeat": self.rope_k_repeat,
                "dropout_p": self.dropout_p,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2VideoMemoryAttentionLayer(layers.Layer):
    """Single memory-attention block with self, cross and feed-forward sub-layers.

    Pre-norm architecture mirroring the original SAM 2 memory attention
    implementation. The current-frame features first self-attend with
    RoPE, then cross-attend to the concatenated spatial memory and
    object-pointer tokens (also with RoPE on the spatial part only), and
    finally pass through a ReLU feed-forward MLP. Residual connections
    are applied around every sub-layer.

    Args:
        hidden_size (int): Query/output dimension. Defaults to ``256``.
        kv_in_dim (int): Memory key/value input dimension. Defaults to
            ``64``.
        num_heads (int): Number of attention heads. Defaults to ``1``.
        ffn_hidden_size (int): Hidden dimension of the feed-forward
            sub-layer. Defaults to ``2048``.
        dropout (float): Dropout probability applied after each sub-layer
            and inside the feed-forward block. Defaults to ``0.1``.
        rope_dropout (float): Dropout probability passed to the internal
            :class:`Sam2VideoRoPEAttention` blocks. Defaults to ``0.1``.
        **kwargs: Additional keyword arguments passed to the base
            ``Layer`` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        hidden_size=256,
        kv_in_dim=64,
        num_heads=1,
        ffn_hidden_size=2048,
        dropout=0.1,
        rope_dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kv_in_dim = kv_in_dim
        self.num_heads = num_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.dropout_rate = dropout
        self.rope_dropout = rope_dropout

        self.self_attn = Sam2VideoRoPEAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_p=rope_dropout,
            name="self_attn",
        )
        self.cross_attn_image = Sam2VideoRoPEAttention(
            hidden_size=hidden_size,
            kv_in_dim=kv_in_dim,
            num_heads=num_heads,
            rope_k_repeat=True,
            dropout_p=rope_dropout,
            name="cross_attn_image",
        )

        self.linear1 = layers.Dense(ffn_hidden_size, name="linear1")
        self.linear2 = layers.Dense(hidden_size, name="linear2")

        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm1")
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm2")
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm3")

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        self.ffn_dropout = layers.Dropout(dropout)

    def call(
        self,
        queries,
        keys,
        key_pos_embed,
        rope_cos_sin,
        num_k_exclude_rope=0,
        training=None,
    ):
        query = self.layer_norm1(queries)
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            rope_cos_sin=rope_cos_sin,
            training=training,
        )
        queries = queries + self.dropout1(query, training=training)

        query = self.layer_norm2(queries)
        query = self.cross_attn_image(
            query=query,
            key=keys + key_pos_embed,
            value=keys,
            rope_cos_sin=rope_cos_sin,
            num_k_exclude_rope=num_k_exclude_rope,
            training=training,
        )
        queries = queries + self.dropout2(query, training=training)

        query = self.layer_norm3(queries)
        query = self.linear2(
            self.ffn_dropout(ops.relu(self.linear1(query)), training=training)
        )
        queries = queries + self.dropout3(query, training=training)

        return queries

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "kv_in_dim": self.kv_in_dim,
                "num_heads": self.num_heads,
                "ffn_hidden_size": self.ffn_hidden_size,
                "dropout": self.dropout_rate,
                "rope_dropout": self.rope_dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2VideoMemoryAttention(layers.Layer):
    """Stack of memory attention blocks used for video feature conditioning.

    Wraps ``num_layers`` :class:`Sam2VideoMemoryAttentionLayer` blocks and
    applies a final layer normalization. The 2D axial RoPE tables are
    precomputed once in :meth:`__init__` from ``rope_feat_sizes`` and
    reused for every forward call. A 0.1-scaled positional encoding is
    added to the current-frame features before the first block, matching
    the original SAM 2 implementation.

    Args:
        hidden_size (int): Query/output dimension. Defaults to ``256``.
        kv_in_dim (int): Memory key/value dimension. Defaults to ``64``.
        num_layers (int): Number of stacked memory attention blocks.
            Defaults to ``4``.
        num_heads (int): Number of attention heads per block. Defaults to
            ``1``.
        ffn_hidden_size (int): Hidden dimension of each block's
            feed-forward sub-layer. Defaults to ``2048``.
        dropout (float): Dropout probability used by all sub-layers.
            Defaults to ``0.1``.
        rope_theta (float): Base frequency for the 2D axial rotary
            embedding tables. Defaults to ``10000.0``.
        rope_feat_sizes (list[int]): Two-element list ``[H, W]`` giving
            the spatial grid used to build the RoPE cache. Defaults to
            ``[64, 64]``.
        **kwargs: Additional keyword arguments passed to the base
            ``Layer`` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        hidden_size=256,
        kv_in_dim=64,
        num_layers=4,
        num_heads=1,
        ffn_hidden_size=2048,
        dropout=0.1,
        rope_theta=10000.0,
        rope_feat_sizes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kv_in_dim = kv_in_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.dropout_rate = dropout
        self.rope_theta = rope_theta
        self.rope_feat_sizes = rope_feat_sizes or [64, 64]

        self.attn_layers = []
        for i in range(num_layers):
            self.attn_layers.append(
                Sam2VideoMemoryAttentionLayer(
                    hidden_size=hidden_size,
                    kv_in_dim=kv_in_dim,
                    num_heads=num_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    dropout=dropout,
                    rope_dropout=dropout,
                    name=f"layers_{i}",
                )
            )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        end_x, end_y = self.rope_feat_sizes
        cos_np, sin_np = build_rope_2d_axial_cache(
            end_x, end_y, hidden_size, theta=rope_theta
        )
        self._rope_cos = cos_np
        self._rope_sin = sin_np

    def call(
        self,
        current_vision_feats,
        memory,
        current_vision_pos_embeds=None,
        memory_pos_embeds=None,
        num_object_pointer_tokens=0,
        training=None,
    ):
        output = current_vision_feats
        if current_vision_pos_embeds is not None:
            output = output + 0.1 * current_vision_pos_embeds

        output = ops.expand_dims(output, 1)
        memory = ops.expand_dims(memory, 1)
        if memory_pos_embeds is not None:
            memory_pos_embeds = ops.expand_dims(memory_pos_embeds, 1)
        else:
            memory_pos_embeds = ops.zeros_like(memory)

        rope_cos = ops.convert_to_tensor(self._rope_cos)
        rope_sin = ops.convert_to_tensor(self._rope_sin)
        rope_cos_sin = (rope_cos, rope_sin)

        for layer in self.attn_layers:
            output = layer(
                queries=output,
                keys=memory,
                key_pos_embed=memory_pos_embeds,
                rope_cos_sin=rope_cos_sin,
                num_k_exclude_rope=num_object_pointer_tokens,
                training=training,
            )

        output = self.layer_norm(output)
        output = ops.squeeze(output, 1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "kv_in_dim": self.kv_in_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ffn_hidden_size": self.ffn_hidden_size,
                "dropout": self.dropout_rate,
                "rope_theta": self.rope_theta,
                "rope_feat_sizes": self.rope_feat_sizes,
            }
        )
        return config
