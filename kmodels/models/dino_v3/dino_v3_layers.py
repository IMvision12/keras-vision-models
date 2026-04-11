import keras
import numpy as np
from keras import InputSpec, layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV3CLSToken(layers.Layer):
    """
    Prepends a learnable CLS token to the input patch sequence.

    The CLS token is a trainable vector broadcast across the batch and
    concatenated at position 0, following the standard ViT convention.

    Args:
        **kwargs: Additional keyword arguments passed to the base ``Layer``.
    """

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        cls = ops.broadcast_to(self.cls_token, [batch_size, 1, inputs.shape[-1]])
        return ops.concatenate([cls, inputs], axis=1)


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV3RegisterTokens(layers.Layer):
    """
    Inserts learnable register tokens after the CLS token.

    Register tokens are extra learnable vectors inserted between the CLS token
    and the patch tokens. They act as memory slots that improve attention map
    quality and reduce artifact tokens.

    Reference:
    - [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)

    Args:
        num_tokens: Integer, number of register tokens to insert.
            Defaults to ``4``.
        **kwargs: Additional keyword arguments passed to the base ``Layer``.
    """

    def __init__(self, num_tokens=4, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens

    def build(self, input_shape):
        self.register_tokens = self.add_weight(
            name="register_tokens",
            shape=(1, self.num_tokens, input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        dim = inputs.shape[-1]
        reg = ops.broadcast_to(self.register_tokens, [batch_size, self.num_tokens, dim])
        cls = inputs[:, :1, :]
        patches = inputs[:, 1:, :]
        return ops.concatenate([cls, reg, patches], axis=1)

    def get_config(self):
        config = super().get_config()
        config["num_tokens"] = self.num_tokens
        return config


def build_rope_2d_cache(grid_h, grid_w, head_dim, theta=100.0):
    """
    Precompute 2D Rotary Position Embedding (RoPE) cos/sin tables.

    Patch center coordinates are normalized to ``[-1, +1]``, multiplied by
    ``2 * pi`` and the inverse-frequency vector, then tiled to fill the full
    head dimension. This matches the HuggingFace DINOv3 implementation.

    Args:
        grid_h: Integer, number of patch rows.
        grid_w: Integer, number of patch columns.
        head_dim: Integer, dimension of each attention head.
        theta: Float, RoPE frequency base. Defaults to ``100.0``.

    Returns:
        Tuple ``(cos, sin)``, each a NumPy array of shape
        ``(grid_h * grid_w, head_dim)``.
    """
    inv_freq = 1.0 / (theta ** np.arange(0, 1, 4 / head_dim, dtype=np.float32))

    coords_h = (np.arange(0.5, grid_h, dtype=np.float32) / grid_h) * 2.0 - 1.0
    coords_w = (np.arange(0.5, grid_w, dtype=np.float32) / grid_w) * 2.0 - 1.0

    gh, gw = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack([gh.ravel(), gw.ravel()], axis=-1)

    angles = 2.0 * np.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.reshape(grid_h * grid_w, -1)
    angles = np.tile(angles, (1, 2))

    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.concatenate([-x2, x1], axis=-1)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV3Attention(layers.Layer):
    """
    Multi-head self-attention with 2D Rotary Position Embeddings (RoPE).

    Uses separate Q, K, V projections following the DINOv3 convention:
    Q and V have bias, K does not. 2D RoPE is applied to Q and K patch
    tokens only; CLS and register tokens are excluded from positional encoding.

    Args:
        dim: Integer, total embedding dimension.
        num_heads: Integer, number of attention heads. Defaults to ``8``.
        attn_drop: Float, dropout rate for attention weights. Defaults to ``0.0``.
        proj_drop: Float, dropout rate for output projection. Defaults to ``0.0``.
        num_prefix_tokens: Integer, number of non-patch tokens (CLS + registers)
            excluded from RoPE. Defaults to ``5``.
        rope_theta: Float, RoPE frequency base. Defaults to ``100.0``.
        block_prefix: String, prefix for layer naming. If None, uses ``"blocks"``.
        **kwargs: Additional keyword arguments passed to the base ``Layer``.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        num_prefix_tokens=5,
        rope_theta=100.0,
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_prefix = block_prefix or "blocks"
        prefix = f"{self.block_prefix}_"

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.rope_theta = rope_theta

        self.q_proj = layers.Dense(dim, use_bias=True, name=prefix + "attn_q_proj")
        self.k_proj = layers.Dense(dim, use_bias=False, name=prefix + "attn_k_proj")
        self.v_proj = layers.Dense(dim, use_bias=True, name=prefix + "attn_v_proj")
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, name=prefix + "attn_proj")
        self.proj_drop = layers.Dropout(proj_drop)

        self._rope_cos = None
        self._rope_sin = None

    def set_rope_cache(self, cos, sin):
        self._rope_cos = cos
        self._rope_sin = sin

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.proj.build(input_shape)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]

        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        q = ops.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
        k = ops.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
        v = ops.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        if self._rope_cos is not None:
            n_prefix = self.num_prefix_tokens
            cos = ops.expand_dims(ops.expand_dims(self._rope_cos, 0), 0)
            sin = ops.expand_dims(ops.expand_dims(self._rope_sin, 0), 0)

            q_prefix = q[:, :, :n_prefix, :]
            q_patches = _apply_rope(q[:, :, n_prefix:, :], cos, sin)
            q = ops.concatenate([q_prefix, q_patches], axis=2)

            k_prefix = k[:, :, :n_prefix, :]
            k_patches = _apply_rope(k[:, :, n_prefix:, :], cos, sin)
            k = ops.concatenate([k_prefix, k_patches], axis=2)

        q = q * self.scale
        attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
        attn = ops.softmax(attn)
        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, [0, 2, 1, 3])
        x = ops.reshape(x, input_shape)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "num_prefix_tokens": self.num_prefix_tokens,
                "rope_theta": self.rope_theta,
                "block_prefix": self.block_prefix,
            }
        )
        return config
