import keras
from keras import InputSpec, layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.layers import ImageNormalizationLayer, LayerScale
from kmodels.model_registry import register_model
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import VIT_MODEL_CONFIG, VIT_WEIGHTS_CONFIG


@keras.saving.register_keras_serializable(package="kmodels")
class ClassDistToken(layers.Layer):
    """
    Implements learnable class and distillation tokens for Vision Transformer (ViT),
    Data-efficient image Transformer (DeiT), and Pyramid Vision Transformer (PiT) architectures.

    This layer can operate in three modes:
    1. Standard ViT mode: Only adds a class token
    2. DeiT mode: Adds separate class and distillation tokens
    3. PiT mode: Adds combined class and distillation tokens in a single weight tensor

    Args:
        use_distillation (bool): If True, adds distillation token(s) alongside class token(s).
            Defaults to False.
        combine_tokens (bool): If True, stores class and distillation tokens in a single weight
            tensor (PiT style). If False, uses separate weight tensors (ViT/DeiT style).
            Only applies when use_distillation=True. Defaults to False.
        name (str, optional): Name for the layer instance.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - ViT: https://arxiv.org/abs/2010.11929
        - DeiT: https://arxiv.org/abs/2012.12877
        - PiT: https://arxiv.org/abs/2103.14030
    """

    def __init__(
        self, use_distillation=False, combine_tokens=False, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.use_distillation = use_distillation
        self.combine_tokens = combine_tokens
        self.num_tokens = 2 if use_distillation else 1

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]

        if self.combine_tokens and self.use_distillation:
            # Combined tokens (PiT-style)
            self.tokens = self.add_weight(
                name="cls_token",
                shape=(1, 2, self.hidden_size),
                initializer="zeros",
                trainable=True,
            )
        else:
            # Class token
            self.cls = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.hidden_size),
                initializer="zeros",
                trainable=True,
            )
            # Distillation token for DeiT
            if self.use_distillation:
                self.dist = self.add_weight(
                    name="dist_token",
                    shape=(1, 1, self.hidden_size),
                    initializer="zeros",
                    trainable=True,
                )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        if self.combine_tokens and self.use_distillation:
            tokens_broadcasted = ops.broadcast_to(
                self.tokens, [batch_size, 2, self.hidden_size]
            )
            return ops.concatenate([tokens_broadcasted, inputs], axis=1)
        else:
            cls_broadcasted = ops.broadcast_to(
                self.cls, [batch_size, 1, self.hidden_size]
            )

            if self.use_distillation:
                dist_broadcasted = ops.broadcast_to(
                    self.dist, [batch_size, 1, self.hidden_size]
                )
                return ops.concatenate(
                    [cls_broadcasted, dist_broadcasted, inputs], axis=1
                )
            else:
                return ops.concatenate([cls_broadcasted, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "use_distillation": self.use_distillation,
                "combine_tokens": self.combine_tokens,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class AddPositionEmbs(layers.Layer):
    """
    A custom Keras layer that adds learnable position embeddings to input tensors with support for
    flexible grid sizes and optional class/distillation token handling.

    The layer supports three modes of operation:
      1. Standard mode: Creates position embeddings for both patches and a class token.
      2. FlexiViT mode: Creates embeddings only for patches, handling a class token separately.
         In this mode, the layer accepts either a pure patch token input (grid_h * grid_w tokens)
         or an input with a leading class token (grid_h * grid_w + 1 tokens).
      3. DeiT mode: Creates embeddings for patches, a class token, and a distillation token.
         In this case the expected sequence length is grid_h * grid_w + 2 tokens.

    For PiT, the recommended usage is to set `no_embed_class=True` so that the patch tokens
    are first augmented with positional embeddings. Then, the class (or combined class/distillation)
    token is added later in the model pipeline.

    Features:
      - Configurable grid dimensions for position embeddings.
      - Dynamic resizing of position embeddings during loading through bilinear interpolation.
      - Compatible with standard Vision Transformer, FlexiViT, DeiT, and PiT architectures.
      - Handles class and distillation token embeddings appropriately based on mode.
      - When no_embed_class=True, the layer applies positional embeddings only to patch tokens.
        If a class token is present at the beginning, it is preserved and concatenated back after
        the patch tokens are positionally embedded.

    Args:
        grid_h (int): Height of the position embedding grid.
        grid_w (int): Width of the position embedding grid.
        no_embed_class (bool):
            - If False (default), operates in standard mode where position embeddings are learned for
              the entire input (patches plus token(s)).
            - If True, operates in FlexiViT or PiT mode where positional embeddings are applied only to
              the patch tokens. In this mode the input can be either a sequence of patch tokens only
              (grid_h * grid_w tokens) or a sequence with a leading class token (grid_h * grid_w + 1 tokens).
        use_distillation (bool):
            If True, operates in DeiT mode with an additional distillation token.
            When no_embed_class is False, the expected input sequence length is grid_h * grid_w + 2.
            Defaults to False.
        name (str, optional): Name of the layer.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input Shape:
        3D tensor with shape: `(batch_size, sequence_length, embedding_dim)`, where sequence_length should be:
          * Standard mode: grid_h * grid_w + 1 (patch tokens + class token)
          * FlexiViT mode / PiT mode: either grid_h * grid_w (patch tokens only) or grid_h * grid_w + 1
          * DeiT mode: grid_h * grid_w + 2 (patch tokens + class token + distillation token)

    Output Shape:
        Same as the input shape: `(batch_size, sequence_length, embedding_dim)`. In FlexiViT or PiT mode,
        if a class token is present at the beginning, it is preserved and positional embeddings are added
        only to the patch tokens.

    The layer supports loading weights from different grid sizes through bilinear interpolation,
    making it flexible for transfer learning and model adaptation scenarios. When loading weights,
    it automatically handles the conversion between different modes (standard / FlexiViT / DeiT),
    adjusting the position embeddings appropriately.
    """

    def __init__(
        self,
        grid_h,
        grid_w,
        no_embed_class=False,
        use_distillation=False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.resize_mode = "bilinear"

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, received: {len(input_shape)}D")

        num_patches = self.grid_h * self.grid_w

        if self.no_embed_class:
            if input_shape[1] == num_patches:
                self.skip_cls = False
            elif input_shape[1] == num_patches + 1:
                self.skip_cls = True
            else:
                raise ValueError(
                    f"Input sequence length {input_shape[1]} does not match expected length "
                    f"{num_patches} or {num_patches + 1} (grid: {self.grid_h}x{self.grid_w})"
                )
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, num_patches, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        else:
            expected_length = num_patches + (2 if self.use_distillation else 1)
            if input_shape[1] != expected_length:
                raise ValueError(
                    f"Input sequence length {input_shape[1]} does not match expected length "
                    f"{expected_length} (grid: {self.grid_h}x{self.grid_w} + tokens)"
                )
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, expected_length, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        if self.no_embed_class:
            if hasattr(self, "skip_cls") and self.skip_cls:
                cls_token = inputs[:, :1]
                patch_tokens = inputs[:, 1:]
                patch_tokens = patch_tokens + self.position_embedding
                return ops.concatenate([cls_token, patch_tokens], axis=1)
            else:
                return inputs + self.position_embedding
        elif self.use_distillation:
            tokens = inputs[:, :2]
            patch_tokens = inputs[:, 2:]
            token_pos_embed = self.position_embedding[:, :2]
            patch_pos_embed = self.position_embedding[:, 2:]
            tokens = tokens + token_pos_embed
            patch_tokens = patch_tokens + patch_pos_embed
            return ops.concatenate([tokens, patch_tokens], axis=1)
        else:
            return inputs + self.position_embedding

    def compute_output_shape(self, input_shape):
        return input_shape

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["grid_h"] = self.grid_h
        store["grid_w"] = self.grid_w
        store["no_embed_class"] = self.no_embed_class
        store["use_distillation"] = self.use_distillation

    def load_own_variables(self, store):
        source_h = int(store["grid_h"][...])
        source_w = int(store["grid_w"][...])

        try:
            source_no_embed_class = bool(store["no_embed_class"][...])
        except KeyError:
            source_no_embed_class = False

        try:
            source_use_distillation = bool(store["use_distillation"][...])
        except KeyError:
            source_use_distillation = False

        if source_h == self.grid_h and source_w == self.grid_w:
            self.position_embedding.assign(store["0"])
            return

        pos_embed = store["0"]

        if not source_no_embed_class:
            if source_use_distillation:
                spatial_pos_embed = pos_embed[:, 2:]
                token_pos_embed = pos_embed[:, :2]
            else:
                spatial_pos_embed = pos_embed[:, 1:]
                token_pos_embed = pos_embed[:, :1]
        else:
            spatial_pos_embed = pos_embed

        embed_dim = spatial_pos_embed.shape[-1]

        spatial_pos_embed = ops.cast(spatial_pos_embed, dtype="float32")
        spatial_pos_embed = ops.reshape(
            spatial_pos_embed, [1, source_h, source_w, embed_dim]
        )

        spatial_pos_embed = ops.image.resize(
            spatial_pos_embed,
            size=[self.grid_h, self.grid_w],
            interpolation=self.resize_mode,
            antialias=True,
        )

        spatial_pos_embed = ops.reshape(
            spatial_pos_embed, [1, self.grid_h * self.grid_w, embed_dim]
        )

        if self.no_embed_class:
            pos_embed = spatial_pos_embed
        else:
            if not source_no_embed_class:
                if self.use_distillation:
                    if source_use_distillation:
                        cls_dist_pos_embed = token_pos_embed
                    else:
                        cls_dist_pos_embed = ops.concatenate(
                            [token_pos_embed, token_pos_embed], axis=1
                        )
                else:
                    if source_use_distillation:
                        cls_dist_pos_embed = token_pos_embed[:, :1]
                    else:
                        cls_dist_pos_embed = token_pos_embed
            else:
                if self.use_distillation:
                    cls_dist_pos_embed = ops.zeros((1, 2, embed_dim))
                else:
                    cls_dist_pos_embed = ops.zeros((1, 1, embed_dim))

            pos_embed = ops.concatenate([cls_dist_pos_embed, spatial_pos_embed], axis=1)

        self.position_embedding.assign(pos_embed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "resize_mode": self.resize_mode,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class MultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention layer implementing scaled dot-product attention.

    This layer implements the standard multi-head self-attention mechanism where input is split
    into multiple attention heads operating in parallel. Each head performs scaled dot-product
    attention independently, after which results are concatenated and projected back to the
    original dimension.

    Key Features:
        - Independent parallel attention heads for capturing different relationship patterns
        - Scaled dot-product attention with optional layer normalization
        - Configurable attention and projection dropout
        - Optional bias terms in query/key/value projections
        - Support for both 3D and 4D input tensors

    Args:
        dim (int): Total dimension of the input and output features. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int, optional): Number of parallel attention heads. Defaults to 8
        qkv_bias (bool, optional): If True, adds learnable bias terms to the query, key,
            and value projections. Defaults to False
        qk_norm (bool, optional): If True, applies layer normalization to query and key
            tensors before attention computation. Defaults to False
        attn_drop (float, optional): Dropout rate applied to attention weights. Defaults to 0.0
        proj_drop (float, optional): Dropout rate applied to the output projection. Defaults to 0.0
        epsilon (float, optional): Small constant used in normalization. Defaults to 1e-6
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - 3D tensor: (batch_size, sequence_length, feature_dim)
        - 4D tensor: (batch_size, height, width, feature_dim)

    Output shape:
        - Same as input shape
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        epsilon=1e-6,
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.epsilon = epsilon

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=prefix + "attn_qkv",
        )

        self.q_norm = (
            layers.LayerNormalization(
                epsilon=self.epsilon,
                dtype=self.dtype_policy,
                name=prefix + "attn_norm1",
            )
            if qk_norm
            else None
        )
        self.k_norm = (
            layers.LayerNormalization(
                epsilon=self.epsilon,
                dtype=self.dtype_policy,
                name=prefix + "attn_norm2",
            )
            if qk_norm
            else None
        )

        self.attn_drop = layers.Dropout(
            attn_drop,
            dtype=self.dtype_policy,
        )
        self.proj = layers.Dense(
            dim, dtype=self.dtype_policy, name=prefix + "attn_proj"
        )
        self.proj_drop = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))

        if self.input_spec.ndim not in (3, 4):
            raise ValueError(
                f"MultiHeadSelfAttention expects 3D or 4D input tensor, but received shape: {input_shape}"
            )

        feature_dim = input_shape[-1]
        if feature_dim != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.dim}"
            )

        self.qkv.build(input_shape)
        self.proj.build(input_shape)

        if self.q_norm is not None:
            norm_shape = (input_shape[-1],)
            self.q_norm.build(norm_shape)
        if self.k_norm is not None:
            norm_shape = (input_shape[-1],)
            self.k_norm.build(norm_shape)

        self.built = True

    def call(self, inputs, training=None):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        ndim = len(inputs.shape)

        qkv = self.qkv(inputs)

        qkv_split = ops.split(qkv, 3, axis=-1)
        q, k, v = qkv_split

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q * self.scale

        if ndim == 3:
            q = ops.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
            k = ops.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
            v = ops.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

            q = ops.transpose(q, [0, 2, 1, 3])
            k = ops.transpose(k, [0, 2, 1, 3])
            v = ops.transpose(v, [0, 2, 1, 3])

            attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
            attn = ops.softmax(attn)
            attn = self.attn_drop(attn, training=training)
            x = ops.matmul(attn, v)

            x = ops.transpose(x, [0, 2, 1, 3])
            x = ops.reshape(x, input_shape)
        else:
            q = ops.reshape(
                q,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )
            k = ops.reshape(
                k,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )
            v = ops.reshape(
                v,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )

            q = ops.transpose(q, [0, 1, 3, 2, 4])
            k = ops.transpose(k, [0, 1, 3, 2, 4])
            v = ops.transpose(v, [0, 1, 3, 2, 4])

            q = ops.reshape(q, [-1, self.num_heads, input_shape[2], self.head_dim])
            k = ops.reshape(k, [-1, self.num_heads, input_shape[2], self.head_dim])
            v = ops.reshape(v, [-1, self.num_heads, input_shape[2], self.head_dim])

            attn = ops.matmul(q, ops.swapaxes(k, -2, -1))
            attn = ops.softmax(attn)
            attn = self.attn_drop(attn, training=training)
            x = ops.matmul(attn, v)

            x = ops.reshape(
                x,
                [
                    batch_size,
                    input_shape[1],
                    self.num_heads,
                    input_shape[2],
                    self.head_dim,
                ],
            )
            x = ops.transpose(x, [0, 1, 3, 2, 4])
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
                "qkv_bias": self.qkv.use_bias,
                "qk_norm": self.q_norm is not None,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "epsilon": self.epsilon,
                "block_prefix": self.block_prefix,
            }
        )
        return config


def mlp_block(inputs, hidden_features, out_features=None, drop=0.0, block_idx=0):
    """
    Implements a Multi-Layer Perceptron (MLP) block typically used in transformer architectures.

    The block consists of two fully connected (dense) layers with GELU activation,
    dropout regularization, and optional feature dimension specification.

    Args:
        inputs: Input tensor to the MLP block.
        hidden_features: Number of neurons in the first (hidden) dense layer.
        out_features: Number of neurons in the second (output) dense layer.
            If None, uses the same number of features as the input. Default is None.
        drop: Dropout rate applied after each dense layer. Default is 0.
        block_idx: Index of the block, used for naming layers. Default is 0.

    Returns:
        Output tensor after passing through the MLP block.
    """
    x = layers.Dense(
        hidden_features, use_bias=True, name=f"blocks_{block_idx}_dense_1"
    )(inputs)
    x = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_1")(x)
    x = layers.Dense(out_features, use_bias=True, name=f"blocks_{block_idx}_dense_2")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_2")(x)
    return x


def transformer_block(
    inputs,
    dim: int,
    num_heads: int,
    channels_axis,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    proj_drop: float = 0.0,
    attn_drop: float = 0.0,
    block_idx: int = 0,
    init_values: float = None,
):
    """
    Implements a standard Transformer block with self-attention and MLP layers.

    The block consists of two main components:
    1. Multi-Head Self-Attention layer with optional normalization
    2. Multi-Layer Perceptron (MLP) layer

    Both components use layer normalization and residual connections.

    Args:
        inputs: Input tensor to the transformer block.
        dim: Dimensionality of the input and output features.
        num_heads: Number of attention heads in the multi-head attention mechanism.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        mlp_ratio: Expansion ratio for the hidden dimension in the MLP layer.
            Hidden layer size will be `dim * mlp_ratio`. Default is 4.0.
        qkv_bias: Whether to use bias in the query, key, and value projections.
            Default is False.
        qk_norm: Whether to apply normalization to query and key before attention.
            Default is False.
        proj_drop: Dropout rate for the projection layers. Default is 0.
        attn_drop: Dropout rate for the attention probabilities. Default is 0.
        block_idx: Index of the block, used for naming layers. Default is 0.
        use_layer_scale: Whether to use LayerScale for scaling residual connections.
        init_valuess: Initial values for LayerScale weights if enabled. Default is None.

    Returns:
        Output tensor after passing through the transformer block,
        with the same shape and dimensionality as the input.
    """

    # Attention branch
    x = layers.LayerNormalization(
        epsilon=1e-6, axis=channels_axis, name=f"blocks_{block_idx}_layernorm_1"
    )(inputs)
    x = MultiHeadSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        block_prefix=f"blocks_{block_idx}",
    )(x)
    if init_values:
        x = LayerScale(
            init_values=init_values,
            name=f"blocks_{block_idx}_layerscale_1",
        )(x)
    x = keras.layers.Add(name=f"blocks_{block_idx}_add_1")([x, inputs])

    # MLP branch
    y = layers.LayerNormalization(
        epsilon=1e-6,
        axis=channels_axis,
        name=f"blocks_{block_idx}_layernorm_2",
    )(x)
    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=proj_drop,
        block_idx=block_idx,
    )
    if init_values:
        y = LayerScale(
            init_values=init_values,
            name=f"blocks_{block_idx}_layerscale_2",
        )(y)
    outputs = keras.layers.Add(name=f"blocks_{block_idx}_add_2")([x, y])
    return outputs


@keras.saving.register_keras_serializable(package="kmodels")
class VisionTransformer(keras.Model):
    """Instantiates the Vision Transformer (ViT) architecture with optional FlexiViT support.

    This implementation supports both the original ViT architecture and FlexiViT modifications,
    allowing for flexible patch sizes and dynamic position embeddings.

    References:
    - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
    - [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013)

    Args:
        patch_size: Integer, size of the patches to extract from the image.
            Defaults to `16`.
        dim: Integer, the embedding dimension for the transformer.
            Defaults to `768`.
        depth: Integer, number of transformer blocks.
            Defaults to `12`.
        num_heads: Integer, number of attention heads in each block.
            Defaults to `12`.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Defaults to `4.0`.
        qkv_bias: Boolean, whether to include bias for query, key, and value projections.
            Defaults to `True`.
        qk_norm: Boolean, whether to apply layer normalization to query and key.
            Defaults to `False`.
        drop_rate: Float, dropout rate applied to the model.
            Defaults to `0.1`.
        attn_drop_rate: Float, dropout rate applied to attention weights.
            Defaults to `0.0`.
        no_embed_class: Boolean, enables FlexiViT mode for position embeddings.
            If True, position embeddings are created only for patches (not class token),
            enabling flexible patch sizes. If False, uses standard ViT embeddings.
            Defaults to `False`.
        include_top: Boolean, whether to include the classification head.
            Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to `False`.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options in `keras-vision`.
        input_shape: Optional tuple specifying the shape of the input data.
            When using FlexiViT (no_embed_class=True), input shape can be flexible.
            For standard ViT, defaults to (224, 224, 3).
        input_tensor: Optional Keras tensor as input.
            Useful for connecting the model to other Keras components.
        pooling: Optional pooling mode when `include_top=False`:
            - `None`: output is the last transformer block's output
            - `"avg"`: global average pooling is applied
            - `"max"`: global max pooling is applied
        num_classes: Integer, the number of output classes for classification.
            Defaults to `1000`.
        classifier_activation: String or callable, activation function for the top layer.
            Set to `None` to return logits. Defaults to `"softmax"`.
        name: String, the name of the model. Defaults to `"ViT"`.

    Returns:
        A Keras `Model` instance.

    Example:
        ```python
        # Standard ViT
        model = ViT(
            patch_size=16,
            dim=768,
            no_embed_class=False,
            input_shape=(224, 224, 3)
        )

        # FlexiViT with dynamic patch sizes
        model = ViT(
            patch_size=16,
            dim=768,
            no_embed_class=True,
            input_shape=(240, 240, 3)  # Can use different input sizes
        )
        ```
    """

    def __init__(
        self,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        no_embed_class=False,
        use_distillation=False,
        init_values=None,
        include_top=True,
        as_backbone=False,
        include_normalization=True,
        normalization_mode="imagenet",
        weights="augreg_in21k_ft_in1k_224",
        input_shape=None,
        input_tensor=None,
        pooling=None,
        num_classes=None,
        classifier_activation="softmax",
        name="ViT",
        **kwargs,
    ):
        if include_top and num_classes is None:
            raise ValueError(
                f"If `include_top` is True, `num_classes` must be specified. "
                f"Received: {num_classes}"
            )

        if include_top and as_backbone:
            raise ValueError(
                "Cannot use `as_backbone=True` with `include_top=True`. "
                f"Received: as_backbone={as_backbone}, include_top={include_top}"
            )

        if pooling is not None and pooling not in ["avg", "max"]:
            raise ValueError(
                "The `pooling` argument should be one of 'avg', 'max', or None. "
                f"Received: pooling={pooling}"
            )

        if (
            include_top
            and weights is not None
            and (weights == "augreg_in21k" or weights.endswith("in21k"))
            and num_classes != 21843
        ):
            raise ValueError(
                f"When using {weights} weights, num_classes must be 21843. "
                f"Received num_classes: {num_classes}"
            )

        if (
            include_top
            and weights is not None
            and weights.endswith(("in1k", "ft_in1k"))
            and num_classes != 1000
        ):
            raise ValueError(
                f"When using {weights}, num_classes must be 1000. "
                f"Received num_classes: {num_classes}"
            )

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        if weights and "384" in weights:
            default_size = 384
        else:
            default_size = 224

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=default_size,
            min_size=32,
            data_format=data_format,
            require_flatten=include_top,
            weights=weights,
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input
        features = []

        if data_format == "channels_first":
            if len(input_shape) == 3:
                _, height, width = input_shape
            else:
                height, width = input_shape[1:]
        else:  # channels_last
            if len(input_shape) == 3:
                height, width, _ = input_shape
            else:
                height, width = input_shape[:2]

        grid_h = height // patch_size
        grid_w = width // patch_size

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(inputs)
            if include_normalization
            else inputs
        )

        x = layers.Conv2D(
            filters=dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            name="conv1",
        )(x)

        x = layers.Reshape((-1, dim))(x)
        x = ClassDistToken(use_distillation=use_distillation, name="cls_token")(x)

        x = AddPositionEmbs(
            name="pos_embed",
            no_embed_class=no_embed_class,
            use_distillation=use_distillation,
            grid_h=grid_h,
            grid_w=grid_w,
        )(x)
        features.append(x)

        x = layers.Dropout(drop_rate)(x)

        for i in range(depth):
            x = transformer_block(
                x,
                dim=dim,
                num_heads=num_heads,
                channels_axis=channels_axis,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                init_values=init_values,
                block_idx=i,
            )
            features.append(x)

        x = layers.LayerNormalization(
            epsilon=1e-6, axis=channels_axis, name="final_layernorm"
        )(x)

        if include_top:
            if use_distillation:
                cls_token = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(x)
                dist_token = layers.Lambda(lambda v: v[:, 1], name="ExtractDistToken")(
                    x
                )

                cls_token = layers.Dropout(drop_rate)(cls_token)
                dist_token = layers.Dropout(drop_rate)(dist_token)

                cls_head = layers.Dense(
                    num_classes, activation=classifier_activation, name="predictions"
                )(cls_token)

                dist_head = layers.Dense(
                    num_classes,
                    activation=classifier_activation,
                    name="predictions_dist",
                )(dist_token)

                x = (cls_head + dist_head) / 2

            else:
                x = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
                x = layers.Dropout(drop_rate)(x)
                x = layers.Dense(
                    num_classes, activation=classifier_activation, name="predictions"
                )(x)
        elif as_backbone:
            x = features
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling1D(
                    data_format=data_format, name="avg_pool"
                )(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling1D(data_format=data_format, name="max_pool")(
                    x
                )

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.init_values = init_values
        self.include_top = include_top
        self.as_backbone = as_backbone
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_norm": self.qk_norm,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "init_values": self.init_values,
                "include_top": self.include_top,
                "as_backbone": self.as_backbone,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "pooling": self.pooling,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model
def ViTTiny16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTTiny16",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTTiny16"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTTiny16", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTSmall16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTSmall16",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTSmall16"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTSmall16", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTSmall32(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTSmall32",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTSmall32"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTSmall32", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTBase16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTBase16",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTBase16"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTBase16", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTBase32(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTBase32",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTBase32"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTBase32", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTLarge16(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="augreg_in21k_ft_in1k_224",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTLarge16",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTLarge16"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTLarge16", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def ViTLarge32(
    include_top=True,
    as_backbone=False,
    include_normalization=True,
    normalization_mode="imagenet",
    weights="orig_in21k_ft_in1k_384",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    name="ViTLarge32",
    **kwargs,
):
    model = VisionTransformer(
        **VIT_MODEL_CONFIG["ViTLarge32"],
        include_top=include_top,
        as_backbone=as_backbone,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        weights=weights,
        name=name,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
    if weights in get_all_weight_names(VIT_WEIGHTS_CONFIG):
        load_weights_from_config("ViTLarge32", weights, model, VIT_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
