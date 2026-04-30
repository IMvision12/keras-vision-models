import keras
from keras import layers, ops, utils

from kmodels.layers import StochasticDepth
from kmodels.model_registry import register_model
from kmodels.weight_utils import load_weights_from_config

from .config import EOMT_MODEL_CONFIG, EOMT_WEIGHTS_CONFIG
from .eomt_layers import (
    EoMTAttention,
    EoMTEmbeddings,
    EoMTLayerScale,
    EoMTQueryInjection,
)


def eomt_mlp(x, hidden_size, mlp_ratio=4, block_prefix="layers_0"):
    """Standard two-layer MLP with GELU activation.

    Applies a dense expansion to `hidden_size * mlp_ratio` units with
    GELU activation, followed by a dense projection back to
    `hidden_size`. Used as the feedforward network in each EoMT
    encoder layer when SwiGLU is not enabled.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        x: Input tensor of shape
            `(batch_size, seq_len, hidden_size)`.
        hidden_size: Integer, input and output feature dimension.
        mlp_ratio: Integer, expansion ratio for the hidden layer.
            Defaults to `4`.
        block_prefix: String, name prefix for all layers in this
            block. Defaults to `"layers_0"`.

    Returns:
        Output tensor of shape
        `(batch_size, seq_len, hidden_size)`.
    """
    hidden_features = int(hidden_size * mlp_ratio)
    x = layers.Dense(hidden_features, name=f"{block_prefix}_mlp_fc1")(x)
    x = layers.Activation("gelu", name=f"{block_prefix}_mlp_gelu")(x)
    x = layers.Dense(hidden_size, name=f"{block_prefix}_mlp_fc2")(x)
    return x


def eomt_swiglu_ffn(x, hidden_size, mlp_ratio=4, block_prefix="layers_0"):
    """SwiGLU gated feed-forward network.

    Applies a gated linear unit with SiLU activation: the input is
    projected to `2 * hidden_features` units, split into two halves,
    one passed through SiLU and multiplied element-wise with the
    other, then projected back to `hidden_size`. The hidden dimension
    is rounded to the nearest multiple of 8 for hardware efficiency.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)
    - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

    Args:
        x: Input tensor of shape
            `(batch_size, seq_len, hidden_size)`.
        hidden_size: Integer, input and output feature dimension.
        mlp_ratio: Integer, expansion ratio used to compute the
            intermediate dimension. Defaults to `4`.
        block_prefix: String, name prefix for all layers in this
            block. Defaults to `"layers_0"`.

    Returns:
        Output tensor of shape
        `(batch_size, seq_len, hidden_size)`.
    """
    hidden_features = int(hidden_size * mlp_ratio)
    hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    x = layers.Dense(2 * hidden_features, name=f"{block_prefix}_mlp_weights_in")(x)
    x1 = x[..., :hidden_features]
    x2 = x[..., hidden_features:]
    hidden = layers.Activation("silu", name=f"{block_prefix}_mlp_silu")(x1)
    hidden = layers.Multiply(name=f"{block_prefix}_mlp_gate")([hidden, x2])
    return layers.Dense(hidden_size, name=f"{block_prefix}_mlp_weights_out")(hidden)


def eomt_encoder_layer(
    hidden_states,
    hidden_size,
    num_heads,
    mlp_ratio=4,
    layerscale_value=1.0,
    drop_path_rate=0.0,
    attention_dropout=0.0,
    use_swiglu_ffn=False,
    layer_norm_eps=1e-6,
    block_prefix="layers_0",
):
    """Single EoMT transformer encoder layer with pre-norm design.

    Applies layer-normalized multi-head self-attention followed by a
    feedforward network (standard MLP or SwiGLU), each with a
    residual connection, learnable LayerScale, and optional stochastic
    depth (DropPath). Follows the DINOv2 encoder block design.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        hidden_states: Input tensor of shape
            `(batch_size, seq_len, hidden_size)`.
        hidden_size: Integer, model hidden dimension.
        num_heads: Integer, number of attention heads.
        mlp_ratio: Integer, expansion ratio for the feedforward
            network. Defaults to `4`.
        layerscale_value: Float, initial value for the learnable
            LayerScale parameters. Defaults to `1.0`.
        drop_path_rate: Float, stochastic depth rate for dropping
            the residual branch. Defaults to `0.0`.
        attention_dropout: Float, dropout rate applied to the
            attention weight matrix. Defaults to `0.0`.
        use_swiglu_ffn: Boolean, whether to use SwiGLU instead of
            the standard GELU MLP. Defaults to `False`.
        layer_norm_eps: Float, epsilon for layer normalization.
            Defaults to `1e-6`.
        block_prefix: String, name prefix for all sub-layers.
            Defaults to `"layers_0"`.

    Returns:
        Output tensor of shape
        `(batch_size, seq_len, hidden_size)`.
    """
    residual = hidden_states
    hidden_states = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_norm1"
    )(hidden_states)
    hidden_states = EoMTAttention(
        hidden_size, num_heads, attention_dropout, name=f"{block_prefix}_attention"
    )(hidden_states)
    hidden_states = EoMTLayerScale(
        init_value=layerscale_value, name=f"{block_prefix}_layer_scale1"
    )(hidden_states)
    drop_path = (
        StochasticDepth(drop_path_rate, name=f"{block_prefix}_drop_path")
        if drop_path_rate > 0.0
        else layers.Identity(name=f"{block_prefix}_identity")
    )
    hidden_states = layers.Add(name=f"{block_prefix}_attn_residual")(
        [drop_path(hidden_states), residual]
    )

    residual = hidden_states
    hidden_states = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_norm2"
    )(hidden_states)

    if use_swiglu_ffn:
        hidden_states = eomt_swiglu_ffn(
            hidden_states, hidden_size, mlp_ratio, block_prefix
        )
    else:
        hidden_states = eomt_mlp(hidden_states, hidden_size, mlp_ratio, block_prefix)

    hidden_states = EoMTLayerScale(
        init_value=layerscale_value, name=f"{block_prefix}_layer_scale2"
    )(hidden_states)
    hidden_states = layers.Add(name=f"{block_prefix}_mlp_residual")(
        [drop_path(hidden_states), residual]
    )

    return hidden_states


def eomt_scale_layer(
    x, hidden_size, data_format="channels_last", block_prefix="upscale_block_0"
):
    """Single 2x spatial upscaling layer for mask feature decoding.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        x: Input tensor.
        hidden_size: int, number of channels.
        data_format: string, image data format. Defaults to ``"channels_last"``.
        block_prefix: string, name prefix for sub-layers.
            Defaults to ``"upscale_block_0"``.

    Returns:
        Output tensor with 2x spatial resolution.
    """
    x = layers.Conv2DTranspose(
        hidden_size,
        kernel_size=2,
        strides=2,
        padding="valid",
        use_bias=True,
        data_format=data_format,
        name=f"{block_prefix}_conv1",
    )(x)
    x = layers.Activation("gelu", name=f"{block_prefix}_gelu")(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_prefix}_conv2",
    )(x)
    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm")(x)
    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2))(x)
    return x


def eomt_scale_block(x, hidden_size, num_upscale_blocks=2, data_format="channels_last"):
    """Stack of spatial upscaling layers for mask feature decoding.

    Args:
        x: Input tensor.
        hidden_size: int, number of channels.
        num_upscale_blocks: int, number of 2x upscaling layers.
            Defaults to ``2``.
        data_format: string, image data format. Defaults to ``"channels_last"``.

    Returns:
        Output tensor with upscaled spatial resolution.
    """
    for i in range(num_upscale_blocks):
        x = eomt_scale_layer(
            x,
            hidden_size,
            data_format=data_format,
            block_prefix=f"upscale_block_{i}",
        )
    return x


def eomt_mask_head(x, hidden_size):
    """Mask prediction head with three dense layers and GELU activations.

    Projects each query token through a 3-layer MLP to produce a
    mask embedding vector. The output is later multiplied with
    upscaled spatial features via einsum to produce per-query mask
    logits.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        x: Input tensor of shape
            `(batch_size, num_queries, hidden_size)`.
        hidden_size: Integer, hidden dimension for all three dense
            layers.

    Returns:
        Mask embedding tensor of shape
        `(batch_size, num_queries, hidden_size)`.
    """
    x = layers.Dense(hidden_size, name="mask_head_fc1")(x)
    x = layers.Activation("gelu", name="mask_head_gelu1")(x)
    x = layers.Dense(hidden_size, name="mask_head_fc2")(x)
    x = layers.Activation("gelu", name="mask_head_gelu2")(x)
    x = layers.Dense(hidden_size, name="mask_head_fc3")(x)
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class EoMT(keras.Model):
    """Encoder-only Mask Transformer (EoMT) for universal image segmentation.

    EoMT repurposes a plain DINOv2-style Vision Transformer for image
    segmentation without a task-specific decoder. Learned object
    queries are injected into the final `num_blocks` encoder layers,
    enabling joint self-attention between image patch tokens and
    query tokens. After the encoder, query tokens are projected to
    class logits via a linear head, and mask logits are computed as
    the bilinear product of query mask embeddings and spatially
    upscaled patch features.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        hidden_size: Integer, transformer hidden dimension.
            Defaults to `1024`.
        num_hidden_layers: Integer, total number of transformer
            encoder layers. Defaults to `24`.
        num_attention_heads: Integer, number of attention heads per
            layer. Defaults to `16`.
        mlp_ratio: Integer, expansion ratio for the feedforward
            network. Defaults to `4`.
        patch_size: Integer, height and width of each image patch.
            Defaults to `16`.
        num_register_tokens: Integer, number of DINOv2-style register
            tokens prepended after the CLS token. Defaults to `4`.
        num_blocks: Integer, number of final encoder blocks that
            receive the injected object queries. Defaults to `4`.
        num_upscale_blocks: Integer, number of 2x upscaling layers
            applied to patch features before mask prediction.
            Defaults to `2`.
        num_queries: Integer, number of learned object queries.
            Defaults to `200`.
        num_labels: Integer, number of segmentation classes.
            Defaults to `133`.
        layerscale_value: Float, initial value for the learnable
            LayerScale parameters. Defaults to `1e-5`.
        drop_path_rate: Float, stochastic depth rate.
            Defaults to `0.0`.
        attention_dropout: Float, dropout rate for the attention
            weight matrix. Defaults to `0.0`.
        use_swiglu_ffn: Boolean, whether to use SwiGLU instead of
            the standard GELU MLP. Defaults to `False`.
        layer_norm_eps: Float, epsilon for layer normalization.
            Defaults to `1e-6`.
        input_shape: Optional tuple of integers specifying the input
            shape (excluding batch size), e.g., `(640, 640, 3)`.
        input_tensor: Optional Keras tensor to use as the model input.
        name: String, the name of the model.
            Defaults to `"EoMT"`.
        **kwargs: Additional keyword arguments passed to the
            `keras.Model` class.

    Returns:
        A `keras.Model` instance with dict outputs:
        - `"class_logits"`: `(batch_size, num_queries, num_labels + 1)`
        - `"mask_logits"`: `(batch_size, num_queries, H_up, W_up)`
          where `H_up = image_size // patch_size * 2^num_upscale_blocks`.
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

        data_format = keras.config.image_data_format()

        if data_format == "channels_first":
            image_size = input_shape[1]
        else:
            image_size = input_shape[0]
        grid_h = image_size // patch_size
        grid_w = image_size // patch_size
        num_prefix_tokens = 1 + num_register_tokens

        embeddings_layer = EoMTEmbeddings(
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_size=image_size,
            num_register_tokens=num_register_tokens,
            name="embeddings",
        )
        hidden_states = embeddings_layer(img_input)

        query_injection = EoMTQueryInjection(num_queries, hidden_size, name="query")
        query_injection_idx = num_hidden_layers - num_blocks

        for i in range(num_hidden_layers):
            if i == query_injection_idx:
                hidden_states = query_injection(hidden_states)

            hidden_states = eomt_encoder_layer(
                hidden_states,
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                layerscale_value=layerscale_value,
                drop_path_rate=drop_path_rate,
                attention_dropout=attention_dropout,
                use_swiglu_ffn=use_swiglu_ffn,
                layer_norm_eps=layer_norm_eps,
                block_prefix=f"layers_{i}",
            )

        sequence_output = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layernorm"
        )(hidden_states)

        query_output = sequence_output[:, :num_queries, :]
        patch_output = sequence_output[:, num_queries + num_prefix_tokens :, :]

        class_logits = layers.Dense(num_labels + 1, name="class_predictor")(
            query_output
        )

        query_mask_tokens = eomt_mask_head(query_output, hidden_size)

        if data_format == "channels_first":
            patch_spatial = ops.reshape(patch_output, (-1, hidden_size, grid_h, grid_w))
        else:
            patch_spatial = ops.reshape(patch_output, (-1, grid_h, grid_w, hidden_size))

        upscaled_features = eomt_scale_block(
            patch_spatial,
            hidden_size,
            num_upscale_blocks,
            data_format=data_format,
        )

        if data_format == "channels_first":
            mask_logits = ops.einsum(
                "bqc,bchw->bqhw", query_mask_tokens, upscaled_features
            )
        else:
            mask_logits = ops.einsum(
                "bqc,bhwc->bqhw", query_mask_tokens, upscaled_features
            )

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
    """Factory function for creating EoMT model variants.

    Looks up the architecture configuration for the given variant
    name, infers `num_labels` and `input_shape` from the weight
    identifier when not specified, instantiates an `EoMT` model, and
    optionally loads pretrained weights.

    Args:
        variant: String, model variant name (e.g., `"EoMTLarge"`).
        num_queries: Integer, number of learned object queries.
            Defaults to `200`.
        num_labels: Integer or `None`, number of segmentation
            classes. Inferred from the weight identifier when `None`.
        input_shape: Optional tuple of integers specifying the input
            shape. Inferred from the weight identifier when `None`
            (defaults to `(640, 640, 3)`).
        input_tensor: Optional Keras tensor to use as the model input.
        weights: String or `None`, pretrained weight identifier
            (e.g., `"coco_panoptic_640"`) or a path to a weights
            file. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to `EoMT`.

    Returns:
        A configured `EoMT` model instance.
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
        mlp_ratio=4,
        patch_size=16,
        num_register_tokens=4,
        num_blocks=config["num_blocks"],
        num_upscale_blocks=2,
        num_queries=num_queries,
        num_labels=num_labels,
        layerscale_value=config["layerscale_value"],
        use_swiglu_ffn=False,
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
def EoMTSmall(
    num_queries=200,
    num_labels=None,
    input_shape=None,
    input_tensor=None,
    weights="coco_panoptic_640",
    **kwargs,
):
    return _create_eomt_model(
        "EoMTSmall",
        num_queries=num_queries,
        num_labels=num_labels,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def EoMTBase(
    num_queries=200,
    num_labels=None,
    input_shape=None,
    input_tensor=None,
    weights="coco_panoptic_640",
    **kwargs,
):
    return _create_eomt_model(
        "EoMTBase",
        num_queries=num_queries,
        num_labels=num_labels,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def EoMTLarge(
    num_queries=200,
    num_labels=None,
    input_shape=None,
    input_tensor=None,
    weights="coco_panoptic_640",
    **kwargs,
):
    return _create_eomt_model(
        "EoMTLarge",
        num_queries=num_queries,
        num_labels=num_labels,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )
