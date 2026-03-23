import keras
from keras import layers, ops, utils

from kmodels.layers import StochasticDepth
from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import EOMT_MODEL_CONFIG, EOMT_WEIGHTS_CONFIG
from .eomt_layers import (
    EoMTAttention,
    EoMTEmbeddings,
    EoMTLayerScale,
    EoMTQueryInjection,
)


def eomt_mlp(x, hidden_size, mlp_ratio=4, block_prefix="layers_0"):
    """Standard MLP with GELU activation (functional).

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_size)``.
        hidden_size: Input/output dimension.
        mlp_ratio: Ratio for hidden dimension.
        block_prefix: Name prefix for sub-layers.

    Returns:
        Output tensor of same shape as ``x``.
    """
    hidden_features = int(hidden_size * mlp_ratio)
    x = layers.Dense(hidden_features, name=f"{block_prefix}_mlp_fc1")(x)
    x = layers.Activation("gelu", name=f"{block_prefix}_mlp_gelu")(x)
    x = layers.Dense(hidden_size, name=f"{block_prefix}_mlp_fc2")(x)
    return x


def eomt_swiglu_ffn(x, hidden_size, mlp_ratio=4, block_prefix="layers_0"):
    """SwiGLU feed-forward network (functional).

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_size)``.
        hidden_size: Input/output dimension.
        mlp_ratio: Ratio for computing hidden dimension.
        block_prefix: Name prefix for sub-layers.

    Returns:
        Output tensor of same shape as ``x``.
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
    """Single transformer encoder layer with LayerScale and DropPath (functional).

    Args:
        hidden_states: Input tensor of shape ``(B, seq_len, hidden_size)``.
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        layerscale_value: Initial LayerScale value.
        drop_path_rate: Stochastic depth rate.
        attention_dropout: Attention dropout rate.
        use_swiglu_ffn: Whether to use SwiGLU FFN.
        layer_norm_eps: Epsilon for LayerNorm.
        block_prefix: Name prefix for sub-layers.

    Returns:
        Output tensor of shape ``(B, seq_len, hidden_size)``.
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


def eomt_scale_layer(x, hidden_size, block_prefix="upscale_block_0"):
    """Single upscaling layer: ConvTranspose2d(2x) + GELU + DepthwiseConv2d + LayerNorm (functional).

    Args:
        x: Input tensor of shape ``(B, H, W, C)``.
        hidden_size: Number of channels.
        block_prefix: Name prefix for sub-layers.

    Returns:
        Output tensor of shape ``(B, 2*H, 2*W, C)``.
    """
    x = layers.Conv2DTranspose(
        hidden_size,
        kernel_size=2,
        strides=2,
        padding="valid",
        use_bias=True,
        name=f"{block_prefix}_conv1",
    )(x)
    x = layers.Activation("gelu", name=f"{block_prefix}_gelu")(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False,
        name=f"{block_prefix}_conv2",
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm")(x)
    return x


def eomt_scale_block(x, hidden_size, num_upscale_blocks=2):
    """Stack of upscaling layers (functional).

    Args:
        x: Input tensor of shape ``(B, H, W, C)``.
        hidden_size: Number of channels.
        num_upscale_blocks: Number of upscaling layers.

    Returns:
        Upscaled output tensor.
    """
    for i in range(num_upscale_blocks):
        x = eomt_scale_layer(x, hidden_size, block_prefix=f"upscale_block_{i}")
    return x


def eomt_mask_head(x, hidden_size):
    """Mask prediction head: 3 Dense layers with GELU activations (functional).

    Args:
        x: Input tensor of shape ``(B, num_queries, hidden_size)``.
        hidden_size: Hidden dimension.

    Returns:
        Mask embedding tensor of same shape.
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

        # Final layer norm
        sequence_output = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layernorm"
        )(hidden_states)

        # Extract query tokens and patch tokens
        query_output = sequence_output[:, :num_queries, :]
        patch_output = sequence_output[:, num_queries + num_prefix_tokens :, :]

        # Class prediction
        class_logits = layers.Dense(num_labels + 1, name="class_predictor")(
            query_output
        )

        # Mask prediction
        query_mask_tokens = eomt_mask_head(query_output, hidden_size)

        # Reshape patch tokens to spatial grid
        patch_spatial = ops.reshape(patch_output, (-1, grid_h, grid_w, hidden_size))

        # Upscale
        upscaled_features = eomt_scale_block(
            patch_spatial, hidden_size, num_upscale_blocks
        )

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
