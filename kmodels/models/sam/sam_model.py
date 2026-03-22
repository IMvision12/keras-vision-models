import keras
from keras import layers, utils

from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import SAM_MODEL_CONFIG, SAM_WEIGHTS_CONFIG
from .sam_layers import (
    SAMAbsolutePositionEmbedding,
    SAMImagePositionalEmbeddings,
    SAMMaskDecoderLayer,
    SAMPositionalEmbedding,
    SAMPromptEncoderLayer,
    SAMVisionLayer,
)


def sam_vision_neck(inputs, hidden_size, output_channels, name="vision_encoder_neck"):
    """Neck that projects vision encoder output to the mask decoder dimension.

    Two Conv2D layers (1x1 then 3x3) with LayerNorm between, converting from
    ``hidden_size`` to ``output_channels``.

    Args:
        inputs: Input tensor from the vision encoder.
        hidden_size: Vision encoder hidden dimension.
        output_channels: Output channel dimension (mask decoder hidden size).
        name: Name prefix for the layers.

    Returns:
        Output tensor of shape ``(B, H, W, output_channels)``.
    """
    x = layers.Conv2D(
        output_channels, kernel_size=1, use_bias=False, name=f"{name}_conv1"
    )(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layer_norm1")(x)
    x = layers.Conv2D(
        output_channels,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layer_norm2")(x)
    return x


def sam_feed_forward(
    inputs, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False, name=""
):
    """Multi-layer perceptron used in the mask decoder for iou/mask heads.

    Args:
        inputs: Input tensor.
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        num_layers: Total number of linear layers.
        sigmoid_output: Apply sigmoid to the final output.
        name: Name prefix for the layers.

    Returns:
        Output tensor.
    """
    x = layers.Dense(hidden_dim, name=f"{name}_proj_in")(inputs)
    x = layers.Activation("relu", name=f"{name}_relu_0")(x)
    for i in range(num_layers - 2):
        x = layers.Dense(hidden_dim, name=f"{name}_layers_{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu_{i + 1}")(x)
    x = layers.Dense(output_dim, name=f"{name}_proj_out")(x)
    if sigmoid_output:
        x = layers.Activation("sigmoid", name=f"{name}_sigmoid")(x)
    return x


def sam_mask_embedding(
    inputs, hidden_size=256, mask_input_channels=16, layer_norm_eps=1e-6, name=""
):
    """Embeds dense mask prompts through a small CNN.

    Three Conv2D layers downsample the mask by 4x total, mapping a single-channel
    mask to ``hidden_size`` channels at the image-embedding resolution.

    Args:
        inputs: Input mask tensor.
        hidden_size: Output embedding dimension.
        mask_input_channels: Intermediate channel count after the second conv.
        layer_norm_eps: Epsilon for layer normalization.
        name: Name prefix for the layers.

    Returns:
        Dense embedding tensor.
    """
    inner_channels = mask_input_channels // 4
    x = layers.Conv2D(inner_channels, kernel_size=2, strides=2, name=f"{name}_conv1")(
        inputs
    )
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm1")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_1")(x)
    x = layers.Conv2D(
        mask_input_channels, kernel_size=2, strides=2, name=f"{name}_conv2"
    )(x)
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm2")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_2")(x)
    x = layers.Conv2D(hidden_size, kernel_size=1, name=f"{name}_conv3")(x)
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class SAM(keras.Model):
    """Segment Anything Model (SAM) for promptable image segmentation.

    SAM consists of three components:

    1. **Vision Encoder** – a ViT backbone with windowed attention and
       relative positional embeddings that produces image embeddings.
    2. **Prompt Encoder** – encodes sparse prompts (points, boxes) and dense
       prompts (masks) into embeddings.
    3. **Mask Decoder** – a lightweight two-way transformer that predicts
       segmentation masks and IoU scores from image and prompt embeddings.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_
          (Kirillov et al., 2023)

    Args:
        vision_hidden_size: Vision encoder hidden dimension.
        vision_num_hidden_layers: Number of vision encoder transformer layers.
        vision_num_attention_heads: Number of attention heads in vision encoder.
        vision_mlp_dim: MLP hidden dimension in vision encoder.
        vision_output_channels: Output channels of the vision neck.
        vision_patch_size: Patch size for patch embeddings.
        vision_image_size: Expected input image resolution.
        vision_window_size: Window size for windowed attention.
        vision_global_attn_indexes: Layer indices that use global attention.
        vision_layer_norm_eps: LayerNorm epsilon for vision encoder.
        vision_qkv_bias: Whether QKV projections have bias.
        vision_use_abs_pos: Whether to use absolute position embeddings.
        vision_use_rel_pos: Whether to use relative position embeddings.
        mask_decoder_hidden_size: Mask decoder hidden dimension.
        mask_decoder_num_hidden_layers: Number of two-way transformer layers.
        mask_decoder_num_attention_heads: Attention heads in mask decoder.
        mask_decoder_mlp_dim: MLP dim in mask decoder.
        mask_decoder_iou_head_depth: Depth of the IoU prediction MLP.
        mask_decoder_iou_head_hidden_dim: Hidden dim of the IoU prediction MLP.
        prompt_encoder_hidden_size: Prompt encoder hidden dimension.
        prompt_encoder_mask_input_channels: Intermediate channels in mask embedding CNN.
        prompt_encoder_num_point_embeddings: Number of point embedding types.
        num_multimask_outputs: Number of mask outputs (default 3).
        input_shape: Input image shape ``(H, W, C)``.
        input_tensor: Optional input tensor.
        name: Model name.
        **kwargs: Additional arguments.

    Example:
        ```python
        model = kmodels.models.sam.SAM_ViT_Huge(
            input_shape=(1024, 1024, 3),
            weights="sa1b",
        )
        ```
    """

    def __init__(
        self,
        vision_hidden_size=768,
        vision_num_hidden_layers=12,
        vision_num_attention_heads=12,
        vision_mlp_dim=3072,
        vision_output_channels=256,
        vision_patch_size=16,
        vision_image_size=1024,
        vision_window_size=14,
        vision_global_attn_indexes=(2, 5, 8, 11),
        vision_layer_norm_eps=1e-6,
        vision_qkv_bias=True,
        vision_use_abs_pos=True,
        vision_use_rel_pos=True,
        mask_decoder_hidden_size=256,
        mask_decoder_num_hidden_layers=2,
        mask_decoder_num_attention_heads=8,
        mask_decoder_mlp_dim=2048,
        mask_decoder_iou_head_depth=3,
        mask_decoder_iou_head_hidden_dim=256,
        prompt_encoder_hidden_size=256,
        prompt_encoder_mask_input_channels=16,
        prompt_encoder_num_point_embeddings=4,
        num_multimask_outputs=3,
        input_shape=None,
        input_tensor=None,
        name="SAM",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (vision_image_size, vision_image_size, 3)

        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=input_shape, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=input_shape, name="pixel_values")

        image_embedding_size = vision_image_size // vision_patch_size

        # Prompt inputs
        input_points = layers.Input(
            shape=(None, None, 2), name="input_points", dtype="float32"
        )
        input_labels = layers.Input(
            shape=(None, None), name="input_labels", dtype="int32"
        )
        # ─── Vision Encoder ───
        hidden_states = layers.Conv2D(
            vision_hidden_size,
            kernel_size=vision_patch_size,
            strides=vision_patch_size,
            padding="valid",
            use_bias=True,
            name="vision_encoder_patch_embed_projection",
        )(pixel_values)

        if vision_use_abs_pos:
            pos_embed_layer = SAMAbsolutePositionEmbedding(
                vision_hidden_size,
                image_embedding_size,
                name="vision_encoder_pos_embed",
            )
            hidden_states = pos_embed_layer(hidden_states)

        for i in range(vision_num_hidden_layers):
            win_size = vision_window_size if i not in vision_global_attn_indexes else 0
            hidden_states = SAMVisionLayer(
                vision_hidden_size,
                vision_num_attention_heads,
                vision_mlp_dim,
                qkv_bias=vision_qkv_bias,
                use_rel_pos=vision_use_rel_pos,
                window_size=win_size,
                image_size=image_embedding_size,
                layer_norm_eps=vision_layer_norm_eps,
                name=f"vision_encoder_layers_{i}",
            )(hidden_states)

        image_embeddings = sam_vision_neck(
            hidden_states,
            vision_hidden_size,
            vision_output_channels,
            name="vision_encoder_neck",
        )

        # ─── Shared Positional Embedding ───
        num_pos_feats = 128
        shared_image_embedding = SAMPositionalEmbedding(
            num_pos_feats=num_pos_feats,
            scale=vision_hidden_size // 2,
            name="shared_image_embedding",
        )

        # ─── Image-wide Positional Embeddings ───
        image_pe = SAMImagePositionalEmbeddings(
            image_embedding_size,
            shared_image_embedding,
            name="image_positional_embeddings",
        )(image_embeddings)

        # ─── Prompt Encoder ───
        prompt_results = SAMPromptEncoderLayer(
            hidden_size=prompt_encoder_hidden_size,
            image_embedding_size=image_embedding_size,
            image_size=vision_image_size,
            num_point_embeddings=prompt_encoder_num_point_embeddings,
            shared_embedding=shared_image_embedding,
            name="prompt_encoder",
        )([input_points, input_labels])

        sparse_embeddings = prompt_results["sparse_embeddings"]
        dense_embeddings = prompt_results["dense_embeddings"]

        # ─── Mask Decoder ───
        decoder_output = SAMMaskDecoderLayer(
            hidden_size=mask_decoder_hidden_size,
            num_hidden_layers=mask_decoder_num_hidden_layers,
            num_attention_heads=mask_decoder_num_attention_heads,
            mlp_dim=mask_decoder_mlp_dim,
            num_multimask_outputs=num_multimask_outputs,
            iou_head_depth=mask_decoder_iou_head_depth,
            iou_head_hidden_dim=mask_decoder_iou_head_hidden_dim,
            name="mask_decoder",
        )(
            [
                image_embeddings,
                image_pe,
                sparse_embeddings,
                dense_embeddings,
            ]
        )

        pred_masks = decoder_output["pred_masks"]
        iou_scores = decoder_output["iou_scores"]

        super().__init__(
            inputs={
                "pixel_values": pixel_values,
                "input_points": input_points,
                "input_labels": input_labels,
            },
            outputs={"pred_masks": pred_masks, "iou_scores": iou_scores},
            name=name,
            **kwargs,
        )

        self.vision_hidden_size = vision_hidden_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_output_channels = vision_output_channels
        self.vision_patch_size = vision_patch_size
        self.vision_image_size = vision_image_size
        self.vision_window_size = vision_window_size
        self.vision_global_attn_indexes = list(vision_global_attn_indexes)
        self.vision_layer_norm_eps = vision_layer_norm_eps
        self.vision_qkv_bias = vision_qkv_bias
        self.vision_use_abs_pos = vision_use_abs_pos
        self.vision_use_rel_pos = vision_use_rel_pos
        self.mask_decoder_hidden_size = mask_decoder_hidden_size
        self.mask_decoder_num_hidden_layers = mask_decoder_num_hidden_layers
        self.mask_decoder_num_attention_heads = mask_decoder_num_attention_heads
        self.mask_decoder_mlp_dim = mask_decoder_mlp_dim
        self.mask_decoder_iou_head_depth = mask_decoder_iou_head_depth
        self.mask_decoder_iou_head_hidden_dim = mask_decoder_iou_head_hidden_dim
        self.prompt_encoder_hidden_size = prompt_encoder_hidden_size
        self.prompt_encoder_mask_input_channels = prompt_encoder_mask_input_channels
        self.prompt_encoder_num_point_embeddings = prompt_encoder_num_point_embeddings
        self.num_multimask_outputs = num_multimask_outputs
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_hidden_size": self.vision_hidden_size,
                "vision_num_hidden_layers": self.vision_num_hidden_layers,
                "vision_num_attention_heads": self.vision_num_attention_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_output_channels": self.vision_output_channels,
                "vision_patch_size": self.vision_patch_size,
                "vision_image_size": self.vision_image_size,
                "vision_window_size": self.vision_window_size,
                "vision_global_attn_indexes": self.vision_global_attn_indexes,
                "vision_layer_norm_eps": self.vision_layer_norm_eps,
                "vision_qkv_bias": self.vision_qkv_bias,
                "vision_use_abs_pos": self.vision_use_abs_pos,
                "vision_use_rel_pos": self.vision_use_rel_pos,
                "mask_decoder_hidden_size": self.mask_decoder_hidden_size,
                "mask_decoder_num_hidden_layers": self.mask_decoder_num_hidden_layers,
                "mask_decoder_num_attention_heads": self.mask_decoder_num_attention_heads,
                "mask_decoder_mlp_dim": self.mask_decoder_mlp_dim,
                "mask_decoder_iou_head_depth": self.mask_decoder_iou_head_depth,
                "mask_decoder_iou_head_hidden_dim": self.mask_decoder_iou_head_hidden_dim,
                "prompt_encoder_hidden_size": self.prompt_encoder_hidden_size,
                "prompt_encoder_mask_input_channels": self.prompt_encoder_mask_input_channels,
                "prompt_encoder_num_point_embeddings": self.prompt_encoder_num_point_embeddings,
                "num_multimask_outputs": self.num_multimask_outputs,
                "input_shape": self._input_shape_val,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_sam_model(
    variant,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    """Creates a SAM model from the given variant configuration.

    Args:
        variant: Model variant name (e.g., ``"SAM_ViT_Huge"``).
        input_shape: Input image shape ``(H, W, C)``.
        input_tensor: Optional input tensor.
        weights: Pretrained weights identifier or file path.
        **kwargs: Additional arguments passed to the ``SAM`` constructor.

    Returns:
        Configured ``SAM`` model instance.
    """
    config = SAM_MODEL_CONFIG[variant]

    valid_model_weights = []
    if variant in SAM_WEIGHTS_CONFIG:
        valid_model_weights = list(SAM_WEIGHTS_CONFIG[variant].keys())

    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights for {variant} are "
            f"{', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )

    if input_shape is None:
        image_size = config["vision_image_size"]
        input_shape = (image_size, image_size, 3)
        print(f"Using default input shape {input_shape}.")

    model = SAM(
        vision_hidden_size=config["vision_hidden_size"],
        vision_num_hidden_layers=config["vision_num_hidden_layers"],
        vision_num_attention_heads=config["vision_num_attention_heads"],
        vision_mlp_dim=config["vision_mlp_dim"],
        vision_output_channels=config["vision_output_channels"],
        vision_patch_size=config["vision_patch_size"],
        vision_image_size=config["vision_image_size"],
        vision_window_size=config["vision_window_size"],
        vision_global_attn_indexes=config["vision_global_attn_indexes"],
        vision_layer_norm_eps=config["vision_layer_norm_eps"],
        vision_qkv_bias=config["vision_qkv_bias"],
        vision_use_abs_pos=config["vision_use_abs_pos"],
        vision_use_rel_pos=config["vision_use_rel_pos"],
        mask_decoder_hidden_size=config["mask_decoder_hidden_size"],
        mask_decoder_num_hidden_layers=config["mask_decoder_num_hidden_layers"],
        mask_decoder_num_attention_heads=config["mask_decoder_num_attention_heads"],
        mask_decoder_mlp_dim=config["mask_decoder_mlp_dim"],
        mask_decoder_iou_head_depth=config["mask_decoder_iou_head_depth"],
        mask_decoder_iou_head_hidden_dim=config["mask_decoder_iou_head_hidden_dim"],
        prompt_encoder_hidden_size=config["prompt_encoder_hidden_size"],
        prompt_encoder_mask_input_channels=config["prompt_encoder_mask_input_channels"],
        prompt_encoder_num_point_embeddings=config[
            "prompt_encoder_num_point_embeddings"
        ],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **kwargs,
    )

    if weights in valid_model_weights:
        print(f"Loading {weights} weights for {variant}.")
        load_weights_from_config(variant, weights, model, SAM_WEIGHTS_CONFIG)
    elif weights is not None and isinstance(weights, str):
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SAM_ViT_Base(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam_model(
        "SAM_ViT_Base",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SAM_ViT_Large(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam_model(
        "SAM_ViT_Large",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SAM_ViT_Huge(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam_model(
        "SAM_ViT_Huge",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )
