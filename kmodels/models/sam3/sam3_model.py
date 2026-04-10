import keras
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import SAM3_MODEL_CONFIG, SAM3_WEIGHTS_CONFIG
from .sam3_clip_tokenizer import SAM3_VOCAB_SIZE
from .sam3_layers import (
    CLIPCausalMask,
    CLIPPositionEmbedding,
    SAM3AddPositionEmbedding,
    SAM3BoxRPB,
    SAM3CLIPAttention,
    SAM3DecoderMLP,
    SAM3GeometryEncoder,
    SAM3LearnableEmbedding,
    SAM3MultiHeadAttention,
    SAM3ViTLayer,
)
from .sam3_utils import (
    box_cxcywh_to_xyxy,
    compute_sine_pos_encoding,
    inverse_sigmoid,
    sine_encode_boxes,
)

LAYER_NORM_EPS = 1e-6

CLIP_HIDDEN_SIZE = 1024
CLIP_NUM_LAYERS = 24
CLIP_NUM_HEADS = 16
CLIP_INTERMEDIATE_SIZE = 4096
CLIP_MAX_POSITION = 32


def compute_rotary_embeddings(
    hidden_size, num_attention_heads, end_x, end_y, rope_theta=10000.0, scale=1.0
):
    """Compute 2-D rotary position embeddings for ViT attention.

    Generates cos/sin frequencies for spatial positions in a 2-D grid,
    used by ``SAM3ViTRoPEAttention`` to encode spatial structure.

    Args:
        hidden_size (int): Total hidden dimension.
        num_attention_heads (int): Number of attention heads.
        end_x (int): Grid width.
        end_y (int): Grid height.
        rope_theta (float): Base frequency for RoPE.
            Defaults to ``10000.0``.
        scale (float): Coordinate scaling factor.
            Defaults to ``1.0``.

    Returns:
        Tuple of ``(cos, sin)``, each ``(end_x * end_y, head_dim)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    head_dim = hidden_size // num_attention_heads
    dim_range = ops.cast(ops.arange(0, head_dim, 4), "float32")[: head_dim // 4]
    freqs = 1.0 / (rope_theta ** (dim_range / head_dim))

    flat_idx = ops.cast(ops.arange(end_x * end_y), "float32")
    x_positions = (flat_idx % end_x) * scale
    y_positions = ops.floor(flat_idx / end_x) * scale

    freqs_x = ops.outer(x_positions, freqs)
    freqs_y = ops.outer(y_positions, freqs)

    inv_freq = ops.concatenate([freqs_x, freqs_y], axis=-1)
    inv_freq = ops.repeat(inv_freq, 2, axis=-1)

    cos = ops.cos(inv_freq)
    sin = ops.sin(inv_freq)
    return cos, sin


def sam3_vision_backbone(
    pixel_values,
    vit_hidden_size,
    vit_num_hidden_layers,
    vit_num_attention_heads,
    vit_intermediate_size,
    vit_image_size,
    vit_patch_size,
    vit_window_size,
    vit_global_attn_indexes,
    vit_rope_theta,
    vit_pretrain_image_size,
    data_format="channels_last",
):
    """ViT backbone: patch embed, position embed, and transformer layers.

    Builds the vision backbone as a functional graph. Patch embedding
    is followed by learnable position embeddings, layer normalization,
    and ``vit_num_hidden_layers`` transformer blocks with windowed or
    global RoPE attention. Internally processes in NHWC; permutes
    to/from NCHW when ``data_format`` is ``"channels_first"``.

    Args:
        pixel_values: Input image tensor.
        vit_hidden_size (int): Hidden dimension.
        vit_num_hidden_layers (int): Number of transformer layers.
        vit_num_attention_heads (int): Number of attention heads.
        vit_intermediate_size (int): MLP intermediate dimension.
        vit_image_size (int): Input image spatial size.
        vit_patch_size (int): Patch embedding kernel/stride size.
        vit_window_size (int): Window size for windowed attention.
        vit_global_attn_indexes (tuple): Layer indices using global attention.
        vit_rope_theta (float): RoPE base frequency.
        vit_pretrain_image_size (int): Image size used during pretraining.
        data_format (str): ``"channels_last"`` or ``"channels_first"``.

    Returns:
        Tuple of ``(backbone_spatial, grid_size)`` where
        ``backbone_spatial`` is ``(B, H, W, C)`` or ``(B, C, H, W)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    grid_size = vit_image_size // vit_patch_size
    pretrain_grid = vit_pretrain_image_size // vit_patch_size
    num_pretrain_patches = pretrain_grid * pretrain_grid

    patch_embed = layers.Conv2D(
        vit_hidden_size,
        kernel_size=vit_patch_size,
        strides=vit_patch_size,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_patch_embed",
    )(pixel_values)

    if data_format == "channels_first":
        patch_embed = layers.Permute((2, 3, 1), name="backbone_patch_to_nhwc")(
            patch_embed
        )

    hidden_states = layers.Reshape(
        (grid_size * grid_size, vit_hidden_size),
        name="backbone_flatten",
    )(patch_embed)

    hidden_states = SAM3AddPositionEmbedding(
        num_patches=num_pretrain_patches,
        hidden_size=vit_hidden_size,
        pretrain_grid=pretrain_grid,
        grid_size=grid_size,
        name="backbone_position_embedding",
    )(hidden_states)

    hidden_states = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name="backbone_layer_norm"
    )(hidden_states)

    hidden_states = layers.Reshape(
        (grid_size, grid_size, vit_hidden_size),
        name="backbone_to_spatial",
    )(hidden_states)

    window_cos, window_sin = compute_rotary_embeddings(
        vit_hidden_size,
        vit_num_attention_heads,
        end_x=vit_window_size,
        end_y=vit_window_size,
        rope_theta=vit_rope_theta,
        scale=1.0,
    )
    global_cos, global_sin = compute_rotary_embeddings(
        vit_hidden_size,
        vit_num_attention_heads,
        end_x=grid_size,
        end_y=grid_size,
        rope_theta=vit_rope_theta,
        scale=vit_window_size / grid_size,
    )

    for i in range(vit_num_hidden_layers):
        win = vit_window_size if i not in vit_global_attn_indexes else 0
        if win > 0:
            cos, sin = window_cos, window_sin
        else:
            cos, sin = global_cos, global_sin
        hidden_states = SAM3ViTLayer(
            hidden_size=vit_hidden_size,
            num_attention_heads=vit_num_attention_heads,
            intermediate_size=vit_intermediate_size,
            window_size=win,
            image_size=grid_size,
            layer_norm_eps=LAYER_NORM_EPS,
            name=f"backbone_layers_{i}",
        )(hidden_states, cos, sin)

    backbone_out_flat = layers.Reshape(
        (grid_size * grid_size, vit_hidden_size),
        name="backbone_out_flatten",
    )(hidden_states)
    backbone_out = layers.Reshape(
        (grid_size, grid_size, vit_hidden_size),
        name="backbone_out_spatial",
    )(backbone_out_flat)

    if data_format == "channels_first":
        backbone_out = layers.Permute((3, 1, 2), name="backbone_to_nchw")(backbone_out)
    return backbone_out, grid_size


def sam3_fpn_neck(
    backbone_spatial,
    vit_hidden_size,
    fpn_hidden_size,
    fpn_scale_factors,
    data_format="channels_last",
):
    """Feature Pyramid Network neck producing multi-scale features.

    Generates a list of feature maps at different spatial scales from the
    single-scale backbone output. Upscaling uses transposed convolutions;
    downscaling uses max pooling. Each level has two 3x3 projection convs.

    Args:
        backbone_spatial: Backbone output tensor.
        vit_hidden_size (int): Backbone channel dimension.
        fpn_hidden_size (int): FPN output channel dimension.
        fpn_scale_factors (tuple): Scale factors for each FPN level.
        data_format (str): ``"channels_last"`` or ``"channels_first"``.

    Returns:
        List of spatial feature tensors, one per scale level.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    fpn_hidden_states = []

    for level_idx, scale_factor in enumerate(fpn_scale_factors):
        if scale_factor == 4.0:
            x = layers.Conv2DTranspose(
                vit_hidden_size // 2,
                kernel_size=2,
                strides=2,
                data_format=data_format,
                name=f"fpn_level_{level_idx}_deconv1",
            )(backbone_spatial)
            x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu1")(x)
            x = layers.Conv2DTranspose(
                vit_hidden_size // 4,
                kernel_size=2,
                strides=2,
                data_format=data_format,
                name=f"fpn_level_{level_idx}_deconv2",
            )(x)
            x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu2")(x)
        elif scale_factor == 2.0:
            x = layers.Conv2DTranspose(
                vit_hidden_size // 2,
                kernel_size=2,
                strides=2,
                data_format=data_format,
                name=f"fpn_level_{level_idx}_deconv1",
            )(backbone_spatial)
            x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu1")(x)
        elif scale_factor == 1.0:
            x = backbone_spatial
        else:
            x = layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                data_format=data_format,
                name=f"fpn_level_{level_idx}_pool",
            )(backbone_spatial)

        x = layers.Conv2D(
            fpn_hidden_size,
            kernel_size=1,
            data_format=data_format,
            name=f"fpn_level_{level_idx}_proj1",
        )(x)
        x = layers.Conv2D(
            fpn_hidden_size,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            name=f"fpn_level_{level_idx}_proj2",
        )(x)
        fpn_hidden_states.append(x)

    return fpn_hidden_states


def sam3_detr_encoder_layer(
    vision_feats,
    text_feats,
    vision_pos,
    text_mask,
    hidden_size,
    num_attention_heads,
    intermediate_size,
    dropout,
    name,
):
    """Single DETR encoder layer: self-attention, text cross-attention, and MLP.

    Pre-norm transformer block that refines vision features through
    self-attention with positional encoding, cross-attention to text
    features, and a feed-forward network.

    Args:
        vision_feats: Vision feature tensor ``(B, H*W, D)``.
        text_feats: Text feature tensor ``(B, seq, D)``.
        vision_pos: Vision position encoding ``(1, H*W, D)``.
        text_mask: Text attention mask ``(B, seq)``.
        hidden_size (int): Hidden dimension.
        num_attention_heads (int): Number of attention heads.
        intermediate_size (int): MLP intermediate dimension.
        dropout (float): Dropout rate.
        name (str): Layer name prefix.

    Returns:
        Updated vision features ``(B, H*W, D)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    self_attn = SAM3MultiHeadAttention(
        hidden_size, num_attention_heads, dropout, name=f"{name}_self_attn"
    )
    cross_attn = SAM3MultiHeadAttention(
        hidden_size, num_attention_heads, dropout, name=f"{name}_cross_attn"
    )
    layer_norm1 = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name=f"{name}_layer_norm1"
    )
    layer_norm2 = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name=f"{name}_layer_norm2"
    )
    layer_norm3 = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name=f"{name}_layer_norm3"
    )
    fc1 = layers.Dense(intermediate_size, name=f"{name}_fc1")
    fc2 = layers.Dense(hidden_size, name=f"{name}_fc2")

    residual = vision_feats
    x = layer_norm1(vision_feats)
    q = k = x + vision_pos
    x = self_attn(q, k, x)
    x = layers.Dropout(dropout, name=f"{name}_dropout1")(x)
    vision_feats = layers.Add(name=f"{name}_add1")([x, residual])

    residual = vision_feats
    x = layer_norm2(vision_feats)
    x = cross_attn(x, text_feats, text_feats, attention_mask=text_mask)
    x = layers.Dropout(dropout, name=f"{name}_dropout2")(x)
    vision_feats = layers.Add(name=f"{name}_add2")([x, residual])

    residual = vision_feats
    x = layer_norm3(vision_feats)
    x = fc1(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout3")(x)
    x = fc2(x)
    vision_feats = layers.Add(name=f"{name}_add3")([x, residual])

    return vision_feats


def sam3_detr_encoder(
    fpn_hidden_states,
    text_projected,
    text_attn_mask,
    grid_size,
    fpn_hidden_size,
    detr_encoder_hidden_size,
    detr_encoder_num_layers,
    detr_encoder_num_attention_heads,
    detr_encoder_intermediate_size,
    detr_encoder_dropout,
    data_format="channels_last",
):
    """DETR encoder: stacked layers of vision self-attention and text cross-attention.

    Flattens the 1x FPN feature map to a sequence, adds sine position
    encoding, and passes through ``detr_encoder_num_layers`` encoder
    layers. Projects text features to the encoder hidden dimension.

    Args:
        fpn_hidden_states (list): FPN feature maps at each scale.
        text_projected: Text features ``(B, seq, text_dim)``.
        text_attn_mask: Text attention mask ``(B, seq)``.
        grid_size (int): Spatial size of the 1x FPN level.
        fpn_hidden_size (int): FPN channel dimension.
        detr_encoder_hidden_size (int): Encoder hidden dimension.
        detr_encoder_num_layers (int): Number of encoder layers.
        detr_encoder_num_attention_heads (int): Number of attention heads.
        detr_encoder_intermediate_size (int): MLP intermediate dimension.
        detr_encoder_dropout (float): Dropout rate.
        data_format (str): ``"channels_last"`` or ``"channels_first"``.

    Returns:
        Tuple of ``(encoder_output, encoder_pos_flat, enc_h)`` where
        ``encoder_output`` is ``(B, H*W, D)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    encoder_vision = fpn_hidden_states[-2]
    enc_h = grid_size

    if data_format == "channels_first":
        encoder_vision_flat = layers.Permute((2, 3, 1), name="encoder_vision_to_nhwc")(
            encoder_vision
        )
    else:
        encoder_vision_flat = encoder_vision
    encoder_vision_flat = layers.Reshape(
        (enc_h * enc_h, fpn_hidden_size), name="encoder_vision_flatten"
    )(encoder_vision_flat)

    enc_pos = compute_sine_pos_encoding(
        enc_h,
        enc_h,
        fpn_hidden_size // 2,
        normalize=True,
    )
    encoder_pos_flat = ops.reshape(enc_pos, (1, enc_h * enc_h, fpn_hidden_size))

    encoder_output = encoder_vision_flat
    for i in range(detr_encoder_num_layers):
        encoder_output = sam3_detr_encoder_layer(
            encoder_output,
            text_projected,
            encoder_pos_flat,
            text_attn_mask,
            hidden_size=detr_encoder_hidden_size,
            num_attention_heads=detr_encoder_num_attention_heads,
            intermediate_size=detr_encoder_intermediate_size,
            dropout=detr_encoder_dropout,
            name=f"detr_encoder_layers_{i}",
        )

    return encoder_output, encoder_pos_flat, enc_h


def sam3_detr_decoder_layer(
    hidden_states,
    query_pos,
    text_feats,
    vision_feats,
    vision_pos,
    text_mask,
    vision_mask,
    hidden_size,
    num_attention_heads,
    intermediate_size,
    dropout,
    name,
):
    """Single DETR decoder layer: self-attention, text and vision cross-attention, and MLP.

    Pre-norm transformer block with four sub-layers: query self-attention,
    text cross-attention, vision cross-attention with box-conditioned RPB,
    and a feed-forward network.

    Args:
        hidden_states: Query tensor ``(B, Q, D)``.
        query_pos: Query position encoding ``(B, Q, D)``.
        text_feats: Projected text features ``(B, seq, D)``.
        vision_feats: Encoded vision features ``(B, H*W, D)``.
        vision_pos: Vision position encoding ``(1, H*W, D)``.
        text_mask: Text attention mask ``(B, seq)``.
        vision_mask: Box RPB attention bias ``(B, heads, Q+1, H*W)``
            or ``None``.
        hidden_size (int): Hidden dimension.
        num_attention_heads (int): Number of attention heads.
        intermediate_size (int): MLP intermediate dimension.
        dropout (float): Dropout rate.
        name (str): Layer name prefix.

    Returns:
        Updated query tensor ``(B, Q, D)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    self_attn = SAM3MultiHeadAttention(
        hidden_size, num_attention_heads, dropout, name=f"{name}_self_attn"
    )
    text_cross_attn = SAM3MultiHeadAttention(
        hidden_size, num_attention_heads, dropout, name=f"{name}_text_cross_attn"
    )
    vision_cross_attn = SAM3MultiHeadAttention(
        hidden_size, num_attention_heads, dropout, name=f"{name}_vision_cross_attn"
    )
    layer_norm1 = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name=f"{name}_layer_norm1"
    )
    layer_norm2 = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name=f"{name}_layer_norm2"
    )
    layer_norm3 = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name=f"{name}_layer_norm3"
    )
    layer_norm4 = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name=f"{name}_layer_norm4"
    )
    fc1 = layers.Dense(intermediate_size, name=f"{name}_fc1")
    fc2 = layers.Dense(hidden_size, name=f"{name}_fc2")

    q = k = hidden_states + query_pos
    x = self_attn(q, k, hidden_states)
    x = layers.Dropout(dropout, name=f"{name}_dropout1")(x)
    hidden_states = layer_norm1(layers.Add(name=f"{name}_add1")([hidden_states, x]))

    x = text_cross_attn(
        hidden_states + query_pos,
        text_feats,
        text_feats,
        attention_mask=text_mask,
    )
    x = layers.Dropout(dropout, name=f"{name}_dropout2")(x)
    hidden_states = layer_norm2(layers.Add(name=f"{name}_add2")([hidden_states, x]))

    x = vision_cross_attn(
        hidden_states + query_pos,
        vision_feats + vision_pos,
        vision_feats,
        attention_mask=vision_mask,
    )
    x = layers.Dropout(dropout, name=f"{name}_dropout3")(x)
    hidden_states = layer_norm3(layers.Add(name=f"{name}_add3")([hidden_states, x]))

    x = fc1(hidden_states)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout4")(x)
    x = fc2(x)
    hidden_states = layer_norm4(layers.Add(name=f"{name}_add4")([hidden_states, x]))

    return hidden_states


def sam3_dot_product_scoring(
    decoder_hidden_states,
    text_features,
    text_mask,
    hidden_size,
    intermediate_size=2048,
    name="dot_product_scoring",
):
    """Dot-product scoring for text-query classification.

    Processes text features through an MLP, projects both text and query
    embeddings to a shared space, and computes classification logits via
    dot product. Used for open-vocabulary detection scoring.

    Args:
        decoder_hidden_states: Decoder output ``(B, Q, D)``.
        text_features: Text features ``(B, seq, text_dim)``.
        text_mask: Text attention mask ``(B, seq)``.
        hidden_size (int): Projection dimension.
        intermediate_size (int): Text MLP intermediate dimension.
            Defaults to ``2048``.
        name (str): Layer name prefix.

    Returns:
        Classification logits ``(B, Q)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    text_mlp_fc1 = layers.Dense(intermediate_size, name=f"{name}_text_mlp_fc1")
    text_mlp_fc2 = layers.Dense(hidden_size, name=f"{name}_text_mlp_fc2")
    text_mlp_out_norm = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS, name=f"{name}_text_mlp_out_norm"
    )
    text_proj = layers.Dense(hidden_size, name=f"{name}_text_proj")
    query_proj = layers.Dense(hidden_size, name=f"{name}_query_proj")

    x = text_mlp_fc1(text_features)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = text_mlp_fc2(x)
    text_feats = text_mlp_out_norm(
        layers.Add(name=f"{name}_residual")([text_features, x])
    )

    mask_expanded = ops.expand_dims(ops.cast(text_mask, "float32"), axis=-1)
    text_pooled = ops.sum(text_feats * mask_expanded, axis=1) / (
        ops.sum(mask_expanded, axis=1) + 1e-8
    )

    t_proj = text_proj(text_pooled)
    q_proj = query_proj(decoder_hidden_states)

    scale = hidden_size**-0.5
    logits = ops.matmul(q_proj, ops.expand_dims(t_proj, axis=-1))
    logits = ops.squeeze(logits, axis=-1) * scale
    logits = ops.clip(logits, -12.0, 12.0)
    return logits


def sam3_detr_decoder(
    encoder_output,
    encoder_pos_flat,
    text_projected,
    text_attn_mask,
    text_attention_mask,
    enc_h,
    detr_decoder_hidden_size,
    detr_decoder_num_layers,
    detr_decoder_num_queries,
    detr_decoder_num_attention_heads,
    detr_decoder_intermediate_size,
    detr_decoder_dropout,
):
    """DETR decoder with iterative box refinement.

    Stacks ``detr_decoder_num_layers`` decoder layers with learnable
    query embeddings and iterative reference point refinement. Each
    layer performs self-attention, text cross-attention, and vision
    cross-attention with box-conditioned relative position bias.
    Produces object queries, refined bounding boxes, classification
    logits, and per-layer presence scores.

    Args:
        encoder_output: Encoded vision features ``(B, H*W, D)``.
        encoder_pos_flat: Vision position encoding ``(1, H*W, D)``.
        text_projected: Projected text features ``(B, seq, D)``.
        text_attn_mask: Text padding mask ``(B, seq)``.
        text_attention_mask: Original text attention mask.
        enc_h (int): Encoder spatial height.
        detr_decoder_hidden_size (int): Decoder hidden dimension.
        detr_decoder_num_layers (int): Number of decoder layers.
        detr_decoder_num_queries (int): Number of object queries.
        detr_decoder_num_attention_heads (int): Number of attention heads.
        detr_decoder_intermediate_size (int): MLP intermediate dimension.
        detr_decoder_dropout (float): Dropout rate.

    Returns:
        Tuple of ``(decoder_hidden, pred_boxes, pred_logits,
        presence_logits)`` where ``pred_boxes`` is ``(B, Q, 4)``
        in cxcywh format.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    query_embed = SAM3LearnableEmbedding(
        detr_decoder_num_queries,
        detr_decoder_hidden_size,
        name="detr_decoder_query_embed",
    )(encoder_output)

    reference_points = SAM3LearnableEmbedding(
        detr_decoder_num_queries,
        4,
        apply_sigmoid=True,
        name="detr_decoder_reference_points",
    )(encoder_output)

    presence_token = SAM3LearnableEmbedding(
        1,
        detr_decoder_hidden_size,
        name="detr_decoder_presence_token",
    )(encoder_output)

    box_head = SAM3DecoderMLP(
        detr_decoder_hidden_size,
        detr_decoder_hidden_size,
        4,
        num_layers=3,
        name="detr_decoder_box_head",
    )
    presence_head = SAM3DecoderMLP(
        detr_decoder_hidden_size,
        detr_decoder_hidden_size,
        1,
        num_layers=3,
        name="detr_decoder_presence_head",
    )
    ref_point_head = SAM3DecoderMLP(
        2 * detr_decoder_hidden_size,
        detr_decoder_hidden_size,
        detr_decoder_hidden_size,
        num_layers=2,
        name="detr_decoder_ref_point_head",
    )

    output_layer_norm = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS,
        name="detr_decoder_output_layer_norm",
    )
    presence_layer_norm = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS,
        name="detr_decoder_presence_layer_norm",
    )
    box_rpb = SAM3BoxRPB(
        hidden_size=detr_decoder_hidden_size,
        num_attention_heads=detr_decoder_num_attention_heads,
        spatial_h=enc_h,
        spatial_w=enc_h,
        name="detr_decoder_box_rpb",
    )

    hidden_states = layers.Concatenate(
        axis=1,
        name="decoder_concat_presence",
    )([presence_token, query_embed])

    all_input_boxes = [reference_points]
    all_presence_logits = []
    all_intermediate = []

    for i in range(detr_decoder_num_layers):
        ref_encoded = sine_encode_boxes(
            reference_points,
            num_pos_feats=detr_decoder_hidden_size // 2,
        )
        query_pos_raw = ref_point_head(ref_encoded)
        zero_pad = ops.zeros_like(query_pos_raw[:, :1, :])
        query_pos = ops.concatenate([zero_pad, query_pos_raw], axis=1)

        vision_cross_attn_mask = box_rpb(reference_points)

        hidden_states = sam3_detr_decoder_layer(
            hidden_states,
            query_pos,
            text_projected,
            encoder_output,
            encoder_pos_flat,
            text_mask=text_attn_mask,
            vision_mask=vision_cross_attn_mask,
            hidden_size=detr_decoder_hidden_size,
            num_attention_heads=detr_decoder_num_attention_heads,
            intermediate_size=detr_decoder_intermediate_size,
            dropout=detr_decoder_dropout,
            name=f"detr_decoder_layers_{i}",
        )

        query_hidden = hidden_states[:, 1:, :]
        presence_hidden = hidden_states[:, :1, :]

        box_delta = box_head(output_layer_norm(query_hidden))
        new_ref = ops.sigmoid(
            inverse_sigmoid(ops.stop_gradient(reference_points)) + box_delta
        )
        reference_points = ops.stop_gradient(new_ref)
        all_input_boxes.append(new_ref)

        presence_logit = presence_head(presence_layer_norm(presence_hidden))
        presence_logit = ops.squeeze(presence_logit, axis=-1)
        presence_logit = ops.clip(presence_logit, -10.0, 10.0)
        all_presence_logits.append(presence_logit)

        all_intermediate.append(output_layer_norm(query_hidden))

    decoder_hidden = all_intermediate[-1]
    last_ref_boxes = all_input_boxes[-2]
    final_box_offsets = box_head(decoder_hidden)
    pred_boxes_cxcywh = ops.sigmoid(inverse_sigmoid(last_ref_boxes) + final_box_offsets)
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes_cxcywh)

    pred_logits = sam3_dot_product_scoring(
        decoder_hidden,
        text_projected,
        text_attention_mask,
        hidden_size=detr_decoder_hidden_size,
    )

    presence_logits_stacked = ops.concatenate(
        [ops.squeeze(p, axis=1) for p in all_presence_logits], axis=-1
    )

    return decoder_hidden, pred_boxes, pred_logits, presence_logits_stacked


def sam3_mask_embedder(x, hidden_size, name_prefix="mask_embedder"):
    """Three-layer MLP that projects decoder queries to mask embeddings.

    Args:
        x: Input tensor ``(B, Q, D)``.
        hidden_size (int): Hidden and output dimension.
        name_prefix (str): Layer name prefix.

    Returns:
        Mask embedding tensor ``(B, Q, D)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    x = layers.Dense(hidden_size, name=f"{name_prefix}_linear1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)
    x = layers.Dense(hidden_size, name=f"{name_prefix}_linear2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)
    x = layers.Dense(hidden_size, name=f"{name_prefix}_linear3")(x)
    return x


def sam3_mask_decoder(
    encoder_output,
    decoder_hidden,
    text_projected,
    text_attn_mask,
    fpn_hidden_states,
    enc_h,
    fpn_hidden_size,
    mask_decoder_hidden_size,
    mask_decoder_num_attention_heads,
    data_format="channels_last",
):
    """Mask decoder: pixel decoder with skip connections and mask prediction.

    Applies prompt-conditioned cross-attention to encoder features,
    upsamples through a pixel decoder with FPN skip connections, and
    produces per-query instance masks via einsum and a semantic
    segmentation map via 1x1 convolution.

    Args:
        encoder_output: Encoded vision features ``(B, H*W, D)``.
        decoder_hidden: Decoder query output ``(B, Q, D)``.
        text_projected: Projected text features ``(B, seq, D)``.
        text_attn_mask: Text attention mask ``(B, seq)``.
        fpn_hidden_states (list): FPN feature maps for skip connections.
        enc_h (int): Encoder spatial height.
        fpn_hidden_size (int): FPN channel dimension.
        mask_decoder_hidden_size (int): Pixel decoder hidden dimension.
        mask_decoder_num_attention_heads (int): Attention heads for
            prompt cross-attention.
        data_format (str): ``"channels_last"`` or ``"channels_first"``.

    Returns:
        Tuple of ``(pred_masks, semantic_seg)`` where ``pred_masks``
        is ``(B, Q, H, W)`` in NCHW format.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    prompt_cross_attn_norm = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS,
        name="mask_decoder_prompt_cross_attn_norm",
    )
    prompt_cross_attn = SAM3MultiHeadAttention(
        hidden_size=mask_decoder_hidden_size,
        num_attention_heads=mask_decoder_num_attention_heads,
        name="mask_decoder_prompt_cross_attn",
    )
    encoder_for_mask = prompt_cross_attn_norm(encoder_output)
    encoder_for_mask = prompt_cross_attn(
        encoder_for_mask,
        text_projected,
        text_projected,
        attention_mask=text_attn_mask,
    )
    encoder_for_mask = encoder_output + encoder_for_mask

    pixel_feat = layers.Reshape(
        (enc_h, enc_h, fpn_hidden_size),
        name="pixel_decoder_reshape_encoder",
    )(encoder_for_mask)
    if data_format == "channels_first":
        pixel_feat = layers.Permute((3, 1, 2), name="pixel_decoder_to_nchw")(pixel_feat)

    num_up = len(fpn_hidden_states) - 2
    for stage_idx in range(num_up):
        pixel_feat = layers.UpSampling2D(
            size=2,
            interpolation="nearest",
            data_format=data_format,
            name=f"pixel_decoder_stage_{stage_idx}_upsample",
        )(pixel_feat)

        skip_idx = len(fpn_hidden_states) - 3 - stage_idx
        if skip_idx >= 0:
            pixel_feat = layers.Add(
                name=f"pixel_decoder_stage_{stage_idx}_add_skip",
            )([pixel_feat, fpn_hidden_states[skip_idx]])

        pixel_feat = layers.Conv2D(
            mask_decoder_hidden_size,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            name=f"pixel_decoder_stage_{stage_idx}_conv",
        )(pixel_feat)
        pixel_feat = layers.GroupNormalization(
            groups=8,
            axis=1 if data_format == "channels_first" else -1,
            name=f"pixel_decoder_stage_{stage_idx}_gn",
        )(pixel_feat)
        pixel_feat = layers.ReLU(
            name=f"pixel_decoder_stage_{stage_idx}_relu",
        )(pixel_feat)

    instance_embed = layers.Conv2D(
        mask_decoder_hidden_size,
        kernel_size=1,
        data_format=data_format,
        name="mask_decoder_instance_proj",
    )(pixel_feat)

    semantic_seg = layers.Conv2D(
        1,
        kernel_size=1,
        data_format=data_format,
        name="mask_decoder_semantic_proj",
    )(pixel_feat)

    mask_embeddings = sam3_mask_embedder(decoder_hidden, mask_decoder_hidden_size)

    if data_format == "channels_first":
        instance_nhwc = ops.transpose(instance_embed, (0, 2, 3, 1))
    else:
        instance_nhwc = instance_embed
    pred_masks = ops.einsum("bqc,bhwc->bqhw", mask_embeddings, instance_nhwc)

    return pred_masks, semantic_seg


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3Main(keras.Model):
    """SAM3: unified open-vocabulary detector, segmenter, and promptable model.

    Builds the complete detection pipeline as a functional graph:
    ViT backbone, FPN neck, DETR encoder/decoder with iterative box
    refinement, dot-product scoring, and mask decoder. Supports both
    ``"channels_last"`` and ``"channels_first"`` via
    ``keras.config.image_data_format()``.

    Args:
        vit_hidden_size (int): ViT hidden dimension. Defaults to ``1024``.
        vit_intermediate_size (int): ViT MLP dimension. Defaults to ``4736``.
        vit_num_hidden_layers (int): Number of ViT layers. Defaults to ``32``.
        vit_num_attention_heads (int): ViT attention heads. Defaults to ``16``.
        vit_image_size (int): Input image size. Defaults to ``1008``.
        vit_patch_size (int): Patch size. Defaults to ``14``.
        vit_window_size (int): Windowed attention size. Defaults to ``24``.
        vit_global_attn_indexes (tuple): Global attention layer indices.
        vit_rope_theta (float): RoPE base frequency. Defaults to ``10000.0``.
        vit_pretrain_image_size (int): Pretrained image size. Defaults to ``336``.
        fpn_hidden_size (int): FPN channel dimension. Defaults to ``256``.
        fpn_scale_factors (tuple): FPN scale factors.
        detr_encoder_hidden_size (int): Encoder dimension. Defaults to ``256``.
        detr_encoder_num_layers (int): Encoder layers. Defaults to ``6``.
        detr_encoder_num_attention_heads (int): Encoder heads. Defaults to ``8``.
        detr_encoder_intermediate_size (int): Encoder MLP dimension.
        detr_encoder_dropout (float): Encoder dropout. Defaults to ``0.1``.
        detr_decoder_hidden_size (int): Decoder dimension. Defaults to ``256``.
        detr_decoder_num_layers (int): Decoder layers. Defaults to ``6``.
        detr_decoder_num_queries (int): Object queries. Defaults to ``200``.
        detr_decoder_num_attention_heads (int): Decoder heads. Defaults to ``8``.
        detr_decoder_intermediate_size (int): Decoder MLP dimension.
        detr_decoder_dropout (float): Decoder dropout. Defaults to ``0.1``.
        mask_decoder_hidden_size (int): Mask decoder dimension.
        mask_decoder_num_upsampling_stages (int): Pixel decoder stages.
        mask_decoder_num_attention_heads (int): Mask decoder heads.
        text_hidden_size (int): Text encoder dimension. Defaults to ``1024``.
        text_projection_dim (int): Text projection dimension.
        input_shape (tuple or None): Model input shape.
        input_tensor: Optional input tensor.
        name (str): Model name. Defaults to ``"SAM3"``.
        **kwargs: Additional keyword arguments.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(
        self,
        vit_hidden_size=1024,
        vit_intermediate_size=4736,
        vit_num_hidden_layers=32,
        vit_num_attention_heads=16,
        vit_image_size=1008,
        vit_patch_size=14,
        vit_window_size=24,
        vit_global_attn_indexes=(7, 15, 23, 31),
        vit_rope_theta=10000.0,
        vit_pretrain_image_size=336,
        fpn_hidden_size=256,
        fpn_scale_factors=(4.0, 2.0, 1.0, 0.5),
        detr_encoder_hidden_size=256,
        detr_encoder_num_layers=6,
        detr_encoder_num_attention_heads=8,
        detr_encoder_intermediate_size=2048,
        detr_encoder_dropout=0.1,
        detr_decoder_hidden_size=256,
        detr_decoder_num_layers=6,
        detr_decoder_num_queries=200,
        detr_decoder_num_attention_heads=8,
        detr_decoder_intermediate_size=2048,
        detr_decoder_dropout=0.1,
        mask_decoder_hidden_size=256,
        mask_decoder_num_upsampling_stages=3,
        mask_decoder_num_attention_heads=8,
        text_hidden_size=1024,
        text_projection_dim=512,
        input_shape=None,
        input_tensor=None,
        name="SAM3Main",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()

        if input_shape is None:
            if data_format == "channels_first":
                input_shape = (3, vit_image_size, vit_image_size)
            else:
                input_shape = (vit_image_size, vit_image_size, 3)

        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=input_shape, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=input_shape, name="pixel_values")

        text_features_input = layers.Input(
            shape=(None, text_hidden_size), name="text_features", dtype="float32"
        )
        text_attention_mask = layers.Input(
            shape=(None,), name="text_attention_mask", dtype="float32"
        )

        grid_size = vit_image_size // vit_patch_size
        backbone_spatial, grid_size = sam3_vision_backbone(
            pixel_values,
            vit_hidden_size=vit_hidden_size,
            vit_num_hidden_layers=vit_num_hidden_layers,
            vit_num_attention_heads=vit_num_attention_heads,
            vit_intermediate_size=vit_intermediate_size,
            vit_image_size=vit_image_size,
            vit_patch_size=vit_patch_size,
            vit_window_size=vit_window_size,
            vit_global_attn_indexes=vit_global_attn_indexes,
            vit_rope_theta=vit_rope_theta,
            vit_pretrain_image_size=vit_pretrain_image_size,
            data_format=data_format,
        )

        fpn_hidden_states = sam3_fpn_neck(
            backbone_spatial,
            vit_hidden_size,
            fpn_hidden_size,
            fpn_scale_factors,
            data_format=data_format,
        )

        text_projected = layers.Dense(detr_encoder_hidden_size, name="text_projection")(
            text_features_input
        )

        text_attn_mask = layers.Lambda(
            lambda m: ops.expand_dims(
                ops.expand_dims((1.0 - m) * (-1e9), axis=1), axis=1
            ),
            name="text_attn_mask",
        )(text_attention_mask)

        encoder_output, encoder_pos_flat, enc_h = sam3_detr_encoder(
            fpn_hidden_states,
            text_projected,
            text_attn_mask,
            grid_size=grid_size,
            fpn_hidden_size=fpn_hidden_size,
            detr_encoder_hidden_size=detr_encoder_hidden_size,
            detr_encoder_num_layers=detr_encoder_num_layers,
            detr_encoder_num_attention_heads=detr_encoder_num_attention_heads,
            detr_encoder_intermediate_size=detr_encoder_intermediate_size,
            detr_encoder_dropout=detr_encoder_dropout,
            data_format=data_format,
        )

        decoder_hidden, pred_boxes, pred_logits, presence_logits = sam3_detr_decoder(
            encoder_output,
            encoder_pos_flat,
            text_projected,
            text_attn_mask,
            text_attention_mask,
            enc_h,
            detr_decoder_hidden_size=detr_decoder_hidden_size,
            detr_decoder_num_layers=detr_decoder_num_layers,
            detr_decoder_num_queries=detr_decoder_num_queries,
            detr_decoder_num_attention_heads=detr_decoder_num_attention_heads,
            detr_decoder_intermediate_size=detr_decoder_intermediate_size,
            detr_decoder_dropout=detr_decoder_dropout,
        )

        pred_masks, semantic_seg = sam3_mask_decoder(
            encoder_output,
            decoder_hidden,
            text_projected,
            text_attn_mask,
            fpn_hidden_states,
            enc_h,
            fpn_hidden_size=fpn_hidden_size,
            mask_decoder_hidden_size=mask_decoder_hidden_size,
            mask_decoder_num_attention_heads=mask_decoder_num_attention_heads,
            data_format=data_format,
        )

        fpn_05x = fpn_hidden_states[-1]

        outputs = {
            "pred_masks": pred_masks,
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": semantic_seg,
            "fpn_05x": fpn_05x,
        }

        super().__init__(
            inputs={
                "pixel_values": pixel_values,
                "text_features": text_features_input,
                "text_attention_mask": text_attention_mask,
            },
            outputs=outputs,
            name=name,
            **kwargs,
        )

        self.text_encoder = build_text_encoder()

        self.geometry_encoder = SAM3GeometryEncoder(
            hidden_size=detr_encoder_hidden_size,
            name="geometry_encoder",
        )
        self.geometry_encoder.build((None, None, 4))

        self.vit_hidden_size = vit_hidden_size
        self.vit_intermediate_size = vit_intermediate_size
        self.vit_num_hidden_layers = vit_num_hidden_layers
        self.vit_num_attention_heads = vit_num_attention_heads
        self.vit_image_size = vit_image_size
        self.vit_patch_size = vit_patch_size
        self.vit_window_size = vit_window_size
        self.vit_global_attn_indexes = list(vit_global_attn_indexes)
        self.vit_rope_theta = vit_rope_theta
        self.vit_pretrain_image_size = vit_pretrain_image_size
        self.fpn_hidden_size = fpn_hidden_size
        self.fpn_scale_factors = list(fpn_scale_factors)
        self.detr_encoder_hidden_size = detr_encoder_hidden_size
        self.detr_encoder_num_layers = detr_encoder_num_layers
        self.detr_encoder_num_attention_heads = detr_encoder_num_attention_heads
        self.detr_encoder_intermediate_size = detr_encoder_intermediate_size
        self.detr_encoder_dropout = detr_encoder_dropout
        self.detr_decoder_hidden_size = detr_decoder_hidden_size
        self.detr_decoder_num_layers = detr_decoder_num_layers
        self.detr_decoder_num_queries = detr_decoder_num_queries
        self.detr_decoder_num_attention_heads = detr_decoder_num_attention_heads
        self.detr_decoder_intermediate_size = detr_decoder_intermediate_size
        self.detr_decoder_dropout = detr_decoder_dropout
        self.mask_decoder_hidden_size = mask_decoder_hidden_size
        self.mask_decoder_num_upsampling_stages = mask_decoder_num_upsampling_stages
        self.mask_decoder_num_attention_heads = mask_decoder_num_attention_heads
        self.text_hidden_size = text_hidden_size
        self.text_projection_dim = text_projection_dim
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor
        self._data_format = data_format

    def build_decoder_model(self):
        """Build a decoder-only sub-model from pre-computed FPN features.

        Returns:
            ``keras.Model`` with inputs ``{fpn_0..3, text_projected,
            text_attention_mask}`` and detection outputs.
        """
        cfg = self.get_config()
        fpn_hidden_size = cfg["fpn_hidden_size"]
        grid_size = cfg["vit_image_size"] // cfg["vit_patch_size"]

        fpn_0_in = layers.Input(shape=(fpn_hidden_size, None, None), name="fpn_0")
        fpn_1_in = layers.Input(shape=(fpn_hidden_size, None, None), name="fpn_1")
        fpn_2_in = layers.Input(shape=(fpn_hidden_size, None, None), name="fpn_2")
        fpn_3_in = layers.Input(shape=(fpn_hidden_size, None, None), name="fpn_3")
        text_proj_in = layers.Input(
            shape=(None, cfg["detr_encoder_hidden_size"]),
            name="text_projected",
            dtype="float32",
        )
        text_mask_in = layers.Input(
            shape=(None,), name="text_attention_mask", dtype="float32"
        )

        fpn_hidden_states = [fpn_0_in, fpn_1_in, fpn_2_in, fpn_3_in]

        text_attn_mask = layers.Lambda(
            lambda m: ops.expand_dims(
                ops.expand_dims((1.0 - m) * (-1e9), axis=1), axis=1
            ),
            name="text_attn_mask",
        )(text_mask_in)

        encoder_output, encoder_pos_flat, enc_h = sam3_detr_encoder(
            fpn_hidden_states,
            text_proj_in,
            text_attn_mask,
            grid_size=grid_size,
            fpn_hidden_size=fpn_hidden_size,
            detr_encoder_hidden_size=cfg["detr_encoder_hidden_size"],
            detr_encoder_num_layers=cfg["detr_encoder_num_layers"],
            detr_encoder_num_attention_heads=cfg["detr_encoder_num_attention_heads"],
            detr_encoder_intermediate_size=cfg["detr_encoder_intermediate_size"],
            detr_encoder_dropout=cfg["detr_encoder_dropout"],
        )

        decoder_hidden, pred_boxes, pred_logits, presence_logits = sam3_detr_decoder(
            encoder_output,
            encoder_pos_flat,
            text_proj_in,
            text_attn_mask,
            text_mask_in,
            enc_h,
            detr_decoder_hidden_size=cfg["detr_decoder_hidden_size"],
            detr_decoder_num_layers=cfg["detr_decoder_num_layers"],
            detr_decoder_num_queries=cfg["detr_decoder_num_queries"],
            detr_decoder_num_attention_heads=cfg["detr_decoder_num_attention_heads"],
            detr_decoder_intermediate_size=cfg["detr_decoder_intermediate_size"],
            detr_decoder_dropout=cfg["detr_decoder_dropout"],
        )

        pred_masks, semantic_seg = sam3_mask_decoder(
            encoder_output,
            decoder_hidden,
            text_proj_in,
            text_attn_mask,
            fpn_hidden_states,
            enc_h,
            fpn_hidden_size=fpn_hidden_size,
            mask_decoder_hidden_size=cfg["mask_decoder_hidden_size"],
            mask_decoder_num_attention_heads=cfg["mask_decoder_num_attention_heads"],
        )

        fpn_3_identity = layers.Identity(name="fpn_3_passthrough")(fpn_3_in)

        outputs = {
            "pred_masks": pred_masks,
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": semantic_seg,
            "fpn_05x": fpn_3_identity,
        }

        decoder_model = keras.Model(
            inputs={
                "fpn_0": fpn_0_in,
                "fpn_1": fpn_1_in,
                "fpn_2": fpn_2_in,
                "fpn_3": fpn_3_in,
                "text_projected": text_proj_in,
                "text_attention_mask": text_mask_in,
            },
            outputs=outputs,
            name="SAM3_decoder",
        )

        orig_weights = {w.path: w.numpy() for w in self.weights}
        for w in decoder_model.weights:
            path = w.path.replace("SAM3_decoder/", "SAM3Main/")
            if path in orig_weights and w.shape == orig_weights[path].shape:
                w.assign(orig_weights[path])

        return decoder_model

    def build_vision_model(self):
        """Build a vision-only sub-model that outputs FPN features.

        Returns:
            ``keras.Model`` with outputs ``{fpn_0..3, text_projected}``.
        """
        outputs = {}
        for i in range(4):
            layer = self.get_layer(f"fpn_level_{i}_proj2")
            outputs[f"fpn_{i}"] = layer.output
        outputs["text_projected"] = self.get_layer("text_projection").output

        return keras.Model(
            inputs=self.input,
            outputs=outputs,
            name="SAM3_vision",
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vit_hidden_size": self.vit_hidden_size,
                "vit_intermediate_size": self.vit_intermediate_size,
                "vit_num_hidden_layers": self.vit_num_hidden_layers,
                "vit_num_attention_heads": self.vit_num_attention_heads,
                "vit_image_size": self.vit_image_size,
                "vit_patch_size": self.vit_patch_size,
                "vit_window_size": self.vit_window_size,
                "vit_global_attn_indexes": self.vit_global_attn_indexes,
                "vit_rope_theta": self.vit_rope_theta,
                "vit_pretrain_image_size": self.vit_pretrain_image_size,
                "fpn_hidden_size": self.fpn_hidden_size,
                "fpn_scale_factors": self.fpn_scale_factors,
                "detr_encoder_hidden_size": self.detr_encoder_hidden_size,
                "detr_encoder_num_layers": self.detr_encoder_num_layers,
                "detr_encoder_num_attention_heads": self.detr_encoder_num_attention_heads,
                "detr_encoder_intermediate_size": self.detr_encoder_intermediate_size,
                "detr_encoder_dropout": self.detr_encoder_dropout,
                "detr_decoder_hidden_size": self.detr_decoder_hidden_size,
                "detr_decoder_num_layers": self.detr_decoder_num_layers,
                "detr_decoder_num_queries": self.detr_decoder_num_queries,
                "detr_decoder_num_attention_heads": self.detr_decoder_num_attention_heads,
                "detr_decoder_intermediate_size": self.detr_decoder_intermediate_size,
                "detr_decoder_dropout": self.detr_decoder_dropout,
                "mask_decoder_hidden_size": self.mask_decoder_hidden_size,
                "mask_decoder_num_upsampling_stages": self.mask_decoder_num_upsampling_stages,
                "mask_decoder_num_attention_heads": self.mask_decoder_num_attention_heads,
                "text_hidden_size": self.text_hidden_size,
                "text_projection_dim": self.text_projection_dim,
                "input_shape": self._input_shape_val,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def sam3_clip_encoder_layer(
    hidden_states,
    hidden_size,
    num_attention_heads,
    intermediate_size,
    attention_mask,
    name,
):
    """Single CLIP encoder layer: self-attention with causal mask and MLP.

    Pre-norm transformer block with GELU-activated feed-forward network.
    Used inside ``build_text_encoder`` to construct the CLIP text model.

    Args:
        hidden_states: Input tensor ``(B, seq, D)``.
        hidden_size (int): Hidden dimension.
        num_attention_heads (int): Number of attention heads.
        intermediate_size (int): MLP intermediate dimension.
        attention_mask: Combined causal + padding mask
            ``(B, 1, seq, seq)``.
        name (str): Layer name prefix.

    Returns:
        Updated hidden states ``(B, seq, D)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    layer_norm1 = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_layer_norm1")
    layer_norm2 = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_layer_norm2")
    self_attn = SAM3CLIPAttention(
        hidden_size, num_attention_heads, name=f"{name}_self_attn"
    )
    fc1 = layers.Dense(intermediate_size, name=f"{name}_fc1")
    fc2 = layers.Dense(hidden_size, name=f"{name}_fc2")

    residual = hidden_states
    hidden_states = layer_norm1(hidden_states)
    hidden_states = self_attn(hidden_states, attention_mask=attention_mask)
    hidden_states = layers.Add(name=f"{name}_add1")([residual, hidden_states])

    residual = hidden_states
    hidden_states = layer_norm2(hidden_states)
    hidden_states = fc1(hidden_states)
    hidden_states = layers.Activation("gelu", name=f"{name}_gelu")(hidden_states)
    hidden_states = fc2(hidden_states)
    hidden_states = layers.Add(name=f"{name}_add2")([residual, hidden_states])
    return hidden_states


def build_text_encoder(
    vocab_size=SAM3_VOCAB_SIZE,
    hidden_size=CLIP_HIDDEN_SIZE,
    num_hidden_layers=CLIP_NUM_LAYERS,
    num_attention_heads=CLIP_NUM_HEADS,
    intermediate_size=CLIP_INTERMEDIATE_SIZE,
    max_position_embeddings=CLIP_MAX_POSITION,
    weights_path=None,
):
    """Build the CLIP text encoder as a functional ``keras.Model``.

    Constructs a 24-layer CLIP text transformer with token and position
    embeddings, causal masking, and final layer normalization. Returns
    the last hidden state for all tokens.

    Args:
        vocab_size (int): Vocabulary size. Defaults to ``49408``.
        hidden_size (int): Hidden dimension. Defaults to ``1024``.
        num_hidden_layers (int): Number of transformer layers.
            Defaults to ``24``.
        num_attention_heads (int): Number of attention heads.
            Defaults to ``16``.
        intermediate_size (int): MLP intermediate dimension.
            Defaults to ``4096``.
        max_position_embeddings (int): Maximum sequence length.
            Defaults to ``32``.
        weights_path (str or None): Path to ``.weights.h5`` file.

    Returns:
        ``keras.Model`` with inputs ``{input_ids, attention_mask}``
        and output ``(B, seq_len, hidden_size)``.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    input_ids = keras.Input(
        shape=(max_position_embeddings,), dtype="int32", name="input_ids"
    )
    attention_mask = keras.Input(
        shape=(max_position_embeddings,), dtype="int32", name="attention_mask"
    )

    hidden_states = layers.Embedding(vocab_size, hidden_size, name="token_embedding")(
        input_ids
    )

    hidden_states = CLIPPositionEmbedding(
        max_position_embeddings, hidden_size, name="add_position"
    )(hidden_states)

    combined_mask = CLIPCausalMask(max_position_embeddings, name="causal_mask")(
        attention_mask
    )

    for i in range(num_hidden_layers):
        hidden_states = sam3_clip_encoder_layer(
            hidden_states,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            combined_mask,
            name=f"layers_{i}",
        )

    output = layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")(
        hidden_states
    )

    model = keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=output,
        name="text_encoder",
    )

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


@register_model
def SAM3(input_shape=None, input_tensor=None, weights=None, **kwargs):
    """SAM3 open-vocabulary detector, segmenter, and promptable model.

    Factory function that builds the full SAM3 model including ViT-L
    backbone, FPN, DETR encoder/decoder, CLIP text encoder, geometry
    encoder, and mask decoder. Supports 839M parameters.

    Args:
        input_shape (tuple or None): Input image shape. Defaults to
            ``(1008, 1008, 3)`` for channels_last.
        input_tensor: Optional input tensor.
        weights (str or None): ``None`` for random init, or a path to
            a ``.weights.h5`` file.
        **kwargs: Additional keyword arguments.

    Returns:
        ``SAM3Main`` instance.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """
    config = SAM3_MODEL_CONFIG["SAM3"]

    valid_model_weights = []
    if "SAM3" in SAM3_WEIGHTS_CONFIG:
        valid_model_weights = list(SAM3_WEIGHTS_CONFIG["SAM3"].keys())
    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported: {', '.join(str(w) for w in valid_weights)}, or a file path."
        )

    model = SAM3Main(
        input_shape=input_shape,
        input_tensor=input_tensor,
        **config,
        **kwargs,
    )

    if weights in valid_model_weights:
        load_weights_from_config(model, SAM3_WEIGHTS_CONFIG["SAM3"], weights)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model
