import keras
from keras import layers, ops, utils

from kmodels.model_registry import register_model

from .config import SAM3_MODEL_CONFIG, SAM3_WEIGHTS_CONFIG
from .sam3_layers import (
    SAM3AddPositionEmbedding,
    SAM3BoxRPB,
    SAM3DecoderMLP,
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


def compute_rotary_embeddings(
    hidden_size, num_attention_heads, end_x, end_y, rope_theta=10000.0, scale=1.0
):
    """Compute 2D rotary position embeddings (cos, sin) using Keras ops.

    Returns:
        cos: (end_x * end_y, head_dim) float32 tensor.
        sin: (end_x * end_y, head_dim) float32 tensor.
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


def _build_vision_backbone(
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
):
    """ViT backbone: patch embed → position embed → LN → N transformer layers.

    Returns:
        backbone_nchw: (B, C, H, W) backbone features in NCHW format.
        grid_size: spatial grid size (H = W).
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
        name="backbone_patch_embed",
    )(pixel_values)

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

    backbone_nchw = layers.Permute((3, 1, 2), name="backbone_to_nchw")(backbone_out)
    return backbone_nchw, grid_size


def _build_fpn_neck(backbone_nchw, vit_hidden_size, fpn_hidden_size, fpn_scale_factors):
    """FPN neck: multi-scale feature pyramid from backbone output.

    Returns:
        fpn_hidden_states: list of (B, C, H, W) tensors at each scale.
    """
    fpn_hidden_states = []

    for level_idx, scale_factor in enumerate(fpn_scale_factors):
        if scale_factor == 4.0:
            x = layers.Conv2DTranspose(
                vit_hidden_size // 2,
                kernel_size=2,
                strides=2,
                data_format="channels_first",
                name=f"fpn_level_{level_idx}_deconv1",
            )(backbone_nchw)
            x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu1")(x)
            x = layers.Conv2DTranspose(
                vit_hidden_size // 4,
                kernel_size=2,
                strides=2,
                data_format="channels_first",
                name=f"fpn_level_{level_idx}_deconv2",
            )(x)
            x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu2")(x)
        elif scale_factor == 2.0:
            x = layers.Conv2DTranspose(
                vit_hidden_size // 2,
                kernel_size=2,
                strides=2,
                data_format="channels_first",
                name=f"fpn_level_{level_idx}_deconv1",
            )(backbone_nchw)
            x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu1")(x)
        elif scale_factor == 1.0:
            x = backbone_nchw
        else:
            x = layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                data_format="channels_first",
                name=f"fpn_level_{level_idx}_pool",
            )(backbone_nchw)

        x = layers.Conv2D(
            fpn_hidden_size,
            kernel_size=1,
            data_format="channels_first",
            name=f"fpn_level_{level_idx}_proj1",
        )(x)
        x = layers.Conv2D(
            fpn_hidden_size,
            kernel_size=3,
            padding="same",
            data_format="channels_first",
            name=f"fpn_level_{level_idx}_proj2",
        )(x)
        fpn_hidden_states.append(x)

    return fpn_hidden_states


def _detr_encoder_layer(
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
    """Single DETR encoder layer (functional): self-attn + text cross-attn + MLP."""
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


def _build_detr_encoder(
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
):
    """DETR encoder: self-attention on vision + cross-attention to text.

    Returns:
        encoder_output: (B, H*W, D) encoded vision features.
        encoder_pos_flat: (1, H*W, D) position encoding constant.
        enc_h: encoder spatial size.
    """
    encoder_vision = fpn_hidden_states[-2]  # 1x scale
    enc_h = grid_size

    encoder_vision_flat = layers.Permute((2, 3, 1), name="encoder_vision_to_nhwc")(
        encoder_vision
    )
    encoder_vision_flat = layers.Reshape(
        (enc_h * enc_h, fpn_hidden_size), name="encoder_vision_flatten"
    )(encoder_vision_flat)

    enc_pos_np = compute_sine_pos_encoding(
        enc_h, enc_h, fpn_hidden_size // 2, normalize=True
    )
    enc_pos_np = enc_pos_np.transpose(0, 2, 3, 1).reshape(
        1, enc_h * enc_h, fpn_hidden_size
    )
    encoder_pos_flat = ops.convert_to_tensor(enc_pos_np, dtype="float32")

    encoder_output = encoder_vision_flat
    for i in range(detr_encoder_num_layers):
        encoder_output = _detr_encoder_layer(
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


def _detr_decoder_layer(
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
    """Single DETR decoder layer (functional): self-attn + text cross-attn
    + vision cross-attn + MLP."""
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


def _dot_product_scoring(
    decoder_hidden_states,
    text_features,
    text_mask,
    hidden_size,
    intermediate_size=2048,
    name="dot_product_scoring",
):
    """Dot-product scoring: text MLP + pooling + projection → logits."""
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


def _build_detr_decoder(
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
    """DETR decoder: iterative box refinement with cross-attention.

    Returns:
        decoder_hidden: (B, Q, D) last layer's output.
        pred_boxes: (B, Q, 4) predicted boxes in xyxy format.
        pred_logits: (B, Q) classification logits.
        presence_logits_stacked: (num_layers,) presence logits.
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

        hidden_states = _detr_decoder_layer(
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

    pred_logits = _dot_product_scoring(
        decoder_hidden,
        text_projected,
        text_attention_mask,
        hidden_size=detr_decoder_hidden_size,
    )

    presence_logits_stacked = ops.concatenate(
        [ops.squeeze(p, axis=1) for p in all_presence_logits], axis=-1
    )

    return decoder_hidden, pred_boxes, pred_logits, presence_logits_stacked


def _mask_embedder(x, hidden_size, name_prefix="mask_embedder"):
    """3-layer MLP: Dense → ReLU → Dense → ReLU → Dense."""
    x = layers.Dense(hidden_size, name=f"{name_prefix}_linear1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)
    x = layers.Dense(hidden_size, name=f"{name_prefix}_linear2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)
    x = layers.Dense(hidden_size, name=f"{name_prefix}_linear3")(x)
    return x


def _build_mask_decoder(
    encoder_output,
    decoder_hidden,
    text_projected,
    text_attn_mask,
    fpn_hidden_states,
    enc_h,
    fpn_hidden_size,
    mask_decoder_hidden_size,
    mask_decoder_num_attention_heads,
):
    """Mask decoder: prompt cross-attention → pixel decoder → mask prediction.

    Returns:
        pred_masks: (B, Q, H, W) mask logits.
        semantic_seg: (B, 1, H, W) semantic segmentation logits.
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
    pixel_feat = layers.Permute((3, 1, 2), name="pixel_decoder_to_nchw")(pixel_feat)

    num_up = len(fpn_hidden_states) - 2
    for stage_idx in range(num_up):
        pixel_feat_nhwc = layers.Permute(
            (2, 3, 1),
            name=f"pixel_decoder_stage_{stage_idx}_to_nhwc",
        )(pixel_feat)
        pixel_feat_nhwc = layers.UpSampling2D(
            size=2,
            interpolation="nearest",
            name=f"pixel_decoder_stage_{stage_idx}_upsample",
        )(pixel_feat_nhwc)

        skip_idx = len(fpn_hidden_states) - 3 - stage_idx
        if skip_idx >= 0:
            skip_nhwc = layers.Permute(
                (2, 3, 1),
                name=f"pixel_decoder_stage_{stage_idx}_skip_to_nhwc",
            )(fpn_hidden_states[skip_idx])
            pixel_feat_nhwc = layers.Add(
                name=f"pixel_decoder_stage_{stage_idx}_add_skip",
            )([pixel_feat_nhwc, skip_nhwc])

        pixel_feat_nhwc = layers.Conv2D(
            mask_decoder_hidden_size,
            kernel_size=3,
            padding="same",
            name=f"pixel_decoder_stage_{stage_idx}_conv",
        )(pixel_feat_nhwc)
        pixel_feat_nhwc = layers.GroupNormalization(
            groups=8,
            name=f"pixel_decoder_stage_{stage_idx}_gn",
        )(pixel_feat_nhwc)
        pixel_feat_nhwc = layers.ReLU(
            name=f"pixel_decoder_stage_{stage_idx}_relu",
        )(pixel_feat_nhwc)

        pixel_feat = layers.Permute(
            (3, 1, 2),
            name=f"pixel_decoder_stage_{stage_idx}_to_nchw",
        )(pixel_feat_nhwc)

    instance_embed = layers.Conv2D(
        mask_decoder_hidden_size,
        kernel_size=1,
        data_format="channels_first",
        name="mask_decoder_instance_proj",
    )(pixel_feat)

    semantic_seg = layers.Conv2D(
        1,
        kernel_size=1,
        data_format="channels_first",
        name="mask_decoder_semantic_proj",
    )(pixel_feat)

    mask_embeddings = _mask_embedder(decoder_hidden, mask_decoder_hidden_size)

    instance_nhwc = layers.Permute((2, 3, 1), name="instance_to_nhwc")(instance_embed)
    pred_masks = ops.einsum("bqc,bhwc->bqhw", mask_embeddings, instance_nhwc)

    return pred_masks, semantic_seg


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3(keras.Model):
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
        name="SAM3",
        **kwargs,
    ):
        if input_shape is None:
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
        backbone_nchw, grid_size = _build_vision_backbone(
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
        )

        fpn_hidden_states = _build_fpn_neck(
            backbone_nchw,
            vit_hidden_size,
            fpn_hidden_size,
            fpn_scale_factors,
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

        encoder_output, encoder_pos_flat, enc_h = _build_detr_encoder(
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
        )

        decoder_hidden, pred_boxes, pred_logits, presence_logits = _build_detr_decoder(
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

        pred_masks, semantic_seg = _build_mask_decoder(
            encoder_output,
            decoder_hidden,
            text_projected,
            text_attn_mask,
            fpn_hidden_states,
            enc_h,
            fpn_hidden_size=fpn_hidden_size,
            mask_decoder_hidden_size=mask_decoder_hidden_size,
            mask_decoder_num_attention_heads=mask_decoder_num_attention_heads,
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


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3Model(keras.Model):
    """Unified SAM3 model containing all components.

    Holds the core detector (vision backbone + FPN + DETR + mask decoder),
    CLIP text encoder, and geometry encoder in one model. All weights are
    saved/loaded together in a single file.

    This is the model returned by the Sam3() factory function.
    Downstream task classes (SAM3ObjectDetection, etc.) wrap this model.
    """

    def __init__(self, detector, **kwargs):
        super().__init__(name=kwargs.pop("name", "SAM3Model"), **kwargs)
        from .sam3_clip import SAM3CLIPTextEncoder
        from .sam3_layers import SAM3GeometryEncoder

        self.detector = detector

        self.text_encoder = SAM3CLIPTextEncoder(name="text_encoder")
        self.text_encoder.build((None, 32))

        self.geometry_encoder = SAM3GeometryEncoder(
            hidden_size=detector.detr_encoder_hidden_size,
            name="geometry_encoder",
        )
        self.geometry_encoder.build((None, None, 4))

    def call(self, inputs, training=None):
        return self.detector(inputs, training=training)

    def build(self, input_shape=None):
        if not self.built:
            self.detector.build(input_shape)
            self.built = True

    def get_config(self):
        config = super().get_config()
        config["detector"] = keras.saving.serialize_keras_object(self.detector)
        return config

    @classmethod
    def from_config(cls, config):
        detector = keras.saving.deserialize_keras_object(config.pop("detector"))
        return cls(detector=detector, **config)


def _create_sam3_model(
    variant, input_shape=None, input_tensor=None, weights=None, **kwargs
):
    config = SAM3_MODEL_CONFIG[variant]

    valid_model_weights = []
    if variant in SAM3_WEIGHTS_CONFIG:
        valid_model_weights = list(SAM3_WEIGHTS_CONFIG[variant].keys())
    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported: {', '.join(str(w) for w in valid_weights)}, or a file path."
        )

    if input_shape is None:
        image_size = config["vit_image_size"]
        input_shape = (image_size, image_size, 3)

    detector = SAM3(
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **config,
        **kwargs,
    )

    model = SAM3Model(detector=detector)
    model.build(None)

    if weights in valid_model_weights:
        from kmodels.utils import load_weights_from_config

        load_weights_from_config(model, SAM3_WEIGHTS_CONFIG[variant], weights)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Sam3(input_shape=None, input_tensor=None, weights=None, **kwargs):
    return _create_sam3_model(
        "Sam3",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


def build_sam3_decoder_model(sam3_model):
    """Build a decoder-only model that takes pre-computed FPN features.

    Takes FPN outputs + projected text (256d) as inputs, skipping the
    vision backbone. Use with build_sam3_vision_model() for efficient
    multi-prompt inference on the same image.

    Args:
        sam3_model: a trained SAM3 model to copy config and weights from.

    Returns:
        decoder_model: keras.Model with inputs:
            - fpn_0: (B, 256, 288, 288) FPN 4x
            - fpn_1: (B, 256, 144, 144) FPN 2x
            - fpn_2: (B, 256, 72, 72) FPN 1x
            - fpn_3: (B, 256, 36, 36) FPN 0.5x
            - text_projected: (B, seq, 256) pre-projected text features
            - text_attention_mask: (B, seq) float mask
    """
    det = sam3_model.detector if hasattr(sam3_model, "detector") else sam3_model
    cfg = det.get_config()
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
        lambda m: ops.expand_dims(ops.expand_dims((1.0 - m) * (-1e9), axis=1), axis=1),
        name="text_attn_mask",
    )(text_mask_in)

    encoder_output, encoder_pos_flat, enc_h = _build_detr_encoder(
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

    decoder_hidden, pred_boxes, pred_logits, presence_logits = _build_detr_decoder(
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

    pred_masks, semantic_seg = _build_mask_decoder(
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

    orig_weights = {w.path: w.numpy() for w in det.weights}
    for w in decoder_model.weights:
        path = w.path.replace("SAM3_decoder/", "SAM3/")
        if path in orig_weights and w.shape == orig_weights[path].shape:
            w.assign(orig_weights[path])

    return decoder_model


def build_sam3_vision_model(sam3_model):
    """Build a vision-only model that outputs FPN features + text projection.

    Use with build_sam3_decoder_model() for efficient multi-prompt inference.

    Args:
        sam3_model: a trained SAM3 model.

    Returns:
        vision_model: keras.Model with same inputs as SAM3, outputs:
            - fpn_0 through fpn_3: FPN level features (NCHW)
            - text_projected: projected text features (256d)
    """
    det = sam3_model.detector
    outputs = {}
    for i in range(4):
        layer = det.get_layer(f"fpn_level_{i}_proj2")
        outputs[f"fpn_{i}"] = layer.output
    outputs["text_projected"] = det.get_layer("text_projection").output

    return keras.Model(
        inputs=det.input,
        outputs=outputs,
        name="SAM3_vision",
    )
