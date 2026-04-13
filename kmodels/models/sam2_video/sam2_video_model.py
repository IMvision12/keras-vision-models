import math

import keras
import numpy as np
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.models.sam2.sam2_layers import (
    SAM2HieraPositionEmbedding,
    SAM2ImagePositionalEmbeddings,
    SAM2MaskDecoderLayer,
    SAM2MultiScaleBlock,
    SAM2NoMemoryEmbedding,
    SAM2PositionalEmbedding,
    SAM2PromptEncoderLayer,
)
from kmodels.utils import load_weights_from_config

from .config import SAM2_VIDEO_MODEL_CONFIG, SAM2_VIDEO_WEIGHTS_CONFIG
from .sam2_video_layers import Sam2VideoMemoryAttention


def sam2_video_feed_forward(input_dim, hidden_dim, output_dim, num_layers, name):
    proj_in = layers.Dense(hidden_dim, name=f"{name}_proj_in")
    hidden_layers = [
        layers.Dense(hidden_dim, name=f"{name}_hidden_{i}")
        for i in range(num_layers - 2)
    ]
    proj_out = layers.Dense(output_dim, name=f"{name}_proj_out")
    return proj_in, hidden_layers, proj_out


def sam2_video_feed_forward_call(x, proj_in, hidden_layers, proj_out):
    x = ops.relu(proj_in(x))
    for layer in hidden_layers:
        x = ops.relu(layer(x))
    return proj_out(x)


def sam2_video_mask_downsampler(embed_dim, name):
    num_stages = 4
    ds_convs = []
    ds_lns = []
    in_ch = 1
    for i in range(num_stages):
        out_ch = in_ch * 4
        ds_convs.append(
            layers.Conv2D(
                out_ch,
                3,
                strides=2,
                padding="valid",
                data_format="channels_first",
                name=f"{name}_conv_{i}",
            )
        )
        ds_lns.append(layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln_{i}"))
        in_ch = out_ch
    final_conv = layers.Conv2D(
        embed_dim,
        1,
        padding="valid",
        data_format="channels_first",
        name=f"{name}_final_conv",
    )
    return ds_convs, ds_lns, final_conv


def sam2_video_mask_downsampler_call(x, ds_convs, ds_lns, final_conv):
    for conv, ln in zip(ds_convs, ds_lns):
        x = ops.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]])
        x = conv(x)
        x = ops.transpose(x, [0, 2, 3, 1])
        x = ln(x)
        x = ops.transpose(x, [0, 3, 1, 2])
        x = ops.nn.gelu(x, approximate=False)
    return final_conv(x)


def sam2_video_cx_block(embed_dim, intermediate_dim, kernel_size, name):
    dw_conv = layers.DepthwiseConv2D(
        kernel_size,
        padding="valid",
        data_format="channels_first",
        name=f"{name}_dw_conv",
    )
    ln = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln")
    pw1 = layers.Dense(intermediate_dim, name=f"{name}_pw1")
    pw2 = layers.Dense(embed_dim, name=f"{name}_pw2")
    return dw_conv, ln, pw1, pw2


def sam2_video_cx_block_call(x, dw_conv, ln, pw1, pw2, scale, padding):
    residual = x
    x = ops.pad(x, [[0, 0], [0, 0], [padding, padding], [padding, padding]])
    x = dw_conv(x)
    x = ops.transpose(x, [0, 2, 3, 1])
    x = ln(x)
    x = pw1(x)
    x = ops.nn.gelu(x, approximate=False)
    x = pw2(x)
    x = scale * x
    x = ops.transpose(x, [0, 3, 1, 2])
    return residual + x


def sam2_video_memory_fuser(num_blocks, embed_dim, intermediate_dim, kernel_size, name):
    dw_convs = []
    lns = []
    pw1s = []
    pw2s = []
    for i in range(num_blocks):
        dw, ln, pw1, pw2 = sam2_video_cx_block(
            embed_dim, intermediate_dim, kernel_size, f"{name}_{i}"
        )
        dw_convs.append(dw)
        lns.append(ln)
        pw1s.append(pw1)
        pw2s.append(pw2)
    return dw_convs, lns, pw1s, pw2s


def sam2_video_memory_fuser_call(x, dw_convs, lns, pw1s, pw2s, scales, padding):
    for dw_conv, ln, pw1, pw2, scale in zip(dw_convs, lns, pw1s, pw2s, scales):
        x = sam2_video_cx_block_call(x, dw_conv, ln, pw1, pw2, scale, padding)
    return x


def sam2_video_sine_position_embedding(x, num_pos_feats, temperature=10000):
    scale = 2.0 * math.pi
    shape = ops.shape(x)
    h, w = shape[2], shape[3]
    y_embed = ops.cast(ops.expand_dims(ops.arange(1, h + 1), 1), dtype="float32")
    x_embed = ops.cast(ops.expand_dims(ops.arange(1, w + 1), 0), dtype="float32")
    y_embed = ops.broadcast_to(y_embed, [h, w])
    x_embed = ops.broadcast_to(x_embed, [h, w])

    eps = 1e-6
    y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, -1:] + eps) * scale

    dim_t = ops.cast(ops.arange(num_pos_feats), dtype="float32")
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = ops.expand_dims(x_embed, -1) / dim_t
    pos_y = ops.expand_dims(y_embed, -1) / dim_t

    pos_x = ops.reshape(
        ops.stack([ops.sin(pos_x[:, :, 0::2]), ops.cos(pos_x[:, :, 1::2])], axis=3),
        [h, w, num_pos_feats],
    )
    pos_y = ops.reshape(
        ops.stack([ops.sin(pos_y[:, :, 0::2]), ops.cos(pos_y[:, :, 1::2])], axis=3),
        [h, w, num_pos_feats],
    )

    pos = ops.concatenate([pos_y, pos_x], axis=-1)
    pos = ops.transpose(pos, [2, 0, 1])
    return ops.expand_dims(pos, 0)


def sam2_video_memory_encoder(hidden_size, output_channels, name):
    ds_convs, ds_lns, ds_final_conv = sam2_video_mask_downsampler(
        hidden_size, f"{name}_mask_ds"
    )
    feature_proj = layers.Conv2D(
        hidden_size,
        1,
        padding="valid",
        data_format="channels_first",
        name=f"{name}_feature_proj",
    )
    fuser_dw_convs, fuser_lns, fuser_pw1s, fuser_pw2s = sam2_video_memory_fuser(
        2, hidden_size, 1024, 7, f"{name}_fuser"
    )
    projection = layers.Conv2D(
        output_channels,
        1,
        padding="valid",
        data_format="channels_first",
        name=f"{name}_projection",
    )
    return (
        ds_convs,
        ds_lns,
        ds_final_conv,
        feature_proj,
        fuser_dw_convs,
        fuser_lns,
        fuser_pw1s,
        fuser_pw2s,
        projection,
    )


def sam2_video_memory_encoder_call(
    vision_features,
    masks,
    ds_convs,
    ds_lns,
    ds_final_conv,
    feature_proj,
    fuser_dw_convs,
    fuser_lns,
    fuser_pw1s,
    fuser_pw2s,
    fuser_scales,
    projection,
    num_pos_feats,
):
    masks = sam2_video_mask_downsampler_call(masks, ds_convs, ds_lns, ds_final_conv)
    vision_features = feature_proj(vision_features)
    vision_features = vision_features + masks
    vision_features = sam2_video_memory_fuser_call(
        vision_features,
        fuser_dw_convs,
        fuser_lns,
        fuser_pw1s,
        fuser_pw2s,
        fuser_scales,
        padding=3,
    )
    vision_features = projection(vision_features)
    pos_enc = sam2_video_sine_position_embedding(vision_features, num_pos_feats)
    return vision_features, pos_enc


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2Video(keras.Model):
    IMAGE_SIZE = 1024
    PATCH_KERNEL = (7, 7)
    PATCH_STRIDE = (4, 4)
    PATCH_PADDING = 3
    QUERY_STRIDE = 2
    NUM_QUERY_POOL_STAGES = 3
    WINDOW_POS_EMBED_BG_SIZE = (7, 7)
    FPN_HIDDEN_SIZE = 256
    NUM_FEATURE_LEVELS = 3
    LAYER_NORM_EPS = 1e-6
    MLP_RATIO = 4.0
    MASK_DECODER_HIDDEN_SIZE = 256
    MASK_DECODER_NUM_HIDDEN_LAYERS = 2
    MASK_DECODER_NUM_ATTENTION_HEADS = 8
    MASK_DECODER_MLP_DIM = 2048
    MASK_DECODER_IOU_HEAD_DEPTH = 3
    MASK_DECODER_IOU_HEAD_HIDDEN_DIM = 256
    MASK_DECODER_ATTENTION_DOWNSAMPLE_RATE = 2
    PROMPT_ENCODER_HIDDEN_SIZE = 256
    PROMPT_ENCODER_MASK_INPUT_CHANNELS = 16
    PROMPT_ENCODER_NUM_POINT_EMBEDDINGS = 4
    PROMPT_ENCODER_PATCH_SIZE = 16
    NUM_MULTIMASK_OUTPUTS = 3
    MEM_DIM = 64
    NUM_MASKMEM = 7

    def __init__(
        self,
        hidden_size=96,
        blocks_per_stage=(1, 2, 7, 2),
        embed_dim_per_stage=(96, 192, 384, 768),
        num_attention_heads_per_stage=(1, 2, 4, 8),
        window_size_per_stage=(8, 4, 14, 7),
        global_attention_blocks=(5, 7, 9),
        backbone_channel_list=(768, 384, 192, 96),
        window_pos_embed_bg_size=None,
        input_shape=None,
        input_tensor=None,
        name="Sam2Video",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()

        if window_pos_embed_bg_size is None:
            window_pos_embed_bg_size = self.WINDOW_POS_EMBED_BG_SIZE

        if input_shape is None:
            if data_format == "channels_first":
                input_shape = (3, self.IMAGE_SIZE, self.IMAGE_SIZE)
            else:
                input_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)

        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=input_shape, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=input_shape, name="pixel_values")

        input_points = layers.Input(
            shape=(None, None, 2), name="input_points", dtype="float32"
        )
        input_labels = layers.Input(
            shape=(None, None), name="input_labels", dtype="int32"
        )

        padded = layers.ZeroPadding2D(
            padding=self.PATCH_PADDING,
            data_format=data_format,
            name="backbone_patch_embed_padding",
        )(pixel_values)
        hidden_states = layers.Conv2D(
            hidden_size,
            kernel_size=self.PATCH_KERNEL,
            strides=self.PATCH_STRIDE,
            padding="valid",
            use_bias=True,
            data_format=data_format,
            name="backbone_patch_embed_projection",
        )(padded)

        if data_format == "channels_first":
            spatial_h, spatial_w = input_shape[1], input_shape[2]
        else:
            spatial_h, spatial_w = input_shape[0], input_shape[1]
        pos_embed_h = spatial_h // self.PATCH_STRIDE[0]
        pos_embed_w = spatial_w // self.PATCH_STRIDE[1]
        pos_embed_layer = SAM2HieraPositionEmbedding(
            hidden_size=hidden_size,
            spatial_size=(pos_embed_h, pos_embed_w),
            window_size=window_size_per_stage[0],
            bg_size=window_pos_embed_bg_size,
            data_format=data_format,
            name="backbone_pos_embed",
        )
        hidden_states = pos_embed_layer(hidden_states)

        stage_ends = (np.cumsum(blocks_per_stage) - 1).tolist()
        intermediate_hidden_states = []
        total_block_idx = 0
        for stage_idx, num_blocks in enumerate(blocks_per_stage):
            for block_idx in range(num_blocks):
                dim_in = (
                    embed_dim_per_stage[stage_idx - 1]
                    if stage_idx > 0 and block_idx == 0
                    else embed_dim_per_stage[stage_idx]
                )
                dim_out = embed_dim_per_stage[stage_idx]

                win = (
                    window_size_per_stage[stage_idx - 1]
                    if stage_idx > 0 and block_idx == 0
                    else window_size_per_stage[stage_idx]
                )
                if total_block_idx in global_attention_blocks:
                    win = 0

                q_stride = (
                    self.QUERY_STRIDE
                    if (0 < stage_idx <= self.NUM_QUERY_POOL_STAGES and block_idx == 0)
                    else None
                )

                hidden_states = SAM2MultiScaleBlock(
                    dim=dim_in,
                    dim_out=dim_out,
                    num_heads=num_attention_heads_per_stage[stage_idx],
                    mlp_ratio=self.MLP_RATIO,
                    window_size=win,
                    query_stride=q_stride,
                    layer_norm_eps=self.LAYER_NORM_EPS,
                    data_format=data_format,
                    name=f"backbone_blocks_{total_block_idx}",
                )(hidden_states)

                if total_block_idx in stage_ends:
                    intermediate_hidden_states.append(hidden_states)

                total_block_idx += 1

        fpn_convs = []
        n = len(backbone_channel_list) - 1
        fpn_hidden_states_list = []

        for i, in_channels in enumerate(backbone_channel_list):
            conv = layers.Conv2D(
                self.FPN_HIDDEN_SIZE,
                kernel_size=1,
                data_format=data_format,
                name=f"neck_convs_{i}",
            )
            fpn_convs.append(conv)

        fpn_top_down_levels = [2, 3]

        prev_features = None
        for i in range(n, -1, -1):
            stage_features = intermediate_hidden_states[i]
            lateral_features = fpn_convs[n - i](stage_features)

            if i not in fpn_top_down_levels or i == n:
                prev_features = lateral_features
            else:
                top_down = layers.UpSampling2D(
                    size=2,
                    interpolation="nearest",
                    data_format=data_format,
                    name=f"neck_upsample_{i}",
                )(prev_features)
                prev_features = layers.Add(name=f"neck_add_{i}")(
                    [lateral_features, top_down]
                )

            fpn_hidden_states_list.append(prev_features)

        fpn_hidden_states_list = fpn_hidden_states_list[-self.NUM_FEATURE_LEVELS :][
            ::-1
        ]
        image_embeddings = fpn_hidden_states_list[-1]

        image_embeddings_raw = image_embeddings
        no_mem_embed_layer = SAM2NoMemoryEmbedding(
            hidden_size=self.FPN_HIDDEN_SIZE,
            data_format=data_format,
            name="no_memory_embedding",
        )
        image_embeddings = no_mem_embed_layer(image_embeddings)

        high_res_feat_s0 = fpn_hidden_states_list[0]
        high_res_feat_s1 = fpn_hidden_states_list[1]

        image_embedding_size = spatial_h // self.PROMPT_ENCODER_PATCH_SIZE

        shared_image_embedding = SAM2PositionalEmbedding(
            num_pos_feats=self.PROMPT_ENCODER_HIDDEN_SIZE // 2,
            scale=1.0,
            name="shared_image_embedding",
        )

        image_pe = SAM2ImagePositionalEmbeddings(
            image_embedding_size,
            shared_image_embedding,
            name="image_positional_embeddings",
        )(image_embeddings)

        prompt_results = SAM2PromptEncoderLayer(
            hidden_size=self.PROMPT_ENCODER_HIDDEN_SIZE,
            image_embedding_size=image_embedding_size,
            image_size=self.IMAGE_SIZE,
            num_point_embeddings=self.PROMPT_ENCODER_NUM_POINT_EMBEDDINGS,
            shared_embedding=shared_image_embedding,
            data_format=data_format,
            name="prompt_encoder",
        )([input_points, input_labels])

        sparse_embeddings = prompt_results["sparse_embeddings"]
        dense_embeddings = prompt_results["dense_embeddings"]

        decoder_output = SAM2MaskDecoderLayer(
            hidden_size=self.MASK_DECODER_HIDDEN_SIZE,
            num_hidden_layers=self.MASK_DECODER_NUM_HIDDEN_LAYERS,
            num_attention_heads=self.MASK_DECODER_NUM_ATTENTION_HEADS,
            mlp_dim=self.MASK_DECODER_MLP_DIM,
            num_multimask_outputs=self.NUM_MULTIMASK_OUTPUTS,
            iou_head_depth=self.MASK_DECODER_IOU_HEAD_DEPTH,
            iou_head_hidden_dim=self.MASK_DECODER_IOU_HEAD_HIDDEN_DIM,
            attention_downsample_rate=self.MASK_DECODER_ATTENTION_DOWNSAMPLE_RATE,
            data_format=data_format,
            name="mask_decoder",
        )(
            [
                image_embeddings,
                image_pe,
                sparse_embeddings,
                dense_embeddings,
                high_res_feat_s0,
                high_res_feat_s1,
            ]
        )

        pred_masks_all = decoder_output["pred_masks"]
        iou_scores_all = decoder_output["iou_scores"]
        mask_tokens_out_all = decoder_output["mask_tokens_out"]
        pred_masks = pred_masks_all[:, :, 1:, :, :]
        iou_scores = iou_scores_all[:, :, 1:]
        object_score_logits = decoder_output["object_score_logits"]

        super().__init__(
            inputs={
                "pixel_values": pixel_values,
                "input_points": input_points,
                "input_labels": input_labels,
            },
            outputs={
                "pred_masks": pred_masks,
                "iou_scores": iou_scores,
                "object_score_logits": object_score_logits,
                "image_embeddings_raw": image_embeddings_raw,
                "image_embeddings": image_embeddings,
                "high_res_feat_s0": high_res_feat_s0,
                "high_res_feat_s1": high_res_feat_s1,
                "image_pe": image_pe,
                "sparse_embeddings": sparse_embeddings,
                "dense_embeddings": dense_embeddings,
                "mask_tokens_out_all": mask_tokens_out_all,
                "pred_masks_all": pred_masks_all,
                "iou_scores_all": iou_scores_all,
            },
            name=name,
            **kwargs,
        )

        self.memory_attention = Sam2VideoMemoryAttention(
            hidden_size=self.FPN_HIDDEN_SIZE,
            kv_in_dim=self.MEM_DIM,
            num_layers=4,
            num_heads=1,
            ffn_hidden_size=2048,
            dropout=0.1,
            rope_theta=10000.0,
            rope_feat_sizes=[64, 64],
            name="memory_attention",
        )
        (
            self.mem_enc_ds_convs,
            self.mem_enc_ds_lns,
            self.mem_enc_ds_final_conv,
            self.mem_enc_feature_proj,
            self.mem_enc_fuser_dw_convs,
            self.mem_enc_fuser_lns,
            self.mem_enc_fuser_pw1s,
            self.mem_enc_fuser_pw2s,
            self.mem_enc_projection,
        ) = sam2_video_memory_encoder(self.FPN_HIDDEN_SIZE, self.MEM_DIM, "mem_enc")
        self.obj_ptr_proj_in, self.obj_ptr_proj_hidden_layers, self.obj_ptr_proj_out = (
            sam2_video_feed_forward(
                self.FPN_HIDDEN_SIZE,
                self.FPN_HIDDEN_SIZE,
                self.FPN_HIDDEN_SIZE,
                num_layers=3,
                name="obj_ptr_proj",
            )
        )
        self.mask_downsample_layer = layers.Conv2D(
            1,
            kernel_size=4,
            strides=4,
            padding="valid",
            data_format="channels_first",
            name="mask_downsample",
        )
        self.temporal_pos_enc_proj = layers.Dense(
            self.MEM_DIM,
            name="temporal_positional_encoding_projection_layer",
        )

        self.hidden_size = hidden_size
        self.blocks_per_stage = list(blocks_per_stage)
        self.embed_dim_per_stage = list(embed_dim_per_stage)
        self.num_attention_heads_per_stage = list(num_attention_heads_per_stage)
        self.window_size_per_stage = list(window_size_per_stage)
        self.global_attention_blocks = list(global_attention_blocks)
        self.backbone_channel_list = list(backbone_channel_list)
        self.window_pos_embed_bg_size = tuple(window_pos_embed_bg_size)
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

        self._build_video_params()

    def _build_video_params(self):
        self.no_memory_positional_encoding = self.add_weight(
            name="no_memory_positional_encoding",
            shape=(1, 1, self.FPN_HIDDEN_SIZE),
            initializer="zeros",
        )
        self.memory_temporal_positional_encoding = self.add_weight(
            name="memory_temporal_positional_encoding",
            shape=(self.NUM_MASKMEM, 1, 1, self.MEM_DIM),
            initializer="zeros",
        )
        self.no_object_pointer = self.add_weight(
            name="no_object_pointer",
            shape=(1, self.FPN_HIDDEN_SIZE),
            initializer="zeros",
        )
        self.occlusion_spatial_embedding_parameter = self.add_weight(
            name="occlusion_spatial_embedding_parameter",
            shape=(1, self.MEM_DIM),
            initializer="zeros",
        )

        dummy_mem = ops.zeros((1, 4096, self.MEM_DIM))
        dummy_q = ops.zeros((1, 4096, self.FPN_HIDDEN_SIZE))
        self.memory_attention(dummy_q, dummy_mem, training=False)

        self.mem_enc_fuser_scales = []
        for i in range(2):
            self.mem_enc_fuser_scales.append(
                self.add_weight(
                    name=f"mem_enc_fuser_{i}_scale",
                    shape=(self.FPN_HIDDEN_SIZE,),
                    initializer="zeros",
                )
            )

        dummy_feat = ops.zeros((1, self.FPN_HIDDEN_SIZE, 4, 4))
        dummy_mask = ops.zeros((1, 1, 64, 64))
        sam2_video_memory_encoder_call(
            dummy_feat,
            dummy_mask,
            self.mem_enc_ds_convs,
            self.mem_enc_ds_lns,
            self.mem_enc_ds_final_conv,
            self.mem_enc_feature_proj,
            self.mem_enc_fuser_dw_convs,
            self.mem_enc_fuser_lns,
            self.mem_enc_fuser_pw1s,
            self.mem_enc_fuser_pw2s,
            self.mem_enc_fuser_scales,
            self.mem_enc_projection,
            self.MEM_DIM // 2,
        )

        dummy_tok = ops.zeros((1, self.FPN_HIDDEN_SIZE))
        sam2_video_feed_forward_call(
            dummy_tok,
            self.obj_ptr_proj_in,
            self.obj_ptr_proj_hidden_layers,
            self.obj_ptr_proj_out,
        )

        dummy_ds_mask = ops.zeros((1, 1, 16, 16))
        self.mask_downsample_layer(dummy_ds_mask)

        dummy_pe = ops.zeros((1, self.FPN_HIDDEN_SIZE))
        self.temporal_pos_enc_proj(dummy_pe)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "blocks_per_stage": self.blocks_per_stage,
                "embed_dim_per_stage": self.embed_dim_per_stage,
                "num_attention_heads_per_stage": self.num_attention_heads_per_stage,
                "window_size_per_stage": self.window_size_per_stage,
                "global_attention_blocks": self.global_attention_blocks,
                "backbone_channel_list": self.backbone_channel_list,
                "window_pos_embed_bg_size": self.window_pos_embed_bg_size,
                "input_shape": self._input_shape_val,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_sam2_video_model(
    variant,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    config = SAM2_VIDEO_MODEL_CONFIG[variant]

    valid_model_weights = []
    if variant in SAM2_VIDEO_WEIGHTS_CONFIG:
        valid_model_weights = list(SAM2_VIDEO_WEIGHTS_CONFIG[variant].keys())

    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights for {variant} are "
            f"{', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )

    if input_shape is None:
        image_size = Sam2Video.IMAGE_SIZE
        df = keras.config.image_data_format()
        if df == "channels_first":
            input_shape = (3, image_size, image_size)
        else:
            input_shape = (image_size, image_size, 3)
        print(f"Using default input shape {input_shape}.")

    model = Sam2Video(
        hidden_size=config["hidden_size"],
        blocks_per_stage=config["blocks_per_stage"],
        embed_dim_per_stage=config["embed_dim_per_stage"],
        num_attention_heads_per_stage=config["num_attention_heads_per_stage"],
        window_size_per_stage=config["window_size_per_stage"],
        global_attention_blocks=config["global_attention_blocks"],
        backbone_channel_list=config["backbone_channel_list"],
        window_pos_embed_bg_size=config.get("window_pos_embed_bg_size"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **kwargs,
    )

    if weights in valid_model_weights:
        print(f"Loading {weights} weights for {variant}.")
        load_weights_from_config(variant, weights, model, SAM2_VIDEO_WEIGHTS_CONFIG)
    elif weights is not None and isinstance(weights, str):
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Sam2VideoTiny(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam2_video_model(
        "Sam2VideoTiny", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def Sam2VideoSmall(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam2_video_model(
        "Sam2VideoSmall", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def Sam2VideoBasePlus(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam2_video_model(
        "Sam2VideoBasePlus", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def Sam2VideoLarge(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam2_video_model(
        "Sam2VideoLarge", input_shape, input_tensor, weights, **kwargs
    )
