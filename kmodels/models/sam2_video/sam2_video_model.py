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
from .sam2_video_layers import (
    Sam2VideoFeedForward,
    Sam2VideoMemoryAttention,
    Sam2VideoMemoryEncoder,
)


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2Video(keras.Model):
    """Segment Anything Model 2 for video segmentation.

    Extends SAM2 with memory-conditioned attention for temporal tracking.
    The architecture adds a memory attention module (4-layer RoPE transformer),
    a memory encoder (ConvNeXt fuser), and object pointer tokens on top of the
    standard Hiera backbone, prompt encoder, and mask decoder.

    For initial frames (no prior memory), the model behaves identically to
    SAM2 image segmentation. For subsequent frames, memory features from
    previous predictions condition the current frame's features via
    cross-attention.

    Reference:
    - [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)

    Args:
        hidden_size: Integer, initial Hiera hidden dimension.
        blocks_per_stage: List of integers, blocks per backbone stage.
        embed_dim_per_stage: List of integers, embedding dims per stage.
        num_attention_heads_per_stage: List of integers, heads per stage.
        window_size_per_stage: List of integers, window sizes per stage.
        global_attention_blocks: List of integers, global attention block indices.
        backbone_channel_list: List of integers, FPN channel dimensions.
        window_pos_embed_bg_size: Tuple of integers, positional embedding
            background size. Defaults to ``(7, 7)``.
        input_shape: Optional tuple for input shape.
        input_tensor: Optional Keras tensor as input.
        name: String, model name. Defaults to ``"Sam2Video"``.

    Returns:
        A Keras ``Model`` instance with dict outputs:
        ``pred_masks``, ``iou_scores``, ``object_score_logits``.
    """

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

        pred_masks = decoder_output["pred_masks"][:, :, 1:, :, :]
        iou_scores = decoder_output["iou_scores"][:, :, 1:]
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
        self.memory_encoder = Sam2VideoMemoryEncoder(
            hidden_size=self.FPN_HIDDEN_SIZE,
            output_channels=self.MEM_DIM,
            name="memory_encoder",
        )
        self.object_pointer_proj = Sam2VideoFeedForward(
            input_dim=self.FPN_HIDDEN_SIZE,
            hidden_dim=self.FPN_HIDDEN_SIZE,
            output_dim=self.FPN_HIDDEN_SIZE,
            num_layers=3,
            name="object_pointer_proj",
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

        dummy_feat = ops.zeros((1, self.FPN_HIDDEN_SIZE, 4, 4))
        dummy_mask = ops.zeros((1, 1, 64, 64))
        self.memory_encoder(dummy_feat, dummy_mask)

        dummy_tok = ops.zeros((1, self.FPN_HIDDEN_SIZE))
        self.object_pointer_proj(dummy_tok)

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
    """
    Instantiates SAM2 Video Tiny (~38 M params).

    Hiera-Tiny backbone with memory attention for video segmentation.

    Reference:
    - [SAM 2](https://arxiv.org/abs/2408.00714)

    Args:
        input_shape: Optional tuple for input shape.
        input_tensor: Optional Keras tensor as input.
        weights: String, ``"sam2_video"`` for HF weights, file path, or None.

    Returns:
        A Keras ``Model`` instance.
    """
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
    """
    Instantiates SAM2 Video Small (~46 M params).

    Hiera-Small backbone with memory attention for video segmentation.

    Reference:
    - [SAM 2](https://arxiv.org/abs/2408.00714)

    Args:
        input_shape: Optional tuple for input shape.
        input_tensor: Optional Keras tensor as input.
        weights: String, ``"sam2_video"`` for HF weights, file path, or None.

    Returns:
        A Keras ``Model`` instance.
    """
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
    """
    Instantiates SAM2 Video Base+ (~81 M params).

    Hiera-Base+ backbone with memory attention for video segmentation.

    Reference:
    - [SAM 2](https://arxiv.org/abs/2408.00714)

    Args:
        input_shape: Optional tuple for input shape.
        input_tensor: Optional Keras tensor as input.
        weights: String, ``"sam2_video"`` for HF weights, file path, or None.

    Returns:
        A Keras ``Model`` instance.
    """
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
    """
    Instantiates SAM2 Video Large (~224 M params).

    Hiera-Large backbone with memory attention for video segmentation.

    Reference:
    - [SAM 2](https://arxiv.org/abs/2408.00714)

    Args:
        input_shape: Optional tuple for input shape.
        input_tensor: Optional Keras tensor as input.
        weights: String, ``"sam2_video"`` for HF weights, file path, or None.

    Returns:
        A Keras ``Model`` instance.
    """
    return _create_sam2_video_model(
        "Sam2VideoLarge", input_shape, input_tensor, weights, **kwargs
    )
