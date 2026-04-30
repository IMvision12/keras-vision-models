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
from kmodels.weight_utils import load_weights_from_config

from .config import SAM2_VIDEO_MODEL_CONFIG, SAM2_VIDEO_WEIGHTS_CONFIG
from .sam2_video_layers import Sam2VideoMemoryAttention


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2VideoLayerScale(layers.Layer):
    """Learnable per-channel scale used inside memory-fuser CX blocks.

    Holds a single ``(dim,)`` weight initialized to ``init_value`` and
    multiplies the last axis of the input tensor. This reproduces the
    ConvNeXt layer-scale trick used by the memory encoder's memory
    fuser.

    Args:
        dim (int): Number of channels; shape of the learned scale
            vector.
        init_value (float): Initial value for every element of the
            scale. Defaults to ``0.0``.
        **kwargs: Additional keyword arguments passed to the base
            ``Layer`` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(self, dim, init_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.init_value = init_value

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(self.dim,),
            initializer=keras.initializers.Constant(self.init_value),
        )

    def call(self, x):
        return self.scale * x

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "init_value": self.init_value})
        return config


def sam2_video_ffn(inputs, hidden_dim, output_dim, num_layers, name=""):
    """Multi-layer perceptron used by the object-pointer projection.

    Builds a feed-forward network of ``num_layers`` Dense layers with
    ReLU activations between hidden layers. Used to project the best
    mask token into the shared object-pointer space consumed by the
    memory attention cross-attention.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        inputs: Input tensor of shape ``(..., input_dim)``.
        hidden_dim: Integer, hidden dimensionality for all intermediate
            linear layers.
        output_dim: Integer, output dimensionality of the final linear
            layer.
        num_layers: Integer, total number of linear layers including the
            input and output projections.
        name: String, name prefix for the sub-layers. Defaults to ``""``.

    Returns:
        Output tensor of shape ``(..., output_dim)``.
    """
    x = layers.Dense(hidden_dim, name=f"{name}_proj_in")(inputs)
    x = layers.Activation("relu", name=f"{name}_relu_0")(x)
    for i in range(num_layers - 2):
        x = layers.Dense(hidden_dim, name=f"{name}_layers_{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu_{i + 1}")(x)
    x = layers.Dense(output_dim, name=f"{name}_proj_out")(x)
    return x


def sam2_video_mask_downsampler(
    inputs, embed_dim, data_format="channels_last", name=""
):
    """Downsample a predicted mask for the memory encoder.

    Progressively reduces a ``(1, 1024, 1024)`` mask to
    ``(embed_dim, 64, 64)`` through four stride-2 3x3 convolutions
    interleaved with layer normalization and GELU activations, then a
    final 1x1 projection to ``embed_dim`` channels.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        inputs: Input mask tensor. Layout matches ``data_format``.
        embed_dim: Integer, number of output channels after the final
            1x1 projection.
        data_format: String, one of ``"channels_first"`` or
            ``"channels_last"``. Defaults to ``"channels_last"``.
        name: String, name prefix for the sub-layers. Defaults to
            ``""``.

    Returns:
        Downsampled tensor in the same data format as the input.
    """
    x = inputs
    in_ch = 1
    for i in range(4):
        out_ch = in_ch * 4
        x = layers.ZeroPadding2D(
            padding=1, data_format=data_format, name=f"{name}_pad_{i}"
        )(x)
        x = layers.Conv2D(
            out_ch,
            3,
            strides=2,
            padding="valid",
            data_format=data_format,
            name=f"{name}_conv_{i}",
        )(x)
        if data_format == "channels_first":
            x = layers.Permute((2, 3, 1), name=f"{name}_permute_a_{i}")(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln_{i}")(x)
        if data_format == "channels_first":
            x = layers.Permute((3, 1, 2), name=f"{name}_permute_b_{i}")(x)
        x = layers.Activation("gelu", name=f"{name}_gelu_{i}")(x)
        in_ch = out_ch
    x = layers.Conv2D(
        embed_dim,
        1,
        padding="valid",
        data_format=data_format,
        name=f"{name}_final_conv",
    )(x)
    return x


def sam2_video_cx_block(
    inputs,
    embed_dim,
    intermediate_dim,
    kernel_size,
    padding,
    data_format="channels_last",
    name="",
):
    """ConvNeXt block used inside the memory-encoder fuser.

    Applies a depthwise spatial convolution followed by a two-layer
    pointwise MLP (with layer normalization and GELU activation),
    scales the residual path by a learned :class:`Sam2VideoLayerScale`,
    and adds the skip connection back onto the original input. When
    ``data_format`` is ``"channels_first"`` the tensor is permuted to
    channels-last for the LayerNorm/Dense/scale chain and permuted back.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        inputs: Input tensor. Layout matches ``data_format``.
        embed_dim: Integer, channel dimension of the input and output.
        intermediate_dim: Integer, hidden dimension of the two-layer
            pointwise MLP.
        kernel_size: Integer, spatial kernel size of the depthwise
            convolution.
        padding: Integer, symmetric zero padding applied before the
            depthwise convolution.
        data_format: String, one of ``"channels_first"`` or
            ``"channels_last"``. Defaults to ``"channels_last"``.
        name: String, name prefix for the sub-layers. Defaults to
            ``""``.

    Returns:
        Tensor of the same shape as ``inputs``.
    """
    residual = inputs
    x = layers.ZeroPadding2D(
        padding=padding, data_format=data_format, name=f"{name}_pad"
    )(inputs)
    x = layers.DepthwiseConv2D(
        kernel_size,
        padding="valid",
        data_format=data_format,
        name=f"{name}_dw_conv",
    )(x)
    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name=f"{name}_permute_a")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln")(x)
    x = layers.Dense(intermediate_dim, name=f"{name}_pw1")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu")(x)
    x = layers.Dense(embed_dim, name=f"{name}_pw2")(x)
    x = Sam2VideoLayerScale(embed_dim, init_value=0.0, name=f"{name}_scale")(x)
    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2), name=f"{name}_permute_b")(x)
    x = layers.Add(name=f"{name}_add")([residual, x])
    return x


def sam2_video_memory_fuser(
    inputs,
    num_blocks,
    embed_dim,
    intermediate_dim,
    kernel_size,
    padding,
    data_format="channels_last",
    name="",
):
    """Sequential stack of CX blocks inside the memory encoder.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        inputs: Input tensor. Layout matches ``data_format``.
        num_blocks: Integer, number of :func:`sam2_video_cx_block`
            blocks to stack.
        embed_dim: Integer, channel dimension of the input and output.
        intermediate_dim: Integer, hidden dimension passed through to
            each CX block's pointwise MLP.
        kernel_size: Integer, spatial kernel size of each block's
            depthwise convolution.
        padding: Integer, symmetric zero padding applied before each
            depthwise convolution.
        data_format: String, one of ``"channels_first"`` or
            ``"channels_last"``. Defaults to ``"channels_last"``.
        name: String, name prefix for the sub-blocks. Defaults to
            ``""``.

    Returns:
        Tensor of the same shape as ``inputs``.
    """
    x = inputs
    for i in range(num_blocks):
        x = sam2_video_cx_block(
            x,
            embed_dim,
            intermediate_dim,
            kernel_size,
            padding,
            data_format=data_format,
            name=f"{name}_{i}",
        )
    return x


def sam2_video_memory_encoder(
    vision_features,
    masks,
    hidden_size,
    output_channels,
    data_format="channels_last",
    name="",
):
    """Fuse backbone features with a predicted mask into memory features.

    Downsamples the high-resolution mask to the backbone feature
    resolution via :func:`sam2_video_mask_downsampler`, projects the
    backbone features with a 1x1 convolution, adds the two streams,
    refines the result with :func:`sam2_video_memory_fuser`, and finally
    projects to ``output_channels`` dimensions. The resulting tensor is
    stored in the per-object memory bank.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        vision_features: Backbone features. Layout matches
            ``data_format``.
        masks: Predicted mask tensor. Layout matches ``data_format``.
        hidden_size: Integer, internal channel dimension.
        output_channels: Integer, channel dimension of the returned
            memory features.
        data_format: String, one of ``"channels_first"`` or
            ``"channels_last"``. Defaults to ``"channels_last"``.
        name: String, name prefix for all sub-layers. Defaults to
            ``""``.

    Returns:
        Memory feature tensor in the same data format as the inputs.
    """
    mask_ds = sam2_video_mask_downsampler(
        masks, hidden_size, data_format=data_format, name=f"{name}_mask_ds"
    )
    vf = layers.Conv2D(
        hidden_size,
        1,
        padding="valid",
        data_format=data_format,
        name=f"{name}_feature_proj",
    )(vision_features)
    fused = layers.Add(name=f"{name}_fuse")([vf, mask_ds])
    fused = sam2_video_memory_fuser(
        fused,
        num_blocks=2,
        embed_dim=hidden_size,
        intermediate_dim=1024,
        kernel_size=7,
        padding=3,
        data_format=data_format,
        name=f"{name}_fuser",
    )
    output = layers.Conv2D(
        output_channels,
        1,
        padding="valid",
        data_format=data_format,
        name=f"{name}_projection",
    )(fused)
    return output


def sam2_video_sine_position_embedding(
    x, num_pos_feats, temperature=10000, data_format="channels_last"
):
    """Compute 2D sine-cosine positional embeddings for memory features.

    Produces the positional encoding paired with the memory features
    returned by :func:`sam2_video_memory_encoder`. Frequencies come in
    pairs following the original Transformer 1D sine encoding and are
    stacked across the ``y`` and ``x`` axes, giving a ``2 *
    num_pos_feats`` channel output.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        x: Reference tensor used to read the spatial dimensions.
            Layout must match ``data_format``.
        num_pos_feats: Integer, number of sine/cosine features per
            axis. The output channel dimension is ``2 * num_pos_feats``.
        temperature: Float, base for the inverse-frequency schedule.
            Defaults to ``10000``.
        data_format: String, one of ``"channels_first"`` or
            ``"channels_last"``. Determines both how the spatial dims
            are read from ``x`` and the layout of the returned tensor.
            Defaults to ``"channels_last"``.

    Returns:
        Positional embedding tensor of shape
        ``(1, 2 * num_pos_feats, H, W)`` when channels-first or
        ``(1, H, W, 2 * num_pos_feats)`` when channels-last.
    """
    scale = 2.0 * math.pi
    shape = ops.shape(x)
    if data_format == "channels_first":
        h, w = shape[2], shape[3]
    else:
        h, w = shape[1], shape[2]
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
    if data_format == "channels_first":
        pos = ops.transpose(pos, [2, 0, 1])
    return ops.expand_dims(pos, 0)


@keras.saving.register_keras_serializable(package="kmodels")
class Sam2Video(keras.Model):
    """Segment Anything Model 2 (Video) for promptable video segmentation.

    Extends the SAM 2 image model with the components required for
    temporal tracking:

    1. **Hiera Backbone + FPN** – identical to the image model, produces
       multi-scale image embeddings for the current frame.
    2. **Prompt Encoder** – encodes sparse (points, boxes) and dense
       (masks) prompts.
    3. **Mask Decoder** – two-way transformer that predicts masks, IoU
       scores, object-presence logits, and the mask tokens used later
       to derive the object pointer.
    4. **Memory Attention** – :class:`Sam2VideoMemoryAttention` stack
       that conditions the current-frame features on a bank of past
       frame memories and object pointers for non-cond frames.
    5. **Memory Encoder Sub-Model** – built via
       :func:`sam2_video_memory_encoder`, fuses backbone features with
       the predicted mask into memory features stored in the memory
       bank.
    6. **Object Pointer Projection Sub-Model** – built via
       :func:`sam2_video_ffn`, projects the selected mask token into a
       compact pointer consumed by the memory attention's
       cross-attention.

    The main functional graph exposes the single-frame outputs
    (``pred_masks``, ``iou_scores``, ``object_score_logits``) along
    with intermediate tensors the dynamic inference loop needs
    (``image_embeddings_raw``, ``high_res_feat_s0``, ``image_pe``, …).

    Reference:
        - `SAM 2: Segment Anything in Images and Videos
          <https://arxiv.org/abs/2408.00714>`_

    Args:
        hidden_size: Integer, initial hidden dimension of the Hiera
            backbone.
        blocks_per_stage: List of integers, number of transformer
            blocks per backbone stage.
        embed_dim_per_stage: List of integers, embedding dimension per
            backbone stage.
        num_attention_heads_per_stage: List of integers, number of
            attention heads per backbone stage.
        window_size_per_stage: List of integers, attention window size
            per backbone stage.
        global_attention_blocks: List of integers, absolute indices of
            blocks that use global attention.
        backbone_channel_list: List of integers, channel dimensions for
            the FPN lateral connections (high-to-low resolution).
        window_pos_embed_bg_size: Tuple of integers ``(H, W)``,
            background size for the windowed positional embeddings.
            Defaults to ``(7, 7)``.
        input_shape: Optional tuple specifying the input image shape.
            Defaults to ``(1024, 1024, 3)`` (channels-last) or
            ``(3, 1024, 1024)`` (channels-first).
        input_tensor: Optional Keras tensor to use as the model input.
        name: String, the name of the model. Defaults to
            ``"Sam2Video"``.
        **kwargs: Additional keyword arguments passed to the
            ``keras.Model`` class.

    Returns:
        A ``keras.Model`` instance with dict outputs:

        - ``"pred_masks"``: ``(batch, num_prompts, 3, 256, 256)``
        - ``"iou_scores"``: ``(batch, num_prompts, 3)``
        - ``"object_score_logits"``: ``(batch, num_prompts, 1)``
        - ``"image_embeddings_raw"``, ``"image_embeddings"``,
          ``"high_res_feat_s0"``, ``"high_res_feat_s1"``,
          ``"image_pe"``, ``"sparse_embeddings"``,
          ``"dense_embeddings"``, ``"mask_tokens_out_all"``,
          ``"pred_masks_all"``, ``"iou_scores_all"`` — intermediate
          tensors consumed by the video inference loop.

    Example:
        ```python
        model = kmodels.models.sam2_video.Sam2VideoSmall(
            input_shape=(1024, 1024, 3),
        )
        ```
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
    MEM_FEATURE_SIZE = 64

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

        if data_format == "channels_first":
            mem_vf_shape = (
                self.FPN_HIDDEN_SIZE,
                self.MEM_FEATURE_SIZE,
                self.MEM_FEATURE_SIZE,
            )
            mem_mask_shape = (1, self.IMAGE_SIZE, self.IMAGE_SIZE)
        else:
            mem_vf_shape = (
                self.MEM_FEATURE_SIZE,
                self.MEM_FEATURE_SIZE,
                self.FPN_HIDDEN_SIZE,
            )
            mem_mask_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        mem_vf_in = layers.Input(shape=mem_vf_shape, name="mem_enc_vf_in")
        mem_mask_in = layers.Input(shape=mem_mask_shape, name="mem_enc_mask_in")
        mem_output = sam2_video_memory_encoder(
            mem_vf_in,
            mem_mask_in,
            hidden_size=self.FPN_HIDDEN_SIZE,
            output_channels=self.MEM_DIM,
            data_format=data_format,
            name="mem_enc",
        )
        self.memory_encoder_submodel = keras.Model(
            inputs=[mem_vf_in, mem_mask_in],
            outputs=mem_output,
            name="memory_encoder",
        )

        ptr_in = layers.Input(shape=(self.FPN_HIDDEN_SIZE,), name="obj_ptr_in")
        ptr_output = sam2_video_ffn(
            ptr_in,
            hidden_dim=self.FPN_HIDDEN_SIZE,
            output_dim=self.FPN_HIDDEN_SIZE,
            num_layers=3,
            name="obj_ptr_proj",
        )
        self.obj_ptr_proj_submodel = keras.Model(
            inputs=ptr_in, outputs=ptr_output, name="obj_ptr_proj"
        )

        self.mask_downsample_layer = layers.Conv2D(
            1,
            kernel_size=4,
            strides=4,
            padding="valid",
            data_format=data_format,
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

        if keras.config.image_data_format() == "channels_first":
            dummy_ds_mask = ops.zeros((1, 1, 16, 16))
        else:
            dummy_ds_mask = ops.zeros((1, 16, 16, 1))
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
    """Factory function for creating Sam2Video model variants.

    Looks up the architecture configuration for the given variant
    name, instantiates a :class:`Sam2Video` model, and optionally
    loads pretrained weights from the configured URL or a local
    file path.

    Args:
        variant: String, model variant name (e.g., ``"Sam2VideoSmall"``).
        input_shape: Optional tuple specifying the input image shape.
            Defaults to ``(1024, 1024, 3)`` (channels-last) or
            ``(3, 1024, 1024)`` (channels-first).
        input_tensor: Optional Keras tensor to use as the model input.
        weights: One of ``None`` (random initialization), a weight
            identifier from ``SAM2_VIDEO_WEIGHTS_CONFIG`` (e.g.,
            ``"sav"``), or a path to a weights file.
        **kwargs: Additional keyword arguments passed to the
            :class:`Sam2Video` constructor.

    Returns:
        A configured :class:`Sam2Video` instance.
    """
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
    weights="sav",
    **kwargs,
):
    """SAM 2 Video Tiny variant (Hiera tiny backbone)."""
    return _create_sam2_video_model(
        "Sam2VideoTiny", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def Sam2VideoSmall(
    input_shape=None,
    input_tensor=None,
    weights="sav",
    **kwargs,
):
    """SAM 2 Video Small variant (Hiera small backbone)."""
    return _create_sam2_video_model(
        "Sam2VideoSmall", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def Sam2VideoBasePlus(
    input_shape=None,
    input_tensor=None,
    weights="sav",
    **kwargs,
):
    """SAM 2 Video Base-Plus variant (Hiera base-plus backbone)."""
    return _create_sam2_video_model(
        "Sam2VideoBasePlus", input_shape, input_tensor, weights, **kwargs
    )


@register_model
def Sam2VideoLarge(
    input_shape=None,
    input_tensor=None,
    weights="sav",
    **kwargs,
):
    """SAM 2 Video Large variant (Hiera large backbone)."""
    return _create_sam2_video_model(
        "Sam2VideoLarge", input_shape, input_tensor, weights, **kwargs
    )
