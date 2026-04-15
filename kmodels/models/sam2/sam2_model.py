import keras
import numpy as np
from keras import layers, utils

from kmodels.model_registry import register_model
from kmodels.weight_utils import load_weights_from_config

from .config import SAM2_MODEL_CONFIG, SAM2_WEIGHTS_CONFIG
from .sam2_layers import (
    SAM2HieraPositionEmbedding,
    SAM2ImagePositionalEmbeddings,
    SAM2MaskDecoderLayer,
    SAM2MultiScaleBlock,
    SAM2NoMemoryEmbedding,
    SAM2PositionalEmbedding,
    SAM2PromptEncoderLayer,
)


def _dynamic_multimask_via_stability(
    all_mask_logits,
    all_iou_scores,
    stability_delta,
    stability_thresh,
):
    """Pick between the single-mask and best multi-mask outputs.

    Mirrors HF's ``Sam2MaskDecoder._dynamic_multimask_via_stability``
    (``transformers/models/sam2/modeling_sam2.py``):

    1. Compute the stability score of the single-mask output
       (decoder token 0) as ``|mask > +delta| / |mask > -delta|``
       over the spatial dims.
    2. If the stability score meets or exceeds ``stability_thresh``
       (default ``0.98``), keep the single-mask output.
    3. Otherwise fall back to the argmax-IoU multi-mask output
       across decoder tokens 1..N.

    Used only when ``multimask_output=False`` at model-build time.
    Returns ``(pred_masks, iou_scores)`` both shaped
    ``(batch, point_batch, 1, H, W)`` / ``(batch, point_batch, 1)``.
    """
    ops_ = keras.ops
    multimask_logits = all_mask_logits[:, :, 1:, :, :]
    multimask_iou = all_iou_scores[:, :, 1:]
    best_idx = ops_.argmax(multimask_iou, axis=-1)
    num_multimask = ops_.shape(multimask_iou)[-1]
    best_one_hot = ops_.one_hot(best_idx, num_multimask, dtype=multimask_iou.dtype)
    best_multimask_iou = ops_.sum(multimask_iou * best_one_hot, axis=-1, keepdims=True)
    best_gather = ops_.reshape(
        best_one_hot,
        (
            ops_.shape(best_one_hot)[0],
            ops_.shape(best_one_hot)[1],
            num_multimask,
            1,
            1,
        ),
    )
    best_multimask_logits = ops_.sum(
        multimask_logits * best_gather, axis=2, keepdims=True
    )

    singlemask_logits = all_mask_logits[:, :, 0:1, :, :]
    singlemask_iou = all_iou_scores[:, :, 0:1]

    flat_single = ops_.reshape(
        singlemask_logits,
        (
            ops_.shape(singlemask_logits)[0],
            ops_.shape(singlemask_logits)[1],
            ops_.shape(singlemask_logits)[2],
            -1,
        ),
    )
    area_i = ops_.sum(
        ops_.cast(flat_single > stability_delta, dtype="float32"), axis=-1
    )
    area_u = ops_.sum(
        ops_.cast(flat_single > -stability_delta, dtype="float32"), axis=-1
    )
    stability_scores = ops_.where(area_u > 0, area_i / ops_.maximum(area_u, 1.0), 1.0)
    is_stable = stability_scores >= stability_thresh

    is_stable_mask = ops_.cast(
        ops_.reshape(
            is_stable,
            (ops_.shape(is_stable)[0], ops_.shape(is_stable)[1], 1, 1, 1),
        ),
        dtype=singlemask_logits.dtype,
    )
    mask_out = (
        is_stable_mask * singlemask_logits
        + (1.0 - is_stable_mask) * best_multimask_logits
    )

    is_stable_iou = ops_.cast(
        ops_.reshape(
            is_stable,
            (ops_.shape(is_stable)[0], ops_.shape(is_stable)[1], 1),
        ),
        dtype=singlemask_iou.dtype,
    )
    iou_out = (
        is_stable_iou * singlemask_iou + (1.0 - is_stable_iou) * best_multimask_iou
    )
    return mask_out, iou_out


def sam2_mask_embedding(
    inputs, hidden_size=256, mask_input_channels=16, layer_norm_eps=1e-6, name=""
):
    """Embeds dense mask prompts through a small convolutional network.

    Downsamples a single-channel mask input by 4x total through
    three Conv2D layers with GELU activations and layer
    normalization, mapping it to ``hidden_size`` channels at the
    image embedding spatial resolution.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        inputs: Input mask tensor of shape
            ``(batch_size, 4*H, 4*W, 1)``.
        hidden_size: Integer, output embedding dimension.
            Defaults to ``256``.
        mask_input_channels: Integer, intermediate channel count
            after the second convolution.
            Defaults to ``16``.
        layer_norm_eps: Float, epsilon for layer normalization.
            Defaults to ``1e-6``.
        name: String, name prefix for all sub-layers.
            Defaults to ``""``.

    Returns:
        Dense embedding tensor of shape
        ``(batch_size, H, W, hidden_size)``.
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
class SAM2(keras.Model):
    """Segment Anything Model 2 (SAM2) for promptable image segmentation.

    SAM2 treats image segmentation as a promptable task, producing
    high-quality masks from flexible user inputs. The architecture
    consists of three components:

    1. **Hiera Backbone** – a hierarchical ViT with multi-scale
       blocks, windowed attention, query pooling at stage transitions,
       and windowed positional embeddings. An FPN neck produces
       multi-scale feature maps with sine-cosine positional encodings.
    2. **Prompt Encoder** – encodes sparse prompts (points, boxes)
       via Fourier positional encoding with learned type embeddings,
       and dense prompts (masks) via a small CNN.
    3. **Mask Decoder** – a lightweight two-way transformer that
       jointly attends between prompt tokens and image embeddings,
       then predicts multiple segmentation masks, IoU quality scores,
       and object-presence scores via hypernetwork MLPs. High-resolution
       feature skip connections from the FPN improve mask quality.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        hidden_size: Integer, initial hidden dimension of the Hiera
            backbone. Defaults to ``96``.
        blocks_per_stage: List of integers, number of transformer
            blocks per stage. Defaults to ``[1, 2, 7, 2]``.
        embed_dim_per_stage: List of integers, embedding dimension
            per stage. Defaults to ``[96, 192, 384, 768]``.
        num_attention_heads_per_stage: List of integers, number of
            attention heads per stage. Defaults to ``[1, 2, 4, 8]``.
        window_size_per_stage: List of integers, window size per
            stage. Defaults to ``[8, 4, 14, 7]``.
        global_attention_blocks: List of integers, absolute block
            indices that use global attention.
            Defaults to ``[5, 7, 9]``.
        backbone_channel_list: List of integers, channel dimensions
            for FPN lateral connections (high-to-low resolution).
            Defaults to ``[768, 384, 192, 96]``.
        window_pos_embed_bg_size: Tuple of integers ``(H, W)``,
            background size for windowed positional embeddings.
            Defaults to ``(7, 7)``.
        num_multimask_outputs: Integer, number of mask outputs
            beyond the single-mask token. Defaults to ``3``.
        input_shape: Optional tuple of integers specifying the
            input image shape ``(H, W, C)``. Defaults to
            ``(1024, 1024, 3)``.
        input_tensor: Optional Keras tensor to use as the model
            input.
        name: String, the name of the model.
            Defaults to ``"SAM2"``.
        **kwargs: Additional keyword arguments passed to the
            ``keras.Model`` class.

    Returns:
        A ``keras.Model`` instance with dict outputs:
        - ``"pred_masks"``: ``(batch_size, num_prompts,
          num_multimask_outputs, 4*H, 4*W)``
        - ``"iou_scores"``: ``(batch_size, num_prompts,
          num_multimask_outputs)``
        - ``"object_score_logits"``: ``(batch_size, num_prompts, 1)``

    Example:
        ```python
        model = kmodels.models.sam2.Sam2Tiny(
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

    DYNAMIC_MULTIMASK_STABILITY_DELTA = 0.05
    DYNAMIC_MULTIMASK_STABILITY_THRESH = 0.98

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
        num_multimask_outputs=3,
        include_box_input=False,
        include_mask_input=False,
        multimask_output=True,
        input_shape=None,
        input_tensor=None,
        name="SAM2",
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
        input_boxes = None
        if include_box_input:
            input_boxes = layers.Input(
                shape=(None, 4), name="input_boxes", dtype="float32"
            )
        input_masks = None
        has_input_masks = None
        if data_format == "channels_first":
            mask_input_size = input_shape[1] // 4
        else:
            mask_input_size = input_shape[0] // 4
        if include_mask_input:
            if data_format == "channels_first":
                input_masks = layers.Input(
                    shape=(1, mask_input_size, mask_input_size),
                    name="input_masks",
                    dtype="float32",
                )
            else:
                input_masks = layers.Input(
                    shape=(mask_input_size, mask_input_size, 1),
                    name="input_masks",
                    dtype="float32",
                )
            has_input_masks = layers.Input(
                shape=(), name="has_input_masks", dtype="float32"
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

        image_pe_layer = SAM2ImagePositionalEmbeddings(
            image_embedding_size,
            shared_image_embedding,
            name="image_positional_embeddings",
        )
        image_pe = image_pe_layer(image_embeddings)

        prompt_inputs = [input_points, input_labels]
        if include_box_input:
            prompt_inputs.append(input_boxes)
        if include_mask_input:
            if not include_box_input:
                empty_boxes = layers.Lambda(
                    lambda p: keras.ops.zeros(
                        (keras.ops.shape(p)[0], keras.ops.shape(p)[1], 0, 4),
                        dtype="float32",
                    ),
                    name="empty_box_placeholder",
                )(input_points)
                prompt_inputs.append(empty_boxes)
            mask_dense = sam2_mask_embedding(
                input_masks,
                hidden_size=self.PROMPT_ENCODER_HIDDEN_SIZE,
                mask_input_channels=self.PROMPT_ENCODER_MASK_INPUT_CHANNELS,
                layer_norm_eps=self.LAYER_NORM_EPS,
                name="prompt_encoder_mask_embed",
            )
            prompt_inputs.append(mask_dense)
            prompt_inputs.append(has_input_masks)

        prompt_encoder_layer = SAM2PromptEncoderLayer(
            hidden_size=self.PROMPT_ENCODER_HIDDEN_SIZE,
            image_embedding_size=image_embedding_size,
            image_size=self.IMAGE_SIZE,
            num_point_embeddings=self.PROMPT_ENCODER_NUM_POINT_EMBEDDINGS,
            shared_embedding=shared_image_embedding,
            data_format=data_format,
            name="prompt_encoder",
        )
        prompt_results = prompt_encoder_layer(prompt_inputs)

        sparse_embeddings = prompt_results["sparse_embeddings"]
        dense_embeddings = prompt_results["dense_embeddings"]

        mask_decoder_layer = SAM2MaskDecoderLayer(
            hidden_size=self.MASK_DECODER_HIDDEN_SIZE,
            num_hidden_layers=self.MASK_DECODER_NUM_HIDDEN_LAYERS,
            num_attention_heads=self.MASK_DECODER_NUM_ATTENTION_HEADS,
            mlp_dim=self.MASK_DECODER_MLP_DIM,
            num_multimask_outputs=num_multimask_outputs,
            iou_head_depth=self.MASK_DECODER_IOU_HEAD_DEPTH,
            iou_head_hidden_dim=self.MASK_DECODER_IOU_HEAD_HIDDEN_DIM,
            attention_downsample_rate=self.MASK_DECODER_ATTENTION_DOWNSAMPLE_RATE,
            data_format=data_format,
            name="mask_decoder",
        )
        decoder_output = mask_decoder_layer(
            [
                image_embeddings,
                image_pe,
                sparse_embeddings,
                dense_embeddings,
                high_res_feat_s0,
                high_res_feat_s1,
            ]
        )

        all_masks = decoder_output["pred_masks"]
        all_iou = decoder_output["iou_scores"]
        object_score_logits = decoder_output["object_score_logits"]

        if multimask_output:
            pred_masks = all_masks[:, :, 1:, :, :]
            iou_scores = all_iou[:, :, 1:]
        else:

            def _select_best_mask(tensors):
                masks, iou = tensors
                return _dynamic_multimask_via_stability(
                    masks,
                    iou,
                    self.DYNAMIC_MULTIMASK_STABILITY_DELTA,
                    self.DYNAMIC_MULTIMASK_STABILITY_THRESH,
                )

            pred_masks, iou_scores = layers.Lambda(
                _select_best_mask,
                name="dynamic_multimask_via_stability",
            )([all_masks, all_iou])

        model_inputs = {
            "pixel_values": pixel_values,
            "input_points": input_points,
            "input_labels": input_labels,
        }
        if include_box_input:
            model_inputs["input_boxes"] = input_boxes
        if include_mask_input:
            model_inputs["input_masks"] = input_masks
            model_inputs["has_input_masks"] = has_input_masks

        super().__init__(
            inputs=model_inputs,
            outputs={
                "pred_masks": pred_masks,
                "iou_scores": iou_scores,
                "object_score_logits": object_score_logits,
            },
            name=name,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.blocks_per_stage = list(blocks_per_stage)
        self.embed_dim_per_stage = list(embed_dim_per_stage)
        self.num_attention_heads_per_stage = list(num_attention_heads_per_stage)
        self.window_size_per_stage = list(window_size_per_stage)
        self.global_attention_blocks = list(global_attention_blocks)
        self.backbone_channel_list = list(backbone_channel_list)
        self.window_pos_embed_bg_size = tuple(window_pos_embed_bg_size)
        self.num_multimask_outputs = num_multimask_outputs
        self.include_box_input = include_box_input
        self.include_mask_input = include_mask_input
        self.multimask_output = multimask_output
        self.image_embedding_size = image_embedding_size
        self.mask_input_size = mask_input_size
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

        self._image_pe_layer = image_pe_layer
        self._prompt_encoder_layer = prompt_encoder_layer
        self._mask_decoder_layer = mask_decoder_layer

        self._build_submodels(
            pixel_values=pixel_values,
            image_embeddings=image_embeddings,
            high_res_feat_s0=high_res_feat_s0,
            high_res_feat_s1=high_res_feat_s1,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
            has_input_masks=has_input_masks,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            name=name,
        )

    def _build_submodels(
        self,
        pixel_values,
        image_embeddings,
        high_res_feat_s0,
        high_res_feat_s1,
        input_points,
        input_labels,
        input_boxes,
        input_masks,
        has_input_masks,
        sparse_embeddings,
        dense_embeddings,
        name,
    ):
        self.vision_encoder_model = keras.Model(
            inputs=pixel_values,
            outputs={
                "image_embeddings": image_embeddings,
                "high_res_feat_s0": high_res_feat_s0,
                "high_res_feat_s1": high_res_feat_s1,
            },
            name=f"{name}_vision_encoder",
        )

        def _drop_batch(shape):
            return tuple(shape[1:])

        img_emb_in = layers.Input(
            shape=_drop_batch(image_embeddings.shape),
            dtype=image_embeddings.dtype,
            name="image_embeddings",
        )
        hr_s0_in = layers.Input(
            shape=_drop_batch(high_res_feat_s0.shape),
            dtype=high_res_feat_s0.dtype,
            name="high_res_feat_s0",
        )
        hr_s1_in = layers.Input(
            shape=_drop_batch(high_res_feat_s1.shape),
            dtype=high_res_feat_s1.dtype,
            name="high_res_feat_s1",
        )

        decoder_image_pe = self._image_pe_layer(img_emb_in)

        decoder_output_sub = self._mask_decoder_layer(
            [
                img_emb_in,
                decoder_image_pe,
                sparse_embeddings,
                dense_embeddings,
                hr_s0_in,
                hr_s1_in,
            ]
        )

        all_masks_sub = decoder_output_sub["pred_masks"]
        all_iou_sub = decoder_output_sub["iou_scores"]
        obj_logits_sub = decoder_output_sub["object_score_logits"]

        if self.multimask_output:
            sub_pred_masks = all_masks_sub[:, :, 1:, :, :]
            sub_iou = all_iou_sub[:, :, 1:]
        else:
            stab_delta = self.DYNAMIC_MULTIMASK_STABILITY_DELTA
            stab_thresh = self.DYNAMIC_MULTIMASK_STABILITY_THRESH

            def _select_best_mask_sub(tensors):
                masks, iou = tensors
                return _dynamic_multimask_via_stability(
                    masks, iou, stab_delta, stab_thresh
                )

            sub_pred_masks, sub_iou = layers.Lambda(
                _select_best_mask_sub,
                name="dynamic_multimask_via_stability_sub",
            )([all_masks_sub, all_iou_sub])

        sub_inputs = {
            "image_embeddings": img_emb_in,
            "high_res_feat_s0": hr_s0_in,
            "high_res_feat_s1": hr_s1_in,
            "input_points": input_points,
            "input_labels": input_labels,
        }
        if self.include_box_input:
            sub_inputs["input_boxes"] = input_boxes
        if self.include_mask_input:
            sub_inputs["input_masks"] = input_masks
            sub_inputs["has_input_masks"] = has_input_masks

        self.prompt_decoder_model = keras.Model(
            inputs=sub_inputs,
            outputs={
                "pred_masks": sub_pred_masks,
                "iou_scores": sub_iou,
                "object_score_logits": obj_logits_sub,
            },
            name=f"{name}_prompt_decoder",
        )

    def get_image_embeddings(self, pixel_values):
        return self.vision_encoder_model(pixel_values)

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
                "num_multimask_outputs": self.num_multimask_outputs,
                "include_box_input": self.include_box_input,
                "include_mask_input": self.include_mask_input,
                "multimask_output": self.multimask_output,
                "input_shape": self._input_shape_val,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_sam2_model(
    variant,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    """Factory function for creating SAM2 model variants.

    Looks up the architecture configuration for the given variant
    name, instantiates a ``SAM2`` model, and optionally loads
    pretrained weights from the configured URL or a local file
    path.

    Args:
        variant: String, model variant name (e.g.,
            ``"Sam2Tiny"``).
        input_shape: Optional tuple of integers specifying the
            input shape ``(H, W, C)``.
        input_tensor: Optional Keras tensor to use as the model
            input.
        weights: String, one of ``None`` (random initialization),
            a weight identifier from the config, or a path to a
            weights file.
        **kwargs: Additional keyword arguments passed to the
            ``SAM2`` constructor.

    Returns:
        A configured ``SAM2`` model instance.
    """
    config = SAM2_MODEL_CONFIG[variant]

    valid_model_weights = []
    if variant in SAM2_WEIGHTS_CONFIG:
        valid_model_weights = list(SAM2_WEIGHTS_CONFIG[variant].keys())

    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights for {variant} are "
            f"{', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )

    if input_shape is None:
        image_size = SAM2.IMAGE_SIZE
        df = keras.config.image_data_format()
        if df == "channels_first":
            input_shape = (3, image_size, image_size)
        else:
            input_shape = (image_size, image_size, 3)
        print(f"Using default input shape {input_shape}.")

    model = SAM2(
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
        load_weights_from_config(variant, weights, model, SAM2_WEIGHTS_CONFIG)
    elif weights is not None and isinstance(weights, str):
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Sam2Tiny(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return create_sam2_model(
        "Sam2Tiny",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def Sam2Small(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return create_sam2_model(
        "Sam2Small",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def Sam2BasePlus(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return create_sam2_model(
        "Sam2BasePlus",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def Sam2Large(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return create_sam2_model(
        "Sam2Large",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )
