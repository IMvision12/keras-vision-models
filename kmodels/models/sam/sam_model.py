import keras
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import SAM_MODEL_CONFIG, SAM_WEIGHTS_CONFIG
from .sam_layers import (
    SAMFeedForward,
    SAMPatchEmbeddings,
    SAMPositionalEmbedding,
    SAMTwoWayAttention,
    SAMTwoWayAttentionBlock,
    SAMVisionLayer,
    SAMVisionNeck,
)


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
        patch_embed = SAMPatchEmbeddings(
            vision_hidden_size,
            patch_size=vision_patch_size,
            name="vision_encoder_patch_embed",
        )
        hidden_states = patch_embed(pixel_values)

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

        neck = SAMVisionNeck(
            vision_hidden_size,
            vision_output_channels,
            name="vision_encoder_neck",
        )
        image_embeddings = neck(hidden_states)

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


@keras.saving.register_keras_serializable(package="kmodels")
class SAMAbsolutePositionEmbedding(layers.Layer):
    """Adds learnable absolute position embeddings to patch embeddings.

    Args:
        hidden_size: Embedding dimension.
        image_embedding_size: Spatial size of the patch grid.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, image_embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_embedding_size = image_embedding_size

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(
                1,
                self.image_embedding_size,
                self.image_embedding_size,
                self.hidden_size,
            ),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, hidden_states):
        return hidden_states + self.pos_embed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "image_embedding_size": self.image_embedding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMImagePositionalEmbeddings(layers.Layer):
    """Generates grid-based positional embeddings for the image feature map.

    Creates a ``(1, H, W, C)`` positional embedding tensor using the shared
    Fourier embedding layer.

    Args:
        image_embedding_size: Spatial size of the image embedding grid.
        shared_embedding: A ``SAMPositionalEmbedding`` layer instance.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, image_embedding_size, shared_embedding, **kwargs):
        super().__init__(**kwargs)
        self.image_embedding_size = image_embedding_size
        self.shared_embedding = shared_embedding

    def call(self, image_embeddings):
        size = self.image_embedding_size
        grid = ops.ones((size, size), dtype="float32")
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        coords = ops.stack([x_embed, y_embed], axis=-1)
        pe = self.shared_embedding(coords)
        pe = ops.transpose(pe, (2, 0, 1))
        pe = ops.expand_dims(pe, axis=0)
        return pe

    def get_config(self):
        config = super().get_config()
        config.update({"image_embedding_size": self.image_embedding_size})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMPromptEncoderLayer(layers.Layer):
    """Encodes sparse (point/box) and dense (mask) prompts.

    Sparse prompts are converted to embedding vectors via positional encoding
    and learned type embeddings.  Dense prompts (masks) are downsampled through
    a small CNN.  When no mask is provided, a learned ``no_mask_embed`` is
    broadcast to the image embedding spatial size.

    Args:
        hidden_size: Output embedding dimension.
        image_embedding_size: Spatial size of the image embedding grid.
        image_size: Original input image resolution.
        mask_input_channels: Intermediate channels for the mask CNN.
        num_point_embeddings: Number of point type embeddings (typically 4).
        shared_embedding: A ``SAMPositionalEmbedding`` for coordinate encoding.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size=256,
        image_embedding_size=64,
        image_size=1024,
        num_point_embeddings=4,
        shared_embedding=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_embedding_size = image_embedding_size
        self.image_size = image_size
        self.num_point_embeddings = num_point_embeddings
        self.shared_embedding = shared_embedding

    def build(self, input_shape):
        self.point_embeddings = []
        for i in range(self.num_point_embeddings):
            w = self.add_weight(
                name=f"point_embed_{i}",
                shape=(1, self.hidden_size),
                initializer="zeros",
                trainable=True,
            )
            self.point_embeddings.append(w)

        self.not_a_point_embed = self.add_weight(
            name="not_a_point_embed",
            shape=(1, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        self.no_mask_embed = self.add_weight(
            name="no_mask_embed",
            shape=(1, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def _embed_points(self, points, point_labels, pad):
        points = points + 0.5
        if pad:
            batch_size = ops.shape(points)[0]
            point_batch_size = ops.shape(points)[1]
            padding_point = ops.zeros(
                (batch_size, point_batch_size, 1, 2), dtype="float32"
            )
            padding_label = -ops.ones((batch_size, point_batch_size, 1), dtype="int32")
            points = ops.concatenate([points, padding_point], axis=2)
            point_labels = ops.concatenate([point_labels, padding_label], axis=2)

        coords_0 = points[..., 0] / float(self.image_size)
        coords_1 = points[..., 1] / float(self.image_size)
        normalized_coords = ops.stack([coords_0, coords_1], axis=-1)
        point_embedding = self.shared_embedding(normalized_coords)

        point_labels_expanded = ops.expand_dims(
            ops.cast(point_labels, "float32"), axis=-1
        )

        not_a_point = ops.broadcast_to(
            self.not_a_point_embed, ops.shape(point_embedding)
        )
        point_embedding = ops.where(
            point_labels_expanded == -1.0, not_a_point, point_embedding
        )

        zeros = ops.zeros_like(point_embedding)
        point_embedding = ops.where(
            point_labels_expanded == -10.0, zeros, point_embedding
        )

        pe_0 = ops.broadcast_to(self.point_embeddings[0], ops.shape(point_embedding))
        point_embedding = ops.where(
            point_labels_expanded == 0.0, point_embedding + pe_0, point_embedding
        )
        pe_1 = ops.broadcast_to(self.point_embeddings[1], ops.shape(point_embedding))
        point_embedding = ops.where(
            point_labels_expanded == 1.0, point_embedding + pe_1, point_embedding
        )
        return point_embedding

    def _embed_boxes(self, boxes):
        boxes = boxes + 0.5
        batch_size = ops.shape(boxes)[0]
        nb_boxes = ops.shape(boxes)[1]
        coords = ops.reshape(boxes, (batch_size, nb_boxes, 2, 2))
        coords_0 = coords[..., 0] / float(self.image_size)
        coords_1 = coords[..., 1] / float(self.image_size)
        normalized_coords = ops.stack([coords_0, coords_1], axis=-1)
        corner_embedding = self.shared_embedding(normalized_coords)
        pe_2 = ops.broadcast_to(
            self.point_embeddings[2],
            ops.shape(corner_embedding[:, :, 0:1, :]),
        )
        pe_3 = ops.broadcast_to(
            self.point_embeddings[3],
            ops.shape(corner_embedding[:, :, 1:2, :]),
        )
        corner_0 = corner_embedding[:, :, 0:1, :] + pe_2
        corner_1 = corner_embedding[:, :, 1:2, :] + pe_3
        return ops.concatenate([corner_0, corner_1], axis=2)

    def call(self, inputs):
        input_points, input_labels = inputs

        batch_size = ops.shape(input_points)[0]

        # Embed points with padding (matches HF's pad=True for points-only).
        sparse_embeddings = self._embed_points(input_points, input_labels, pad=True)

        no_mask = ops.reshape(self.no_mask_embed, (1, 1, 1, -1))
        dense_embeddings = ops.broadcast_to(
            no_mask,
            (
                batch_size,
                self.image_embedding_size,
                self.image_embedding_size,
                self.hidden_size,
            ),
        )

        return {
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "image_embedding_size": self.image_embedding_size,
                "image_size": self.image_size,
                "num_point_embeddings": self.num_point_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMMaskDecoderLayer(layers.Layer):
    """Mask decoder that predicts segmentation masks and IoU scores.

    Uses a two-way transformer to attend between prompt tokens and image
    embeddings, then upscales image features and generates per-token mask
    predictions via hypernetwork MLPs.

    Args:
        hidden_size: Hidden dimension.
        num_hidden_layers: Number of two-way transformer layers.
        num_attention_heads: Number of attention heads.
        mlp_dim: MLP dimension in the two-way transformer.
        num_multimask_outputs: Number of mask outputs (default 3).
        iou_head_depth: Depth of the IoU prediction head.
        iou_head_hidden_dim: Hidden dim of the IoU prediction head.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        mlp_dim=2048,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_dim = mlp_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim

        self.transformer_layers = []
        for i in range(num_hidden_layers):
            self.transformer_layers.append(
                SAMTwoWayAttentionBlock(
                    hidden_size,
                    num_attention_heads,
                    mlp_dim=mlp_dim,
                    attention_downsample_rate=2,
                    skip_first_layer_pe=(i == 0),
                    name=f"transformer_layers_{i}",
                )
            )
        self.final_attn_token_to_image = SAMTwoWayAttention(
            hidden_size,
            num_attention_heads,
            downsample_rate=2,
            name="final_attn_token_to_image",
        )
        self.layer_norm_final_attn = layers.LayerNormalization(
            epsilon=1e-5, name="layer_norm_final_attn"
        )

        self.upscale_conv1 = layers.Conv2DTranspose(
            hidden_size // 4,
            kernel_size=2,
            strides=2,
            name="upscale_conv1",
        )
        self.upscale_layer_norm = layers.LayerNormalization(
            epsilon=1e-6, name="upscale_layer_norm"
        )
        self.upscale_conv2 = layers.Conv2DTranspose(
            hidden_size // 8,
            kernel_size=2,
            strides=2,
            name="upscale_conv2",
        )

        self.output_hypernetworks_mlps = []
        for i in range(self.num_mask_tokens):
            self.output_hypernetworks_mlps.append(
                SAMFeedForward(
                    hidden_size,
                    hidden_size,
                    hidden_size // 8,
                    3,
                    name=f"output_hypernetworks_mlps_{i}",
                )
            )

        self.iou_prediction_head = SAMFeedForward(
            hidden_size,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            name="iou_prediction_head",
        )

    def build(self, input_shape):
        self.iou_token = self.add_weight(
            name="iou_token",
            shape=(1, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        self.mask_tokens = self.add_weight(
            name="mask_tokens",
            shape=(self.num_mask_tokens, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        image_embeddings, image_pe, sparse_embeddings, dense_embeddings = inputs

        batch_size = ops.shape(image_embeddings)[0]
        point_batch_size = ops.shape(sparse_embeddings)[1]

        output_tokens = ops.concatenate(
            [
                ops.expand_dims(self.iou_token, axis=0),
                ops.expand_dims(self.mask_tokens, axis=0),
            ],
            axis=1,
        )
        output_tokens = ops.broadcast_to(
            ops.expand_dims(output_tokens, axis=0),
            (batch_size, point_batch_size, 1 + self.num_mask_tokens, self.hidden_size),
        )

        tokens = ops.concatenate([output_tokens, sparse_embeddings], axis=2)

        image_embeddings_with_dense = image_embeddings + dense_embeddings

        num_channels = ops.shape(image_embeddings)[3]
        height = ops.shape(image_embeddings)[1]
        width = ops.shape(image_embeddings)[2]

        image_emb_flat = ops.reshape(
            image_embeddings_with_dense,
            (batch_size, height * width, num_channels),
        )
        image_emb_flat = ops.expand_dims(image_emb_flat, axis=1)
        image_emb_flat = ops.broadcast_to(
            image_emb_flat,
            (batch_size, point_batch_size, height * width, num_channels),
        )

        image_pe_flat = ops.reshape(image_pe, (1, num_channels, height * width))
        image_pe_flat = ops.transpose(image_pe_flat, (0, 2, 1))
        image_pe_flat = ops.expand_dims(image_pe_flat, axis=1)
        image_pe_flat = ops.broadcast_to(
            image_pe_flat,
            (batch_size, point_batch_size, height * width, num_channels),
        )

        queries = tokens
        keys = image_emb_flat

        for layer in self.transformer_layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=tokens,
                key_point_embedding=image_pe_flat,
            )

        query = queries + tokens
        key = keys + image_pe_flat
        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)

        iou_token_out = queries[:, :, 0, :]
        mask_tokens_out = queries[:, :, 1 : 1 + self.num_mask_tokens, :]

        keys_spatial = ops.reshape(
            keys,
            (batch_size * point_batch_size, height, width, num_channels),
        )

        upscaled = self.upscale_conv1(keys_spatial)
        upscaled = self.upscale_layer_norm(upscaled)
        upscaled = ops.nn.gelu(upscaled)
        upscaled = self.upscale_conv2(upscaled)
        upscaled = ops.nn.gelu(upscaled)

        up_channels = ops.shape(upscaled)[3]
        up_height = ops.shape(upscaled)[1]
        up_width = ops.shape(upscaled)[2]

        upscaled_flat = ops.reshape(
            upscaled,
            (batch_size, point_batch_size, up_height * up_width, up_channels),
        )

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i, :])
            )
        hyper_in = ops.stack(hyper_in_list, axis=2)

        masks = ops.matmul(hyper_in, ops.transpose(upscaled_flat, (0, 1, 3, 2)))
        masks = ops.reshape(
            masks,
            (batch_size, point_batch_size, self.num_mask_tokens, up_height, up_width),
        )

        iou_pred = self.iou_prediction_head(iou_token_out)

        return {"pred_masks": masks, "iou_scores": iou_pred}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "mlp_dim": self.mlp_dim,
                "num_multimask_outputs": self.num_multimask_outputs,
                "iou_head_depth": self.iou_head_depth,
                "iou_head_hidden_dim": self.iou_head_hidden_dim,
            }
        )
        return config


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
