import keras
import numpy as np
from keras import layers, ops


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
        attention_downsample_rate: Downsample factor for cross-attention.
        layer_norm_eps: LayerNorm epsilon.
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
        attention_downsample_rate=2,
        layer_norm_eps=1e-6,
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
        self.attention_downsample_rate = attention_downsample_rate
        self.layer_norm_eps = layer_norm_eps

        self.transformer_self_attns = []
        self.transformer_layer_norm1s = []
        self.transformer_cross_attn_token_to_images = []
        self.transformer_layer_norm2s = []
        self.transformer_mlp_lin1s = []
        self.transformer_mlp_lin2s = []
        self.transformer_layer_norm3s = []
        self.transformer_layer_norm4s = []
        self.transformer_cross_attn_image_to_tokens = []

        for i in range(num_hidden_layers):
            prefix = f"transformer_layers_{i}"
            self.transformer_self_attns.append(
                SAMTwoWayAttention(
                    hidden_size,
                    num_attention_heads,
                    downsample_rate=1,
                    name=f"{prefix}_self_attn",
                )
            )
            self.transformer_layer_norm1s.append(
                layers.LayerNormalization(
                    epsilon=layer_norm_eps, name=f"{prefix}_layer_norm1"
                )
            )
            self.transformer_cross_attn_token_to_images.append(
                SAMTwoWayAttention(
                    hidden_size,
                    num_attention_heads,
                    downsample_rate=attention_downsample_rate,
                    name=f"{prefix}_cross_attn_token_to_image",
                )
            )
            self.transformer_layer_norm2s.append(
                layers.LayerNormalization(
                    epsilon=layer_norm_eps, name=f"{prefix}_layer_norm2"
                )
            )
            self.transformer_mlp_lin1s.append(
                layers.Dense(mlp_dim, name=f"{prefix}_mlp_lin1")
            )
            self.transformer_mlp_lin2s.append(
                layers.Dense(hidden_size, name=f"{prefix}_mlp_lin2")
            )
            self.transformer_layer_norm3s.append(
                layers.LayerNormalization(
                    epsilon=layer_norm_eps, name=f"{prefix}_layer_norm3"
                )
            )
            self.transformer_layer_norm4s.append(
                layers.LayerNormalization(
                    epsilon=layer_norm_eps, name=f"{prefix}_layer_norm4"
                )
            )
            self.transformer_cross_attn_image_to_tokens.append(
                SAMTwoWayAttention(
                    hidden_size,
                    num_attention_heads,
                    downsample_rate=attention_downsample_rate,
                    name=f"{prefix}_cross_attn_image_to_token",
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

        self.output_hypernetworks_mlps_proj_ins = []
        self.output_hypernetworks_mlps_hidden_layers = []
        self.output_hypernetworks_mlps_proj_outs = []
        for i in range(self.num_mask_tokens):
            prefix = f"output_hypernetworks_mlps_{i}"
            self.output_hypernetworks_mlps_proj_ins.append(
                layers.Dense(hidden_size, name=f"{prefix}_proj_in")
            )
            h_layers = []
            for j in range(3 - 2):  # num_layers=3
                h_layers.append(layers.Dense(hidden_size, name=f"{prefix}_layers_{j}"))
            self.output_hypernetworks_mlps_hidden_layers.append(h_layers)
            self.output_hypernetworks_mlps_proj_outs.append(
                layers.Dense(hidden_size // 8, name=f"{prefix}_proj_out")
            )

        iou_prefix = "iou_prediction_head"
        self.iou_head_proj_in = layers.Dense(
            iou_head_hidden_dim, name=f"{iou_prefix}_proj_in"
        )
        self.iou_head_hidden_layers = []
        for j in range(iou_head_depth - 2):
            self.iou_head_hidden_layers.append(
                layers.Dense(iou_head_hidden_dim, name=f"{iou_prefix}_layers_{j}")
            )
        self.iou_head_proj_out = layers.Dense(
            self.num_mask_tokens, name=f"{iou_prefix}_proj_out"
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

        for i in range(self.num_hidden_layers):
            if i == 0:
                queries = self.transformer_self_attns[i](
                    query=queries, key=queries, value=queries
                )
            else:
                query = queries + tokens
                attn_out = self.transformer_self_attns[i](
                    query=query, key=query, value=queries
                )
                queries = queries + attn_out
            queries = self.transformer_layer_norm1s[i](queries)

            query = queries + tokens
            key = keys + image_pe_flat
            attn_out = self.transformer_cross_attn_token_to_images[i](
                query=query, key=key, value=keys
            )
            queries = queries + attn_out
            queries = self.transformer_layer_norm2s[i](queries)

            mlp_out = self.transformer_mlp_lin1s[i](queries)
            mlp_out = ops.nn.relu(mlp_out)
            mlp_out = self.transformer_mlp_lin2s[i](mlp_out)
            queries = queries + mlp_out
            queries = self.transformer_layer_norm3s[i](queries)

            query = queries + tokens
            key = keys + image_pe_flat
            attn_out = self.transformer_cross_attn_image_to_tokens[i](
                query=key, key=query, value=queries
            )
            keys = keys + attn_out
            keys = self.transformer_layer_norm4s[i](keys)

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
            h = self.output_hypernetworks_mlps_proj_ins[i](mask_tokens_out[:, :, i, :])
            h = ops.nn.relu(h)
            for hidden_layer in self.output_hypernetworks_mlps_hidden_layers[i]:
                h = ops.nn.relu(hidden_layer(h))
            h = self.output_hypernetworks_mlps_proj_outs[i](h)
            hyper_in_list.append(h)
        hyper_in = ops.stack(hyper_in_list, axis=2)

        masks = ops.matmul(hyper_in, ops.transpose(upscaled_flat, (0, 1, 3, 2)))
        masks = ops.reshape(
            masks,
            (batch_size, point_batch_size, self.num_mask_tokens, up_height, up_width),
        )

        iou_pred = self.iou_head_proj_in(iou_token_out)
        iou_pred = ops.nn.relu(iou_pred)
        for hidden_layer in self.iou_head_hidden_layers:
            iou_pred = ops.nn.relu(hidden_layer(iou_pred))
        iou_pred = self.iou_head_proj_out(iou_pred)

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
                "attention_downsample_rate": self.attention_downsample_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMVisionAttention(layers.Layer):
    """Multi-head attention with decomposed relative position embeddings.

    Supports both windowed and global attention modes. When ``window_size > 0``,
    relative position parameters have shape ``(2*window_size - 1, head_dim)``; for
    global attention layers, the input_size is derived from the full feature map.

    Args:
        hidden_size: Total hidden dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projection.
        use_rel_pos: Whether to add decomposed relative position bias.
        input_size: Spatial resolution ``(H, W)`` used for relative position tables.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        qkv_bias=True,
        use_rel_pos=True,
        input_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        self.qkv = layers.Dense(hidden_size * 3, use_bias=qkv_bias, name="qkv")
        self.proj = layers.Dense(hidden_size, name="proj")

    def build(self, input_shape):
        if self.use_rel_pos and self.input_size is not None:
            self.rel_pos_h = self.add_weight(
                name="rel_pos_h",
                shape=(2 * self.input_size[0] - 1, self.head_dim),
                initializer="zeros",
                trainable=True,
            )
            self.rel_pos_w = self.add_weight(
                name="rel_pos_w",
                shape=(2 * self.input_size[1] - 1, self.head_dim),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def _get_rel_pos(self, q_size, k_size, rel_pos):
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        rel_pos_shape = ops.shape(rel_pos)
        if rel_pos_shape[0] != max_rel_dist:
            rel_pos_resized = ops.transpose(
                ops.reshape(rel_pos, (1, rel_pos_shape[0], -1)), (0, 2, 1)
            )
            rel_pos_resized = ops.image.resize(
                ops.expand_dims(rel_pos_resized, axis=-1),
                (ops.shape(rel_pos_resized)[1], max_rel_dist),
                interpolation="bilinear",
            )
            rel_pos_resized = ops.squeeze(rel_pos_resized, axis=-1)
            rel_pos_resized = ops.transpose(
                ops.reshape(rel_pos_resized, (-1, max_rel_dist)), (1, 0)
            )
        else:
            rel_pos_resized = rel_pos

        q_coords = ops.cast(
            ops.expand_dims(ops.arange(q_size), axis=1), dtype="float32"
        ) * max(k_size / q_size, 1.0)
        k_coords = ops.cast(
            ops.expand_dims(ops.arange(k_size), axis=0), dtype="float32"
        ) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
            q_size / k_size, 1.0
        )
        relative_coords = ops.cast(relative_coords, dtype="int32")
        return ops.take(rel_pos_resized, relative_coords, axis=0)

    def _get_decomposed_rel_pos(self, query, q_size, k_size):
        query_height, query_width = q_size
        key_height, key_width = k_size
        rel_pos_h = self._get_rel_pos(query_height, key_height, self.rel_pos_h)
        rel_pos_w = self._get_rel_pos(query_width, key_width, self.rel_pos_w)

        batch_size = ops.shape(query)[0]
        dim = ops.shape(query)[2]
        reshaped_query = ops.reshape(
            query, (batch_size, query_height, query_width, dim)
        )
        rel_h = ops.einsum("bhwc,hkc->bhwk", reshaped_query, rel_pos_h)
        rel_w = ops.einsum("bhwc,wkc->bhwk", reshaped_query, rel_pos_w)
        return ops.expand_dims(rel_h, axis=-1) + ops.expand_dims(rel_w, axis=-2)

    def call(self, hidden_states):
        batch_size = ops.shape(hidden_states)[0]
        height = ops.shape(hidden_states)[1]
        width = ops.shape(hidden_states)[2]

        qkv = self.qkv(hidden_states)
        qkv = ops.reshape(qkv, (batch_size, height * width, 3, self.num_heads, -1))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        qkv = ops.reshape(qkv, (3, batch_size * self.num_heads, height * width, -1))
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_weights = ops.matmul(query * self.scale, ops.transpose(key, (0, 2, 1)))

        if self.use_rel_pos:
            decomposed_rel_pos = self._get_decomposed_rel_pos(
                query, (height, width), (height, width)
            )
            decomposed_rel_pos = ops.reshape(
                decomposed_rel_pos, ops.shape(attn_weights)
            )
            attn_weights = attn_weights + decomposed_rel_pos

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, value)
        attn_output = ops.reshape(
            attn_output, (batch_size, self.num_heads, height, width, -1)
        )
        attn_output = ops.transpose(attn_output, (0, 2, 3, 1, 4))
        attn_output = ops.reshape(attn_output, (batch_size, height, width, -1))
        attn_output = self.proj(attn_output)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "use_rel_pos": self.use_rel_pos,
                "input_size": self.input_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMVisionLayer(layers.Layer):
    """Single transformer block in the SAM vision encoder.

    Implements windowed or global attention with optional relative position bias,
    followed by a two-layer MLP.  When ``window_size > 0``, the input is
    partitioned into non-overlapping windows before attention.

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension.
        qkv_bias: QKV bias flag.
        use_rel_pos: Relative position flag.
        window_size: Window size (0 = global attention).
        image_size: Full image patch grid size.
        layer_norm_eps: LayerNorm epsilon.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_dim,
        qkv_bias=True,
        use_rel_pos=True,
        window_size=0,
        image_size=64,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps

        if window_size == 0:
            input_size = (image_size, image_size)
        else:
            input_size = (window_size, window_size)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm1"
        )
        self.attn = SAMVisionAttention(
            hidden_size,
            num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size,
            name="attn",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm2"
        )
        self.mlp_lin1 = layers.Dense(mlp_dim, name="mlp_lin1")
        self.mlp_lin2 = layers.Dense(hidden_size, name="mlp_lin2")

    def _window_partition(self, hidden_states, window_size):
        batch_size = ops.shape(hidden_states)[0]
        height = ops.shape(hidden_states)[1]
        width = ops.shape(hidden_states)[2]
        channel = ops.shape(hidden_states)[3]

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            hidden_states = ops.pad(
                hidden_states, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
            )
        pad_height = height + pad_h
        pad_width = width + pad_w

        hidden_states = ops.reshape(
            hidden_states,
            (
                batch_size,
                pad_height // window_size,
                window_size,
                pad_width // window_size,
                window_size,
                channel,
            ),
        )
        windows = ops.reshape(
            ops.transpose(hidden_states, (0, 1, 3, 2, 4, 5)),
            (-1, window_size, window_size, channel),
        )
        return windows, (pad_height, pad_width)

    def _window_unpartition(self, windows, window_size, padding_shape, original_shape):
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = ops.shape(windows)[0] // (
            pad_height * pad_width // window_size // window_size
        )
        hidden_states = ops.reshape(
            windows,
            (
                batch_size,
                pad_height // window_size,
                pad_width // window_size,
                window_size,
                window_size,
                -1,
            ),
        )
        hidden_states = ops.reshape(
            ops.transpose(hidden_states, (0, 1, 3, 2, 4, 5)),
            (batch_size, pad_height, pad_width, -1),
        )
        return hidden_states[:, :height, :width, :]

    def call(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        if self.window_size > 0:
            height = ops.shape(hidden_states)[1]
            width = ops.shape(hidden_states)[2]
            hidden_states, padding_shape = self._window_partition(
                hidden_states, self.window_size
            )

        hidden_states = self.attn(hidden_states)

        if self.window_size > 0:
            hidden_states = self._window_unpartition(
                hidden_states, self.window_size, padding_shape, (height, width)
            )

        hidden_states = residual + hidden_states
        ln_out = self.layer_norm2(hidden_states)
        mlp_out = self.mlp_lin1(ln_out)
        mlp_out = ops.nn.gelu(mlp_out)
        mlp_out = self.mlp_lin2(mlp_out)
        hidden_states = hidden_states + mlp_out
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "qkv_bias": self.qkv_bias,
                "use_rel_pos": self.use_rel_pos,
                "window_size": self.window_size,
                "image_size": self.image_size,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMPositionalEmbedding(layers.Layer):
    """Random Fourier feature positional encoding used in prompt and mask encoders.

    Encodes 2-D coordinates normalized to ``[0, 1]`` into a fixed-dimensional
    feature vector via ``sin``/``cos`` of a learned random projection.

    Args:
        num_pos_feats: Half of the output dimension (full output = ``2 * num_pos_feats``).
        scale: Standard deviation of the initial random projection matrix.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, num_pos_feats=128, scale=None, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.scale = scale if scale is not None else 1.0

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(2, self.num_pos_feats),
            initializer=keras.initializers.RandomNormal(stddev=self.scale),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, coordinates):
        """Encode pre-normalized coordinates in [-1, 1].

        Args:
            coordinates: Tensor with last dim = 2, already normalized to [0, 1].
        """
        coordinates = 2.0 * coordinates - 1.0
        coordinates = ops.cast(coordinates, dtype=self.positional_embedding.dtype)
        coordinates = ops.matmul(coordinates, self.positional_embedding)
        coordinates = 2.0 * np.pi * coordinates
        return ops.concatenate([ops.sin(coordinates), ops.cos(coordinates)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"num_pos_feats": self.num_pos_feats, "scale": self.scale})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMTwoWayAttention(layers.Layer):
    """Attention layer used in the mask decoder's two-way transformer.

    Supports an optional ``downsample_rate`` that reduces the internal dimension
    of Q/K/V projections for efficiency.

    Args:
        hidden_size: Input hidden dimension.
        num_heads: Number of attention heads.
        downsample_rate: Factor to reduce internal dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_size, num_heads, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = hidden_size // downsample_rate
        self.head_dim = self.internal_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = layers.Dense(self.internal_dim, name="q_proj")
        self.k_proj = layers.Dense(self.internal_dim, name="k_proj")
        self.v_proj = layers.Dense(self.internal_dim, name="v_proj")
        self.out_proj = layers.Dense(hidden_size, name="out_proj")

    def call(self, query, key, value, attention_similarity=None):
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self._separate_heads(query)
        key = self._separate_heads(key)
        value = self._separate_heads(value)

        attn_weights = ops.matmul(query, ops.transpose(key, (0, 1, 2, 4, 3)))
        attn_weights = attn_weights * self.scale

        if attention_similarity is not None:
            attn_weights = attn_weights + attention_similarity

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, value)
        attn_output = self._recombine_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output

    def _separate_heads(self, x):
        batch = ops.shape(x)[0]
        point_batch = ops.shape(x)[1]
        n_tokens = ops.shape(x)[2]
        x = ops.reshape(
            x, (batch, point_batch, n_tokens, self.num_heads, self.head_dim)
        )
        return ops.transpose(x, (0, 1, 3, 2, 4))

    def _recombine_heads(self, x):
        batch = ops.shape(x)[0]
        point_batch = ops.shape(x)[1]
        n_tokens = ops.shape(x)[3]
        x = ops.transpose(x, (0, 1, 3, 2, 4))
        return ops.reshape(x, (batch, point_batch, n_tokens, self.internal_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config
