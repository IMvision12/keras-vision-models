import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class SAMAbsolutePositionEmbedding(layers.Layer):
    """Learnable absolute position embeddings for vision encoder patches.

    Adds a trainable position embedding tensor to the patch embeddings
    produced by the vision encoder's patch projection. The embedding
    has the same spatial dimensions as the patch grid and is broadcast
    across the batch dimension.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        hidden_size: Integer, embedding dimension for each spatial
            position. Must match the channel dimension of the patch
            embeddings.
        image_embedding_size: Integer, spatial size of the patch
            grid (both height and width).
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        4D tensor: ``(batch_size, image_embedding_size,
        image_embedding_size, hidden_size)``.

    Output Shape:
        Same as input: ``(batch_size, image_embedding_size,
        image_embedding_size, hidden_size)``.
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

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["image_embedding_size"] = self.image_embedding_size

    def load_own_variables(self, store):
        source_size = int(store["image_embedding_size"][...])

        if source_size == self.image_embedding_size:
            self.pos_embed.assign(store["0"])
            return

        pos_embed = store["0"]
        pos_embed = ops.cast(pos_embed, dtype="float32")
        pos_embed = ops.image.resize(
            pos_embed,
            size=(self.image_embedding_size, self.image_embedding_size),
            interpolation="bilinear",
            antialias=True,
        )
        self.pos_embed.assign(pos_embed)

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
class SAMPromptEncoderLayer(layers.Layer):
    """Prompt encoder for sparse (point/box) and dense (mask) prompts.

    Encodes user-provided prompts into embedding vectors that condition
    the mask decoder. Sparse prompts (points and boxes) are converted
    to positional embeddings via a shared Fourier encoding layer, then
    augmented with learned type embeddings that distinguish foreground
    points, background points, and box corners. When no mask prompt is
    provided, a learned ``no_mask_embed`` is broadcast to the image
    embedding spatial size to serve as the dense embedding.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        hidden_size: Integer, output embedding dimension.
            Defaults to ``256``.
        image_embedding_size: Integer, spatial size of the image
            embedding grid (both height and width).
            Defaults to ``64``.
        image_size: Integer, original input image resolution in
            pixels. Used to normalize point coordinates.
            Defaults to ``1024``.
        num_point_embeddings: Integer, number of learned point type
            embeddings (foreground, background, top-left corner,
            bottom-right corner). Defaults to ``4``.
        shared_embedding: A ``SAMPositionalEmbedding`` layer instance
            used for Fourier coordinate encoding. Shared with the
            image positional embedding generation.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        List of two tensors:
        - ``input_points``: ``(batch_size, num_prompts, num_points, 2)``
        - ``input_labels``: ``(batch_size, num_prompts, num_points)``

    Output Shape:
        Dictionary with:
        - ``"sparse_embeddings"``: ``(batch_size, num_prompts,
          num_points + 1, hidden_size)``
        - ``"dense_embeddings"``: ``(batch_size, image_embedding_size,
          image_embedding_size, hidden_size)``
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

    Uses a lightweight two-way transformer to jointly attend between
    prompt tokens and image embeddings. The decoder first concatenates
    learned IoU and mask tokens with the sparse prompt embeddings,
    then alternates self-attention on tokens, cross-attention from
    tokens to image features, an MLP block, and cross-attention from
    image features back to tokens. After the transformer, image
    features are upscaled via two transposed convolutions with GELU
    activation, and per-mask predictions are generated by
    hypernetwork MLPs that produce dynamic convolution weights. An
    IoU prediction head estimates the quality of each mask output.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        hidden_size: Integer, hidden dimension for the transformer
            and embedding layers. Defaults to ``256``.
        num_hidden_layers: Integer, number of two-way transformer
            layers. Defaults to ``2``.
        num_attention_heads: Integer, number of attention heads in
            each attention layer. Defaults to ``8``.
        mlp_dim: Integer, hidden dimension of the feedforward
            network in each transformer layer.
            Defaults to ``2048``.
        num_multimask_outputs: Integer, number of mask outputs
            (excluding the single-mask output token).
            Defaults to ``3``.
        iou_head_depth: Integer, total number of linear layers in
            the IoU prediction MLP. Defaults to ``3``.
        iou_head_hidden_dim: Integer, hidden dimension of the IoU
            prediction MLP. Defaults to ``256``.
        attention_downsample_rate: Integer, factor by which to
            reduce the internal dimension in cross-attention
            layers for efficiency. Defaults to ``2``.
        layer_norm_eps: Float, epsilon for layer normalization.
            Defaults to ``1e-6``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        List of four tensors:
        - ``image_embeddings``: ``(batch_size, H, W, hidden_size)``
        - ``image_pe``: ``(1, hidden_size, H, W)``
        - ``sparse_embeddings``: ``(batch_size, num_prompts,
          num_tokens, hidden_size)``
        - ``dense_embeddings``: ``(batch_size, H, W, hidden_size)``

    Output Shape:
        Dictionary with:
        - ``"pred_masks"``: ``(batch_size, num_prompts,
          num_mask_tokens, 4*H, 4*W)``
        - ``"iou_scores"``: ``(batch_size, num_prompts,
          num_mask_tokens)``
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
            epsilon=1e-6, name="layer_norm_final_attn"
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
        self._hyper_num_hidden = 3 - 2  # num_layers=3 -> 1 hidden layer each
        for i in range(self.num_mask_tokens):
            prefix = f"output_hypernetworks_mlps_{i}"
            self.output_hypernetworks_mlps_proj_ins.append(
                layers.Dense(hidden_size, name=f"{prefix}_proj_in")
            )
            for j in range(self._hyper_num_hidden):
                self.output_hypernetworks_mlps_hidden_layers.append(
                    layers.Dense(hidden_size, name=f"{prefix}_layers_{j}")
                )
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
        upscaled = ops.nn.gelu(upscaled, approximate=False)
        upscaled = self.upscale_conv2(upscaled)
        upscaled = ops.nn.gelu(upscaled, approximate=False)

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
            for j in range(self._hyper_num_hidden):
                h = ops.nn.relu(
                    self.output_hypernetworks_mlps_hidden_layers[
                        i * self._hyper_num_hidden + j
                    ](h)
                )
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
    """Multi-head attention with decomposed relative position bias.

    Implements scaled dot-product multi-head attention with an
    optional decomposed relative position bias. The bias is added
    to the attention logits and decomposes into separate
    height-axis and width-axis learned embeddings, reducing the
    parameter count from ``O(H*W * H*W)`` to ``O(H + W)`` per
    head. Supports both windowed and global attention modes: when
    operating within windows, the relative position tables have
    shape ``(2 * window_size - 1, head_dim)``; for global
    attention, ``input_size`` corresponds to the full feature map.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        hidden_size: Integer, total hidden dimension. Must be
            divisible by ``num_heads``.
        num_heads: Integer, number of parallel attention heads.
        qkv_bias: Boolean, whether to include bias terms in the
            fused QKV projection. Defaults to ``True``.
        use_rel_pos: Boolean, whether to add decomposed relative
            position bias to the attention logits.
            Defaults to ``True``.
        input_size: Tuple of two integers ``(H, W)`` specifying
            the spatial resolution used to size the relative
            position tables. ``None`` disables relative position
            even if ``use_rel_pos`` is ``True``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        4D tensor: ``(batch_size, height, width, hidden_size)``.

    Output Shape:
        4D tensor: ``(batch_size, height, width, hidden_size)``.
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

    def load_own_variables(self, store):
        if not self.use_rel_pos or self.input_size is None:
            super().load_own_variables(store)
            return

        target_vars = self._trainable_variables + self._non_trainable_variables
        rel_h_idx = None
        rel_w_idx = None
        for i, var in enumerate(target_vars):
            if var is self.rel_pos_h:
                rel_h_idx = i
            elif var is self.rel_pos_w:
                rel_w_idx = i

        source_h = (store[str(rel_h_idx)].shape[0] + 1) // 2
        source_w = (store[str(rel_w_idx)].shape[0] + 1) // 2

        if source_h == self.input_size[0] and source_w == self.input_size[1]:
            super().load_own_variables(store)
            return

        for i, var in enumerate(target_vars):
            if i in (rel_h_idx, rel_w_idx):
                continue
            var.assign(store[str(i)])

        for rel_idx, target_size in [
            (rel_h_idx, 2 * self.input_size[0] - 1),
            (rel_w_idx, 2 * self.input_size[1] - 1),
        ]:
            rel_pos = store[str(rel_idx)]
            rel_pos = ops.cast(rel_pos, dtype="float32")
            source_len = rel_pos.shape[0]
            head_dim = rel_pos.shape[1]
            rel_pos = ops.reshape(rel_pos, (1, source_len, head_dim))
            rel_pos = ops.transpose(rel_pos, (0, 2, 1))
            rel_pos = ops.expand_dims(rel_pos, axis=-1)
            rel_pos = ops.image.resize(
                rel_pos,
                size=(head_dim, target_size),
                interpolation="bilinear",
                antialias=True,
            )
            rel_pos = ops.squeeze(rel_pos, axis=-1)
            rel_pos = ops.transpose(rel_pos, (0, 2, 1))
            rel_pos = ops.reshape(rel_pos, (target_size, head_dim))
            target_var = self.rel_pos_h if rel_idx == rel_h_idx else self.rel_pos_w
            target_var.assign(rel_pos)

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

    Applies pre-norm layer normalization, multi-head attention with
    decomposed relative position bias, and a two-layer MLP with
    GELU activation. Both sub-blocks use residual connections. When
    ``window_size > 0``, the spatial input is partitioned into
    non-overlapping windows before attention and unpartitioned
    afterward, limiting the attention receptive field to reduce
    compute. Layers at global attention indices use
    ``window_size=0`` to attend over the full feature map.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        hidden_size: Integer, hidden dimension for attention and
            MLP output projections.
        num_heads: Integer, number of parallel attention heads.
        mlp_dim: Integer, hidden dimension of the two-layer
            feedforward network.
        qkv_bias: Boolean, whether to include bias in the QKV
            projection. Defaults to ``True``.
        use_rel_pos: Boolean, whether to use decomposed relative
            position bias in attention. Defaults to ``True``.
        window_size: Integer, size of the attention window. Set
            to ``0`` for global attention. Defaults to ``0``.
        image_size: Integer, full patch grid size used to
            determine the relative position table size for global
            attention layers. Defaults to ``64``.
        layer_norm_eps: Float, epsilon for layer normalization.
            Defaults to ``1e-6``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        4D tensor: ``(batch_size, height, width, hidden_size)``.

    Output Shape:
        4D tensor: ``(batch_size, height, width, hidden_size)``.
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
        mlp_out = ops.nn.gelu(mlp_out, approximate=False)
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
class SAMImagePositionalEmbeddings(layers.Layer):
    """Grid-based positional embeddings for the image feature map.

    Builds a normalized ``[0, 1]`` coordinate grid over the image
    embedding spatial dimensions and encodes it with the shared
    random Fourier feature layer (``SAMPositionalEmbedding``). The
    resulting tensor provides fixed positional information that is
    added to the keys in the mask decoder's cross-attention layers.

    This computation depends on the ``positional_embedding`` weight
    inside the shared embedding layer, so it must be a ``Layer``
    subclass to remain part of the Keras computation graph.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        image_embedding_size: Integer, spatial size of the image
            embedding grid (both height and width).
        shared_embedding: A ``SAMPositionalEmbedding`` layer
            instance used to encode the coordinate grid. Shared
            with the prompt encoder.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        Not used. A dummy input is required to trigger the
        ``call`` method in Keras functional API.

    Output Shape:
        4D tensor: ``(1, 2 * num_pos_feats, H, W)``.
    """

    def __init__(self, image_embedding_size, shared_embedding, **kwargs):
        super().__init__(**kwargs)
        self.image_embedding_size = image_embedding_size
        self.shared_embedding = shared_embedding

    def call(self, inputs):
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
        config.update(
            {
                "image_embedding_size": self.image_embedding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAMPositionalEmbedding(layers.Layer):
    """Random Fourier feature positional encoding for 2-D coordinates.

    Encodes 2-D coordinates normalized to ``[0, 1]`` into a
    fixed-dimensional feature vector using random Fourier features.
    Coordinates are first mapped to ``[-1, 1]``, linearly projected
    by a frozen random weight matrix, scaled by ``2 * pi``, then
    passed through concatenated ``sin`` and ``cos`` functions. The
    projection matrix is initialized once and not updated during
    training, producing a stable positional encoding shared between
    the prompt encoder and image positional embedding generation.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        num_pos_feats: Integer, half of the output embedding
            dimension. The full output has ``2 * num_pos_feats``
            channels. Defaults to ``128``.
        scale: Float, standard deviation of the initial random
            projection matrix. Defaults to ``1.0``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        Tensor of shape ``(..., 2)`` with coordinates in
        ``[0, 1]``.

    Output Shape:
        Tensor of shape ``(..., 2 * num_pos_feats)``.
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
    """Multi-head attention for the mask decoder's two-way transformer.

    Implements scaled dot-product multi-head attention with separate
    query, key, and value projections followed by an output
    projection. An optional ``downsample_rate`` reduces the internal
    dimension of Q/K/V projections for efficiency, which is used in
    the cross-attention layers of the mask decoder to lower compute
    cost. The projection naming (``q_proj``, ``k_proj``, ``v_proj``,
    ``out_proj``) matches the HuggingFace SAM layout for direct
    weight transfer.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        hidden_size: Integer, input and output hidden dimension.
            Must be divisible by ``num_heads * downsample_rate``.
        num_heads: Integer, number of parallel attention heads.
        downsample_rate: Integer, factor by which to reduce the
            internal projection dimension. Defaults to ``1``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        Three 4D tensors:
        - ``query``: ``(batch_size, num_prompts, seq_len_q,
          hidden_size)``
        - ``key``: ``(batch_size, num_prompts, seq_len_k,
          hidden_size)``
        - ``value``: ``(batch_size, num_prompts, seq_len_k,
          hidden_size)``

    Output Shape:
        4D tensor: ``(batch_size, num_prompts, seq_len_q,
        hidden_size)``.
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
