"""Sam3Tracker layers: prompt encoder, mask decoder, two-way transformer."""

import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class ChannelsFirstLayerNorm(layers.Layer):
    """LayerNorm for NCHW tensors -- normalizes over C dimension.

    Standard LayerNormalization(axis=1) does NOT work on torch backend
    for NCHW tensors because torch.layer_norm requires normalized dims
    to be trailing. This custom implementation handles it correctly.
    """

    def __init__(self, num_channels, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="weight", shape=(self.num_channels,), initializer="ones"
        )
        self.beta = self.add_weight(
            name="bias", shape=(self.num_channels,), initializer="zeros"
        )
        self.built = True

    def call(self, x):
        mean = ops.mean(x, axis=1, keepdims=True)
        var = ops.var(x, axis=1, keepdims=True)
        x = (x - mean) / ops.sqrt(var + self.eps)
        w = ops.reshape(self.gamma, (1, -1, 1, 1))
        b = ops.reshape(self.beta, (1, -1, 1, 1))
        return x * w + b

    def get_config(self):
        config = super().get_config()
        config.update({"num_channels": self.num_channels, "eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerPositionalEmbedding(layers.Layer):
    """Sinusoidal positional embedding with learned projection."""

    def __init__(self, hidden_size=256, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.scale = scale

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(2, self.hidden_size // 2),
            initializer=keras.initializers.RandomNormal(stddev=self.scale),
            trainable=True,
        )
        self.built = True

    def call(self, coords, input_shape=None):
        if input_shape is not None:
            h, w = input_shape
            coords_x = coords[..., 0:1] / ops.cast(w, coords.dtype)
            coords_y = coords[..., 1:2] / ops.cast(h, coords.dtype)
            coords = ops.concatenate([coords_x, coords_y], axis=-1)

        coords = 2.0 * coords - 1.0
        coords = ops.matmul(coords, self.positional_embedding)
        coords = 2.0 * math.pi * coords
        return ops.concatenate([ops.sin(coords), ops.cos(coords)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "scale": self.scale})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerMaskEmbedding(layers.Layer):
    """Downsamples high-res mask (288x288) to embedding space (72x72, 256d)."""

    def __init__(self, hidden_size=256, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            4, kernel_size=2, strides=2, data_format="channels_first", name="conv1"
        )
        self.conv1.build((None, 1, None, None))
        self.layer_norm1 = ChannelsFirstLayerNorm(4, name="layer_norm1")
        self.layer_norm1.build((None, 4, None, None))
        self.conv2 = layers.Conv2D(
            16, kernel_size=2, strides=2, data_format="channels_first", name="conv2"
        )
        self.conv2.build((None, 4, None, None))
        self.layer_norm2 = ChannelsFirstLayerNorm(16, name="layer_norm2")
        self.layer_norm2.build((None, 16, None, None))
        self.conv3 = layers.Conv2D(
            self.hidden_size,
            kernel_size=1,
            data_format="channels_first",
            name="conv3",
        )
        self.conv3.build((None, 16, None, None))
        self.built = True

    def call(self, masks):
        x = self.conv1(masks)
        x = self.layer_norm1(x)
        x = ops.nn.gelu(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = ops.nn.gelu(x)
        return self.conv3(x)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerFeedForward(layers.Layer):
    """MLP with configurable depth and optional sigmoid output."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        sigmoid_output=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output

    def build(self, input_shape):
        h = self.hidden_dim
        self.proj_in = layers.Dense(h, name="proj_in")
        self.proj_in.build((None, None, None, self.input_dim))
        self.hidden_layers = []
        for i in range(self.num_layers - 2):
            d = layers.Dense(h, name=f"layers_{i}")
            d.build((None, None, None, h))
            self.hidden_layers.append(d)
        self.proj_out = layers.Dense(self.output_dim, name="proj_out")
        self.proj_out.build((None, None, None, h))
        self.built = True

    def call(self, x):
        out = ops.nn.relu(self.proj_in(x))
        for layer in self.hidden_layers:
            out = ops.nn.relu(layer(out))
        out = self.proj_out(out)
        if self.sigmoid_output:
            out = ops.sigmoid(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "sigmoid_output": self.sigmoid_output,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerPromptEncoder(layers.Layer):
    """Encodes points, boxes, and masks into sparse/dense embeddings."""

    def __init__(
        self,
        hidden_size=256,
        image_size=1008,
        patch_size=14,
        num_point_embeddings=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_point_embeddings = num_point_embeddings
        self.image_embedding_size = image_size // patch_size
        self.mask_input_size = 4 * self.image_embedding_size

    def build(self, input_shape):
        self.shared_embedding = Sam3TrackerPositionalEmbedding(
            self.hidden_size, name="shared_embedding"
        )
        self.shared_embedding.build(None)

        self.mask_embed = Sam3TrackerMaskEmbedding(self.hidden_size, name="mask_embed")
        self.mask_embed.build(None)

        self.no_mask_embed = layers.Embedding(1, self.hidden_size, name="no_mask_embed")
        self.no_mask_embed.build((None,))

        self.point_embed = layers.Embedding(
            self.num_point_embeddings, self.hidden_size, name="point_embed"
        )
        self.point_embed.build((None,))

        self.not_a_point_embed = layers.Embedding(
            1, self.hidden_size, name="not_a_point_embed"
        )
        self.not_a_point_embed.build((None,))
        self.built = True

    def _embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            pad_point = ops.zeros_like(points[..., :1, :])
            points = ops.concatenate([points, pad_point], axis=-2)
            pad_label = -ops.ones_like(labels[..., :1])
            labels = ops.concatenate([labels, pad_label], axis=-1)

        input_shape = (self.image_size, self.image_size)
        pe = self.shared_embedding(points, input_shape=input_shape)

        not_a_point = self.not_a_point_embed(ops.zeros((), dtype="int32"))
        is_neg1 = ops.expand_dims(ops.cast(labels == -1, pe.dtype), axis=-1)
        pe = pe * (1.0 - is_neg1) + not_a_point * is_neg1

        is_valid = ops.expand_dims(ops.cast(labels != -10, pe.dtype), axis=-1)
        pe = pe * is_valid

        labels_clamped = ops.maximum(labels, 0)
        point_emb = self.point_embed(ops.cast(labels_clamped, "int32"))
        is_positive = ops.expand_dims(ops.cast(labels >= 0, pe.dtype), axis=-1)
        pe = pe + point_emb * is_positive
        return pe

    def _embed_boxes(self, boxes):
        boxes = boxes + 0.5
        shape = ops.shape(boxes)
        coords = ops.reshape(boxes, (shape[0], shape[1], 2, 2))
        pad_coords = ops.zeros_like(coords[..., :1, :])
        coords = ops.concatenate([coords, pad_coords], axis=-2)

        input_shape = (self.image_size, self.image_size)
        corner_emb = self.shared_embedding(coords, input_shape=input_shape)

        tl_emb = self.point_embed(ops.convert_to_tensor(2, dtype="int32"))
        br_emb = self.point_embed(ops.convert_to_tensor(3, dtype="int32"))
        not_a_pt = self.not_a_point_embed(ops.zeros((), dtype="int32"))

        corner_emb_0 = corner_emb[..., 0:1, :] + tl_emb
        corner_emb_1 = corner_emb[..., 1:2, :] + br_emb
        not_a_pt_expanded = ops.broadcast_to(
            not_a_pt, ops.shape(corner_emb[..., 2:3, :])
        )
        return ops.concatenate([corner_emb_0, corner_emb_1, not_a_pt_expanded], axis=-2)

    def call(
        self,
        inputs=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        input_masks=None,
    ):
        if inputs is not None:
            input_points, input_labels = inputs[0], inputs[1]

        sparse_embeddings = None
        batch_size = 1

        if input_points is not None:
            batch_size = ops.shape(input_points)[0]
            if input_labels is None:
                raise ValueError("Labels required with points.")
            point_emb = self._embed_points(
                input_points, input_labels, pad=(input_boxes is None)
            )
            sparse_embeddings = point_emb

        if input_boxes is not None:
            batch_size = ops.shape(input_boxes)[0]
            box_emb = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_emb
            else:
                sparse_embeddings = ops.concatenate(
                    [sparse_embeddings, box_emb], axis=-2
                )

        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            emb = self.no_mask_embed(ops.zeros((1,), dtype="int32"))
            emb = ops.reshape(emb, (1, self.hidden_size, 1, 1))
            dense_embeddings = ops.broadcast_to(
                emb,
                (
                    batch_size,
                    self.hidden_size,
                    self.image_embedding_size,
                    self.image_embedding_size,
                ),
            )

        return {
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
        }

    def compute_output_spec(self, *args, **kwargs):
        return {
            "sparse_embeddings": keras.KerasTensor(
                shape=(None, None, None, self.hidden_size), dtype="float32"
            ),
            "dense_embeddings": keras.KerasTensor(
                shape=(
                    None,
                    self.hidden_size,
                    self.image_embedding_size,
                    self.image_embedding_size,
                ),
                dtype="float32",
            ),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_point_embeddings": self.num_point_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerAttention(layers.Layer):
    """Multi-head attention with optional dimension downsampling."""

    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        downsample_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = hidden_size // downsample_rate
        self.head_dim = self.internal_dim // num_attention_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.internal_dim, name="q_proj")
        self.q_proj.build((None, None, None, self.hidden_size))
        self.k_proj = layers.Dense(self.internal_dim, name="k_proj")
        self.k_proj.build((None, None, None, self.hidden_size))
        self.v_proj = layers.Dense(self.internal_dim, name="v_proj")
        self.v_proj.build((None, None, None, self.hidden_size))
        self.o_proj = layers.Dense(self.hidden_size, name="o_proj")
        self.o_proj.build((None, None, None, self.internal_dim))
        self.built = True

    def call(self, query, key, value, attention_similarity=None):
        shape_q = ops.shape(query)
        batch_size, point_batch_size = shape_q[0], shape_q[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        def reshape_heads(x):
            s = ops.shape(x)
            x = ops.reshape(
                x, (s[0] * s[1], s[2], self.num_attention_heads, self.head_dim)
            )
            return ops.transpose(x, (0, 2, 1, 3))

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if attention_similarity is not None:
            attn = attn + attention_similarity
        attn = ops.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(
            out,
            (
                batch_size,
                point_batch_size,
                -1,
                self.num_attention_heads * self.head_dim,
            ),
        )
        return self.o_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerTwoWayAttentionBlock(layers.Layer):
    """Bidirectional attention block: sparse <-> dense."""

    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        attention_downsample_rate=2,
        mlp_dim=2048,
        skip_first_layer_pe=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.mlp_dim = mlp_dim
        self.skip_first_layer_pe = skip_first_layer_pe

    def build(self, input_shape):
        self.self_attn = Sam3TrackerAttention(
            self.hidden_size,
            self.num_attention_heads,
            downsample_rate=1,
            name="self_attn",
        )
        self.self_attn.build(None)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm1")
        self.layer_norm1.build((None, None, None, self.hidden_size))

        self.cross_attn_token_to_image = Sam3TrackerAttention(
            self.hidden_size,
            self.num_attention_heads,
            self.attention_downsample_rate,
            name="cross_attn_token_to_image",
        )
        self.cross_attn_token_to_image.build(None)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm2")
        self.layer_norm2.build((None, None, None, self.hidden_size))

        self.mlp = Sam3TrackerFeedForward(
            self.hidden_size,
            self.mlp_dim,
            self.hidden_size,
            num_layers=2,
            name="mlp",
        )
        self.mlp.build((None, None, None, self.hidden_size))
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm3")
        self.layer_norm3.build((None, None, None, self.hidden_size))

        self.cross_attn_image_to_token = Sam3TrackerAttention(
            self.hidden_size,
            self.num_attention_heads,
            self.attention_downsample_rate,
            name="cross_attn_image_to_token",
        )
        self.cross_attn_image_to_token.build(None)
        self.layer_norm4 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm4")
        self.layer_norm4.build((None, None, None, self.hidden_size))
        self.built = True

    def call(
        self,
        queries,
        keys,
        query_point_embedding,
        key_point_embedding,
        attention_similarity=None,
    ):
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q_with_pe = queries + query_point_embedding
            attn_out = self.self_attn(q_with_pe, q_with_pe, queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        q_with_pe = queries + query_point_embedding
        k_with_pe = keys + key_point_embedding
        attn = self.cross_attn_token_to_image(
            q_with_pe, k_with_pe, keys, attention_similarity
        )
        queries = self.layer_norm2(queries + attn)

        mlp_out = self.mlp(queries)
        queries = self.layer_norm3(queries + mlp_out)

        q_with_pe = queries + query_point_embedding
        k_with_pe = keys + key_point_embedding
        attn = self.cross_attn_image_to_token(k_with_pe, q_with_pe, queries)
        keys = self.layer_norm4(keys + attn)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "attention_downsample_rate": self.attention_downsample_rate,
                "mlp_dim": self.mlp_dim,
                "skip_first_layer_pe": self.skip_first_layer_pe,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerTwoWayTransformer(layers.Layer):
    """Two-way transformer with final token-to-image attention."""

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        mlp_dim=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.transformer_layers = []
        for i in range(self.num_hidden_layers):
            block = Sam3TrackerTwoWayAttentionBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                attention_downsample_rate=self.attention_downsample_rate,
                mlp_dim=self.mlp_dim,
                skip_first_layer_pe=(i == 0),
                name=f"layers_{i}",
            )
            block.build(None)
            self.transformer_layers.append(block)

        self.final_attn_token_to_image = Sam3TrackerAttention(
            self.hidden_size,
            self.num_attention_heads,
            self.attention_downsample_rate,
            name="final_attn_token_to_image",
        )
        self.final_attn_token_to_image.build(None)

        self.layer_norm_final_attn = layers.LayerNormalization(
            epsilon=1e-5, name="layer_norm_final_attn"
        )
        self.layer_norm_final_attn.build((None, None, None, self.hidden_size))
        self.built = True

    def call(
        self,
        point_embeddings,
        image_embeddings,
        image_positional_embeddings,
        attention_similarity=None,
    ):
        shape = ops.shape(image_embeddings)
        image_flat = ops.reshape(
            ops.transpose(image_embeddings, (0, 2, 3, 1)),
            (shape[0], 1, shape[2] * shape[3], shape[1]),
        )
        image_pe_flat = ops.reshape(
            ops.transpose(image_positional_embeddings, (0, 2, 3, 1)),
            (shape[0], 1, shape[2] * shape[3], shape[1]),
        )

        queries = point_embeddings
        keys = image_flat

        for block in self.transformer_layers:
            queries, keys = block(
                queries,
                keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_pe_flat,
                attention_similarity=attention_similarity,
            )

        q = queries + point_embeddings
        k = keys + image_pe_flat
        attn = self.final_attn_token_to_image(q, k, keys)
        queries = self.layer_norm_final_attn(queries + attn)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "attention_downsample_rate": self.attention_downsample_rate,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3TrackerMaskDecoder(layers.Layer):
    """Decodes masks from image + prompt embeddings."""

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        mlp_dim=2048,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = 1 + num_multimask_outputs
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
        self._transformer_config = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "attention_downsample_rate": attention_downsample_rate,
            "mlp_dim": mlp_dim,
        }
        self._iou_config = {
            "depth": iou_head_depth,
            "hidden_dim": iou_head_hidden_dim,
        }

    def build(self, input_shape):
        d = self.hidden_size

        self.iou_token = layers.Embedding(1, d, name="iou_token")
        self.iou_token.build((None,))
        self.mask_tokens = layers.Embedding(self.num_mask_tokens, d, name="mask_tokens")
        self.mask_tokens.build((None,))
        self.obj_score_token = layers.Embedding(1, d, name="obj_score_token")
        self.obj_score_token.build((None,))

        self.transformer = Sam3TrackerTwoWayTransformer(
            name="transformer", **self._transformer_config
        )
        self.transformer.build(None)

        self.upscale_conv1 = layers.Conv2DTranspose(
            64,
            kernel_size=2,
            strides=2,
            data_format="channels_first",
            name="upscale_conv1",
        )
        self.upscale_conv1.build((None, d, None, None))
        self.upscale_layer_norm = ChannelsFirstLayerNorm(64, name="upscale_layer_norm")
        self.upscale_layer_norm.build((None, 64, None, None))
        self.upscale_conv2 = layers.Conv2DTranspose(
            32,
            kernel_size=2,
            strides=2,
            data_format="channels_first",
            name="upscale_conv2",
        )
        self.upscale_conv2.build((None, 64, None, None))

        self.output_hypernetworks_mlps = []
        for i in range(self.num_mask_tokens):
            mlp = Sam3TrackerFeedForward(
                d, d, 32, num_layers=3, name=f"output_hypernetworks_mlps_{i}"
            )
            mlp.build(None)
            self.output_hypernetworks_mlps.append(mlp)

        self.iou_prediction_head = Sam3TrackerFeedForward(
            d,
            self._iou_config["hidden_dim"],
            self.num_mask_tokens,
            num_layers=self._iou_config["depth"],
            sigmoid_output=True,
            name="iou_prediction_head",
        )
        self.iou_prediction_head.build(None)

        self.conv_s0 = layers.Conv2D(
            32, kernel_size=1, data_format="channels_first", name="conv_s0"
        )
        self.conv_s0.build((None, d, None, None))
        self.conv_s1 = layers.Conv2D(
            64, kernel_size=1, data_format="channels_first", name="conv_s1"
        )
        self.conv_s1.build((None, d, None, None))

        self.pred_obj_score_head = Sam3TrackerFeedForward(
            d, d, 1, num_layers=3, name="pred_obj_score_head"
        )
        self.pred_obj_score_head.build(None)
        self.built = True

    def call(
        self,
        inputs=None,
        image_embeddings=None,
        image_positional_embeddings=None,
        sparse_prompt_embeddings=None,
        dense_prompt_embeddings=None,
        multimask_output=True,
        high_resolution_features=None,
        attention_similarity=None,
    ):
        if inputs is not None:
            image_embeddings = inputs[0]
            image_positional_embeddings = inputs[1]
            sparse_prompt_embeddings = inputs[2]
            dense_prompt_embeddings = inputs[3]
            high_resolution_features = [inputs[4], inputs[5]]

        shape = ops.shape(image_embeddings)
        batch_size = shape[0]
        num_channels = shape[1]
        height = shape[2]
        width = shape[3]
        point_batch_size = ops.shape(sparse_prompt_embeddings)[1]

        obj_token = self.obj_score_token(ops.zeros((1,), dtype="int32"))
        iou_tok = self.iou_token(ops.zeros((1,), dtype="int32"))
        mask_tok = self.mask_tokens(ops.arange(self.num_mask_tokens, dtype="int32"))
        output_tokens = ops.concatenate([obj_token, iou_tok, mask_tok], axis=0)
        output_tokens = ops.broadcast_to(
            ops.reshape(output_tokens, (1, 1, -1, self.hidden_size)),
            (batch_size, point_batch_size, 6, self.hidden_size),
        )

        tokens = ops.concatenate([output_tokens, sparse_prompt_embeddings], axis=2)

        image_emb = image_embeddings + dense_prompt_embeddings

        queries, keys = self.transformer(
            point_embeddings=tokens,
            image_embeddings=image_emb,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
        )

        iou_token_out = queries[:, :, 1:2, :]
        mask_tokens_out = queries[:, :, 2:6, :]

        keys_spatial = ops.reshape(
            keys, (batch_size * point_batch_size, height, width, num_channels)
        )
        keys_spatial = ops.transpose(keys_spatial, (0, 3, 1, 2))

        feat_s0 = high_resolution_features[0]
        feat_s1 = high_resolution_features[1]
        if ops.shape(feat_s0)[1] == self.hidden_size:
            feat_s0 = self.conv_s0(feat_s0)
        if ops.shape(feat_s1)[1] == self.hidden_size:
            feat_s1 = self.conv_s1(feat_s1)

        upscaled = self.upscale_conv1(keys_spatial) + ops.repeat(
            feat_s1, point_batch_size, axis=0
        )
        upscaled = ops.nn.gelu(self.upscale_layer_norm(upscaled))
        upscaled = ops.nn.gelu(
            self.upscale_conv2(upscaled) + ops.repeat(feat_s0, point_batch_size, axis=0)
        )

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i : i + 1, :])
            )
        hyper_in = ops.concatenate(hyper_in_list, axis=2)

        up_shape = ops.shape(upscaled)
        upscaled_flat = ops.reshape(
            upscaled,
            (batch_size, point_batch_size, up_shape[1], up_shape[2] * up_shape[3]),
        )
        masks = ops.matmul(hyper_in, upscaled_flat)
        masks = ops.reshape(
            masks,
            (
                batch_size,
                point_batch_size,
                self.num_mask_tokens,
                up_shape[2],
                up_shape[3],
            ),
        )

        iou_pred = self.iou_prediction_head(iou_token_out)
        iou_pred = ops.squeeze(iou_pred, axis=2)
        object_score = self.pred_obj_score_head(queries[:, :, 0:1, :])
        object_score = ops.squeeze(object_score, axis=2)

        return {
            "pred_masks": masks,
            "iou_scores": iou_pred,
            "sam_tokens": mask_tokens_out,
            "object_score_logits": object_score,
        }

    def compute_output_spec(self, *args, **kwargs):
        return {
            "pred_masks": keras.KerasTensor(
                shape=(None, None, self.num_mask_tokens, None, None), dtype="float32"
            ),
            "iou_scores": keras.KerasTensor(
                shape=(None, None, self.num_mask_tokens), dtype="float32"
            ),
            "sam_tokens": keras.KerasTensor(
                shape=(None, None, self.num_mask_tokens, self.hidden_size),
                dtype="float32",
            ),
            "object_score_logits": keras.KerasTensor(
                shape=(None, None, 1), dtype="float32"
            ),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_multimask_outputs": self.num_multimask_outputs,
                "dynamic_multimask_via_stability": self.dynamic_multimask_via_stability,
                "dynamic_multimask_stability_delta": self.dynamic_multimask_stability_delta,
                "dynamic_multimask_stability_thresh": self.dynamic_multimask_stability_thresh,
                **self._transformer_config,
                "iou_head_depth": self._iou_config["depth"],
                "iou_head_hidden_dim": self._iou_config["hidden_dim"],
            }
        )
        return config
