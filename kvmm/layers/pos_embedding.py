import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class AddPositionEmbs(layers.Layer):
    """
    A custom Keras layer that adds learnable position embeddings to input tensors with support for
    flexible grid sizes and optional class/distillation token handling.

    The layer supports three modes of operation:
      1. Standard mode: Creates position embeddings for both patches and a class token.
      2. FlexiViT mode: Creates embeddings only for patches, handling a class token separately.
         In this mode, the layer accepts either a pure patch token input (grid_h * grid_w tokens)
         or an input with a leading class token (grid_h * grid_w + 1 tokens).
      3. DeiT mode: Creates embeddings for patches, a class token, and a distillation token.
         In this case the expected sequence length is grid_h * grid_w + 2 tokens.

    For PiT, the recommended usage is to set `no_embed_class=True` so that the patch tokens
    are first augmented with positional embeddings. Then, the class (or combined class/distillation)
    token is added later in the model pipeline.

    Features:
      - Configurable grid dimensions for position embeddings.
      - Dynamic resizing of position embeddings during loading through bilinear interpolation.
      - Compatible with standard Vision Transformer, FlexiViT, DeiT, and PiT architectures.
      - Handles class and distillation token embeddings appropriately based on mode.
      - When no_embed_class=True, the layer applies positional embeddings only to patch tokens.
        If a class token is present at the beginning, it is preserved and concatenated back after
        the patch tokens are positionally embedded.

    Args:
        grid_h (int): Height of the position embedding grid.
        grid_w (int): Width of the position embedding grid.
        no_embed_class (bool):
            - If False (default), operates in standard mode where position embeddings are learned for
              the entire input (patches plus token(s)).
            - If True, operates in FlexiViT or PiT mode where positional embeddings are applied only to
              the patch tokens. In this mode the input can be either a sequence of patch tokens only
              (grid_h * grid_w tokens) or a sequence with a leading class token (grid_h * grid_w + 1 tokens).
        use_distillation (bool):
            If True, operates in DeiT mode with an additional distillation token.
            When no_embed_class is False, the expected input sequence length is grid_h * grid_w + 2.
            Defaults to False.
        name (str, optional): Name of the layer.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input Shape:
        3D tensor with shape: `(batch_size, sequence_length, embedding_dim)`, where sequence_length should be:
          * Standard mode: grid_h * grid_w + 1 (patch tokens + class token)
          * FlexiViT mode / PiT mode: either grid_h * grid_w (patch tokens only) or grid_h * grid_w + 1
          * DeiT mode: grid_h * grid_w + 2 (patch tokens + class token + distillation token)

    Output Shape:
        Same as the input shape: `(batch_size, sequence_length, embedding_dim)`. In FlexiViT or PiT mode,
        if a class token is present at the beginning, it is preserved and positional embeddings are added
        only to the patch tokens.

    Example:
        ```python
        # Standard mode (e.g., ViT with a 24x24 grid)
        layer = AddPositionEmbs(grid_h=24, grid_w=24)
        inputs = tf.random.normal((batch_size, 577, embedding_dim))  # 576 patches + 1 class token
        outputs = layer(inputs)

        # FlexiViT mode (e.g., FlexiViT with a 15x15 grid)
        layer = AddPositionEmbs(grid_h=15, grid_w=15, no_embed_class=True)
        inputs = tf.random.normal((batch_size, 226, embedding_dim))  # 225 patches + 1 class token or 225 patches only
        outputs = layer(inputs)

        # DeiT mode (e.g., DeiT with a 16x16 grid)
        layer = AddPositionEmbs(grid_h=16, grid_w=16, use_distillation=True)
        inputs = tf.random.normal((batch_size, 258, embedding_dim))  # 256 patches + 2 tokens (class and distillation)
        outputs = layer(inputs)

        # PiT mode (e.g., PiT with a 14x14 grid)
        # In PiT, set no_embed_class=True so that the patch tokens are positionally embedded first.
        layer = AddPositionEmbs(grid_h=14, grid_w=14, no_embed_class=True)
        inputs = tf.random.normal((batch_size, 197, embedding_dim))  # 196 patches + 1 class token
        outputs = layer(inputs)
        ```

    The layer supports loading weights from different grid sizes through bilinear interpolation,
    making it flexible for transfer learning and model adaptation scenarios. When loading weights,
    it automatically handles the conversion between different modes (standard / FlexiViT / DeiT),
    adjusting the position embeddings appropriately.
    """

    def __init__(
        self,
        grid_h,
        grid_w,
        no_embed_class=False,
        use_distillation=False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.resize_mode = "bilinear"

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, received: {len(input_shape)}D")

        num_patches = self.grid_h * self.grid_w

        if self.no_embed_class:
            if input_shape[1] == num_patches:
                self.skip_cls = False
            elif input_shape[1] == num_patches + 1:
                self.skip_cls = True
            else:
                raise ValueError(
                    f"Input sequence length {input_shape[1]} does not match expected length "
                    f"{num_patches} or {num_patches + 1} (grid: {self.grid_h}x{self.grid_w})"
                )
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, num_patches, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        else:
            expected_length = num_patches + (2 if self.use_distillation else 1)
            if input_shape[1] != expected_length:
                raise ValueError(
                    f"Input sequence length {input_shape[1]} does not match expected length "
                    f"{expected_length} (grid: {self.grid_h}x{self.grid_w} + tokens)"
                )
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, expected_length, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        if self.no_embed_class:
            if hasattr(self, "skip_cls") and self.skip_cls:
                cls_token = inputs[:, :1]
                patch_tokens = inputs[:, 1:]
                patch_tokens = patch_tokens + self.position_embedding
                return ops.concatenate([cls_token, patch_tokens], axis=1)
            else:
                return inputs + self.position_embedding
        elif self.use_distillation:
            tokens = inputs[:, :2]
            patch_tokens = inputs[:, 2:]
            token_pos_embed = self.position_embedding[:, :2]
            patch_pos_embed = self.position_embedding[:, 2:]
            tokens = tokens + token_pos_embed
            patch_tokens = patch_tokens + patch_pos_embed
            return ops.concatenate([tokens, patch_tokens], axis=1)
        else:
            return inputs + self.position_embedding

    def compute_output_shape(self, input_shape):
        return input_shape

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["grid_h"] = self.grid_h
        store["grid_w"] = self.grid_w
        store["no_embed_class"] = self.no_embed_class
        store["use_distillation"] = self.use_distillation

    def load_own_variables(self, store):
        source_h = int(store["grid_h"][...])
        source_w = int(store["grid_w"][...])

        try:
            source_no_embed_class = bool(store["no_embed_class"][...])
        except KeyError:
            source_no_embed_class = False

        try:
            source_use_distillation = bool(store["use_distillation"][...])
        except KeyError:
            source_use_distillation = False

        if source_h == self.grid_h and source_w == self.grid_w:
            self.position_embedding.assign(store["0"])
            return

        pos_embed = store["0"]

        if not source_no_embed_class:
            if source_use_distillation:
                spatial_pos_embed = pos_embed[:, 2:]
                token_pos_embed = pos_embed[:, :2]
            else:
                spatial_pos_embed = pos_embed[:, 1:]
                token_pos_embed = pos_embed[:, :1]
        else:
            spatial_pos_embed = pos_embed

        embed_dim = spatial_pos_embed.shape[-1]

        spatial_pos_embed = ops.cast(spatial_pos_embed, dtype="float32")
        spatial_pos_embed = ops.reshape(
            spatial_pos_embed, [1, source_h, source_w, embed_dim]
        )

        spatial_pos_embed = ops.image.resize(
            spatial_pos_embed,
            size=[self.grid_h, self.grid_w],
            interpolation=self.resize_mode,
            antialias=True,
        )

        spatial_pos_embed = ops.reshape(
            spatial_pos_embed, [1, self.grid_h * self.grid_w, embed_dim]
        )

        if self.no_embed_class:
            pos_embed = spatial_pos_embed
        else:
            if not source_no_embed_class:
                if self.use_distillation:
                    if source_use_distillation:
                        cls_dist_pos_embed = token_pos_embed
                    else:
                        cls_dist_pos_embed = ops.concatenate(
                            [token_pos_embed, token_pos_embed], axis=1
                        )
                else:
                    if source_use_distillation:
                        cls_dist_pos_embed = token_pos_embed[:, :1]
                    else:
                        cls_dist_pos_embed = token_pos_embed
            else:
                if self.use_distillation:
                    cls_dist_pos_embed = ops.zeros((1, 2, embed_dim))
                else:
                    cls_dist_pos_embed = ops.zeros((1, 1, embed_dim))

            pos_embed = ops.concatenate([cls_dist_pos_embed, spatial_pos_embed], axis=1)

        self.position_embedding.assign(pos_embed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "resize_mode": self.resize_mode,
            }
        )
        return config
