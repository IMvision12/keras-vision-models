from keras import layers, ops


class AddPositionEmbs(layers.Layer):
    """
    A custom Keras layer that adds learnable position embeddings to input tensors with support for
    flexible grid sizes and optional class token handling.

    The layer supports two modes of operation:
    1. Standard mode: Creates position embeddings for both patches and class token
    2. FlexiViT mode: Creates embeddings only for patches, handling class token separately

    Features:
    - Configurable grid dimensions for position embeddings
    - Dynamic resizing of position embeddings during loading
    - Bilinear interpolation support for smooth resizing
    - Compatible with both standard Vision Transformer and FlexiViT architectures
    - Handles class token embeddings appropriately based on mode

    Args:
        grid_h (int): Height of the position embedding grid
        grid_w (int): Width of the position embedding grid
        no_embed_class (bool): If True, operates in FlexiViT mode where class token is handled
            separately. If False, uses standard mode where position embeddings include class token.
            Defaults to False.
        name (str, optional): Name of the layer
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input Shape:
        3D tensor with shape: `(batch_size, sequence_length, embedding_dim)`
        - sequence_length should be grid_h * grid_w + 1 (including class token)
        - embedding_dim is determined by the input shape

    Output Shape:
        Same as input shape: `(batch_size, sequence_length, embedding_dim)`

    Example:
        ```python
        # Standard mode (24x24 grid)
        layer = AddPositionEmbs(grid_h=24, grid_w=24, no_embed_class=False)
        inputs = tf.random.normal((batch_size, 577, embedding_dim))  # 576 patches + 1 class token
        outputs = layer(inputs)

        # FlexiViT mode (15x15 grid)
        layer = AddPositionEmbs(grid_h=15, grid_w=15, no_embed_class=True)
        inputs = tf.random.normal((batch_size, 226, embedding_dim))  # 225 patches + 1 class token
        outputs = layer(inputs)
        ```

    The layer supports loading weights from different grid sizes through bilinear interpolation,
    making it flexible for transfer learning and model adaptation scenarios.
    """

    def __init__(self, grid_h, grid_w, no_embed_class=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.no_embed_class = no_embed_class
        self.resize_mode = "bilinear"

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, received: {len(input_shape)}D")

        num_patches = self.grid_h * self.grid_w

        # Input sequence length should be num_patches + 1 (for class token)
        if input_shape[1] != num_patches + 1:
            raise ValueError(
                f"Input sequence length {input_shape[1]} does not match expected length "
                f"{num_patches + 1} (grid: {self.grid_h}x{self.grid_w} + class token)"
            )

        if self.no_embed_class:
            # For FlexiViT
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, num_patches, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        else:
            # For standard ViT
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, input_shape[1], input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )

    def call(self, inputs):
        if self.no_embed_class:
            cls_token = inputs[:, :1]
            patch_tokens = inputs[:, 1:]
            patch_tokens = patch_tokens + self.position_embedding
            return ops.concatenate([cls_token, patch_tokens], axis=1)
        else:
            return inputs + self.position_embedding

    def compute_output_shape(self, input_shape):
        return input_shape

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["grid_h"] = self.grid_h
        store["grid_w"] = self.grid_w
        store["no_embed_class"] = self.no_embed_class

    def load_own_variables(self, store):
        source_h = int(store["grid_h"][...])
        source_w = int(store["grid_w"][...])

        try:
            source_no_embed_class = bool(store["no_embed_class"][...])
        except KeyError:
            source_no_embed_class = False

        if source_h == self.grid_h and source_w == self.grid_w:
            self.position_embedding.assign(store["0"])
            return

        pos_embed = store["0"]

        if not source_no_embed_class:
            spatial_pos_embed = pos_embed[:, 1:]
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
                cls_pos_embed = pos_embed[:, :1]
            else:
                cls_pos_embed = ops.zeros((1, 1, embed_dim))
            pos_embed = ops.concatenate([cls_pos_embed, spatial_pos_embed], axis=1)

        self.position_embedding.assign(pos_embed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "no_embed_class": self.no_embed_class,
                "resize_mode": self.resize_mode,
            }
        )
        return config
