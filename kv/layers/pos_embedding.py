from keras import layers, ops


class AddPositionEmbs(layers.Layer):
    """
    A custom Keras layer that adds position embeddings to input tensors with advanced interpolation.

    This layer creates learnable position embeddings and supports interpolation of these
    embeddings to handle variable patch sizes. It's particularly useful in vision transformers
    and similar architectures that require positional encoding.

    Attributes:
        interpolate_mode (str): Interpolation mode for resizing position embeddings.
            Defaults to 'bilinear'.
        num_patches (int): Total number of patches, used for position embedding calculation.
            Defaults to 576.

    Args:
        name (str, optional): Name of the layer. Defaults to None.
        num_patches (int, optional): Number of patches in the input. Defaults to 576.
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Methods:
        build(input_shape):
            Creates learnable position embeddings based on the input shape.
        interpolate_pos_encoding(height, width, num_patches):
            Interpolates position embeddings to match the current number of patches.
        call(inputs):
            Adds interpolated position embeddings to the input tensor.
        get_config():
            Returns the layer configuration.

    Raises:
        ValueError: If the input tensor does not have exactly 3 dimensions.

    Example:
        >>> layer = AddPositionEmbs(num_patches=576)
        >>> input_tensor = tf.random.normal((batch_size, 577, embedding_dim))
        >>> output = layer(input_tensor)  # Adds position embeddings to input
    """

    def __init__(self, name=None, num_patches=576, **kwargs):
        super().__init__(name=name, **kwargs)
        self.interpolate_mode = "bilinear"
        self.num_patches = num_patches

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"Number of dimensions should be 3, got {len(input_shape)}"
            )

        num_patches = 576
        self.pe = self.add_weight(
            name="pos_embed",
            shape=(1, num_patches + 1, input_shape[2]),
            initializer="random_normal",
            trainable=True,
        )

    def interpolate_pos_encoding(self, height, width, num_patches):
        cls_token_embed = self.pe[:, :1, :]
        pos_embed = self.pe[:, 1:, :]
        gs_h = gs_w = int((self.num_patches) ** 0.5)
        gh = gw = int(num_patches**0.5)
        pos_embed = ops.reshape(pos_embed, (1, gs_h, gs_w, -1))
        pos_embed = ops.image.resize(
            pos_embed,
            size=[gh, gw],
            interpolation=self.interpolate_mode,
            antialias=False,
        )
        pos_embed = ops.reshape(pos_embed, (1, gh * gw, -1))
        pos_embed = ops.concatenate([cls_token_embed, pos_embed], axis=1)
        return pos_embed

    def call(self, inputs):
        num_patches = inputs.shape[1] - 1
        pos_embed = self.interpolate_pos_encoding(None, None, num_patches)
        return inputs + pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({"interpolate_mode": self.interpolate_mode})
        return config
