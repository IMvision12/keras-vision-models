from keras import layers, ops


class AddPositionEmbs(layers.Layer):
    def __init__(
        self,
        name=None,
        num_patches=None,
        no_embed_class=False,  # FlexiViT parameter
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.interpolate_mode = "bilinear"
        if no_embed_class:
            self.num_patches = 225
        else:
            self.num_patches = 576

        self.no_embed_class = no_embed_class

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"Number of dimensions should be 3, got {len(input_shape)}"
            )

        if self.no_embed_class:
            # FlexiViT: only create embeddings for patches, not class token
            self.pe = self.add_weight(
                name="pos_embed",
                shape=(1, self.num_patches, input_shape[2]),  # Use pretrained size
                initializer="random_normal",
                trainable=True,
            )
        else:
            # Original: create embeddings for both patches and class token
            self.pe = self.add_weight(
                name="pos_embed",
                shape=(1, self.num_patches + 1, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )

    def interpolate_pos_encoding(self, height, width, num_patches):
        if self.no_embed_class:
            pos_embed = self.pe
            gs_h = gs_w = int(self.num_patches**0.5)
            gh = gw = int(num_patches**0.5)
            pos_embed = ops.reshape(pos_embed, (1, gs_h, gs_w, -1))
            pos_embed = ops.image.resize(
                pos_embed,
                size=[gh, gw],
                interpolation=self.interpolate_mode,
                antialias=False,
            )
            pos_embed = ops.reshape(pos_embed, (1, gh * gw, -1))
            return pos_embed
        else:
            cls_token_embed = self.pe[:, :1, :]
            pos_embed = self.pe[:, 1:, :]

            gs_h = gs_w = int(self.num_patches**0.5)
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
        if self.no_embed_class:
            cls_token = inputs[:, :1]
            patch_tokens = inputs[:, 1:]
            num_patches = patch_tokens.shape[1]

            pos_embed = self.interpolate_pos_encoding(None, None, num_patches)
            patch_tokens = patch_tokens + pos_embed

            return ops.concatenate([cls_token, patch_tokens], axis=1)
        else:
            num_patches = inputs.shape[1] - 1
            pos_embed = self.interpolate_pos_encoding(None, None, num_patches)
            return inputs + pos_embed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "interpolate_mode": self.interpolate_mode,
                "no_embed_class": self.no_embed_class,
            }
        )
        return config
