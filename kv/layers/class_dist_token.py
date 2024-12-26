from keras import layers, ops


class ClassDistToken(layers.Layer):
    """
    Implements learnable class and distillation tokens for Vision Transformer (ViT) and
    Data-efficient image Transformer (DeiT) architectures.

    This layer can operate in two modes:
    1. Standard ViT mode: Only adds a class token
    2. DeiT mode: Adds both a class token and a distillation token

    Args:
        use_distillation (bool): If True, adds both class and distillation tokens (DeiT mode).
            If False, only adds class token (ViT mode). Defaults to False.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Example:
        ```python
        # Standard ViT mode
        layer = ClassToken(use_distillation=False)
        x = tf.random.normal((batch_size, 196, 768))  # 14x14 patches
        output = layer(x)  # Shape: (batch_size, 197, 768)

        # DeiT mode
        layer = ClassToken(use_distillation=True)
        x = tf.random.normal((batch_size, 196, 768))  # 14x14 patches
        output = layer(x)  # Shape: (batch_size, 198, 768)
        ```
    """

    def __init__(self, use_distillation=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.use_distillation = use_distillation

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        # Class token
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        # Distillation token for DeiT
        if self.use_distillation:
            self.dist = self.add_weight(
                name="dist_token",
                shape=(1, 1, self.hidden_size),
                initializer="zeros",
                trainable=True,
            )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        cls_broadcasted = ops.broadcast_to(self.cls, [batch_size, 1, self.hidden_size])

        if self.use_distillation:
            dist_broadcasted = ops.broadcast_to(
                self.dist, [batch_size, 1, self.hidden_size]
            )
            return ops.concatenate([cls_broadcasted, dist_broadcasted, inputs], axis=1)
        else:
            return ops.concatenate([cls_broadcasted, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"use_distillation": self.use_distillation})
        return config
