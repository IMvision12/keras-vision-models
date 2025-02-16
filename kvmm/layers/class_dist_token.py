import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kvmm")
class ClassDistToken(layers.Layer):
    """
    Implements learnable class and distillation tokens for Vision Transformer (ViT),
    Data-efficient image Transformer (DeiT), and Pyramid Vision Transformer (PiT) architectures.

    This layer can operate in three modes:
    1. Standard ViT mode: Only adds a class token
    2. DeiT mode: Adds separate class and distillation tokens
    3. PiT mode: Adds combined class and distillation tokens in a single weight tensor

    Args:
        use_distillation (bool): If True, adds distillation token(s) alongside class token(s).
            Defaults to False.
        combine_tokens (bool): If True, stores class and distillation tokens in a single weight
            tensor (PiT style). If False, uses separate weight tensors (ViT/DeiT style).
            Only applies when use_distillation=True. Defaults to False.
        name (str, optional): Name for the layer instance.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Example:
        ```python
        # Standard ViT mode (class token only)
        layer = ClassDistToken(use_distillation=False)
        x = tf.random.normal((batch_size, 196, 768))  # 14x14 patches
        output = layer(x)  # Shape: (batch_size, 197, 768)

        # DeiT mode (separate class and distillation tokens)
        layer = ClassDistToken(use_distillation=True, combine_tokens=False)
        x = tf.random.normal((batch_size, 196, 768))  # 14x14 patches
        output = layer(x)  # Shape: (batch_size, 198, 768)

        # PiT mode (combined class and distillation tokens)
        layer = ClassDistToken(use_distillation=True, combine_tokens=True)
        x = tf.random.normal((batch_size, 196, 768))  # 14x14 patches
        output = layer(x)  # Shape: (batch_size, 198, 768)
        ```

    References:
        - ViT: https://arxiv.org/abs/2010.11929
        - DeiT: https://arxiv.org/abs/2012.12877
        - PiT: https://arxiv.org/abs/2103.14030
    """

    def __init__(
        self, use_distillation=False, combine_tokens=False, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.use_distillation = use_distillation
        self.combine_tokens = combine_tokens
        self.num_tokens = 2 if use_distillation else 1

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]

        if self.combine_tokens and self.use_distillation:
            # Combined tokens (PiT-style)
            self.tokens = self.add_weight(
                name="cls_token",
                shape=(1, 2, self.hidden_size),
                initializer="zeros",
                trainable=True,
            )
        else:
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
        if self.combine_tokens and self.use_distillation:
            tokens_broadcasted = ops.broadcast_to(
                self.tokens, [batch_size, 2, self.hidden_size]
            )
            return ops.concatenate([tokens_broadcasted, inputs], axis=1)
        else:
            cls_broadcasted = ops.broadcast_to(
                self.cls, [batch_size, 1, self.hidden_size]
            )

            if self.use_distillation:
                dist_broadcasted = ops.broadcast_to(
                    self.dist, [batch_size, 1, self.hidden_size]
                )
                return ops.concatenate(
                    [cls_broadcasted, dist_broadcasted, inputs], axis=1
                )
            else:
                return ops.concatenate([cls_broadcasted, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "use_distillation": self.use_distillation,
                "combine_tokens": self.combine_tokens,
            }
        )
        return config
