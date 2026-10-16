import keras
from keras import layers, ops


def masked_mean_pool(token_embeddings, attention_mask):
    """Attention-masked mean over the sequence (ST ``pooling_mode_mean_tokens``)."""
    mask = ops.cast(attention_mask, token_embeddings.dtype)
    mask = ops.expand_dims(mask, axis=-1)
    summed = ops.sum(token_embeddings * mask, axis=1)
    counts = ops.maximum(ops.sum(mask, axis=1), 1e-9)
    return summed / counts


def cls_pool(token_embeddings, attention_mask=None):
    """First-token pool (ST ``pooling_mode_cls_token``)."""
    del attention_mask
    return token_embeddings[:, 0]


def masked_max_pool(token_embeddings, attention_mask):
    """Attention-masked max over the sequence (ST ``pooling_mode_max_tokens``)."""
    mask = ops.cast(attention_mask, token_embeddings.dtype)
    mask = ops.expand_dims(mask, axis=-1)
    # Large negative so padded positions never win the max.
    masked = token_embeddings + (1.0 - mask) * (-1e9)
    return ops.max(masked, axis=1)


POOLERS = {
    "mean": masked_mean_pool,
    "cls": cls_pool,
    "max": masked_max_pool,
}


@keras.saving.register_keras_serializable(package="zeromodels")
class SentenceTransformerPooling(layers.Layer):
    """Mean / CLS / max pool over token embeddings with an attention mask."""

    def __init__(self, pooling="mean", **kwargs):
        super().__init__(**kwargs)
        if pooling not in POOLERS:
            raise ValueError(
                f"Unknown pooling={pooling!r}; expected one of {sorted(POOLERS)}"
            )
        self.pooling = pooling

    def call(self, inputs):
        token_embeddings, attention_mask = inputs
        if self.pooling not in POOLERS:
            raise ValueError(
                f"Unknown pooling={self.pooling!r}; expected one of {sorted(POOLERS)}"
            )
        return POOLERS[self.pooling](token_embeddings, attention_mask)

    def get_config(self):
        config = super().get_config()
        config.update({"pooling": self.pooling})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SentenceTransformerNormalize(layers.Layer):
    """L2-normalize embeddings along the last axis."""

    def __init__(self, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, embeddings):
        norm = ops.sqrt(
            ops.maximum(
                ops.sum(ops.square(embeddings), axis=-1, keepdims=True), self.eps
            )
        )
        return embeddings / norm

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config
