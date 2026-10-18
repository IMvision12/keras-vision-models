import math
from typing import Callable, NamedTuple

import keras
from keras import layers, ops

from zeromodels.models.bert.bert_model import bert_backbone
from zeromodels.models.mpnet.mpnet_model import mpnet_backbone
from zeromodels.models.roberta.roberta_model import roberta_backbone


def _attention(query, key, value, attention_mask, num_heads):
    batch = ops.shape(query)[0]
    length = ops.shape(query)[1]
    head_dim = query.shape[-1] // num_heads

    def split_heads(x):
        x = ops.reshape(x, (batch, length, num_heads, head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    q, k, v = (split_heads(x) for x in (query, key, value))
    scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
    scores = scores / math.sqrt(head_dim)
    mask = ops.cast(attention_mask[:, None, None, :], "bool")
    scores = ops.where(mask, scores, ops.cast(-1e4, scores.dtype))
    context = ops.matmul(ops.softmax(scores, axis=-1), v)
    context = ops.transpose(context, (0, 2, 1, 3))
    return ops.reshape(context, (batch, length, num_heads * head_dim))


@keras.saving.register_keras_serializable(package="zeromodels")
class DistilBertEncoderBlock(layers.Layer):
    """One post-LayerNorm DistilBERT transformer layer."""

    def __init__(
        self, embed_dim, num_heads, mlp_dim, activation="gelu", norm_eps=1e-12, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.activation = activation
        self.norm_eps = norm_eps
        self.q_lin = layers.Dense(embed_dim, name="q_lin")
        self.k_lin = layers.Dense(embed_dim, name="k_lin")
        self.v_lin = layers.Dense(embed_dim, name="v_lin")
        self.out_lin = layers.Dense(embed_dim, name="out_lin")
        self.sa_layer_norm = layers.LayerNormalization(
            epsilon=norm_eps, name="sa_layer_norm"
        )
        self.ffn_lin1 = layers.Dense(mlp_dim, name="ffn_lin1")
        self.ffn_lin2 = layers.Dense(embed_dim, name="ffn_lin2")
        self.output_layer_norm = layers.LayerNormalization(
            epsilon=norm_eps, name="output_layer_norm"
        )

    def call(self, inputs):
        x, attention_mask = inputs
        context = _attention(
            self.q_lin(x), self.k_lin(x), self.v_lin(x), attention_mask, self.num_heads
        )
        x = self.sa_layer_norm(x + self.out_lin(context))
        hidden = self.ffn_lin1(x)
        if self.activation == "gelu":
            hidden = ops.gelu(hidden, approximate=False)
        else:
            raise ValueError(f"Unsupported DistilBERT activation: {self.activation}")
        return self.output_layer_norm(x + self.ffn_lin2(hidden))

    def get_config(self):
        return {
            **super().get_config(),
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "activation": self.activation,
            "norm_eps": self.norm_eps,
        }


@keras.saving.register_keras_serializable(package="zeromodels")
class AlbertEncoderBlock(layers.Layer):
    """Shared ALBERT attention and feed-forward group."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        activation="gelu_new",
        norm_eps=1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.activation = activation
        self.norm_eps = norm_eps
        self.query = layers.Dense(embed_dim, name="query")
        self.key = layers.Dense(embed_dim, name="key")
        self.value = layers.Dense(embed_dim, name="value")
        self.attention_dense = layers.Dense(embed_dim, name="attention_dense")
        self.attention_layer_norm = layers.LayerNormalization(
            epsilon=norm_eps, name="attention_layer_norm"
        )
        self.ffn = layers.Dense(mlp_dim, name="ffn")
        self.ffn_output = layers.Dense(embed_dim, name="ffn_output")
        self.full_layer_norm = layers.LayerNormalization(
            epsilon=norm_eps, name="full_layer_norm"
        )

    def call(self, inputs):
        x, attention_mask = inputs
        context = _attention(
            self.query(x), self.key(x), self.value(x), attention_mask, self.num_heads
        )
        x = self.attention_layer_norm(x + self.attention_dense(context))
        hidden = self.ffn(x)
        if self.activation == "gelu_new":
            hidden = (
                0.5
                * hidden
                * (
                    1.0
                    + ops.tanh(
                        math.sqrt(2.0 / math.pi)
                        * (hidden + 0.044715 * ops.power(hidden, 3))
                    )
                )
            )
        elif self.activation == "gelu":
            hidden = ops.gelu(hidden, approximate=False)
        else:
            raise ValueError(f"Unsupported ALBERT activation: {self.activation}")
        return self.full_layer_norm(x + self.ffn_output(hidden))

    def get_config(self):
        return {
            **super().get_config(),
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "activation": self.activation,
            "norm_eps": self.norm_eps,
        }


def distilbert_backbone(
    input_ids,
    attention_mask,
    *,
    vocab_size,
    embed_dim,
    num_layers,
    num_heads,
    mlp_dim,
    max_position_embeddings,
    hidden_act,
    layer_norm_eps,
):
    positions = ops.cumsum(ops.ones_like(input_ids), axis=1) - 1
    x = layers.Embedding(vocab_size, embed_dim, name="distil_word_embeddings")(
        input_ids
    )
    x = x + layers.Embedding(
        max_position_embeddings, embed_dim, name="distil_position_embeddings"
    )(positions)
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, name="distil_embeddings_norm"
    )(x)
    for index in range(num_layers):
        x = DistilBertEncoderBlock(
            embed_dim,
            num_heads,
            mlp_dim,
            hidden_act,
            layer_norm_eps,
            name=f"distil_layer_{index}",
        )([x, attention_mask])
    return x


def albert_backbone(
    input_ids,
    attention_mask,
    token_type_ids,
    *,
    vocab_size,
    embed_dim,
    embedding_size,
    num_layers,
    num_heads,
    mlp_dim,
    max_position_embeddings,
    type_vocab_size,
    hidden_act,
    layer_norm_eps,
    num_hidden_groups=1,
    inner_group_num=1,
):
    if num_hidden_groups != 1 or inner_group_num != 1:
        raise ValueError(
            "Only ALBERT with one shared hidden group and one inner group is supported"
        )
    positions = ops.cumsum(ops.ones_like(input_ids), axis=1) - 1
    x = layers.Embedding(vocab_size, embedding_size, name="albert_word_embeddings")(
        input_ids
    )
    x = x + layers.Embedding(
        max_position_embeddings, embedding_size, name="albert_position_embeddings"
    )(positions)
    x = x + layers.Embedding(
        type_vocab_size, embedding_size, name="albert_token_type_embeddings"
    )(token_type_ids)
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, name="albert_embeddings_norm"
    )(x)
    x = layers.Dense(embed_dim, name="albert_embedding_projection")(x)
    shared = AlbertEncoderBlock(
        embed_dim,
        num_heads,
        mlp_dim,
        hidden_act,
        layer_norm_eps,
        name="albert_shared_block",
    )
    for _ in range(num_layers):
        x = shared([x, attention_mask])
    return x


def encoder_kwargs(config):
    """The encoder hyperparameters every backbone builder takes."""
    return {
        "vocab_size": config.vocab_size,
        "embed_dim": config.embed_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "mlp_dim": config.mlp_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_act": config.hidden_act,
        "layer_norm_eps": config.layer_norm_eps,
    }


def build_mpnet(inputs, config):
    sequence_output, _ = mpnet_backbone(
        inputs["input_ids"],
        inputs["attention_mask"],
        **encoder_kwargs(config),
        relative_attention_num_buckets=config.relative_attention_num_buckets,
        pad_token_id=config.pad_token_id,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        add_pooler=False,
    )
    return sequence_output


def build_bert(inputs, config):
    sequence_output, _ = bert_backbone(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"],
        **encoder_kwargs(config),
        type_vocab_size=config.type_vocab_size,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        add_pooler=False,
    )
    return sequence_output


def build_roberta(inputs, config):
    sequence_output, _ = roberta_backbone(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"],
        **encoder_kwargs(config),
        type_vocab_size=config.type_vocab_size,
        pad_token_id=config.pad_token_id,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        add_pooler=False,
    )
    return sequence_output


def build_distilbert(inputs, config):
    return distilbert_backbone(
        inputs["input_ids"], inputs["attention_mask"], **encoder_kwargs(config)
    )


def build_albert(inputs, config):
    return albert_backbone(
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"],
        **encoder_kwargs(config),
        embedding_size=config.embedding_size or config.embed_dim,
        type_vocab_size=config.type_vocab_size,
        num_hidden_groups=config.num_hidden_groups,
        inner_group_num=config.inner_group_num,
    )


class SentenceTransformerBackbone(NamedTuple):
    """One encoder ``ZMSentenceTransformer`` can be built on.

    ``build(inputs, config)`` wires the encoder into the functional graph and
    returns the token embeddings; ``token_type_ids`` says whether the encoder (and
    so the model and the tokenizer) takes segment ids.
    """

    build: Callable
    token_type_ids: bool


# backbone_type -> encoder. MPNet / BERT / RoBERTa / XLM-R reuse the zeromodels.models
# encoders; DistilBERT and ALBERT are built above.
SENTENCE_TRANSFORMER_BACKBONES = {
    "mpnet": SentenceTransformerBackbone(build_mpnet, token_type_ids=False),
    "bert": SentenceTransformerBackbone(build_bert, token_type_ids=True),
    "roberta": SentenceTransformerBackbone(build_roberta, token_type_ids=True),
    "xlm-roberta": SentenceTransformerBackbone(build_roberta, token_type_ids=True),
    "distilbert": SentenceTransformerBackbone(build_distilbert, token_type_ids=False),
    "albert": SentenceTransformerBackbone(build_albert, token_type_ids=True),
}


def get_backbone(backbone_type):
    backbone = SENTENCE_TRANSFORMER_BACKBONES.get(backbone_type)
    if backbone is None:
        raise ValueError(
            f"Unsupported sentence embedding backbone: {backbone_type!r}; expected "
            f"one of {sorted(SENTENCE_TRANSFORMER_BACKBONES)}."
        )
    return backbone
