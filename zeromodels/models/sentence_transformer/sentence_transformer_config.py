from typing import Optional

from zeromodels.base import BaseConfig


class SentenceTransformerConfig(BaseConfig):
    """Configuration for :class:`SentenceTransformerModel`.

    A Sentence-Transformers-style stack: encoder backbone + pooling (+ optional
    dense projection) + optional L2 normalize. ``backbone_type`` selects which
    ZeroModels encoder family builds the tower.

    Fields mirror the flat model constructor and serialize into ``zm_config.json``.
    """

    model_type = "sentence_transformer"

    backbone_type: str = "mpnet"
    vocab_size: int = 30527
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    max_position_embeddings: int = 514
    relative_attention_num_buckets: int = 32
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5  # matches microsoft/mpnet-base / ST hubs
    pad_token_id: int = 1
    type_vocab_size: int = 2
    embedding_size: Optional[int] = None
    num_hidden_groups: int = 1
    inner_group_num: int = 1
    dropout: float = 0.0
    attention_dropout: float = 0.0

    pooling: str = "mean"  # mean | cls | max
    normalize_embeddings: bool = True
    projection_dim: Optional[int] = None
    projection_activation: str = "linear"
    similarity_fn_name: str = "cosine"
    max_seq_len: int = 384  # all-mpnet-base-v2 sentence_bert_config.json
