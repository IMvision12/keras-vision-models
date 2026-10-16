"""Sentence Transformer models and tokenizers for the ZeroModels embedding API."""

import importlib
import sys

from .sentence_transformer_config import SentenceTransformerConfig
from .sentence_transformer_model import (
    SentenceTransformerTextEmbed,
    ZMSentenceTransformer,
)
from .sentence_transformer_tokenizer import ZMSentenceTransformerTokenizer

SentenceTransformer = ZMSentenceTransformer
SentenceTransformerModel = ZMSentenceTransformer
SentenceTransformerTokenizer = ZMSentenceTransformerTokenizer

# Keep the former nested import paths usable without a nested implementation.
sentence_transformer = sys.modules[__name__]
sys.modules[f"{__name__}.sentence_transformer"] = sentence_transformer
for _name in (
    "convert_sentence_transformer_hf_to_keras",
    "sentence_transformer_backbones",
    "sentence_transformer_config",
    "sentence_transformer_layers",
    "sentence_transformer_model",
    "sentence_transformer_tokenizer",
):
    sys.modules[f"{__name__}.sentence_transformer.{_name}"] = importlib.import_module(
        f"{__name__}.{_name}"
    )

__all__ = [
    "SentenceTransformerConfig",
    "SentenceTransformerTextEmbed",
    "ZMSentenceTransformer",
    "ZMSentenceTransformerTokenizer",
]
