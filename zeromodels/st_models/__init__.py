"""Compatibility aliases for the renamed ``zeromodels.sentence_transformers`` API."""

import importlib
import sys

from zeromodels.sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerConfig,
    SentenceTransformerModel,
    SentenceTransformerTextEmbed,
    SentenceTransformerTokenizer,
    ZMSentenceTransformer,
    ZMSentenceTransformerTokenizer,
    sentence_transformer,
)

# Saved configs and deep imports must resolve to the canonical classes, not
# load a second copy with a different module name or Keras registration.
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
        f"{sentence_transformer.__name__}.{_name}"
    )

__all__ = [
    "SentenceTransformer",
    "SentenceTransformerConfig",
    "SentenceTransformerModel",
    "SentenceTransformerTextEmbed",
    "SentenceTransformerTokenizer",
    "ZMSentenceTransformer",
    "ZMSentenceTransformerTokenizer",
]
