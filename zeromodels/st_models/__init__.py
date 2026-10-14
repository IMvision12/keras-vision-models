"""Sentence-embedding models, separate from the general model zoo."""

from .sentence_transformer import (
    SentenceTransformerConfig,
    SentenceTransformerModel,
    SentenceTransformerTextEmbed,
    SentenceTransformerTokenizer,
)

# The compact public API mirrors sentence-transformers' main class name while
# preserving the existing ZeroModels class and Hub model_class metadata.
SentenceTransformer = SentenceTransformerModel

__all__ = [
    "SentenceTransformer",
    "SentenceTransformerConfig",
    "SentenceTransformerModel",
    "SentenceTransformerTextEmbed",
    "SentenceTransformerTokenizer",
]
