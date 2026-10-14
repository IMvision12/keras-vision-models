"""Compatibility imports for Sentence Transformer models moved to ``st_models``."""

import sys

from zeromodels.st_models import (
    SentenceTransformerConfig,
    SentenceTransformerModel,
    SentenceTransformerTextEmbed,
    SentenceTransformerTokenizer,
)
from zeromodels.st_models.sentence_transformer import (
    convert_sentence_transformer_hf_to_keras as _converter,
)
from zeromodels.st_models.sentence_transformer import (
    sentence_transformer_backbones as _backbones,
)
from zeromodels.st_models.sentence_transformer import (
    sentence_transformer_config as _config,
)
from zeromodels.st_models.sentence_transformer import (
    sentence_transformer_layers as _layers,
)
from zeromodels.st_models.sentence_transformer import (
    sentence_transformer_model as _model,
)
from zeromodels.st_models.sentence_transformer import (
    sentence_transformer_tokenizer as _tokenizer,
)

# Old deep import paths may appear in user code and saved Keras objects. Keep
# them importable without retaining a second copy of the implementation.
for _name, _module in {
    "convert_sentence_transformer_hf_to_keras": _converter,
    "sentence_transformer_backbones": _backbones,
    "sentence_transformer_config": _config,
    "sentence_transformer_layers": _layers,
    "sentence_transformer_model": _model,
    "sentence_transformer_tokenizer": _tokenizer,
}.items():
    sys.modules[f"{__name__}.{_name}"] = _module

__all__ = [
    "SentenceTransformerConfig",
    "SentenceTransformerModel",
    "SentenceTransformerTextEmbed",
    "SentenceTransformerTokenizer",
]
