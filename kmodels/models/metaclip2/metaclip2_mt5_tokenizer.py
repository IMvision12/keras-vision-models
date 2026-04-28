import os
import string
from typing import List, Union

import keras
import numpy as np

from kmodels.base import BaseTokenizer
from kmodels.weight_utils import download_file

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

METACLIP2_MT5_EOS_TOKEN_ID = 1
METACLIP2_MT5_PAD_TOKEN_ID = 1
METACLIP2_MT5_UNK_TOKEN_ID = 2

DEFAULT_MT5_SENTENCEPIECE_URL = (
    "https://github.com/IMvision12/keras-models/releases/download/metaclip-2/"
    "spiece.model"
)


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2Mt5Tokenizer(BaseTokenizer):
    """SigLIP-style tokenizer for the MetaCLIP 2 mT5 variants.

    Loads a SentencePiece model file (``spiece.model``) and replicates
    HF ``SiglipTokenizer`` exactly: **lowercase** the text, then SP-encode,
    append ``eos_token_id=1``, pad with ``pad_token_id=1`` (same as eos).
    No BOS token. Vocab size is 250 100.

    Args:
        sentencepiece_model_file: Path to ``spiece.model``. When ``None``,
            downloads from the default MetaCLIP 2 release URL.
        context_length: Maximum sequence length. Defaults to ``77``.
    """

    def __init__(
        self,
        sentencepiece_model_file: str = None,
        context_length: int = 77,
        eos_token_id: int = METACLIP2_MT5_EOS_TOKEN_ID,
        pad_token_id: int = METACLIP2_MT5_PAD_TOKEN_ID,
        unk_token_id: int = METACLIP2_MT5_UNK_TOKEN_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "MetaClip2Mt5Tokenizer requires the `sentencepiece` package. "
                "Install with `pip install sentencepiece`."
            ) from exc

        if sentencepiece_model_file is None:
            sentencepiece_model_file = download_file(DEFAULT_MT5_SENTENCEPIECE_URL)
        if not os.path.exists(sentencepiece_model_file):
            raise FileNotFoundError(sentencepiece_model_file)

        self.sentencepiece_model_file = sentencepiece_model_file
        self.context_length = context_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(sentencepiece_model_file)

    def _encode_one(self, text: str):
        text = text.lower().translate(_PUNCT_TABLE)
        ids = self._sp.encode(text, out_type=int)
        ids = ids + [self.eos_token_id]
        if len(ids) > self.context_length:
            ids = ids[: self.context_length - 1] + [self.eos_token_id]
        attention_mask = [1] * len(ids) + [0] * (self.context_length - len(ids))
        ids = ids + [self.pad_token_id] * (self.context_length - len(ids))
        return ids, attention_mask

    def call(self, inputs: Union[str, List[str]]):
        if isinstance(inputs, str):
            texts = [inputs]
        else:
            texts = list(inputs)
        token_ids = np.zeros((len(texts), self.context_length), dtype=np.int32)
        attention_mask = np.zeros((len(texts), self.context_length), dtype=np.int32)
        for i, t in enumerate(texts):
            row_ids, row_mask = self._encode_one(t)
            token_ids[i] = row_ids
            attention_mask[i] = row_mask
        return {
            "token_ids": keras.ops.convert_to_tensor(token_ids, dtype="int32"),
            "padding_mask": keras.ops.convert_to_tensor(attention_mask, dtype="int32"),
        }

    def decode(self, ids) -> List[str]:
        if hasattr(ids, "numpy"):
            ids = ids.numpy()
        ids = np.asarray(ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        out = []
        for row in ids:
            keep = [
                int(i)
                for i in row
                if int(i) not in (self.eos_token_id, self.pad_token_id)
            ]
            out.append(self._sp.decode(keep))
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sentencepiece_model_file": self.sentencepiece_model_file,
                "context_length": self.context_length,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
            }
        )
        return config
