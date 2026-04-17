import os
from typing import List, Union

import keras
import numpy as np

from kmodels.weight_utils import download_file

from .config import (
    METACLIP2_BOS_TOKEN_ID,
    METACLIP2_EOS_TOKEN_ID,
    METACLIP2_PAD_TOKEN_ID,
    METACLIP2_UNK_TOKEN_ID,
)

DEFAULT_SENTENCEPIECE_URL = (
    "https://huggingface.co/facebook/metaclip-2-worldwide-b32/resolve/main/"
    "sentencepiece.bpe.model"
)


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2Tokenizer(keras.Layer):
    """XLM-RoBERTa SentencePiece tokenizer wrapper for MetaCLIP 2.

    Wraps ``sentencepiece.SentencePieceProcessor`` to match the exact
    tokenization used by HuggingFace's ``XLMRobertaTokenizer`` on the
    MetaCLIP 2 checkpoints (multilingual 901629-token vocab).

    Args:
        sentencepiece_model_file: Path to the ``sentencepiece.bpe.model`` file.
            When ``None``, downloads it from the default MetaCLIP 2 hub repo.
        context_length: Maximum sequence length. MetaCLIP 2 uses ``77``.
        bos_token_id / eos_token_id / pad_token_id / unk_token_id: Override
            special-token ids. Defaults match XLM-R.
    """

    def __init__(
        self,
        sentencepiece_model_file: str = None,
        context_length: int = 77,
        bos_token_id: int = METACLIP2_BOS_TOKEN_ID,
        eos_token_id: int = METACLIP2_EOS_TOKEN_ID,
        pad_token_id: int = METACLIP2_PAD_TOKEN_ID,
        unk_token_id: int = METACLIP2_UNK_TOKEN_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)

        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "MetaClip2Tokenizer requires the `sentencepiece` package. "
                "Install with `pip install sentencepiece`."
            ) from exc

        if sentencepiece_model_file is None:
            sentencepiece_model_file = download_file(DEFAULT_SENTENCEPIECE_URL)
        if not os.path.exists(sentencepiece_model_file):
            raise FileNotFoundError(sentencepiece_model_file)

        self.sentencepiece_model_file = sentencepiece_model_file
        self.context_length = context_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(sentencepiece_model_file)

        self.fairseq_offset = 1

    def _encode_one(self, text: str) -> List[int]:
        pieces = self._sp.encode(text, out_type=int)
        ids = [p + self.fairseq_offset for p in pieces]
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if len(ids) > self.context_length:
            ids = ids[: self.context_length - 1] + [self.eos_token_id]
        pad_len = self.context_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.pad_token_id] * pad_len
        return ids, attention_mask

    def call(self, inputs: Union[str, List[str]]):
        if isinstance(inputs, str):
            texts = [inputs]
        else:
            texts = list(inputs)
        ids = np.zeros((len(texts), self.context_length), dtype=np.int32)
        mask = np.zeros((len(texts), self.context_length), dtype=np.int32)
        for i, t in enumerate(texts):
            row_ids, row_mask = self._encode_one(t)
            ids[i] = row_ids
            mask[i] = row_mask
        return {
            "token_ids": keras.ops.convert_to_tensor(ids, dtype="int32"),
            "padding_mask": keras.ops.convert_to_tensor(mask, dtype="int32"),
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
                int(i) - self.fairseq_offset
                for i in row
                if int(i)
                not in (
                    self.bos_token_id,
                    self.eos_token_id,
                    self.pad_token_id,
                )
                and int(i) >= self.fairseq_offset
            ]
            out.append(self._sp.decode(keep))
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sentencepiece_model_file": self.sentencepiece_model_file,
                "context_length": self.context_length,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
            }
        )
        return config
