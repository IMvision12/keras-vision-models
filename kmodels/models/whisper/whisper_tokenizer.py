import json
import os
from typing import List, Union

import keras
import numpy as np
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from kmodels.base import BaseTokenizer
from kmodels.weight_utils import download_file

from .config import WHISPER_TOKENIZER_FILES


@keras.saving.register_keras_serializable(package="kmodels")
class WhisperTokenizer(BaseTokenizer):
    """Whisper byte-level BPE tokenizer, built on HuggingFace ``tokenizers`` (Rust).

    Matches HF ``WhisperTokenizer`` exactly for text encoding / decoding.
    The pipeline is: ByteLevel pre-tokenizer + BPE(vocab.json, merges.txt) +
    ByteLevel decoder, plus the ~1607 Whisper special tokens (languages,
    timestamps, task tokens) registered via ``add_special_tokens``.

    Vocab files are pulled from the ``whisper`` release tag on
    ``github.com/IMvision12/keras-models`` unless explicit paths are given.

    Args:
        variant: Which tokenizer set to use.
            * ``"v1"`` -> tiny / base / small / medium / large / large-v2
              (51865 vocab).
            * ``"v3"`` -> large-v3 / large-v3-turbo (51866 vocab).
        vocab_file / merges_file / added_tokens_file: Optional explicit
            paths. When ``None``, the file is downloaded from the kmodels
            release URL corresponding to ``variant``.
        bos_token_id / eos_token_id / pad_token_id: Whisper special ids.
    """

    def __init__(
        self,
        variant: str = "v1",
        vocab_file: str = None,
        merges_file: str = None,
        added_tokens_file: str = None,
        bos_token_id: int = 50257,
        eos_token_id: int = 50257,
        pad_token_id: int = 50257,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if variant not in WHISPER_TOKENIZER_FILES:
            raise ValueError(
                f"Unknown Whisper tokenizer variant {variant!r}. "
                f"Available: {list(WHISPER_TOKENIZER_FILES)}"
            )
        urls = WHISPER_TOKENIZER_FILES[variant]
        if vocab_file is None or not os.path.exists(vocab_file):
            vocab_file = download_file(urls["vocab"])
        if merges_file is None or not os.path.exists(merges_file):
            merges_file = download_file(urls["merges"])
        if added_tokens_file is None or not os.path.exists(added_tokens_file):
            added_tokens_file = download_file(urls["added_tokens"])

        self.variant = variant
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.added_tokens_file = added_tokens_file
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        with open(added_tokens_file, "r", encoding="utf-8") as f:
            self.added_tokens = json.load(f)

        tok = Tokenizer(
            BPE(
                vocab=vocab_file,
                merges=merges_file,
                unk_token="<|endoftext|>",
                fuse_unk=False,
            )
        )
        tok.pre_tokenizer = ByteLevel(
            add_prefix_space=False, trim_offsets=True, use_regex=True
        )
        tok.decoder = ByteLevelDecoder(
            add_prefix_space=True, trim_offsets=True, use_regex=True
        )
        tok.add_special_tokens(
            [
                AddedToken(content, special=True, normalized=False)
                for content in self.added_tokens.keys()
            ]
        )
        self._tok = tok

        self._special_id_set = set(self.added_tokens.values())
        self._special_id_set.add(eos_token_id)
        self._special_id_set.add(pad_token_id)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=False).ids
        encs = self._tok.encode_batch(text, add_special_tokens=False)
        return [e.ids for e in encs]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids = [int(i) for i in token_ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_id_set]
        return self._tok.decode(ids, skip_special_tokens=False)

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()
        out = []
        for row in token_ids_batch:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            out.append(self.decode(row, skip_special_tokens=skip_special_tokens))
        return out

    def call(self, inputs: Union[str, List[str]]):
        if inputs is None:
            raise ValueError("No text inputs provided to WhisperTokenizer")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        lens = [len(e.ids) for e in encs]
        max_len = max(lens) if lens else 0
        ids = np.full((len(texts), max_len), self.pad_token_id, dtype=np.int32)
        mask = np.zeros((len(texts), max_len), dtype=np.int32)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        return {
            "input_ids": keras.ops.convert_to_tensor(ids, dtype="int32"),
            "attention_mask": keras.ops.convert_to_tensor(mask, dtype="int32"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
                "added_tokens_file": self.added_tokens_file,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
        )
        return config
