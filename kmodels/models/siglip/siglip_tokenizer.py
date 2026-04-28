import string
from typing import Dict, List, Optional, Union

import keras
import numpy as np
import sentencepiece as spm
from keras import ops
from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.normalizers import Lowercase, Replace, Strip
from tokenizers.normalizers import Sequence as NormSeq
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing

from kmodels.base import BaseTokenizer

_PUNCT_CHARS = "".join(
    f"\\{c}" if c in r"\^$.|?*+()[]{}" else c for c in string.punctuation
)
_PUNCT_REGEX = f"[{_PUNCT_CHARS}]"


@keras.saving.register_keras_serializable(package="kmodels")
class SigLIPTokenizer(BaseTokenizer):
    """SigLIP SentencePiece Unigram tokenizer, built on HuggingFace `tokenizers`.

    Matches HF ``SiglipTokenizer`` exactly: when ``do_lower_case=True``
    (HF default) it canonicalizes text (strip ASCII punctuation, collapse
    whitespace, strip), then SP-encodes, then appends EOS. The SP model
    itself case-folds (nmt_nfkc_cf).

    Args:
        vocab_file: Path to the SentencePiece ``.model`` file.
        context_length: Maximum sequence length (default 64).
        do_lower_case: When True, apply HF's ``canonicalize_text``.
        unk_token / pad_token / eos_token: Special token strings.
    """

    def __init__(
        self,
        vocab_file: str,
        context_length: int = 64,
        do_lower_case: bool = False,
        unk_token: str = "<unk>",
        pad_token: str = "</s>",
        eos_token: str = "</s>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file
        self.context_length = context_length
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(vocab_file)

        vocab = [
            (self.sp_model.id_to_piece(i), self.sp_model.get_score(i))
            for i in range(self.sp_model.get_piece_size())
        ]
        self.encoder = {piece: i for i, (piece, _) in enumerate(vocab)}
        self.decoder = {i: piece for i, (piece, _) in enumerate(vocab)}

        self.unk_token_id = self.sp_model.piece_to_id(unk_token)
        self.pad_token_id = self.sp_model.piece_to_id(pad_token)
        self.eos_token_id = self.sp_model.piece_to_id(eos_token)

        tok = Tokenizer(Unigram(vocab, unk_id=self.unk_token_id, byte_fallback=True))
        if do_lower_case:
            normalizer_steps = [
                Lowercase(),
                Replace(Regex(_PUNCT_REGEX), ""),
                Replace(Regex(r"\s+"), " "),
                Strip(left=True, right=True),
            ]
        else:
            normalizer_steps = [
                Replace(Regex(r"\s+"), " "),
                Strip(left=True, right=True),
            ]
        tok.normalizer = NormSeq(normalizer_steps)
        tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
        tok.post_processor = TemplateProcessing(
            single="$A </s>",
            special_tokens=[("</s>", self.eos_token_id)],
        )
        tok.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")
        tok.enable_truncation(max_length=context_length)
        tok.enable_padding(
            pad_id=self.pad_token_id, pad_token=pad_token, length=context_length
        )
        self._tok = tok

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def canonicalize_text(
        self, text: str, keep_punctuation_exact_string: Optional[str] = None
    ) -> str:
        import re

        if keep_punctuation_exact_string:
            text = keep_punctuation_exact_string.join(
                self.remove_punctuation(part)
                for part in text.split(keep_punctuation_exact_string)
            )
        else:
            text = self.remove_punctuation(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            enc = self._tok.encode(text, add_special_tokens=False)
            return enc.ids + [self.eos_token_id]
        encs = self._tok.encode_batch(text, add_special_tokens=False)
        return [e.ids + [self.eos_token_id] for e in encs]

    def detokenize(
        self, token_ids, skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        if hasattr(token_ids, "numpy"):
            token_ids = token_ids.numpy()
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        def _one(ids):
            if skip_special_tokens:
                skip = {self.pad_token_id, self.eos_token_id, self.unk_token_id}
                ids = [int(i) for i in ids if int(i) not in skip]
            return self.sp_model.decode_ids([int(i) for i in ids]).strip()

        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            return [_one(r) for r in token_ids]
        return _one(token_ids)

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        return list(token_ids) + [self.eos_token_id]

    def prepare_for_model_tensor(
        self, token_ids_list: List[List[int]]
    ) -> Dict[str, keras.KerasTensor]:
        padded = []
        for seq in token_ids_list:
            seq = list(seq)[: self.context_length]
            pad_len = self.context_length - len(seq)
            if pad_len > 0:
                seq = seq + [self.pad_token_id] * pad_len
            padded.append(seq)
        ids = np.array(padded, dtype=np.int32)
        return {"input_ids": ops.convert_to_tensor(ids, dtype="int32")}

    def prepare_for_model(self, text: Union[str, List[int]]) -> Dict[str, List[int]]:
        token_ids = self.tokenize(text) if isinstance(text, str) else list(text)
        token_ids = token_ids[: self.context_length]
        pad_len = self.context_length - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [self.pad_token_id] * pad_len
        return {"input_ids": token_ids}

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def call(self, inputs):
        if inputs is None:
            raise ValueError("No text inputs provided to SigLIPTokenizer")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        encs = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int32)
        return {"input_ids": ops.convert_to_tensor(ids, dtype="int32")}

    def batch_detokenize(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()
        out = []
        for row in token_ids_batch:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            if skip_special_tokens:
                row = [int(i) for i in row if int(i) != self.pad_token_id]
            out.append(self.detokenize(row))
        return out

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        return self.batch_detokenize(token_ids_batch, skip_special_tokens)

    def get_sequence_length(self, input_ids: keras.KerasTensor) -> keras.KerasTensor:
        pad = ops.convert_to_tensor(self.pad_token_id, dtype="int32")
        mask = ops.not_equal(input_ids, pad)
        return ops.sum(ops.cast(mask, dtype="int32"), axis=1)

    def truncate_sequences(
        self, input_ids: keras.KerasTensor, max_length: int
    ) -> keras.KerasTensor:
        if max_length >= input_ids.shape[1]:
            return input_ids
        return input_ids[:, :max_length]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "context_length": self.context_length,
                "do_lower_case": self.do_lower_case,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "eos_token": self.eos_token,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
