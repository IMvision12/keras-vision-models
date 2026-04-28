from typing import Dict, List, Union

import keras
import numpy as np
import sentencepiece as spm
from keras import ops

from kmodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="kmodels")
class SigLIP2Tokenizer(BaseTokenizer):
    """SigLIP2 SentencePiece tokenizer matching HF ``google/siglip2-*``.

    HF's SigLIP2 (Gemma) tokenizer.json always appends ``<eos>`` via a
    ``TemplateProcessing`` step, regardless of the ``add_eos_token`` flag.
    This layer mirrors that behavior: EOS is always appended on ``call``.

    Args:
        vocab_file: Path to the SentencePiece ``.model`` file.
        context_length: Maximum sequence length (default 64).
        add_bos / add_eos: Exposed for compatibility with legacy code
            paths like ``prepare_for_model`` — ``call`` always follows
            HF's template and appends EOS.
        pad_token / bos_token / eos_token / unk_token: Special token strings.
    """

    def __init__(
        self,
        vocab_file: str,
        context_length: int = 64,
        add_bos: bool = False,
        add_eos: bool = True,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file
        self.context_length = context_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(vocab_file)

        vocab_size = self.sp_model.get_piece_size()
        self.encoder = {self.sp_model.id_to_piece(i): i for i in range(vocab_size)}
        self.decoder = {i: self.sp_model.id_to_piece(i) for i in range(vocab_size)}

        self.pad_token_id = self._resolve_id(pad_token, self.sp_model.pad_id(), 0)
        self.bos_token_id = self._resolve_id(bos_token, self.sp_model.bos_id(), 1)
        self.eos_token_id = self._resolve_id(eos_token, self.sp_model.eos_id(), 2)
        self.unk_token_id = self._resolve_id(unk_token, self.sp_model.unk_id(), 3)

    def _resolve_id(self, token: str, sp_id: int, fallback: int) -> int:
        tid = self.encoder.get(token, sp_id)
        return fallback if tid == -1 else tid

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return [] if not text else self.sp_model.encode_as_ids(text)
        return [[] if not s else self.sp_model.encode_as_ids(s) for s in text]

    def detokenize(self, token_ids: List[int]) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        skip = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        keep = [int(i) for i in token_ids if int(i) not in skip]
        return self.sp_model.decode_ids(keep) if keep else ""

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        result = list(token_ids)
        if self.add_bos and self.bos_token_id >= 0:
            if not result or result[0] != self.bos_token_id:
                result = [self.bos_token_id] + result
        if self.add_eos and self.eos_token_id >= 0:
            if not result or result[-1] != self.eos_token_id:
                result = result + [self.eos_token_id]
        return result

    def _encode_for_call(self, text: str) -> List[int]:
        ids = self.sp_model.encode_as_ids(text) if text else []
        ids = ids + [self.eos_token_id]
        return ids[: self.context_length]

    def prepare_for_model_tensor(
        self, token_ids_list: List[List[int]]
    ) -> Dict[str, keras.KerasTensor]:
        padded = []
        for seq in token_ids_list:
            seq = self.build_inputs_with_special_tokens(seq)[: self.context_length]
            pad_len = self.context_length - len(seq)
            if pad_len > 0:
                seq = seq + [self.pad_token_id] * pad_len
            padded.append(seq)
        ids = np.array(padded, dtype=np.int32)
        return {"input_ids": ops.convert_to_tensor(ids, dtype="int32")}

    def prepare_for_model(self, text: Union[str, List[int]]) -> Dict[str, List[int]]:
        token_ids = self.tokenize(text) if isinstance(text, str) else list(text)
        token_ids = self.build_inputs_with_special_tokens(token_ids)
        token_ids = token_ids[: self.context_length]
        pad_len = self.context_length - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [self.pad_token_id] * pad_len
        return {"input_ids": token_ids}

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocabulary(self) -> List[str]:
        return [self.sp_model.id_to_piece(i) for i in range(self.vocab_size)]

    def id_to_token(self, id: int) -> str:
        if id >= self.vocab_size or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocab_size - 1}]. Received: {id}"
            )
        return self.sp_model.id_to_piece(id)

    def token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def batch_decode(
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

    def call(self, inputs):
        if inputs is None:
            raise ValueError("No text inputs provided to SigLIP2Tokenizer")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        rows = []
        for t in texts:
            seq = self._encode_for_call(t)
            pad_len = self.context_length - len(seq)
            if pad_len > 0:
                seq = seq + [self.pad_token_id] * pad_len
            rows.append(seq)
        ids = np.array(rows, dtype=np.int32)
        return {"input_ids": ops.convert_to_tensor(ids, dtype="int32")}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "context_length": self.context_length,
                "add_bos": self.add_bos,
                "add_eos": self.add_eos,
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
