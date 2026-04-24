import json
from typing import Dict, List, Union

import keras
import numpy as np
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Lowercase, Replace
from tokenizers.normalizers import Sequence as NormSeq
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.pre_tokenizers import Sequence as PreSeq
from tokenizers.processors import RobertaProcessing


@keras.saving.register_keras_serializable(package="kmodels")
class CLIPTokenizer(keras.Layer):
    """CLIP BPE tokenizer, built on HuggingFace `tokenizers` (Rust).

    Wraps a programmatically-assembled `tokenizers.Tokenizer` that matches
    OpenAI/HF CLIP exactly: NFC + whitespace collapse + lowercase, CLIP's
    regex split, byte-level BPE over ``vocab.json`` + ``merges.txt``, and
    ``[BOS] ... [EOS]`` RoBERTa-style post-processing with EOS padding.

    Args:
        vocab_file: Path to ``vocab.json``.
        merges_file: Path to ``merges.txt``.
        context_length: Max sequence length (default 77).
        unk_token / bos_token / eos_token / pad_token: Special token strings.
    """

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        context_length: int = 77,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.context_length = context_length
        self.errors = errors
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.bos_token_id = self.encoder.get(bos_token, 49406)
        self.eos_token_id = self.encoder.get(eos_token, 49407)
        self.pad_token_id = self.encoder.get(pad_token, 49407)
        self.unk_token_id = self.encoder.get(unk_token, 49407)

        tok = Tokenizer(
            BPE(
                vocab=vocab_file,
                merges=merges_file,
                unk_token=unk_token,
                end_of_word_suffix="</w>",
                fuse_unk=False,
            )
        )
        tok.normalizer = NormSeq([NFC(), Replace(Regex(r"\s+"), " "), Lowercase()])
        tok.pre_tokenizer = PreSeq(
            [
                Split(
                    pattern=Regex(
                        r"<\|startoftext\|>|<\|endoftext\|>"
                        r"|'s|'t|'re|'ve|'m|'ll|'d"
                        r"|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
                    ),
                    behavior="removed",
                    invert=True,
                ),
                ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
            ]
        )
        tok.post_processor = RobertaProcessing(
            sep=(eos_token, self.eos_token_id),
            cls=(bos_token, self.bos_token_id),
            trim_offsets=False,
            add_prefix_space=False,
        )
        tok.decoder = ByteLevelDecoder()
        tok.add_special_tokens(
            [
                AddedToken(bos_token, special=True, normalized=False),
                AddedToken(eos_token, special=True, normalized=False),
            ]
        )
        tok.enable_truncation(max_length=context_length)
        tok.enable_padding(
            pad_id=self.pad_token_id,
            pad_token=pad_token,
            length=context_length,
        )
        self._tok = tok

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            enc = self._tok.encode(text, add_special_tokens=False)
            return enc.ids
        encs = self._tok.encode_batch(text, add_special_tokens=False)
        return [e.ids for e in encs]

    def detokenize(self, token_ids) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        skip = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        keep = [int(i) for i in token_ids if int(i) not in skip]
        text = self._tok.decode(keep, skip_special_tokens=False)
        return text.replace("</w>", " ").strip()

    def prepare_for_model(self, text: str) -> Dict[str, List[int]]:
        enc = self._tok.encode(text)
        return {"input_ids": enc.ids, "attention_mask": enc.attention_mask}

    def call(self, inputs: Union[str, List[str]]):
        if inputs is None:
            raise ValueError("No text inputs provided to CLIPTokenizer")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        encs = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int32)
        mask = np.array([e.attention_mask for e in encs], dtype=np.bool_)
        return {
            "input_ids": keras.ops.convert_to_tensor(ids, dtype="int32"),
            "attention_mask": keras.ops.convert_to_tensor(mask, dtype="bool"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
                "context_length": self.context_length,
                "errors": self.errors,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
            }
        )
        return config
