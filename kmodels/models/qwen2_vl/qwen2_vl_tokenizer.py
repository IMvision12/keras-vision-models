import json
import os
from typing import List, Union

import keras
import numpy as np
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.pre_tokenizers import Sequence as PreSeq
from tokenizers.processors import ByteLevel as ByteLevelProcessor

_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _download(repo_id, fname):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, fname)


@keras.saving.register_keras_serializable(package="kmodels")
class Qwen2VLTokenizer(keras.Layer):
    IM_START = 151644
    IM_END = 151645
    VISION_START = 151652
    VISION_END = 151653
    IMAGE_PAD = 151655
    VIDEO_PAD = 151656

    def __init__(
        self,
        vocab_file: str = None,
        merges_file: str = None,
        repo_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vocab_file is None or not os.path.exists(vocab_file):
            vocab_file = _download(repo_id, "vocab.json")
        if merges_file is None or not os.path.exists(merges_file):
            merges_file = _download(repo_id, "merges.txt")

        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.repo_id = repo_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        try:
            added_path = _download(repo_id, "added_tokens.json")
            with open(added_path, "r", encoding="utf-8") as f:
                self.added_tokens = json.load(f)
        except Exception:
            self.added_tokens = {
                "<|endoftext|>": 151643,
                "<|im_start|>": 151644,
                "<|im_end|>": 151645,
                "<|object_ref_start|>": 151646,
                "<|object_ref_end|>": 151647,
                "<|box_start|>": 151648,
                "<|box_end|>": 151649,
                "<|quad_start|>": 151650,
                "<|quad_end|>": 151651,
                "<|vision_start|>": 151652,
                "<|vision_end|>": 151653,
                "<|vision_pad|>": 151654,
                "<|image_pad|>": 151655,
                "<|video_pad|>": 151656,
            }

        tok = Tokenizer(
            BPE(
                vocab=vocab_file,
                merges=merges_file,
                unk_token=None,
                fuse_unk=False,
            )
        )
        tok.normalizer = NFC()
        tok.pre_tokenizer = PreSeq(
            [
                Split(pattern=Regex(_SPLIT_REGEX), behavior="isolated", invert=False),
                ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
            ]
        )
        tok.post_processor = ByteLevelProcessor(
            add_prefix_space=False, trim_offsets=False, use_regex=False
        )
        tok.decoder = ByteLevelDecoder(
            add_prefix_space=False, trim_offsets=False, use_regex=False
        )
        tok.add_special_tokens(
            [
                AddedToken(s, special=True, normalized=False)
                for s in self.added_tokens.keys()
            ]
        )
        self._tok = tok
        self._special_ids = set(self.added_tokens.values()) | {pad_token_id}

    def tokenize(
        self, text: Union[str, List[str]], add_special_tokens: bool = False
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=add_special_tokens).ids
        encs = self._tok.encode_batch(text, add_special_tokens=add_special_tokens)
        return [e.ids for e in encs]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids = [int(i) for i in token_ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_ids]
        return self._tok.decode(ids, skip_special_tokens=False)

    def batch_decode(self, token_ids_batch, skip_special_tokens: bool = True):
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()
        return [
            self.decode(row, skip_special_tokens=skip_special_tokens)
            for row in token_ids_batch
        ]

    def call(self, inputs: Union[str, List[str]]):
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

    def build_chat_prompt(
        self,
        text: str,
        image_patch_counts: List[int] = None,
        system_message: str = "You are a helpful assistant.",
    ) -> str:
        parts = [
            f"<|im_start|>system\n{system_message}<|im_end|>\n",
            "<|im_start|>user\n",
        ]
        if image_patch_counts:
            for n in image_patch_counts:
                parts.append("<|vision_start|>")
                parts.append("<|image_pad|>" * n)
                parts.append("<|vision_end|>")
        parts.append(text)
        parts.append("<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
                "repo_id": self.repo_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
        )
        return cfg
