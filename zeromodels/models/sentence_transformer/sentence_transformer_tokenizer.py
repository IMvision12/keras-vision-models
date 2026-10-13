"""Fast-tokenizer adapter for supported Sentence Transformers encoders."""

import re
from typing import List, Union

import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class SentenceTransformerTokenizer(BaseTokenizer):
    """Tokenizer for supported ST models, preserving their tokenizer.json rules.

    Loads ``tokenizer.json`` from a Hub repo (via ``from_weights("org/repo")`` /
    ``from_hub_repo``) or an explicit ``tokenizer_file``. Defaults match
    ``sentence-transformers/all-mpnet-base-v2`` (``max_seq_len=384``, MPNet
    special tokens). ``call`` returns ``input_ids`` / ``attention_mask`` only.

    Args:
        variant: Optional Hub variant key; resolves
            ``zeromodels/<variant>/tokenizer.json`` when ``tokenizer_file`` is
            omitted.
        tokenizer_file: Explicit ``tokenizer.json`` path (overrides variant).
        max_seq_len: Truncation length (default 384).
        bos_token / eos_token / unk_token / pad_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 384,
        backbone_type: str = "mpnet",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "[UNK]",
        pad_token: str = None,
        mask_token: str = "<mask>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            (f"zeromodels/{self.variant}" if self.variant else None), tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.max_seq_len = max_seq_len
        self.backbone_type = backbone_type
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        tok = Tokenizer.from_file(tokenizer_file)
        if pad_token is None:
            pad_token = next(
                (
                    candidate
                    for candidate in ("<pad>", "[PAD]", "<PAD>")
                    if tok.token_to_id(candidate) is not None
                ),
                None,
            )
        if pad_token is None:
            raise ValueError("tokenizer.json has no supported padding token")
        self.pad_token = pad_token
        self.bos_token_id = tok.token_to_id(bos_token)
        self.eos_token_id = tok.token_to_id(eos_token)
        self.unk_token_id = tok.token_to_id(unk_token)
        self.pad_token_id = tok.token_to_id(pad_token)
        self.mask_token_id = tok.token_to_id(mask_token)
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(pad_id=self.pad_token_id, pad_token=pad_token)
        self._tok = tok

    @classmethod
    def from_hf(cls, repo, **kwargs):
        import json
        import os

        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        token = os.environ.get("HF_TOKEN")
        revision = kwargs.pop("revision", None)
        if not (isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)):
            revision = HfApi(token=token).model_info(repo, revision=revision).sha
        with open(
            hf_hub_download(repo, "config.json", token=token, revision=revision)
        ) as f:
            hf_config = json.load(f)
        kwargs.setdefault("backbone_type", hf_config["model_type"])
        if "max_seq_len" not in kwargs:
            try:
                path = hf_hub_download(
                    repo, "sentence_bert_config.json", token=token, revision=revision
                )
            except EntryNotFoundError:
                pass
            else:
                with open(path) as f:
                    kwargs["max_seq_len"] = int(json.load(f)["max_seq_length"])
        path = hf_hub_download(repo, "tokenizer.json", token=token, revision=revision)
        return cls(tokenizer_file=path, **kwargs)

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        """Load ``tokenizer.json`` and honour ``max_seq_len`` from zm / ST configs."""
        import json
        import os

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        token = os.environ.get("HF_TOKEN")
        path = hf_hub_download(repo_id.rstrip("/"), "tokenizer.json", token=token)
        if "max_seq_len" not in kwargs:
            for filename in ("zm_config.json", "sentence_bert_config.json"):
                try:
                    cfg_path = hf_hub_download(
                        repo_id.rstrip("/"), filename, token=token
                    )
                except EntryNotFoundError:
                    continue
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                if "max_seq_len" in cfg:
                    kwargs["max_seq_len"] = int(cfg["max_seq_len"])
                elif "max_seq_length" in cfg:
                    kwargs["max_seq_len"] = int(cfg["max_seq_length"])
                if "backbone_type" in cfg:
                    kwargs.setdefault("backbone_type", cfg["backbone_type"])
                if "max_seq_len" in kwargs:
                    break
        return cls(tokenizer_file=path, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=False).ids
        encs = self._tok.encode_batch(text, add_special_tokens=False)
        return [e.ids for e in encs]

    def detokenize(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def call(self, inputs: Union[str, List[str]], text_pair=None):
        return self.encode_batch_to_inputs(
            inputs,
            text_pair,
            token_type_ids=self.backbone_type
            in ("bert", "roberta", "xlm-roberta", "albert"),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "max_seq_len": self.max_seq_len,
                "backbone_type": self.backbone_type,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "mask_token": self.mask_token,
            }
        )
        return config
