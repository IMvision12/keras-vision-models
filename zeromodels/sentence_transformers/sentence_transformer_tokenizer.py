import json
import os
import re
from pathlib import PurePosixPath
from typing import List, Union

import keras
from tokenizers import Tokenizer, normalizers

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
@keras.saving.register_keras_serializable(
    package="zeromodels", name="SentenceTransformerTokenizer"
)
class ZMSentenceTransformerTokenizer(BaseTokenizer):
    """Tokenizer for supported ST models, preserving their tokenizer.json rules.

    Loads ``tokenizer.json`` from a Hub repo (via ``from_weights("org/repo")`` /
    ``from_hub_repo``) or an explicit ``tokenizer_file``. ``zm_preprocessor.json``
    keeps official tokenizer settings separate from Sentence Transformers'
    additional preprocessing. The JSON pipeline owns normalization and special
    token insertion; metadata supplies special token identities and batching.
    Repos without preprocessing metadata retain the legacy loading path.

    Args:
        variant: Optional Hub variant key; resolves
            ``zeromodels/<variant>/tokenizer.json`` when ``tokenizer_file`` is
            omitted.
        tokenizer_file: Explicit ``tokenizer.json`` path (overrides variant).
        max_seq_len: Explicit truncation override; otherwise read from metadata.
        bos_token / eos_token / unk_token / pad_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = None,
        backbone_type: str = "mpnet",
        bos_token: str = None,
        eos_token: str = None,
        unk_token: str = None,
        pad_token: str = None,
        mask_token: str = None,
        tokenizer_config: dict = None,
        sentence_transformer_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            (f"zeromodels/{self.variant}" if self.variant else None), tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.backbone_type = backbone_type
        self.tokenizer_config = dict(tokenizer_config or {})
        self.sentence_transformer_config = dict(sentence_transformer_config or {})
        tc, sc = self.tokenizer_config, self.sentence_transformer_config
        if max_seq_len is None:
            max_seq_len = sc.get("max_seq_length")
        if max_seq_len is None:
            candidate = tc.get("model_max_length")
            max_seq_len = (
                candidate
                if isinstance(candidate, int) and 0 < candidate < 10**9
                else 384
            )
        if (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, int)
            or max_seq_len < 1
        ):
            raise ValueError("max_seq_len must be a positive integer")
        self.max_seq_len = max_seq_len

        tok = Tokenizer.from_file(tokenizer_file)
        if sc.get("do_lower_case", False) and not self._has_lowercase(
            json.loads(tok.to_str()).get("normalizer")
        ):
            tok.normalizer = normalizers.Sequence(
                [normalizers.Lowercase()] + ([tok.normalizer] if tok.normalizer else [])
            )
        for name, explicit, fallback in (
            ("bos_token", bos_token, "<s>"),
            ("eos_token", eos_token, "</s>"),
            ("unk_token", unk_token, "[UNK]"),
            ("pad_token", pad_token, None),
            ("mask_token", mask_token, "<mask>"),
            ("cls_token", None, None),
            ("sep_token", None, None),
        ):
            value = explicit if explicit is not None else tc.get(name, fallback)
            if isinstance(value, dict):
                value = value["content"]
            setattr(self, name, value)
            setattr(
                self,
                f"{name}_id",
                tok.token_to_id(value) if value is not None else None,
            )
        pad_token = self.pad_token
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
        self.pad_token_id = tok.token_to_id(pad_token)
        if self.pad_token_id is None:
            raise ValueError(
                f"Padding token {pad_token!r} is absent from tokenizer.json"
            )
        tok.enable_truncation(
            max_length=max_seq_len,
            direction=tc.get("truncation_side", "right"),
            strategy="longest_first",
        )
        tok.enable_padding(
            direction=tc.get("padding_side", "right"),
            pad_id=self.pad_token_id,
            pad_token=pad_token,
        )
        self._tok = tok

    @classmethod
    def from_variant(cls, variant, /, **kwargs):
        if kwargs.get("tokenizer_file") is not None:
            return cls(variant=variant, **kwargs)
        return cls.from_hub_repo(f"zeromodels/{variant}", **kwargs)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        token = kwargs.pop("token", os.environ.get("HF_TOKEN"))
        revision = cls._snapshot(repo, kwargs.pop("revision", None), token)
        modules = cls._optional_json(repo, "modules.json", token, revision)
        prefix = ""
        if modules and modules[0].get("path"):
            prefix = cls._repo_file(modules[0]["path"]) + "/"
        hf_config = cls._optional_json(repo, prefix + "config.json", token, revision)
        if hf_config is None:
            raise ValueError(f"{repo}: missing Transformer config.json")
        kwargs.setdefault("backbone_type", hf_config["model_type"])
        tc = (
            cls._optional_json(repo, prefix + "tokenizer_config.json", token, revision)
            or {}
        )
        tc.update(
            cls._optional_json(
                repo, prefix + "special_tokens_map.json", token, revision
            )
            or {}
        )
        kwargs.setdefault("tokenizer_config", tc)
        kwargs.setdefault(
            "sentence_transformer_config",
            cls._optional_json(
                repo, prefix + "sentence_bert_config.json", token, revision
            )
            or {},
        )
        path = hf_hub_download(
            repo, prefix + "tokenizer.json", token=token, revision=revision
        )
        model = cls(tokenizer_file=path, **kwargs)
        model.source_revision = revision
        return model

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        """Load official preprocessing metadata, with legacy config fallback."""
        from huggingface_hub import hf_hub_download

        repo_id = repo_id.rstrip("/")
        token = kwargs.pop("token", os.environ.get("HF_TOKEN"))
        revision = cls._snapshot(repo_id, kwargs.pop("revision", None), token)
        spec = cls._optional_json(repo_id, "zm_preprocessor.json", token, revision)
        filename = "tokenizer.json"
        if spec is not None:
            if spec.get("preprocessor_class") not in (
                cls.__name__,
                "SentenceTransformerTokenizer",
            ):
                raise ValueError(
                    f"{repo_id}: zm_preprocessor.json must declare {cls.__name__}"
                )
            filename = cls._repo_file(spec.get("tokenizer_file", filename))
            for field in (
                "tokenizer_config",
                "sentence_transformer_config",
                "backbone_type",
                "max_seq_len",
            ):
                if field in spec:
                    kwargs.setdefault(field, spec[field])
        else:
            spec = cls._optional_json(repo_id, "zm_config.json", token, revision) or {}
            config = {**spec.get("text_config", {}), **spec}
            for field in ("backbone_type", "max_seq_len"):
                if field in config:
                    kwargs.setdefault(field, config[field])
            kwargs.setdefault(
                "tokenizer_config",
                cls._optional_json(repo_id, "tokenizer_config.json", token, revision)
                or {},
            )
            kwargs.setdefault(
                "sentence_transformer_config",
                cls._optional_json(
                    repo_id, "sentence_bert_config.json", token, revision
                )
                or {},
            )
        path = hf_hub_download(repo_id, filename, token=token, revision=revision)
        model = cls(tokenizer_file=path, **kwargs)
        model.source_revision = revision
        return model

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
                "tokenizer_config": self.tokenizer_config,
                "sentence_transformer_config": self.sentence_transformer_config,
            }
        )
        return config

    def get_hub_config(self):
        """Portable preprocessing recipe; tokenizer.json is stored alongside it."""
        tc = dict(self.tokenizer_config)
        for name in (
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "mask_token",
            "cls_token",
            "sep_token",
        ):
            if getattr(self, f"{name}_id") is not None:
                tc[name] = getattr(self, name)
        sc = dict(self.sentence_transformer_config)
        sc["max_seq_length"] = self.max_seq_len
        return {
            "tokenizer_file": "tokenizer.json",
            "backbone_type": self.backbone_type,
            "tokenizer_config": tc,
            "sentence_transformer_config": sc,
        }

    @staticmethod
    def _repo_file(filename):
        """Accept only filenames inside the selected Hub repository."""
        if (
            not isinstance(filename, str)
            or not filename
            or "\\" in filename
            or ":" in filename
            or filename.startswith("/")
            or any(part in ("", ".", "..") for part in filename.split("/"))
        ):
            raise ValueError(
                f"Expected a repository-relative filename, got {filename!r}"
            )
        return str(PurePosixPath(filename))

    @staticmethod
    def _optional_json(repo, filename, token, revision):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            path = hf_hub_download(repo, filename, token=token, revision=revision)
        except EntryNotFoundError:
            return None
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _snapshot(repo, revision, token):
        from huggingface_hub import HfApi

        if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
            return revision
        sha = HfApi(token=token).model_info(repo, revision=revision).sha
        if not sha:
            raise OSError(f"{repo}: Hub returned no commit SHA for {revision!r}")
        return sha

    @staticmethod
    def _has_lowercase(config):
        return config is not None and (
            config.get("type") == "Lowercase"
            or config.get("lowercase") is True
            or any(
                ZMSentenceTransformerTokenizer._has_lowercase(item)
                for item in config.get("normalizers", [])
            )
        )


SentenceTransformerTokenizer = ZMSentenceTransformerTokenizer
