"""Sentence Transformer text embedding models (Keras 3)."""

from __future__ import annotations

import json
import os
import re

import keras
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.base.base_mixin import warn_skipped
from zeromodels.conversion.hf_download_utils import download_hf_state_dict
from zeromodels.conversion.weight_transfer_util import skip_mismatched_weights

from .sentence_transformer_backbones import albert_backbone, distilbert_backbone
from .sentence_transformer_config import SentenceTransformerConfig
from .sentence_transformer_layers import (
    SentenceTransformerNormalize,
    SentenceTransformerPooling,
)

SENTENCE_TRANSFORMER_HUB_SIBLINGS = frozenset(
    {"ZMSentenceTransformer", "SentenceTransformerModel", "SentenceTransformer"}
)


@keras.saving.register_keras_serializable(package="zeromodels")
@keras.saving.register_keras_serializable(
    package="zeromodels", name="SentenceTransformerModel"
)
class ZMSentenceTransformer(BaseModel):
    """Sentence embedding model: encoder + pooling (+ optional Dense) + normalize.

    Matches popular Sentence-Transformers stacks such as
    ``sentence-transformers/all-mpnet-base-v2`` (MPNet encoder, mean pooling,
    L2 normalize). Load an upstream checkpoint with::

        model = ZMSentenceTransformer.from_weights(
            "hf:sentence-transformers/all-mpnet-base-v2"
        )
        tok = ZMSentenceTransformerTokenizer.from_weights(
            "hf:sentence-transformers/all-mpnet-base-v2",
            revision=model.source_revision,
        )
        emb = model(tok(["a cat", "a dog"]))["embeddings"]

    Or the ST-style helper::

        emb = model.encode(["a cat", "a dog"], tokenizer=tok)

    Args:
        backbone_type: Encoder family (MPNet, BERT, RoBERTa, XLM-R, DistilBERT, ALBERT).
        pooling: ``\"mean\"`` | ``\"cls\"`` | ``\"max\"``.
        normalize_embeddings: If True, L2-normalize the output vectors.
        projection_dim: Optional Dense width after pooling (None = identity).
        max_seq_len: Default tokenizer truncation length (stored in config).
        Other kwargs: forwarded to the backbone (``vocab_size``, ``embed_dim``, …).
    """

    config_class = SentenceTransformerConfig
    # Accept zeromodels ``sentence_transformer`` configs and upstream ST hubs
    # whose root ``config.json`` is the backbone transformers config
    # (``model_type: mpnet`` for all-mpnet-base-v2). Pooling / normalize come
    # from sibling ST files read in :meth:`from_hf`.
    HF_MODEL_TYPE = (
        "sentence_transformer",
        "mpnet",
        "bert",
        "roberta",
        "xlm-roberta",
        "distilbert",
        "albert",
    )
    HUB_REPO_SIBLINGS = SENTENCE_TRANSFORMER_HUB_SIBLINGS

    def __init__(
        self,
        backbone_type="mpnet",
        vocab_size=30527,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=514,
        relative_attention_num_buckets=32,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        pad_token_id=1,
        type_vocab_size=2,
        embedding_size=None,
        num_hidden_groups=1,
        inner_group_num=1,
        dropout=0.0,
        attention_dropout=0.0,
        pooling="mean",
        normalize_embeddings=True,
        projection_dim=None,
        projection_activation="linear",
        similarity_fn_name="cosine",
        max_seq_len=384,
        name="ZMSentenceTransformer",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        if backbone_type not in (
            "mpnet",
            "bert",
            "roberta",
            "xlm-roberta",
            "distilbert",
            "albert",
        ):
            raise ValueError(
                f"Unsupported sentence embedding backbone: {backbone_type!r}"
            )
        if pooling not in ("mean", "cls", "max"):
            raise ValueError(f"pooling must be mean|cls|max, got {pooling!r}")
        if projection_activation not in ("linear", "tanh"):
            raise ValueError(
                f"Unsupported projection activation: {projection_activation!r}"
            )
        if similarity_fn_name not in ("cosine", "dot"):
            raise ValueError(f"Unsupported similarity function: {similarity_fn_name!r}")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        if backbone_type in ("bert", "roberta", "xlm-roberta", "albert"):
            inputs["token_type_ids"] = layers.Input(
                shape=(None,), dtype="int32", name="token_type_ids"
            )
        common = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "mlp_dim": mlp_dim,
            "max_position_embeddings": max_position_embeddings,
            "hidden_act": hidden_act,
            "layer_norm_eps": layer_norm_eps,
        }
        if backbone_type == "mpnet":
            from zeromodels.models.mpnet.mpnet_model import mpnet_backbone

            sequence_output, _ = mpnet_backbone(
                inputs["input_ids"],
                inputs["attention_mask"],
                **common,
                relative_attention_num_buckets=relative_attention_num_buckets,
                pad_token_id=pad_token_id,
                dropout=dropout,
                attention_dropout=attention_dropout,
                add_pooler=False,
            )
        elif backbone_type in ("roberta", "xlm-roberta"):
            from zeromodels.models.roberta.roberta_model import roberta_backbone

            sequence_output, _ = roberta_backbone(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"],
                **common,
                type_vocab_size=type_vocab_size,
                pad_token_id=pad_token_id,
                dropout=dropout,
                attention_dropout=attention_dropout,
                add_pooler=False,
            )
        elif backbone_type == "bert":
            from zeromodels.models.bert.bert_model import bert_backbone

            sequence_output, _ = bert_backbone(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"],
                **common,
                type_vocab_size=type_vocab_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                add_pooler=False,
            )
        elif backbone_type == "distilbert":
            sequence_output = distilbert_backbone(
                inputs["input_ids"],
                inputs["attention_mask"],
                **common,
            )
        else:
            sequence_output = albert_backbone(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"],
                **common,
                embedding_size=embedding_size or embed_dim,
                type_vocab_size=type_vocab_size,
                num_hidden_groups=num_hidden_groups,
                inner_group_num=inner_group_num,
            )

        pooled = SentenceTransformerPooling(pooling=pooling, name="pooling")(
            [sequence_output, inputs["attention_mask"]]
        )

        if projection_dim is not None:
            pooled = layers.Dense(
                int(projection_dim), activation=projection_activation, name="projection"
            )(pooled)
            out_dim = int(projection_dim)
        else:
            out_dim = embed_dim

        if normalize_embeddings:
            embeddings = SentenceTransformerNormalize(name="normalize")(pooled)
        else:
            embeddings = pooled

        super().__init__(
            inputs=inputs,
            outputs={
                "embeddings": embeddings,
                "token_embeddings": sequence_output,
            },
            name=name,
            **kwargs,
        )
        self.backbone_type = backbone_type
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.embedding_size = embedding_size
        self.num_hidden_groups = num_hidden_groups
        self.inner_group_num = inner_group_num
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.projection_dim = projection_dim
        self.projection_activation = projection_activation
        self.similarity_fn_name = similarity_fn_name
        self.max_seq_len = max_seq_len
        self.output_embedding_dim = out_dim
        self.config = SentenceTransformerConfig(
            backbone_type=backbone_type,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            relative_attention_num_buckets=relative_attention_num_buckets,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            type_vocab_size=type_vocab_size,
            embedding_size=embedding_size,
            num_hidden_groups=num_hidden_groups,
            inner_group_num=inner_group_num,
            dropout=dropout,
            attention_dropout=attention_dropout,
            pooling=pooling,
            normalize_embeddings=normalize_embeddings,
            projection_dim=projection_dim,
            projection_activation=projection_activation,
            similarity_fn_name=similarity_fn_name,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def config_from_hf(cls, hf_config):
        """Map a transformers backbone ``config.json`` to constructor kwargs.

        Pooling / normalize / max_seq_len are filled by :meth:`from_hf` from the
        Sentence-Transformers sidecar files when present.
        """
        kind = hf_config["model_type"]
        if kind == "distilbert":
            return {
                "backbone_type": kind,
                "vocab_size": hf_config["vocab_size"],
                "embed_dim": hf_config["dim"],
                "num_layers": hf_config["n_layers"],
                "num_heads": hf_config["n_heads"],
                "mlp_dim": hf_config["hidden_dim"],
                "max_position_embeddings": hf_config["max_position_embeddings"],
                "hidden_act": hf_config.get("activation", "gelu"),
                "layer_norm_eps": 1e-12,
                "pad_token_id": hf_config.get("pad_token_id", 0),
            }
        result = {
            "backbone_type": kind,
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_dim": hf_config["intermediate_size"],
            "max_position_embeddings": hf_config["max_position_embeddings"],
            "hidden_act": hf_config.get("hidden_act", "gelu"),
            "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-5),
            "pad_token_id": hf_config.get("pad_token_id", 1),
            "type_vocab_size": hf_config.get("type_vocab_size", 2),
        }
        if kind == "mpnet":
            result["relative_attention_num_buckets"] = hf_config.get(
                "relative_attention_num_buckets", 32
            )
        if kind == "albert":
            result.update(
                embedding_size=hf_config["embedding_size"],
                num_hidden_groups=hf_config.get("num_hidden_groups", 1),
                inner_group_num=hf_config.get("inner_group_num", 1),
            )
        return result

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_sentence_transformer_hf_to_keras import (
            transfer_sentence_transformer_weights,
        )

        transfer_sentence_transformer_weights(keras_model, state_dict)

    @classmethod
    def from_hf(
        cls,
        hf_id,
        load_weights=True,
        variant=None,
        skip_mismatch=False,
        quantization=None,
        **kwargs,
    ):
        """Load from a Sentence-Transformers (or plain MPNet) Hub repo.

        Reads the backbone ``config.json`` via the base loader, then overlays
        pooling / normalize / ``max_seq_len`` from ``modules.json``,
        ``*_Pooling/config.json``, and ``sentence_bert_config.json`` when present.
        """
        from huggingface_hub import HfApi, hf_hub_download

        token = os.environ.get("HF_TOKEN")
        revision = kwargs.pop("revision", None)
        if variant is not None:
            raise ValueError("variant= is not used for Sentence Transformers Hub repos")
        # Resolve once: config, module metadata, tokenizer and weight files must
        # describe the same commit even if the upstream default branch moves.
        if not (isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)):
            revision = HfApi(token=token).model_info(hf_id, revision=revision).sha
        with open(
            hf_hub_download(hf_id, "config.json", token=token, revision=revision)
        ) as f:
            hf_config = json.load(f)
        cls.assert_hf_model_type(hf_id, hf_config)
        if hf_config["model_type"] == "sentence_transformer":
            raise ValueError(
                "Use the ZeroModels repo loader for a converted checkpoint"
            )
        sidecar, dense_path = cls._st_sidecar_kwargs(
            hf_id, token=token, revision=revision
        )
        model_kwargs = cls.config_from_hf(hf_config)
        model_kwargs.update(sidecar)
        model_kwargs.update(kwargs)
        model = cls(**model_kwargs)
        model.source_revision = revision
        if not load_weights:
            return model
        state_dict = download_hf_state_dict(hf_id, token=token, revision=revision)
        completed = False
        try:
            if dense_path:
                from .convert_sentence_transformer_hf_to_keras import (
                    load_dense_state_dict,
                )

                model._st_dense_state = load_dense_state_dict(
                    hf_id, dense_path, token=token, revision=revision
                )
            with skip_mismatched_weights(skip_mismatch) as skipped:
                cls._quantized_transfer(model, state_dict, quantization, skip_mismatch)
            completed = True
        finally:
            close = getattr(state_dict, "close", None)
            if callable(close):
                close(completed=completed)
            if hasattr(model, "_st_dense_state"):
                del model._st_dense_state
        warn_skipped(skipped)
        return model

    @classmethod
    def _st_sidecar_kwargs(cls, hf_id, token=None, revision=None):
        """Parse Sentence-Transformers module configs into constructor kwargs."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        out = {}
        try:
            path = hf_hub_download(
                hf_id,
                "config_sentence_transformers.json",
                token=token,
                revision=revision,
            )
            with open(path, "r") as f:
                st_config = json.load(f)
            if st_config.get("similarity_fn_name") is not None:
                out["similarity_fn_name"] = st_config["similarity_fn_name"]
        except EntryNotFoundError:
            pass
        try:
            path = hf_hub_download(
                hf_id, "sentence_bert_config.json", token=token, revision=revision
            )
            with open(path, "r") as f:
                sbert = json.load(f)
            if "max_seq_length" in sbert:
                out["max_seq_len"] = int(sbert["max_seq_length"])
        except EntryNotFoundError:
            pass

        modules = None
        try:
            path = hf_hub_download(
                hf_id, "modules.json", token=token, revision=revision
            )
            with open(path, "r") as f:
                modules = json.load(f)
        except EntryNotFoundError:
            modules = None

        if modules is None:
            raise ValueError(
                f"{hf_id} has no modules.json; it is not a Sentence Transformers checkpoint"
            )
        types = [mod.get("type") for mod in modules]
        allowed = [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
        ]
        if types[:2] != allowed or types[2:] not in (
            [],
            ["sentence_transformers.models.Normalize"],
            ["sentence_transformers.models.Dense"],
            [
                "sentence_transformers.models.Dense",
                "sentence_transformers.models.Normalize",
            ],
        ):
            raise ValueError(
                f"Unsupported Sentence Transformers module recipe in {hf_id}: {types}"
            )
        out["pooling"] = cls._pooling_from_module_dir(
            hf_id, modules[1].get("path"), token=token, revision=revision
        )
        out["normalize_embeddings"] = (
            types[-1] == "sentence_transformers.models.Normalize"
        )
        dense_path = None
        if "sentence_transformers.models.Dense" in types:
            dense_path = modules[2].get("path")
            dense = cls._dense_config(hf_id, dense_path, token=token, revision=revision)
            out["projection_dim"] = int(dense["out_features"])
            activation = dense.get("activation_function")
            if activation in (None, "torch.nn.modules.linear.Identity"):
                out["projection_activation"] = "linear"
            elif activation == "torch.nn.modules.activation.Tanh":
                out["projection_activation"] = "tanh"
            else:
                raise ValueError(
                    f"Unsupported Sentence Transformers Dense activation: {activation}"
                )
        return out, dense_path

    @classmethod
    def _pooling_from_module_dir(cls, hf_id, rel_path, token=None, revision=None):
        from huggingface_hub import hf_hub_download

        filename = "config.json" if not rel_path else f"{rel_path}/config.json"
        path = hf_hub_download(hf_id, filename, token=token, revision=revision)
        with open(path, "r") as f:
            cfg = json.load(f)
        flags = {
            "cls": cfg.get("pooling_mode_cls_token", False),
            "mean": cfg.get("pooling_mode_mean_tokens", False),
            "max": cfg.get("pooling_mode_max_tokens", False),
        }
        unsupported = (
            cfg.get("pooling_mode_mean_sqrt_len_tokens")
            or cfg.get("pooling_mode_weightedmean_tokens")
            or cfg.get("pooling_mode_lasttoken")
            or cfg.get("include_prompt") is False
        )
        selected = [name for name, enabled in flags.items() if enabled]
        if unsupported or len(selected) != 1:
            raise ValueError(f"Unsupported Sentence Transformers pooling config: {cfg}")
        return selected[0]

    @classmethod
    def _dense_config(cls, hf_id, rel_path, token=None, revision=None):
        from huggingface_hub import hf_hub_download

        filename = "config.json" if not rel_path else f"{rel_path}/config.json"
        path = hf_hub_download(hf_id, filename, token=token, revision=revision)
        with open(path, "r") as f:
            return json.load(f)

    def encode(self, texts, tokenizer, normalize=None, batch_size=32):
        """Embed ``texts`` and return a numpy array ``(N, D)``.

        Args:
            texts: String or list of strings.
            tokenizer: :class:`ZMSentenceTransformerTokenizer` (or any callable that
                returns ``input_ids`` / ``attention_mask``).
            normalize: Override config L2-normalize. ``None`` keeps model output.
            batch_size: Encode in chunks of this size.
        """
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch)
            out = self(inputs, training=False)
            emb = ops.convert_to_numpy(out["embeddings"])
            if normalize is False and self.normalize_embeddings:
                raise ValueError(
                    "This model has a Normalize module; encode(normalize=False) cannot undo it"
                )
            if normalize is True:
                emb = emb / np.clip(
                    np.linalg.norm(emb, axis=-1, keepdims=True), 1e-12, None
                )
            chunks.append(emb)
        if not chunks:
            return np.zeros((0, self.output_embedding_dim), dtype="float32")
        return np.concatenate(chunks, axis=0)

    def similarity(self, a, b):
        """Pairwise scores using this checkpoint's declared similarity function."""
        if self.similarity_fn_name == "dot":
            return ops.matmul(a, ops.transpose(b))
        a = ops.convert_to_tensor(a)
        b = ops.convert_to_tensor(b)
        a = a / ops.sqrt(
            ops.maximum(ops.sum(ops.square(a), axis=-1, keepdims=True), 1e-12)
        )
        b = b / ops.sqrt(
            ops.maximum(ops.sum(ops.square(b), axis=-1, keepdims=True), 1e-12)
        )
        return ops.matmul(a, ops.transpose(b))

    def get_config(self):
        config = super().get_config()
        config.update(self.config.constructor_kwargs())
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SentenceTransformerTextEmbed(ZMSentenceTransformer):
    """Text-embedding task view of :class:`ZMSentenceTransformer`."""


# Preserve old deep imports and serialized objects without duplicating classes.
SentenceTransformerModel = ZMSentenceTransformer
SentenceTransformer = ZMSentenceTransformer
