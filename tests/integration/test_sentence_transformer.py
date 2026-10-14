"""Offline contract tests for the multi-backbone sentence-embedding API."""

import importlib
import json

import numpy as np
import pytest
from keras import ops

from tests.base.model_test_registry import MODEL_TEST_CONFIGS
from zeromodels.st_models import (
    SentenceTransformer,
    SentenceTransformerModel,
    SentenceTransformerTextEmbed,
)


def test_top_level_st_api_and_legacy_imports_resolve_to_same_classes():
    legacy = importlib.import_module("zeromodels.models.sentence_transformer")
    legacy_model_module = importlib.import_module(
        "zeromodels.models.sentence_transformer.sentence_transformer_model"
    )
    assert SentenceTransformer is SentenceTransformerModel
    assert legacy.SentenceTransformerModel is SentenceTransformerModel
    assert legacy_model_module.SentenceTransformerModel is SentenceTransformerModel


def test_existing_hub_config_with_old_module_path_still_builds(monkeypatch):
    from zeromodels.conversion import zm_config

    spec = {
        "model_module": "zeromodels.models.sentence_transformer",
        "model_class": "SentenceTransformerModel",
        "model_type": "sentence_transformer",
        "backbone_type": "bert",
        "vocab_size": 32,
        "embed_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "mlp_dim": 16,
        "max_position_embeddings": 16,
    }
    monkeypatch.setattr(zm_config, "load_zm_config", lambda repo_id: spec)

    model = SentenceTransformer.from_hub_repo(
        "zeromodels/example-sentence-transformer", load_weights=False
    )
    assert isinstance(model, SentenceTransformerModel)
    assert model.config.backbone_type == "bert"


def test_all_published_sentence_transformer_variants_are_in_offline_registry():
    published = {
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-distilroberta-v1",
        "all-MiniLM-L12-v2",
        "multi-qa-distilbert-cos-v1",
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "paraphrase-multilingual-mpnet-base-v2",
        "paraphrase-albert-small-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-MiniLM-L3-v2",
        "distiluse-base-multilingual-cased-v1",
        "distiluse-base-multilingual-cased-v2",
    }
    registered = {
        name.removeprefix("SentenceTransformer_")
        for name in MODEL_TEST_CONFIGS
        if name.startswith("SentenceTransformer_")
    }
    assert registered == {name.replace("-", "_") for name in published}


@pytest.mark.parametrize(
    "backbone_type",
    ("mpnet", "bert", "roberta", "xlm-roberta", "distilbert", "albert"),
)
def test_backbone_builds_and_returns_sentence_embeddings(backbone_type):
    model = SentenceTransformerModel(
        backbone_type=backbone_type,
        vocab_size=32,
        embed_dim=16,
        embedding_size=8,
        num_layers=1,
        num_heads=2,
        mlp_dim=32,
        max_position_embeddings=32,
        relative_attention_num_buckets=8,
    )
    inputs = {
        "input_ids": np.array([[2, 3, 4, 1]], dtype="int32"),
        "attention_mask": np.array([[1, 1, 1, 0]], dtype="int32"),
    }
    if backbone_type in ("bert", "roberta", "xlm-roberta", "albert"):
        inputs["token_type_ids"] = np.zeros((1, 4), dtype="int32")
    actual = model(inputs, training=False)
    assert actual["embeddings"].shape == (1, 16)
    assert actual["token_embeddings"].shape == (1, 4, 16)


@pytest.mark.parametrize("pooling", ("mean", "cls", "max"))
def test_pooling_parser_accepts_exactly_one_supported_mode(
    monkeypatch, tmp_path, pooling
):
    flags = {
        "pooling_mode_mean_tokens": pooling == "mean",
        "pooling_mode_cls_token": pooling == "cls",
        "pooling_mode_max_tokens": pooling == "max",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(flags), encoding="utf-8")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(config_path)
    )
    assert (
        SentenceTransformerModel._pooling_from_module_dir("example/model", "1_Pooling")
        == pooling
    )


def test_pooling_parser_rejects_ambiguous_modes(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(config_path)
    )
    with pytest.raises(ValueError, match="pooling config"):
        SentenceTransformerModel._pooling_from_module_dir("example/model", "1_Pooling")


def test_sentence_transformer_registered_for_text_embed_task():
    from zeromodels.auto import AutoZMTextEmbed

    assert (
        AutoZMTextEmbed.mapping()["sentence_transformer"]
        is SentenceTransformerTextEmbed
    )


def test_dot_similarity_is_not_cosine_normalized():
    model = SentenceTransformerModel(
        backbone_type="bert",
        vocab_size=32,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_dim=16,
        max_position_embeddings=16,
        similarity_fn_name="dot",
    )
    a = np.array([[2.0, 0.0]], dtype="float32")
    b = np.array([[3.0, 0.0]], dtype="float32")
    np.testing.assert_allclose(ops.convert_to_numpy(model.similarity(a, b)), [[6.0]])
    assert model.get_config()["similarity_fn_name"] == "dot"


def test_cosine_similarity_accepts_numpy_embeddings():
    model = SentenceTransformerModel(
        backbone_type="bert",
        vocab_size=32,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_dim=16,
        max_position_embeddings=16,
    )
    a = np.array([[2.0, 0.0]], dtype="float32")
    b = np.array([[3.0, 0.0]], dtype="float32")
    np.testing.assert_allclose(ops.convert_to_numpy(model.similarity(a, b)), [[1.0]])


def test_sentence_transformer_h5_weights_round_trip(tmp_path):
    kwargs = {
        "backbone_type": "bert",
        "vocab_size": 32,
        "embed_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "mlp_dim": 16,
        "max_position_embeddings": 16,
    }
    model = SentenceTransformerModel(**kwargs)
    inputs = {
        "input_ids": np.array([[2, 3, 4]], dtype="int32"),
        "attention_mask": np.ones((1, 3), dtype="int32"),
        "token_type_ids": np.zeros((1, 3), dtype="int32"),
    }
    expected = ops.convert_to_numpy(model(inputs)["embeddings"])
    path = tmp_path / "model.weights.h5"
    model.save_weights(str(path))

    reloaded = SentenceTransformerModel(**model.config.constructor_kwargs())
    reloaded.load_weights(str(path))
    actual = ops.convert_to_numpy(reloaded(inputs)["embeddings"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
