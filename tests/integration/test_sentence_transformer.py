"""Offline contract tests for the multi-backbone sentence-embedding API."""

import json

import numpy as np
import pytest
from keras import ops, saving

from tests.base.model_test_registry import MODEL_TEST_CONFIGS
from zeromodels.sentence_transformers import (
    ZMSentenceTransformer,
    ZMSentenceTransformerTokenizer,
)
from zeromodels.sentence_transformers.sentence_transformer_backbones import (
    SENTENCE_TRANSFORMER_BACKBONES,
)


def test_public_api_is_the_model_and_the_tokenizer():
    import zeromodels.sentence_transformers as sentence_transformers

    assert sentence_transformers.__all__ == [
        "ZMSentenceTransformer",
        "ZMSentenceTransformerTokenizer",
    ]
    for removed in (
        "SentenceTransformer",
        "SentenceTransformerModel",
        "SentenceTransformerTokenizer",
        "SentenceTransformerTextEmbed",
        "SentenceTransformerConfig",
    ):
        assert not hasattr(sentence_transformers, removed), removed


def test_backbone_registry_matches_accepted_hf_model_types():
    assert set(ZMSentenceTransformer.HF_MODEL_TYPE) == {
        "sentence_transformer",
        *SENTENCE_TRANSFORMER_BACKBONES,
    }


@pytest.mark.parametrize(
    "module",
    (
        "zeromodels.models.sentence_transformer",
        "zeromodels.st_models.sentence_transformer",
        "zeromodels.sentence_transformers.sentence_transformer",
        "zeromodels.sentence_transformers",
    ),
)
@pytest.mark.parametrize(
    "declared",
    ("ZMSentenceTransformer", "SentenceTransformerModel", "SentenceTransformer"),
)
def test_existing_hub_config_with_old_module_path_still_builds(
    monkeypatch, module, declared
):
    from zeromodels.conversion import zm_config

    spec = {
        "model_module": module,
        "model_class": declared,
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

    model = ZMSentenceTransformer.from_hub_repo(
        "zeromodels/example-sentence-transformer", load_weights=False
    )
    assert isinstance(model, ZMSentenceTransformer)
    assert model.config.backbone_type == "bert"


@pytest.mark.parametrize(
    "canonical,legacy",
    (
        (ZMSentenceTransformer, "SentenceTransformerModel"),
        (ZMSentenceTransformerTokenizer, "SentenceTransformerTokenizer"),
    ),
)
def test_keras_registration_uses_new_name_and_preserves_legacy_name(canonical, legacy):
    assert saving.get_registered_name(canonical) == f"zeromodels>{canonical.__name__}"
    assert saving.get_registered_object(f"zeromodels>{legacy}") is canonical


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
        name.removeprefix("ZMSentenceTransformer_")
        for name in MODEL_TEST_CONFIGS
        if name.startswith("ZMSentenceTransformer_")
    }
    assert registered == {name.replace("-", "_") for name in published}


@pytest.mark.parametrize(
    "backbone_type",
    ("mpnet", "bert", "roberta", "xlm-roberta", "distilbert", "albert"),
)
def test_backbone_builds_and_returns_sentence_embeddings(backbone_type):
    model = ZMSentenceTransformer(
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
    if SENTENCE_TRANSFORMER_BACKBONES[backbone_type].token_type_ids:
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
        ZMSentenceTransformer._pooling_from_module_dir("example/model", "1_Pooling")
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
        ZMSentenceTransformer._pooling_from_module_dir("example/model", "1_Pooling")


def test_sentence_transformer_registered_for_text_embed_task():
    from zeromodels.auto import AutoZMTextEmbed

    assert AutoZMTextEmbed.mapping()["sentence_transformer"] is ZMSentenceTransformer


def test_dot_similarity_is_not_cosine_normalized():
    model = ZMSentenceTransformer(
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
    model = ZMSentenceTransformer(
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


@pytest.mark.parametrize(
    "declared", ("ZMSentenceTransformer", "SentenceTransformerModel")
)
def test_sentence_transformer_h5_weights_round_trip_through_auto(
    monkeypatch, tmp_path, declared
):
    from zeromodels import AutoZModel
    from zeromodels.base import base_mixin
    from zeromodels.conversion import zm_config

    kwargs = {
        "backbone_type": "bert",
        "vocab_size": 32,
        "embed_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "mlp_dim": 16,
        "max_position_embeddings": 16,
    }
    model = ZMSentenceTransformer(name="SentenceTransformerModel", **kwargs)
    inputs = {
        "input_ids": np.array([[2, 3, 4]], dtype="int32"),
        "attention_mask": np.ones((1, 3), dtype="int32"),
        "token_type_ids": np.zeros((1, 3), dtype="int32"),
    }
    expected = ops.convert_to_numpy(model(inputs)["embeddings"])
    path = tmp_path / "model.weights.h5"
    model.save_weights(str(path))

    config_path = zm_config.write_zm_config(
        tmp_path, ZMSentenceTransformer, "example", model.config, weight_dtype="float32"
    )
    with open(config_path, encoding="utf-8") as stream:
        spec = json.load(stream)
    spec["model_class"] = declared
    monkeypatch.setattr(zm_config, "load_zm_config", lambda repo_id: spec)
    monkeypatch.setattr(base_mixin, "resolve_weights_url", lambda url: url)
    monkeypatch.setattr(base_mixin, "download_weights", lambda url: str(path))

    reloaded = AutoZModel.from_weights("zeromodels/example")
    assert isinstance(reloaded, ZMSentenceTransformer)
    assert model.name == "SentenceTransformerModel"
    assert reloaded.name == "ZMSentenceTransformer"
    actual = ops.convert_to_numpy(reloaded(inputs)["embeddings"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
