"""Checkpoint metadata must control batching without replacing the JSON pipeline."""

import json
from types import SimpleNamespace

import keras
import numpy as np
import pytest
from huggingface_hub.utils import EntryNotFoundError
from keras import ops
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from zeromodels.st_models import SentenceTransformerTokenizer

SHA = "a" * 40


@pytest.fixture
def tokenizer_path(tmp_path):
    vocab = {
        token: i
        for i, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "Hello"]
        )
    }
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


def metadata():
    return {
        "preprocessor_module": "zeromodels.st_models",
        "preprocessor_class": "SentenceTransformerTokenizer",
        "tokenizer_file": "assets/tokenizer.json",
        "backbone_type": "bert",
        "tokenizer_config": {
            "model_max_length": 512,
            "do_lower_case": False,
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": {"content": "[PAD]", "special": True},
        },
        "sentence_transformer_config": {
            "max_seq_length": 6,
            "do_lower_case": False,
        },
    }


def mock_repo(monkeypatch, tmp_path, files):
    calls = []

    def download(repo_id, filename, token=None, revision=None):
        calls.append((filename, revision))
        if filename not in files:
            raise EntryNotFoundError(f"Missing {filename}")
        value = files[filename]
        if not isinstance(value, dict) and not isinstance(value, list):
            return str(value)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda *args, **kwargs: SimpleNamespace(sha=SHA),
    )
    return calls


def test_hub_metadata_uses_one_snapshot_and_official_sequence_limit(
    monkeypatch, tmp_path, tokenizer_path
):
    calls = mock_repo(
        monkeypatch,
        tmp_path,
        {
            "zm_preprocessor.json": metadata(),
            "assets/tokenizer.json": tokenizer_path,
        },
    )
    tok = SentenceTransformerTokenizer.from_weights(
        "zeromodels/example", revision="release"
    )
    encoded = tok(["hello world " * 10, "hello"])
    assert tok.max_seq_len == 6
    assert tok.cls_token_id == 2 and tok.sep_token_id == 3
    assert tok.pad_token_id == 0
    assert encoded["input_ids"].shape == (2, 6)
    assert ops.convert_to_numpy(encoded["input_ids"])[0, -1] == 3
    np.testing.assert_array_equal(
        ops.convert_to_numpy(encoded["attention_mask"])[1], [1, 1, 1, 0, 0, 0]
    )
    assert tok.source_revision == SHA
    assert all(revision == SHA for _, revision in calls)
    assert {name for name, _ in calls} == {
        "zm_preprocessor.json",
        "assets/tokenizer.json",
    }


def test_portable_preprocessor_export_loads_again(
    monkeypatch, tmp_path, tokenizer_path
):
    from zeromodels.conversion.zm_config import write_zm_preprocessor

    original = SentenceTransformerTokenizer(
        tokenizer_file=str(tokenizer_path),
        backbone_type="bert",
        max_seq_len=6,
        tokenizer_config={"padding_side": "left"},
        sentence_transformer_config={"do_lower_case": True},
    )
    path = write_zm_preprocessor(tmp_path, original, "example")
    with open(path, encoding="utf-8") as stream:
        spec = json.load(stream)
    assert spec["tokenizer_file"] == "tokenizer.json"
    mock_repo(
        monkeypatch,
        tmp_path,
        {"zm_preprocessor.json": spec, "tokenizer.json": tokenizer_path},
    )
    reloaded = SentenceTransformerTokenizer.from_weights("example", revision=SHA)
    for key, expected in original(["Hello world", "hello"]).items():
        np.testing.assert_array_equal(
            ops.convert_to_numpy(reloaded(["Hello world", "hello"])[key]),
            ops.convert_to_numpy(expected),
        )


def test_legacy_nested_config_and_explicit_limit_keep_backbone(
    monkeypatch, tmp_path, tokenizer_path
):
    mock_repo(
        monkeypatch,
        tmp_path,
        {
            "zm_config.json": {
                "model_type": "sentence_transformer",
                "text_config": {"max_seq_len": 8, "backbone_type": "bert"},
            },
            "tokenizer.json": tokenizer_path,
        },
    )
    tok = SentenceTransformerTokenizer.from_weights("zeromodels/example", max_seq_len=5)
    assert tok.max_seq_len == 5
    assert tok.backbone_type == "bert"
    assert "token_type_ids" in tok(["hello"])


def test_sentence_transformer_lowercase_is_separate_from_tokenizer_normalization(
    tokenizer_path,
):
    tc = {"do_lower_case": False, "pad_token": "[PAD]"}
    plain = SentenceTransformerTokenizer(
        tokenizer_file=str(tokenizer_path),
        tokenizer_config=tc,
        sentence_transformer_config={"do_lower_case": False},
    )
    extra = SentenceTransformerTokenizer(
        tokenizer_file=str(tokenizer_path),
        tokenizer_config=tc,
        sentence_transformer_config={"do_lower_case": True},
    )
    assert plain.tokenize("Hello") == [7]
    assert extra.tokenize("Hello") == [5]
    raw = Tokenizer.from_file(str(tokenizer_path))
    raw.normalizer = normalizers.BertNormalizer(lowercase=True)
    raw.save(str(tokenizer_path))
    inherited = SentenceTransformerTokenizer(
        tokenizer_file=str(tokenizer_path),
        tokenizer_config={"do_lower_case": True},
        sentence_transformer_config={"do_lower_case": False},
    )
    assert inherited.tokenize("Hello") == [5]


def test_padding_truncation_pairs_and_serialization(tokenizer_path):
    tok = SentenceTransformerTokenizer(
        tokenizer_file=str(tokenizer_path),
        backbone_type="bert",
        tokenizer_config={"padding_side": "left", "truncation_side": "left"},
        sentence_transformer_config={"max_seq_length": 6},
    )
    clone = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(tok)
    )
    inputs = ["hello world hello world hello", "hello"]
    for current in (tok, clone):
        encoded = current(inputs, text_pair=["world", "world"])
        ids = ops.convert_to_numpy(encoded["input_ids"])
        np.testing.assert_array_equal(ids[1], [0, 2, 5, 3, 6, 3])
        np.testing.assert_array_equal(
            ops.convert_to_numpy(encoded["token_type_ids"])[1], [0, 0, 0, 0, 1, 1]
        )
        assert ids[0, 0] == 2 and ids[0, -1] == 3


def test_upstream_module_path_and_sidecars(monkeypatch, tmp_path, tokenizer_path):
    calls = mock_repo(
        monkeypatch,
        tmp_path,
        {
            "modules.json": [{"path": "0_Transformer"}],
            "0_Transformer/config.json": {"model_type": "bert"},
            "0_Transformer/tokenizer_config.json": {"model_max_length": 512},
            "0_Transformer/special_tokens_map.json": {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
            },
            "0_Transformer/sentence_bert_config.json": {
                "max_seq_length": 5,
                "do_lower_case": True,
            },
            "0_Transformer/tokenizer.json": tokenizer_path,
        },
    )
    tok = SentenceTransformerTokenizer.from_weights("hf:sentence-transformers/example")
    assert tok.max_seq_len == 5 and tok.cls_token_id == 2
    assert tok.tokenize("Hello") == [5]
    assert all(revision == SHA for _, revision in calls)


@pytest.mark.parametrize(
    "filename",
    [
        "../tokenizer.json",
        "/tmp/tokenizer.json",
        "https://example.com/tokenizer.json",
        "C:\\tokenizer.json",
    ],
)
def test_hub_rejects_non_repository_tokenizer_paths(monkeypatch, tmp_path, filename):
    spec = metadata()
    spec["tokenizer_file"] = filename
    mock_repo(monkeypatch, tmp_path, {"zm_preprocessor.json": spec})
    with pytest.raises(ValueError, match="repository-relative"):
        SentenceTransformerTokenizer.from_weights("zeromodels/example")
