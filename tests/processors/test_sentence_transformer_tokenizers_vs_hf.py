"""Optional official-tokenizer parity for all 13 ST checkpoints; no weights.

Run with transformers installed:
    KERAS_BACKEND=torch pytest tests/processors/test_sentence_transformer_tokenizers_vs_hf.py
"""

import numpy as np
import pytest
from keras import ops

from tests.base.model_test_registry import _SENTENCE_TRANSFORMER_VARIANTS
from zeromodels.sentence_transformers import ZMSentenceTransformerTokenizer

transformers = pytest.importorskip("transformers")

TEXTS = [
    "",
    "  Hello, WORLD!  ",
    "Café naïve résumé",
    "你好，世界！ مرحبا بالعالم",
    "line one\nline two\t123",
    "[CLS] hello [SEP]",
    "hello world " * 600,
]
PAIRS = ["a pair", "", "deuxième phrase", "हिन्दी पाठ", "test", "[MASK]", "world " * 500]


@pytest.mark.parametrize("name", list(_SENTENCE_TRANSFORMER_VARIANTS))
def test_official_tokenizer_parity_without_weights(name):
    repo = f"sentence-transformers/{name}"
    ours = ZMSentenceTransformerTokenizer.from_hf(repo)
    reference = transformers.AutoTokenizer.from_pretrained(
        repo, revision=ours.source_revision, use_fast=True, trust_remote_code=False
    )
    # The 13 supported checkpoints all disable extra ST lowercasing; their own
    # tokenizer.json normalizers still apply. The offline suite covers True.
    assert ours.sentence_transformer_config["do_lower_case"] is False
    for pairs in (None, PAIRS):
        actual = ours(TEXTS, text_pair=pairs)
        expected = reference(
            TEXTS,
            text_pair=pairs,
            padding=True,
            truncation="longest_first",
            max_length=ours.max_seq_len,
            return_tensors="np",
            return_token_type_ids="token_type_ids" in actual,
        )
        for key, value in actual.items():
            np.testing.assert_array_equal(
                ops.convert_to_numpy(value),
                expected[key],
                err_msg=f"{name}: {key}, pairs={pairs is not None}",
            )
