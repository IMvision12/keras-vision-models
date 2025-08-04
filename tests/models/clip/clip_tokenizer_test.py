import os

from keras import ops
from keras.src.testing import TestCase

from kvmm.models.clip import CLIPTokenizer
from kvmm.utils import download_file


class TestCLIPTokenizer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_file = download_file(
            "https://github.com/IMvision12/keras-vision-models/releases/download/clip/vocab.json"
        )
        cls.merges_file = download_file(
            "https://github.com/IMvision12/keras-vision-models/releases/download/clip/merges.txt"
        )

        if not os.path.exists(cls.vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {cls.vocab_file}")
        if not os.path.exists(cls.merges_file):
            raise FileNotFoundError(f"Merges file not found: {cls.merges_file}")

    def setUp(self):
        self.tokenizer = CLIPTokenizer(
            vocab_file=self.vocab_file, merges_file=self.merges_file, context_length=77
        )

    def test_raw_tokenization_values(self):
        test_cases = [
            {
                "text": "cat",
                "expected_tokens": [2368],
                "description": "Simple single word",
            },
            {
                "text": "a photo of a cat",
                "expected_tokens": [320, 1125, 539, 320, 2368],
                "description": "Common CLIP prompt",
            },
            {
                "text": "dog",
                "expected_tokens": [1929],
                "description": "Another simple word",
            },
            {
                "text": "",
                "expected_tokens": [],
                "description": "Empty string edge case",
            },
        ]

        for case in test_cases:
            with self.subTest(text=case["text"], description=case["description"]):
                actual_tokens = self.tokenizer._tokenize_single_text(case["text"])
                self.assertListEqual(
                    actual_tokens,
                    case["expected_tokens"],
                    f"Raw tokenization mismatch for '{case['text']}'. "
                    f"Expected: {case['expected_tokens']}, Got: {actual_tokens}",
                )

    def test_full_tokenization_with_special_tokens(self):
        test_cases = [
            {
                "text": "cat",
                "expected_tokens": [49406, 2368, 49407],  # BOS + cat + EOS
                "description": "Simple single word with special tokens",
            },
            {
                "text": "a photo of a cat",
                "expected_tokens": [
                    49406,
                    320,
                    1125,
                    539,
                    320,
                    2368,
                    49407,
                ],  # BOS + tokens + EOS
                "description": "Common CLIP prompt with special tokens",
            },
        ]

        for case in test_cases:
            with self.subTest(text=case["text"], description=case["description"]):
                result = self.tokenizer(inputs=case["text"])
                input_ids = ops.convert_to_numpy(result["input_ids"])[0]
                pad_token_id = self.tokenizer.pad_token_id
                eos_token_id = self.tokenizer.eos_token_id
                actual_length = len(input_ids)
                eos_found = False

                for i, token_id in enumerate(input_ids):
                    if token_id == eos_token_id and not eos_found:
                        eos_found = True
                        continue
                    elif token_id == pad_token_id and eos_found:
                        actual_length = i
                        break

                actual_tokens = input_ids[:actual_length].tolist()
                self.assertListEqual(
                    actual_tokens,
                    case["expected_tokens"],
                    f"Full tokenization mismatch for '{case['text']}'. "
                    f"Expected: {case['expected_tokens']}, Got: {actual_tokens}",
                )

    def test_tokenize_detokenize_roundtrip(self):
        test_texts = [
            "hello world",
            "a photo of a cat",
            "The quick brown fox jumps over the lazy dog",
            "123 456 789",
            "Hello, World!",
        ]

        for text in test_texts:
            tokens = self.tokenizer.tokenize(text)
            detokenized = self.tokenizer.detokenize(tokens)

            words = text.lower().split()
            for word in words:
                if word.isalpha():
                    self.assertIn(word, detokenized.lower())

    def test_different_context_lengths(self):
        for context_length in [16, 32, 77, 128]:
            tokenizer = CLIPTokenizer(
                vocab_file=self.vocab_file,
                merges_file=self.merges_file,
                context_length=context_length,
            )

            result = tokenizer(inputs="test text")
            self.assertEqual(result["input_ids"].shape[1], context_length)
            self.assertEqual(result["attention_mask"].shape[1], context_length)

    def test_tokenization_consistency(self):
        test_cases = [
            "a photo of a cat",
            "The quick brown fox",
            "Hello, World!",
            "123 456",
            "",
            "   ",
        ]

        for text in test_cases:
            with self.subTest(text=text):
                result1 = self.tokenizer(inputs=text)
                result2 = self.tokenizer(inputs=text)

                self.assertListEqual(
                    ops.convert_to_numpy(result1["input_ids"]).tolist(),
                    ops.convert_to_numpy(result2["input_ids"]).tolist(),
                )
                self.assertListEqual(
                    ops.convert_to_numpy(result1["attention_mask"]).tolist(),
                    ops.convert_to_numpy(result2["attention_mask"]).tolist(),
                )

    def test_text_tokenization_details(self):
        result = self.tokenizer(inputs="A photo of a cat")
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]
        self.assertEqual(ops.shape(input_ids), ops.shape(attention_mask))
        attention_mask_np = ops.convert_to_numpy(attention_mask)
        unique_values = set(attention_mask_np.flatten())
        for val in unique_values:
            self.assertIn(val, [0, 1])

    def test_text_edge_cases(self):
        long_text = " ".join(["word"] * 200)
        result_long = self.tokenizer(inputs=long_text)
        self.assertEqual(ops.shape(result_long["input_ids"])[1], 77)
        result_empty = self.tokenizer(inputs="")
        self.assertEqual(ops.shape(result_empty["input_ids"])[1], 77)
