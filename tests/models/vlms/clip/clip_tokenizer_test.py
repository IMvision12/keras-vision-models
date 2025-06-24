import os

from keras import ops
from keras.src.testing import TestCase

from kvmm.models.vlms.clip import CLIPTokenizer
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

    def test_tokenizer_initialization(self):
        self.assertIsInstance(self.tokenizer, CLIPTokenizer)
        self.assertEqual(self.tokenizer.context_length, 77)
        self.assertEqual(self.tokenizer.bos_token, "<|startoftext|>")
        self.assertEqual(self.tokenizer.eos_token, "<|endoftext|>")
        self.assertEqual(self.tokenizer.pad_token, "<|endoftext|>")
        self.assertGreater(len(self.tokenizer.encoder), 0)

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.bos_token_id, 49406)
        self.assertEqual(self.tokenizer.eos_token_id, 49407)

    def test_vocab_size(self):
        vocab_size = self.tokenizer.vocab_size
        self.assertIsInstance(vocab_size, int)
        self.assertGreater(vocab_size, 0)
        self.assertEqual(vocab_size, len(self.tokenizer.encoder))

    def test_tokenize_single_text(self):
        text = "a photo of a cat"
        tokens = self.tokenizer.tokenize(text)

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(token, int) for token in tokens))

    def test_tokenize_empty_text(self):
        tokens = self.tokenizer.tokenize("")
        self.assertIsInstance(tokens, list)

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

    def test_expected_tokenization_values(self):
        test_cases = [
            {
                "text": "cat",
                "expected_tokens": [49406, 2368, 49407],
                "description": "Simple single word",
            },
            {
                "text": "a photo of a cat",
                "expected_tokens": [49406, 320, 1125, 539, 320, 2368, 49407],
                "description": "Common CLIP prompt",
            },
            {
                "text": "dog",
                "expected_tokens": [49406, 1929, 49407],
                "description": "Another simple word",
            },
        ]

        for case in test_cases:
            with self.subTest(text=case["text"], description=case["description"]):
                result = self.tokenizer(inputs=case["text"])
                input_ids = ops.convert_to_numpy(result["input_ids"])[0]
                eos_positions = [
                    i
                    for i, token_id in enumerate(input_ids)
                    if token_id == self.tokenizer.eos_token_id
                ]
                if eos_positions:
                    actual_tokens = input_ids[: eos_positions[0] + 1].tolist()
                else:
                    actual_tokens = [
                        token_id
                        for token_id in input_ids
                        if token_id != self.tokenizer.eos_token_id
                        or token_id == self.tokenizer.eos_token_id
                    ]
                    actual_tokens = actual_tokens[: len(case["expected_tokens"])]

                self.assertListEqual(
                    actual_tokens,
                    case["expected_tokens"],
                    f"Tokenization mismatch for '{case['text']}'. "
                    f"Expected: {case['expected_tokens']}, Got: {actual_tokens}",
                )

    def test_build_inputs_with_special_tokens(self):
        token_ids = [1, 2, 3, 4, 5]
        result = self.tokenizer.build_inputs_with_special_tokens(token_ids)

        self.assertEqual(result[0], self.tokenizer.bos_token_id)
        self.assertEqual(result[-1], self.tokenizer.eos_token_id)
        self.assertEqual(result[1:-1], token_ids)

    def test_create_attention_mask(self):
        token_ids = [1, 2, 3, 4, 5]
        mask = self.tokenizer.create_attention_mask(token_ids)

        self.assertEqual(len(mask), self.tokenizer.context_length)
        self.assertTrue(all(mask[: len(token_ids)]))

        if len(mask) > len(token_ids):
            self.assertTrue(all(not m for m in mask[len(token_ids) :]))

    def test_call_single_text(self):
        text = "a photo of a cat"
        result = self.tokenizer(inputs=text)

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        self.assertEqual(len(input_ids.shape), 2)
        self.assertEqual(input_ids.shape[0], 1)
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)
        self.assertEqual(attention_mask.shape, input_ids.shape)

    def test_call_batch_texts(self):
        texts = ["a photo of a cat", "a painting of a dog", "hello world"]
        result = self.tokenizer(inputs=texts)

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        self.assertEqual(len(input_ids.shape), 2)
        self.assertEqual(input_ids.shape[0], len(texts))
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)
        self.assertEqual(attention_mask.shape, input_ids.shape)

    def test_call_empty_input(self):
        with self.assertRaises(ValueError):
            self.tokenizer(inputs=None)

    def test_context_length_truncation(self):
        long_text = " ".join(["word"] * 200)
        result = self.tokenizer(inputs=long_text)
        input_ids = result["input_ids"]
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)

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

    def test_error_handling(self):
        with self.assertRaises(FileNotFoundError):
            CLIPTokenizer(
                vocab_file="nonexistent_vocab.json",
                merges_file="nonexistent_merges.txt",
            )

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
