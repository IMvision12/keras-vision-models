import os

from keras import ops
from keras.src.testing import TestCase

from kvmm.models.siglip2 import SigLIP2Tokenizer
from kvmm.utils import download_file


class TestSigLIP2Tokenizer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_file = download_file(
            "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2_vocab.model"
        )
        if not os.path.exists(cls.vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {cls.vocab_file}")

    def setUp(self):
        self.tokenizer = SigLIP2Tokenizer(
            vocab_file=self.vocab_file, context_length=64, add_bos=False, add_eos=False
        )

    def test_initialization(self):
        self.assertIsInstance(self.tokenizer, SigLIP2Tokenizer)
        self.assertEqual(self.tokenizer.context_length, 64)
        self.assertGreater(self.tokenizer.vocab_size, 0)
        self.assertIsInstance(self.tokenizer.unk_token_id, int)
        self.assertIsInstance(self.tokenizer.pad_token_id, int)

    def test_raw_tokenization_values(self):
        test_cases = [
            {
                "text": "a photo of a cat",
                "expected_tokens": [235250, 2686, 576, 476, 4401],
                "description": "Basic example - single text",
            },
            {
                "text": "The quick brown fox jumped.",
                "expected_tokens": [651, 4320, 8426, 25341, 32292, 235265],
                "description": "Simple sentence with punctuation",
            },
            {
                "text": "",
                "expected_tokens": [],
                "description": "Empty string edge case",
            },
            {
                "text": "dog",
                "expected_tokens": [12240],
                "description": "Single word",
            },
            {
                "text": "A beautiful sunset over the mountains with golden light reflecting on a crystal clear lake",
                "expected_tokens": [
                    235280,
                    4964,
                    22097,
                    1163,
                    573,
                    17200,
                    675,
                    13658,
                    2611,
                    44211,
                    611,
                    476,
                    16162,
                    3110,
                    13492,
                ],
                "description": "Complex description",
            },
            {
                "text": "Photo #123: A dog's best friend! Cost: $50.99",
                "expected_tokens": [
                    7869,
                    1700,
                    235274,
                    235284,
                    235304,
                    235292,
                    586,
                    5929,
                    235303,
                    235256,
                    1963,
                    4034,
                    235341,
                    8449,
                    235292,
                    697,
                    235308,
                    235276,
                    235265,
                    235315,
                    235315,
                ],
                "description": "Text with punctuation and numbers",
            },
            {
                "text": "AI/ML Model Training @2024 #DeepLearning",
                "expected_tokens": [
                    11716,
                    235283,
                    3939,
                    5708,
                    13206,
                    1118,
                    235284,
                    235276,
                    235284,
                    235310,
                    1700,
                    26843,
                    26231,
                ],
                "description": "Mixed case and special characters",
            },
        ]

        for case in test_cases:
            with self.subTest(text=case["text"], description=case["description"]):
                actual_tokens = self.tokenizer.tokenize(case["text"])

                self.assertListEqual(
                    actual_tokens,
                    case["expected_tokens"],
                    f"Raw tokenization mismatch for '{case['text']}'. "
                    f"Expected: {case['expected_tokens']}, Got: {actual_tokens}",
                )

    def test_batch_tokenization(self):
        test_cases = [
            {
                "texts": ["The quick brown fox jumps", "a fox jumps"],
                "expected_tokens": [
                    [651, 4320, 8426, 25341, 36271],
                    [235250, 25341, 36271],
                ],
                "description": "Batch of texts",
            },
            {
                "texts": [
                    "Dog",
                    "A painting of flowers in a vase on a wooden table",
                    "Night sky filled with stars and a bright full moon over a quiet lake surrounded by tall pine trees",
                ],
                "expected_tokens": [
                    [21401],
                    [235280, 11825, 576, 9148, 575, 476, 34557, 611, 476, 11769, 3037],
                    [
                        25738,
                        8203,
                        10545,
                        675,
                        8995,
                        578,
                        476,
                        8660,
                        2247,
                        11883,
                        1163,
                        476,
                        11067,
                        13492,
                        22442,
                        731,
                        13754,
                        23780,
                        8195,
                    ],
                ],
                "description": "Varied length batch",
            },
        ]

        for case in test_cases:
            with self.subTest(texts=case["texts"], description=case["description"]):
                actual_tokens = self.tokenizer.tokenize(case["texts"])

                self.assertListEqual(
                    actual_tokens,
                    case["expected_tokens"],
                    f"Batch tokenization mismatch for {case['texts']}. "
                    f"Expected: {case['expected_tokens']}, Got: {actual_tokens}",
                )

    def test_special_tokens(self):
        tokenizer_bos = SigLIP2Tokenizer(
            vocab_file=self.vocab_file, context_length=64, add_bos=True, add_eos=False
        )
        token_ids = [1, 2, 3, 4, 5]
        result_bos = tokenizer_bos.build_inputs_with_special_tokens(token_ids)
        self.assertEqual(result_bos[0], tokenizer_bos.bos_token_id)

        tokenizer_eos = SigLIP2Tokenizer(
            vocab_file=self.vocab_file, context_length=64, add_bos=False, add_eos=True
        )
        result_eos = tokenizer_eos.build_inputs_with_special_tokens(token_ids)
        self.assertEqual(result_eos[-1], tokenizer_eos.eos_token_id)

    def test_detokenization(self):
        test_texts = ["hello world", "a photo of a cat"]

        for text in test_texts:
            with self.subTest(text=text):
                tokens = self.tokenizer.tokenize(text)
                detokenized = self.tokenizer.detokenize(tokens)

                words = text.lower().split()
                for word in words:
                    if word.isalpha():
                        self.assertIn(word, detokenized.lower())

    def test_context_length_handling(self):
        long_text = " ".join(["word"] * 200)
        result = self.tokenizer(inputs=long_text)
        input_ids = result["input_ids"]
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)

        for context_length in [32, 64, 128]:
            tokenizer = SigLIP2Tokenizer(
                vocab_file=self.vocab_file,
                context_length=context_length,
                add_bos=False,
                add_eos=False,
            )
            result = tokenizer(inputs="test text")
            self.assertEqual(result["input_ids"].shape[1], context_length)

    def test_sequence_operations(self):
        texts = ["hello", "a much longer text with many words"]
        result = self.tokenizer(inputs=texts)
        input_ids = result["input_ids"]

        lengths = self.tokenizer.get_sequence_length(input_ids)
        self.assertEqual(lengths.shape[0], len(texts))
        lengths_np = ops.convert_to_numpy(lengths)
        self.assertLess(lengths_np[0], lengths_np[1])

        decoded_texts = self.tokenizer.batch_decode(input_ids)
        self.assertIsInstance(decoded_texts, list)
        self.assertEqual(len(decoded_texts), len(texts))

    def test_detokenize_functionality(self):
        test_texts = [
            "hello world",
            "a photo of a cat",
            "The quick brown fox",
        ]

        for text in test_texts:
            with self.subTest(text=text):
                tokens = self.tokenizer.tokenize(text)
                detokenized = self.tokenizer.detokenize(tokens)
                self.assertIsInstance(detokenized, str)
                self.assertNotIn(self.tokenizer.eos_token, detokenized)
                self.assertNotIn(self.tokenizer.pad_token, detokenized)
                self.assertNotIn(self.tokenizer.bos_token, detokenized)

    def test_edge_cases(self):
        long_word = "a" * 1000
        result = self.tokenizer(inputs=long_word)
        self.assertEqual(result["input_ids"].shape[1], self.tokenizer.context_length)

        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        result = self.tokenizer(inputs=special_chars)
        self.assertIn("input_ids", result)

        unicode_text = "Café naïve résumé"
        result = self.tokenizer(inputs=unicode_text)
        self.assertIn("input_ids", result)

        numbers = "123456789"
        result = self.tokenizer(inputs=numbers)
        self.assertIn("input_ids", result)

    def test_batch_decode(self):
        texts = ["hello world", "a photo of a cat"]
        result = self.tokenizer(inputs=texts)
        input_ids = result["input_ids"]

        decoded_texts = self.tokenizer.batch_decode(input_ids)

        self.assertIsInstance(decoded_texts, list)
        self.assertEqual(len(decoded_texts), len(texts))

        for decoded_text in decoded_texts:
            self.assertIsInstance(decoded_text, str)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.tokenizer(inputs=None)

        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(-1)

        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(self.tokenizer.vocab_size)

    def test_serialization(self):
        config = self.tokenizer.get_config()

        expected_keys = [
            "vocab_file",
            "context_length",
            "add_bos",
            "add_eos",
            "pad_token",
            "bos_token",
            "eos_token",
            "unk_token",
        ]

        for key in expected_keys:
            self.assertIn(key, config)

        new_tokenizer = SigLIP2Tokenizer.from_config(config)
        self.assertEqual(new_tokenizer.context_length, self.tokenizer.context_length)
        self.assertEqual(new_tokenizer.vocab_size, self.tokenizer.vocab_size)
