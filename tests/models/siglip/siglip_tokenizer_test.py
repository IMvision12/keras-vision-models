import os

from keras.src.testing import TestCase

from kvmm.models.vlms.siglip import SigLIPTokenizer
from kvmm.utils import download_file


class TestSigLIPTokenizer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_file = download_file(
            "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip_vocab.model"
        )
        if not os.path.exists(cls.vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {cls.vocab_file}")

    def setUp(self):
        self.tokenizer = SigLIPTokenizer(vocab_file=self.vocab_file, context_length=64)

    def test_tokenize_single_text(self):
        text = "a photo of a cat"
        tokens = self.tokenizer.tokenize(text)

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(token, int) for token in tokens))
        self.assertEqual(tokens[-1], self.tokenizer.eos_token_id)

    def test_tokenize_empty_text(self):
        tokens = self.tokenizer.tokenize("")
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0], self.tokenizer.eos_token_id)

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

    def test_raw_tokenization_values(self):
        test_cases = [
            {
                "text": "a photo of a cat",
                "expected_tokens": [262, 266, 1304, 267, 262, 266, 2783, 1],
                "description": "Basic example - single text",
            },
            {
                "text": "The quick brown fox jumps",
                "expected_tokens": [281, 1476, 3033, 262, 13220, 3535, 264, 1],
                "description": "Simple sentence",
            },
            {
                "text": "",
                "expected_tokens": [1],
                "description": "Empty string edge case",
            },
            {
                "text": "cat",
                "expected_tokens": [2783, 1],
                "description": "Single word",
            },
            {
                "text": "A beautiful sunset over the mountains with golden light reflecting on a crystal clear lake",
                "expected_tokens": [
                    323,
                    825,
                    8969,
                    363,
                    260,
                    5486,
                    275,
                    5274,
                    691,
                    12858,
                    277,
                    262,
                    266,
                    5988,
                    974,
                    4324,
                    1,
                ],
                "description": "Complex description",
            },
            {
                "text": "Photo #123: A cat's best friend! Cost: $50.99",
                "expected_tokens": [
                    4035,
                    4777,
                    4480,
                    304,
                    323,
                    2783,
                    285,
                    264,
                    404,
                    1377,
                    306,
                    6066,
                    304,
                    8567,
                    10588,
                    1,
                ],
                "description": "Text with punctuation and numbers",
            },
            {
                "text": "AI/ML Model Training @2024 #DeepLearning",
                "expected_tokens": [
                    4938,
                    335,
                    9778,
                    4108,
                    4198,
                    2566,
                    2621,
                    3658,
                    1692,
                    521,
                    14579,
                    23045,
                    309,
                    1,
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
                    [281, 1476, 3033, 262, 13220, 3535, 264, 1],
                    [262, 266, 262, 13220, 3535, 264, 1],
                ],
                "description": "Batch of texts",
            },
            {
                "texts": [
                    "hello",
                    "A painting of flowers in a vase on a wooden table",
                    "Night sky filled with stars and a bright full moon over a quiet lake surrounded by tall pine trees",
                ],
                "expected_tokens": [
                    [14647, 1],
                    [
                        323,
                        3023,
                        267,
                        2778,
                        268,
                        262,
                        266,
                        15802,
                        277,
                        262,
                        266,
                        4104,
                        1029,
                        1,
                    ],
                    [
                        4086,
                        4333,
                        2453,
                        275,
                        3583,
                        263,
                        262,
                        266,
                        2923,
                        553,
                        5744,
                        363,
                        262,
                        266,
                        3193,
                        4324,
                        5974,
                        294,
                        4672,
                        9682,
                        2460,
                        1,
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

    def test_call_functionality(self):
        text = "a photo of a cat"
        result = self.tokenizer(inputs=text)
        self.assertIn("input_ids", result)
        input_ids = result["input_ids"]
        self.assertEqual(len(input_ids.shape), 2)
        self.assertEqual(input_ids.shape[0], 1)
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)

        texts = ["a photo of a cat", "a painting of a dog", "hello world"]
        result = self.tokenizer(inputs=texts)
        input_ids = result["input_ids"]
        self.assertEqual(input_ids.shape[0], len(texts))
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)

    def test_detokenize_functionality(self):
        test_texts = ["hello world", "a photo of a cat", "The quick brown fox"]

        for text in test_texts:
            with self.subTest(text=text):
                tokens = self.tokenizer.tokenize(text)
                detokenized = self.tokenizer.detokenize(tokens)
                self.assertIsInstance(detokenized, str)
                self.assertNotIn(self.tokenizer.eos_token, detokenized)
                self.assertNotIn(self.tokenizer.pad_token, detokenized)

    def test_context_length_handling(self):
        for context_length in [16, 32, 128]:
            tokenizer = SigLIPTokenizer(
                vocab_file=self.vocab_file,
                context_length=context_length,
            )
            result = tokenizer(inputs="test text")
            self.assertEqual(result["input_ids"].shape[1], context_length)

        long_text = " ".join(["word"] * 200)
        result = self.tokenizer(inputs=long_text)
        input_ids = result["input_ids"]
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)

    def test_text_processing_methods(self):
        """Test canonicalize_text, remove_punctuation, and preprocess_text methods"""
        canonicalize = self.tokenizer.canonicalize_text
        remove_punct = self.tokenizer.remove_punctuation
        preprocess = self.tokenizer._preprocess_text

        self.assertEqual(canonicalize("Hello, World!"), "Hello World")
        self.assertEqual(canonicalize("Test! @#$ String"), "Test String")
        self.assertEqual(canonicalize("Multiple   spaces"), "Multiple spaces")

        self.assertEqual(remove_punct("Hello, World!"), "Hello World")
        self.assertEqual(remove_punct("Test@#$String"), "TestString")
        self.assertEqual(remove_punct("No punctuation"), "No punctuation")

        self.assertEqual(preprocess("  Multiple   spaces  "), "Multiple spaces")
        self.assertEqual(preprocess("\nNew\nlines\n"), "New lines")
        self.assertEqual(preprocess("Normal text"), "Normal text")

    def test_config_serialization(self):
        config = self.tokenizer.get_config()
        expected_keys = [
            "vocab_file",
            "context_length",
            "do_lower_case",
            "unk_token",
            "pad_token",
            "eos_token",
        ]
        for key in expected_keys:
            self.assertIn(key, config)

        new_tokenizer = SigLIPTokenizer.from_config(config)
        self.assertEqual(new_tokenizer.context_length, self.tokenizer.context_length)
        self.assertEqual(new_tokenizer.vocab_file, self.tokenizer.vocab_file)
