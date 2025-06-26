import os

from keras import ops
from keras.src.testing import TestCase

from kvmm.models.vlms.siglip2 import SigLIP2Tokenizer
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

    def test_tokenizer_initialization(self):
        self.assertIsInstance(self.tokenizer, SigLIP2Tokenizer)
        self.assertEqual(self.tokenizer.context_length, 64)
        self.assertEqual(self.tokenizer.unk_token, "<unk>")
        self.assertEqual(self.tokenizer.pad_token, "<pad>")
        self.assertEqual(self.tokenizer.eos_token, "<eos>")
        self.assertEqual(self.tokenizer.bos_token, "<bos>")
        self.assertGreater(len(self.tokenizer.encoder), 0)

    def test_special_tokens(self):
        self.assertIsInstance(self.tokenizer.unk_token_id, int)
        self.assertIsInstance(self.tokenizer.pad_token_id, int)
        self.assertIsInstance(self.tokenizer.eos_token_id, int)
        self.assertIsInstance(self.tokenizer.bos_token_id, int)

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
        self.assertEqual(len(tokens), 0)

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

    def test_build_inputs_with_special_tokens(self):
        token_ids = [1, 2, 3, 4, 5]
        result = self.tokenizer.build_inputs_with_special_tokens(token_ids)

        self.assertEqual(result, token_ids)
        tokenizer_with_bos = SigLIP2Tokenizer(
            vocab_file=self.vocab_file, context_length=64, add_bos=True, add_eos=False
        )
        result_bos = tokenizer_with_bos.build_inputs_with_special_tokens(token_ids)
        self.assertEqual(result_bos[0], tokenizer_with_bos.bos_token_id)
        self.assertEqual(result_bos[1:], token_ids)
        tokenizer_with_eos = SigLIP2Tokenizer(
            vocab_file=self.vocab_file, context_length=64, add_bos=False, add_eos=True
        )
        result_eos = tokenizer_with_eos.build_inputs_with_special_tokens(token_ids)
        self.assertEqual(result_eos[-1], tokenizer_with_eos.eos_token_id)
        self.assertEqual(result_eos[:-1], token_ids)

    def test_prepare_for_model(self):
        text = "a photo of a cat"
        result = self.tokenizer.prepare_for_model(text)

        self.assertIn("input_ids", result)
        input_ids = result["input_ids"]

        self.assertIsInstance(input_ids, list)
        self.assertEqual(len(input_ids), self.tokenizer.context_length)

    def test_call_single_text(self):
        text = "a photo of a cat"
        result = self.tokenizer(inputs=text)

        self.assertIn("input_ids", result)
        input_ids = result["input_ids"]

        self.assertEqual(len(input_ids.shape), 2)
        self.assertEqual(input_ids.shape[0], 1)
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)

    def test_call_batch_texts(self):
        texts = ["a photo of a cat", "a painting of a dog", "hello world"]
        result = self.tokenizer(inputs=texts)

        self.assertIn("input_ids", result)
        input_ids = result["input_ids"]

        self.assertEqual(len(input_ids.shape), 2)
        self.assertEqual(input_ids.shape[0], len(texts))
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)

    def test_call_empty_input(self):
        with self.assertRaises(ValueError):
            self.tokenizer(inputs=None)

    def test_context_length_truncation(self):
        long_text = " ".join(["word"] * 200)
        result = self.tokenizer(inputs=long_text)
        input_ids = result["input_ids"]
        self.assertEqual(input_ids.shape[1], self.tokenizer.context_length)

    def test_different_context_lengths(self):
        for context_length in [16, 32, 64, 128]:
            tokenizer = SigLIP2Tokenizer(
                vocab_file=self.vocab_file,
                context_length=context_length,
                add_bos=False,
                add_eos=False,
            )

            result = tokenizer(inputs="test text")
            self.assertEqual(result["input_ids"].shape[1], context_length)

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

    def test_batch_decode(self):
        texts = ["hello world", "a photo of a cat"]
        result = self.tokenizer(inputs=texts)
        input_ids = result["input_ids"]

        decoded_texts = self.tokenizer.batch_decode(input_ids)

        self.assertIsInstance(decoded_texts, list)
        self.assertEqual(len(decoded_texts), len(texts))

        for decoded_text in decoded_texts:
            self.assertIsInstance(decoded_text, str)

    def test_get_sequence_length(self):
        texts = ["hello", "a much longer text with many words"]
        result = self.tokenizer(inputs=texts)
        input_ids = result["input_ids"]

        lengths = self.tokenizer.get_sequence_length(input_ids)

        self.assertEqual(lengths.shape[0], len(texts))
        lengths_np = ops.convert_to_numpy(lengths)
        self.assertLess(lengths_np[0], lengths_np[1])

    def test_truncate_sequences(self):
        text = "a very long text that should be truncated"
        result = self.tokenizer(inputs=text)
        input_ids = result["input_ids"]

        max_length = 10
        truncated = self.tokenizer.truncate_sequences(input_ids, max_length)

        self.assertEqual(truncated.shape[1], max_length)

    def test_vocab_methods(self):
        """Test vocabulary-related methods"""
        vocab_size = self.tokenizer.vocab_size
        self.assertIsInstance(vocab_size, int)
        self.assertGreater(vocab_size, 0)

        vocabulary = self.tokenizer.get_vocabulary()
        self.assertIsInstance(vocabulary, list)
        self.assertEqual(len(vocabulary), vocab_size)

        for i in range(min(100, vocab_size)):
            token = self.tokenizer.id_to_token(i)
            self.assertIsInstance(token, str)
            token_id = self.tokenizer.token_to_id(token)
            self.assertEqual(token_id, i)

    def test_special_token_configuration(self):
        tokenizer_both = SigLIP2Tokenizer(
            vocab_file=self.vocab_file, context_length=64, add_bos=True, add_eos=True
        )

        text = "hello world"
        tokens = tokenizer_both.tokenize(text)
        processed_tokens = tokenizer_both.build_inputs_with_special_tokens(tokens)
        self.assertEqual(processed_tokens[0], tokenizer_both.bos_token_id)
        self.assertEqual(processed_tokens[-1], tokenizer_both.eos_token_id)

    def test_custom_special_tokens(self):
        custom_tokenizer = SigLIP2Tokenizer(
            vocab_file=self.vocab_file,
            context_length=64,
            pad_token="<custom_pad>",
            bos_token="<custom_bos>",
            eos_token="<custom_eos>",
            unk_token="<custom_unk>",
        )

        self.assertEqual(custom_tokenizer.pad_token, "<custom_pad>")
        self.assertEqual(custom_tokenizer.bos_token, "<custom_bos>")
        self.assertEqual(custom_tokenizer.eos_token, "<custom_eos>")
        self.assertEqual(custom_tokenizer.unk_token, "<custom_unk>")

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

    def test_config_serialization(self):
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
        self.assertEqual(new_tokenizer.vocab_file, self.tokenizer.vocab_file)
        self.assertEqual(new_tokenizer.add_bos, self.tokenizer.add_bos)
        self.assertEqual(new_tokenizer.add_eos, self.tokenizer.add_eos)

    def test_sentencepiece_specific_features(self):
        self.assertIsNotNone(self.tokenizer.sp_model)

        text = "hello world"
        sp_tokens = self.tokenizer.sp_model.encode_as_ids(text)
        tokenizer_tokens = self.tokenizer.tokenize(text)

        self.assertEqual(sp_tokens, tokenizer_tokens)
        decoded_sp = self.tokenizer.sp_model.decode_ids(sp_tokens)
        decoded_tokenizer = self.tokenizer.detokenize(tokenizer_tokens)

        self.assertEqual(decoded_sp, decoded_tokenizer)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(-1)

        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(self.tokenizer.vocab_size)

        # Test None inputs
        with self.assertRaises(ValueError):
            self.tokenizer(inputs=None)
