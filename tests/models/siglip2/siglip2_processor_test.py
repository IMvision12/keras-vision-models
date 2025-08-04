import keras
from keras import ops
from keras.src.testing import TestCase

from kvmm.models.siglip2.siglip2_processor import SigLIP2Processor


class TestSigLIP2Processor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_image_array = ops.cast(
            keras.random.randint(shape=(256, 256, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        cls.sample_texts = [
            "A photo of a cat",
            "An image of a dog",
            "A beautiful landscape",
        ]
        cls.sample_text_single = "A photo of a cat"

    def setUp(self):
        self.processor = SigLIP2Processor()

    def test_text_only_processing(self):
        result = self.processor(text=self.sample_text_single)
        self.assertIn("input_ids", result)
        self.assertNotIn("images", result)

        input_ids = result["input_ids"]
        self.assertEqual(len(ops.shape(input_ids)), 2)
        self.assertEqual(ops.shape(input_ids)[0], 1)
        self.assertEqual(ops.shape(input_ids)[1], 64)

    def test_image_only_processing(self):
        result = self.processor(images=self.sample_image_array)
        self.assertIn("images", result)
        self.assertNotIn("input_ids", result)

        processed_images = result["images"]
        self.assertEqual(len(ops.shape(processed_images)), 4)
        self.assertEqual(ops.shape(processed_images)[0], 1)
        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (224, 224))

    def test_combined_processing(self):
        result = self.processor(
            text=self.sample_text_single, images=self.sample_image_array
        )
        self.assertIn("input_ids", result)
        self.assertIn("images", result)

        input_ids = result["input_ids"]
        images = result["images"]
        self.assertEqual(ops.shape(input_ids)[0], 1)
        self.assertEqual(ops.shape(images)[0], 1)

    def test_batch_processing(self):
        result = self.processor(text=self.sample_texts)
        input_ids = result["input_ids"]
        self.assertEqual(ops.shape(input_ids)[0], len(self.sample_texts))

        batch_images = ops.cast(
            keras.random.randint(shape=(3, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result = self.processor(images=batch_images)
        processed_batch = result["images"]
        self.assertEqual(ops.shape(processed_batch)[0], 3)

    def test_custom_configurations(self):
        processor = SigLIP2Processor(image_resolution=336, context_length=128)
        result = processor(text=self.sample_text_single, images=self.sample_image_array)

        processed_images = result["images"]
        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (336, 336))

        input_ids = result["input_ids"]
        self.assertEqual(ops.shape(input_ids)[1], 128)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.processor()

        with self.assertRaises(ValueError):
            self.processor(images=self.sample_image_array, image_paths="dummy_path.jpg")

    def test_token_properties(self):
        self.assertIsInstance(self.processor.vocab_size, int)
        self.assertGreater(self.processor.vocab_size, 0)
        self.assertIsInstance(self.processor.pad_token_id, int)
        self.assertIsInstance(self.processor.eos_token_id, int)

    def test_serialization(self):
        processor = SigLIP2Processor(
            image_resolution=336,
            context_length=128,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        original_result = processor(
            text=self.sample_text_single, images=self.sample_image_array
        )

        config = processor.get_config()
        recreated_processor = SigLIP2Processor.from_config(config)
        recreated_result = recreated_processor(
            text=self.sample_text_single, images=self.sample_image_array
        )

        self.assertAllClose(
            original_result["input_ids"],
            recreated_result["input_ids"],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertAllClose(
            original_result["images"], recreated_result["images"], rtol=1e-6, atol=1e-6
        )
