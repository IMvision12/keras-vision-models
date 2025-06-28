# test_clip_processor.py - Main processor integration tests
import os
import tempfile

import keras
from keras import ops
from keras.src.testing import TestCase
from PIL import Image

from kvmm.models import clip


class TestCLIPProcessor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_image_array = ops.cast(
            keras.random.randint(shape=(256, 256, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        cls.sample_image_pil = Image.fromarray(
            ops.convert_to_numpy(cls.sample_image_array)
        )

        cls.temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cls.sample_image_path = cls.temp_file.name
        cls.sample_image_pil.save(cls.sample_image_path)
        cls.temp_file.close()

        cls.sample_texts = [
            "A photo of a cat",
            "An image of a dog",
            "A beautiful sunset over the mountains",
            "A red car driving on the highway",
        ]
        cls.single_text = "A photo of a cat"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.sample_image_path):
            os.remove(cls.sample_image_path)

    def setUp(self):
        self.processor = clip.CLIPProcessor()

    def test_processor_basic_functionality(self):
        """Test basic text and image processing functionality"""
        # Text processing
        result = self.processor(text=self.single_text)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(len(ops.shape(result["input_ids"])), 2)
        self.assertEqual(ops.shape(result["input_ids"])[0], 1)
        self.assertEqual(ops.shape(result["input_ids"])[1], 77)

        # Image processing
        result = self.processor(images=self.sample_image_array)
        self.assertIn("images", result)
        processed_images = result["images"]
        self.assertEqual(len(ops.shape(processed_images)), 4)
        self.assertEqual(ops.shape(processed_images)[0], 1)
        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (224, 224))
        self.assertEqual(ops.shape(processed_images)[3], 3)

        # Combined processing
        result = self.processor(text=self.single_text, images=self.sample_image_array)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("images", result)

    def test_batch_processing(self):
        """Test batch processing for both text and images"""
        # Batch images
        batch_size = 3
        batch_images = ops.cast(
            keras.random.randint(shape=(batch_size, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result_batch = self.processor(images=batch_images)
        self.assertEqual(ops.shape(result_batch["images"])[0], batch_size)

        # Batch text
        result = self.processor(text=self.sample_texts[:2])
        self.assertEqual(ops.shape(result["input_ids"])[0], 2)

        # Mixed batch sizes
        result = self.processor(text=self.sample_texts[:2], images=batch_images)
        self.assertEqual(ops.shape(result["input_ids"])[0], 2)
        self.assertEqual(ops.shape(result["images"])[0], 3)

    def test_custom_processor_parameters(self):
        """Test processor with custom parameters"""
        custom_processor = clip.CLIPProcessor(image_resolution=336, context_length=49)
        result = custom_processor(text=self.single_text, images=self.sample_image_array)
        self.assertEqual(tuple(ops.shape(result["images"])[1:3]), (336, 336))
        self.assertEqual(ops.shape(result["input_ids"])[1], 49)

    def test_invalid_input_combinations(self):
        """Test error handling for invalid inputs"""
        # Multiple image input types
        with self.assertRaises(ValueError):
            self.processor(
                text=self.single_text,
                images=self.sample_image_array,
                image_paths=self.sample_image_path,
            )

        # No inputs
        with self.assertRaises((ValueError, TypeError)):
            self.processor()

    def test_processing_consistency(self):
        """Test that processing is deterministic"""
        # Text consistency
        result1 = self.processor(text=self.single_text)
        result2 = self.processor(text=self.single_text)
        self.assertAllClose(
            result1["input_ids"], result2["input_ids"], rtol=1e-6, atol=1e-6
        )
        self.assertAllClose(
            result1["attention_mask"], result2["attention_mask"], rtol=1e-6, atol=1e-6
        )

        # Image consistency
        result1 = self.processor(images=self.sample_image_array)
        result2 = self.processor(images=self.sample_image_array)
        self.assertAllClose(result1["images"], result2["images"], rtol=1e-6, atol=1e-6)

    def test_serialization(self):
        """Test processor serialization and deserialization"""
        processor = clip.CLIPProcessor(
            image_resolution=336,
            mean=[0.5, 0.5, 0.5],
            std=[0.3, 0.3, 0.3],
            do_center_crop=False,
            context_length=49,
        )

        original_result = processor(
            text=self.single_text, images=self.sample_image_array
        )

        config = processor.get_config()
        recreated_processor = clip.CLIPProcessor.from_config(config)
        recreated_result = recreated_processor(
            text=self.single_text, images=self.sample_image_array
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
