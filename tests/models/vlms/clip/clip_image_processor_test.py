import os
import tempfile

import keras
from keras import ops
from keras.src.testing import TestCase
from PIL import Image

from kvmm.models import clip


class TestCLIPImageProcessor(TestCase):
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

        cls.temp_files = []
        cls.sample_image_paths = []
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix=f"_{i}.png", delete=False)
            sample_array = ops.cast(
                keras.random.randint(shape=(128, 128, 3), minval=0, maxval=256),
                dtype="uint8",
            )
            sample_pil = Image.fromarray(ops.convert_to_numpy(sample_array))
            sample_pil.save(temp_file.name)
            cls.temp_files.append(temp_file)
            cls.sample_image_paths.append(temp_file.name)
            temp_file.close()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.sample_image_path):
            os.remove(cls.sample_image_path)

        for temp_file in cls.temp_files:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def setUp(self):
        self.processor = clip.CLIPImageProcessor()

    def test_image_processing_basic(self):
        result = self.processor(inputs=self.sample_image_array)
        self.assertIn("images", result)
        processed_images = result["images"]
        self.assertEqual(len(ops.shape(processed_images)), 4)
        self.assertEqual(ops.shape(processed_images)[0], 1)
        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (224, 224))
        self.assertEqual(ops.shape(processed_images)[3], 3)

    def test_batch_image_processing(self):
        batch_size = 3
        batch_images = ops.cast(
            keras.random.randint(shape=(batch_size, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result_batch = self.processor(inputs=batch_images)
        self.assertEqual(ops.shape(result_batch["images"])[0], batch_size)

    def test_image_path_processing(self):
        result_path = self.processor(image_paths=self.sample_image_path)
        self.assertEqual(ops.shape(result_path["images"])[0], 1)
        result_paths = self.processor(image_paths=self.sample_image_paths)
        self.assertEqual(ops.shape(result_paths["images"])[0], 3)

    def test_channel_handling(self):
        grayscale_image = ops.cast(
            keras.random.randint(shape=(100, 100, 1), minval=0, maxval=256),
            dtype="uint8",
        )
        result_gray = self.processor(inputs=grayscale_image)["images"]
        self.assertEqual(ops.shape(result_gray)[3], 3)
        rgba_image = ops.cast(
            keras.random.randint(shape=(100, 100, 4), minval=0, maxval=256),
            dtype="uint8",
        )
        result_rgba = self.processor(inputs=rgba_image)["images"]
        self.assertEqual(ops.shape(result_rgba)[3], 3)

    def test_custom_image_parameters(self):
        custom_processor = clip.CLIPImageProcessor(image_resolution=336)
        result = custom_processor(inputs=self.sample_image_array)
        self.assertEqual(tuple(ops.shape(result["images"])[1:3]), (336, 336))
        custom_processor = clip.CLIPImageProcessor(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        result_custom = custom_processor(inputs=self.sample_image_array)
        result_default = self.processor(inputs=self.sample_image_array)

        with self.assertRaises(AssertionError):
            self.assertAllClose(
                result_custom["images"], result_default["images"], rtol=1e-4, atol=1e-4
            )

    def test_float_input_handling(self):
        float_image_01 = keras.random.uniform((100, 100, 3))
        result_01 = self.processor(inputs=float_image_01)["images"]
        self.assertEqual(tuple(ops.shape(result_01)[1:3]), (224, 224))

        float_image_255 = keras.random.uniform((100, 100, 3)) * 255.0
        result_255 = self.processor(inputs=float_image_255)["images"]
        self.assertEqual(tuple(ops.shape(result_255)[1:3]), (224, 224))

    def test_invalid_image_inputs(self):
        with self.assertRaises(ValueError):
            invalid_image = ops.zeros((100, 100))
            self.processor(inputs=invalid_image)

        with self.assertRaises(ValueError):
            invalid_channels = ops.zeros((100, 100, 5))
            self.processor(inputs=invalid_channels)

        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            self.processor(image_paths="nonexistent_file.jpg")

    def test_image_edge_cases(self):
        small_image = ops.cast(
            keras.random.randint(shape=(1, 1, 3), minval=0, maxval=256), dtype="uint8"
        )
        result_small = self.processor(inputs=small_image)["images"]
        self.assertEqual(tuple(ops.shape(result_small)[1:3]), (224, 224))

    def test_image_processing_consistency(self):
        result1 = self.processor(inputs=self.sample_image_array)
        result2 = self.processor(inputs=self.sample_image_array)
        self.assertAllClose(result1["images"], result2["images"], rtol=1e-6, atol=1e-6)
