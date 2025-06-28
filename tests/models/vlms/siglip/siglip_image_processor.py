import os
import tempfile

import keras
from keras import ops
from keras.src.testing import TestCase
from PIL import Image

from kvmm.models import siglip


class TestSigLIPImageProcessor(TestCase):
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

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.sample_image_path):
            os.remove(cls.sample_image_path)

    def setUp(self):
        self.processor = siglip.SigLIPImageProcessor()

    def test_basic_functionality(self):
        result = self.processor(self.sample_image_array)
        self.assertEqual(len(ops.shape(result)), 4)
        self.assertEqual(ops.shape(result)[0], 1)
        self.assertEqual(tuple(ops.shape(result)[1:3]), (224, 224))
        self.assertEqual(ops.shape(result)[3], 3)

    def test_custom_image_resolution(self):
        custom_processor = siglip.SigLIPImageProcessor(image_resolution=336)
        result = custom_processor(self.sample_image_array)
        self.assertEqual(tuple(ops.shape(result)[1:3]), (336, 336))

    def test_batch_processing(self):
        batch_size = 4
        batch_images = ops.cast(
            keras.random.randint(shape=(batch_size, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result = self.processor(batch_images)
        self.assertEqual(ops.shape(result)[0], batch_size)
        self.assertEqual(tuple(ops.shape(result)[1:3]), (224, 224))

    def test_channel_handling(self):
        grayscale_image = ops.cast(
            keras.random.randint(shape=(100, 100, 1), minval=0, maxval=256),
            dtype="uint8",
        )
        result_gray = self.processor(grayscale_image)
        self.assertEqual(ops.shape(result_gray)[3], 3)

        rgba_image = ops.cast(
            keras.random.randint(shape=(100, 100, 4), minval=0, maxval=256),
            dtype="uint8",
        )
        result_rgba = self.processor(rgba_image)
        self.assertEqual(ops.shape(result_rgba)[3], 3)

    def test_normalization_options(self):
        processor_default = siglip.SigLIPImageProcessor()
        result_default = processor_default(self.sample_image_array)
        processor_no_norm = siglip.SigLIPImageProcessor(do_normalize=False)
        result_no_norm = processor_no_norm(self.sample_image_array)

        with self.assertRaises(AssertionError):
            self.assertAllClose(result_default, result_no_norm, rtol=1e-4, atol=1e-4)

    def test_resize_and_crop_functionality(self):
        processor_resize = siglip.SigLIPImageProcessor(
            do_resize=True, do_center_crop=False
        )
        result_resize = processor_resize(self.sample_image_array)
        self.assertEqual(tuple(ops.shape(result_resize)[1:3]), (224, 224))
        large_image = ops.cast(
            keras.random.randint(shape=(400, 400, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        processor_crop = siglip.SigLIPImageProcessor(
            do_center_crop=True, do_resize=False
        )
        result_crop = processor_crop(large_image)
        self.assertEqual(tuple(ops.shape(result_crop)[1:3]), (224, 224))

    def test_invalid_input_handling(self):
        with self.assertRaises(ValueError):
            invalid_channels = ops.zeros((100, 100, 5))
            self.processor(invalid_channels)

        with self.assertRaises(ValueError):
            invalid_dims = ops.zeros((100, 100))
            self.processor(invalid_dims)

        with self.assertRaises(ValueError):
            self.processor()

    def test_edge_cases(self):
        small_image = ops.cast(
            keras.random.randint(shape=(1, 1, 3), minval=0, maxval=256), dtype="uint8"
        )
        result_small = self.processor(small_image)
        self.assertEqual(tuple(ops.shape(result_small)[1:3]), (224, 224))
        square_image = ops.cast(
            keras.random.randint(shape=(224, 224, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result_square = self.processor(square_image)
        self.assertEqual(tuple(ops.shape(result_square)[1:3]), (224, 224))

    def test_serialization(self):
        processor = siglip.SigLIPImageProcessor(
            image_resolution=336,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            do_center_crop=False,
        )
        original_result = processor(self.sample_image_array)
        config = processor.get_config()
        recreated_processor = siglip.SigLIPImageProcessor.from_config(config)
        recreated_result = recreated_processor(self.sample_image_array)
        self.assertAllClose(original_result, recreated_result, rtol=1e-6, atol=1e-6)
