import os
import tempfile

import keras
import numpy as np
from keras import ops
from keras.src.testing import TestCase
from PIL import Image

from kvmm.models import segformer


class TestSegFormerImageProcessor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_image_array = ops.cast(
            keras.random.randint(shape=(10, 10, 3), minval=0, maxval=256), dtype="uint8"
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
        self.processor = segformer.SegFormerImageProcessor

    def test_processor_basic_functionality(self):
        result = self.processor(self.sample_image_array)

        self.assertEqual(len(ops.shape(result)), 4)
        self.assertEqual(ops.shape(result)[0], 1)
        self.assertEqual(tuple(ops.shape(result)[1:3]), (512, 512))
        self.assertEqual(ops.shape(result)[3], 3)

    def test_input_type_compatibility(self):
        result_array = self.processor(self.sample_image_array, return_tensor=False)

        result_pil = self.processor(self.sample_image_pil, return_tensor=False)

        result_path = self.processor(self.sample_image_path, return_tensor=False)

        self.assertEqual(ops.shape(result_array), ops.shape(result_pil))
        self.assertEqual(ops.shape(result_pil), ops.shape(result_path))

    def test_resize_parameters(self):
        custom_size = {"height": 256, "width": 300}
        result = self.processor(
            self.sample_image_array, size=custom_size, return_tensor=False
        )
        self.assertEqual(tuple(ops.shape(result)[1:3]), (256, 300))

        result_no_resize = self.processor(
            self.sample_image_array, do_resize=False, return_tensor=False
        )
        self.assertEqual(tuple(ops.shape(result_no_resize)[1:3]), (10, 10))

    def test_resample_methods(self):
        resample_methods = ["nearest", "bilinear", "bicubic"]

        for method in resample_methods:
            with self.subTest(method=method):
                result = self.processor(
                    self.sample_image_array, resample=method, return_tensor=False
                )
                self.assertEqual(tuple(ops.shape(result)[1:3]), (512, 512))

    def test_rescale_functionality(self):
        result_rescale = self.processor(
            self.sample_image_array, do_normalize=False, return_tensor=False
        )
        self.assertLessEqual(ops.max(result_rescale), 1.0)
        self.assertGreaterEqual(ops.min(result_rescale), 0.0)

        result_no_rescale = self.processor(
            self.sample_image_array,
            do_rescale=False,
            do_normalize=False,
            return_tensor=False,
        )
        self.assertGreater(ops.max(result_no_rescale), 1.0)

    def test_custom_rescale_factor(self):
        custom_factor = 1 / 100
        result_custom = self.processor(
            self.sample_image_array,
            rescale_factor=custom_factor,
            do_normalize=False,
            return_tensor=False,
        )

        expected = ops.cast(self.sample_image_array, "float32") * custom_factor
        expected = ops.expand_dims(expected, 0)
        expected = ops.image.resize(
            expected,
            size=(512, 512),
            interpolation="bilinear",
        )

        self.assertAllClose(result_custom, expected, rtol=1e-4, atol=1e-6)

    def test_normalization_options(self):
        result_default = self.processor(self.sample_image_array, return_tensor=False)

        custom_mean = (0.5, 0.5, 0.5)
        custom_std = (0.1, 0.1, 0.1)
        result_custom = self.processor(
            self.sample_image_array,
            image_mean=custom_mean,
            image_std=custom_std,
            return_tensor=False,
        )

        result_no_norm = self.processor(
            self.sample_image_array, do_normalize=False, return_tensor=False
        )

        with self.assertRaises(AssertionError):
            self.assertAllClose(result_default, result_custom, rtol=1e-4, atol=1e-4)

        with self.assertRaises(AssertionError):
            self.assertAllClose(result_default, result_no_norm, rtol=1e-4, atol=1e-4)

        with self.assertRaises(AssertionError):
            self.assertAllClose(result_custom, result_no_norm, rtol=1e-4, atol=1e-4)

    def test_return_tensor_types(self):
        result_tensor = self.processor(self.sample_image_array, return_tensor=True)
        tensor_np = ops.convert_to_numpy(result_tensor)
        self.assertIsInstance(tensor_np, np.ndarray)

        result_numpy = self.processor(self.sample_image_array, return_tensor=False)
        self.assertIsInstance(result_numpy, np.ndarray)

    def test_float_input_handling(self):
        float_image = ops.cast(keras.random.uniform((10, 10, 3)), "float32")
        result = self.processor(float_image, return_tensor=False)
        self.assertEqual(tuple(ops.shape(result)[1:3]), (512, 512))

    def test_batch_input_handling(self):
        batch_image = ops.cast(
            keras.random.randint(shape=(1, 10, 10, 3), minval=0, maxval=256), "uint8"
        )
        result = self.processor(batch_image, return_tensor=False)
        self.assertEqual(ops.shape(result)[0], 1)

    def test_invalid_input_handling(self):
        with self.assertRaises(ValueError):
            self.processor("nonexistent_file.jpg")

        with self.assertRaises(ValueError):
            self.processor(ops.zeros((10, 10)))

        with self.assertRaises(TypeError):
            self.processor(123)

        with self.assertRaises(ValueError):
            self.processor(ops.ones((10, 10, 3)) * 2.0)

    def test_processing_consistency(self):
        test_images = [
            self.sample_image_array,
            self.sample_image_pil,
            self.sample_image_path,
        ]

        for img in test_images:
            with self.subTest(input_type=type(img).__name__):
                result1 = self.processor(img, return_tensor=False)
                result2 = self.processor(img, return_tensor=False)

                self.assertAllClose(
                    result1,
                    result2,
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_edge_cases(self):
        small_image = ops.cast(
            keras.random.randint(shape=(1, 1, 3), minval=0, maxval=256), "uint8"
        )
        result = self.processor(small_image, return_tensor=False)
        self.assertEqual(tuple(ops.shape(result)[1:3]), (512, 512))

        grayscale_image = ops.cast(
            keras.random.randint(shape=(10, 10), minval=0, maxval=256), "uint8"
        )
        grayscale_pil = Image.fromarray(
            ops.convert_to_numpy(grayscale_image), mode="L"
        ).convert("RGB")
        result = self.processor(grayscale_pil, return_tensor=False)
        self.assertEqual(ops.shape(result)[3], 3)

    def test_parameter_validation(self):
        with self.assertRaises((ValueError, TypeError)):
            self.processor(self.sample_image_array, size={"height": -1, "width": 300})

        with self.assertRaises((ValueError, TypeError)):
            self.processor(self.sample_image_array, resample="invalid_method")

        with self.assertRaises((ValueError, TypeError)):
            self.processor(self.sample_image_array, rescale_factor=-1.0)
