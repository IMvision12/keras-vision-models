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

    def test_processor_basic_functionality(self):
        result = self.processor(self.sample_image_array)
        processed_images = result["images"]

        self.assertEqual(len(ops.shape(processed_images)), 4)
        self.assertEqual(ops.shape(processed_images)[0], 1)
        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (224, 224))
        self.assertEqual(ops.shape(processed_images)[3], 3)

    def test_custom_image_resolution(self):
        custom_processor = clip.CLIPImageProcessor(image_resolution=336)
        result = custom_processor(self.sample_image_array)
        processed_images = result["images"]

        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (336, 336))

    def test_input_type_compatibility(self):
        result_array = self.processor(self.sample_image_array)["images"]
        pil_array = keras.utils.img_to_array(self.sample_image_pil)
        result_pil = self.processor(pil_array)["images"]
        self.assertEqual(ops.shape(result_array), ops.shape(result_pil))

    def test_image_path_processing(self):
        result_single = self.processor(image_paths=self.sample_image_path)
        processed_single = result_single["images"]
        self.assertEqual(ops.shape(processed_single)[0], 1)
        self.assertEqual(tuple(ops.shape(processed_single)[1:3]), (224, 224))
        result_multiple = self.processor(image_paths=self.sample_image_paths)
        processed_multiple = result_multiple["images"]
        self.assertEqual(ops.shape(processed_multiple)[0], 3)
        self.assertEqual(tuple(ops.shape(processed_multiple)[1:3]), (224, 224))

    def test_batch_processing(self):
        batch_size = 4
        batch_images = ops.cast(
            keras.random.randint(shape=(batch_size, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result = self.processor(batch_images)
        processed_batch = result["images"]
        self.assertEqual(ops.shape(processed_batch)[0], batch_size)
        self.assertEqual(tuple(ops.shape(processed_batch)[1:3]), (224, 224))

    def test_channel_handling(self):
        grayscale_image = ops.cast(
            keras.random.randint(shape=(100, 100, 1), minval=0, maxval=256),
            dtype="uint8",
        )
        result_gray = self.processor(grayscale_image)["images"]
        self.assertEqual(ops.shape(result_gray)[3], 3)
        rgba_image = ops.cast(
            keras.random.randint(shape=(100, 100, 4), minval=0, maxval=256),
            dtype="uint8",
        )
        result_rgba = self.processor(rgba_image)["images"]
        self.assertEqual(ops.shape(result_rgba)[3], 3)

    def test_normalization_options(self):
        processor_default = clip.CLIPImageProcessor()
        result_default = processor_default(self.sample_image_array)["images"]
        processor_custom = clip.CLIPImageProcessor(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        result_custom = processor_custom(self.sample_image_array)["images"]
        processor_no_norm = clip.CLIPImageProcessor(do_normalize=False)
        result_no_norm = processor_no_norm(self.sample_image_array)["images"]
        with self.assertRaises(AssertionError):
            self.assertAllClose(result_default, result_custom, rtol=1e-4, atol=1e-4)

        with self.assertRaises(AssertionError):
            self.assertAllClose(result_default, result_no_norm, rtol=1e-4, atol=1e-4)

    def test_resize_functionality(self):
        processor_resize = clip.CLIPImageProcessor(do_resize=True)
        result_resize = processor_resize(self.sample_image_array)["images"]
        self.assertEqual(tuple(ops.shape(result_resize)[1:3]), (224, 224))
        processor_no_resize = clip.CLIPImageProcessor(
            do_resize=False, do_center_crop=False
        )
        result_no_resize = processor_no_resize(self.sample_image_array)["images"]
        self.assertEqual(tuple(ops.shape(result_no_resize)[1:3]), (256, 256))

    def test_center_crop_functionality(self):
        large_image = ops.cast(
            keras.random.randint(shape=(400, 400, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        processor_crop = clip.CLIPImageProcessor(do_center_crop=True)
        result_crop = processor_crop(large_image)["images"]
        self.assertEqual(tuple(ops.shape(result_crop)[1:3]), (224, 224))
        processor_no_crop = clip.CLIPImageProcessor(do_center_crop=False)
        result_no_crop = processor_no_crop(large_image)["images"]
        self.assertEqual(tuple(ops.shape(result_no_crop)[1:3]), (224, 224))

    def test_aspect_ratio_resize(self):
        rect_image = ops.cast(
            keras.random.randint(shape=(100, 200, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        processor = clip.CLIPImageProcessor()
        result = processor(rect_image)["images"]
        self.assertEqual(tuple(ops.shape(result)[1:3]), (224, 224))

    def test_float_input_handling(self):
        float_image_01 = keras.random.uniform((100, 100, 3))
        result_01 = self.processor(float_image_01)["images"]
        self.assertEqual(tuple(ops.shape(result_01)[1:3]), (224, 224))

        float_image_255 = keras.random.uniform((100, 100, 3)) * 255.0
        result_255 = self.processor(float_image_255)["images"]
        self.assertEqual(tuple(ops.shape(result_255)[1:3]), (224, 224))

    def test_invalid_input_handling(self):
        with self.assertRaises(ValueError):
            invalid_channels = ops.zeros((100, 100, 5))
            self.processor(invalid_channels)

        with self.assertRaises(ValueError):
            invalid_dims = ops.zeros((100, 100))
            self.processor(invalid_dims)

        with self.assertRaises(ValueError):
            self.processor(
                inputs=self.sample_image_array, image_paths=self.sample_image_path
            )

        with self.assertRaises(ValueError):
            self.processor()

    def test_edge_cases(self):
        small_image = ops.cast(
            keras.random.randint(shape=(1, 1, 3), minval=0, maxval=256), dtype="uint8"
        )
        result_small = self.processor(small_image)["images"]
        self.assertEqual(tuple(ops.shape(result_small)[1:3]), (224, 224))
        square_image = ops.cast(
            keras.random.randint(shape=(224, 224, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result_square = self.processor(square_image)["images"]
        self.assertEqual(tuple(ops.shape(result_square)[1:3]), (224, 224))

    def test_processing_consistency(self):
        result1 = self.processor(self.sample_image_array)["images"]
        result2 = self.processor(self.sample_image_array)["images"]
        self.assertAllClose(result1, result2, rtol=1e-6, atol=1e-6)

    def test_serialization(self):
        processor = clip.CLIPImageProcessor(
            image_resolution=336,
            mean=[0.5, 0.5, 0.5],
            std=[0.3, 0.3, 0.3],
            do_center_crop=False,
        )

        original_result = processor(self.sample_image_array)["images"]
        config = processor.get_config()
        recreated_processor = clip.CLIPImageProcessor.from_config(config)
        recreated_result = recreated_processor(self.sample_image_array)["images"]
        self.assertAllClose(original_result, recreated_result, rtol=1e-6, atol=1e-6)

    def test_output_format(self):
        result = self.processor(self.sample_image_array)
        self.assertIsInstance(result, dict)
        self.assertIn("images", result)
        images = result["images"]
        self.assertEqual(len(ops.shape(images)), 4)
        self.assertEqual(ops.shape(images)[0], 1)

    def test_multiple_configuration_combinations(self):
        configs = [
            {
                "image_resolution": 224,
                "do_resize": True,
                "do_center_crop": True,
                "do_normalize": True,
            },
            {
                "image_resolution": 336,
                "do_resize": True,
                "do_center_crop": False,
                "do_normalize": True,
            },
            {
                "image_resolution": 224,
                "do_resize": False,
                "do_center_crop": False,
                "do_normalize": False,
            },
            {
                "image_resolution": 512,
                "do_resize": True,
                "do_center_crop": True,
                "do_normalize": False,
            },
        ]

        for config in configs:
            with self.subTest(config=config):
                processor = clip.CLIPImageProcessor(**config)
                result = processor(self.sample_image_array)["images"]

                self.assertEqual(len(ops.shape(result)), 4)
                self.assertEqual(ops.shape(result)[0], 1)
                self.assertEqual(ops.shape(result)[3], 3)

                if config.get("do_resize", True):
                    expected_res = config.get("image_resolution", 224)
                    if config.get("do_center_crop", True):
                        self.assertEqual(
                            tuple(ops.shape(result)[1:3]), (expected_res, expected_res)
                        )

    def test_nonexistent_image_path(self):
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            self.processor(image_paths="nonexistent_file.jpg")

    def test_empty_image_paths_list(self):
        with self.assertRaises((ValueError, IndexError)):
            self.processor(image_paths=[])
