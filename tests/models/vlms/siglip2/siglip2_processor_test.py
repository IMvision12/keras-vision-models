import os
import tempfile

import keras
from keras import ops
from keras.src.testing import TestCase
from PIL import Image

from kvmm.models.vlms.siglip2.siglip2_processor import SigLIP2Processor


class TestSigLIP2Processor(TestCase):
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

        cls.sample_texts = [
            "A photo of a cat",
            "An image of a dog",
            "A beautiful landscape",
        ]
        cls.sample_text_single = "A photo of a cat"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.sample_image_path):
            os.remove(cls.sample_image_path)

        for temp_file in cls.temp_files:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

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

    def test_text_batch_processing(self):
        result = self.processor(text=self.sample_texts)
        self.assertIn("input_ids", result)
        input_ids = result["input_ids"]
        self.assertEqual(ops.shape(input_ids)[0], len(self.sample_texts))

    def test_image_only_processing(self):
        result = self.processor(images=self.sample_image_array)

        self.assertIn("images", result)
        self.assertNotIn("input_ids", result)

        processed_images = result["images"]
        self.assertEqual(len(ops.shape(processed_images)), 4)
        self.assertEqual(ops.shape(processed_images)[0], 1)
        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (224, 224))
        self.assertEqual(ops.shape(processed_images)[3], 3)

    def test_image_path_only_processing(self):
        result_single = self.processor(image_paths=self.sample_image_path)
        self.assertIn("images", result_single)
        processed_single = result_single["images"]
        self.assertEqual(ops.shape(processed_single)[0], 1)

        result_multiple = self.processor(image_paths=self.sample_image_paths)
        processed_multiple = result_multiple["images"]
        self.assertEqual(ops.shape(processed_multiple)[0], 3)

    def test_combined_text_and_image_processing(self):
        result = self.processor(
            text=self.sample_text_single, images=self.sample_image_array
        )

        self.assertIn("input_ids", result)
        self.assertIn("images", result)

        input_ids = result["input_ids"]
        images = result["images"]
        self.assertEqual(ops.shape(input_ids)[0], 1)
        self.assertEqual(ops.shape(images)[0], 1)

    def test_combined_text_and_image_paths_processing(self):
        result = self.processor(
            text=self.sample_texts, image_paths=self.sample_image_paths
        )
        self.assertIn("input_ids", result)
        self.assertIn("images", result)
        input_ids = result["input_ids"]
        images = result["images"]
        self.assertEqual(ops.shape(input_ids)[0], len(self.sample_texts))
        self.assertEqual(ops.shape(images)[0], len(self.sample_image_paths))

    def test_custom_image_resolution(self):
        processor = SigLIP2Processor(image_resolution=336)
        result = processor(images=self.sample_image_array)
        processed_images = result["images"]
        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (336, 336))

    def test_custom_context_length(self):
        processor = SigLIP2Processor(context_length=128)
        result = processor(text=self.sample_text_single)
        input_ids = result["input_ids"]
        self.assertEqual(ops.shape(input_ids)[1], 128)

    def test_sequence_length_calculation(self):
        result = self.processor(text=self.sample_texts)
        input_ids = result["input_ids"]
        seq_lengths = self.processor.get_sequence_length(input_ids)
        self.assertEqual(len(ops.shape(seq_lengths)), 1)
        self.assertEqual(ops.shape(seq_lengths)[0], len(self.sample_texts))

    def test_batch_image_processing(self):
        batch_size = 4
        batch_images = ops.cast(
            keras.random.randint(shape=(batch_size, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result = self.processor(images=batch_images)
        processed_batch = result["images"]
        self.assertEqual(ops.shape(processed_batch)[0], batch_size)
        self.assertEqual(tuple(ops.shape(processed_batch)[1:3]), (224, 224))

    def test_image_normalization_options(self):
        processor_default = SigLIP2Processor()
        result_default = processor_default(images=self.sample_image_array)["images"]

        processor_custom = SigLIP2Processor(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        result_custom = processor_custom(images=self.sample_image_array)["images"]

        with self.assertRaises(AssertionError):
            self.assertAllClose(result_default, result_custom, rtol=1e-4, atol=1e-4)

    def test_image_resize_and_crop_options(self):
        processor_resize = SigLIP2Processor(do_resize=True, do_center_crop=True)
        result_resize = processor_resize(images=self.sample_image_array)["images"]
        self.assertEqual(tuple(ops.shape(result_resize)[1:3]), (224, 224))

        processor_no_resize = SigLIP2Processor(do_resize=False, do_center_crop=False)
        result_no_resize = processor_no_resize(images=self.sample_image_array)["images"]
        self.assertEqual(tuple(ops.shape(result_no_resize)[1:3]), (256, 256))

    def test_channel_handling(self):
        grayscale_image = ops.cast(
            keras.random.randint(shape=(100, 100, 1), minval=0, maxval=256),
            dtype="uint8",
        )
        result_gray = self.processor(images=grayscale_image)["images"]
        self.assertEqual(ops.shape(result_gray)[3], 3)

        rgba_image = ops.cast(
            keras.random.randint(shape=(100, 100, 4), minval=0, maxval=256),
            dtype="uint8",
        )
        result_rgba = self.processor(images=rgba_image)["images"]
        self.assertEqual(ops.shape(result_rgba)[3], 3)

    def test_float_input_handling(self):
        float_image_01 = keras.random.uniform((100, 100, 3))
        result_01 = self.processor(images=float_image_01)["images"]
        self.assertEqual(tuple(ops.shape(result_01)[1:3]), (224, 224))
        float_image_255 = keras.random.uniform((100, 100, 3)) * 255.0
        result_255 = self.processor(images=float_image_255)["images"]
        self.assertEqual(tuple(ops.shape(result_255)[1:3]), (224, 224))

    def test_error_handling_no_inputs(self):
        with self.assertRaises(ValueError):
            self.processor()

    def test_error_handling_both_images_and_paths(self):
        with self.assertRaises(ValueError):
            self.processor(
                images=self.sample_image_array, image_paths=self.sample_image_path
            )

    def test_error_handling_empty_image_paths(self):
        with self.assertRaises(ValueError):
            self.processor(image_paths=[])

    def test_error_handling_invalid_image_dimensions(self):
        with self.assertRaises(ValueError):
            invalid_dims = ops.zeros((100, 100))
            self.processor(images=invalid_dims)

        with self.assertRaises(ValueError):
            invalid_channels = ops.zeros((100, 100, 5))
            self.processor(images=invalid_channels)

    def test_nonexistent_image_path(self):
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            self.processor(image_paths="nonexistent_file.jpg")

    def test_processing_consistency(self):
        result1 = self.processor(
            text=self.sample_text_single, images=self.sample_image_array
        )
        result2 = self.processor(
            text=self.sample_text_single, images=self.sample_image_array
        )

        self.assertAllClose(
            result1["input_ids"], result2["input_ids"], rtol=1e-6, atol=1e-6
        )
        self.assertAllClose(result1["images"], result2["images"], rtol=1e-6, atol=1e-6)

    def test_serialization(self):
        processor = SigLIP2Processor(
            image_resolution=336,
            context_length=128,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            do_center_crop=False,
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

    def test_output_format_consistency(self):
        result_text = self.processor(text=self.sample_text_single)
        self.assertIsInstance(result_text, dict)
        self.assertIn("input_ids", result_text)

        result_images = self.processor(images=self.sample_image_array)
        self.assertIsInstance(result_images, dict)
        self.assertIn("images", result_images)

        result_both = self.processor(
            text=self.sample_text_single, images=self.sample_image_array
        )
        self.assertIsInstance(result_both, dict)
        self.assertIn("input_ids", result_both)
        self.assertIn("images", result_both)

    def test_token_properties(self):
        self.assertIsInstance(self.processor.vocab_size, int)
        self.assertGreater(self.processor.vocab_size, 0)

        self.assertIsInstance(self.processor.pad_token_id, int)
        self.assertGreaterEqual(self.processor.pad_token_id, 0)

        self.assertIsInstance(self.processor.eos_token_id, int)
        self.assertGreaterEqual(self.processor.eos_token_id, 0)

        self.assertIsInstance(self.processor.unk_token_id, int)
        self.assertGreaterEqual(self.processor.unk_token_id, 0)

    def test_edge_cases_small_images(self):
        small_image = ops.cast(
            keras.random.randint(shape=(1, 1, 3), minval=0, maxval=256), dtype="uint8"
        )
        result_small = self.processor(images=small_image)["images"]
        self.assertEqual(tuple(ops.shape(result_small)[1:3]), (224, 224))

    def test_edge_cases_square_images(self):
        square_image = ops.cast(
            keras.random.randint(shape=(224, 224, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result_square = self.processor(images=square_image)["images"]
        self.assertEqual(tuple(ops.shape(result_square)[1:3]), (224, 224))

    def test_multiple_configuration_combinations(self):
        configs = [
            {
                "image_resolution": 224,
                "context_length": 64,
                "do_resize": True,
                "do_center_crop": True,
                "do_normalize": True,
            },
            {
                "image_resolution": 336,
                "context_length": 128,
                "do_resize": True,
                "do_center_crop": False,
                "do_normalize": True,
            },
            {
                "image_resolution": 224,
                "context_length": 32,
                "do_resize": False,
                "do_center_crop": False,
                "do_normalize": False,
            },
        ]

        for config in configs:
            with self.subTest(config=config):
                processor = SigLIP2Processor(**config)
                result = processor(
                    text=self.sample_text_single, images=self.sample_image_array
                )

                input_ids = result["input_ids"]
                self.assertEqual(len(ops.shape(input_ids)), 2)
                self.assertEqual(ops.shape(input_ids)[0], 1)
                self.assertEqual(ops.shape(input_ids)[1], config["context_length"])

                images = result["images"]
                self.assertEqual(len(ops.shape(images)), 4)
                self.assertEqual(ops.shape(images)[0], 1)
                self.assertEqual(ops.shape(images)[3], 3)

                if config.get("do_resize", True):
                    expected_res = config.get("image_resolution", 224)
                    if config.get("do_center_crop", True):
                        self.assertEqual(
                            tuple(ops.shape(images)[1:3]), (expected_res, expected_res)
                        )
