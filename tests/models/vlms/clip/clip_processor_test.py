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
            "A beautiful sunset over the mountains",
            "A red car driving on the highway",
        ]
        cls.single_text = "A photo of a cat"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.sample_image_path):
            os.remove(cls.sample_image_path)

        for temp_file in cls.temp_files:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def setUp(self):
        self.processor = clip.CLIPProcessor()

    def test_processor_basic_functionality(self):
        result = self.processor(text=self.single_text)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(len(ops.shape(result["input_ids"])), 2)
        self.assertEqual(ops.shape(result["input_ids"])[0], 1)
        self.assertEqual(ops.shape(result["input_ids"])[1], 77)

        result = self.processor(images=self.sample_image_array)
        self.assertIn("images", result)
        processed_images = result["images"]
        self.assertEqual(len(ops.shape(processed_images)), 4)
        self.assertEqual(ops.shape(processed_images)[0], 1)
        self.assertEqual(tuple(ops.shape(processed_images)[1:3]), (224, 224))
        self.assertEqual(ops.shape(processed_images)[3], 3)

        result = self.processor(text=self.single_text, images=self.sample_image_array)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("images", result)

    def test_image_processing_variations(self):
        result_single = self.processor(images=self.sample_image_array)
        self.assertEqual(ops.shape(result_single["images"])[0], 1)

        batch_size = 3
        batch_images = ops.cast(
            keras.random.randint(shape=(batch_size, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result_batch = self.processor(images=batch_images)
        self.assertEqual(ops.shape(result_batch["images"])[0], batch_size)

        result_path = self.processor(image_paths=self.sample_image_path)
        self.assertEqual(ops.shape(result_path["images"])[0], 1)

        result_paths = self.processor(image_paths=self.sample_image_paths)
        self.assertEqual(ops.shape(result_paths["images"])[0], 3)

    def test_combined_text_image_processing(self):
        result = self.processor(text=self.single_text, images=self.sample_image_array)
        self.assertEqual(ops.shape(result["input_ids"])[0], 1)
        self.assertEqual(ops.shape(result["images"])[0], 1)

        result = self.processor(
            text=self.sample_texts[:2], images=self.sample_image_array
        )
        self.assertEqual(ops.shape(result["input_ids"])[0], 2)
        self.assertEqual(ops.shape(result["images"])[0], 1)

        batch_images = ops.cast(
            keras.random.randint(shape=(2, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result = self.processor(text=self.single_text, images=batch_images)
        self.assertEqual(ops.shape(result["input_ids"])[0], 1)
        self.assertEqual(ops.shape(result["images"])[0], 2)

        result = self.processor(
            text=self.sample_texts[:3], image_paths=self.sample_image_paths
        )
        self.assertEqual(ops.shape(result["input_ids"])[0], 3)
        self.assertEqual(ops.shape(result["images"])[0], 3)

    def test_custom_processor_parameters(self):
        custom_processor = clip.CLIPProcessor(image_resolution=336, context_length=49)
        result = custom_processor(text=self.single_text, images=self.sample_image_array)
        self.assertEqual(tuple(ops.shape(result["images"])[1:3]), (336, 336))
        self.assertEqual(ops.shape(result["input_ids"])[1], 49)

        custom_processor = clip.CLIPProcessor(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        result_custom = custom_processor(images=self.sample_image_array)
        result_default = self.processor(images=self.sample_image_array)

        with self.assertRaises(AssertionError):
            self.assertAllClose(
                result_custom["images"], result_default["images"], rtol=1e-4, atol=1e-4
            )

    def test_image_input_type_compatibility(self):
        result_array = self.processor(images=self.sample_image_array)["images"]

        pil_array = keras.utils.img_to_array(self.sample_image_pil)
        result_pil = self.processor(images=pil_array)["images"]

        self.assertEqual(ops.shape(result_array), ops.shape(result_pil))

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

    def test_text_tokenization_details(self):
        result = self.processor(text=self.single_text)

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]
        self.assertEqual(ops.shape(input_ids), ops.shape(attention_mask))

        attention_mask_np = ops.convert_to_numpy(attention_mask)
        unique_values = set(attention_mask_np.flatten())
        for val in unique_values:
            self.assertIn(val, [0, 1])

    def test_invalid_input_combinations(self):
        with self.assertRaises(ValueError):
            self.processor(
                text=self.single_text,
                images=self.sample_image_array,
                image_paths=self.sample_image_path,
            )

        with self.assertRaises((ValueError, TypeError)):
            self.processor()

        with self.assertRaises(ValueError):
            invalid_image = ops.zeros((100, 100))
            self.processor(images=invalid_image)

        with self.assertRaises(ValueError):
            invalid_channels = ops.zeros((100, 100, 5))
            self.processor(images=invalid_channels)

    def test_edge_cases(self):
        small_image = ops.cast(
            keras.random.randint(shape=(1, 1, 3), minval=0, maxval=256), dtype="uint8"
        )
        result_small = self.processor(images=small_image)["images"]
        self.assertEqual(tuple(ops.shape(result_small)[1:3]), (224, 224))

        long_text = " ".join(["word"] * 200)
        result_long = self.processor(text=long_text)
        self.assertEqual(ops.shape(result_long["input_ids"])[1], 77)

        result_empty = self.processor(text="")
        self.assertEqual(ops.shape(result_empty["input_ids"])[1], 77)

    def test_processing_consistency(self):
        result1 = self.processor(text=self.single_text)
        result2 = self.processor(text=self.single_text)
        self.assertAllClose(
            result1["input_ids"], result2["input_ids"], rtol=1e-6, atol=1e-6
        )
        self.assertAllClose(
            result1["attention_mask"], result2["attention_mask"], rtol=1e-6, atol=1e-6
        )

        result1 = self.processor(images=self.sample_image_array)
        result2 = self.processor(images=self.sample_image_array)
        self.assertAllClose(result1["images"], result2["images"], rtol=1e-6, atol=1e-6)

    def test_serialization(self):
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

    def test_output_format(self):
        result_text = self.processor(text=self.single_text)
        self.assertIsInstance(result_text, dict)
        self.assertIn("input_ids", result_text)
        self.assertIn("attention_mask", result_text)
        self.assertNotIn("images", result_text)

        result_image = self.processor(images=self.sample_image_array)
        self.assertIsInstance(result_image, dict)
        self.assertIn("images", result_image)
        self.assertNotIn("input_ids", result_image)
        self.assertNotIn("attention_mask", result_image)

        result_combined = self.processor(
            text=self.single_text, images=self.sample_image_array
        )
        self.assertIsInstance(result_combined, dict)
        self.assertIn("input_ids", result_combined)
        self.assertIn("attention_mask", result_combined)
        self.assertIn("images", result_combined)

    def test_multiple_configuration_combinations(self):
        configs = [
            {
                "image_resolution": 224,
                "context_length": 77,
                "do_resize": True,
                "do_center_crop": True,
                "do_normalize": True,
            },
            {
                "image_resolution": 336,
                "context_length": 49,
                "do_resize": True,
                "do_center_crop": False,
                "do_normalize": True,
            },
            {
                "image_resolution": 224,
                "context_length": 77,
                "do_resize": False,
                "do_center_crop": False,
                "do_normalize": False,
            },
            {
                "image_resolution": 512,
                "context_length": 128,
                "do_resize": True,
                "do_center_crop": True,
                "do_normalize": False,
            },
        ]

        for config in configs:
            with self.subTest(config=config):
                processor = clip.CLIPProcessor(**config)
                result = processor(
                    text=self.single_text, images=self.sample_image_array
                )

                self.assertEqual(len(ops.shape(result["input_ids"])), 2)
                self.assertEqual(ops.shape(result["input_ids"])[0], 1)
                self.assertEqual(
                    ops.shape(result["input_ids"])[1], config.get("context_length", 77)
                )

                self.assertEqual(len(ops.shape(result["images"])), 4)
                self.assertEqual(ops.shape(result["images"])[0], 1)
                self.assertEqual(ops.shape(result["images"])[3], 3)

                if config.get("do_resize", True):
                    expected_res = config.get("image_resolution", 224)
                    if config.get("do_center_crop", True):
                        self.assertEqual(
                            tuple(ops.shape(result["images"])[1:3]),
                            (expected_res, expected_res),
                        )

    def test_nonexistent_image_path(self):
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            self.processor(text=self.single_text, image_paths="nonexistent_file.jpg")

    def test_empty_inputs(self):
        with self.assertRaises((ValueError, IndexError)):
            self.processor(image_paths=[])

        with self.assertRaises((ValueError, TypeError)):
            self.processor(text=None, images=None, image_paths=None)

    def test_float_input_handling(self):
        float_image_01 = keras.random.uniform((100, 100, 3))
        result_01 = self.processor(images=float_image_01)["images"]
        self.assertEqual(tuple(ops.shape(result_01)[1:3]), (224, 224))
        float_image_255 = keras.random.uniform((100, 100, 3)) * 255.0
        result_255 = self.processor(images=float_image_255)["images"]
        self.assertEqual(tuple(ops.shape(result_255)[1:3]), (224, 224))

    def test_special_tokens_handling(self):
        special_text = "<|startoftext|> This is a test <|endoftext|>"
        result = self.processor(text=special_text)
        self.assertIn("input_ids", result)
        self.assertEqual(ops.shape(result["input_ids"])[1], 77)

    def test_batch_size_mismatch_handling(self):
        batch_texts = self.sample_texts[:2]
        batch_images = ops.cast(
            keras.random.randint(shape=(3, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )

        result = self.processor(text=batch_texts, images=batch_images)
        self.assertEqual(ops.shape(result["input_ids"])[0], 2)
        self.assertEqual(ops.shape(result["images"])[0], 3)
