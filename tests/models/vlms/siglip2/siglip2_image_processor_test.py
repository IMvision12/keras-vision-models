import os
import tempfile

import keras
from keras import ops
from keras.src.testing import TestCase
from PIL import Image

from kvmm.models import siglip2


class TestSigLIP2ImageProcessor(TestCase):
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
        self.processor = siglip2.SigLIP2ImageProcessor()

    def test_custom_resolution(self):
        custom_processor = siglip2.SigLIP2ImageProcessor(image_resolution=336)
        result = custom_processor(self.sample_image_array)
        self.assertEqual(tuple(ops.shape(result)[1:3]), (336, 336))

    def test_image_path_processing(self):
        result_single = self.processor(image_paths=self.sample_image_path)
        processed_single = result_single
        self.assertEqual(ops.shape(processed_single)[0], 1)
        self.assertEqual(tuple(ops.shape(processed_single)[1:3]), (224, 224))

        result_multiple = self.processor(
            image_paths=[
                self.sample_image_path,
                self.sample_image_path,
                self.sample_image_path,
            ]
        )
        processed_multiple = result_multiple
        self.assertEqual(ops.shape(processed_multiple)[0], 3)
        self.assertEqual(tuple(ops.shape(processed_multiple)[1:3]), (224, 224))

    def test_batch_processing(self):
        batch_images = ops.cast(
            keras.random.randint(shape=(3, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        result = self.processor(batch_images)
        self.assertEqual(ops.shape(result)[0], 3)
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
        processor_default = siglip2.SigLIP2ImageProcessor()
        result_default = processor_default(self.sample_image_array)

        processor_custom = siglip2.SigLIP2ImageProcessor(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        result_custom = processor_custom(self.sample_image_array)

        with self.assertRaises(AssertionError):
            self.assertAllClose(result_default, result_custom, rtol=1e-4, atol=1e-4)

    def test_resize_and_center_crop_functionality(self):
        processor_resize = siglip2.SigLIP2ImageProcessor(do_resize=True)
        processor_no_resize = siglip2.SigLIP2ImageProcessor(
            do_resize=False, do_center_crop=False
        )
        processor_crop = siglip2.SigLIP2ImageProcessor(do_center_crop=True)
        processor_no_crop = siglip2.SigLIP2ImageProcessor(do_center_crop=False)

        large_image = ops.cast(
            keras.random.randint(shape=(400, 400, 3), minval=0, maxval=256),
            dtype="uint8",
        )

        result_resize = processor_resize(self.sample_image_array)
        result_no_resize = processor_no_resize(self.sample_image_array)

        result_crop = processor_crop(large_image)
        result_no_crop = processor_no_crop(large_image)

        self.assertEqual(tuple(ops.shape(result_resize)[1:3]), (224, 224))
        self.assertEqual(tuple(ops.shape(result_no_resize)[1:3]), (256, 256))
        self.assertEqual(tuple(ops.shape(result_crop)[1:3]), (224, 224))
        self.assertEqual(tuple(ops.shape(result_no_crop)[1:3]), (224, 224))

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            invalid_channels = ops.zeros((100, 100, 5))
            self.processor(invalid_channels)

        with self.assertRaises(ValueError):
            invalid_dims = ops.zeros((100, 100))
            self.processor(invalid_dims)

        with self.assertRaises(ValueError):
            self.processor()

    def test_serialization(self):
        processor = siglip2.SigLIP2ImageProcessor(
            image_resolution=336,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        original_result = processor(self.sample_image_array)
        config = processor.get_config()
        recreated_processor = siglip2.SigLIP2ImageProcessor.from_config(config)
        recreated_result = recreated_processor(self.sample_image_array)

        self.assertAllClose(original_result, recreated_result, rtol=1e-6, atol=1e-6)

    def test_layer_inheritance(self):
        self.assertIsInstance(self.processor, keras.layers.Layer)
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = self.processor(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        test_input = ops.cast(
            keras.random.randint(shape=(1, 128, 128, 3), minval=0, maxval=256),
            dtype="uint8",
        )
        prediction = model.predict(test_input, verbose=0)
        self.assertEqual(tuple(ops.shape(prediction)[1:3]), (224, 224))
