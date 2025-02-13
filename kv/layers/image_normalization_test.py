import keras
import numpy as np
from keras import ops
from keras.src.testing import TestCase

from kv.layers.image_normalization import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_DPN_MEAN,
    IMAGENET_DPN_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageNormalizationLayer,
)


class TestImageNormalizationLayer(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.height = 224
        self.width = 224
        self.channels = 3
        self.input_shape = (self.batch_size, self.height, self.width, self.channels)
        self.test_inputs = ops.cast(
            keras.random.uniform(
                (self.batch_size, self.height, self.width, self.channels), 0, 255
            ),
            dtype="uint8",
        )

    def test_init(self):
        modes = [
            "imagenet",
            "inception",
            "dpn",
            "clip",
            "zero_to_one",
            "minus_one_to_one",
        ]
        for mode in modes:
            layer = ImageNormalizationLayer(mode=mode)
            self.assertEqual(layer.mode, mode)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            ImageNormalizationLayer(mode="invalid_mode")

    def test_imagenet_preprocessing(self):
        layer = ImageNormalizationLayer(mode="imagenet")
        output = layer(self.test_inputs)
        self.assertEqual(output.shape, self.input_shape)
        output_np = output.numpy()
        inputs_float = self.test_inputs.numpy().astype(np.float32) / 255.0
        expected = (inputs_float - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(
            IMAGENET_DEFAULT_STD
        )
        self.assertTrue(np.allclose(output_np, expected, rtol=1e-5, atol=1e-5))

    def test_inception_preprocessing(self):
        layer = ImageNormalizationLayer(mode="inception")
        output = layer(self.test_inputs)
        output_np = output.numpy()
        inputs_float = self.test_inputs.numpy().astype(np.float32) / 255.0
        expected = (inputs_float - np.array(IMAGENET_INCEPTION_MEAN)) / np.array(
            IMAGENET_INCEPTION_STD
        )
        self.assertTrue(np.allclose(output_np, expected, rtol=1e-5, atol=1e-5))

    def test_dpn_preprocessing(self):
        layer = ImageNormalizationLayer(mode="dpn")
        output = layer(self.test_inputs)
        output_np = output.numpy()
        inputs_float = self.test_inputs.numpy().astype(np.float32) / 255.0
        expected = (inputs_float - np.array(IMAGENET_DPN_MEAN)) / np.array(
            IMAGENET_DPN_STD
        )
        self.assertTrue(np.allclose(output_np, expected, rtol=1e-5, atol=1e-5))

    def test_clip_preprocessing(self):
        layer = ImageNormalizationLayer(mode="clip")
        output = layer(self.test_inputs)
        output_np = output.numpy()
        inputs_float = self.test_inputs.numpy().astype(np.float32) / 255.0
        expected = (inputs_float - np.array(OPENAI_CLIP_MEAN)) / np.array(
            OPENAI_CLIP_STD
        )
        self.assertTrue(np.allclose(output_np, expected, rtol=1e-5, atol=1e-5))

    def test_zero_to_one_preprocessing(self):
        layer = ImageNormalizationLayer(mode="zero_to_one")
        output = layer(self.test_inputs)

        output_np = output.numpy()

        self.assertTrue(np.all(output_np >= 0.0))
        self.assertTrue(np.all(output_np <= 1.0))

        expected = self.test_inputs.numpy().astype(np.float32) / 255.0
        self.assertTrue(np.allclose(output_np, expected, rtol=1e-5, atol=1e-5))

    def test_minus_one_to_one_preprocessing(self):
        layer = ImageNormalizationLayer(mode="minus_one_to_one")
        output = layer(self.test_inputs)

        output_np = output.numpy()
        self.assertTrue(np.all(output_np >= -1.0))
        self.assertTrue(np.all(output_np <= 1.0))

        expected = (self.test_inputs.numpy().astype(np.float32) / 255.0) * 2.0 - 1.0
        self.assertTrue(np.allclose(output_np, expected, rtol=1e-5, atol=1e-5))

    def test_different_input_shapes(self):
        test_shapes = [
            (2, 160, 160, 3),
            (1, 299, 299, 3),
            (8, 32, 32, 3),
        ]

        layer = ImageNormalizationLayer(mode="imagenet")

        for shape in test_shapes:
            inputs = ops.cast(keras.random.uniform(shape, 0, 255), dtype="uint8")
            output = layer(inputs)
            self.assertEqual(output.shape, shape)

    def test_get_config(self):
        layer = ImageNormalizationLayer(mode="imagenet")
        config = layer.get_config()

        self.assertIn("mode", config)
        self.assertEqual(config["mode"], "imagenet")

        reconstructed_layer = ImageNormalizationLayer.from_config(config)
        self.assertEqual(reconstructed_layer.mode, "imagenet")

    def test_output_dtypes(self):
        layer = ImageNormalizationLayer(mode="imagenet")
        output = layer(self.test_inputs)
        self.assertEqual(output.dtype, "float32")

        inputs_float32 = ops.cast(self.test_inputs, "float32")
        output_float32 = layer(inputs_float32)
        self.assertEqual(output_float32.dtype, "float32")

        inputs_int32 = ops.cast(self.test_inputs, "int32")
        output_int32 = layer(inputs_int32)
        self.assertEqual(output_int32.dtype, "float32")

    def test_data_format(self):
        input_channels_last = ops.cast(
            keras.random.uniform((2, 224, 224, 3), 0, 255),
            dtype="uint8",
        )

        input_channels_first = ops.cast(
            keras.random.uniform((2, 3, 224, 224), 0, 255),
            dtype="uint8",
        )

        layer = ImageNormalizationLayer(mode="imagenet")

        original_data_format = keras.config.image_data_format()
        keras.config.set_image_data_format("channels_last")
        try:
            output_channels_last = layer(input_channels_last)

            self.assertEqual(output_channels_last.shape, (2, 224, 224, 3))

            output_np = output_channels_last.numpy()
            inputs_float = input_channels_last.numpy().astype(np.float32) / 255.0
            mean = np.reshape(IMAGENET_DEFAULT_MEAN, (1, 1, 1, 3))
            std = np.reshape(IMAGENET_DEFAULT_STD, (1, 1, 1, 3))
            expected = (inputs_float - mean) / std
            self.assertTrue(np.allclose(output_np, expected, rtol=1e-5, atol=1e-5))
        finally:
            keras.config.set_image_data_format(original_data_format)

        keras.config.set_image_data_format("channels_first")
        try:
            output_channels_first = layer(input_channels_first)

            self.assertEqual(output_channels_first.shape, (2, 3, 224, 224))

            output_np = output_channels_first.numpy()
            inputs_float = input_channels_first.numpy().astype(np.float32) / 255.0
            mean = np.reshape(IMAGENET_DEFAULT_MEAN, (1, 3, 1, 1))
            std = np.reshape(IMAGENET_DEFAULT_STD, (1, 3, 1, 1))
            expected = (inputs_float - mean) / std
            self.assertTrue(np.allclose(output_np, expected, rtol=1e-5, atol=1e-5))
        finally:
            keras.config.set_image_data_format(original_data_format)
