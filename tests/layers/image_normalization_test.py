import keras
from keras import ops
from keras.src.testing import TestCase

from kvmm.layers.image_normalization import (
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
        output_np = layer(self.test_inputs)
        self.assertEqual(output_np.shape, self.input_shape)
        inputs_float = ops.cast(self.test_inputs, dtype="float32") / 255.0

        # Create properly shaped mean and std according to data format
        if keras.config.image_data_format() == "channels_last":
            mean = ops.reshape(ops.convert_to_tensor(IMAGENET_DEFAULT_MEAN), (1, 1, -1))
            std = ops.reshape(ops.convert_to_tensor(IMAGENET_DEFAULT_STD), (1, 1, -1))
        else:
            mean = ops.reshape(ops.convert_to_tensor(IMAGENET_DEFAULT_MEAN), (-1, 1, 1))
            std = ops.reshape(ops.convert_to_tensor(IMAGENET_DEFAULT_STD), (-1, 1, 1))

        expected = (inputs_float - mean) / std
        self.assertAllClose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_inception_preprocessing(self):
        layer = ImageNormalizationLayer(mode="inception")
        output = layer(self.test_inputs)
        inputs_float = ops.cast(self.test_inputs, dtype="float32") / 255.0

        # Create properly shaped mean and std according to data format
        if keras.config.image_data_format() == "channels_last":
            mean = ops.reshape(
                ops.convert_to_tensor(IMAGENET_INCEPTION_MEAN), (1, 1, -1)
            )
            std = ops.reshape(ops.convert_to_tensor(IMAGENET_INCEPTION_STD), (1, 1, -1))
        else:
            mean = ops.reshape(
                ops.convert_to_tensor(IMAGENET_INCEPTION_MEAN), (-1, 1, 1)
            )
            std = ops.reshape(ops.convert_to_tensor(IMAGENET_INCEPTION_STD), (-1, 1, 1))

        expected = (inputs_float - mean) / std
        self.assertAllClose(output, expected, rtol=1e-5, atol=1e-5)

    def test_dpn_preprocessing(self):
        layer = ImageNormalizationLayer(mode="dpn")
        output = layer(self.test_inputs)
        inputs_float = ops.cast(self.test_inputs, dtype="float32") / 255.0

        # Create properly shaped mean and std according to data format
        if keras.config.image_data_format() == "channels_last":
            mean = ops.reshape(ops.convert_to_tensor(IMAGENET_DPN_MEAN), (1, 1, -1))
            std = ops.reshape(ops.convert_to_tensor(IMAGENET_DPN_STD), (1, 1, -1))
        else:
            mean = ops.reshape(ops.convert_to_tensor(IMAGENET_DPN_MEAN), (-1, 1, 1))
            std = ops.reshape(ops.convert_to_tensor(IMAGENET_DPN_STD), (-1, 1, 1))

        expected = (inputs_float - mean) / std
        self.assertAllClose(output, expected, rtol=1e-5, atol=1e-5)

    def test_clip_preprocessing(self):
        layer = ImageNormalizationLayer(mode="clip")
        output = layer(self.test_inputs)
        inputs_float = ops.cast(self.test_inputs, dtype="float32") / 255.0

        # Create properly shaped mean and std according to data format
        if keras.config.image_data_format() == "channels_last":
            mean = ops.reshape(ops.convert_to_tensor(OPENAI_CLIP_MEAN), (1, 1, -1))
            std = ops.reshape(ops.convert_to_tensor(OPENAI_CLIP_STD), (1, 1, -1))
        else:
            mean = ops.reshape(ops.convert_to_tensor(OPENAI_CLIP_MEAN), (-1, 1, 1))
            std = ops.reshape(ops.convert_to_tensor(OPENAI_CLIP_STD), (-1, 1, 1))

        expected = (inputs_float - mean) / std
        self.assertAllClose(output, expected, rtol=1e-5, atol=1e-5)

    def test_zero_to_one_preprocessing(self):
        layer = ImageNormalizationLayer(mode="zero_to_one")
        output = layer(self.test_inputs)

        self.assertTrue(ops.all(output >= 0.0))
        self.assertTrue(ops.all(output <= 1.0))

        expected = ops.cast(self.test_inputs, dtype="float32") / 255.0
        self.assertAllClose(output, expected, rtol=1e-5, atol=1e-5)

    def test_minus_one_to_one_preprocessing(self):
        layer = ImageNormalizationLayer(mode="minus_one_to_one")
        output = layer(self.test_inputs)
        self.assertTrue(ops.all(output >= -1.0))
        self.assertTrue(ops.all(output <= 1.0))

        expected = (ops.cast(self.test_inputs, dtype="float32") / 255.0) * 2.0 - 1.0
        self.assertAllClose(output, expected, rtol=1e-5, atol=1e-5)

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
        import torch

        layer = ImageNormalizationLayer(mode="imagenet")
        output = layer(self.test_inputs)

        if keras.backend.backend() == "torch":
            self.assertEqual(output.dtype, torch.float32)
        else:
            self.assertEqual(output.dtype, "float32")

        inputs_float32 = ops.cast(self.test_inputs, "float32")
        output_float32 = layer(inputs_float32)

        if keras.backend.backend() == "torch":
            self.assertEqual(output_float32.dtype, torch.float32)
        else:
            self.assertEqual(output_float32.dtype, "float32")

        inputs_int32 = ops.cast(self.test_inputs, "int32")
        output_int32 = layer(inputs_int32)

        if keras.backend.backend() == "torch":
            self.assertEqual(output_int32.dtype, torch.float32)
        else:
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
            inputs_float = ops.cast(input_channels_last, dtype="float32") / 255.0
            mean = ops.reshape(ops.convert_to_tensor(IMAGENET_DEFAULT_MEAN), (1, 1, -1))
            std = ops.reshape(ops.convert_to_tensor(IMAGENET_DEFAULT_STD), (1, 1, -1))
            expected = (inputs_float - mean) / std
            self.assertAllClose(output_channels_last, expected, rtol=1e-5, atol=1e-5)
        finally:
            keras.config.set_image_data_format(original_data_format)

        keras.config.set_image_data_format("channels_first")
        try:
            output_channels_first = layer(input_channels_first)
            self.assertEqual(output_channels_first.shape, (2, 3, 224, 224))
            inputs_float = ops.cast(input_channels_first, dtype="float32") / 255.0
            mean = ops.reshape(ops.convert_to_tensor(IMAGENET_DEFAULT_MEAN), (-1, 1, 1))
            std = ops.reshape(ops.convert_to_tensor(IMAGENET_DEFAULT_STD), (-1, 1, 1))
            expected = (inputs_float - mean) / std
            self.assertAllClose(output_channels_first, expected, rtol=1e-5, atol=1e-5)
        finally:
            keras.config.set_image_data_format(original_data_format)
