import keras
from keras import layers, ops
from keras.src.testing import TestCase

from kvmm.layers import ImageToPatchesLayer


class TestImageToPatchesLayer(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.height = 32
        self.width = 32
        self.channels = 3
        self.patch_size = 8
        self.num_patches_h = self.height // self.patch_size
        self.num_patches_w = self.width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.input_shape_channels_last = (
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        self.input_shape_channels_first = (
            self.batch_size,
            self.channels,
            self.height,
            self.width,
        )
        self.test_inputs_channels_last = ops.ones(self.input_shape_channels_last)
        self.test_inputs_channels_first = ops.ones(self.input_shape_channels_first)

    def test_init_default(self):
        layer = ImageToPatchesLayer(patch_size=self.patch_size)
        self.assertEqual(layer.patch_size, self.patch_size)
        self.assertFalse(layer.resize)
        self.assertIn(layer.data_format, ["channels_first", "channels_last"])

    def test_build_invalid_input_shape(self):
        layer = ImageToPatchesLayer(patch_size=self.patch_size)
        with self.assertRaises(ValueError):
            invalid_shape = (self.batch_size, self.height, self.width)
            invalid_input = ops.ones(invalid_shape)
            layer(invalid_input)

    def test_call_channels_last(self):
        layer = ImageToPatchesLayer(patch_size=self.patch_size)
        outputs = layer(self.test_inputs_channels_last)
        expected_shape = (
            self.batch_size,
            self.patch_size * self.patch_size,
            self.num_patches,
            self.channels,
        )
        self.assertEqual(outputs.shape, expected_shape)
        self.assertFalse(layer.resize)

    def test_uneven_dimensions(self):
        uneven_height = 30
        uneven_width = 34
        input_shape = (self.batch_size, uneven_height, uneven_width, self.channels)
        inputs = ops.ones(input_shape)

        layer = ImageToPatchesLayer(patch_size=self.patch_size)
        outputs = layer(inputs)

        # Convert tensor values to integers
        expected_height = int(ops.ceil(uneven_height / self.patch_size))
        expected_width = int(ops.ceil(uneven_width / self.patch_size))
        expected_patches = expected_height * expected_width

        expected_shape = (
            self.batch_size,
            self.patch_size * self.patch_size,
            expected_patches,
            self.channels,
        )
        self.assertEqual(outputs.shape, expected_shape)
        self.assertTrue(layer.resize)

    def test_single_patch(self):
        input_shape = (self.batch_size, self.patch_size, self.patch_size, self.channels)
        inputs = ops.ones(input_shape)

        layer = ImageToPatchesLayer(patch_size=self.patch_size)
        outputs = layer(inputs)

        expected_shape = (
            self.batch_size,
            self.patch_size * self.patch_size,
            1,
            self.channels,
        )
        self.assertEqual(outputs.shape, expected_shape)
        self.assertFalse(layer.resize)

    def test_patch_content(self):
        height = width = 4
        patch_size = 2
        channels = 1

        input_data = keras.ops.arange(16, dtype="float32")
        input_data = keras.ops.reshape(input_data, (1, height, width, channels))
        inputs = keras.ops.convert_to_tensor(input_data)

        layer = ImageToPatchesLayer(patch_size=patch_size)
        outputs = layer(inputs)

        expected_patches = keras.ops.convert_to_tensor(
            [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]],
            dtype="float32",
        )

        for i in range(4):
            patch = outputs[0, :, i, 0]
            is_equal = keras.ops.all(keras.ops.equal(patch, expected_patches[i]))
            self.assertTrue(keras.ops.convert_to_numpy(is_equal).item())

    def test_get_config(self):
        layer = ImageToPatchesLayer(patch_size=self.patch_size)
        config = layer.get_config()

        self.assertIn("patch_size", config)
        self.assertEqual(config["patch_size"], self.patch_size)

        reconstructed_layer = ImageToPatchesLayer.from_config(config)
        self.assertEqual(reconstructed_layer.patch_size, self.patch_size)

    def test_model_integration(self):
        inputs = layers.Input(shape=(self.height, self.width, self.channels))
        patch_layer = ImageToPatchesLayer(patch_size=self.patch_size)
        outputs = patch_layer(inputs)

        model = keras.Model(inputs=inputs, outputs=outputs)

        test_input = ops.ones((1, self.height, self.width, self.channels))
        output = model(test_input)

        expected_shape = (
            1,
            self.patch_size * self.patch_size,
            self.num_patches,
            self.channels,
        )
        self.assertEqual(output.shape, expected_shape)
