import keras
import numpy as np
from keras import layers, ops
from keras.src.testing import TestCase

from kvmm.layers import PatchesToImageLayer


class TestPatchesToImageLayer(TestCase):
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
            self.patch_size * self.patch_size,
            self.num_patches,
            self.channels,
        )
        self.input_shape_channels_first = (
            self.batch_size,
            self.channels,
            self.patch_size * self.patch_size,
            self.num_patches,
        )
        self.test_inputs_channels_last = ops.ones(self.input_shape_channels_last)
        self.test_inputs_channels_first = ops.ones(self.input_shape_channels_first)

    def test_init_default(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        self.assertEqual(layer.patch_size, self.patch_size)
        self.assertIn(layer.data_format, ["channels_first", "channels_last"])

    def test_build(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        layer.build(self.input_shape_channels_last)
        self.assertEqual(layer.c, self.channels)
        self.assertIsNone(layer.h)
        self.assertIsNone(layer.w)

    def test_call_channels_last(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        outputs = layer(self.test_inputs_channels_last)
        expected_shape = (
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        self.assertEqual(outputs.shape, expected_shape)

    def test_call_with_original_size(self):
        original_height = 30
        original_width = 34
        num_patches_h = (original_height + self.patch_size - 1) // self.patch_size
        num_patches_w = (original_width + self.patch_size - 1) // self.patch_size
        num_patches = num_patches_h * num_patches_w

        input_shape = (
            self.batch_size,
            self.patch_size * self.patch_size,
            num_patches,
            self.channels,
        )
        inputs = ops.ones(input_shape)

        layer = PatchesToImageLayer(patch_size=self.patch_size)
        outputs = layer(
            inputs, original_size=(original_height, original_width), resize=True
        )

        expected_shape = (
            self.batch_size,
            original_height,
            original_width,
            self.channels,
        )
        self.assertEqual(outputs.shape, expected_shape)

    def test_single_patch(self):
        input_shape = (
            self.batch_size,
            self.patch_size * self.patch_size,
            1,
            self.channels,
        )
        inputs = ops.ones(input_shape)

        layer = PatchesToImageLayer(patch_size=self.patch_size)
        outputs = layer(inputs)

        expected_shape = (
            self.batch_size,
            self.patch_size,
            self.patch_size,
            self.channels,
        )
        self.assertEqual(outputs.shape, expected_shape)

    def test_patch_content(self):
        patch_size = 2
        height = width = 4
        channels = 1

        patches_data = np.array(
            [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
        )
        inputs_data = patches_data.T.reshape(1, patch_size * patch_size, 4, channels)
        inputs = ops.convert_to_tensor(inputs_data)

        layer = PatchesToImageLayer(patch_size=patch_size)
        outputs = layer(inputs)

        expected_output = np.arange(16).reshape(1, height, width, channels)
        outputs_np = outputs.numpy()

        self.assertTrue(np.array_equal(outputs_np, expected_output))

    def test_get_config(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        config = layer.get_config()

        self.assertIn("patch_size", config)
        self.assertEqual(config["patch_size"], self.patch_size)

        reconstructed_layer = PatchesToImageLayer.from_config(config)
        self.assertEqual(reconstructed_layer.patch_size, self.patch_size)

    def test_model_integration(self):
        inputs = layers.Input(
            shape=(self.patch_size * self.patch_size, self.num_patches, self.channels)
        )
        patch_layer = PatchesToImageLayer(patch_size=self.patch_size)
        outputs = patch_layer(inputs)

        model = keras.Model(inputs=inputs, outputs=outputs)

        test_input = ops.ones(
            (1, self.patch_size * self.patch_size, self.num_patches, self.channels)
        )
        output = model(test_input)

        expected_shape = (1, self.height, self.width, self.channels)
        self.assertEqual(output.shape, expected_shape)

    def test_resize_without_original_size(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        outputs = layer(self.test_inputs_channels_last, resize=True)

        expected_shape = (
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        self.assertEqual(outputs.shape, expected_shape)
