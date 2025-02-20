import keras
import numpy as np
from keras import layers, ops
from keras.src.testing import TestCase

from .patches_to_images import PatchesToImageLayer


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
        self.input_shape = (
            self.batch_size,
            self.patch_size * self.patch_size,
            self.num_patches,
            self.channels,
        )
        self.test_inputs = ops.ones(self.input_shape)

    def test_init_default(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        assert layer.patch_size == self.patch_size
        assert layer.data_format in ["channels_first", "channels_last"]

    def test_build(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        layer.build(self.input_shape)
        assert layer.c == self.channels

    def test_call_default(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        outputs = layer(self.test_inputs)

        expected_shape_channels_last = (
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        expected_shape_channels_first = (
            self.batch_size,
            self.channels,
            self.height,
            self.width,
        )

        expected_shape = (
            expected_shape_channels_first
            if layer.data_format == "channels_first"
            else expected_shape_channels_last
        )
        assert outputs.shape == expected_shape

    def test_call_with_original_size(self):
        original_height = 32
        original_width = 32
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        outputs = layer(
            self.test_inputs, original_size=(original_height, original_width)
        )

        expected_shape_channels_last = (
            self.batch_size,
            original_height,
            original_width,
            self.channels,
        )
        expected_shape_channels_first = (
            self.batch_size,
            self.channels,
            original_height,
            original_width,
        )

        expected_shape = (
            expected_shape_channels_first
            if layer.data_format == "channels_first"
            else expected_shape_channels_last
        )
        assert outputs.shape == expected_shape

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

        expected_shape_channels_last = (
            self.batch_size,
            self.patch_size,
            self.patch_size,
            self.channels,
        )
        expected_shape_channels_first = (
            self.batch_size,
            self.channels,
            self.patch_size,
            self.patch_size,
        )

        expected_shape = (
            expected_shape_channels_first
            if layer.data_format == "channels_first"
            else expected_shape_channels_last
        )
        assert outputs.shape == expected_shape

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
        assert np.array_equal(outputs_np, expected_output)

    def test_get_config(self):
        layer = PatchesToImageLayer(patch_size=self.patch_size)
        config = layer.get_config()

        assert "patch_size" in config
        assert config["patch_size"] == self.patch_size

        reconstructed_layer = PatchesToImageLayer.from_config(config)
        assert reconstructed_layer.patch_size == self.patch_size

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

        expected_shape = (
            (
                1,
                self.height,
                self.width,
                self.channels,
            )
            if patch_layer.data_format == "channels_last"
            else (
                1,
                self.channels,
                self.height,
                self.width,
            )
        )
        assert output.shape == expected_shape
