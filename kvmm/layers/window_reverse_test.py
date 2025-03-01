import keras
import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .window_reverse import WindowReverse


class TestWindowReverse(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.height = 56
        self.width = 56
        self.channels = 96
        self.window_size = 7
        self.windows_height = self.height // self.window_size
        self.windows_width = self.width // self.window_size
        self.num_heads = 3
        self.total_windows = self.batch_size * self.windows_height * self.windows_width
        self.window_elements = self.window_size**2

    def test_init(self):
        layer = WindowReverse(window_size=self.window_size)
        assert layer.window_size == self.window_size
        assert not layer.fused
        assert layer.num_heads is None

        layer = WindowReverse(
            window_size=self.window_size, fused=True, num_heads=self.num_heads
        )
        assert layer.window_size == self.window_size
        assert layer.fused
        assert layer.num_heads == self.num_heads

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            WindowReverse(window_size=self.window_size, fused=True)

    def test_call_standard(self):
        layer = WindowReverse(window_size=self.window_size)
        inputs = ops.ones((self.total_windows, self.window_elements, self.channels))
        outputs = layer(inputs, height=self.height, width=self.width)

        expected_shape = (self.batch_size, self.height, self.width, self.channels)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(expected_shape)
        assert all(
            output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
        )

        manual_reshape = ops.reshape(
            inputs,
            [
                self.batch_size,
                self.windows_height,
                self.windows_width,
                self.window_size,
                self.window_size,
                self.channels,
            ],
        )
        manual_transpose = ops.transpose(manual_reshape, [0, 1, 3, 2, 4, 5])
        manual_result = ops.reshape(
            manual_transpose, [self.batch_size, self.height, self.width, self.channels]
        )

        assert np.allclose(outputs.numpy(), manual_result.numpy())

    def test_call_fused(self):
        channels_per_head = self.channels // self.num_heads
        fused_inputs = ops.ones(
            (
                self.total_windows,
                self.num_heads,
                self.window_elements,
                channels_per_head,
            )
        )

        layer = WindowReverse(
            window_size=self.window_size, fused=True, num_heads=self.num_heads
        )
        outputs = layer(fused_inputs, height=self.height, width=self.width)

        expected_shape = (self.batch_size, self.height, self.width, self.channels)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(expected_shape)
        assert all(
            output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
        )

    def test_call_validation(self):
        layer = WindowReverse(window_size=self.window_size)

        invalid_rank_inputs = ops.ones((self.total_windows, self.window_elements))
        with self.assertRaises(ValueError):
            layer(invalid_rank_inputs, height=self.height, width=self.width)

        layer_fused = WindowReverse(
            window_size=self.window_size, fused=True, num_heads=self.num_heads
        )
        invalid_rank_inputs_fused = ops.ones(
            (self.total_windows, self.num_heads, self.window_elements)
        )
        with self.assertRaises(ValueError):
            layer_fused(invalid_rank_inputs_fused, height=self.height, width=self.width)

        valid_inputs = ops.ones(
            (self.total_windows, self.window_elements, self.channels)
        )
        with self.assertRaises(ValueError):
            layer(valid_inputs)

        with self.assertRaises(ValueError):
            layer(valid_inputs, height=None, width=self.width)

        with self.assertRaises(ValueError):
            layer(valid_inputs, height=self.height, width=None)

        inputs_unknown_channels = keras.Input(
            shape=(self.window_elements, None), batch_size=self.total_windows
        )
        with self.assertRaises(ValueError):
            layer(inputs_unknown_channels, height=self.height, width=self.width)

    def test_different_window_sizes(self):
        test_window_sizes = [4, 8, 14]
        for window_size in test_window_sizes:
            layer = WindowReverse(window_size=window_size)

            height = width = window_size * 8
            windows_height = height // window_size
            windows_width = width // window_size
            total_windows = self.batch_size * windows_height * windows_width
            window_elements = window_size**2

            inputs = ops.ones((total_windows, window_elements, self.channels))
            outputs = layer(inputs, height=height, width=width)

            expected_shape = (self.batch_size, height, width, self.channels)
            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_input_dimensions(self):
        layer = WindowReverse(window_size=self.window_size)

        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            total_windows = batch_size * self.windows_height * self.windows_width
            inputs = ops.ones((total_windows, self.window_elements, self.channels))
            outputs = layer(inputs, height=self.height, width=self.width)

            expected_shape = (batch_size, self.height, self.width, self.channels)
            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

        test_channels = [32, 64, 128]
        for channels in test_channels:
            inputs = ops.ones((self.total_windows, self.window_elements, channels))
            outputs = layer(inputs, height=self.height, width=self.width)

            expected_shape = (self.batch_size, self.height, self.width, channels)
            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_fused_with_different_params(self):
        test_num_heads = [1, 4, 8]
        for num_heads in test_num_heads:
            channels_per_head = 32
            full_channels = num_heads * channels_per_head
            inputs = ops.ones(
                (self.total_windows, num_heads, self.window_elements, channels_per_head)
            )

            layer = WindowReverse(
                window_size=self.window_size, fused=True, num_heads=num_heads
            )
            outputs = layer(inputs, height=self.height, width=self.width)

            expected_shape = (self.batch_size, self.height, self.width, full_channels)
            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_get_config(self):
        layer = WindowReverse(window_size=self.window_size)
        config = layer.get_config()

        assert "window_size" in config
        assert config["window_size"] == self.window_size
        assert "fused" in config
        assert not config["fused"]
        assert "num_heads" in config
        assert config["num_heads"] is None

        reconstructed_layer = WindowReverse.from_config(config)
        assert reconstructed_layer.window_size == layer.window_size
        assert reconstructed_layer.fused == layer.fused
        assert reconstructed_layer.num_heads == layer.num_heads

        layer = WindowReverse(
            window_size=self.window_size, fused=True, num_heads=self.num_heads
        )
        config = layer.get_config()

        assert config["window_size"] == self.window_size
        assert config["fused"]
        assert config["num_heads"] == self.num_heads

        reconstructed_layer = WindowReverse.from_config(config)
        assert reconstructed_layer.window_size == layer.window_size
        assert reconstructed_layer.fused == layer.fused
        assert reconstructed_layer.num_heads == layer.num_heads
