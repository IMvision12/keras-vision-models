import keras
from keras import ops
from keras.src.testing import TestCase

from kvmm.layers import WindowPartition


class TestWindowPartition(TestCase):
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
        self.qkv_mult = 3
        self.input_shape = (self.batch_size, self.height, self.width, self.channels)
        self.test_inputs = ops.ones(self.input_shape)

    def test_init(self):
        layer = WindowPartition(window_size=self.window_size)
        assert layer.window_size == self.window_size
        assert not layer.fused
        assert layer.num_heads is None
        assert layer.qkv_mult == 3

        layer = WindowPartition(
            window_size=self.window_size, fused=True, num_heads=self.num_heads
        )
        assert layer.window_size == self.window_size
        assert layer.fused
        assert layer.num_heads == self.num_heads
        assert layer.qkv_mult == 3

        layer = WindowPartition(window_size=self.window_size, qkv_mult=1)
        assert layer.qkv_mult == 1

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            WindowPartition(window_size=self.window_size, fused=True)

    def test_call_standard(self):
        layer = WindowPartition(window_size=self.window_size)
        outputs = layer(self.test_inputs, height=self.height, width=self.width)

        expected_windows = self.batch_size * self.windows_height * self.windows_width
        expected_window_elements = self.window_size**2
        expected_shape = (expected_windows, expected_window_elements, self.channels)

        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(expected_shape)
        assert all(
            output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
        )

        manual_reshape = ops.reshape(
            self.test_inputs,
            [
                self.batch_size,
                self.windows_height,
                self.window_size,
                self.windows_width,
                self.window_size,
                self.channels,
            ],
        )
        manual_transpose = ops.transpose(manual_reshape, [0, 1, 3, 2, 4, 5])
        manual_result = ops.reshape(
            manual_transpose,
            [
                self.batch_size * self.windows_height * self.windows_width,
                self.window_size**2,
                self.channels,
            ],
        )

        self.assertAllClose(outputs, manual_result)

    def test_call_fused(self):
        channels_per_head = self.channels // self.num_heads
        fused_inputs = ops.ones(
            (
                self.batch_size,
                self.height,
                self.width,
                self.qkv_mult * self.num_heads * channels_per_head,
            )
        )

        layer = WindowPartition(
            window_size=self.window_size, fused=True, num_heads=self.num_heads
        )
        outputs = layer(fused_inputs, height=self.height, width=self.width)

        expected_windows = self.batch_size * self.windows_height * self.windows_width
        expected_window_elements = self.window_size**2
        expected_shape = (
            self.qkv_mult,
            expected_windows,
            self.num_heads,
            expected_window_elements,
            channels_per_head,
        )

        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(expected_shape)
        assert all(
            output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
        )

    def test_call_validation(self):
        layer = WindowPartition(window_size=self.window_size)

        invalid_rank_inputs = ops.ones((self.batch_size, self.height, self.width))
        with self.assertRaises(ValueError):
            layer(invalid_rank_inputs, height=self.height, width=self.width)

        with self.assertRaises(ValueError):
            layer(self.test_inputs)

        with self.assertRaises(ValueError):
            layer(self.test_inputs, height=None, width=self.width)

        with self.assertRaises(ValueError):
            layer(self.test_inputs, height=self.height, width=None)

        inputs_unknown_channels = keras.Input(
            shape=(self.height, self.width, None), batch_size=self.batch_size
        )
        with self.assertRaises(ValueError):
            layer(inputs_unknown_channels, height=self.height, width=self.width)

    def test_different_window_sizes(self):
        test_window_sizes = [4, 8, 14]
        for window_size in test_window_sizes:
            layer = WindowPartition(window_size=window_size)

            height = width = window_size * 8
            inputs = ops.ones((self.batch_size, height, width, self.channels))

            outputs = layer(inputs, height=height, width=width)
            windows_height = height // window_size
            windows_width = width // window_size
            expected_windows = self.batch_size * windows_height * windows_width
            expected_window_elements = window_size**2
            expected_shape = (expected_windows, expected_window_elements, self.channels)

            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_input_dimensions(self):
        layer = WindowPartition(window_size=self.window_size)

        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.height, self.width, self.channels))
            outputs = layer(inputs, height=self.height, width=self.width)

            expected_windows = batch_size * self.windows_height * self.windows_width
            expected_shape = (expected_windows, self.window_size**2, self.channels)

            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

        test_channels = [32, 64, 128]
        for channels in test_channels:
            inputs = ops.ones((self.batch_size, self.height, self.width, channels))
            outputs = layer(inputs, height=self.height, width=self.width)

            expected_windows = (
                self.batch_size * self.windows_height * self.windows_width
            )
            expected_shape = (expected_windows, self.window_size**2, channels)

            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_fused_with_different_params(self):
        test_num_heads = [1, 4, 8]
        for num_heads in test_num_heads:
            channels_per_head = 32
            full_channels = self.qkv_mult * num_heads * channels_per_head
            inputs = ops.ones((self.batch_size, self.height, self.width, full_channels))

            layer = WindowPartition(
                window_size=self.window_size, fused=True, num_heads=num_heads
            )
            outputs = layer(inputs, height=self.height, width=self.width)

            expected_windows = (
                self.batch_size * self.windows_height * self.windows_width
            )
            expected_shape = (
                self.qkv_mult,
                expected_windows,
                num_heads,
                self.window_size**2,
                channels_per_head,
            )

            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

        test_qkv_mult = [1, 2, 4]
        for qkv_mult in test_qkv_mult:
            num_heads = 4
            channels_per_head = 32
            full_channels = qkv_mult * num_heads * channels_per_head
            inputs = ops.ones((self.batch_size, self.height, self.width, full_channels))

            layer = WindowPartition(
                window_size=self.window_size,
                fused=True,
                num_heads=num_heads,
                qkv_mult=qkv_mult,
            )
            outputs = layer(inputs, height=self.height, width=self.width)

            expected_windows = (
                self.batch_size * self.windows_height * self.windows_width
            )
            expected_shape = (
                qkv_mult,
                expected_windows,
                num_heads,
                self.window_size**2,
                channels_per_head,
            )

            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_get_config(self):
        layer = WindowPartition(window_size=self.window_size)
        config = layer.get_config()

        assert "window_size" in config
        assert config["window_size"] == self.window_size
        assert "fused" in config
        assert not config["fused"]
        assert "num_heads" in config
        assert config["num_heads"] is None
        assert "qkv_mult" in config
        assert config["qkv_mult"] == 3

        reconstructed_layer = WindowPartition.from_config(config)
        assert reconstructed_layer.window_size == layer.window_size
        assert reconstructed_layer.fused == layer.fused
        assert reconstructed_layer.num_heads == layer.num_heads
        assert reconstructed_layer.qkv_mult == layer.qkv_mult

        layer = WindowPartition(
            window_size=self.window_size,
            fused=True,
            num_heads=self.num_heads,
            qkv_mult=2,
        )
        config = layer.get_config()

        assert config["window_size"] == self.window_size
        assert config["fused"]
        assert config["num_heads"] == self.num_heads
        assert config["qkv_mult"] == 2

        reconstructed_layer = WindowPartition.from_config(config)
        assert reconstructed_layer.window_size == layer.window_size
        assert reconstructed_layer.fused == layer.fused
        assert reconstructed_layer.num_heads == layer.num_heads
        assert reconstructed_layer.qkv_mult == layer.qkv_mult
