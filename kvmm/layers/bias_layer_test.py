import keras
import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .bias_layer import BiasLayer


class TestBiasLayer(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.seq_length = 196
        self.channels = 768
        self.input_shape_channels_last = (
            self.batch_size,
            self.seq_length,
            self.channels,
        )
        self.input_shape_channels_first = (
            self.batch_size,
            self.channels,
            self.seq_length,
        )
        self.test_inputs_channels_last = ops.ones(self.input_shape_channels_last)
        self.test_inputs_channels_first = ops.ones(self.input_shape_channels_first)

    def test_init(self):
        layer = BiasLayer()
        assert layer.trainable_bias == True
        assert not layer.built
        assert isinstance(layer.initializer, keras.initializers.Zeros)

        custom_layer = BiasLayer(
            trainable=False, initializer="ones", data_format="channels_first"
        )
        assert custom_layer.trainable_bias == False
        assert isinstance(custom_layer.initializer, keras.initializers.Ones)
        assert custom_layer.data_format == "channels_first"

    def test_build_channels_last(self):
        layer = BiasLayer(data_format="channels_last")
        layer.build(self.input_shape_channels_last)
        assert hasattr(layer, "bias")
        assert layer.bias.shape == (self.channels,)
        expected_values = np.zeros((self.channels,))
        assert np.allclose(layer.bias.numpy(), expected_values)

    def test_build_channels_first(self):
        layer = BiasLayer(data_format="channels_first")
        layer.build(self.input_shape_channels_first)
        assert hasattr(layer, "bias")
        assert layer.bias.shape == (self.channels,)
        expected_values = np.zeros((self.channels,))
        assert np.allclose(layer.bias.numpy(), expected_values)

    def test_call_channels_last(self):
        layer = BiasLayer(data_format="channels_last", initializer="ones")
        outputs = layer(self.test_inputs_channels_last)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_shape_channels_last)
        assert all(
            output_shape[i] == self.input_shape_channels_last[i]
            for i in range(len(self.input_shape_channels_last))
        )
        expected_output = ops.ones_like(self.test_inputs_channels_last) * 2
        assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_call_channels_first(self):
        layer = BiasLayer(data_format="channels_first", initializer="ones")
        outputs = layer(self.test_inputs_channels_first)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_shape_channels_first)
        assert all(
            output_shape[i] == self.input_shape_channels_first[i]
            for i in range(len(self.input_shape_channels_first))
        )
        expected_output = ops.ones_like(self.test_inputs_channels_first) * 2
        assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_get_config(self):
        layer = BiasLayer(
            trainable=False, initializer="ones", data_format="channels_first"
        )
        config = layer.get_config()
        assert "trainable" in config
        assert "initializer" in config
        assert "data_format" in config
        assert config["trainable"] == False
        assert config["data_format"] == "channels_first"

        reconstructed_layer = BiasLayer.from_config(config)
        assert reconstructed_layer.trainable_bias == layer.trainable_bias
        assert reconstructed_layer.data_format == layer.data_format
        assert isinstance(reconstructed_layer.initializer, keras.initializers.Ones)

    def test_different_batch_sizes(self):
        for data_format in ["channels_last", "channels_first"]:
            layer = BiasLayer(data_format=data_format)
            test_batch_sizes = [1, 8, 16]
            for batch_size in test_batch_sizes:
                if data_format == "channels_last":
                    inputs = ops.ones((batch_size, self.seq_length, self.channels))
                    expected_shape = (batch_size, self.seq_length, self.channels)
                else:
                    inputs = ops.ones((batch_size, self.channels, self.seq_length))
                    expected_shape = (batch_size, self.channels, self.seq_length)

                outputs = layer(inputs)
                output_shape = ops.shape(outputs)

                assert all(
                    output_shape[i] == expected_shape[i]
                    for i in range(len(expected_shape))
                )

    def test_different_sequence_lengths(self):
        for data_format in ["channels_last", "channels_first"]:
            layer = BiasLayer(data_format=data_format)
            test_seq_lengths = [64, 128, 256]
            for seq_length in test_seq_lengths:
                if data_format == "channels_last":
                    inputs = ops.ones((self.batch_size, seq_length, self.channels))
                    expected_shape = (self.batch_size, seq_length, self.channels)
                else:
                    inputs = ops.ones((self.batch_size, self.channels, seq_length))
                    expected_shape = (self.batch_size, self.channels, seq_length)

                outputs = layer(inputs)
                output_shape = ops.shape(outputs)

                assert all(
                    output_shape[i] == expected_shape[i]
                    for i in range(len(expected_shape))
                )

    def test_different_channel_dims(self):
        test_channel_dims = [256, 512, 1024]
        for data_format in ["channels_last", "channels_first"]:
            for channels in test_channel_dims:
                layer = BiasLayer(data_format=data_format)

                if data_format == "channels_last":
                    inputs = ops.ones((self.batch_size, self.seq_length, channels))
                    expected_shape = (self.batch_size, self.seq_length, channels)
                else:
                    inputs = ops.ones((self.batch_size, channels, self.seq_length))
                    expected_shape = (self.batch_size, channels, self.seq_length)

                outputs = layer(inputs)
                output_shape = ops.shape(outputs)

                assert all(
                    output_shape[i] == expected_shape[i]
                    for i in range(len(expected_shape))
                )

                assert layer.bias.shape == (channels,)

    def test_different_initializers(self):
        test_initializers = ["zeros", "ones", "random_normal", "constant"]

        for initializer in test_initializers:
            if initializer == "constant":
                layer = BiasLayer(initializer=keras.initializers.Constant(0.5))
                expected_value = 0.5
            else:
                layer = BiasLayer(initializer=initializer)
                if initializer == "zeros":
                    expected_value = 0.0
                elif initializer == "ones":
                    expected_value = 1.0
                else:
                    expected_value = None

            layer.build(self.input_shape_channels_last)

            if expected_value is not None:
                if initializer == "zeros":
                    assert np.allclose(layer.bias.numpy(), np.zeros((self.channels,)))
                elif initializer == "ones":
                    assert np.allclose(layer.bias.numpy(), np.ones((self.channels,)))
                elif initializer == "constant":
                    assert np.allclose(
                        layer.bias.numpy(), np.full((self.channels,), expected_value)
                    )

    def test_trainable_bias(self):
        layer = BiasLayer(trainable=True)
        layer.build(self.input_shape_channels_last)
        assert layer.bias.trainable

        initial_bias = layer.bias.numpy().copy()
        new_values = initial_bias + 0.01
        layer.bias.assign(new_values)
        assert np.allclose(layer.bias.numpy(), new_values)
        assert not np.allclose(layer.bias.numpy(), initial_bias)

        layer_non_trainable = BiasLayer(trainable=False)
        layer_non_trainable.build(self.input_shape_channels_last)
        assert not layer_non_trainable.bias.trainable

    def test_higher_dimensions(self):
        height, width = 28, 28
        channels = 32

        for data_format in ["channels_last", "channels_first"]:
            layer = BiasLayer(data_format=data_format)

            if data_format == "channels_last":
                inputs_4d = ops.ones((self.batch_size, height, width, channels))
                expected_shape = (self.batch_size, height, width, channels)
            else:
                inputs_4d = ops.ones((self.batch_size, channels, height, width))
                expected_shape = (self.batch_size, channels, height, width)

            outputs = layer(inputs_4d)
            output_shape = ops.shape(outputs)

            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

            assert layer.bias.shape == (channels,)
