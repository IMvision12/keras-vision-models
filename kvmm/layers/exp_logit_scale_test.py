import math

import keras
import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .exp_logit_scale import ExpLogitScale


class TestExpLogitScale(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.num_heads = 8
        self.seq_length = 64
        self.input_shape = (
            self.batch_size,
            self.num_heads,
            self.seq_length,
            self.seq_length,
        )
        self.test_inputs = ops.ones(self.input_shape)
        self.default_init_value = math.log(10.0)
        self.default_max_value = math.log(100.0)

    def test_init(self):
        layer = ExpLogitScale()
        assert layer.init_value == self.default_init_value
        assert layer.max_value == self.default_max_value
        assert not layer.built

        custom_init = math.log(5.0)
        custom_max = math.log(50.0)
        custom_layer = ExpLogitScale(init_value=custom_init, max_value=custom_max)
        assert custom_layer.init_value == custom_init
        assert custom_layer.max_value == custom_max

    def test_build(self):
        layer = ExpLogitScale()
        layer.build(self.input_shape)
        assert hasattr(layer, "scale")
        assert layer.scale.shape == (self.num_heads,)
        expected_values = np.full((self.num_heads,), self.default_init_value)
        assert np.allclose(layer.scale.numpy(), expected_values)

    def test_call(self):
        layer = ExpLogitScale()
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)

        assert len(output_shape) == len(self.input_shape)
        assert all(
            output_shape[i] == self.input_shape[i] for i in range(len(self.input_shape))
        )

        expected_scale = math.exp(self.default_init_value)
        expected_output = ops.ones_like(self.test_inputs) * expected_scale
        assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_max_value_capping(self):
        layer = ExpLogitScale(max_value=math.log(20.0))
        layer.build(self.input_shape)

        above_max = math.log(30.0)
        layer.scale.assign(np.full((self.num_heads,), above_max))

        outputs = layer(self.test_inputs)

        expected_scale = math.exp(math.log(20.0))
        expected_output = ops.ones_like(self.test_inputs) * expected_scale
        assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_get_config(self):
        custom_init = math.log(5.0)
        custom_max = math.log(50.0)
        layer = ExpLogitScale(init_value=custom_init, max_value=custom_max)

        config = layer.get_config()
        assert "init_value" in config
        assert "max_value" in config
        assert config["init_value"] == custom_init
        assert config["max_value"] == custom_max

        reconstructed_layer = ExpLogitScale.from_config(config)
        assert reconstructed_layer.init_value == layer.init_value
        assert reconstructed_layer.max_value == layer.max_value

    def test_different_batch_sizes(self):
        layer = ExpLogitScale()
        test_batch_sizes = [1, 4, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones(
                (batch_size, self.num_heads, self.seq_length, self.seq_length)
            )
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (
                batch_size,
                self.num_heads,
                self.seq_length,
                self.seq_length,
            )
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_num_heads(self):
        test_head_counts = [2, 4, 16]
        for num_heads in test_head_counts:
            layer = ExpLogitScale()
            inputs = ops.ones(
                (self.batch_size, num_heads, self.seq_length, self.seq_length)
            )
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (
                self.batch_size,
                num_heads,
                self.seq_length,
                self.seq_length,
            )
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

            assert layer.scale.shape == (num_heads,)

    def test_different_sequence_lengths(self):
        layer = ExpLogitScale()
        test_seq_lengths = [32, 128, 256]
        for seq_length in test_seq_lengths:
            inputs = ops.ones((self.batch_size, self.num_heads, seq_length, seq_length))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, self.num_heads, seq_length, seq_length)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_init_values(self):
        test_init_values = [math.log(2.0), math.log(20.0), math.log(50.0)]
        for init_value in test_init_values:
            layer = ExpLogitScale(init_value=init_value)
            layer.build(self.input_shape)
            expected_values = np.full((self.num_heads,), init_value)
            assert np.allclose(layer.scale.numpy(), expected_values)

            outputs = layer(self.test_inputs)
            expected_scale = math.exp(init_value)
            expected_output = ops.ones_like(self.test_inputs) * expected_scale
            assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_trainable_scale(self):
        layer = ExpLogitScale()
        layer.build(self.input_shape)
        assert layer.scale.trainable

        initial_scale = layer.scale.numpy().copy()
        new_values = initial_scale + 0.5
        layer.scale.assign(new_values)
        assert np.allclose(layer.scale.numpy(), new_values)
        assert not np.allclose(layer.scale.numpy(), initial_scale)

        outputs = layer(self.test_inputs)
        expected_scale = np.exp(new_values).reshape((1, self.num_heads, 1, 1))
        expected_output = ops.ones_like(self.test_inputs) * expected_scale
        assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_non_uniform_scales(self):
        layer = ExpLogitScale()
        layer.build(self.input_shape)

        scales = np.array([math.log(i + 1) for i in range(self.num_heads)])
        layer.scale.assign(scales)

        outputs = layer(self.test_inputs)

        for head_idx in range(self.num_heads):
            head_output = outputs[:, head_idx, :, :]
            expected_scale = math.exp(scales[head_idx])
            expected_head_output = ops.ones_like(head_output) * expected_scale
            assert np.allclose(head_output.numpy(), expected_head_output.numpy())

    def test_different_input_values(self):
        layer = ExpLogitScale()

        varied_inputs = keras.random.uniform(
            shape=self.input_shape, minval=-2.0, maxval=2.0
        )

        outputs = layer(varied_inputs)

        expected_scale = math.exp(self.default_init_value)
        expected_output = varied_inputs * expected_scale
        assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_zero_inputs(self):
        layer = ExpLogitScale()

        zero_inputs = ops.zeros(self.input_shape)

        outputs = layer(zero_inputs)

        expected_output = ops.zeros_like(zero_inputs)
        assert np.allclose(outputs.numpy(), expected_output.numpy())
