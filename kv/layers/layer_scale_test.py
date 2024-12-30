import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .layer_scale import LayerScale


class TestLayerScale(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.seq_length = 196
        self.projection_dim = 768
        self.init_values = 0.1
        self.input_shape = (self.batch_size, self.seq_length, self.projection_dim)
        self.test_inputs = ops.ones(self.input_shape)

    def test_init(self):
        layer = LayerScale(init_values=self.init_values)
        assert layer.init_values == self.init_values
        assert not layer.built

    def test_build(self):
        layer = LayerScale(init_values=self.init_values)
        layer.build(self.input_shape)
        assert hasattr(layer, "gamma")
        assert layer.gamma.shape == (self.projection_dim,)
        expected_values = np.full((self.projection_dim,), self.init_values)
        assert np.allclose(layer.gamma.numpy(), expected_values)

    def test_call(self):
        layer = LayerScale(init_values=self.init_values)
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_shape)
        assert all(
            output_shape[i] == self.input_shape[i] for i in range(len(self.input_shape))
        )
        expected_output = self.test_inputs * layer.gamma
        assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_get_config(self):
        layer = LayerScale(init_values=self.init_values)
        config = layer.get_config()
        assert "init_values" in config
        assert config["init_values"] == self.init_values
        reconstructed_layer = LayerScale.from_config(config)
        assert reconstructed_layer.init_values == layer.init_values

    def test_different_batch_sizes(self):
        layer = LayerScale(init_values=self.init_values)
        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.seq_length, self.projection_dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.seq_length, self.projection_dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_sequence_lengths(self):
        layer = LayerScale(init_values=self.init_values)
        test_seq_lengths = [64, 128, 256]
        for seq_length in test_seq_lengths:
            inputs = ops.ones((self.batch_size, seq_length, self.projection_dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, seq_length, self.projection_dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_projection_dims(self):
        test_projection_dims = [256, 512, 1024]
        for projection_dim in test_projection_dims:
            layer = LayerScale(init_values=self.init_values)
            inputs = ops.ones((self.batch_size, self.seq_length, projection_dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, self.seq_length, projection_dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_init_values(self):
        test_init_values = [0.001, 1.0, 10.0]
        for init_value in test_init_values:
            layer = LayerScale(init_values=init_value)
            layer.build((self.batch_size, self.seq_length, self.projection_dim))
            expected_values = np.full((self.projection_dim,), init_value)
            assert np.allclose(layer.gamma.numpy(), expected_values)

    def test_trainable_gamma(self):
        layer = LayerScale(init_values=self.init_values)
        layer.build(self.input_shape)
        assert layer.gamma.trainable
        initial_gamma = layer.gamma.numpy().copy()
        new_values = initial_gamma + 0.01
        layer.gamma.assign(new_values)
        assert np.allclose(layer.gamma.numpy(), new_values)
        assert not np.allclose(layer.gamma.numpy(), initial_gamma)
