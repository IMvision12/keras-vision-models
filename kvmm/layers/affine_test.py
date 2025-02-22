import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .affine import Affine


class TestAffine(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.seq_length = 196
        self.projection_dim = 768
        self.input_shape = (self.batch_size, self.seq_length, self.projection_dim)
        self.test_inputs = ops.ones(self.input_shape)

    def test_init(self):
        layer = Affine()
        assert not layer.built
        assert layer.dim is None
        assert layer.alpha is None
        assert layer.beta is None

        layer_with_dim = Affine(dim=512)
        assert layer_with_dim.dim == 512

    def test_build(self):
        layer = Affine()
        layer.build(self.input_shape)
        assert layer.dim == self.projection_dim
        assert hasattr(layer, "alpha")
        assert hasattr(layer, "beta")
        
        expected_shape = (1, 1, self.projection_dim)
        assert layer.alpha.shape == expected_shape
        assert layer.beta.shape == expected_shape
        
        assert np.allclose(layer.alpha.numpy(), np.ones(expected_shape))
        assert np.allclose(layer.beta.numpy(), np.zeros(expected_shape))

    def test_call(self):
        layer = Affine()
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)
        
        assert len(output_shape) == len(self.input_shape)
        assert all(
            output_shape[i] == self.input_shape[i] for i in range(len(self.input_shape))
        )
        
        expected_output = self.test_inputs * layer.alpha + layer.beta
        assert np.allclose(outputs.numpy(), expected_output.numpy())

    def test_get_config(self):
        layer = Affine(dim=self.projection_dim)
        config = layer.get_config()
        assert "dim" in config
        assert config["dim"] == self.projection_dim
        reconstructed_layer = Affine.from_config(config)
        assert reconstructed_layer.dim == layer.dim

        layer = Affine()
        layer.build(self.input_shape)
        config = layer.get_config()
        assert "dim" in config
        assert config["dim"] == self.projection_dim
        reconstructed_layer = Affine.from_config(config)
        assert reconstructed_layer.dim == layer.dim

    def test_different_batch_sizes(self):
        layer = Affine()
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
        layer = Affine()
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
            layer = Affine(dim=projection_dim)
            inputs = ops.ones((self.batch_size, self.seq_length, projection_dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, self.seq_length, projection_dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_trainable_parameters(self):
        layer = Affine()
        layer.build(self.input_shape)
        
        assert layer.alpha.trainable
        initial_alpha = layer.alpha.numpy().copy()
        new_alpha_values = initial_alpha + 0.5
        layer.alpha.assign(new_alpha_values)
        assert np.allclose(layer.alpha.numpy(), new_alpha_values)
        assert not np.allclose(layer.alpha.numpy(), initial_alpha)
        
        assert layer.beta.trainable
        initial_beta = layer.beta.numpy().copy()
        new_beta_values = initial_beta + 0.3
        layer.beta.assign(new_beta_values)
        assert np.allclose(layer.beta.numpy(), new_beta_values)
        assert not np.allclose(layer.beta.numpy(), initial_beta)

    def test_non_standard_input_dims(self):
        input_2d = ops.ones((self.batch_size, self.projection_dim))
        layer = Affine()
        layer.build((self.batch_size, self.projection_dim))
        output_2d = layer(input_2d)
        assert ops.shape(output_2d) == (self.batch_size, self.projection_dim)
        
        input_4d = ops.ones((self.batch_size, 32, 32, self.projection_dim))
        layer = Affine()
        layer.build((self.batch_size, 32, 32, self.projection_dim))
        output_4d = layer(input_4d)
        assert ops.shape(output_4d) == (self.batch_size, 32, 32, self.projection_dim)