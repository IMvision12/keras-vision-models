import numpy as np
from keras import ops
from keras.src.testing import TestCase

from kvmm.layers import GlobalResponseNorm


class TestGlobalResponseNorm(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.height = 32
        self.width = 32
        self.dim = 64
        self.input_shape = (self.batch_size, self.height, self.width, self.dim)
        self.test_inputs = ops.ones(self.input_shape)

    def test_init(self):
        layer = GlobalResponseNorm()
        assert not layer.built

    def test_build(self):
        layer = GlobalResponseNorm()
        layer.build(self.input_shape)
        assert hasattr(layer, "weight")
        assert hasattr(layer, "bias")
        assert layer.weight.shape == (1, 1, 1, self.dim)
        assert layer.bias.shape == (1, 1, 1, self.dim)
        assert np.allclose(layer.weight.numpy(), np.zeros((1, 1, 1, self.dim)))
        assert np.allclose(layer.bias.numpy(), np.zeros((1, 1, 1, self.dim)))

    def test_call(self):
        layer = GlobalResponseNorm()
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_shape)
        assert all(
            output_shape[i] == self.input_shape[i] for i in range(len(self.input_shape))
        )
        global_features = ops.sqrt(
            ops.sum(ops.square(self.test_inputs), axis=(1, 2), keepdims=True)
        )
        norm_features = global_features / (
            ops.mean(global_features, axis=-1, keepdims=True) + 1e-6
        )
        expected_output = (
            layer.weight * (self.test_inputs * norm_features)
            + layer.bias
            + self.test_inputs
        )
        assert np.allclose(outputs.numpy(), expected_output.numpy(), rtol=1e-5)

    def test_different_batch_sizes(self):
        layer = GlobalResponseNorm()
        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.height, self.width, self.dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.height, self.width, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_spatial_dimensions(self):
        layer = GlobalResponseNorm()
        test_sizes = [(16, 16), (64, 64), (128, 128)]
        for height, width in test_sizes:
            inputs = ops.ones((self.batch_size, height, width, self.dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, height, width, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_channels(self):
        test_dims = [32, 128, 256]
        for dim in test_dims:
            layer = GlobalResponseNorm()
            inputs = ops.ones((self.batch_size, self.height, self.width, dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, self.height, self.width, dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_numerical_stability(self):
        layer = GlobalResponseNorm()
        small_inputs = self.test_inputs * 1e-10
        small_outputs = layer(small_inputs)
        assert not np.any(np.isnan(small_outputs.numpy()))
        assert not np.any(np.isinf(small_outputs.numpy()))
        large_inputs = self.test_inputs * 1e10
        large_outputs = layer(large_inputs)
        assert not np.any(np.isnan(large_outputs.numpy()))
        assert not np.any(np.isinf(large_outputs.numpy()))
