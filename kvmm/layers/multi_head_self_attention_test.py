import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .multi_head_self_attention import MultiHeadSelfAttention


class TestMultiHeadSelfAttention(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.seq_length = 16
        self.height = 8
        self.width = 8
        self.dim = 64
        self.num_heads = 8
        self.head_dim = self.dim // self.num_heads
        self.input_3d_shape = (self.batch_size, self.seq_length, self.dim)
        self.input_4d_shape = (self.batch_size, self.height, self.width, self.dim)
        self.test_3d_inputs = ops.ones(self.input_3d_shape)
        self.test_4d_inputs = ops.ones(self.input_4d_shape)
        self.default_epsilon = 1e-6

    def test_init_default(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        assert layer.dim == self.dim
        assert layer.num_heads == 8
        assert layer.head_dim == self.dim // 8
        assert layer.scale == (self.dim // 8) ** -0.5
        assert layer.block_prefix == "blocks"
        assert not layer.built
        assert layer.q_norm is None
        assert layer.k_norm is None
        assert layer.epsilon == self.default_epsilon

    def test_init_with_options(self):
        custom_epsilon = 1e-5
        layer = MultiHeadSelfAttention(
            dim=self.dim,
            num_heads=4,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.1,
            proj_drop=0.1,
            block_prefix="custom_block",
            epsilon=custom_epsilon,
        )
        assert layer.dim == self.dim
        assert layer.num_heads == 4
        assert layer.head_dim == self.dim // 4
        assert layer.block_prefix == "custom_block"
        assert layer.q_norm is not None
        assert layer.k_norm is not None
        assert layer.epsilon == custom_epsilon

        assert layer.q_norm.epsilon == custom_epsilon
        assert layer.k_norm.epsilon == custom_epsilon

    def test_invalid_dim(self):
        with self.assertRaises(AssertionError):
            MultiHeadSelfAttention(dim=65, num_heads=8)

    def test_invalid_input_dims(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        with self.assertRaises(ValueError):
            layer.build(
                (self.batch_size, self.seq_length, self.height, self.width, self.dim)
            )

    def test_build_3d(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        layer.build(self.input_3d_shape)
        assert hasattr(layer, "qkv")
        assert hasattr(layer, "proj")
        assert layer.qkv.kernel.shape == (self.dim, self.dim * 3)
        assert layer.proj.kernel.shape == (self.dim, self.dim)

    def test_build_4d(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        layer.build(self.input_4d_shape)
        assert hasattr(layer, "qkv")
        assert hasattr(layer, "proj")
        assert layer.qkv.kernel.shape == (self.dim, self.dim * 3)
        assert layer.proj.kernel.shape == (self.dim, self.dim)

    def test_call_3d(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        outputs = layer(self.test_3d_inputs)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_3d_shape)
        assert all(
            output_shape[i] == self.input_3d_shape[i]
            for i in range(len(self.input_3d_shape))
        )

    def test_call_4d(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        outputs = layer(self.test_4d_inputs)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_4d_shape)
        assert all(
            output_shape[i] == self.input_4d_shape[i]
            for i in range(len(self.input_4d_shape))
        )

    def test_training_vs_inference_3d(self):
        layer = MultiHeadSelfAttention(dim=self.dim, attn_drop=0.5, proj_drop=0.5)
        train_output = layer(self.test_3d_inputs, training=True)
        infer_output = layer(self.test_3d_inputs, training=False)
        assert ops.shape(train_output) == ops.shape(infer_output)
        assert not np.allclose(train_output.numpy(), infer_output.numpy())

    def test_training_vs_inference_4d(self):
        layer = MultiHeadSelfAttention(dim=self.dim, attn_drop=0.5, proj_drop=0.5)
        train_output = layer(self.test_4d_inputs, training=True)
        infer_output = layer(self.test_4d_inputs, training=False)
        assert ops.shape(train_output) == ops.shape(infer_output)
        assert not np.allclose(train_output.numpy(), infer_output.numpy())

    def test_qk_norm_3d(self):
        custom_epsilon = 1e-5
        layer = MultiHeadSelfAttention(
            dim=self.dim, qk_norm=True, epsilon=custom_epsilon
        )
        outputs = layer(self.test_3d_inputs)
        assert ops.shape(outputs) == self.input_3d_shape
        assert layer.q_norm is not None
        assert layer.k_norm is not None
        assert layer.q_norm.epsilon == custom_epsilon
        assert layer.k_norm.epsilon == custom_epsilon

    def test_qk_norm_4d(self):
        custom_epsilon = 1e-5
        layer = MultiHeadSelfAttention(
            dim=self.dim, qk_norm=True, epsilon=custom_epsilon
        )
        outputs = layer(self.test_4d_inputs)
        assert ops.shape(outputs) == self.input_4d_shape
        assert layer.q_norm is not None
        assert layer.k_norm is not None
        assert layer.q_norm.epsilon == custom_epsilon
        assert layer.k_norm.epsilon == custom_epsilon

    def test_get_config(self):
        custom_epsilon = 1e-5
        layer = MultiHeadSelfAttention(
            dim=self.dim,
            num_heads=4,
            block_prefix="custom_block",
            epsilon=custom_epsilon,
        )
        config = layer.get_config()
        assert "dim" in config
        assert "num_heads" in config
        assert "block_prefix" in config
        assert "epsilon" in config
        assert config["dim"] == self.dim
        assert config["num_heads"] == 4
        assert config["block_prefix"] == "custom_block"
        assert config["epsilon"] == custom_epsilon
        reconstructed_layer = MultiHeadSelfAttention.from_config(config)
        assert reconstructed_layer.dim == layer.dim
        assert reconstructed_layer.num_heads == layer.num_heads
        assert reconstructed_layer.block_prefix == layer.block_prefix
        assert reconstructed_layer.epsilon == layer.epsilon

    def test_different_batch_sizes_3d(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.seq_length, self.dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.seq_length, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_batch_sizes_4d(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
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
        layer = MultiHeadSelfAttention(dim=self.dim)
        test_sizes = [(8, 8), (16, 16), (32, 32)]
        for h, w in test_sizes:
            inputs = ops.ones((self.batch_size, h, w, self.dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, h, w, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_numerical_stability_3d(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        small_inputs = self.test_3d_inputs * 1e-10
        small_outputs = layer(small_inputs)
        assert not np.any(np.isnan(small_outputs.numpy()))
        assert not np.any(np.isinf(small_outputs.numpy()))
        large_inputs = self.test_3d_inputs * 1e10
        large_outputs = layer(large_inputs)
        assert not np.any(np.isnan(large_outputs.numpy()))
        assert not np.any(np.isinf(large_outputs.numpy()))

    def test_numerical_stability_4d(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        small_inputs = self.test_4d_inputs * 1e-10
        small_outputs = layer(small_inputs)
        assert not np.any(np.isnan(small_outputs.numpy()))
        assert not np.any(np.isinf(small_outputs.numpy()))
        large_inputs = self.test_4d_inputs * 1e10
        large_outputs = layer(large_inputs)
        assert not np.any(np.isnan(large_outputs.numpy()))
        assert not np.any(np.isinf(large_outputs.numpy()))

    def test_attention_computation_3d(self):
        layer = MultiHeadSelfAttention(dim=self.dim, num_heads=self.num_heads)
        x = ops.eye(self.seq_length)
        x = ops.expand_dims(x, axis=0)
        x = ops.repeat(x, self.dim // self.seq_length, axis=-1)
        x = ops.repeat(x, self.batch_size, axis=0)
        outputs = layer(x)
        assert ops.shape(outputs) == (self.batch_size, self.seq_length, self.dim)

    def test_attention_computation_4d(self):
        layer = MultiHeadSelfAttention(dim=self.dim, num_heads=self.num_heads)
        x = ops.eye(self.height * self.width)
        x = ops.reshape(x, (self.height, self.width, self.height * self.width))
        x = ops.expand_dims(x, axis=0)
        x = ops.repeat(x, self.dim // (self.height * self.width), axis=-1)
        x = ops.repeat(x, self.batch_size, axis=0)
        outputs = layer(x)
        assert ops.shape(outputs) == (
            self.batch_size,
            self.height,
            self.width,
            self.dim,
        )

    def test_epsilon_layer_norm_stability(self):
        test_epsilon_values = [1e-12, 1e-6, 1e-3]
        for epsilon in test_epsilon_values:
            layer_3d = MultiHeadSelfAttention(
                dim=self.dim, qk_norm=True, epsilon=epsilon
            )
            layer_4d = MultiHeadSelfAttention(
                dim=self.dim, qk_norm=True, epsilon=epsilon
            )

            outputs_3d = layer_3d(self.test_3d_inputs)
            assert not np.any(np.isnan(outputs_3d.numpy()))
            assert not np.any(np.isinf(outputs_3d.numpy()))

            outputs_4d = layer_4d(self.test_4d_inputs)
            assert not np.any(np.isnan(outputs_4d.numpy()))
            assert not np.any(np.isinf(outputs_4d.numpy()))
