import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .multi_head_self_attention import MultiHeadSelfAttention


class TestMultiHeadSelfAttention(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.seq_length = 16
        self.dim = 64
        self.num_heads = 8
        self.head_dim = self.dim // self.num_heads
        self.input_shape = (self.batch_size, self.seq_length, self.dim)
        self.test_inputs = ops.ones(self.input_shape)

    def test_init_default(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        assert layer.dim == self.dim
        assert layer.num_heads == 8
        assert layer.head_dim == self.dim // 8
        assert layer.scale == (self.dim // 8) ** -0.5
        assert layer.block_idx == 0
        assert not layer.built
        assert layer.q_norm is None
        assert layer.k_norm is None

    def test_init_with_options(self):
        layer = MultiHeadSelfAttention(
            dim=self.dim,
            num_heads=4,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.1,
            proj_drop=0.1,
            block_idx=1,
        )
        assert layer.dim == self.dim
        assert layer.num_heads == 4
        assert layer.head_dim == self.dim // 4
        assert layer.block_idx == 1
        assert layer.q_norm is not None
        assert layer.k_norm is not None

    def test_invalid_dim(self):
        with self.assertRaises(AssertionError):
            MultiHeadSelfAttention(dim=65, num_heads=8)

    def test_build(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        layer.build(self.input_shape)
        assert hasattr(layer, "qkv")
        assert hasattr(layer, "proj")
        assert layer.qkv.kernel.shape == (self.dim, self.dim * 3)
        assert layer.proj.kernel.shape == (self.dim, self.dim)

    def test_call(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_shape)
        assert all(
            output_shape[i] == self.input_shape[i] for i in range(len(self.input_shape))
        )

    def test_attention_mask(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        x = ops.ones(self.input_shape)
        outputs = layer(x)
        assert ops.shape(outputs) == self.input_shape

    def test_training_vs_inference(self):
        layer = MultiHeadSelfAttention(dim=self.dim, attn_drop=0.5, proj_drop=0.5)
        train_output = layer(self.test_inputs, training=True)
        infer_output = layer(self.test_inputs, training=False)
        assert ops.shape(train_output) == ops.shape(infer_output)
        assert not np.allclose(train_output.numpy(), infer_output.numpy())

    def test_qk_norm(self):
        layer = MultiHeadSelfAttention(dim=self.dim, qk_norm=True)
        outputs = layer(self.test_inputs)
        assert ops.shape(outputs) == self.input_shape
        assert layer.q_norm is not None
        assert layer.k_norm is not None

    def test_get_config(self):
        layer = MultiHeadSelfAttention(dim=self.dim, num_heads=4, block_idx=1)
        config = layer.get_config()
        assert "dim" in config
        assert "num_heads" in config
        assert "block_idx" in config
        assert config["dim"] == self.dim
        assert config["num_heads"] == 4
        assert config["block_idx"] == 1
        reconstructed_layer = MultiHeadSelfAttention.from_config(config)
        assert reconstructed_layer.dim == layer.dim
        assert reconstructed_layer.num_heads == layer.num_heads
        assert reconstructed_layer.block_idx == layer.block_idx

    def test_different_batch_sizes(self):
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

    def test_different_sequence_lengths(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        test_seq_lengths = [8, 32, 64]
        for seq_length in test_seq_lengths:
            inputs = ops.ones((self.batch_size, seq_length, self.dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, seq_length, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_numerical_stability(self):
        layer = MultiHeadSelfAttention(dim=self.dim)
        small_inputs = self.test_inputs * 1e-10
        small_outputs = layer(small_inputs)
        assert not np.any(np.isnan(small_outputs.numpy()))
        assert not np.any(np.isinf(small_outputs.numpy()))
        large_inputs = self.test_inputs * 1e10
        large_outputs = layer(large_inputs)
        assert not np.any(np.isnan(large_outputs.numpy()))
        assert not np.any(np.isinf(large_outputs.numpy()))

    def test_attention_computation(self):
        layer = MultiHeadSelfAttention(dim=self.dim, num_heads=self.num_heads)
        x = ops.eye(self.seq_length)
        x = ops.expand_dims(x, axis=0)
        x = ops.repeat(x, self.dim // self.seq_length, axis=-1)
        x = ops.repeat(x, self.batch_size, axis=0)
        outputs = layer(x)
        assert ops.shape(outputs) == (self.batch_size, self.seq_length, self.dim)
