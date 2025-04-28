import keras
from keras import ops
from keras.src.testing import TestCase

from kvmm.layers import TalkingHeadAttention


class TestTalkingHeadAttention(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.seq_length = 16
        self.dim = 64
        self.num_heads = 8
        self.head_dim = self.dim // self.num_heads
        self.input_shape_channels_last = (self.batch_size, self.seq_length, self.dim)
        self.input_shape_channels_first = (self.batch_size, self.dim, self.seq_length)
        self.test_inputs_channels_last = ops.ones(self.input_shape_channels_last)
        self.test_inputs_channels_first = ops.ones(self.input_shape_channels_first)

    def test_init_default(self):
        layer = TalkingHeadAttention(dim=self.dim, num_heads=self.num_heads)
        assert layer.dim == self.dim
        assert layer.num_heads == self.num_heads
        assert layer.head_dim == self.dim // self.num_heads
        assert layer.scale == (self.dim // self.num_heads) ** -0.5
        assert layer.block_prefix is None
        assert layer.data_format == "channels_last"
        assert not layer.built

    def test_init_with_options(self):
        layer = TalkingHeadAttention(
            dim=self.dim,
            num_heads=4,
            qkv_bias=False,
            attn_drop=0.1,
            proj_drop=0.1,
            data_format="channels_first",
            block_prefix="custom_block",
        )
        assert layer.dim == self.dim
        assert layer.num_heads == 4
        assert layer.head_dim == self.dim // 4
        assert layer.block_prefix == "custom_block"
        assert layer.attn_drop.rate == 0.1
        assert layer.proj_drop.rate == 0.1
        assert layer.data_format == "channels_first"

    def test_invalid_dim(self):
        with self.assertRaises(AssertionError):
            TalkingHeadAttention(dim=65, num_heads=8)

    def test_invalid_data_format(self):
        with self.assertRaises(AssertionError):
            TalkingHeadAttention(
                dim=self.dim, num_heads=self.num_heads, data_format="invalid_format"
            )

    def test_invalid_input_dims(self):
        layer = TalkingHeadAttention(dim=self.dim, num_heads=self.num_heads)
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.seq_length, self.dim, 10))
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.seq_length, self.dim + 1))

    def test_build_channels_last(self):
        layer = TalkingHeadAttention(dim=self.dim, num_heads=self.num_heads)
        layer.build(self.input_shape_channels_last)
        assert hasattr(layer, "qkv")
        assert hasattr(layer, "proj")
        assert hasattr(layer, "proj_l")
        assert hasattr(layer, "proj_w")
        assert layer.qkv.kernel.shape == (self.dim, self.dim * 3)
        assert layer.proj.kernel.shape == (self.dim, self.dim)
        assert layer.proj_l.kernel.shape == (self.num_heads, self.num_heads)
        assert layer.proj_w.kernel.shape == (self.num_heads, self.num_heads)

    def test_build_channels_first(self):
        layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        layer.build(self.input_shape_channels_first)
        assert hasattr(layer, "qkv")
        assert hasattr(layer, "proj")
        assert hasattr(layer, "proj_l")
        assert hasattr(layer, "proj_w")
        assert layer.qkv.kernel.shape == (self.dim, self.dim * 3)
        assert layer.proj.kernel.shape == (self.dim, self.dim)
        assert layer.proj_l.kernel.shape == (self.num_heads, self.num_heads)
        assert layer.proj_w.kernel.shape == (self.num_heads, self.num_heads)

    def test_call_channels_last(self):
        layer = TalkingHeadAttention(dim=self.dim, num_heads=self.num_heads)
        outputs = layer(self.test_inputs_channels_last)
        output_shape = ops.shape(outputs)

        assert len(output_shape) == len(self.input_shape_channels_last)
        assert all(
            output_shape[i] == self.input_shape_channels_last[i]
            for i in range(len(self.input_shape_channels_last))
        )

    def test_call_channels_first(self):
        layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        outputs = layer(self.test_inputs_channels_first)
        output_shape = ops.shape(outputs)

        assert len(output_shape) == len(self.input_shape_channels_first)
        assert all(
            output_shape[i] == self.input_shape_channels_first[i]
            for i in range(len(self.input_shape_channels_first))
        )

    def test_training_vs_inference_channels_last(self):
        layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, attn_drop=0.5, proj_drop=0.5
        )
        train_output = layer(self.test_inputs_channels_last, training=True)
        infer_output = layer(self.test_inputs_channels_last, training=False)

        assert ops.shape(train_output) == ops.shape(infer_output)

        self.assertNotAllClose(train_output, infer_output)

    def test_training_vs_inference_channels_first(self):
        layer = TalkingHeadAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            attn_drop=0.5,
            proj_drop=0.5,
            data_format="channels_first",
        )
        train_output = layer(self.test_inputs_channels_first, training=True)
        infer_output = layer(self.test_inputs_channels_first, training=False)

        assert ops.shape(train_output) == ops.shape(infer_output)

        self.assertNotAllClose(train_output, infer_output)

    def test_get_config(self):
        layer = TalkingHeadAttention(
            dim=self.dim,
            num_heads=4,
            qkv_bias=False,
            attn_drop=0.1,
            proj_drop=0.2,
            data_format="channels_first",
            block_prefix="custom_block",
        )
        config = layer.get_config()

        assert "dim" in config
        assert "num_heads" in config
        assert "qkv_bias" in config
        assert "attn_drop" in config
        assert "proj_drop" in config
        assert "data_format" in config
        assert "block_prefix" in config

        assert config["dim"] == self.dim
        assert config["num_heads"] == 4
        assert config["qkv_bias"] is False
        assert config["attn_drop"] == 0.1
        assert config["proj_drop"] == 0.2
        assert config["data_format"] == "channels_first"
        assert config["block_prefix"] == "custom_block"

        reconstructed_layer = TalkingHeadAttention.from_config(config)
        assert reconstructed_layer.dim == layer.dim
        assert reconstructed_layer.num_heads == layer.num_heads
        assert reconstructed_layer.data_format == layer.data_format
        assert reconstructed_layer.block_prefix == layer.block_prefix
        assert reconstructed_layer.attn_drop.rate == layer.attn_drop.rate
        assert reconstructed_layer.proj_drop.rate == layer.proj_drop.rate

    def test_different_batch_sizes_channels_last(self):
        layer = TalkingHeadAttention(dim=self.dim, num_heads=self.num_heads)
        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.seq_length, self.dim))
            outputs = layer(inputs)

            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.seq_length, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_batch_sizes_channels_first(self):
        layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.dim, self.seq_length))
            outputs = layer(inputs)

            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.dim, self.seq_length)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_sequence_lengths_channels_last(self):
        layer = TalkingHeadAttention(dim=self.dim, num_heads=self.num_heads)
        test_seq_lengths = [8, 32, 64]
        for seq_len in test_seq_lengths:
            inputs = ops.ones((self.batch_size, seq_len, self.dim))
            outputs = layer(inputs)

            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, seq_len, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_sequence_lengths_channels_first(self):
        layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        test_seq_lengths = [8, 32, 64]
        for seq_len in test_seq_lengths:
            inputs = ops.ones((self.batch_size, self.dim, seq_len))
            outputs = layer(inputs)

            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, self.dim, seq_len)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_numerical_stability_channels_last(self):
        layer = TalkingHeadAttention(dim=self.dim, num_heads=self.num_heads)
        small_inputs = self.test_inputs_channels_last * 1e-10
        small_outputs = layer(small_inputs)
        assert not ops.any(ops.isnan(small_outputs))
        assert not ops.any(ops.isinf(small_outputs))

        large_inputs = self.test_inputs_channels_last * 1e10
        large_outputs = layer(large_inputs)
        assert not ops.any(ops.isnan(large_outputs))
        assert not ops.any(ops.isinf(large_outputs))

    def test_numerical_stability_channels_first(self):
        layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        small_inputs = self.test_inputs_channels_first * 1e-10
        small_outputs = layer(small_inputs)
        assert not ops.any(ops.isnan(small_outputs))
        assert not ops.any(ops.isinf(small_outputs))

        large_inputs = self.test_inputs_channels_first * 1e10
        large_outputs = layer(large_inputs)
        assert not ops.any(ops.isnan(large_outputs))
        assert not ops.any(ops.isinf(large_outputs))

    def test_attention_computation_channels_last(self):
        layer = TalkingHeadAttention(dim=self.dim, num_heads=self.num_heads)
        x = ops.eye(self.seq_length)
        x = ops.expand_dims(x, axis=0)
        x = ops.repeat(x, self.dim // self.seq_length, axis=-1)
        x = ops.repeat(x, self.batch_size, axis=0)
        outputs = layer(x)

        assert ops.shape(outputs) == (self.batch_size, self.seq_length, self.dim)

    def test_attention_computation_channels_first(self):
        layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        x = ops.eye(self.seq_length)
        x = ops.expand_dims(x, axis=0)
        x = ops.repeat(x, self.dim // self.seq_length, axis=-1)
        x = ops.repeat(x, self.batch_size, axis=0)
        x = ops.transpose(x, (0, 2, 1))
        outputs = layer(x)

        assert ops.shape(outputs) == (self.batch_size, self.dim, self.seq_length)

    def test_format_consistency(self):
        channels_last_layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_last"
        )
        channels_first_layer = TalkingHeadAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )

        channels_last_layer.build(self.input_shape_channels_last)
        channels_first_layer.build(self.input_shape_channels_first)

        initial_qkv_kernel = keras.random.normal(
            shape=channels_last_layer.qkv.kernel.shape
        )
        initial_qkv_bias = keras.random.normal(shape=channels_last_layer.qkv.bias.shape)
        initial_proj_kernel = keras.random.normal(
            shape=channels_last_layer.proj.kernel.shape
        )
        initial_proj_bias = keras.random.normal(
            shape=channels_last_layer.proj.bias.shape
        )
        initial_proj_l_kernel = keras.random.normal(
            shape=channels_last_layer.proj_l.kernel.shape
        )
        initial_proj_l_bias = keras.random.normal(
            shape=channels_last_layer.proj_l.bias.shape
        )
        initial_proj_w_kernel = keras.random.normal(
            shape=channels_last_layer.proj_w.kernel.shape
        )
        initial_proj_w_bias = keras.random.normal(
            shape=channels_last_layer.proj_w.bias.shape
        )

        channels_last_layer.qkv.kernel.assign(initial_qkv_kernel)
        channels_last_layer.qkv.bias.assign(initial_qkv_bias)
        channels_last_layer.proj.kernel.assign(initial_proj_kernel)
        channels_last_layer.proj.bias.assign(initial_proj_bias)
        channels_last_layer.proj_l.kernel.assign(initial_proj_l_kernel)
        channels_last_layer.proj_l.bias.assign(initial_proj_l_bias)
        channels_last_layer.proj_w.kernel.assign(initial_proj_w_kernel)
        channels_last_layer.proj_w.bias.assign(initial_proj_w_bias)

        channels_first_layer.qkv.kernel.assign(initial_qkv_kernel)
        channels_first_layer.qkv.bias.assign(initial_qkv_bias)
        channels_first_layer.proj.kernel.assign(initial_proj_kernel)
        channels_first_layer.proj.bias.assign(initial_proj_bias)
        channels_first_layer.proj_l.kernel.assign(initial_proj_l_kernel)
        channels_first_layer.proj_l.bias.assign(initial_proj_l_bias)
        channels_first_layer.proj_w.kernel.assign(initial_proj_w_kernel)
        channels_first_layer.proj_w.bias.assign(initial_proj_w_bias)

        test_input = ops.ones((2, self.seq_length, self.dim))
        output_channels_last = channels_last_layer(test_input)
        test_input_channels_first = ops.transpose(test_input, (0, 2, 1))
        output_channels_first = channels_first_layer(test_input_channels_first)
        output_channels_first_converted = ops.transpose(
            output_channels_first, (0, 2, 1)
        )

        self.assertAllClose(
            output_channels_last,
            output_channels_first_converted,
            atol=1e-5,
        )
