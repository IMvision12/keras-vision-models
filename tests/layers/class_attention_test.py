import keras
from keras import ops
from keras.src.testing import TestCase

from kvmm.layers import ClassAttention


class TestClassAttention(TestCase):
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
        layer = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        assert layer.dim == self.dim
        assert layer.num_heads == self.num_heads
        assert layer.head_dim == self.dim // self.num_heads
        assert layer.scale == (self.dim // self.num_heads) ** -0.5
        assert layer.block_prefix is None
        assert layer.data_format == "channels_last"
        assert not layer.built

    def test_init_with_options(self):
        layer = ClassAttention(
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
        assert layer.data_format == "channels_first"
        assert not layer.q.use_bias
        assert not layer.k.use_bias
        assert not layer.v.use_bias

    def test_invalid_dim(self):
        with self.assertRaises(AssertionError):
            ClassAttention(dim=65, num_heads=8)

    def test_invalid_data_format(self):
        with self.assertRaises(AssertionError):
            ClassAttention(
                dim=self.dim, num_heads=self.num_heads, data_format="invalid_format"
            )

    def test_invalid_input_dims(self):
        layer = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.seq_length, self.seq_length, self.dim))

    def test_build_channels_last(self):
        layer = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        layer.build(self.input_shape_channels_last)
        assert hasattr(layer, "q")
        assert hasattr(layer, "k")
        assert hasattr(layer, "v")
        assert hasattr(layer, "proj")
        assert layer.q.kernel.shape == (self.dim, self.dim)
        assert layer.k.kernel.shape == (self.dim, self.dim)
        assert layer.v.kernel.shape == (self.dim, self.dim)
        assert layer.proj.kernel.shape == (self.dim, self.dim)

    def test_build_channels_first(self):
        layer = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        layer.build(self.input_shape_channels_first)
        assert hasattr(layer, "q")
        assert hasattr(layer, "k")
        assert hasattr(layer, "v")
        assert hasattr(layer, "proj")
        assert layer.q.kernel.shape == (self.dim, self.dim)
        assert layer.k.kernel.shape == (self.dim, self.dim)
        assert layer.v.kernel.shape == (self.dim, self.dim)
        assert layer.proj.kernel.shape == (self.dim, self.dim)

    def test_call_channels_last(self):
        layer = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        outputs = layer(self.test_inputs_channels_last)
        output_shape = ops.shape(outputs)
        expected_shape = (self.batch_size, 1, self.dim)
        assert len(output_shape) == len(expected_shape)
        assert all(
            output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
        )

    def test_call_channels_first(self):
        layer = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        outputs = layer(self.test_inputs_channels_first)
        output_shape = ops.shape(outputs)
        expected_shape = (self.batch_size, self.dim, 1)
        assert len(output_shape) == len(expected_shape)
        assert all(
            output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
        )

    def test_training_vs_inference(self):
        keras.utils.set_random_seed(42)
        layer_cl = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, attn_drop=0.5, proj_drop=0.5
        )
        train_output_cl = layer_cl(self.test_inputs_channels_last, training=True)
        infer_output_cl = layer_cl(self.test_inputs_channels_last, training=False)
        assert ops.shape(train_output_cl) == ops.shape(infer_output_cl)

        diff_cl = ops.sum(ops.abs(train_output_cl - infer_output_cl))
        assert diff_cl > 0, "Training and inference outputs should be different"

        layer_cf = ClassAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            attn_drop=0.5,
            proj_drop=0.5,
            data_format="channels_first",
        )
        train_output_cf = layer_cf(self.test_inputs_channels_first, training=True)
        infer_output_cf = layer_cf(self.test_inputs_channels_first, training=False)
        assert ops.shape(train_output_cf) == ops.shape(infer_output_cf)

        diff_cf = ops.sum(ops.abs(train_output_cf - infer_output_cf))
        assert diff_cf > 0, "Training and inference outputs should be different"

    def test_get_config(self):
        layer = ClassAttention(
            dim=self.dim,
            num_heads=4,
            qkv_bias=False,
            data_format="channels_first",
            block_prefix="custom_block",
        )
        config = layer.get_config()
        assert "dim" in config
        assert "num_heads" in config
        assert "qkv_bias" in config
        assert "data_format" in config
        assert "block_prefix" in config
        assert config["dim"] == self.dim
        assert config["num_heads"] == 4
        assert config["qkv_bias"] is False
        assert config["data_format"] == "channels_first"
        assert config["block_prefix"] == "custom_block"

        reconstructed_layer = ClassAttention.from_config(config)
        assert reconstructed_layer.dim == layer.dim
        assert reconstructed_layer.num_heads == layer.num_heads
        assert reconstructed_layer.data_format == layer.data_format
        assert reconstructed_layer.block_prefix == layer.block_prefix
        assert reconstructed_layer.q.use_bias == layer.q.use_bias

    def test_different_batch_sizes(self):
        test_batch_sizes = [1, 8, 16]

        layer_cl = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.seq_length, self.dim))
            outputs = layer_cl(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, 1, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

        layer_cf = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.dim, self.seq_length))
            outputs = layer_cf(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.dim, 1)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_sequence_lengths(self):
        test_seq_lengths = [8, 32, 64]

        layer_cl = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        for seq_length in test_seq_lengths:
            inputs = ops.ones((self.batch_size, seq_length, self.dim))
            outputs = layer_cl(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, 1, self.dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

        layer_cf = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        for seq_length in test_seq_lengths:
            inputs = ops.ones((self.batch_size, self.dim, seq_length))
            outputs = layer_cf(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, self.dim, 1)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_numerical_stability(self):
        layer_cl = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        small_inputs = self.test_inputs_channels_last * 1e-10
        small_outputs = layer_cl(small_inputs)
        assert not ops.any(ops.isnan(small_outputs))
        assert not ops.any(ops.isinf(small_outputs))

        large_inputs = self.test_inputs_channels_last * 1e10
        large_outputs = layer_cl(large_inputs)
        assert not ops.any(ops.isnan(large_outputs))
        assert not ops.any(ops.isinf(large_outputs))

        layer_cf = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        small_inputs = self.test_inputs_channels_first * 1e-10
        small_outputs = layer_cf(small_inputs)
        assert not ops.any(ops.isnan(small_outputs))
        assert not ops.any(ops.isinf(small_outputs))

        large_inputs = self.test_inputs_channels_first * 1e10
        large_outputs = layer_cf(large_inputs)
        assert not ops.any(ops.isnan(large_outputs))
        assert not ops.any(ops.isinf(large_outputs))

    def test_attention_computation(self):
        layer_cl = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        x_cl = ops.eye(self.seq_length)
        x_cl = ops.expand_dims(x_cl, axis=0)
        x_cl = ops.repeat(x_cl, self.dim // self.seq_length, axis=-1)
        x_cl = ops.repeat(x_cl, self.batch_size, axis=0)

        outputs_cl = layer_cl(x_cl)
        assert ops.shape(outputs_cl) == (self.batch_size, 1, self.dim)
        assert ops.sum(ops.abs(outputs_cl)) > 0

        layer_cf = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        x_cf = ops.eye(self.seq_length)
        x_cf = ops.expand_dims(x_cf, axis=0)
        x_cf = ops.repeat(x_cf, self.dim // self.seq_length, axis=-1)
        x_cf = ops.transpose(x_cf, (0, 2, 1))
        x_cf = ops.repeat(x_cf, self.batch_size, axis=0)

        outputs_cf = layer_cf(x_cf)
        assert ops.shape(outputs_cf) == (self.batch_size, self.dim, 1)
        assert ops.sum(ops.abs(outputs_cf)) > 0

    def test_class_token_attention(self):
        layer_cl = ClassAttention(dim=self.dim, num_heads=self.num_heads)
        inputs_cl = ops.zeros(self.input_shape_channels_last)
        temp_ones = ops.ones((self.batch_size, 1, self.dim))
        if inputs_cl.shape[1] > 1:
            zeros_part = ops.zeros((self.batch_size, inputs_cl.shape[1] - 1, self.dim))
            inputs_cl = ops.concatenate([temp_ones, zeros_part], axis=1)
        else:
            inputs_cl = temp_ones

        outputs_cl = layer_cl(inputs_cl)
        assert ops.shape(outputs_cl) == (self.batch_size, 1, self.dim)
        assert ops.sum(ops.abs(outputs_cl)) > 0

        temp_zeros = ops.zeros((self.batch_size, 1, self.dim))
        if self.input_shape_channels_last[1] > 1:
            ones_part = ops.ones(
                (self.batch_size, self.input_shape_channels_last[1] - 1, self.dim)
            )
            inputs_cl = ops.concatenate([temp_zeros, ones_part], axis=1)
        else:
            inputs_cl = temp_zeros

        outputs_cl = layer_cl(inputs_cl)
        assert ops.sum(ops.abs(outputs_cl)) > 0

        layer_cf = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )

        temp_ones = ops.ones((self.batch_size, self.dim, 1))
        if self.input_shape_channels_first[2] > 1:
            zeros_part = ops.zeros(
                (self.batch_size, self.dim, self.input_shape_channels_first[2] - 1)
            )
            inputs_cf = ops.concatenate([temp_ones, zeros_part], axis=2)
        else:
            inputs_cf = temp_ones

        outputs_cf = layer_cf(inputs_cf)
        assert ops.shape(outputs_cf) == (self.batch_size, self.dim, 1)
        assert ops.sum(ops.abs(outputs_cf)) > 0

        temp_zeros = ops.zeros((self.batch_size, self.dim, 1))
        if self.input_shape_channels_first[2] > 1:
            ones_part = ops.ones(
                (self.batch_size, self.dim, self.input_shape_channels_first[2] - 1)
            )
            inputs_cf = ops.concatenate([temp_zeros, ones_part], axis=2)
        else:
            inputs_cf = temp_zeros

        outputs_cf = layer_cf(inputs_cf)
        assert ops.sum(ops.abs(outputs_cf)) > 0

    def test_output_format_matches_input_format(self):
        layer_cl = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_last"
        )
        outputs_cl = layer_cl(self.test_inputs_channels_last)
        assert ops.shape(outputs_cl)[0] == self.batch_size
        assert ops.shape(outputs_cl)[1] == 1
        assert ops.shape(outputs_cl)[2] == self.dim

        layer_cf = ClassAttention(
            dim=self.dim, num_heads=self.num_heads, data_format="channels_first"
        )
        outputs_cf = layer_cf(self.test_inputs_channels_first)
        assert ops.shape(outputs_cf)[0] == self.batch_size
        assert ops.shape(outputs_cf)[1] == self.dim
        assert ops.shape(outputs_cf)[2] == 1
