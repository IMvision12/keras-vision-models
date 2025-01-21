import numpy as np
from keras import ops
from keras.src.testing import TestCase

from kv.layers import EfficientMultiheadSelfAttention


class TestEfficientMultiheadSelfAttention(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.seq_length = 64
        self.project_dim = 64
        self.num_heads = 8
        self.head_dim = self.project_dim // self.num_heads
        self.input_shape = (self.batch_size, self.seq_length, self.project_dim)
        self.test_inputs = ops.ones(self.input_shape)

    def test_init_default(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim, sr_ratio=1, block_prefix="block1"
        )
        assert layer.project_dim == self.project_dim
        assert layer.num_heads == 8
        assert layer.sr_ratio == 1
        assert layer.block_prefix == "block1"
        assert layer.scale == (self.project_dim // 8) ** -0.5
        assert layer.epsilon == 1e-6
        assert not layer.built
        assert not layer.q.use_bias
        assert layer.attn_drop.rate == 0.1

    def test_init_with_options(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim,
            sr_ratio=2,
            block_prefix="block2",
            num_heads=4,
            proj_drop=0.1,
            attn_drop=0.1,
            qkv_bias=True,
            epsilon=1e-5,
        )
        assert layer.project_dim == self.project_dim
        assert layer.num_heads == 4
        assert layer.sr_ratio == 2
        assert layer.block_prefix == "block2"
        assert layer.epsilon == 1e-5
        assert hasattr(layer, "sr")
        assert hasattr(layer, "norm")
        assert layer.attn_drop.rate == 0.1
        assert layer.proj_drop.rate == 0.1
        assert layer.q.use_bias
        assert layer.kv.use_bias

    def test_invalid_project_dim(self):
        with self.assertRaises(AssertionError):
            EfficientMultiheadSelfAttention(
                project_dim=65, sr_ratio=1, block_prefix="block1", num_heads=8
            )

    def test_build(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim,
            sr_ratio=1,
            block_prefix="block1",
            qkv_bias=True,
        )
        layer.build(self.input_shape)
        assert hasattr(layer, "q")
        assert hasattr(layer, "kv")
        assert hasattr(layer, "proj")
        assert layer.q.kernel.shape == (self.project_dim, self.project_dim)
        assert layer.kv.kernel.shape == (self.project_dim, self.project_dim * 2)
        assert layer.proj.kernel.shape == (self.project_dim, self.project_dim)
        assert layer.q.bias is not None
        assert layer.kv.bias is not None

    def test_call_no_reduction(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim, sr_ratio=1, block_prefix="block1"
        )
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_shape)
        assert all(
            output_shape[i] == self.input_shape[i] for i in range(len(self.input_shape))
        )

    def test_call_with_reduction(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim, sr_ratio=2, block_prefix="block1"
        )
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)
        assert len(output_shape) == len(self.input_shape)
        assert all(
            output_shape[i] == self.input_shape[i] for i in range(len(self.input_shape))
        )

    def test_training_vs_inference(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim,
            sr_ratio=2,
            block_prefix="block1",
            proj_drop=0.9,
            attn_drop=0.9,
        )

        for _ in range(5):
            train_output = layer(self.test_inputs, training=True)
            infer_output = layer(self.test_inputs, training=False)

            assert ops.shape(train_output) == ops.shape(infer_output)

            try:
                assert not np.allclose(
                    train_output.numpy(), infer_output.numpy(), rtol=1e-3
                )
                return
            except AssertionError:
                continue

        raise AssertionError("Dropout does not appear to be working during training")

    def test_get_config(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim,
            sr_ratio=2,
            block_prefix="block1",
            num_heads=4,
            attn_drop=0.1,
            qkv_bias=True,
            epsilon=1e-5,
        )
        config = layer.get_config()
        assert "project_dim" in config
        assert "sr_ratio" in config
        assert "block_prefix" in config
        assert "num_heads" in config
        assert "attn_drop" in config
        assert "qkv_bias" in config
        assert "epsilon" in config
        assert config["project_dim"] == self.project_dim
        assert config["sr_ratio"] == 2
        assert config["block_prefix"] == "block1"
        assert config["num_heads"] == 4
        assert config["attn_drop"] == 0.1
        assert config["qkv_bias"] is True
        assert config["epsilon"] == 1e-5

        reconstructed_layer = EfficientMultiheadSelfAttention.from_config(config)
        assert reconstructed_layer.project_dim == layer.project_dim
        assert reconstructed_layer.sr_ratio == layer.sr_ratio
        assert reconstructed_layer.block_prefix == layer.block_prefix
        assert reconstructed_layer.num_heads == layer.num_heads
        assert reconstructed_layer.attn_drop.rate == layer.attn_drop.rate
        assert reconstructed_layer.q.use_bias == layer.q.use_bias
        assert reconstructed_layer.epsilon == layer.epsilon

    def test_qkv_bias_computation(self):
        for use_bias in [True, False]:
            layer = EfficientMultiheadSelfAttention(
                project_dim=self.project_dim,
                sr_ratio=1,
                block_prefix="block1",
                qkv_bias=use_bias,
            )

            outputs = layer(self.test_inputs)
            assert ops.shape(outputs) == self.input_shape

            assert layer.q.use_bias == use_bias
            assert layer.kv.use_bias == use_bias
            if use_bias:
                assert layer.q.bias is not None
                assert layer.kv.bias is not None
            else:
                assert layer.q.bias is None
                assert layer.kv.bias is None

    def test_attention_dropout(self):
        for attn_drop_rate in [0.0, 0.3, 0.7]:
            layer = EfficientMultiheadSelfAttention(
                project_dim=self.project_dim,
                sr_ratio=1,
                block_prefix="block1",
                attn_drop=attn_drop_rate,
            )

            train_output = layer(self.test_inputs, training=True)
            assert ops.shape(train_output) == self.input_shape

            infer_output = layer(self.test_inputs, training=False)
            assert ops.shape(infer_output) == self.input_shape

            assert layer.attn_drop.rate == attn_drop_rate

    def test_different_batch_sizes(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim, sr_ratio=2, block_prefix="block1"
        )
        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.seq_length, self.project_dim))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.seq_length, self.project_dim)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_sr_ratios(self):
        test_sr_ratios = [1, 2, 4]
        for sr_ratio in test_sr_ratios:
            layer = EfficientMultiheadSelfAttention(
                project_dim=self.project_dim,
                sr_ratio=sr_ratio,
                block_prefix=f"block_{sr_ratio}",
                qkv_bias=True,
                attn_drop=0.1,
            )
            outputs = layer(self.test_inputs)
            output_shape = ops.shape(outputs)
            assert all(
                output_shape[i] == self.input_shape[i]
                for i in range(len(self.input_shape))
            )

    def test_attention_computation(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim,
            sr_ratio=2,
            block_prefix="block1",
            qkv_bias=True,
            attn_drop=0.1,
        )
        x = ops.eye(self.seq_length)
        x = ops.expand_dims(x, axis=0)
        x = ops.repeat(x, self.project_dim // self.seq_length, axis=-1)
        x = ops.repeat(x, self.batch_size, axis=0)
        outputs = layer(x)
        assert ops.shape(outputs) == (
            self.batch_size,
            self.seq_length,
            self.project_dim,
        )

    def test_layer_normalization_epsilon(self):
        layer = EfficientMultiheadSelfAttention(
            project_dim=self.project_dim,
            sr_ratio=2,
            block_prefix="block1",
            epsilon=1e-5,
        )
        layer.build(self.input_shape)
        assert hasattr(layer, "norm")
        assert layer.norm.epsilon == 1e-5
