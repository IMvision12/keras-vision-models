import keras
import numpy as np
from keras import ops
from keras.src.testing import TestCase

from kvmm.layers import WindowAttention


class TestWindowAttention(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.height = 16
        self.width = 16
        self.dim = 64
        self.num_heads = 8
        self.window_size = 4
        self.bias_table_window_size = 8
        self.head_dim = self.dim // self.num_heads
        self.input_shape = (self.batch_size, self.height, self.width, self.dim)
        self.test_inputs = ops.ones(self.input_shape)

        coords_h = ops.arange(self.window_size)
        coords_w = ops.arange(self.window_size)
        coords = ops.stack(ops.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = ops.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = ops.transpose(relative_coords, [1, 2, 0])
        relative_coords_h = relative_coords[:, :, 0]
        relative_coords_w = relative_coords[:, :, 1]

        self.relative_position_index = (relative_coords_h + self.window_size - 1) * (
            2 * self.window_size - 1
        ) + (relative_coords_w + self.window_size - 1)

        self.num_windows = (self.height // self.window_size) * (
            self.width // self.window_size
        )

    def test_init_default(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        self.assertEqual(layer.dim, self.dim)
        self.assertEqual(layer.num_heads, self.num_heads)
        self.assertEqual(layer.window_size, self.window_size)
        self.assertEqual(layer.bias_table_window_size, self.bias_table_window_size)
        self.assertEqual(layer.head_dim, self.dim // self.num_heads)
        self.assertAlmostEqual(layer.scale, (self.dim // self.num_heads) ** -0.5)
        self.assertTrue(layer.qkv_bias)
        self.assertEqual(layer.block_prefix, "blocks")
        self.assertFalse(layer.built)

    def test_init_with_options(self):
        custom_qk_scale = 0.125
        layer = WindowAttention(
            dim=self.dim,
            num_heads=4,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
            qkv_bias=False,
            qk_scale=custom_qk_scale,
            attn_drop=0.1,
            proj_drop=0.1,
            block_prefix="custom_block",
        )
        self.assertEqual(layer.dim, self.dim)
        self.assertEqual(layer.num_heads, 4)
        self.assertEqual(layer.window_size, self.window_size)
        self.assertEqual(layer.bias_table_window_size, self.bias_table_window_size)
        self.assertEqual(layer.head_dim, self.dim // 4)
        self.assertEqual(layer.scale, custom_qk_scale)
        self.assertFalse(layer.qkv_bias)
        self.assertEqual(layer.block_prefix, "custom_block")

    def test_invalid_dim(self):
        with self.assertRaises(AssertionError):
            WindowAttention(
                dim=65,
                num_heads=8,
                window_size=self.window_size,
                bias_table_window_size=self.bias_table_window_size,
            )

    def test_build(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        layer.build([self.input_shape, (), self.relative_position_index.shape, None])
        self.assertTrue(hasattr(layer, "qkv"))
        self.assertTrue(hasattr(layer, "proj"))
        self.assertTrue(hasattr(layer, "relative_bias"))
        self.assertEqual(layer.qkv.kernel.shape, (self.dim, self.dim * 3))
        self.assertEqual(layer.proj.kernel.shape, (self.dim, self.dim))
        self.assertEqual(
            layer.relative_bias.shape,
            [(2 * self.bias_table_window_size - 1) ** 2, self.num_heads],
        )

    def test_invalid_input_dims(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        invalid_shape = (self.batch_size, self.height, self.width, self.dim + 1)
        with self.assertRaises(ValueError):
            layer.build([invalid_shape, (), self.relative_position_index.shape, None])

    def test_call_basic(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        window_size_tensor = ops.convert_to_tensor(self.window_size)
        layer.build([self.input_shape, (), self.relative_position_index.shape, None])
        outputs = layer(
            [self.test_inputs, window_size_tensor, self.relative_position_index, None]
        )
        output_shape = ops.shape(outputs)
        self.assertEqual(len(output_shape), len(self.input_shape))
        for i in range(len(self.input_shape)):
            self.assertEqual(output_shape[i], self.input_shape[i])

    def test_call_with_mask(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        window_size_tensor = ops.convert_to_tensor(self.window_size)

        B = self.batch_size
        H, W = self.height, self.width
        num_windows = (H // self.window_size) * (W // self.window_size)
        attn_mask = ops.zeros(
            (num_windows, B, self.num_heads, self.window_size**2, self.window_size**2)
        )

        layer.build(
            [self.input_shape, (), self.relative_position_index.shape, attn_mask.shape]
        )
        outputs = layer(
            [
                self.test_inputs,
                window_size_tensor,
                self.relative_position_index,
                attn_mask,
            ]
        )
        output_shape = ops.shape(outputs)
        self.assertEqual(len(output_shape), len(self.input_shape))
        for i in range(len(self.input_shape)):
            self.assertEqual(output_shape[i], self.input_shape[i])

    def test_training_vs_inference(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
            attn_drop=0.5,
            proj_drop=0.5,
        )
        window_size_tensor = ops.convert_to_tensor(self.window_size)
        layer.build([self.input_shape, (), self.relative_position_index.shape, None])
        train_output = layer(
            [self.test_inputs, window_size_tensor, self.relative_position_index, None],
            training=True,
        )
        infer_output = layer(
            [self.test_inputs, window_size_tensor, self.relative_position_index, None],
            training=False,
        )
        self.assertEqual(ops.shape(train_output), ops.shape(infer_output))
        self.assertNotEqual(ops.mean(train_output), ops.mean(infer_output))

    def test_get_config(self):
        custom_qk_scale = 0.125
        layer = WindowAttention(
            dim=self.dim,
            num_heads=4,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
            qkv_bias=False,
            qk_scale=custom_qk_scale,
            attn_drop=0.1,
            proj_drop=0.1,
            block_prefix="custom_block",
        )
        config = layer.get_config()
        self.assertIn("dim", config)
        self.assertIn("num_heads", config)
        self.assertIn("window_size", config)
        self.assertIn("bias_table_window_size", config)
        self.assertIn("qkv_bias", config)
        self.assertIn("qk_scale", config)
        self.assertIn("attn_drop", config)
        self.assertIn("proj_drop", config)
        self.assertIn("block_prefix", config)

        self.assertEqual(config["dim"], self.dim)
        self.assertEqual(config["num_heads"], 4)
        self.assertEqual(config["window_size"], self.window_size)
        self.assertEqual(config["bias_table_window_size"], self.bias_table_window_size)
        self.assertEqual(config["qkv_bias"], False)
        self.assertEqual(config["qk_scale"], custom_qk_scale)
        self.assertEqual(config["attn_drop"], 0.1)
        self.assertEqual(config["proj_drop"], 0.1)
        self.assertEqual(config["block_prefix"], "custom_block")

        reconstructed_layer = WindowAttention.from_config(config)
        self.assertEqual(reconstructed_layer.dim, layer.dim)
        self.assertEqual(reconstructed_layer.num_heads, layer.num_heads)
        self.assertEqual(reconstructed_layer.window_size, layer.window_size)
        self.assertEqual(
            reconstructed_layer.bias_table_window_size, layer.bias_table_window_size
        )
        self.assertEqual(reconstructed_layer.qkv_bias, layer.qkv_bias)
        self.assertEqual(reconstructed_layer.scale, layer.scale)
        self.assertEqual(reconstructed_layer.block_prefix, layer.block_prefix)

    def test_different_batch_sizes(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        layer.build([self.input_shape, (), self.relative_position_index.shape, None])
        window_size_tensor = ops.convert_to_tensor(self.window_size)
        test_batch_sizes = [1, 8, 16]

        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.height, self.width, self.dim))
            outputs = layer(
                [inputs, window_size_tensor, self.relative_position_index, None]
            )
            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.height, self.width, self.dim)
            for i in range(len(expected_shape)):
                self.assertEqual(output_shape[i], expected_shape[i])

    def test_different_spatial_dimensions(self):
        test_sizes = [(8, 8), (16, 16), (32, 32)]

        for h, w in test_sizes:
            if h % self.window_size == 0 and w % self.window_size == 0:
                layer = WindowAttention(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    bias_table_window_size=self.bias_table_window_size,
                )
                window_size_tensor = ops.convert_to_tensor(self.window_size)

                input_shape = (self.batch_size, h, w, self.dim)
                inputs = ops.ones(input_shape)

                layer.build([input_shape, (), self.relative_position_index.shape, None])

                outputs = layer(
                    [inputs, window_size_tensor, self.relative_position_index, None]
                )
                output_shape = ops.shape(outputs)
                expected_shape = (self.batch_size, h, w, self.dim)
                for i in range(len(expected_shape)):
                    self.assertEqual(output_shape[i], expected_shape[i])

    def test_numerical_stability(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        window_size_tensor = ops.convert_to_tensor(self.window_size)
        layer.build([self.input_shape, (), self.relative_position_index.shape, None])

        small_inputs = self.test_inputs * 0.001
        small_outputs = layer(
            [small_inputs, window_size_tensor, self.relative_position_index, None]
        )
        self.assertFalse(ops.any(ops.isnan(small_outputs)))
        self.assertFalse(ops.any(ops.isinf(small_outputs)))

        large_inputs = self.test_inputs * 1000
        large_outputs = layer(
            [large_inputs, window_size_tensor, self.relative_position_index, None]
        )
        self.assertFalse(ops.any(ops.isnan(large_outputs)))
        self.assertFalse(ops.any(ops.isinf(large_outputs)))

    def test_attention_computation(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        window_size_tensor = ops.convert_to_tensor(self.window_size)
        layer.build([self.input_shape, (), self.relative_position_index.shape, None])

        test_input = ops.ones(self.input_shape)
        outputs = layer(
            [test_input, window_size_tensor, self.relative_position_index, None]
        )

        self.assertEqual(ops.shape(outputs), self.input_shape)

    def test_mask_application(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        window_size_tensor = ops.convert_to_tensor(self.window_size)

        num_windows = (self.height // self.window_size) * (
            self.width // self.window_size
        )
        B = self.batch_size

        random_inputs = keras.ops.convert_to_tensor(
            keras.random.normal(
                shape=self.input_shape, mean=0, stddev=1, dtype="float32"
            )
        )

        zero_mask = ops.zeros(
            (num_windows, B, self.num_heads, self.window_size**2, self.window_size**2)
        )

        negative_mask = ops.zeros(
            (num_windows, B, self.num_heads, self.window_size**2, self.window_size**2)
        )
        negative_mask_np = negative_mask.numpy()
        negative_mask_np[0, :, :, :, :] = np.random.choice(
            [-1000.0, 0.0], size=negative_mask_np[0, :, :, :, :].shape, p=[0.5, 0.5]
        )
        negative_mask = ops.convert_to_tensor(negative_mask_np)

        layer.build(
            [self.input_shape, (), self.relative_position_index.shape, zero_mask.shape]
        )

        output_zero_mask = layer(
            [random_inputs, window_size_tensor, self.relative_position_index, zero_mask]
        )
        output_negative_mask = layer(
            [
                random_inputs,
                window_size_tensor,
                self.relative_position_index,
                negative_mask,
            ]
        )

        self.assertFalse(
            np.allclose(
                output_zero_mask.numpy(),
                output_negative_mask.numpy(),
                rtol=1e-3,
                atol=1e-3,
            )
        )

    def test_compute_output_shape(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
        )
        output_shape = layer.compute_output_shape([self.input_shape, None, None, None])
        self.assertEqual(output_shape, self.input_shape)

    def test_relative_bias_effect(self):
        layer = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            bias_table_window_size=self.bias_table_window_size,
            qk_scale=1.0,
        )
        layer.build([self.input_shape, (), self.relative_position_index.shape, None])
        window_size_tensor = ops.convert_to_tensor(self.window_size)

        random_inputs = ops.convert_to_tensor(
            np.random.normal(0, 1, self.input_shape).astype(np.float32)
        )

        output_original = layer(
            [random_inputs, window_size_tensor, self.relative_position_index, None]
        )

        original_bias = layer.relative_bias.numpy().copy()

        extreme_bias = ops.ones_like(layer.relative_bias) * 1000.0
        layer.relative_bias.assign(extreme_bias)

        output_extreme_bias = layer(
            [random_inputs, window_size_tensor, self.relative_position_index, None]
        )

        self.assertFalse(
            np.allclose(
                output_original.numpy(),
                output_extreme_bias.numpy(),
                rtol=1e-3,
                atol=1e-3,
            )
        )

        layer.relative_bias.assign(ops.convert_to_tensor(original_bias))
