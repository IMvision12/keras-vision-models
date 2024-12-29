import keras
import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .stochastic_depth import StochasticDepth


class TestStochasticDepth(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.height = 32
        self.width = 32
        self.channels = 3
        self.input_shape = (self.batch_size, self.height, self.width, self.channels)
        self.test_inputs = ops.ones(self.input_shape)
        self.drop_rate = 0.2

    def test_init(self):
        layer = StochasticDepth(drop_path_rate=self.drop_rate)
        self.assertEqual(layer.drop_path_rate, self.drop_rate)

        layer = StochasticDepth(drop_path_rate=0.0)
        self.assertEqual(layer.drop_path_rate, 0.0)

        layer = StochasticDepth(drop_path_rate=1.0)
        self.assertEqual(layer.drop_path_rate, 1.0)

    def test_invalid_drop_rate(self):
        with self.assertRaises(ValueError):
            StochasticDepth(drop_path_rate=-0.1)

        with self.assertRaises(ValueError):
            StochasticDepth(drop_path_rate=1.1)

    def test_call_training_mode(self):
        layer = StochasticDepth(drop_path_rate=self.drop_rate)
        output = layer(self.test_inputs, training=True)

        self.assertEqual(output.shape, self.input_shape)
        self.assertFalse(np.allclose(output.numpy(), self.test_inputs.numpy()))

        keep_prob = 1 - self.drop_rate
        max_expected_value = 1.0 / keep_prob
        self.assertTrue(np.any(output.numpy() > 1.0))
        self.assertTrue(np.all(output.numpy() <= max_expected_value + 1e-5))

    def test_call_inference_mode(self):
        layer = StochasticDepth(drop_path_rate=self.drop_rate)
        output = layer(self.test_inputs, training=False)

        self.assertEqual(output.shape, self.input_shape)
        self.assertTrue(np.allclose(output.numpy(), self.test_inputs.numpy()))

    def test_different_input_shapes(self):
        test_shapes = [(2, 16, 16, 3), (4, 10, 128), (8, 64), (1, 32, 32, 32, 3)]

        layer = StochasticDepth(drop_path_rate=self.drop_rate)

        for shape in test_shapes:
            inputs = ops.ones(shape)
            output = layer(inputs, training=True)
            self.assertEqual(output.shape, shape)

    def test_zero_drop_rate(self):
        layer = StochasticDepth(drop_path_rate=0.0)
        output = layer(self.test_inputs, training=True)

        self.assertTrue(np.allclose(output.numpy(), self.test_inputs.numpy()))

    def test_get_config(self):
        layer = StochasticDepth(drop_path_rate=self.drop_rate)
        config = layer.get_config()

        self.assertIn("drop_path_rate", config)
        self.assertEqual(config["drop_path_rate"], self.drop_rate)

        reconstructed_layer = StochasticDepth.from_config(config)
        self.assertEqual(reconstructed_layer.drop_path_rate, self.drop_rate)

    def test_different_seeds(self):
        layer = StochasticDepth(drop_path_rate=self.drop_rate)

        keras.utils.set_random_seed(42)
        output1 = layer(self.test_inputs, training=True)

        keras.utils.set_random_seed(43)
        output2 = layer(self.test_inputs, training=True)

        diff = ops.abs(ops.mean(output1) - ops.mean(output2))
        self.assertGreater(diff.numpy(), 1e-3)

    def test_deterministic_behavior(self):
        layer = StochasticDepth(drop_path_rate=self.drop_rate)

        keras.utils.set_random_seed(42)
        output1 = layer(self.test_inputs, training=True)
        mean1 = ops.mean(output1).numpy()

        keras.utils.set_random_seed(42)
        output2 = layer(self.test_inputs, training=True)

        mean2 = ops.mean(output2).numpy()
        self.assertAlmostEqual(mean1, mean2, decimal=1e-2)
        keep_prob = 1 - self.drop_rate
        max_value = 1.0 / keep_prob
        self.assertTrue(ops.all(output1 <= max_value + 1e-5))
        self.assertTrue(ops.all(output2 <= max_value + 1e-5))

    def test_statistical_properties(self):
        layer = StochasticDepth(drop_path_rate=self.drop_rate)
        num_runs = 100
        means = []
        for _ in range(num_runs):
            output = layer(self.test_inputs, training=True)
            means.append(ops.mean(output).numpy())
        overall_mean = np.mean(means)
        self.assertAlmostEqual(overall_mean, 1.0, decimal=0.1)
        variance = np.var(means)
        self.assertGreater(variance, 0)
        self.assertLess(variance, 1)
