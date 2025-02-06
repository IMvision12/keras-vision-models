import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .std_conv2d import StdConv2D


class TestStdConv2D(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.height = 32
        self.width = 32
        self.in_channels = 3
        self.out_channels = 16
        self.kernel_size = 3
        self.input_shape = (self.batch_size, self.height, self.width, self.in_channels)
        self.test_inputs = ops.ones(self.input_shape)
        self.eps = 1e-8

    def test_init(self):
        layer = StdConv2D(
            filters=self.out_channels, kernel_size=self.kernel_size, eps=self.eps
        )
        self.assertEqual(layer.filters, self.out_channels)
        self.assertEqual(layer.kernel_size, (self.kernel_size, self.kernel_size))
        self.assertEqual(layer.eps, self.eps)

    def test_invalid_eps(self):
        with self.assertRaises(ValueError):
            StdConv2D(
                filters=self.out_channels, kernel_size=self.kernel_size, eps=-1e-8
            )

    def test_output_shape(self):
        layer = StdConv2D(
            filters=self.out_channels, kernel_size=self.kernel_size, padding="same"
        )
        output = layer(self.test_inputs)
        expected_shape = (self.batch_size, self.height, self.width, self.out_channels)
        self.assertEqual(output.shape, expected_shape)

    def test_kernel_standardization(self):
        layer = StdConv2D(filters=self.out_channels, kernel_size=self.kernel_size)
        _ = layer(self.test_inputs)

        std_kernel = layer.standardize_kernel(layer.kernel)

        kernel_mean = ops.mean(std_kernel, axis=[0, 1, 2])
        self.assertTrue(np.allclose(kernel_mean.numpy(), 0, atol=1e-6))

        kernel_var = ops.var(std_kernel, axis=[0, 1, 2])
        self.assertTrue(np.allclose(kernel_var.numpy(), 1, atol=1e-6))

    def test_different_kernel_sizes(self):
        test_sizes = [(1, 1), (3, 3), (5, 5), (7, 7)]

        for size in test_sizes:
            layer = StdConv2D(
                filters=self.out_channels, kernel_size=size, padding="same"
            )
            output = layer(self.test_inputs)
            self.assertEqual(
                output.shape,
                (self.batch_size, self.height, self.width, self.out_channels),
            )

    def test_different_paddings(self):
        paddings = ["valid", "same"]

        for padding in paddings:
            layer = StdConv2D(
                filters=self.out_channels, kernel_size=self.kernel_size, padding=padding
            )
            output = layer(self.test_inputs)

            if padding == "valid":
                expected_height = self.height - self.kernel_size + 1
                expected_width = self.width - self.kernel_size + 1
            else:
                expected_height = self.height
                expected_width = self.width

            self.assertEqual(
                output.shape,
                (self.batch_size, expected_height, expected_width, self.out_channels),
            )

    def test_with_bias(self):
        layer = StdConv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            use_bias=True,
            padding="same",
        )
        output_with_bias = layer(self.test_inputs)

        layer_no_bias = StdConv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            use_bias=False,
            padding="same",
        )
        layer_no_bias.build(self.input_shape)
        layer_no_bias.kernel.assign(layer.kernel)

        output_without_bias = layer_no_bias(self.test_inputs)

        self.assertFalse(
            np.allclose(output_with_bias.numpy(), output_without_bias.numpy())
        )

    def test_activation(self):
        layer = StdConv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            activation="relu",
            padding="same",
        )
        output = layer(self.test_inputs)

        self.assertTrue(np.all(output.numpy() >= 0))

    def test_different_data_formats(self):
        formats = ["channels_last", "channels_first"]

        for data_format in formats:
            layer = StdConv2D(
                filters=self.out_channels,
                kernel_size=self.kernel_size,
                data_format=data_format,
                padding="same",
            )

            if data_format == "channels_first":
                inputs = ops.transpose(self.test_inputs, [0, 3, 1, 2])
            else:
                inputs = self.test_inputs

            output = layer(inputs)

            if data_format == "channels_first":
                expected_shape = (
                    self.batch_size,
                    self.out_channels,
                    self.height,
                    self.width,
                )
            else:
                expected_shape = (
                    self.batch_size,
                    self.height,
                    self.width,
                    self.out_channels,
                )

            self.assertEqual(output.shape, expected_shape)
