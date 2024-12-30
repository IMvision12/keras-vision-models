import numpy as np
from keras import ops
from keras.src.testing import TestCase

from .class_dist_token import ClassDistToken


class TestClassDistToken(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.num_patches = 196
        self.hidden_size = 768
        self.input_shape = (self.batch_size, self.num_patches, self.hidden_size)
        self.test_inputs = ops.ones(self.input_shape)

    def test_init_default(self):
        layer = ClassDistToken()
        assert not layer.use_distillation

    def test_init_with_distillation(self):
        layer = ClassDistToken(use_distillation=True)
        assert layer.use_distillation

    def test_build_vit_mode(self):
        layer = ClassDistToken()
        layer.build(self.input_shape)
        assert hasattr(layer, "cls")
        assert layer.cls.shape == (1, 1, self.hidden_size)
        assert not hasattr(layer, "dist")

    def test_build_deit_mode(self):
        layer = ClassDistToken(use_distillation=True)
        layer.build(self.input_shape)
        assert hasattr(layer, "cls")
        assert hasattr(layer, "dist")
        assert layer.cls.shape == (1, 1, self.hidden_size)
        assert layer.dist.shape == (1, 1, self.hidden_size)

    def test_call_vit_mode(self):
        layer = ClassDistToken()
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)
        expected_shape = (self.batch_size, self.num_patches + 1, self.hidden_size)
        assert len(output_shape) == len(expected_shape)
        assert all(
            output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
        )

    def test_call_deit_mode(self):
        layer = ClassDistToken(use_distillation=True)
        outputs = layer(self.test_inputs)
        output_shape = ops.shape(outputs)
        expected_shape = (self.batch_size, self.num_patches + 2, self.hidden_size)
        assert len(output_shape) == len(expected_shape)
        assert all(
            output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
        )

    def test_get_config(self):
        layer = ClassDistToken(use_distillation=True)
        config = layer.get_config()
        assert "use_distillation" in config
        assert config["use_distillation"] is True
        reconstructed_layer = ClassDistToken.from_config(config)
        assert reconstructed_layer.use_distillation == layer.use_distillation

    def test_different_batch_sizes(self):
        layer = ClassDistToken()
        test_batch_sizes = [1, 8, 16]
        for batch_size in test_batch_sizes:
            inputs = ops.ones((batch_size, self.num_patches, self.hidden_size))
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (batch_size, self.num_patches + 1, self.hidden_size)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_different_hidden_sizes(self):
        test_hidden_sizes = [256, 512, 1024]
        for hidden_size in test_hidden_sizes:
            inputs = ops.ones((self.batch_size, self.num_patches, hidden_size))
            layer = ClassDistToken()
            outputs = layer(inputs)
            output_shape = ops.shape(outputs)
            expected_shape = (self.batch_size, self.num_patches + 1, hidden_size)
            assert all(
                output_shape[i] == expected_shape[i] for i in range(len(expected_shape))
            )

    def test_token_broadcasting_vit(self):
        layer = ClassDistToken(use_distillation=False)
        outputs = layer(self.test_inputs)
        cls_tokens = outputs[:, 0:1, :]
        assert np.allclose(cls_tokens[0], cls_tokens[1])

    def test_token_broadcasting_deit(self):
        layer = ClassDistToken(use_distillation=True)
        outputs = layer(self.test_inputs)
        cls_tokens = outputs[:, 0:1, :]
        dist_tokens = outputs[:, 1:2, :]
        assert np.allclose(cls_tokens[0], cls_tokens[1])
        assert np.allclose(dist_tokens[0], dist_tokens[1])
