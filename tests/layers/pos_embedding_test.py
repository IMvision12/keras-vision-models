from keras import ops
from keras.src.testing import TestCase

from kvmm.layers import AddPositionEmbs


class TestAddPositionEmbs(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.grid_h = 14
        self.grid_w = 14
        self.hidden_size = 768
        self.num_patches = self.grid_h * self.grid_w
        self.sequence_length = self.num_patches + 1
        self.input_shape = (self.batch_size, self.sequence_length, self.hidden_size)
        self.test_inputs = ops.ones(self.input_shape)

    def test_init_default(self):
        layer = AddPositionEmbs(grid_h=self.grid_h, grid_w=self.grid_w)
        assert not layer.no_embed_class
        assert not layer.use_distillation
        assert layer.grid_h == self.grid_h
        assert layer.grid_w == self.grid_w
        assert layer.resize_mode == "bilinear"

    def test_init_flexivit_mode(self):
        layer = AddPositionEmbs(
            grid_h=self.grid_h, grid_w=self.grid_w, no_embed_class=True
        )
        assert layer.no_embed_class
        assert not layer.use_distillation

    def test_init_deit_mode(self):
        layer = AddPositionEmbs(
            grid_h=self.grid_h, grid_w=self.grid_w, use_distillation=True
        )
        assert not layer.no_embed_class
        assert layer.use_distillation

    def test_build_invalid_input_shape(self):
        layer = AddPositionEmbs(grid_h=self.grid_h, grid_w=self.grid_w)
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.sequence_length))

    def test_build_invalid_sequence_length(self):
        layer = AddPositionEmbs(grid_h=self.grid_h, grid_w=self.grid_w)
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.sequence_length + 1, self.hidden_size))

    def test_build_standard_mode(self):
        layer = AddPositionEmbs(grid_h=self.grid_h, grid_w=self.grid_w)
        layer.build(self.input_shape)
        expected_shape = (1, self.sequence_length, self.hidden_size)
        assert layer.position_embedding.shape == expected_shape

    def test_build_flexivit_mode_patches_only(self):
        layer = AddPositionEmbs(
            grid_h=self.grid_h, grid_w=self.grid_w, no_embed_class=True
        )
        patches_only_shape = (self.batch_size, self.num_patches, self.hidden_size)
        layer.build(patches_only_shape)
        expected_shape = (1, self.num_patches, self.hidden_size)
        assert layer.position_embedding.shape == expected_shape
        assert not hasattr(layer, "skip_cls") or not layer.skip_cls

    def test_build_flexivit_mode_with_class(self):
        layer = AddPositionEmbs(
            grid_h=self.grid_h, grid_w=self.grid_w, no_embed_class=True
        )
        with_class_shape = (self.batch_size, self.num_patches + 1, self.hidden_size)
        layer.build(with_class_shape)
        expected_shape = (1, self.num_patches, self.hidden_size)
        assert layer.position_embedding.shape == expected_shape
        assert layer.skip_cls

    def test_call_standard_mode(self):
        layer = AddPositionEmbs(grid_h=self.grid_h, grid_w=self.grid_w)
        outputs = layer(self.test_inputs)
        assert outputs.shape == self.input_shape
        self.assertNotAllClose(outputs, self.test_inputs)

    def test_call_flexivit_mode_patches_only(self):
        layer = AddPositionEmbs(
            grid_h=self.grid_h, grid_w=self.grid_w, no_embed_class=True
        )
        patches_only_inputs = ops.ones(
            (self.batch_size, self.num_patches, self.hidden_size)
        )
        layer.build((self.batch_size, self.num_patches, self.hidden_size))
        outputs = layer(patches_only_inputs)
        assert outputs.shape == patches_only_inputs.shape
        self.assertNotAllClose(outputs, patches_only_inputs)

    def test_call_flexivit_mode_with_class(self):
        layer = AddPositionEmbs(
            grid_h=self.grid_h, grid_w=self.grid_w, no_embed_class=True
        )
        with_class_inputs = self.test_inputs
        outputs = layer(with_class_inputs)
        assert outputs.shape == with_class_inputs.shape
        self.assertAllClose(outputs[:, 0:1, :], with_class_inputs[:, 0:1, :])
        self.assertNotAllClose(outputs[:, 1:, :], with_class_inputs[:, 1:, :])

    def test_call_deit_mode(self):
        deit_sequence_length = self.num_patches + 2
        deit_inputs = ops.ones(
            (self.batch_size, deit_sequence_length, self.hidden_size)
        )
        layer = AddPositionEmbs(
            grid_h=self.grid_h, grid_w=self.grid_w, use_distillation=True
        )
        outputs = layer(deit_inputs)
        assert outputs.shape == (
            self.batch_size,
            deit_sequence_length,
            self.hidden_size,
        )

    def test_get_config(self):
        layer = AddPositionEmbs(
            grid_h=self.grid_h, grid_w=self.grid_w, no_embed_class=True
        )
        config = layer.get_config()
        assert "grid_h" in config
        assert "grid_w" in config
        assert "no_embed_class" in config
        assert "use_distillation" in config
        assert "resize_mode" in config
        assert config["grid_h"] == self.grid_h
        assert config["grid_w"] == self.grid_w
        assert config["no_embed_class"] is True
        assert config["use_distillation"] is False
        assert config["resize_mode"] == "bilinear"

    def test_save_load_variables(self):
        layer = AddPositionEmbs(grid_h=self.grid_h, grid_w=self.grid_w)
        layer.build(self.input_shape)
        store = {}
        layer.save_own_variables(store)

        assert "grid_h" in store
        assert "grid_w" in store
        assert "no_embed_class" in store
        assert "use_distillation" in store
        assert "0" in store
