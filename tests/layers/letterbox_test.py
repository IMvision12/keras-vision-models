import keras
from keras import ops
from keras.src.testing import TestCase
import numpy as np

from kvmm.layers import Letterbox


class TestLetterbox(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 4
        self.height = 480
        self.width = 640
        self.channels = 3
        self.input_shape = (self.batch_size, self.height, self.width, self.channels)
        self.single_image_shape = (self.height, self.width, self.channels)
        
        self.test_inputs_batch = keras.random.uniform(self.input_shape, minval=0.0, maxval=1.0)
        self.test_inputs_single = keras.random.uniform(self.single_image_shape, minval=0.0, maxval=1.0)

    def test_init_default(self):
        layer = Letterbox()
        assert layer.new_shape == (640, 640)
        assert layer.color == (114, 114, 114)
        assert layer.auto is True
        assert layer.scaleFill is False
        assert layer.scaleup is True
        assert layer.stride == 32
        assert layer.color_norm == [114/255.0, 114/255.0, 114/255.0]

    def test_init_custom_parameters(self):
        layer = Letterbox(
            new_shape=(416, 416),
            color=(128, 128, 128),
            auto=False,
            scaleFill=True,
            scaleup=False,
            stride=16
        )
        assert layer.new_shape == (416, 416)
        assert layer.color == (128, 128, 128)
        assert layer.auto is False
        assert layer.scaleFill is True
        assert layer.scaleup is False
        assert layer.stride == 16

    def test_init_square_shape(self):
        layer = Letterbox(new_shape=512)
        assert layer.new_shape == (512, 512)

    def test_call_single_image(self):
        layer = Letterbox()
        letterboxed, ratios, paddings = layer(self.test_inputs_single)
        
        assert len(ops.shape(letterboxed)) == 3
        assert ops.shape(ratios) == (2,)
        assert ops.shape(paddings) == (2,)
        assert ops.all(ratios > 0)
        assert ops.all(paddings >= 0)

    def test_call_batch_images(self):
        layer = Letterbox()
        letterboxed, ratios, paddings = layer(self.test_inputs_batch)
        
        letterboxed_shape = ops.shape(letterboxed)
        assert letterboxed_shape[0] == self.batch_size
        assert len(letterboxed_shape) == 4
        assert ops.shape(ratios) == (self.batch_size, 2)
        assert ops.shape(paddings) == (self.batch_size, 2)
        
        assert ops.all(ratios > 0)
        assert ops.all(paddings >= 0)

    def test_different_target_shapes(self):
        test_shapes = [(320, 320), (416, 416), (512, 512), (1024, 1024)]
        
        for target_shape in test_shapes:
            layer = Letterbox(new_shape=target_shape, auto=False)
            letterboxed, ratios, paddings = layer(self.test_inputs_single)
            
            assert len(ops.shape(letterboxed)) == 3

    def test_different_input_shapes(self):
        test_input_shapes = [
            (224, 224, 3),
            (300, 400, 3),
            (400, 300, 3),
            (100, 800, 3),
            (800, 100, 3), 
        ]
        
        layer = Letterbox()
        
        for input_shape in test_input_shapes:
            test_input = keras.random.uniform(input_shape, minval=0.0, maxval=1.0)
            letterboxed, ratios, paddings = layer(test_input)
            
            assert len(ops.shape(letterboxed)) == 3

    def test_scale_fill_mode(self):
        layer = Letterbox(scaleFill=True)
        letterboxed, ratios, paddings = layer(self.test_inputs_single)
        
        assert ops.all(paddings == 0.0)
        
        assert len(ops.shape(letterboxed)) == 3

    def test_no_scaleup_mode(self):
        small_image = keras.random.uniform((100, 100, 3), minval=0.0, maxval=1.0)
        
        layer_scaleup = Letterbox(scaleup=True)
        layer_no_scaleup = Letterbox(scaleup=False)
        
        _, ratios_scaleup, _ = layer_scaleup(small_image)
        _, ratios_no_scaleup, _ = layer_no_scaleup(small_image)
        
        assert ops.all(ratios_no_scaleup <= 1.0 + 1e-6)
        
        assert ops.all(ratios_scaleup > 0)
        assert ops.all(ratios_no_scaleup > 0)

    def test_auto_stride_adjustment(self):
        layer_auto = Letterbox(auto=True, stride=32)
        layer_no_auto = Letterbox(auto=False, stride=32)
        
        _, _, paddings_auto = layer_auto(self.test_inputs_single)
        _, _, paddings_no_auto = layer_no_auto(self.test_inputs_single)
        
        assert ops.all(paddings_auto >= 0)
        assert ops.all(paddings_no_auto >= 0)

    def test_different_colors(self):
        colors = [(0, 0, 0), (255, 255, 255), (128, 64, 192)]
        
        for color in colors:
            layer = Letterbox(color=color)
            letterboxed, _, _ = layer(self.test_inputs_single)
            
            assert len(ops.shape(letterboxed)) == 3
            
            expected_color_norm = [c / 255.0 for c in color]
            assert layer.color_norm == expected_color_norm

    def test_different_batch_sizes(self):
        layer = Letterbox()
        test_batch_sizes = [1, 2, 8, 16]
        
        for batch_size in test_batch_sizes:
            inputs = keras.random.uniform(
                (batch_size, self.height, self.width, self.channels),
                minval=0.0, maxval=1.0
            )
            letterboxed, ratios, paddings = layer(inputs)
            
            letterboxed_shape = ops.shape(letterboxed)
            assert letterboxed_shape[0] == batch_size
            assert len(letterboxed_shape) == 4
            assert ops.shape(ratios) == (batch_size, 2)
            assert ops.shape(paddings) == (batch_size, 2)

    def test_compute_output_shape(self):
        layer = Letterbox(new_shape=(416, 416))
        
        output_shape = layer.compute_output_shape((480, 640, 3))
        assert output_shape == (416, 416, 3)
        
        output_shape = layer.compute_output_shape((4, 480, 640, 3))
        assert output_shape == (4, 416, 416, 3)

    def test_get_config(self):
        layer = Letterbox(
            new_shape=(512, 512),
            color=(100, 150, 200),
            auto=False,
            scaleFill=True,
            scaleup=False,
            stride=16
        )
        
        config = layer.get_config()
        
        assert config["new_shape"] == (512, 512)
        assert config["color"] == (100, 150, 200)
        assert config["auto"] is False
        assert config["scaleFill"] is True
        assert config["scaleup"] is False
        assert config["stride"] == 16
        
        reconstructed_layer = Letterbox.from_config(config)
        assert reconstructed_layer.new_shape == layer.new_shape
        assert reconstructed_layer.color == layer.color
        assert reconstructed_layer.auto == layer.auto
        assert reconstructed_layer.scaleFill == layer.scaleFill
        assert reconstructed_layer.scaleup == layer.scaleup
        assert reconstructed_layer.stride == layer.stride

    def test_aspect_ratio_preservation(self):
        rect_image = keras.random.uniform((300, 600, 3), minval=0.0, maxval=1.0)
        
        layer = Letterbox(new_shape=(640, 640))
        letterboxed, ratios, paddings = layer(rect_image)
        
        assert ops.abs(ratios[0] - ratios[1]) < 1e-6
        
        padding_sum = paddings[0] + paddings[1]
        assert padding_sum >= 0.0

    def test_numpy_input_compatibility(self):
        numpy_input = np.random.uniform(0.0, 1.0, self.single_image_shape).astype(np.float32)
        
        layer = Letterbox()
        letterboxed, ratios, paddings = layer(numpy_input)
        
        assert len(ops.shape(letterboxed)) == 3
        assert ops.shape(ratios) == (2,)
        assert ops.shape(paddings) == (2,)

    def test_edge_case_same_size(self):
        same_size_input = keras.random.uniform((640, 640, 3), minval=0.0, maxval=1.0)
        
        layer = Letterbox(new_shape=(640, 640), auto=False)
        letterboxed, ratios, paddings = layer(same_size_input)
        
        assert len(ops.shape(letterboxed)) == 3
        assert ops.all(ratios > 0)
        assert ops.all(paddings >= 0)

    def test_output_pixel_range(self):
        layer = Letterbox()
        letterboxed, _, _ = layer(self.test_inputs_single)
        
        assert ops.all(letterboxed >= 0.0)
        assert ops.all(letterboxed <= 1.0)

    def test_different_strides(self):
        test_strides = [8, 16, 32, 64]
        
        for stride in test_strides:
            layer = Letterbox(auto=True, stride=stride)
            letterboxed, ratios, paddings = layer(self.test_inputs_single)
            
            assert len(ops.shape(letterboxed)) == 3
            assert ops.all(ratios > 0)
            assert ops.all(paddings >= 0)