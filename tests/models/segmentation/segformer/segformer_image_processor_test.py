import os
import tempfile

import keras
import numpy as np
import pytest
from PIL import Image

from kvmm.models import segformer


class TestSegFormerImageProcessor:
    @pytest.fixture
    def sample_image_array(self):
        return np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_image_pil(self, sample_image_array):
        return Image.fromarray(sample_image_array)

    @pytest.fixture
    def sample_image_path(self, sample_image_pil):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name
            sample_image_pil.save(temp_path)
        yield temp_path
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_basic_functionality(self, sample_image_array):
        result = segformer.SegFormerImageProcessor(sample_image_array)
        result_np = keras.ops.convert_to_numpy(result)

        assert len(result_np.shape) == 4
        assert result_np.shape[0] == 1  # Batch dimension
        assert result_np.shape[1:3] == (512, 512)  # Height, Width
        assert result_np.shape[3] == 3  # RGB channels

    def test_input_types(self, sample_image_array, sample_image_pil, sample_image_path):
        result_array = segformer.SegFormerImageProcessor(
            sample_image_array, return_tensor=False
        )
        result_pil = segformer.SegFormerImageProcessor(
            sample_image_pil, return_tensor=False
        )
        result_path = segformer.SegFormerImageProcessor(
            sample_image_path, return_tensor=False
        )
        assert result_array.shape == result_pil.shape == result_path.shape

    def test_resize_parameters(self, sample_image_array):
        custom_size = {"height": 256, "width": 300}
        result = segformer.SegFormerImageProcessor(
            sample_image_array, size=custom_size, return_tensor=False
        )

        assert result.shape[1:3] == (256, 300)

        result_no_resize = segformer.SegFormerImageProcessor(
            sample_image_array, do_resize=False, return_tensor=False
        )

        assert result_no_resize.shape[1:3] == (10, 10)

    def test_resample_methods(self, sample_image_array):
        for method in ["nearest", "bilinear", "bicubic"]:
            result = segformer.SegFormerImageProcessor(
                sample_image_array, resample=method, return_tensor=False
            )
            assert result.shape[1:3] == (512, 512)

    def test_rescale_options(self, sample_image_array):
        result_rescale = segformer.SegFormerImageProcessor(
            sample_image_array, do_normalize=False, return_tensor=False
        )

        assert np.max(result_rescale) <= 1.0
        assert np.min(result_rescale) >= 0.0

        result_no_rescale = segformer.SegFormerImageProcessor(
            sample_image_array,
            do_rescale=False,
            do_normalize=False,
            return_tensor=False,
        )

        assert np.max(result_no_rescale) > 1.0

        custom_factor = 1 / 100
        result_custom = segformer.SegFormerImageProcessor(
            sample_image_array,
            rescale_factor=custom_factor,
            do_normalize=False,
            return_tensor=False,
        )

        expected = sample_image_array.astype(float) * custom_factor
        expected = np.expand_dims(expected, 0)
        expected = keras.ops.image.resize(
            keras.ops.convert_to_tensor(expected, dtype="float32"),
            size=(512, 512),
            interpolation="bilinear",
        )
        expected = keras.ops.convert_to_numpy(expected)

        np.testing.assert_allclose(result_custom, expected, rtol=1e-4, atol=1e-6)

    def test_normalization(self, sample_image_array):
        result_default = segformer.SegFormerImageProcessor(
            sample_image_array, return_tensor=False
        )

        custom_mean = (0.5, 0.5, 0.5)
        custom_std = (0.1, 0.1, 0.1)
        result_custom = segformer.SegFormerImageProcessor(
            sample_image_array,
            image_mean=custom_mean,
            image_std=custom_std,
            return_tensor=False,
        )

        result_no_norm = segformer.SegFormerImageProcessor(
            sample_image_array, do_normalize=False, return_tensor=False
        )

        assert not np.allclose(result_default, result_custom)

        assert not np.allclose(result_default, result_no_norm)
        assert not np.allclose(result_custom, result_no_norm)

    def test_return_types(self, sample_image_array):
        result_tensor = segformer.SegFormerImageProcessor(
            sample_image_array, return_tensor=True
        )
        tensor_np = keras.ops.convert_to_numpy(result_tensor)
        assert isinstance(tensor_np, np.ndarray)

        result_numpy = segformer.SegFormerImageProcessor(
            sample_image_array, return_tensor=False
        )
        assert isinstance(result_numpy, np.ndarray)

    def test_float_input(self):
        float_image = np.random.random((10, 10, 3)).astype(np.float32)
        result = segformer.SegFormerImageProcessor(float_image, return_tensor=False)
        assert result.shape[1:3] == (512, 512)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            segformer.SegFormerImageProcessor("nonexistent_file.jpg")

        with pytest.raises(ValueError):
            segformer.SegFormerImageProcessor(np.zeros((10, 10)))

        with pytest.raises(TypeError):
            segformer.SegFormerImageProcessor(123)

        with pytest.raises(ValueError):
            segformer.SegFormerImageProcessor(np.ones((10, 10, 3)) * 2.0)

    def test_4d_input(self):
        batch_image = np.random.randint(0, 256, size=(1, 10, 10, 3), dtype=np.uint8)
        result = segformer.SegFormerImageProcessor(batch_image, return_tensor=False)
        assert result.shape[0] == 1
