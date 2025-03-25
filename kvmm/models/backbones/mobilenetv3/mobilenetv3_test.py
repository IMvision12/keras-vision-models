import pytest

from kvmm.models import mobilenetv3
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestMobileNetV3(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=mobilenetv3.MobileNetV3Small075, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
