import pytest

from kvmm.models.resnetv2 import ResNetV2_50x1
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestResNetV2(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=ResNetV2_50x1, input_shape=(224, 224, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
