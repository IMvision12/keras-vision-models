import pytest

from kvmm.models.efficientnetv2 import EfficientNetV2S
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestEfficientNetV2(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=EfficientNetV2S, input_shape=(384, 384, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
