import pytest

from kvmm.models.senet import SEResNeXt50_32x4d
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestSeNet(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=SEResNeXt50_32x4d, input_shape=(224, 224, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
