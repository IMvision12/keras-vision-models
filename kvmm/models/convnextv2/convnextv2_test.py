import pytest

from kvmm.models.convnextv2 import ConvNeXtV2Atto
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestConvNeXtV2(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=ConvNeXtV2Atto, input_shape=(224, 224, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
