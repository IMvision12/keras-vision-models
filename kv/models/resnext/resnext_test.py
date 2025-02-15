import pytest

from kv.models.resnext import ResNeXt50_32x4d
from kv.tests.test_modeling import BaseVisionTest, ModelConfig


class TestResNeXt(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=ResNeXt50_32x4d, input_shape=(224, 224, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
