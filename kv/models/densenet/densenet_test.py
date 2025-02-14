import pytest

from kv.models.densenet import DenseNet121
from kv.tests.test_modeling import BaseVisionTest, ModelConfig


class TestDenseNet(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=DenseNet121, input_shape=(224, 224, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
