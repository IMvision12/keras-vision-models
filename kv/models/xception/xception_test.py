import pytest

from kv.models.xception import Xception
from kv.tests.test_modeling import BaseVisionTest, ModelConfig


class TestXception(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=Xception, input_shape=(299, 299, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
