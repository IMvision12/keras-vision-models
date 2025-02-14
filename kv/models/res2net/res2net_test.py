import pytest

from kv.models.res2net import Res2Net50_26w_4s
from kv.tests.test_modeling import BaseVisionTest, ModelConfig


class TestRes2Net(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=Res2Net50_26w_4s, input_shape=(224, 224, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
