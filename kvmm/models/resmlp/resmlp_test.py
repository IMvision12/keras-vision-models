import pytest

from kvmm.models.resmlp import ResMLP12
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestResMLP(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=ResMLP12, input_shape=(224, 224, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
