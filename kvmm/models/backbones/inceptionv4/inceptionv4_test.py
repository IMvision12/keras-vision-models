import pytest

from kvmm.models import inceptionv4
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestInceptionV4(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=inceptionv4.InceptionV4, input_shape=(75, 75, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
