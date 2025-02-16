import pytest

from kvmm.models.inceptionv4 import InceptionV4
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestInceptionV4(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=InceptionV4, input_shape=(299, 299, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
