import pytest

from kvmm.models.inceptionv3 import InceptionV3
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestInceptionV3(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=InceptionV3, input_shape=(299, 299, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
