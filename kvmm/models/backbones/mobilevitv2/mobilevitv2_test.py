import pytest

from kvmm.models import mobilevitv2
from kvmm.tests.test_modeling import BaseVisionTest, ModelConfig


class TestMobileViTV2(BaseVisionTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            model_cls=mobilevitv2.MobileViTV2M050, input_shape=(32, 32, 3)
        )

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
