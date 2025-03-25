import pytest

from kvmm.models import resnetv2
from kvmm.tests.test_backbone_modeling import BackboneTest, ModelConfig


class TestResNetV2(BackboneTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=resnetv2.ResNetV2_50x1, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
