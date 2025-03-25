import pytest

from kvmm.models import resnext
from kvmm.tests.test_backbone_modeling import BackboneTest, ModelConfig


class TestResNeXt(BackboneTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=resnext.ResNeXt50_32x4d, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
