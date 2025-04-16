import pytest

from kvmm.models import convmixer

from ....test_backbone_modeling import BackboneTest, ModelConfig


class TestConvMixer(BackboneTest):
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(model_cls=convmixer.ConvMixer768D32, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
