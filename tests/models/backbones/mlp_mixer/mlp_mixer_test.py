from kvmm.models import mlp_mixer

from ....test_backbone_modeling import BackboneTestCase


class TestMLPMixer(BackboneTestCase):
    """Test case for the MLPMixer model."""

    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(model_cls=mlp_mixer.MLPMixerB16, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }

    def test_weight_loading(self):
        custom_model = mlp_mixer.MLPMixerB16(
            input_shape=(224, 224, 3),
        )
        return super().test_weight_loading(custom_model)
