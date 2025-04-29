from kvmm.models import convmixer

from ....test_backbone_modeling import BackboneTestCase


class TestConvMixer(BackboneTestCase):
    """Test case for the ConvMixer model."""

    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(model_cls=convmixer.ConvMixer768D32, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }

    def test_weight_loading(self):
        custom_model = convmixer.ConvMixer768D32(
            input_shape=(32, 32, 3),
        )
        return super().test_weight_loading(custom_model)
