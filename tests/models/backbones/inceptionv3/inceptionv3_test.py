from kvmm.models import inceptionv3

from ....test_backbone_modeling import BackboneTestCase


class TestInceptionV3(BackboneTestCase):
    """Test case for the InceptionV3 model."""

    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(model_cls=inceptionv3.InceptionV3, input_shape=(75, 75, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }

    def test_weight_loading(self):
        custom_model = inceptionv3.InceptionV3(
            input_shape=(75, 75, 3),
        )
        return super().test_weight_loading(custom_model)
