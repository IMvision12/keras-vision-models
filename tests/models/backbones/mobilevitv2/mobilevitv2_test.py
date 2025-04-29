from kvmm.models import mobilevitv2

from ....test_backbone_modeling import BackboneTestCase


class TestMobileViTV2(BackboneTestCase):
    """Test case for the MobileViTV2 model."""

    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(model_cls=mobilevitv2.MobileViTV2M050, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }

    def test_weight_loading(self):
        custom_model = mobilevitv2.MobileViTV2M050(
            input_shape=(32, 32, 3),
        )
        return super().test_weight_loading(custom_model)
