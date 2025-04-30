from kvmm.models import res2net

from ....test_backbone_modeling import BackboneTestCase


class TestRes2Net(BackboneTestCase):
    """Test case for the Res2Net model."""

    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(model_cls=res2net.Res2Net50_26w_4s, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }

    def test_weight_loading(self):
        custom_model = res2net.Res2Net50_26w_4s(
            input_shape=(32, 32, 3),
        )
        return super().test_weight_loading(custom_model)
