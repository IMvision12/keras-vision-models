from kvmm.models import inception_resnetv2
from ....test_backbone_modeling import BackboneTestCase


class TestInceptionResNetV2(BackboneTestCase):
    """Test case for the InceptionResNetV2 model."""
    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(
            model_cls=inception_resnetv2.InceptionResNetV2, 
            input_shape=(75, 75, 3)
        )
    
    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
    
    def test_weight_loading(self):
        custom_model = inception_resnetv2.InceptionResNetV2(
            input_shape=(75, 75, 3),
        )
        return super().test_weight_loading(custom_model)