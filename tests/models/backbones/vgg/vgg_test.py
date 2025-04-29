from kvmm.models import vgg
from ....test_backbone_modeling import BackboneTestCase


class TestVGG(BackboneTestCase):
    """Test case for the VGG model."""
    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(
            model_cls=vgg.VGG16, 
            input_shape=(224, 224, 3)
        )
    
    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }
    
    def test_weight_loading(self):
        custom_model = vgg.VGG16(
            input_shape=(224, 224, 3),
        )
        return super().test_weight_loading(custom_model)