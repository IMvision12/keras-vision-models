from kvmm.models import resmlp

from ....test_backbone_modeling import BackboneTestCase


class TestResMLP(BackboneTestCase):
    """Test case for the ResMLP model."""

    __test__ = True

    def setUp(self):
        super().setUp()
        self.configure(model_cls=resmlp.ResMLP12, input_shape=(32, 32, 3))

    def get_default_kwargs(self) -> dict:
        return {
            "include_normalization": True,
            "normalization_mode": "imagenet",
            "classifier_activation": "softmax",
            "weights": None,
        }

    def test_weight_loading(self):
        custom_model = resmlp.ResMLP12(
            input_shape=(224, 224, 3),
        )
        return super().test_weight_loading(custom_model)

from kvmm.models import resmlp
from keras import ops
from ....test_modelling import ModelTestCase

class TestResMLP(ModelTestCase):
    __test__ = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setUp(self):
        super().setUp()
        
        self.input_data = ops.ones((2, 32, 32, 3))
        self.expected_output_shape = (2, 1000)
        
        self.configure(
            model_cls=resmlp.ResMLP12,
            model_type="backbone",
            init_kwargs={
                "weights": None,
                "input_shape": (32, 32, 3),
                "include_top": True,
            },
            input_data=self.input_data,
            expected_output_shape=self.expected_output_shape
        )
    
    def test_weight_initialization(self):
        custom_model = resmlp.ResMLP12(
            input_shape=(224, 224, 3),
        )
        return super().test_weight_initialization(custom_model)