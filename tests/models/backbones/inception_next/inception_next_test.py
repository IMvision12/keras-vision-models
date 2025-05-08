from kvmm.models import inception_next
from keras import ops
from ....test_modelling import ModelTestCase

class TestInceptionNeXt(ModelTestCase):
    __test__ = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setUp(self):
        super().setUp()
        
        self.input_data = ops.ones((2, 32, 32, 3))
        self.expected_output_shape = (2, 1000)
        
        self.configure(
            model_cls=inception_next.InceptionNeXtTiny,
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
        custom_model = inception_next.InceptionNeXtTiny(
            input_shape=(32, 32, 3),
        )
        return super().test_weight_initialization(custom_model)