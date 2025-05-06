from kvmm.models import segformer
from keras import ops
from ....test_modelling import ModelTestCase

class TestSegFormer(ModelTestCase):
    __test__ = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setUp(self):
        super().setUp()

        self.input_data = ops.ones((2, 32, 32, 3))
        self.expected_output_shape = (2, 32, 32, 150)
        self.configure(
            model_cls=segformer.SegFormerB0,
            model_type="segmentation",
            init_kwargs={
                "weights": None,
                "input_shape": (32, 32, 3),
                "num_classes": 150
            },
            input_data=self.input_data,
            expected_output_shape=self.expected_output_shape
        )
    
    def test_weight_initialization(self):
        custom_model = segformer.SegFormerB0(
            input_shape=(32, 32, 3),
            weights="ade20k_512",
            num_classes=150
        )
        return super().test_weight_initialization(custom_model)