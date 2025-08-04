from keras import ops

from kvmm.models import inceptionv4

from ...test_modelling import ModelTestCase


class TestInceptionV4(ModelTestCase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        super().setUp()

        self.input_data = ops.ones((2, 75, 75, 3))
        self.expected_output_shape = (2, 1000)

        self.configure(
            model_cls=inceptionv4.InceptionV4,
            model_type="backbone",
            init_kwargs={
                "weights": None,
                "input_shape": (75, 75, 3),
                "include_top": True,
            },
            input_data=self.input_data,
            expected_output_shape=self.expected_output_shape,
        )

    def test_weight_initialization(self):
        custom_model = inceptionv4.InceptionV4(
            input_shape=(75, 75, 3),
        )
        return super().test_weight_initialization(custom_model)
