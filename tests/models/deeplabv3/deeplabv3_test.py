from keras import ops

from kmodels.models import deeplabv3

from ...test_modelling import ModelTestCase


class TestDeepLabV3(ModelTestCase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        super().setUp()

        self.input_data = ops.ones((2, 64, 64, 3))
        self.expected_output_shape = (2, 64, 64, 21)

        self.configure(
            model_cls=deeplabv3.DeepLabV3ResNet50,
            model_type="segmentation",
            init_kwargs={
                "weights": None,
                "input_shape": (64, 64, 3),
                "num_classes": 21,
            },
            input_data=self.input_data,
            expected_output_shape=self.expected_output_shape,
        )

    def test_weight_initialization(self):
        self.skipTest("Skipped: BatchNorm beta is initialized to zeros by design")

    def test_different_num_classes(self):
        for num_classes in [10, 150]:
            with self.subTest(num_classes=num_classes):
                model = deeplabv3.DeepLabV3ResNet50(
                    input_shape=(64, 64, 3),
                    num_classes=num_classes,
                    weights=None,
                )
                output = model(self.input_data)
                self.assertEqual(
                    output.shape[-1],
                    num_classes,
                    f"Expected {num_classes} output channels",
                )

    def test_output_spatial_matches_input(self):
        model = self.create_model()
        output = model(self.input_data)
        self.assertEqual(
            output.shape[1:3],
            (64, 64),
            "Output spatial dims should match input",
        )

    def test_resnet101_variant(self):
        model = deeplabv3.DeepLabV3ResNet101(
            input_shape=(64, 64, 3),
            num_classes=21,
            weights=None,
        )
        output = model(self.input_data)
        self.assertEqual(output.shape, (2, 64, 64, 21))
        self.assertGreater(
            model.count_params(),
            50_000_000,
            "ResNet101 variant should have more than 50M params",
        )
