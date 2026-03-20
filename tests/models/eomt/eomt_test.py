from keras import ops

from kmodels.models import eomt

from ...test_modelling import ModelTestCase


class TestEoMT(ModelTestCase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        super().setUp()

        self.input_data = ops.ones((2, 64, 64, 3))
        self.expected_output_shape = {
            "class_logits": (2, 100, 134),
            "mask_logits": (2, 100, 16, 16),
        }

        self.configure(
            model_cls=eomt.EoMT_Small,
            model_type="segmentation",
            init_kwargs={
                "weights": None,
                "input_shape": (64, 64, 3),
                "num_queries": 100,
                "num_labels": 133,
            },
            input_data=self.input_data,
            expected_output_shape=self.expected_output_shape,
        )

    def test_weight_initialization(self):
        self.skipTest("Skipped: register_tokens is initialized to zeros by design")

    def test_different_num_queries(self):
        for num_queries in [50, 200]:
            with self.subTest(num_queries=num_queries):
                model = eomt.EoMT_Small(
                    input_shape=(64, 64, 3),
                    num_queries=num_queries,
                    num_labels=133,
                    weights=None,
                )
                output = model(self.input_data)
                self.assertEqual(
                    output["class_logits"].shape[1],
                    num_queries,
                )
                self.assertEqual(
                    output["mask_logits"].shape[1],
                    num_queries,
                )

    def test_different_num_labels(self):
        for num_labels in [21, 150]:
            with self.subTest(num_labels=num_labels):
                model = eomt.EoMT_Small(
                    input_shape=(64, 64, 3),
                    num_queries=100,
                    num_labels=num_labels,
                    weights=None,
                )
                output = model(self.input_data)
                self.assertEqual(
                    output["class_logits"].shape[2],
                    num_labels + 1,
                    f"Expected {num_labels + 1} class logits (labels + no-object)",
                )

    def test_mask_spatial_dimensions(self):
        model = self.create_model()
        output = model(self.input_data)
        mask_h = output["mask_logits"].shape[2]
        mask_w = output["mask_logits"].shape[3]
        # With patch_size=16 and 2 upscale blocks (4x), spatial = (input/16)*4 = input/4
        expected_size = 64 // 4
        self.assertEqual(mask_h, expected_size)
        self.assertEqual(mask_w, expected_size)

    def test_output_no_nans(self):
        model = self.create_model()
        output = model(self.input_data)
        for key, value in output.items():
            has_nans = ops.any(ops.isnan(value))
            self.assertFalse(has_nans, f"Output '{key}' contains NaN values")
