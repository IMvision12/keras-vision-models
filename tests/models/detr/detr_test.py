from keras import ops

from kmodels.models import detr

from ...test_modelling import ModelTestCase


class TestDETR(ModelTestCase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        super().setUp()

        self.input_data = ops.ones((2, 32, 32, 3))
        self.expected_output_shape = {
            "logits": (2, 10, 92),
            "pred_boxes": (2, 10, 4),
        }

        self.configure(
            model_cls=detr.DETRResNet50,
            model_type="object_detection",
            init_kwargs={
                "weights": None,
                "input_shape": (32, 32, 3),
                "num_classes": 92,
                "num_queries": 10,
                "include_normalization": False,
            },
            input_data=self.input_data,
            expected_output_shape=self.expected_output_shape,
        )

    def test_weight_initialization(self):
        custom_model = detr.DETRResNet50(
            input_shape=(32, 32, 3), weights="coco", num_classes=92
        )
        return super().test_weight_initialization(custom_model)

    def test_different_num_queries(self):
        for num_queries in [5, 20]:
            with self.subTest(num_queries=num_queries):
                model = detr.DETRResNet50(
                    input_shape=(32, 32, 3),
                    num_classes=92,
                    num_queries=num_queries,
                    weights=None,
                )
                output = model(self.input_data)
                self.assertEqual(output["logits"].shape[1], num_queries)
                self.assertEqual(output["pred_boxes"].shape[1], num_queries)

    def test_different_num_classes(self):
        for num_classes in [10, 50]:
            with self.subTest(num_classes=num_classes):
                model = detr.DETRResNet50(
                    input_shape=(32, 32, 3),
                    num_classes=num_classes,
                    num_queries=10,
                    weights=None,
                )
                output = model(self.input_data)
                self.assertEqual(output["logits"].shape[2], num_classes)

    def test_pred_boxes_range(self):
        model = self.create_model()
        output = model(self.input_data)
        boxes = output["pred_boxes"]
        self.assertTrue(
            ops.all(boxes >= 0) and ops.all(boxes <= 1),
            "Predicted boxes should be in [0, 1] range (normalized)",
        )
