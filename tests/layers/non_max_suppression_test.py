import numpy as np
from keras import ops
from keras.src.testing import TestCase

from kvmm.layers import NonMaxSuppression


class TestNonMaxSuppression(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.num_predictions = 8400
        self.num_classes = 80
        self.input_channels = 4 + self.num_classes  # 84
        self.input_shape = (self.batch_size, self.num_predictions, self.input_channels)

        self.test_inputs = self._create_test_predictions()

    def _create_test_predictions(self):
        predictions = np.zeros(self.input_shape, dtype=np.float32)

        for batch_idx in range(self.batch_size):
            predictions[batch_idx, 0, 0] = 50.0
            predictions[batch_idx, 0, 1] = 50.0
            predictions[batch_idx, 0, 2] = 20.0
            predictions[batch_idx, 0, 3] = 20.0
            predictions[batch_idx, 0, 4] = 0.9

            predictions[batch_idx, 1, 0] = 55.0
            predictions[batch_idx, 1, 1] = 55.0
            predictions[batch_idx, 1, 2] = 20.0
            predictions[batch_idx, 1, 3] = 20.0
            predictions[batch_idx, 1, 4] = 0.8

            predictions[batch_idx, 2, 0] = 150.0
            predictions[batch_idx, 2, 1] = 150.0
            predictions[batch_idx, 2, 2] = 30.0
            predictions[batch_idx, 2, 3] = 30.0
            predictions[batch_idx, 2, 5] = 0.85

        return ops.convert_to_tensor(predictions)

    def test_init(self):
        layer = NonMaxSuppression()
        assert layer.conf_threshold == 0.25
        assert layer.iou_threshold == 0.7
        assert layer.max_detections == 300
        assert layer.num_classes == 80

        layer_custom = NonMaxSuppression(
            conf_threshold=0.5, iou_threshold=0.6, max_detections=100, num_classes=20
        )
        assert layer_custom.conf_threshold == 0.5
        assert layer_custom.iou_threshold == 0.6
        assert layer_custom.max_detections == 100
        assert layer_custom.num_classes == 20

    def test_call_basic(self):
        layer = NonMaxSuppression(conf_threshold=0.1, num_classes=self.num_classes)
        outputs = layer(self.test_inputs)

        assert isinstance(outputs, list)
        assert len(outputs) == self.batch_size

        for batch_output in outputs:
            output_shape = ops.shape(batch_output)
            assert len(output_shape) == 2
            assert output_shape[1] == 6

    def test_confidence_filtering(self):
        layer_high_thresh = NonMaxSuppression(
            conf_threshold=0.95, num_classes=self.num_classes
        )
        outputs_high = layer_high_thresh(self.test_inputs)

        layer_low_thresh = NonMaxSuppression(
            conf_threshold=0.1, num_classes=self.num_classes
        )
        outputs_low = layer_low_thresh(self.test_inputs)

        for i in range(self.batch_size):
            high_count = ops.shape(outputs_high[i])[0]
            low_count = ops.shape(outputs_low[i])[0]
            assert low_count >= high_count

    def test_max_detections_limit(self):
        max_det = 5
        layer = NonMaxSuppression(
            conf_threshold=0.1, max_detections=max_det, num_classes=self.num_classes
        )
        outputs = layer(self.test_inputs)

        for batch_output in outputs:
            num_detections = ops.shape(batch_output)[0]
            assert num_detections <= max_det

    def test_empty_results(self):
        low_conf_inputs = ops.ones(self.input_shape) * 0.01

        layer = NonMaxSuppression(conf_threshold=0.5, num_classes=self.num_classes)
        outputs = layer(low_conf_inputs)

        for batch_output in outputs:
            assert ops.shape(batch_output)[0] == 0
            assert ops.shape(batch_output)[1] == 6

    def test_different_batch_sizes(self):
        layer = NonMaxSuppression(conf_threshold=0.1, num_classes=self.num_classes)
        test_batch_sizes = [1, 4, 8]

        for batch_size in test_batch_sizes:
            inputs = (
                ops.ones((batch_size, self.num_predictions, self.input_channels)) * 0.3
            )
            outputs = layer(inputs)

            assert len(outputs) == batch_size
            for batch_output in outputs:
                output_shape = ops.shape(batch_output)
                assert len(output_shape) == 2
                assert output_shape[1] == 6

    def test_different_num_predictions(self):
        layer = NonMaxSuppression(conf_threshold=0.1, num_classes=self.num_classes)
        test_num_preds = [100, 1000, 5000]

        for num_preds in test_num_preds:
            inputs = ops.ones((self.batch_size, num_preds, self.input_channels)) * 0.3
            outputs = layer(inputs)

            assert len(outputs) == self.batch_size
            for batch_output in outputs:
                output_shape = ops.shape(batch_output)
                assert len(output_shape) == 2
                assert output_shape[1] == 6

    def test_different_num_classes(self):
        test_num_classes = [20, 40, 90]

        for num_classes in test_num_classes:
            layer = NonMaxSuppression(conf_threshold=0.1, num_classes=num_classes)
            input_channels = 4 + num_classes
            inputs = ops.ones((self.batch_size, 100, input_channels)) * 0.3
            outputs = layer(inputs)

            assert len(outputs) == self.batch_size
            for batch_output in outputs:
                output_shape = ops.shape(batch_output)
                assert len(output_shape) == 2
                assert output_shape[1] == 6

    def test_iou_threshold_effect(self):
        overlapping_inputs = self._create_overlapping_boxes()

        layer_low_iou = NonMaxSuppression(
            conf_threshold=0.1, iou_threshold=0.3, num_classes=self.num_classes
        )
        outputs_low_iou = layer_low_iou(overlapping_inputs)

        layer_high_iou = NonMaxSuppression(
            conf_threshold=0.1, iou_threshold=0.9, num_classes=self.num_classes
        )
        outputs_high_iou = layer_high_iou(overlapping_inputs)

        for i in range(self.batch_size):
            low_count = ops.shape(outputs_low_iou[i])[0]
            high_count = ops.shape(outputs_high_iou[i])[0]
            assert high_count >= low_count

    def _create_overlapping_boxes(self):
        predictions = np.zeros(
            (self.batch_size, 10, self.input_channels), dtype=np.float32
        )

        for batch_idx in range(self.batch_size):
            for i in range(5):
                predictions[batch_idx, i, 0] = 50.0 + i * 2
                predictions[batch_idx, i, 1] = 50.0 + i * 2
                predictions[batch_idx, i, 2] = 20.0
                predictions[batch_idx, i, 3] = 20.0
                predictions[batch_idx, i, 4] = 0.9 - i * 0.1

        return ops.convert_to_tensor(predictions)

    def test_output_format(self):
        layer = NonMaxSuppression(conf_threshold=0.1, num_classes=self.num_classes)
        outputs = layer(self.test_inputs)

        for batch_output in outputs:
            if ops.shape(batch_output)[0] > 0:
                boxes = batch_output[:, :4]
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

                assert ops.all(x1 <= x2)
                assert ops.all(y1 <= y2)

                confidences = batch_output[:, 4]
                assert ops.all(confidences >= 0.0)
                assert ops.all(confidences <= 1.0)

                class_ids = batch_output[:, 5]
                assert ops.all(class_ids >= 0)
                assert ops.all(class_ids < self.num_classes)

    def test_get_config(self):
        layer = NonMaxSuppression(
            conf_threshold=0.3, iou_threshold=0.6, max_detections=200, num_classes=20
        )
        config = layer.get_config()

        assert "conf_threshold" in config
        assert "iou_threshold" in config
        assert "max_detections" in config
        assert "num_classes" in config

        assert config["conf_threshold"] == 0.3
        assert config["iou_threshold"] == 0.6
        assert config["max_detections"] == 200
        assert config["num_classes"] == 20

        reconstructed_layer = NonMaxSuppression.from_config(config)
        assert reconstructed_layer.conf_threshold == layer.conf_threshold
        assert reconstructed_layer.iou_threshold == layer.iou_threshold
        assert reconstructed_layer.max_detections == layer.max_detections
        assert reconstructed_layer.num_classes == layer.num_classes

    def test_xywh_to_xyxy_conversion(self):
        simple_inputs = np.zeros((1, 1, self.input_channels), dtype=np.float32)
        simple_inputs[0, 0, 0] = 100.0
        simple_inputs[0, 0, 1] = 100.0
        simple_inputs[0, 0, 2] = 20.0
        simple_inputs[0, 0, 3] = 20.0
        simple_inputs[0, 0, 4] = 0.9
        simple_inputs = ops.convert_to_tensor(simple_inputs)

        layer = NonMaxSuppression(conf_threshold=0.1, num_classes=self.num_classes)
        outputs = layer(simple_inputs)

        if ops.shape(outputs[0])[0] > 0:
            box = outputs[0][0, :4]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            self.assertAllClose(x1, 90.0, atol=1e-3)
            self.assertAllClose(y1, 90.0, atol=1e-3)
            self.assertAllClose(x2, 110.0, atol=1e-3)
            self.assertAllClose(y2, 110.0, atol=1e-3)

    def test_class_selection(self):
        class_test_inputs = np.zeros((1, 1, self.input_channels), dtype=np.float32)
        class_test_inputs[0, 0, 0] = 100.0
        class_test_inputs[0, 0, 1] = 100.0
        class_test_inputs[0, 0, 2] = 20.0
        class_test_inputs[0, 0, 3] = 20.0

        class_test_inputs[0, 0, 4 + 5] = 0.9
        class_test_inputs[0, 0, 4 + 3] = 0.7
        class_test_inputs = ops.convert_to_tensor(class_test_inputs)

        layer = NonMaxSuppression(conf_threshold=0.1, num_classes=self.num_classes)
        outputs = layer(class_test_inputs)

        if ops.shape(outputs[0])[0] > 0:
            predicted_class = outputs[0][0, 5]
            self.assertAllClose(predicted_class, 5.0, atol=1e-3)

    def test_deterministic_behavior(self):
        layer = NonMaxSuppression(conf_threshold=0.1, num_classes=self.num_classes)

        outputs1 = layer(self.test_inputs)
        outputs2 = layer(self.test_inputs)

        assert len(outputs1) == len(outputs2)
        for i in range(len(outputs1)):
            self.assertAllClose(outputs1[i], outputs2[i], atol=1e-6)
