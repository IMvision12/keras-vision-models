import json
import os
import tempfile
from typing import Any, Dict, Tuple, Type, Union

import keras
import tensorflow as tf
from keras import Model, ops
from keras.src.testing import TestCase


class ModelTestCase(TestCase):
    __test__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default attributes
        self.model_cls = None
        self.model_type = None
        self.init_kwargs = {}
        self.input_data = None
        self.expected_output_shape = None
        self.batch_size = 2

    def configure(
        self,
        model_cls: Type[Model],
        model_type: str,
        init_kwargs: Dict[str, Any] = None,
        input_data: Union[Dict[str, Any], keras.KerasTensor] = None,
        expected_output_shape: Union[Dict[str, Tuple], Tuple] = None,
        batch_size: int = 2,
    ):
        self.model_cls = model_cls
        self.model_type = model_type.lower()
        self.init_kwargs = init_kwargs or {}
        self.input_data = input_data
        self.expected_output_shape = expected_output_shape
        self.batch_size = batch_size

        return self

    def create_model(self, **kwargs: Any) -> Model:
        self.assertIsNotNone(
            self.model_cls, "Model class not configured. Call configure() first."
        )

        combined_kwargs = self.init_kwargs.copy()
        combined_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        return self.model_cls(**combined_kwargs)

    def get_input_data(self, **kwargs) -> Union[Dict[str, Any], keras.KerasTensor]:
        self.assertIsNotNone(
            self.input_data, "Input data not configured. Call configure() first."
        )
        return self.input_data

    def convert_data_format(
        self, data: keras.KerasTensor, to_format: str
    ) -> keras.KerasTensor:
        if not isinstance(data, (keras.KerasTensor, tf.Tensor)):
            return data

        if len(data.shape) == 4:
            if to_format == "channels_first":
                return ops.transpose(data, (0, 3, 1, 2))
            return ops.transpose(data, (0, 2, 3, 1))
        elif len(data.shape) == 3:
            if to_format == "channels_first":
                return ops.transpose(data, (2, 0, 1))
            return ops.transpose(data, (1, 2, 0))
        return data

    def convert_dict_data_format(self, data_dict, to_format):
        if not isinstance(data_dict, dict):
            return self.convert_data_format(data_dict, to_format)

        result = {}
        for key, value in data_dict.items():
            if (
                isinstance(value, (keras.KerasTensor, tf.Tensor))
                and len(value.shape) == 4
            ):
                if (value.shape[-1] <= 4 and value.shape[-1] >= 1) or (
                    value.shape[1] <= 4 and value.shape[1] >= 1
                ):
                    result[key] = self.convert_data_format(value, to_format)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def check_output_shape(self, output, expected_shape):
        if expected_shape is None:
            return

        if isinstance(output, dict) and isinstance(expected_shape, dict):
            for key, shape in expected_shape.items():
                self.assertIn(
                    key,
                    output,
                    f"Expected output key '{key}' not found in model output",
                )
                self.assertEqual(
                    tuple(output[key].shape),
                    shape,
                    f"Output shape mismatch for '{key}'. Expected {shape}, got {output[key].shape}",
                )
        elif isinstance(output, list) and isinstance(expected_shape, list):
            self.assertEqual(
                len(output),
                len(expected_shape),
                f"Expected {len(expected_shape)} outputs, got {len(output)}",
            )
            for i, (out, shape) in enumerate(zip(output, expected_shape)):
                self.assertEqual(
                    tuple(out.shape),
                    shape,
                    f"Output {i} shape mismatch. Expected {shape}, got {out.shape}",
                )
        elif isinstance(output, list) and not isinstance(expected_shape, list):
            main_output = output[-1]
            self.assertEqual(
                tuple(main_output.shape),
                expected_shape,
                f"Output shape mismatch. Expected {expected_shape}, got {main_output.shape}",
            )
        else:
            self.assertEqual(
                tuple(output.shape),
                expected_shape,
                f"Output shape mismatch. Expected {expected_shape}, got {output.shape}",
            )

    def test_model_creation(self):
        model = self.create_model()
        self.assertIsInstance(model, Model)

    def test_weight_initialization(self, model=None):
        self.assertIsNotNone(model, "Model not provided for weight initialization test")
        self.assertIsNotNone(model.weights, "Model weights not initialized")
        self.assertGreater(
            len(model.trainable_weights), 0, "Model has no trainable weights"
        )

        for weight in model.weights:
            has_nans = ops.any(ops.isnan(weight))
            self.assertFalse(has_nans, f"Weight '{weight.name}' contains NaN values")

            is_all_zeros = ops.all(ops.equal(weight, 0))
            self.assertFalse(
                is_all_zeros,
                f"Weight '{weight.name}' contains all zeros, suggesting improper initialization",
            )

    def test_model_forward_pass(self):
        model = self.create_model()
        input_data = self.get_input_data()
        output = model(input_data)

        if self.expected_output_shape is not None:
            self.check_output_shape(output, self.expected_output_shape)

    def test_data_formats(self):
        if isinstance(self.input_data, dict) and len(self.input_data) > 1:
            self.skipTest(
                "Data format test not implemented for complex dictionary inputs"
            )

        original_data_format = keras.config.image_data_format()
        input_data = self.get_input_data()

        try:
            keras.config.set_image_data_format("channels_last")
            model_last = self.create_model()
            output_last = model_last(input_data)

            if (
                keras.config.backend() == "tensorflow"
                and tf.config.list_physical_devices("GPU")
            ):
                keras.config.set_image_data_format("channels_first")

                if isinstance(input_data, dict):
                    current_data = self.convert_dict_data_format(
                        input_data, "channels_first"
                    )
                else:
                    current_data = self.convert_data_format(
                        input_data, "channels_first"
                    )

                model_first = self.create_model()

                try:
                    model_first.set_weights(model_last.get_weights())
                except ValueError:
                    pass

                output_first = model_first(current_data)

                if (
                    not isinstance(output_first, dict)
                    and not isinstance(output_last, dict)
                    and not isinstance(output_first, list)
                    and not isinstance(output_last, list)
                    and len(output_first.shape) == len(output_last.shape)
                ):
                    if len(output_first.shape) == 4:
                        output_first_converted = self.convert_data_format(
                            output_first, "channels_last"
                        )
                        try:
                            self.assertAllClose(
                                output_first_converted,
                                output_last,
                                rtol=1e-5,
                                atol=1e-5,
                            )
                        except AssertionError:
                            pass
        finally:
            keras.config.set_image_data_format(original_data_format)

    def test_model_saving(self):
        model = self.create_model()
        input_data = self.get_input_data()
        original_output = model(input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, f"test_{self.model_type}_model.keras")

            model.save(save_path)
            loaded_model = keras.models.load_model(save_path)

            self.assertIsInstance(
                loaded_model,
                model.__class__,
                f"Loaded model should be an instance of {model.__class__.__name__}",
            )

            loaded_output = loaded_model(input_data)

            if isinstance(original_output, dict) and isinstance(loaded_output, dict):
                for key in original_output:
                    self.assertIn(key, loaded_output)
                    self.assertAllClose(
                        original_output[key],
                        loaded_output[key],
                        rtol=1e-5,
                        atol=1e-5,
                    )
            elif isinstance(original_output, list) and isinstance(loaded_output, list):
                self.assertEqual(len(original_output), len(loaded_output))
                for orig, loaded in zip(original_output, loaded_output):
                    self.assertAllClose(orig, loaded, rtol=1e-5, atol=1e-5)
            else:
                self.assertAllClose(
                    original_output, loaded_output, rtol=1e-5, atol=1e-5
                )

    def test_serialization(self):
        model = self.create_model()

        cfg = model.get_config()
        cfg_json = json.dumps(cfg, sort_keys=True, indent=4)

        cls = model.__class__
        revived_instance = cls.from_config(cfg)
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)

        self.assertEqual(
            cfg_json,
            revived_cfg_json,
            "Config JSON mismatch after from_config roundtrip",
        )

        serialized = keras.saving.serialize_keras_object(model)
        serialized_json = json.dumps(serialized, sort_keys=True, indent=4)
        revived_instance = keras.saving.deserialize_keras_object(
            json.loads(serialized_json)
        )
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)

        self.assertEqual(
            cfg_json,
            revived_cfg_json,
            "Config JSON mismatch after full serialization roundtrip",
        )

    def test_training_mode(self):
        model = self.create_model()
        model.trainable = True
        self.assertTrue(model.trainable)

        input_data = self.get_input_data()

        training_output = model(input_data, training=True)
        inference_output = model(input_data, training=False)

        if isinstance(training_output, dict) and isinstance(inference_output, dict):
            self.assertEqual(set(training_output.keys()), set(inference_output.keys()))
            for key in training_output:
                self.assertEqual(
                    training_output[key].shape, inference_output[key].shape
                )
        elif isinstance(training_output, list) and isinstance(inference_output, list):
            self.assertEqual(len(training_output), len(inference_output))
            for train_out, infer_out in zip(training_output, inference_output):
                self.assertEqual(train_out.shape, infer_out.shape)
        else:
            self.assertEqual(training_output.shape, inference_output.shape)

    def test_backbone_features(self):
        if self.model_type != "backbone":
            self.skipTest("This test is only for backbone models")

        backbone_kwargs = self.init_kwargs.copy()
        if "include_top" in backbone_kwargs:
            backbone_kwargs["include_top"] = False
        backbone_kwargs["as_backbone"] = True

        model = self.create_model(**backbone_kwargs)
        input_data = self.get_input_data()
        features = model(input_data)

        self.assertIsInstance(
            features, list, "Backbone output should be a list of feature maps"
        )
        self.assertGreaterEqual(
            len(features), 1, "Backbone should output at least 1 feature map"
        )

        for i, feature in enumerate(features):
            if len(feature.shape) == 3:
                self.assertEqual(
                    feature.shape[0],
                    input_data.shape[0],
                    "Batch dimension mismatch",
                )
            elif len(feature.shape) == 4:
                self.assertEqual(
                    feature.shape[0],
                    input_data.shape[0],
                    "Batch dimension mismatch",
                )

                if i > 0 and len(features[i - 1].shape) == 4:
                    prev_h_idx = (
                        1 if keras.config.image_data_format() == "channels_last" else 2
                    )
                    prev_w_idx = (
                        2 if keras.config.image_data_format() == "channels_last" else 3
                    )
                    curr_h_idx = (
                        1 if keras.config.image_data_format() == "channels_last" else 2
                    )
                    curr_w_idx = (
                        2 if keras.config.image_data_format() == "channels_last" else 3
                    )

                    prev_h, prev_w = (
                        features[i - 1].shape[prev_h_idx],
                        features[i - 1].shape[prev_w_idx],
                    )
                    curr_h, curr_w = (
                        feature.shape[curr_h_idx],
                        feature.shape[curr_w_idx],
                    )

                    self.assertLessEqual(
                        curr_h,
                        prev_h,
                        f"Feature map {i} height should be <= previous feature map height",
                    )
                    self.assertLessEqual(
                        curr_w,
                        prev_w,
                        f"Feature map {i} width should be <= previous feature map width",
                    )

    def test_different_input_sizes(self):
        if self.model_type != "segmentation":
            self.skipTest("This test is only for segmentation models")

        input_data = self.get_input_data()
        self.assertFalse(
            isinstance(input_data, dict),
            "Input size test not implemented for dictionary inputs",
        )
        self.assertEqual(
            len(input_data.shape),
            4,
            "Input must be a 4D tensor (batch, height, width, channels)",
        )

        if keras.config.image_data_format() == "channels_last":
            height_idx, width_idx, channel_idx = 1, 2, 3
        else:
            channel_idx, height_idx, width_idx = 1, 2, 3

        original_height = input_data.shape[height_idx]
        original_width = input_data.shape[width_idx]
        channels = input_data.shape[channel_idx]

        larger_height = original_height + 32
        larger_width = original_width + 32

        if keras.config.image_data_format() == "channels_last":
            larger_shape = (larger_height, larger_width, channels)
            larger_input = keras.random.uniform(
                (self.batch_size, larger_height, larger_width, channels),
                dtype="float32",
            )
        else:
            larger_shape = (channels, larger_height, larger_width)
            larger_input = keras.random.uniform(
                (self.batch_size, channels, larger_height, larger_width),
                dtype="float32",
            )

        model_kwargs = self.init_kwargs.copy()
        if "input_shape" in model_kwargs:
            model_kwargs["input_shape"] = larger_shape

        larger_model = self.create_model(**model_kwargs)
        larger_output = larger_model(larger_input)

        if isinstance(larger_output, list):
            main_output = larger_output[-1]
        else:
            main_output = larger_output

        if keras.config.image_data_format() == "channels_last":
            self.assertEqual(
                main_output.shape[1:3],
                (larger_height, larger_width),
                "Output spatial dimensions don't match input dimensions",
            )
        else:
            self.assertEqual(
                main_output.shape[2:4],
                (larger_height, larger_width),
                "Output spatial dimensions don't match input dimensions",
            )

    def test_vlm_text_image_inputs(self):
        if self.model_type != "vlm":
            self.skipTest("This test is only for VLM models")

        model = self.create_model()
        input_data = self.get_input_data()

        self.assertIsInstance(input_data, dict, "VLM input should be a dictionary")

        output = model(input_data)
        self.check_output_shape(output, self.expected_output_shape)

        output_train = model(input_data, training=True)
        output_infer = model(input_data, training=False)

        if isinstance(output_train, dict) and isinstance(output_infer, dict):
            self.assertEqual(set(output_train.keys()), set(output_infer.keys()))
