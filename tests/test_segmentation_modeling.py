import json
import os
import tempfile
from typing import Any, Dict, Tuple, Type

import keras
import tensorflow as tf
from keras import Model, ops
from keras.src.testing import TestCase


class SegmentationTestCase(TestCase):
    """Base test case for segmentation models."""

    __test__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_cls = None
        self.input_shape = (32, 32, 3)
        self.batch_size = 2
        self.num_classes = 21

    def configure(
        self,
        model_cls: Type[Model],
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        batch_size: int = 2,
        num_classes: int = 21,
    ):
        self.model_cls = model_cls
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        return self

    def get_default_kwargs(self) -> Dict[str, Any]:
        return {}

    def get_input_data(self) -> keras.KerasTensor:
        return keras.random.uniform(
            (self.batch_size,) + self.input_shape, dtype="float32"
        )

    def create_model(self, **kwargs: Any) -> Model:
        if self.model_cls is None:
            self.skipTest("Model class not configured. Call configure() first.")

        default_kwargs = {
            "weights": None,
            "input_shape": kwargs.get("input_shape", self.input_shape),
            "num_classes": self.num_classes,
            **self.get_default_kwargs(),
        }
        default_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        return self.model_cls(**default_kwargs)

    def test_model_creation(self):
        model = self.create_model()
        self.assertIsInstance(model, Model)

    def test_weight_loading(self, model):
        self.assertIsNotNone(model.weights, "Model weights not initialized")
        self.assertTrue(
            len(model.trainable_weights) > 0, "Model has no trainable weights"
        )
        for weight in model.weights:
            has_nans = ops.any(ops.isnan(weight))
            self.assertFalse(has_nans, f"Weight '{weight.name}' contains NaN values")

            is_all_zeros = ops.all(ops.equal(weight, 0))
            self.assertFalse(
                is_all_zeros,
                f"Weight '{weight.name}' contains all zeros, suggesting improper initialization",
            )
        input_data = self.get_input_data()
        original_output = model(input_data)
        second_model = keras.models.clone_model(model)
        second_model.set_weights(model.get_weights())
        second_output = second_model(input_data)

        if isinstance(original_output, list) and isinstance(second_output, list):
            self.assertEqual(
                len(original_output),
                len(second_output),
                "Number of outputs doesn't match after loading weights",
            )

            for i, (orig, loaded) in enumerate(zip(original_output, second_output)):
                self.assertAllClose(
                    orig,
                    loaded,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Output {i} mismatch after loading weights",
                )
        else:
            self.assertAllClose(
                original_output,
                second_output,
                rtol=1e-5,
                atol=1e-5,
                msg="Output mismatch after loading weights",
            )

    def test_model_forward_pass(self):
        model = self.create_model()
        input_data = self.get_input_data()
        output = model(input_data)

        if isinstance(output, list):
            main_output = output[-1]
        else:
            main_output = output

        if keras.config.image_data_format() == "channels_last":
            expected_shape = (
                self.batch_size,
                self.input_shape[0],
                self.input_shape[1],
                self.num_classes,
            )
        else:
            expected_shape = (
                self.batch_size,
                self.num_classes,
                self.input_shape[0],
                self.input_shape[1],
            )

        self.assertEqual(
            main_output.shape,
            expected_shape,
            f"Output shape mismatch. Expected {expected_shape}, got {main_output.shape}",
        )

    def test_data_formats(self):
        original_data_format = keras.config.image_data_format()
        input_data = self.get_input_data()

        try:
            keras.config.set_image_data_format("channels_last")
            model_last = self.create_model()
            output_last = model_last(input_data)

            if isinstance(output_last, list):
                output_last = output_last[-1]

            expected_shape_last = (
                self.batch_size,
                self.input_shape[0],
                self.input_shape[1],
                self.num_classes,
            )
            self.assertEqual(output_last.shape, expected_shape_last)

            if (
                keras.config.backend() == "tensorflow"
                and tf.config.list_physical_devices("GPU")
            ):
                keras.config.set_image_data_format("channels_first")
                current_shape = (
                    self.input_shape[2],
                    self.input_shape[0],
                    self.input_shape[1],
                )
                current_data = ops.transpose(input_data, (0, 3, 1, 2))

                model_first = self.create_model(input_shape=current_shape)
                model_first.set_weights(model_last.get_weights())

                output_first = model_first(current_data)

                if isinstance(output_first, list):
                    output_first = output_first[-1]

                expected_shape_first = (
                    self.batch_size,
                    self.num_classes,
                    self.input_shape[0],
                    self.input_shape[1],
                )
                self.assertEqual(output_first.shape, expected_shape_first)

                if len(output_first.shape) == 4:
                    output_first_converted = ops.transpose(output_first, [0, 2, 3, 1])
                    self.assertAllClose(
                        output_first_converted, output_last, rtol=1e-5, atol=1e-5
                    )
        finally:
            keras.config.set_image_data_format(original_data_format)

    def test_model_saving(self):
        model = self.create_model()
        input_data = self.get_input_data()
        original_output = model(input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_segmentation_model.keras")

            model.save(save_path)
            loaded_model = keras.models.load_model(save_path)

            self.assertIsInstance(
                loaded_model,
                model.__class__,
                f"Loaded model should be an instance of {model.__class__.__name__}",
            )

            loaded_output = loaded_model(input_data)

            # Handle multi-output models
            if isinstance(original_output, list) and isinstance(loaded_output, list):
                self.assertEqual(
                    len(original_output),
                    len(loaded_output),
                    "Number of outputs doesn't match after loading model",
                )

                for i, (orig, loaded) in enumerate(zip(original_output, loaded_output)):
                    self.assertAllClose(
                        orig,
                        loaded,
                        rtol=1e-5,
                        atol=1e-5,
                        msg=f"Output {i} mismatch after loading model",
                    )
            else:
                self.assertAllClose(
                    original_output, loaded_output, rtol=1e-5, atol=1e-5
                )

    def test_serialization(self):
        model = self.create_model()

        run_dir_test = not keras.config.backend() == "tensorflow"

        cls = model.__class__
        cfg = model.get_config()
        cfg_json = json.dumps(cfg, sort_keys=True, indent=4)
        ref_dir = dir(model)[:]

        revived_instance = cls.from_config(cfg)
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        self.assertEqual(
            cfg_json,
            revived_cfg_json,
            "Config JSON mismatch after from_config roundtrip",
        )

        if run_dir_test:
            self.assertEqual(
                set(ref_dir),
                set(dir(revived_instance)),
                "Dir mismatch after from_config roundtrip",
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

        if run_dir_test:
            new_dir = dir(revived_instance)[:]
            for lst in [ref_dir, new_dir]:
                if "__annotations__" in lst:
                    lst.remove("__annotations__")
            self.assertEqual(
                set(ref_dir),
                set(new_dir),
                "Dir mismatch after full serialization roundtrip",
            )

    def test_training_mode(self):
        model = self.create_model()
        model.trainable = True
        self.assertTrue(model.trainable)

        input_data = self.get_input_data()

        training_output = model(input_data, training=True)
        inference_output = model(input_data, training=False)

        if isinstance(training_output, list) and isinstance(inference_output, list):
            self.assertEqual(
                len(training_output),
                len(inference_output),
                "Number of outputs doesn't match between training and inference modes",
            )

            for i, (train_out, infer_out) in enumerate(
                zip(training_output, inference_output)
            ):
                self.assertEqual(
                    train_out.shape,
                    infer_out.shape,
                    f"Output {i} shape mismatch between training and inference modes",
                )
        else:
            self.assertEqual(training_output.shape, inference_output.shape)

    def test_auxiliary_outputs(self):
        """Test models with auxiliary outputs (like DeepLabV3+)"""
        model = self.create_model()
        input_data = self.get_input_data()
        outputs = model(input_data)

        if isinstance(outputs, list):
            self.assertGreater(
                len(outputs), 1, "Expected multiple outputs but got only one"
            )

            main_output = outputs[-1]

            if keras.config.image_data_format() == "channels_last":
                expected_shape = (
                    self.batch_size,
                    self.input_shape[0],
                    self.input_shape[1],
                    self.num_classes,
                )
            else:
                expected_shape = (
                    self.batch_size,
                    self.num_classes,
                    self.input_shape[0],
                    self.input_shape[1],
                )

            self.assertEqual(
                main_output.shape,
                expected_shape,
                f"Main output shape mismatch. Expected {expected_shape}, got {main_output.shape}",
            )

            for i, aux_output in enumerate(outputs[:-1]):
                self.assertEqual(
                    len(aux_output.shape),
                    4,
                    f"Auxiliary output {i} should be a 4D tensor, got shape {aux_output.shape}",
                )

                self.assertEqual(
                    int(aux_output.shape[0]),
                    self.batch_size,
                    f"Auxiliary output {i} has incorrect batch size. "
                    f"Expected {self.batch_size}, got {aux_output.shape[0]}",
                )

    def test_different_input_sizes(self):
        larger_shape = (
            self.input_shape[0] + 64,
            self.input_shape[1] + 64,
            self.input_shape[2],
        )

        kwargs = {"input_shape": larger_shape}
        larger_model = self.create_model(**kwargs)

        larger_input = keras.random.uniform(
            (self.batch_size,) + larger_shape, dtype="float32"
        )
        larger_output = larger_model(larger_input)

        if isinstance(larger_output, list):
            main_output = larger_output[-1]
        else:
            main_output = larger_output

        if keras.config.image_data_format() == "channels_last":
            self.assertEqual(
                main_output.shape[1:3],
                larger_shape[0:2],
                f"Output spatial dimensions don't match input. "
                f"Expected {larger_shape[0:2]}, got {main_output.shape[1:3]}",
            )
        else:
            self.assertEqual(
                main_output.shape[2:4],
                larger_shape[0:2],
                f"Output spatial dimensions don't match input. "
                f"Expected {larger_shape[0:2]}, got {main_output.shape[2:4]}",
            )
