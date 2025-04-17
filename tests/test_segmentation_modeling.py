import json
import os
import tempfile
from typing import Any, Dict, Tuple, Type

import keras
import numpy as np
import tensorflow as tf
from keras import Model


class SegmentationModelConfig:
    def __init__(
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


class SegmentationTest:
    def get_default_kwargs(self) -> Dict[str, Any]:
        return {}

    def get_input_data(self, config: SegmentationModelConfig) -> np.ndarray:
        return np.random.random((config.batch_size,) + config.input_shape).astype(
            np.float32
        )

    def create_model(self, config: SegmentationModelConfig, **kwargs: Any) -> Model:
        default_kwargs = {
            "weights": None,
            "input_shape": kwargs.get("input_shape", config.input_shape),
            "num_classes": config.num_classes,
            **self.get_default_kwargs(),
        }
        default_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        return config.model_cls(**default_kwargs)

    def convert_data_format(self, data: np.ndarray, to_format: str) -> np.ndarray:
        if len(data.shape) == 4:
            if to_format == "channels_first":
                return np.transpose(data, (0, 3, 1, 2))
            return np.transpose(data, (0, 2, 3, 1))
        elif len(data.shape) == 3:
            if to_format == "channels_first":
                return np.transpose(data, (2, 0, 1))
            return np.transpose(data, (1, 2, 0))
        return data

    def test_model_creation(self, model_config):
        model = self.create_model(model_config)
        assert isinstance(model, Model)

    def test_model_forward_pass(self, model_config):
        model = self.create_model(model_config)
        input_data = self.get_input_data(model_config)
        output = model(input_data)

        if isinstance(output, list):
            main_output = output[-1]
        else:
            main_output = output

        if keras.config.image_data_format() == "channels_last":
            expected_shape = (
                model_config.batch_size,
                model_config.input_shape[0],
                model_config.input_shape[1],
                model_config.num_classes,
            )
        else:
            expected_shape = (
                model_config.batch_size,
                model_config.num_classes,
                model_config.input_shape[0],
                model_config.input_shape[1],
            )

        assert main_output.shape == expected_shape, (
            f"Output shape mismatch. Expected {expected_shape}, got {main_output.shape}"
        )

    def test_data_formats(self, model_config):
        original_data_format = keras.config.image_data_format()
        input_data = self.get_input_data(model_config)

        try:
            keras.config.set_image_data_format("channels_last")
            model_last = self.create_model(model_config)
            output_last = model_last(input_data)

            if isinstance(output_last, list):
                output_last = output_last[-1]

            expected_shape_last = (
                model_config.batch_size,
                model_config.input_shape[0],
                model_config.input_shape[1],
                model_config.num_classes,
            )
            assert output_last.shape == expected_shape_last

            if (
                keras.config.backend() == "tensorflow"
                and tf.config.list_physical_devices("GPU")
            ):
                keras.config.set_image_data_format("channels_first")
                current_shape = (
                    model_config.input_shape[2],
                    model_config.input_shape[0],
                    model_config.input_shape[1],
                )
                current_data = self.convert_data_format(input_data, "channels_first")

                model_first = self.create_model(model_config, input_shape=current_shape)
                model_first.set_weights(model_last.get_weights())

                output_first = model_first(current_data)

                if isinstance(output_first, list):
                    output_first = output_first[-1]

                expected_shape_first = (
                    model_config.batch_size,
                    model_config.num_classes,
                    model_config.input_shape[0],
                    model_config.input_shape[1],
                )
                assert output_first.shape == expected_shape_first

                if len(output_first.shape) == 4:
                    output_first_converted = tf.transpose(output_first, [0, 2, 3, 1])

                    np.testing.assert_allclose(
                        output_first_converted.numpy(),
                        output_last.numpy(),
                        rtol=1e-5,
                        atol=1e-5,
                    )
        finally:
            keras.config.set_image_data_format(original_data_format)

    def test_model_saving(self, model_config):
        model = self.create_model(model_config)
        input_data = self.get_input_data(model_config)
        original_output = model(input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_segmentation_model.keras")

            model.save(save_path)
            loaded_model = keras.models.load_model(save_path)

            assert isinstance(loaded_model, model.__class__), (
                f"Loaded model should be an instance of {model.__class__.__name__}"
            )

            loaded_output = loaded_model(input_data)

            # Handle multi-output models
            if isinstance(original_output, list) and isinstance(loaded_output, list):
                assert len(original_output) == len(loaded_output), (
                    "Number of outputs doesn't match after loading model"
                )

                for i, (orig, loaded) in enumerate(zip(original_output, loaded_output)):
                    np.testing.assert_allclose(
                        orig.numpy(),
                        loaded.numpy(),
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Output {i} mismatch after loading model",
                    )
            else:
                np.testing.assert_allclose(
                    original_output.numpy(), loaded_output.numpy(), rtol=1e-5, atol=1e-5
                )

    def test_serialization(self, model_config):
        model = self.create_model(model_config)

        run_dir_test = not keras.config.backend() == "tensorflow"

        cls = model.__class__
        cfg = model.get_config()
        cfg_json = json.dumps(cfg, sort_keys=True, indent=4)
        ref_dir = dir(model)[:]

        revived_instance = cls.from_config(cfg)
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        assert cfg_json == revived_cfg_json, (
            "Config JSON mismatch after from_config roundtrip"
        )

        if run_dir_test:
            assert set(ref_dir) == set(dir(revived_instance)), (
                "Dir mismatch after from_config roundtrip"
            )

        serialized = keras.saving.serialize_keras_object(model)
        serialized_json = json.dumps(serialized, sort_keys=True, indent=4)
        revived_instance = keras.saving.deserialize_keras_object(
            json.loads(serialized_json)
        )
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        assert cfg_json == revived_cfg_json, (
            "Config JSON mismatch after full serialization roundtrip"
        )

        if run_dir_test:
            new_dir = dir(revived_instance)[:]
            for lst in [ref_dir, new_dir]:
                if "__annotations__" in lst:
                    lst.remove("__annotations__")
            assert set(ref_dir) == set(new_dir), (
                "Dir mismatch after full serialization roundtrip"
            )

    def test_training_mode(self, model_config):
        model = self.create_model(model_config)
        model.trainable = True
        assert model.trainable

        input_data = self.get_input_data(model_config)

        training_output = model(input_data, training=True)
        inference_output = model(input_data, training=False)

        if isinstance(training_output, list) and isinstance(inference_output, list):
            assert len(training_output) == len(inference_output), (
                "Number of outputs doesn't match between training and inference modes"
            )

            for i, (train_out, infer_out) in enumerate(
                zip(training_output, inference_output)
            ):
                assert train_out.shape == infer_out.shape, (
                    f"Output {i} shape mismatch between training and inference modes"
                )
        else:
            assert training_output.shape == inference_output.shape

    def test_auxiliary_outputs(self, model_config):
        """Test models with auxiliary outputs (like DeepLabV3+)"""
        model = self.create_model(model_config)
        input_data = self.get_input_data(model_config)
        outputs = model(input_data)

        if isinstance(outputs, list):
            assert len(outputs) > 1, "Expected multiple outputs but got only one"

            main_output = outputs[-1]

            if keras.config.image_data_format() == "channels_last":
                expected_shape = (
                    model_config.batch_size,
                    model_config.input_shape[0],
                    model_config.input_shape[1],
                    model_config.num_classes,
                )
            else:
                expected_shape = (
                    model_config.batch_size,
                    model_config.num_classes,
                    model_config.input_shape[0],
                    model_config.input_shape[1],
                )

            assert main_output.shape == expected_shape, (
                f"Main output shape mismatch. Expected {expected_shape}, got {main_output.shape}"
            )

            for i, aux_output in enumerate(outputs[:-1]):
                assert len(aux_output.shape) == 4, (
                    f"Auxiliary output {i} should be a 4D tensor, "
                    f"got shape {aux_output.shape}"
                )

                assert aux_output.shape[0] == model_config.batch_size, (
                    f"Auxiliary output {i} has incorrect batch size. "
                    f"Expected {model_config.batch_size}, got {aux_output.shape[0]}"
                )

    def test_different_input_sizes(self, model_config):
        larger_shape = (
            model_config.input_shape[0] + 64,
            model_config.input_shape[1] + 64,
            model_config.input_shape[2],
        )

        kwargs = {"input_shape": larger_shape}
        larger_model = self.create_model(model_config, **kwargs)

        larger_input = np.random.random(
            (model_config.batch_size,) + larger_shape
        ).astype(np.float32)
        larger_output = larger_model(larger_input)

        if isinstance(larger_output, list):
            main_output = larger_output[-1]
        else:
            main_output = larger_output

        if keras.config.image_data_format() == "channels_last":
            assert main_output.shape[1:3] == larger_shape[0:2], (
                f"Output spatial dimensions don't match input. "
                f"Expected {larger_shape[0:2]}, got {main_output.shape[1:3]}"
            )
        else:
            assert main_output.shape[2:4] == larger_shape[0:2], (
                f"Output spatial dimensions don't match input. "
                f"Expected {larger_shape[0:2]}, got {main_output.shape[2:4]}"
            )
