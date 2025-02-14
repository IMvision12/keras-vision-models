import os
import tempfile
from typing import Any, Dict, Tuple, Type

import keras
import numpy as np
import tensorflow as tf
from keras import Model


class ModelConfig:
    def __init__(
        self,
        model_cls: Type[Model],
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        batch_size: int = 2,
        num_classes: int = 1000,
    ):
        self.model_cls = model_cls
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes


class BaseVisionTest:
    def get_default_kwargs(self) -> Dict[str, Any]:
        return {}

    def get_input_data(self, config: ModelConfig) -> np.ndarray:
        return np.random.random((config.batch_size,) + config.input_shape).astype(
            np.float32
        )

    def create_model(self, config: ModelConfig, **kwargs: Any) -> Model:
        default_kwargs = {
            "include_top": True,
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
        assert output.shape == (model_config.batch_size, model_config.num_classes)

    def test_data_formats(self, model_config):
        original_data_format = keras.config.image_data_format()
        input_data = self.get_input_data(model_config)

        try:
            keras.config.set_image_data_format("channels_last")
            model_last = self.create_model(model_config)
            output_last = model_last(input_data)
            assert output_last.shape == (
                model_config.batch_size,
                model_config.num_classes,
            )

            if (
                keras.config.backend() == "tensorflow"
                and not tf.config.list_physical_devices("GPU")
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
                assert output_first.shape == (
                    model_config.batch_size,
                    model_config.num_classes,
                )

                np.testing.assert_allclose(
                    output_first.numpy(), output_last.numpy(), rtol=1e-5, atol=1e-5
                )
        finally:
            keras.config.set_image_data_format(original_data_format)

    def test_serialization(self, model_config):
        model = self.create_model(model_config)
        input_data = self.get_input_data(model_config)
        original_output = model(input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model.keras")

            model.save(save_path)
            loaded_model = keras.models.load_model(save_path)
            loaded_output = loaded_model(input_data)
            np.testing.assert_allclose(
                original_output.numpy(), loaded_output.numpy(), rtol=1e-5, atol=1e-5
            )

            config = model.get_config()
            restored_model = model.__class__.from_config(config)
            assert isinstance(restored_model, Model)

            restored_output = restored_model(input_data)
            np.testing.assert_allclose(
                original_output.numpy(), restored_output.numpy(), rtol=1e-5, atol=1e-5
            )
