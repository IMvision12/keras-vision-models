import json
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
                assert output_first.shape == (
                    model_config.batch_size,
                    model_config.num_classes,
                )

                np.testing.assert_allclose(
                    output_first.numpy(), output_last.numpy(), rtol=1e-5, atol=1e-5
                )
        finally:
            keras.config.set_image_data_format(original_data_format)

    def test_model_saving(self, model_config):
        model = self.create_model(model_config)
        input_data = self.get_input_data(model_config)
        original_output = model(input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model.keras")

            model.save(save_path)
            loaded_model = keras.models.load_model(save_path)

            assert isinstance(loaded_model, model.__class__), (
                f"Loaded model should be an instance of {model.__class__.__name__}"
            )

            loaded_output = loaded_model(input_data)
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

        assert training_output.shape == inference_output.shape

    def test_backbone_features(self, model_config):
        model = self.create_model(model_config, include_top=False, as_backbone=True)
        input_data = self.get_input_data(model_config)
        features = model(input_data)

        assert isinstance(features, list), (
            "Backbone output should be a list of feature maps"
        )

        assert len(features) >= 2, "Backbone should output at least 2 feature maps"

        for i, feature_map in enumerate(features):
            assert len(feature_map.shape) == 4, (
                f"Feature map {i} should be a 4D tensor, got shape {feature_map.shape}"
            )

            assert feature_map.shape[0] == model_config.batch_size, (
                f"Feature map {i} has incorrect batch size. "
                f"Expected {model_config.batch_size}, got {feature_map.shape[0]}"
            )

            if keras.config.image_data_format() == "channels_last":
                h, w, c = feature_map.shape[1:]
            else:
                c, h, w = feature_map.shape[1:]

            assert h > 0 and w > 0 and c > 0, (
                f"Feature map {i} has invalid dimensions: height={h}, width={w}, channels={c}"
            )

            if i > 0:
                prev_map = features[i - 1]
                prev_h = (
                    prev_map.shape[1]
                    if keras.config.image_data_format() == "channels_last"
                    else prev_map.shape[2]
                )
                prev_w = (
                    prev_map.shape[2]
                    if keras.config.image_data_format() == "channels_last"
                    else prev_map.shape[3]
                )

                assert h <= prev_h and w <= prev_w, (
                    f"Feature map {i} has larger spatial dimensions than previous feature map. "
                    f"Got {h}x{w}, previous was {prev_h}x{prev_w}"
                )

                assert prev_h / h <= 4 and prev_w / w <= 4, (
                    f"Feature map {i} has too large spatial reduction from previous feature map. "
                    f"Got {h}x{w}, previous was {prev_h}x{prev_w}"
                )

        features_train = model(input_data, training=True)
        assert len(features_train) == len(features), (
            "Number of feature maps should be consistent between training and inference modes"
        )

        if model_config.batch_size > 1:
            single_input = input_data[:1]
            single_features = model(single_input)
            assert len(single_features) == len(features), (
                "Number of feature maps should be consistent across different batch sizes"
            )

            for i, (single_feat, batch_feat) in enumerate(
                zip(single_features, features)
            ):
                assert single_feat.shape[1:] == batch_feat.shape[1:], (
                    f"Feature map {i} shapes don't match between different batch sizes"
                )
