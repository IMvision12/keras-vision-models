import json
import os
import tempfile
from typing import Any, Dict, Tuple, Type

import keras
from keras import ops

import tensorflow as tf
from keras import Model
from keras.src.testing import TestCase


class BackboneTestCase(TestCase):
    __test__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_cls = None
        self.input_shape = (224, 224, 3)
        self.batch_size = 2
        self.num_classes = 1000

    def configure(
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
        return self

    def get_default_kwargs(self) -> Dict[str, Any]:
        return {}

    def get_input_data(self) -> keras.KerasTensor:
        return keras.random.uniform((self.batch_size,) + self.input_shape, dtype="float32")

    def create_model(self, **kwargs: Any) -> Model:
        if self.model_cls is None:
            self.skipTest("Model class not configured. Call configure() first.")
            
        default_kwargs = {
            "include_top": True,
            "weights": None,
            "input_shape": kwargs.get("input_shape", self.input_shape),
            "num_classes": self.num_classes,
            **self.get_default_kwargs(),
        }
        default_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        return self.model_cls(**default_kwargs)

    def convert_data_format(self, data: keras.KerasTensor, to_format: str) -> keras.KerasTensor:
        if len(data.shape) == 4:
            if to_format == "channels_first":
                return ops.transpose(data, (0, 3, 1, 2))
            return ops.transpose(data, (0, 2, 3, 1))
        elif len(data.shape) == 3:
            if to_format == "channels_first":
                return ops.transpose(data, (2, 0, 1))
            return ops.transpose(data, (1, 2, 0))
        return data

    def test_model_creation(self):
        model = self.create_model()
        self.assertIsInstance(model, Model)

    def test_model_forward_pass(self):
        model = self.create_model()
        input_data = self.get_input_data()
        output = model(input_data)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_weight_loading(self, model):        
        self.assertIsNotNone(model.weights, "Model weights not initialized")
        self.assertTrue(len(model.trainable_weights) > 0, "Model has no trainable weights")
        
        for weight in model.weights:
            has_nans = ops.any(ops.isnan(weight))
            self.assertFalse(
                has_nans,
                f"Weight '{weight.name}' contains NaN values"
            )
            
            is_all_zeros = ops.all(ops.equal(weight, 0))
            self.assertFalse(
                is_all_zeros,
                f"Weight '{weight.name}' contains all zeros, suggesting improper initialization"
            )
            
    def test_data_formats(self):
        original_data_format = keras.config.image_data_format()
        input_data = self.get_input_data()

        try:
            keras.config.set_image_data_format("channels_last")
            model_last = self.create_model()
            output_last = model_last(input_data)
            self.assertEqual(output_last.shape, (self.batch_size, self.num_classes))

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
                current_data = self.convert_data_format(input_data, "channels_first")

                model_first = self.create_model(input_shape=current_shape)
                model_first.set_weights(model_last.get_weights())

                output_first = model_first(current_data)
                self.assertEqual(output_first.shape, (self.batch_size, self.num_classes))

                self.assertAllClose(
                    output_first, output_last, rtol=1e-5, atol=1e-5
                )
        finally:
            keras.config.set_image_data_format(original_data_format)

    def test_model_saving(self):
        model = self.create_model()
        input_data = self.get_input_data()
        original_output = model(input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model.keras")

            model.save(save_path)
            loaded_model = keras.models.load_model(save_path)

            self.assertIsInstance(
                loaded_model, model.__class__,
                f"Loaded model should be an instance of {model.__class__.__name__}"
            )

            loaded_output = loaded_model(input_data)
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
            cfg_json, revived_cfg_json,
            "Config JSON mismatch after from_config roundtrip"
        )

        if run_dir_test:
            self.assertEqual(
                set(ref_dir), set(dir(revived_instance)),
                "Dir mismatch after from_config roundtrip"
            )

        serialized = keras.saving.serialize_keras_object(model)
        serialized_json = json.dumps(serialized, sort_keys=True, indent=4)
        revived_instance = keras.saving.deserialize_keras_object(
            json.loads(serialized_json)
        )
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        self.assertEqual(
            cfg_json, revived_cfg_json,
            "Config JSON mismatch after full serialization roundtrip"
        )

        if run_dir_test:
            new_dir = dir(revived_instance)[:]
            for lst in [ref_dir, new_dir]:
                if "__annotations__" in lst:
                    lst.remove("__annotations__")
            self.assertEqual(
                set(ref_dir), set(new_dir),
                "Dir mismatch after full serialization roundtrip"
            )

    def test_training_mode(self):
        model = self.create_model()
        model.trainable = True
        self.assertTrue(model.trainable)

        input_data = self.get_input_data()

        training_output = model(input_data, training=True)
        inference_output = model(input_data, training=False)

        self.assertEqual(training_output.shape, inference_output.shape)

    def test_backbone_features(self):
        model = self.create_model(include_top=False, as_backbone=True)
        input_data = self.get_input_data()
        features = model(input_data)

        self.assertIsInstance(
            features, list,
            "Backbone output should be a list of feature maps"
        )

        self.assertGreaterEqual(
            len(features), 2,
            "Backbone should output at least 2 feature maps"
        )

        for i, feature_map in enumerate(features):
            is_transformer_output = len(feature_map.shape) == 3

            self.assertIn(
                len(feature_map.shape), (3, 4),
                f"Feature map {i} should be a 3D (transformer) or 4D (CNN) tensor, "
                f"got shape {feature_map.shape}"
            )

            self.assertEqual(
                int(feature_map.shape[0]), self.batch_size,
                f"Feature map {i} has incorrect batch size. "
                f"Expected {self.batch_size}, got {feature_map.shape[0]}"
            )

            if is_transformer_output:
                seq_len, channels = int(feature_map.shape[1]), int(feature_map.shape[2])
                self.assertGreater(seq_len, 0, f"Feature map {i} has invalid sequence length: {seq_len}")
                self.assertGreater(channels, 0, f"Feature map {i} has invalid channel count: {channels}")

                if i > 0:
                    prev_map = features[i - 1]
                    if len(prev_map.shape) == 3:  # Only compare with previous transformer outputs
                        prev_seq_len = int(prev_map.shape[1])
                        self.assertLessEqual(
                            seq_len, prev_seq_len,
                            f"Feature map {i} has larger sequence length than previous feature map. "
                            f"Got {seq_len}, previous was {prev_seq_len}"
                        )

            else:  # CNN output (4D)
                if keras.config.image_data_format() == "channels_last":
                    h, w, c = int(feature_map.shape[1]), int(feature_map.shape[2]), int(feature_map.shape[3])
                else:
                    c, h, w = int(feature_map.shape[1]), int(feature_map.shape[2]), int(feature_map.shape[3])

                self.assertGreater(h, 0, f"Feature map {i} has invalid height: {h}")
                self.assertGreater(w, 0, f"Feature map {i} has invalid width: {w}")
                self.assertGreater(c, 0, f"Feature map {i} has invalid channel count: {c}")

                if i > 0:
                    prev_map = features[i - 1]
                    if len(prev_map.shape) == 4:  # Only compare with previous CNN outputs
                        prev_h = int(
                            prev_map.shape[1]
                            if keras.config.image_data_format() == "channels_last"
                            else prev_map.shape[2]
                        )
                        prev_w = int(
                            prev_map.shape[2]
                            if keras.config.image_data_format() == "channels_last"
                            else prev_map.shape[3]
                        )

                        self.assertLessEqual(
                            h, prev_h,
                            f"Feature map {i} has larger height than previous feature map. "
                            f"Got {h}, previous was {prev_h}"
                        )
                        self.assertLessEqual(
                            w, prev_w,
                            f"Feature map {i} has larger width than previous feature map. "
                            f"Got {w}, previous was {prev_w}"
                        )

                        self.assertLessEqual(
                            prev_h / h, 4,
                            f"Feature map {i} has too large height reduction from previous feature map. "
                            f"Got {h}, previous was {prev_h}"
                        )
                        self.assertLessEqual(
                            prev_w / w, 4,
                            f"Feature map {i} has too large width reduction from previous feature map. "
                            f"Got {w}, previous was {prev_w}"
                        )

        features_train = model(input_data, training=True)
        self.assertEqual(
            len(features_train), len(features),
            "Number of feature maps should be consistent between training and inference modes"
        )

        if self.batch_size > 1:
            single_input = input_data[:1]
            single_features = model(single_input)
            self.assertEqual(
                len(single_features), len(features),
                "Number of feature maps should be consistent across different batch sizes"
            )

            for i, (single_feat, batch_feat) in enumerate(zip(single_features, features)):
                self.assertEqual(
                    tuple(single_feat.shape[1:]), tuple(batch_feat.shape[1:]),
                    f"Feature map {i} shapes don't match between different batch sizes"
                )