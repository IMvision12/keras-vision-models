import json
import os

import keras
import pytest

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    import_model_class,
)

BACKEND = os.environ.get("KERAS_BACKEND", "torch")
MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())

# Models that cause backend-specific issues during serialization
SKIP_SERIALIZATION_TF = {"Sam2Tiny"}


@pytest.mark.serialization
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_config_roundtrip(model_name):
    if BACKEND == "tensorflow" and model_name in SKIP_SERIALIZATION_TF:
        pytest.skip(f"{model_name} causes TF backend segfault during serialization")

    config = MODEL_TEST_CONFIGS[model_name]
    model_cls = import_model_class(config)
    model = model_cls(**config["init_kwargs"])

    cfg = model.get_config()
    revived = model.__class__.from_config(cfg)

    assert isinstance(revived, model.__class__), (
        f"{model_name}: from_config produced wrong type: {type(revived).__name__}"
    )

    input_data = create_test_input(config)
    output = revived(input_data)
    assert output is not None, f"{model_name}: revived model produced None output"


@pytest.mark.serialization
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_keras_serialization_roundtrip(model_name):
    if BACKEND == "tensorflow" and model_name in SKIP_SERIALIZATION_TF:
        pytest.skip(f"{model_name} causes TF backend segfault during serialization")

    config = MODEL_TEST_CONFIGS[model_name]
    model_cls = import_model_class(config)
    model = model_cls(**config["init_kwargs"])

    serialized = keras.saving.serialize_keras_object(model)
    json_str = json.dumps(serialized, indent=4, default=str)
    revived = keras.saving.deserialize_keras_object(json.loads(json_str))

    assert isinstance(revived, model.__class__), (
        f"{model_name}: keras deserialization produced wrong type: "
        f"{type(revived).__name__}"
    )

    input_data = create_test_input(config)
    output = revived(input_data)
    assert output is not None, f"{model_name}: revived model produced None output"
