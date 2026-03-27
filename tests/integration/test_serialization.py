import json

import keras
import pytest

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    import_model_class,
)

MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())


@pytest.mark.serialization
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_config_roundtrip(model_name):
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
