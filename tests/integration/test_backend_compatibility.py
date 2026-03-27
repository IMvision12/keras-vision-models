import pytest
from keras import ops

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    import_model_class,
)

MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())


@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_model_forward_pass(model_name):
    config = MODEL_TEST_CONFIGS[model_name]
    model_cls = import_model_class(config)
    model = model_cls(**config["init_kwargs"])
    input_data = create_test_input(config)
    output = model(input_data)

    expected = config["expected_output_shape"]
    if expected is None:
        # Models with dynamic output shapes (e.g. SAM) — just check it runs
        return

    if isinstance(expected, dict):
        assert isinstance(output, dict), (
            f"{model_name}: expected dict output, got {type(output)}"
        )
        for key, shape in expected.items():
            if shape is not None:
                assert key in output, f"{model_name}: missing output key '{key}'"
                assert output[key].shape == shape, (
                    f"{model_name}[{key}]: expected {shape}, got {output[key].shape}"
                )
    else:
        assert output.shape == expected, (
            f"{model_name}: expected {expected}, got {output.shape}"
        )


@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_model_no_nans(model_name):
    config = MODEL_TEST_CONFIGS[model_name]
    model_cls = import_model_class(config)
    model = model_cls(**config["init_kwargs"])
    input_data = create_test_input(config)
    output = model(input_data)

    if isinstance(output, dict):
        for key, value in output.items():
            has_nans = bool(ops.any(ops.isnan(value)))
            assert not has_nans, f"{model_name}[{key}] contains NaN values"
    else:
        has_nans = bool(ops.any(ops.isnan(output)))
        assert not has_nans, f"{model_name} output contains NaN values"
