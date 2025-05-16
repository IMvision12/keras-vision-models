import keras

from kvmm.models import xception
from kvmm.utils.model_equivalence_tester import verify_cls_model_equivalence

model_config = {
    "input_shape": (299, 299, 3),
    "include_top": True,
    "include_normalization": False,
    "classifier_activation": "linear",
}

original_model = keras.applications.Xception(
    input_shape=model_config["input_shape"],
    classifier_activation=model_config["classifier_activation"],
    weights="imagenet",
    include_top=model_config["include_top"],
)

custom_model = xception.Xception(
    weights=None,
    input_shape=model_config["input_shape"],
    include_top=model_config["include_top"],
    include_normalization=model_config["include_normalization"],
    classifier_activation=model_config["classifier_activation"],
)

if not original_model or not custom_model:
    raise ValueError("Failed to create one or both models")


original_weights = original_model.get_weights()
custom_model.set_weights(original_weights)

results = verify_cls_model_equivalence(
    original_model,
    custom_model,
    input_shape=(299, 299, 3),
    output_specs={"num_classes": 1000},
    comparison_type="keras_to_keras",
    run_performance=False,
)

if not results["standard_input"]:
    raise ValueError(
        "Model equivalence test failed - model outputs do not match for standard input"
    )

model_filename: str = "keras_org_xception.weights.h5"
custom_model.save_weights(model_filename)
print(f"Model saved successfully as {model_filename}")
