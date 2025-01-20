from typing import Any, Dict, Optional

import keras

from kv.models import Xception
from kv.utils.model_equivalence_tester import verify_cls_model_equivalence


def create_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[keras.Model]:
    if not config:
        print("Missing configuration.")
        return None

    try:
        if model_type == "original":
            return keras.applications.Xception(
                input_shape=config["input_shape"],
                classifier_activation=config["classifier_activation"],
                weights="imagenet",
                include_top=config["include_top"],
            )
        elif model_type == "custom":
            return Xception(
                weights=None,
                input_shape=config["input_shape"],
                include_top=config["include_top"],
                include_preprocessing=config["include_preprocessing"],
                classifier_activation=config["classifier_activation"],
            )
        else:
            print("Invalid model type.")
            return None
    except Exception as e:
        print(f"Error creating {model_type} model: {e}")
        return None


model_config = {
    "input_shape": (299, 299, 3),
    "include_top": True,
    "include_preprocessing": False,
    "classifier_activation": "linear",
}

original_model = create_model("original", config=model_config)
custom_model = create_model("custom", config=model_config)

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

model_filename: str = "keras_org_xception.keras"
custom_model.save(model_filename)
print(f"Model saved successfully as {model_filename}")
