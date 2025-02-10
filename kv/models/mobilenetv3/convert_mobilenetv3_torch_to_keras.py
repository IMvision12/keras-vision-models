import keras
import tensorflow as tf

from kv.models.mobilenetv3 import MobileNetV3LargeMinimal100
from kv.utils.model_equivalence_tester import verify_cls_model_equivalence

model_config = {
    "input_shape": (224, 224, 3),
    "include_top": True,
    "include_normalization": False,
    "classifier_activation": "linear",
}

original_model = keras.applications.MobileNetV3Large(
    input_shape=(224,224,3),
    alpha=1.0,
    minimalistic=True,
    include_top=True,
    weights='imagenet',
    classifier_activation='linear',
    include_preprocessing=False
)

custom_model = MobileNetV3LargeMinimal100(
    weights=None,
    input_shape=model_config["input_shape"],
    include_top=model_config["include_top"],
    include_normalization=False,
    classifier_activation=model_config["classifier_activation"],
)

original_weights = original_model.get_weights()
custom_model.set_weights(original_weights)

results = verify_cls_model_equivalence(
    original_model,
    custom_model,
    input_shape=(224, 224, 3),
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
