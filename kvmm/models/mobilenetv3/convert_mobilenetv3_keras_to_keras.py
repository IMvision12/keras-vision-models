import keras

from kvmm.models import mobilenetv3
from kvmm.utils.model_equivalence_tester import verify_cls_model_equivalence

model_config = {
    "input_shape": (224, 224, 3),
    "include_top": True,
    "alpha": 0.75,
    "minimalistic": False,
    "include_normalization": False,
    "include_preprocessing": False,
    "classifier_activation": "linear",
    "model_cls": mobilenetv3.MobileNetV3Small075,
}

original_model = keras.applications.MobileNetV3Small(
    input_shape=model_config["input_shape"],
    alpha=model_config["alpha"],
    minimalistic=model_config["minimalistic"],
    include_top=model_config["include_top"],
    weights="imagenet",
    classifier_activation=model_config["classifier_activation"],
    include_preprocessing=model_config["include_preprocessing"],
)

custom_model = model_config["model_cls"](
    weights=None,
    input_shape=model_config["input_shape"],
    include_top=model_config["include_top"],
    include_normalization=model_config["include_normalization"],
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


model_filename: str = (
    f"keras_org_{model_config['model_cls'].__name__.lower()}.weights.h5"
)
custom_model.save_weights(model_filename)
print(f"Model saved successfully as {model_filename}")
