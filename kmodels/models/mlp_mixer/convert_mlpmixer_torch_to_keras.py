from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kmodels.models import mlp_mixer
from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.model_equivalence_tester import verify_cls_model_equivalence
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "dense.1": "mlp_tokens.fc1",
    "dense.2": "mlp_tokens.fc2",
    "dense.3": "mlp_channels.fc1",
    "dense.4": "mlp_channels.fc2",
    "stem.conv": "stem.proj",
    "final.layernomr": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "predictions": "head",
}

model_configs: List[Dict[str, Union[str, type]]] = [
    {
        "keras_cls": mlp_mixer.MLPMixerB16,
        "torch_name": "mixer_b16_224.goog_in21k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": mlp_mixer.MLPMixerB16,
        "torch_name": "mixer_b16_224.goog_in21k",
        "input_shape": [224, 224, 3],
        "num_classes": 21843,
    },
    {
        "keras_cls": mlp_mixer.MLPMixerB16,
        "torch_name": "mixer_b16_224.miil_in21k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": mlp_mixer.MLPMixerL16,
        "torch_name": "mixer_l16_224.goog_in21k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": mlp_mixer.MLPMixerL16,
        "torch_name": "mixer_l16_224.goog_in21k",
        "input_shape": [224, 224, 3],
        "num_classes": 21843,
    },
]

for model_config in model_configs:
    torch_model_name: str = model_config["torch_name"]
    print(f"\n{'=' * 60}")
    print(f"Converting {torch_model_name}...")
    print(f"{'=' * 60}")

    keras_model: keras.Model = model_config["keras_cls"](
        include_top=True,
        input_shape=model_config["input_shape"],
        classifier_activation="linear",
        num_classes=model_config["num_classes"],
        include_normalization=False,
        weights=None,
    )

    torch_model: torch.nn.Module = timm.create_model(
        torch_model_name, pretrained=True
    ).eval()

    trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
        torch_model
    )
    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model
    )

    torch_weights_dict: Dict[str, torch.Tensor] = {
        **trainable_torch_weights,
        **non_trainable_torch_weights,
    }

    for keras_weight, keras_weight_name in tqdm(
        trainable_keras_weights + non_trainable_keras_weights,
        total=len(trainable_keras_weights + non_trainable_keras_weights),
        desc="Transferring weights",
    ):
        torch_weight_name: str = keras_weight_name
        for keras_name_part, torch_name_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )

        if torch_weight_name not in torch_weights_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name,
                keras_weight.shape,
                torch_weight_name,
                torch_weight.shape,
            )

        transfer_weights(keras_weight_name, keras_weight, torch_weight)

    results = verify_cls_model_equivalence(
        model_a=torch_model,
        model_b=keras_model,
        input_shape=tuple(model_config["input_shape"]),
        output_specs={"num_classes": model_config["num_classes"]},
        run_performance=False,
        atol=1e-4,
        rtol=1e-4,
    )

    if not results["standard_input"]:
        raise ValueError(
            f"Model equivalence test failed for {torch_model_name} - "
            "model outputs do not match for standard input"
        )

    model_filename: str = f"{torch_model_name.replace('.', '_')}.weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")

    del keras_model, torch_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
