import re
from typing import Dict, List, Union

import keras
import timm
import torch
from tqdm import tqdm

from kmodels.models import convnext
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.model_equivalence_tester import verify_cls_model_equivalence
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "stem_conv_": "stem.0.",
    "stem_layernorm_": "stem.1.",
    "_": ".",
    "layernorm": "norm",
    "depthwise.conv": "conv_dw",
    "grn": "mlp.grn",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "conv.1": "mlp.fc1",
    "conv.2": "mlp.fc2",
    "downsampling.norm": "downsample.0",
    "downsampling.conv": "downsample.1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "final.norm": "head.norm",
    "predictions": "head.fc",
}

model_configs: List[Dict[str, Union[type, str, List[int], int]]] = [
    # ConvNeXtAtto
    {
        "keras_cls": convnext.ConvNeXtAtto,
        "torch_name": "convnext_atto.d2_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    # ConvNeXtFemto
    {
        "keras_cls": convnext.ConvNeXtFemto,
        "torch_name": "convnext_femto.d1_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    # ConvNeXtPico
    {
        "keras_cls": convnext.ConvNeXtPico,
        "torch_name": "convnext_pico.d1_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    # ConvNeXtNano
    {
        "keras_cls": convnext.ConvNeXtNano,
        "torch_name": "convnext_nano.d1h_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtNano,
        "torch_name": "convnext_nano.in12k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    # ConvNeXtTiny
    {
        "keras_cls": convnext.ConvNeXtTiny,
        "torch_name": "convnext_tiny.fb_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtTiny,
        "torch_name": "convnext_tiny.fb_in22k",
        "input_shape": [224, 224, 3],
        "num_classes": 21841,
    },
    {
        "keras_cls": convnext.ConvNeXtTiny,
        "torch_name": "convnext_tiny.fb_in22k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtTiny,
        "torch_name": "convnext_tiny.fb_in22k_ft_in1k_384",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    # ConvNeXtSmall
    {
        "keras_cls": convnext.ConvNeXtSmall,
        "torch_name": "convnext_small.fb_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtSmall,
        "torch_name": "convnext_small.fb_in22k",
        "input_shape": [224, 224, 3],
        "num_classes": 21841,
    },
    {
        "keras_cls": convnext.ConvNeXtSmall,
        "torch_name": "convnext_small.fb_in22k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtSmall,
        "torch_name": "convnext_small.fb_in22k_ft_in1k_384",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    # ConvNeXtBase
    {
        "keras_cls": convnext.ConvNeXtBase,
        "torch_name": "convnext_base.fb_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtBase,
        "torch_name": "convnext_base.fb_in22k",
        "input_shape": [224, 224, 3],
        "num_classes": 21841,
    },
    {
        "keras_cls": convnext.ConvNeXtBase,
        "torch_name": "convnext_base.fb_in22k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtBase,
        "torch_name": "convnext_base.fb_in22k_ft_in1k_384",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    # ConvNeXtLarge
    {
        "keras_cls": convnext.ConvNeXtLarge,
        "torch_name": "convnext_large.fb_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtLarge,
        "torch_name": "convnext_large.fb_in22k",
        "input_shape": [224, 224, 3],
        "num_classes": 21841,
    },
    {
        "keras_cls": convnext.ConvNeXtLarge,
        "torch_name": "convnext_large.fb_in22k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtLarge,
        "torch_name": "convnext_large.fb_in22k_ft_in1k_384",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
    },
    # ConvNeXtXLarge
    {
        "keras_cls": convnext.ConvNeXtXLarge,
        "torch_name": "convnext_xlarge.fb_in22k",
        "input_shape": [224, 224, 3],
        "num_classes": 21841,
    },
    {
        "keras_cls": convnext.ConvNeXtXLarge,
        "torch_name": "convnext_xlarge.fb_in22k_ft_in1k",
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
    },
    {
        "keras_cls": convnext.ConvNeXtXLarge,
        "torch_name": "convnext_xlarge.fb_in22k_ft_in1k_384",
        "input_shape": [384, 384, 3],
        "num_classes": 1000,
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
        torch_weight_name = re.sub(
            r"layer\.scale\.variable(\.\d+)?", "gamma", torch_weight_name
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
