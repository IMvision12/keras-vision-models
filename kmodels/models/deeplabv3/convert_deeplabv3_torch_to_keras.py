"""
Script to convert DeepLabV3 PyTorch weights to Keras 3 format.

Usage:
    python convert_deeplabv3_torch_to_keras.py
"""

from typing import Dict, List, Union

import keras
import numpy as np
import torch
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)
from tqdm import tqdm

from kmodels.models import deeplabv3
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "backbone_layer": "backbone.layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.bn": "downsample.1",
    "backbone.conv1": "backbone.conv1",
    "backbone.bn1": "backbone.bn1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}

classifier_conv_bn_pairs = [
    (
        "classifier_0_convs_0_0",
        "classifier_0_convs_0_1",
        "classifier.0.convs.0.0",
        "classifier.0.convs.0.1",
    ),
    (
        "classifier_0_convs_1_0",
        "classifier_0_convs_1_1",
        "classifier.0.convs.1.0",
        "classifier.0.convs.1.1",
    ),
    (
        "classifier_0_convs_2_0",
        "classifier_0_convs_2_1",
        "classifier.0.convs.2.0",
        "classifier.0.convs.2.1",
    ),
    (
        "classifier_0_convs_3_0",
        "classifier_0_convs_3_1",
        "classifier.0.convs.3.0",
        "classifier.0.convs.3.1",
    ),
    (
        "classifier_0_convs_4_1",
        "classifier_0_convs_4_2",
        "classifier.0.convs.4.1",
        "classifier.0.convs.4.2",
    ),
    (
        "classifier_0_project_0",
        "classifier_0_project_1",
        "classifier.0.project.0",
        "classifier.0.project.1",
    ),
    ("classifier_1", "classifier_2", "classifier.1", "classifier.2"),
]

model_configs: List[Dict[str, Union[type, str, List[int], int, bool]]] = [
    {
        "keras_model_cls": deeplabv3.DeepLabV3ResNet50,
        "torch_model_fn": deeplabv3_resnet50,
        "torch_weights": DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        "variant_name": "DeepLabV3ResNet50",
        "input_shape": [520, 520, 3],
        "num_classes": 21,
        "include_normalization": False,
    },
    {
        "keras_model_cls": deeplabv3.DeepLabV3ResNet101,
        "torch_model_fn": deeplabv3_resnet101,
        "torch_weights": DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
        "variant_name": "DeepLabV3ResNet101",
        "input_shape": [520, 520, 3],
        "num_classes": 21,
        "include_normalization": False,
    },
]

for model_config in model_configs:
    print(f"\n{'=' * 60}")
    print(f"Converting {model_config['variant_name']}...")
    print(f"{'=' * 60}")

    keras_model: keras.Model = model_config["keras_model_cls"](
        weights=None,
        input_shape=model_config["input_shape"],
        num_classes=model_config["num_classes"],
        include_normalization=model_config["include_normalization"],
    )

    torch_model: torch.nn.Module = model_config["torch_model_fn"](
        weights=model_config["torch_weights"]
    ).eval()

    trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
        torch_model
    )
    pytorch_state_dict = {**trainable_torch_weights, **non_trainable_torch_weights}

    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model
    )
    all_keras_weights = trainable_keras_weights + non_trainable_keras_weights

    # Filter to backbone-only weights for name-mapping transfer
    backbone_weights = [
        (w, name) for w, name in all_keras_weights if name.startswith("backbone_")
    ]

    for keras_weight, keras_weight_name in tqdm(
        backbone_weights,
        total=len(backbone_weights),
        desc="Transferring backbone weights",
    ):
        torch_weight_name = keras_weight_name
        for keras_part, torch_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(keras_part, torch_part)

        if torch_weight_name not in pytorch_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = pytorch_state_dict[torch_weight_name]

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

    for k_conv, k_bn, pt_conv, pt_bn in tqdm(
        classifier_conv_bn_pairs,
        desc="Transferring classifier weights",
    ):
        conv_layer = keras_model.get_layer(k_conv)
        conv_w = pytorch_state_dict[f"{pt_conv}.weight"].numpy()
        conv_layer.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))

        bn_layer = keras_model.get_layer(k_bn)
        bn_layer.weights[0].assign(pytorch_state_dict[f"{pt_bn}.weight"].numpy())
        bn_layer.weights[1].assign(pytorch_state_dict[f"{pt_bn}.bias"].numpy())
        bn_layer.weights[2].assign(pytorch_state_dict[f"{pt_bn}.running_mean"].numpy())
        bn_layer.weights[3].assign(pytorch_state_dict[f"{pt_bn}.running_var"].numpy())

    cls_layer = keras_model.get_layer("classifier_4")
    conv_w = pytorch_state_dict["classifier.4.weight"].numpy()
    cls_layer.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))
    cls_layer.weights[1].assign(pytorch_state_dict["classifier.4.bias"].numpy())

    print("\nVerifying model equivalence...")
    np.random.seed(42)
    input_shape = model_config["input_shape"]
    test_input = np.random.rand(1, *input_shape).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    normalized_input = (test_input - mean) / std

    torch_input = torch.tensor(normalized_input).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        torch_output = torch_model(torch_input)["out"]
        torch_output_np = torch_output.permute(0, 2, 3, 1).numpy()

    keras_output = keras_model(normalized_input.astype(np.float32), training=False)
    keras_output_np = keras.ops.convert_to_numpy(keras_output)

    torch_h, torch_w = torch_output_np.shape[1], torch_output_np.shape[2]
    keras_h, keras_w = keras_output_np.shape[1], keras_output_np.shape[2]

    if torch_h != keras_h or torch_w != keras_w:
        print(
            f"Note: Spatial size mismatch (torch: {torch_h}x{torch_w}, "
            f"keras: {keras_h}x{keras_w}), comparing pre-upsample output..."
        )
        keras_model_no_upsample = keras.Model(
            inputs=keras_model.input,
            outputs=keras_model.layers[-2].output,
        )
        keras_raw = keras_model_no_upsample(
            normalized_input.astype(np.float32), training=False
        )
        keras_raw_np = keras.ops.convert_to_numpy(keras_raw)
        max_diff = np.max(np.abs(torch_output_np - keras_raw_np))
    else:
        max_diff = np.max(np.abs(torch_output_np - keras_output_np))

    print(f"Max output diff: {max_diff:.6f}")

    if max_diff > 1e-3:
        raise ValueError(
            f"Model equivalence test failed - outputs do not match (diff: {max_diff:.6f})"
        )

    print("Model equivalence test passed!")

    model_filename = (
        f"deeplabv3_{model_config['variant_name'].replace('DeepLabV3', '').lower()}"
        "_coco_voc.weights.h5"
    )
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")

    del keras_model, torch_model, pytorch_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
