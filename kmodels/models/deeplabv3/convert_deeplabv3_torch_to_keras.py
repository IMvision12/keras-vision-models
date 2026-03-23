from typing import Dict

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

classifier_weight_name_mapping: Dict[str, str] = {
    "classifier.0.convs": "classifier.0.convs",
    "classifier.0.project": "classifier.0.project",
    "classifier.1": "classifier.1",
    "classifier.2": "classifier.2",
    "classifier.4": "classifier.4",
    "_": ".",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}


def convert_model(
    variant: str = "DeepLabV3ResNet50",
    input_shape=(520, 520, 3),
):
    """Convert torchvision DeepLabV3 weights to Keras format.

    Args:
        variant: "DeepLabV3ResNet50" or "DeepLabV3ResNet101"
        input_shape: Input shape tuple (H, W, C).
    """
    if variant == "DeepLabV3ResNet50":
        torch_model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        ).eval()
        keras_model_fn = deeplabv3.DeepLabV3ResNet50
    elif variant == "DeepLabV3ResNet101":
        torch_model = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        ).eval()
        keras_model_fn = deeplabv3.DeepLabV3ResNet101
    else:
        raise ValueError(f"Unknown variant: {variant}")

    keras_model = keras_model_fn(
        weights=None,
        input_shape=input_shape,
        num_classes=21,
        include_normalization=False,
    )

    pytorch_state_dict = {
        k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
    }

    # Transfer backbone weights
    backbone_layers = [
        layer for layer in keras_model.layers if layer.name.startswith("backbone_")
    ]

    backbone_weights = []
    for layer in backbone_layers:
        for weight in layer.trainable_weights:
            backbone_weights.append((weight, f"{layer.name}_{weight.name}"))
        for weight in layer.non_trainable_weights:
            backbone_weights.append((weight, f"{layer.name}_{weight.name}"))

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

    _transfer_conv_bn(
        keras_model,
        pytorch_state_dict,
        keras_conv_name="classifier_0_convs_0_0",
        keras_bn_name="classifier_0_convs_0_1",
        torch_conv_key="classifier.0.convs.0.0",
        torch_bn_key="classifier.0.convs.0.1",
    )

    for i in range(1, 4):
        _transfer_conv_bn(
            keras_model,
            pytorch_state_dict,
            keras_conv_name=f"classifier_0_convs_{i}_0",
            keras_bn_name=f"classifier_0_convs_{i}_1",
            torch_conv_key=f"classifier.0.convs.{i}.0",
            torch_bn_key=f"classifier.0.convs.{i}.1",
        )

    _transfer_conv_bn(
        keras_model,
        pytorch_state_dict,
        keras_conv_name="classifier_0_convs_4_1",
        keras_bn_name="classifier_0_convs_4_2",
        torch_conv_key="classifier.0.convs.4.1",
        torch_bn_key="classifier.0.convs.4.2",
    )

    _transfer_conv_bn(
        keras_model,
        pytorch_state_dict,
        keras_conv_name="classifier_0_project_0",
        keras_bn_name="classifier_0_project_1",
        torch_conv_key="classifier.0.project.0",
        torch_bn_key="classifier.0.project.1",
    )

    _transfer_conv_bn(
        keras_model,
        pytorch_state_dict,
        keras_conv_name="classifier_1",
        keras_bn_name="classifier_2",
        torch_conv_key="classifier.1",
        torch_bn_key="classifier.2",
    )

    cls_layer = keras_model.get_layer("classifier_4")
    conv_w = pytorch_state_dict["classifier.4.weight"]
    cls_layer.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))
    cls_layer.weights[1].assign(pytorch_state_dict["classifier.4.bias"])

    print("\nVerifying model equivalence...")
    np.random.seed(42)
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

    from PIL import Image

    torch_h, torch_w = torch_output_np.shape[1], torch_output_np.shape[2]
    keras_h, keras_w = keras_output_np.shape[1], keras_output_np.shape[2]

    if torch_h != keras_h or torch_w != keras_w:
        keras_resized = np.array(
            Image.fromarray(keras_output_np[0, :, :, 0]).resize(
                (torch_w, torch_h), Image.BILINEAR
            )
        )
        diff = np.max(np.abs(torch_output_np[0, :, :, 0] - keras_resized))
        print(
            f"Note: Comparing at different resolutions (torch: {torch_h}x{torch_w}, keras: {keras_h}x{keras_w})"
        )
        print(f"Max diff (channel 0, resized): {diff:.6f}")

        print("Building model without upsampling for exact comparison...")
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
        f"deeplabv3_{variant.replace('DeepLabV3', '').lower()}_coco_voc.weights.h5"
    )
    keras_model.save_weights(model_filename)
    print(f"Model saved as {model_filename}")

    return keras_model


def _transfer_conv_bn(
    keras_model,
    pytorch_state_dict,
    keras_conv_name,
    keras_bn_name,
    torch_conv_key,
    torch_bn_key,
):
    conv_layer = keras_model.get_layer(keras_conv_name)
    conv_w = pytorch_state_dict[f"{torch_conv_key}.weight"]
    conv_layer.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))

    bn_layer = keras_model.get_layer(keras_bn_name)
    bn_layer.weights[0].assign(pytorch_state_dict[f"{torch_bn_key}.weight"])
    bn_layer.weights[1].assign(pytorch_state_dict[f"{torch_bn_key}.bias"])
    bn_layer.weights[2].assign(pytorch_state_dict[f"{torch_bn_key}.running_mean"])
    bn_layer.weights[3].assign(pytorch_state_dict[f"{torch_bn_key}.running_var"])


if __name__ == "__main__":
    print("Converting DeepLabV3-ResNet50...")
    convert_model("DeepLabV3ResNet50", input_shape=(520, 520, 3))

    print("\n" + "=" * 60 + "\n")

    print("Converting DeepLabV3-ResNet101...")
    convert_model("DeepLabV3ResNet101", input_shape=(520, 520, 3))
