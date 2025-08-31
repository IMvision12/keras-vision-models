from typing import Dict

import keras
import torch
from tqdm import tqdm

from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)
from ultralytics import YOLO
from kvmm.models import yolo

weight_name_mapping: Dict[str, str] = {
    "cv2_new_conv":"cv2",
    "cv3_new_conv":"cv3",
    "_bias":".bias",
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
}

keras_model = yolo.YoloV5s(input_shape=(640, 640, 3), nc=80)
torch_model = YOLO("yolov5su.pt").eval()

trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
    torch_model
)
trainable_keras_weights, non_trainable_keras_weights = split_model_weights(keras_model)

for keras_weight, keras_weight_name in tqdm(
    trainable_keras_weights + non_trainable_keras_weights,
    total=len(trainable_keras_weights + non_trainable_keras_weights),
    desc="Transferring weights",
):
    torch_weight_name: str = keras_weight_name
    for keras_name_part, torch_name_part in weight_name_mapping.items():
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)

    torch_weights_dict: Dict[str, torch.Tensor] = {
        **trainable_torch_weights,
        **non_trainable_torch_weights,
    }

    if torch_weight_name not in torch_weights_dict:
        raise WeightMappingError(keras_weight_name, torch_weight_name)

    torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

    if not compare_keras_torch_names(
        keras_weight_name, keras_weight, torch_weight_name, torch_weight
    ):
        raise WeightShapeMismatchError(
            keras_weight_name, keras_weight.shape, torch_weight_name, torch_weight.shape
        )
    transfer_weights(keras_weight_name, keras_weight, torch_weight)