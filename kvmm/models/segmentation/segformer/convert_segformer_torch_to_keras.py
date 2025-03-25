from typing import Dict

import keras
import numpy as np
import torch
from tqdm import tqdm

from kvmm.models import segformer
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.model_weights_util import download_weights
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "block": "backbone.block",
    "dense.1": "fc1",
    "dense.2": "fc2",
    "layernorm": "norm",
    ".conv": ".proj",
    "dwconv": "dwconv.dwconv",
    "overlap.patch.embed": "patch_embed",
    "patch_embed": "backbone.patch_embed",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "predictions": "head",
}

torch_model_path_mit_b0 = "https://huggingface.co/IMvision12/Test/resolve/main/segformer.b0.512x512.ade.160k.pth"  # TODO: change once repo is open sourced

keras_model: keras.Model = segformer.SegFormerB0(
    weights=None, num_classes=150, input_shape=(512, 512, 3)
)
temp_path = download_weights(torch_model_path_mit_b0)
torch_model: torch.nn.Module = torch.load(temp_path, map_location="cpu")

trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
    keras_model.backbone
)

for keras_weight, keras_weight_name in tqdm(
    trainable_keras_weights + non_trainable_keras_weights,
    total=len(trainable_keras_weights + non_trainable_keras_weights),
    desc="Transferring weights",
):
    torch_weight_name: str = keras_weight_name
    for keras_name_part, torch_name_part in weight_name_mapping.items():
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)

    torch_weights_dict: Dict[str, torch.Tensor] = torch_model

    if "attention" in torch_weight_name:
        transfer_attention_weights(
            keras_weight_name,
            keras_weight,
            torch_weights_dict["state_dict"],
            name_replacements={"block": "backbone.block"},
        )
        continue

    if torch_weight_name == "norm1.weight":
        torch_weight_name = torch_weight_name.replace("norm1", "backbone.norm1")
    if torch_weight_name == "norm1.bias":
        torch_weight_name = torch_weight_name.replace("norm1", "backbone.norm1")
    if torch_weight_name == "norm2.weight":
        torch_weight_name = torch_weight_name.replace("norm2", "backbone.norm2")
    if torch_weight_name == "norm2.bias":
        torch_weight_name = torch_weight_name.replace("norm2", "backbone.norm2")
    if torch_weight_name == "norm3.weight":
        torch_weight_name = torch_weight_name.replace("norm3", "backbone.norm3")
    if torch_weight_name == "norm3.bias":
        torch_weight_name = torch_weight_name.replace("norm3", "backbone.norm3")
    if torch_weight_name == "norm4.weight":
        torch_weight_name = torch_weight_name.replace("norm4", "backbone.norm4")
    if torch_weight_name == "norm4.bias":
        torch_weight_name = torch_weight_name.replace("norm4", "backbone.norm4")

    if torch_weight_name not in torch_weights_dict["state_dict"].keys():
        raise WeightMappingError(keras_weight_name, torch_weight_name)

    torch_weight: torch.Tensor = torch_weights_dict["state_dict"][torch_weight_name]

    if not compare_keras_torch_names(
        keras_weight_name, keras_weight, torch_weight_name, torch_weight
    ):
        raise WeightShapeMismatchError(
            keras_weight_name, keras_weight.shape, torch_weight_name, torch_weight.shape
        )

    transfer_weights(keras_weight_name, keras_weight, torch_weight)


pytorch_state_dict = torch_model["state_dict"]

# Linear C1 projection
keras_model.get_layer("SegFormer_B0_head_linear_c1").weights[0].assign(
    pytorch_state_dict["decode_head.linear_c1.proj.weight"].cpu().numpy().T
)
keras_model.get_layer("SegFormer_B0_head_linear_c1").weights[1].assign(
    pytorch_state_dict["decode_head.linear_c1.proj.bias"].cpu().numpy()
)

# Linear C2 projection
keras_model.get_layer("SegFormer_B0_head_linear_c2").weights[0].assign(
    pytorch_state_dict["decode_head.linear_c2.proj.weight"].cpu().numpy().T
)
keras_model.get_layer("SegFormer_B0_head_linear_c2").weights[1].assign(
    pytorch_state_dict["decode_head.linear_c2.proj.bias"].cpu().numpy()
)

# Linear C3 projection
keras_model.get_layer("SegFormer_B0_head_linear_c3").weights[0].assign(
    pytorch_state_dict["decode_head.linear_c3.proj.weight"].cpu().numpy().T
)
keras_model.get_layer("SegFormer_B0_head_linear_c3").weights[1].assign(
    pytorch_state_dict["decode_head.linear_c3.proj.bias"].cpu().numpy()
)

# Linear C4 projection
keras_model.get_layer("SegFormer_B0_head_linear_c4").weights[0].assign(
    pytorch_state_dict["decode_head.linear_c4.proj.weight"].cpu().numpy().T
)
keras_model.get_layer("SegFormer_B0_head_linear_c4").weights[1].assign(
    pytorch_state_dict["decode_head.linear_c4.proj.bias"].cpu().numpy()
)

# Conv2D (linear fuse conv)
conv_weight = pytorch_state_dict["decode_head.linear_fuse.conv.weight"].cpu().numpy()
conv_weight = np.transpose(conv_weight, (2, 3, 1, 0))
keras_model.get_layer("SegFormer_B0_head_fusion_conv").weights[0].assign(conv_weight)

# Batch Normalization
bn_layer = keras_model.get_layer("SegFormer_B0_head_fusion_bn")
bn_layer.weights[0].assign(
    pytorch_state_dict["decode_head.linear_fuse.bn.weight"].cpu().numpy()
)
bn_layer.weights[1].assign(
    pytorch_state_dict["decode_head.linear_fuse.bn.bias"].cpu().numpy()
)
bn_layer.weights[2].assign(
    pytorch_state_dict["decode_head.linear_fuse.bn.running_mean"].cpu().numpy()
)
bn_layer.weights[3].assign(
    pytorch_state_dict["decode_head.linear_fuse.bn.running_var"].cpu().numpy()
)

# Final Conv Layer
final_conv_weight = pytorch_state_dict["decode_head.linear_pred.weight"].cpu().numpy()
final_conv_weight = np.transpose(final_conv_weight, (2, 3, 1, 0))
keras_model.get_layer("SegFormer_B0_head_classifier").weights[0].assign(
    final_conv_weight
)
keras_model.get_layer("SegFormer_B0_head_classifier").weights[1].assign(
    pytorch_state_dict["decode_head.linear_pred.bias"].cpu().numpy()
)

# Save the model
keras_model.save("SegFormer_B0_ADE.keras")
