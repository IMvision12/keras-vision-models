from typing import Dict

import keras
import numpy as np
import torch
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from kvmm.models import segformer
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "block": "segformer.encoder.block",
    "patch.embed": "segformer.encoder.patch_embeddings",
    "layernorm": "layer_norm",
    "layer_norm.1": "layer_norm_1",
    "layer_norm.2": "layer_norm_2",
    "conv.proj": "proj",
    "dense.1": "dense1",
    "dense.2": "dense2",
    "dwconv": "dwconv.dwconv",
    "final": "segformer.encoder",
    "segformer.encoder.layer_norm_1": "segformer.encoder.layer_norm.1",
    "segformer.encoder.layer_norm_2": "segformer.encoder.layer_norm.2",
    "segformer.encoder.layer_norm_3": "segformer.encoder.layer_norm.3",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "predictions": "classifier",
}

attn_name_replace = {
    "block": "segformer.encoder.block",
    "attn.q": "attention.self.query",
    "attn.k": "attention.self.key",
    "attn.v": "attention.self.value",
    "attn.proj": "attention.output.dense",
    "attn.sr": "attention.self.sr",
    "attn.norm": "attention.self.layer_norm",
}

keras_model: keras.Model = segformer.SegFormerB0(
    weights=None,
    num_classes=150,
    input_shape=(512, 512, 3),
    backbone=None,
)
torch_model: torch.nn.Module = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
).eval()
trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
    torch_model
)
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

    torch_weights_dict: Dict[str, torch.Tensor] = {
        **trainable_torch_weights,
        **non_trainable_torch_weights,
    }

    if "attention" in torch_weight_name:
        transfer_attention_weights(
            keras_weight_name, keras_weight, torch_weights_dict, attn_name_replace
        )
        continue

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


pytorch_state_dict = torch_model.state_dict()

# Linear C1 projection
keras_model.get_layer("SegFormerB0_head_linear_c1").weights[0].assign(
    pytorch_state_dict["decode_head.linear_c.0.proj.weight"].cpu().numpy().T
)
keras_model.get_layer("SegFormerB0_head_linear_c1").weights[1].assign(
    pytorch_state_dict["decode_head.linear_c.0.proj.bias"].cpu().numpy()
)

# Linear C2 projection
keras_model.get_layer("SegFormerB0_head_linear_c2").weights[0].assign(
    pytorch_state_dict["decode_head.linear_c.1.proj.weight"].cpu().numpy().T
)
keras_model.get_layer("SegFormerB0_head_linear_c2").weights[1].assign(
    pytorch_state_dict["decode_head.linear_c.1.proj.bias"].cpu().numpy()
)

# Linear C3 projection
keras_model.get_layer("SegFormerB0_head_linear_c3").weights[0].assign(
    pytorch_state_dict["decode_head.linear_c.2.proj.weight"].cpu().numpy().T
)
keras_model.get_layer("SegFormerB0_head_linear_c3").weights[1].assign(
    pytorch_state_dict["decode_head.linear_c.2.proj.bias"].cpu().numpy()
)

# Linear C4 projection
keras_model.get_layer("SegFormerB0_head_linear_c4").weights[0].assign(
    pytorch_state_dict["decode_head.linear_c.3.proj.weight"].cpu().numpy().T
)
keras_model.get_layer("SegFormerB0_head_linear_c4").weights[1].assign(
    pytorch_state_dict["decode_head.linear_c.3.proj.bias"].cpu().numpy()
)

# Conv2D (linear fuse conv)
conv_weight = pytorch_state_dict["decode_head.linear_fuse.weight"].cpu().numpy()
conv_weight = np.transpose(conv_weight, (2, 3, 1, 0))
keras_model.get_layer("SegFormerB0_head_fusion_conv").weights[0].assign(conv_weight)

# Batch Normalization
bn_layer = keras_model.get_layer("SegFormerB0_head_fusion_bn")
bn_layer.weights[0].assign(
    pytorch_state_dict["decode_head.batch_norm.weight"].cpu().numpy()
)
bn_layer.weights[1].assign(
    pytorch_state_dict["decode_head.batch_norm.bias"].cpu().numpy()
)
bn_layer.weights[2].assign(
    pytorch_state_dict["decode_head.batch_norm.running_mean"].cpu().numpy()
)
bn_layer.weights[3].assign(
    pytorch_state_dict["decode_head.batch_norm.running_var"].cpu().numpy()
)

# Final Conv Layer
final_conv_weight = pytorch_state_dict["decode_head.classifier.weight"].cpu().numpy()
final_conv_weight = np.transpose(final_conv_weight, (2, 3, 1, 0))
keras_model.get_layer("SegFormerB0_head_classifier").weights[0].assign(
    final_conv_weight
)
keras_model.get_layer("SegFormerB0_head_classifier").weights[1].assign(
    pytorch_state_dict["decode_head.classifier.bias"].cpu().numpy()
)

# Save the model
keras_model.save_weights("SegFormer_B0_ade.weights.h5")
