from typing import Dict, List, Union

import keras
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from transformers import DetrForObjectDetection

from kmodels.models import detr
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "backbone_layer": "model.backbone.model.layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.bn": "downsample.1",
    "backbone.conv1": "model.backbone.model.conv1",
    "backbone.bn1": "model.backbone.model.bn1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}

model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": detr.DETRResNet50,
    "hf_model_name": "facebook/detr-resnet-50",
    "input_shape": [800, 800, 3],
    "num_classes": 92,
    "num_queries": 100,
    "include_normalization": False,
}

keras_model: keras.Model = model_config["keras_model_cls"](
    weights=None,
    input_shape=model_config["input_shape"],
    num_classes=model_config["num_classes"],
    num_queries=model_config["num_queries"],
    include_normalization=model_config["include_normalization"],
)

torch_model: torch.nn.Module = DetrForObjectDetection.from_pretrained(
    model_config["hf_model_name"]
).eval()

pytorch_state_dict = {k: v.cpu().numpy() for k, v in torch_model.state_dict().items()}

backbone_layers = [
    layer for layer in keras_model.layers if layer.name.startswith("backbone_")
]

backbone_trainable = []
backbone_non_trainable = []
for layer in backbone_layers:
    for weight in layer.trainable_weights:
        backbone_trainable.append((weight, f"{layer.name}_{weight.name}"))
    for weight in layer.non_trainable_weights:
        backbone_non_trainable.append((weight, f"{layer.name}_{weight.name}"))

for keras_weight, keras_weight_name in tqdm(
    backbone_trainable + backbone_non_trainable,
    total=len(backbone_trainable + backbone_non_trainable),
    desc="Transferring backbone weights",
):
    torch_weight_name: str = keras_weight_name
    for keras_name_part, torch_name_part in weight_name_mapping.items():
        torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)

    torch_weights_dict: Dict[str, np.ndarray] = pytorch_state_dict

    if torch_weight_name not in torch_weights_dict:
        raise WeightMappingError(keras_weight_name, torch_weight_name)

    torch_weight = torch_weights_dict[torch_weight_name]

    if not compare_keras_torch_names(
        keras_weight_name, keras_weight, torch_weight_name, torch_weight
    ):
        raise WeightShapeMismatchError(
            keras_weight_name, keras_weight.shape, torch_weight_name, torch_weight.shape
        )

    transfer_weights(keras_weight_name, keras_weight, torch_weight)

input_proj = keras_model.get_layer("input_projection")
conv_w = pytorch_state_dict["model.input_projection.weight"]
input_proj.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))
input_proj.weights[1].assign(pytorch_state_dict["model.input_projection.bias"])

query_layer = keras_model.get_layer("query_position_embeddings")
query_layer.weights[0].assign(
    pytorch_state_dict["model.query_position_embeddings.weight"]
)

for i in tqdm(range(6), desc="Transferring encoder weights"):
    hf_prefix = f"model.encoder.layers.{i}"
    enc_layer = keras_model.get_layer(f"encoder_layers_{i}")

    # Self attention: q_proj, k_proj, v_proj, o_proj (out_proj in keras)
    for proj, hf_proj in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", "o_proj"),
    ]:
        dense = getattr(enc_layer.self_attn, proj)
        w = pytorch_state_dict[f"{hf_prefix}.self_attn.{hf_proj}.weight"]
        dense.weights[0].assign(w.T)
        dense.weights[1].assign(
            pytorch_state_dict[f"{hf_prefix}.self_attn.{hf_proj}.bias"]
        )

    # Self attention layer norm
    ln = enc_layer.self_attn_layer_norm
    ln.weights[0].assign(pytorch_state_dict[f"{hf_prefix}.self_attn_layer_norm.weight"])
    ln.weights[1].assign(pytorch_state_dict[f"{hf_prefix}.self_attn_layer_norm.bias"])

    # FFN
    fc1_w = pytorch_state_dict[f"{hf_prefix}.mlp.fc1.weight"]
    enc_layer.fc1.weights[0].assign(fc1_w.T)
    enc_layer.fc1.weights[1].assign(pytorch_state_dict[f"{hf_prefix}.mlp.fc1.bias"])

    fc2_w = pytorch_state_dict[f"{hf_prefix}.mlp.fc2.weight"]
    enc_layer.fc2.weights[0].assign(fc2_w.T)
    enc_layer.fc2.weights[1].assign(pytorch_state_dict[f"{hf_prefix}.mlp.fc2.bias"])

    # Final layer norm
    ln2 = enc_layer.final_layer_norm
    ln2.weights[0].assign(pytorch_state_dict[f"{hf_prefix}.final_layer_norm.weight"])
    ln2.weights[1].assign(pytorch_state_dict[f"{hf_prefix}.final_layer_norm.bias"])


for i in tqdm(range(6), desc="Transferring decoder weights"):
    hf_prefix = f"model.decoder.layers.{i}"
    dec_layer = keras_model.get_layer(f"decoder_layers_{i}")

    # Self attention
    for proj, hf_proj in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", "o_proj"),
    ]:
        dense = getattr(dec_layer.self_attn, proj)
        w = pytorch_state_dict[f"{hf_prefix}.self_attn.{hf_proj}.weight"]
        dense.weights[0].assign(w.T)
        dense.weights[1].assign(
            pytorch_state_dict[f"{hf_prefix}.self_attn.{hf_proj}.bias"]
        )

    ln = dec_layer.self_attn_layer_norm
    ln.weights[0].assign(pytorch_state_dict[f"{hf_prefix}.self_attn_layer_norm.weight"])
    ln.weights[1].assign(pytorch_state_dict[f"{hf_prefix}.self_attn_layer_norm.bias"])

    # Cross attention (encoder_attn)
    for proj, hf_proj in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", "o_proj"),
    ]:
        dense = getattr(dec_layer.cross_attn, proj)
        w = pytorch_state_dict[f"{hf_prefix}.encoder_attn.{hf_proj}.weight"]
        dense.weights[0].assign(w.T)
        dense.weights[1].assign(
            pytorch_state_dict[f"{hf_prefix}.encoder_attn.{hf_proj}.bias"]
        )

    ln = dec_layer.cross_attn_layer_norm
    ln.weights[0].assign(
        pytorch_state_dict[f"{hf_prefix}.encoder_attn_layer_norm.weight"]
    )
    ln.weights[1].assign(
        pytorch_state_dict[f"{hf_prefix}.encoder_attn_layer_norm.bias"]
    )

    # FFN
    fc1_w = pytorch_state_dict[f"{hf_prefix}.mlp.fc1.weight"]
    dec_layer.fc1.weights[0].assign(fc1_w.T)
    dec_layer.fc1.weights[1].assign(pytorch_state_dict[f"{hf_prefix}.mlp.fc1.bias"])

    fc2_w = pytorch_state_dict[f"{hf_prefix}.mlp.fc2.weight"]
    dec_layer.fc2.weights[0].assign(fc2_w.T)
    dec_layer.fc2.weights[1].assign(pytorch_state_dict[f"{hf_prefix}.mlp.fc2.bias"])

    # Final layer norm
    ln2 = dec_layer.final_layer_norm
    ln2.weights[0].assign(pytorch_state_dict[f"{hf_prefix}.final_layer_norm.weight"])
    ln2.weights[1].assign(pytorch_state_dict[f"{hf_prefix}.final_layer_norm.bias"])

# Decoder layernorm
dec_ln = keras_model.get_layer("decoder_layernorm")
dec_ln.weights[0].assign(pytorch_state_dict["model.decoder.layernorm.weight"])
dec_ln.weights[1].assign(pytorch_state_dict["model.decoder.layernorm.bias"])

# Class head
cls_layer = keras_model.get_layer("class_labels_classifier")
cls_layer.weights[0].assign(pytorch_state_dict["class_labels_classifier.weight"].T)
cls_layer.weights[1].assign(pytorch_state_dict["class_labels_classifier.bias"])

# BBox head (3-layer MLP)
for layer_idx in range(3):
    bbox_layer = keras_model.get_layer(f"bbox_predictor_{layer_idx}")
    bbox_layer.weights[0].assign(
        pytorch_state_dict[f"bbox_predictor.layers.{layer_idx}.weight"].T
    )
    bbox_layer.weights[1].assign(
        pytorch_state_dict[f"bbox_predictor.layers.{layer_idx}.bias"]
    )

print("\nVerifying model equivalence...")

np.random.seed(42)
test_input = np.random.rand(1, 800, 800, 3).astype(np.float32)

hf_input = torch.tensor(test_input).permute(0, 3, 1, 2)
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
hf_input_norm = normalize(hf_input)

with torch.no_grad():
    hf_output = torch_model(hf_input_norm)
    hf_logits = hf_output.logits.numpy()
    hf_boxes = hf_output.pred_boxes.numpy()

mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
keras_input_norm = (test_input - mean) / std

keras_output = keras_model(keras_input_norm.astype(np.float32), training=False)
keras_logits = keras.ops.convert_to_numpy(keras_output["logits"])
keras_boxes = keras.ops.convert_to_numpy(keras_output["pred_boxes"])

logits_diff = np.max(np.abs(hf_logits - keras_logits))
boxes_diff = np.max(np.abs(hf_boxes - keras_boxes))

print(f"Max logits diff:  {logits_diff:.6f}")
print(f"Max boxes diff:   {boxes_diff:.6f}")

if logits_diff > 1e-3 or boxes_diff > 1e-3:
    raise ValueError(
        "Model equivalence test failed - model outputs do not match "
        f"(logits: {logits_diff:.6f}, boxes: {boxes_diff:.6f})"
    )

print("Model equivalence test passed!")


model_filename: str = (
    f"{model_config['hf_model_name'].split('/')[-1].replace('-', '_')}_coco.weights.h5"
)
keras_model.save_weights(model_filename)
print(f"Model saved successfully as {model_filename}")
