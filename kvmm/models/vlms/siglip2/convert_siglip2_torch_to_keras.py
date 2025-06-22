import re
from typing import Dict

import keras
import torch
from tqdm import tqdm
from transformers import AutoModel

from kvmm.models import siglip2
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "patch.embedding.conv": "patch_embedding",
    "position.embedding.embeddings": "position_embedding.weight",
    "token.embedding.embeddings": "token_embedding.weight",
    "text_model.post_layernorm": "text_model.final_layer_norm",
    "layernorm.1": "layer_norm1",
    "layernorm.2": "layer_norm2",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "vision_model.final.layernorm": "vision_model.post_layernorm",
    "text_model.final.layernorm": "text_model.final_layer_norm",
    "probe.probe": "probe",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}

attn_name_replace = {
    "_": ".",
    "self.attn": "self_attn",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "in.proj": "in_proj",
    "out.proj": "out_proj",
    "q.proj": "q_proj",
    "k.proj": "k_proj",
    "v.proj": "v_proj",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}

keras_model: keras.Model = siglip2.SigLIP2BaseP16(
    weights=None, input_shape=(224, 224, 3)
)
torch_model: torch.nn.Module = AutoModel.from_pretrained(
    "google/siglip2-base-patch16-224"
).eval()

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

    if "attention" in torch_weight_name:
        if "in_proj" in keras_weight.path:
            if "kernel" in keras_weight.path:
                torch_in_proj_weight = (
                    torch_model.vision_model.head.attention.in_proj_weight.detach()
                    .numpy()
                    .T
                )
                keras_weight.assign(torch_in_proj_weight)
            else:
                torch_in_proj_bias = torch_model.vision_model.head.attention.in_proj_bias.detach().numpy()
                keras_weight.assign(torch_in_proj_bias)
            continue
        transfer_attention_weights(
            keras_weight_name, keras_weight, torch_weights_dict, attn_name_replace
        )
        continue

    # Handle probe weights
    if "probe" in torch_weight_name:
        keras_weight.assign(torch_model.vision_model.head.probe.detach().cpu().numpy())
        continue

    if "logit" in torch_weight_name:
        if torch_weight_name.split(".")[-1] == "scale":
            torch_weight_name = re.sub(
                r"logit.scale.bias.\d+.logit.scale", "logit_scale", torch_weight_name
            )
            keras_weight.assign(torch_model.logit_scale[0].detach().cpu().numpy())
        else:
            torch_weight_name = re.sub(
                r"logit.bias.\d+.logit.bias", "logit_bias", torch_weight_name
            )
            keras_weight.assign(torch_model.logit_bias[0].detach().cpu().numpy())
        continue

    if "position.ids" in torch_weight_name:
        if "vision_model" in torch_weight_name:
            keras_weight.assign(
                torch_model.vision_model.embeddings.position_ids.detach().cpu().numpy()
            )
        elif "text_model" in torch_weight_name:
            keras_weight.assign(
                torch_model.text_model.embeddings.position_ids.detach().cpu().numpy()
            )
        continue

    if "head.attention" in torch_weight_name:
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


# Save the model
weight_name = f"{keras_model.name.lower()}_{list(siglip2.config.SigLIP2_WEIGHTS_CONFIG[keras_model.name].keys())[0]}.weights.h5"
keras_model.save_weights(weight_name)  # use max_shard_size if >2GB
