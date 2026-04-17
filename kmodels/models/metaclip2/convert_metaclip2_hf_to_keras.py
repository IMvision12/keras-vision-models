from typing import Dict

import keras
import torch
from tqdm import tqdm
from transformers import AutoModel

from kmodels.models import metaclip2
from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "conv": "embeddings.patch_embedding",
    "class.embedding": "class_embedding",
    "pos.embed": "position_embedding.weight",
    "vision_model.layernorm.1": "vision_model.pre_layrnorm",
    "text_model.encoder": "text_model.encoder.layers",
    "vision_model.encoder": "vision_model.encoder.layers",
    "text_model.layernorm": "text_model.final_layer_norm",
    "layernorm.1": "layer_norm1",
    "layernorm.2": "layer_norm2",
    "vision_model.layer_norm2": "vision_model.post_layernorm",
    "text.projection": "text_projection",
    "visual.projection": "visual_projection",
    "logit_scale_logit_scale": "logit_scale",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}

attn_name_replace = {
    "text.model": "text_model",
    "vision.model": "vision_model",
    "encoder": "encoder.layers",
    "attn": "self_attn",
    "q.proj": "q_proj",
    "k.proj": "k_proj",
    "v.proj": "v_proj",
    "out.proj": "out_proj",
}


HF_REPO = {
    "MetaClip2WorldwideS16": "facebook/metaclip-2-worldwide-s16",
    "MetaClip2WorldwideS16_384": "facebook/metaclip-2-worldwide-s16-384",
    "MetaClip2WorldwideM16": "facebook/metaclip-2-worldwide-m16",
    "MetaClip2WorldwideM16_384": "facebook/metaclip-2-worldwide-m16-384",
    "MetaClip2WorldwideB16": "facebook/metaclip-2-worldwide-b16",
    "MetaClip2WorldwideB16_384": "facebook/metaclip-2-worldwide-b16-384",
    "MetaClip2WorldwideB32": "facebook/metaclip-2-worldwide-b32",
    "MetaClip2WorldwideB32_384": "facebook/metaclip-2-worldwide-b32-384",
    "MetaClip2WorldwideL14": "facebook/metaclip-2-worldwide-l14",
    "MetaClip2WorldwideHugeQuickgelu": "facebook/metaclip-2-worldwide-huge-quickgelu",
    "MetaClip2WorldwideHuge378": "facebook/metaclip-2-worldwide-huge-378",
    "MetaClip2WorldwideGiant": "facebook/metaclip-2-worldwide-giant",
    "MetaClip2WorldwideGiant378": "facebook/metaclip-2-worldwide-giant-378",
    "MetaClip2Mt5WorldwideS16": "facebook/metaclip-2-mt5-worldwide-s16",
    "MetaClip2Mt5WorldwideM16": "facebook/metaclip-2-mt5-worldwide-m16",
    "MetaClip2Mt5WorldwideB32": "facebook/metaclip-2-mt5-worldwide-b32",
}


def convert(variant: str):
    cfg = metaclip2.config.METACLIP2_MODEL_CONFIG[variant]
    image_size = cfg["image_resolution"]
    input_shape = (image_size, image_size, 3)
    keras_model: keras.Model = getattr(metaclip2, variant)(
        weights=None, input_shape=input_shape
    )
    torch_model: torch.nn.Module = AutoModel.from_pretrained(HF_REPO[variant]).eval()

    trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
        torch_model
    )
    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model
    )

    for keras_weight, keras_weight_name in tqdm(
        trainable_keras_weights + non_trainable_keras_weights,
        total=len(trainable_keras_weights + non_trainable_keras_weights),
        desc=f"Transferring {variant}",
    ):
        torch_weight_name: str = keras_weight_name
        for keras_name_part, torch_name_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )

        torch_weights_dict: Dict[str, torch.Tensor] = {
            **trainable_torch_weights,
            **non_trainable_torch_weights,
        }

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, torch_weights_dict, attn_name_replace
            )
            continue

        if keras_weight_name == "text_model_embedding_embeddings":
            if "token_embedding" in keras_weight.path:
                torch_token_embedding = (
                    torch_model.text_model.embeddings.token_embedding.weight
                )
                keras_weight.assign(torch_token_embedding.detach().cpu().numpy())
                continue
            elif "positional_embedding" in keras_weight.path:
                torch_position_embedding = (
                    torch_model.text_model.embeddings.position_embedding.weight
                )
                keras_weight.assign(torch_position_embedding.detach().cpu().numpy())
                continue

        if keras_weight_name == "logit_scale_logit_scale":
            torch_logit_scale = torch_model.logit_scale
            keras_weight.assign(torch_logit_scale.detach().cpu().numpy())
            continue

        if keras_weight_name == "vision_model_embeddings_pos_embed":
            torch_pos_embed = (
                torch_model.vision_model.embeddings.position_embedding.weight
            )
            torch_pos_embed_expanded = torch_pos_embed.unsqueeze(0)
            keras_weight.assign(torch_pos_embed_expanded.detach().cpu().numpy())
            continue

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

    weight_name = (
        f"{keras_model.name.lower()}_"
        f"{list(metaclip2.config.METACLIP2_WEIGHTS_CONFIG[keras_model.name].keys())[0]}"
        f".weights.h5"
    )
    total_gb = keras_model.count_params() * 4 / (1024**3)
    if total_gb > 2.0:
        keras_model.save_weights(weight_name, max_shard_size=2)
    else:
        keras_model.save_weights(weight_name)
    print(f"Saved {weight_name} ({total_gb:.2f} GB)")


if __name__ == "__main__":
    import sys

    variant = sys.argv[1] if len(sys.argv) > 1 else "MetaClip2WorldwideB32"
    convert(variant)
