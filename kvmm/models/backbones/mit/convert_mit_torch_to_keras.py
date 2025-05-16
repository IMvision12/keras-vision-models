from typing import Dict, List, Union

import keras
import torch
from tqdm import tqdm
from transformers import SegformerForImageClassification

from kvmm.models import mit
from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.model_equivalence_tester import verify_cls_model_equivalence
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
model_config: Dict[str, Union[type, str, List[int], int, bool]] = {
    "keras_model_cls": mit.MiT_B0,
    "input_shape": [224, 224, 3],
    "num_classes": 1000,
    "include_top": True,
    "include_normalization": False,
    "classifier_activation": "linear",
}


keras_model: keras.Model = model_config["keras_model_cls"](
    include_top=model_config["include_top"],
    input_shape=model_config["input_shape"],
    classifier_activation=model_config["classifier_activation"],
    num_classes=model_config["num_classes"],
    include_normalization=model_config["include_normalization"],
    weights=None,
)

torch_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0").eval()

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

results = verify_cls_model_equivalence(
    model_a=torch_model,
    model_b=keras_model,
    input_shape=(224, 224, 3),
    comparison_type="hf_to_keras",
    output_specs={"num_classes": 1000},
    run_performance=False,
    atol=1e-3,
    rtol=1e-3,
)

if not results["standard_input"]:
    raise ValueError(
        "Model equivalence test failed - model outputs do not match for standard input"
    )

model_filename: str = f"{keras_model.name}.weights.h5"
keras_model.save_weights(model_filename)
print(f"Model saved successfully as {model_filename}")
