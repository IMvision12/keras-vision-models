import gc
import re
from typing import Dict, List, Tuple, Type

import keras
import timm
import torch
from tqdm import tqdm

from kmodels.models import vit
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.model_equivalence_tester import verify_cls_model_equivalence
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "conv1": "patch_embed.proj",
    "pos.embed.pos.embed": "pos_embed",
    "cls.token.cls.token": "cls_token",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head",
}

# (keras_model_cls, timm_base_name, variant_key, resolution, num_classes)
VIT_WEIGHTS_CONFIG: List[Tuple[Type[keras.Model], str, str, int, int]] = [
    # ViTTiny16
    (vit.ViTTiny16, "vit_tiny_patch16", "augreg_in21k_ft_in1k", 224, 1000),
    (vit.ViTTiny16, "vit_tiny_patch16", "augreg_in21k_ft_in1k", 384, 1000),
    (vit.ViTTiny16, "vit_tiny_patch16", "augreg_in21k", 224, 21843),
    # ViTSmall16
    (vit.ViTSmall16, "vit_small_patch16", "augreg_in21k_ft_in1k", 224, 1000),
    (vit.ViTSmall16, "vit_small_patch16", "augreg_in21k_ft_in1k", 384, 1000),
    (vit.ViTSmall16, "vit_small_patch16", "augreg_in1k", 224, 1000),
    (vit.ViTSmall16, "vit_small_patch16", "augreg_in1k", 384, 1000),
    (vit.ViTSmall16, "vit_small_patch16", "augreg_in21k", 224, 21843),
    # ViTSmall32
    (vit.ViTSmall32, "vit_small_patch32", "augreg_in21k_ft_in1k", 224, 1000),
    (vit.ViTSmall32, "vit_small_patch32", "augreg_in21k_ft_in1k", 384, 1000),
    (vit.ViTSmall32, "vit_small_patch32", "augreg_in21k", 224, 21843),
    # ViTBase16
    (vit.ViTBase16, "vit_base_patch16", "augreg_in21k_ft_in1k", 224, 1000),
    (vit.ViTBase16, "vit_base_patch16", "augreg_in21k_ft_in1k", 384, 1000),
    (vit.ViTBase16, "vit_base_patch16", "orig_in21k_ft_in1k", 224, 1000),
    (vit.ViTBase16, "vit_base_patch16", "orig_in21k_ft_in1k", 384, 1000),
    (vit.ViTBase16, "vit_base_patch16", "augreg_in1k", 224, 1000),
    (vit.ViTBase16, "vit_base_patch16", "augreg_in1k", 384, 1000),
    (vit.ViTBase16, "vit_base_patch16", "augreg_in21k", 224, 21843),
    # ViTBase32
    (vit.ViTBase32, "vit_base_patch32", "augreg_in21k_ft_in1k", 224, 1000),
    (vit.ViTBase32, "vit_base_patch32", "augreg_in21k_ft_in1k", 384, 1000),
    (vit.ViTBase32, "vit_base_patch32", "augreg_in1k", 224, 1000),
    (vit.ViTBase32, "vit_base_patch32", "augreg_in1k", 384, 1000),
    (vit.ViTBase32, "vit_base_patch32", "augreg_in21k", 224, 21843),
    # ViTLarge16
    (vit.ViTLarge16, "vit_large_patch16", "augreg_in21k_ft_in1k", 224, 1000),
    (vit.ViTLarge16, "vit_large_patch16", "augreg_in21k_ft_in1k", 384, 1000),
    (vit.ViTLarge16, "vit_large_patch16", "augreg_in21k", 224, 21843),
    # ViTLarge32
    (vit.ViTLarge32, "vit_large_patch32", "orig_in21k_ft_in1k", 384, 1000),
]

for keras_model_cls, timm_base, variant, resolution, num_classes in VIT_WEIGHTS_CONFIG:
    torch_model_name = f"{timm_base}_{resolution}.{variant}"
    print(f"\n{'=' * 60}")
    print(f"Converting: {torch_model_name}")
    print(f"{'=' * 60}")

    input_shape = [resolution, resolution, 3]

    keras_model: keras.Model = keras_model_cls(
        include_top=True,
        input_shape=input_shape,
        classifier_activation="linear",
        num_classes=num_classes,
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
            r"pos_embed_variable_\d+$", "pos_embed", torch_weight_name
        )
        torch_weight_name = re.sub(
            r"cls_token_variable_\d+$", "cls_token", torch_weight_name
        )

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, torch_weights_dict
            )
            continue

        if torch_weight_name not in torch_weights_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

        if torch_weight_name == "cls_token":
            keras_weight.assign(torch_weight)
            continue

        if torch_weight_name == "pos_embed":
            if torch_weight.shape[1] == keras_weight.shape[1] + 1:
                torch_weight = torch_weight[:, 1:, :]
            keras_weight.assign(torch_weight)
            continue

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
        input_shape=tuple(input_shape),
        output_specs={"num_classes": num_classes},
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

    del keras_model, torch_model, torch_weights_dict
    del trainable_torch_weights, non_trainable_torch_weights
    del trainable_keras_weights, non_trainable_keras_weights
    keras.backend.clear_session()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
