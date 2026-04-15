import gc
import re
from typing import Dict, List, Tuple, Type

import keras
import numpy as np
import torch
from tqdm import tqdm

from kmodels.models import dino
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

DINO_TORCH_HUB = {
    "DinoViTSmall16": "dino_vits16",
    "DinoViTSmall8": "dino_vits8",
    "DinoViTBase16": "dino_vitb16",
    "DinoViTBase8": "dino_vitb8",
    "DinoResNet50": "dino_resnet50",
}

VIT_NAME_MAPPING: Dict[str, str] = {
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
}

RESNET_NAME_MAPPING: Dict[str, str] = {
    "resnet_layer": "layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.batchnorm": "downsample.1",
    "batchnorm1": "bn1",
    "batchnorm2": "bn2",
    "batchnorm3": "bn3",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}

DINO_WEIGHTS_CONFIG: List[Tuple[Type[keras.Model], str, str, int, str]] = [
    (dino.DinoViTSmall16, "DinoViTSmall16", "vit", 224, "dino_vits16"),
    (dino.DinoViTSmall8, "DinoViTSmall8", "vit", 224, "dino_vits8"),
    (dino.DinoViTBase16, "DinoViTBase16", "vit", 224, "dino_vitb16"),
    (dino.DinoViTBase8, "DinoViTBase8", "vit", 224, "dino_vitb8"),
    (dino.DinoResNet50, "DinoResNet50", "resnet", 224, "dino_resnet50"),
]

for keras_model_cls, hub_key, kind, resolution, save_name in DINO_WEIGHTS_CONFIG:
    torch_hub_name = DINO_TORCH_HUB[hub_key]
    input_shape = [resolution, resolution, 3]

    print(f"\n{'=' * 60}")
    print(f"Converting: {hub_key}  (torch.hub: {torch_hub_name})")
    print(f"{'=' * 60}")

    torch_model: torch.nn.Module = torch.hub.load(
        "facebookresearch/dino:main", torch_hub_name, pretrained=True
    ).eval()

    trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
        torch_model
    )
    torch_weights_dict: Dict[str, torch.Tensor] = {
        **trainable_torch_weights,
        **non_trainable_torch_weights,
    }

    keras_model: keras.Model = keras_model_cls(
        include_top=False,
        include_normalization=False,
        input_shape=input_shape,
        pooling="avg" if kind == "resnet" else None,
        weights=None,
    )

    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model
    )

    name_mapping = VIT_NAME_MAPPING if kind == "vit" else RESNET_NAME_MAPPING

    for keras_weight, keras_weight_name in tqdm(
        trainable_keras_weights + non_trainable_keras_weights,
        total=len(trainable_keras_weights + non_trainable_keras_weights),
        desc="Transferring weights",
    ):
        torch_weight_name: str = keras_weight_name
        for old, new in name_mapping.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if kind == "vit":
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

        if torch_weight_name in ("cls_token", "pos_embed"):
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

    h, w, c = input_shape
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, c, h, w)).astype(np.float32)

    with torch.no_grad():
        t_out = torch_model(torch.from_numpy(x)).cpu().numpy()

    k_in = np.transpose(x, (0, 2, 3, 1))
    k_raw = keras_model(k_in, training=False)
    k_out = (
        k_raw.detach().cpu().numpy() if hasattr(k_raw, "detach") else np.asarray(k_raw)
    )

    if kind == "vit":
        k_out = k_out[:, 0]

    diff = float(np.abs(k_out - t_out).max())
    assert diff < 1e-3, f"{hub_key}: max diff {diff:.2e}"
    print(f"  Verification OK (max diff = {diff:.2e})")

    model_filename = f"{save_name}.weights.h5"
    keras_model.save_weights(model_filename)
    print(f"  Saved -> {model_filename}")

    del keras_model, torch_model, torch_weights_dict
    del trainable_torch_weights, non_trainable_torch_weights
    del trainable_keras_weights, non_trainable_keras_weights
    keras.backend.clear_session()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
