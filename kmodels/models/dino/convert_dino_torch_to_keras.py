"""Convert Facebook DINO checkpoints (backbone-only) to kmodels Keras weights.

Backbone weights come from **torch.hub** (``facebookresearch/dino:main``).

Usage:
    python convert_dino_torch_to_keras.py
"""

import gc
import re
from typing import Dict

import keras
import numpy as np
import torch
from tqdm import tqdm

from kmodels.models import dino as dino_kmodels
from kmodels.models.dino.config import DINO_TORCH_HUB
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.weight_split_torch_and_keras import split_model_weights
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

DINO_VARIANTS = {
    "DinoViTSmall16": {
        "kind": "vit",
        "ctor": dino_kmodels.DinoViTSmall16,
        "input_shape": (224, 224, 3),
        "save_name": "dino_vits16",
    },
    "DinoViTSmall8": {
        "kind": "vit",
        "ctor": dino_kmodels.DinoViTSmall8,
        "input_shape": (224, 224, 3),
        "save_name": "dino_vits8",
    },
    "DinoViTBase16": {
        "kind": "vit",
        "ctor": dino_kmodels.DinoViTBase16,
        "input_shape": (224, 224, 3),
        "save_name": "dino_vitb16",
    },
    "DinoViTBase8": {
        "kind": "vit",
        "ctor": dino_kmodels.DinoViTBase8,
        "input_shape": (224, 224, 3),
        "save_name": "dino_vitb8",
    },
    "DinoResNet50": {
        "kind": "resnet",
        "ctor": dino_kmodels.DinoResNet50,
        "input_shape": (224, 224, 3),
        "save_name": "dino_resnet50",
    },
}

# --------------------------------------------------------------------------- #
# Keras -> PyTorch name mappings
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# Backbone weight transfer
# --------------------------------------------------------------------------- #


def _transfer_backbone(
    keras_model: keras.Model, torch_sd: Dict[str, torch.Tensor], kind: str
) -> None:
    name_mapping = VIT_NAME_MAPPING if kind == "vit" else RESNET_NAME_MAPPING
    trainable, non_trainable = split_model_weights(keras_model)
    all_weights = trainable + non_trainable

    for keras_weight, keras_weight_name in tqdm(
        all_weights,
        total=len(all_weights),
        desc="  backbone",
    ):
        torch_name = keras_weight_name
        for old, new in name_mapping.items():
            torch_name = torch_name.replace(old, new)

        if kind == "vit":
            torch_name = re.sub(r"pos_embed_variable_\d+$", "pos_embed", torch_name)
            torch_name = re.sub(r"cls_token_variable_\d+$", "cls_token", torch_name)

            if "attention" in torch_name:
                transfer_attention_weights(keras_weight_name, keras_weight, torch_sd)
                continue

        if torch_name not in torch_sd:
            raise WeightMappingError(keras_weight_name, torch_name)

        tw = torch_sd[torch_name]
        if torch_name in ("cls_token", "pos_embed"):
            keras_weight.assign(tw.numpy())
            continue

        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_name, tw
        ):
            raise WeightShapeMismatchError(
                keras_weight_name, keras_weight.shape, torch_name, tw.shape
            )
        transfer_weights(keras_weight_name, keras_weight, tw)


# --------------------------------------------------------------------------- #
# Forward-pass verification
# --------------------------------------------------------------------------- #


def _verify_backbone(
    name: str,
    kind: str,
    keras_model: keras.Model,
    torch_sd: Dict[str, torch.Tensor],
    input_shape,
) -> None:
    hub_name = DINO_TORCH_HUB[name]
    if kind == "vit":
        ref = torch.hub.load("facebookresearch/dino:main", hub_name, pretrained=False)
        ref.load_state_dict(torch_sd, strict=False)
    else:
        from torchvision.models import resnet50

        ref = resnet50(weights=None)
        ref.fc = torch.nn.Identity()
        ref.load_state_dict(torch_sd, strict=False)
    ref.eval()

    h, w, c = input_shape
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, c, h, w)).astype(np.float32)

    with torch.no_grad():
        t_out = ref(torch.from_numpy(x)).cpu().numpy()

    k_in = np.transpose(x, (0, 2, 3, 1))
    k_raw = keras_model(k_in, training=False)
    k_out = (
        k_raw.detach().cpu().numpy() if hasattr(k_raw, "detach") else np.asarray(k_raw)
    )

    if kind == "vit":
        k_out = k_out[:, 0]  # [CLS] token

    diff = float(np.abs(k_out - t_out).max())
    assert diff < 1e-3, f"{name} backbone: max diff {diff:.2e}"
    print(f"  backbone OK  (max diff = {diff:.2e})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    for name, info in DINO_VARIANTS.items():
        print("\n" + "=" * 60)
        print(f"Converting: {name}")
        print("=" * 60)

        kind = info["kind"]
        ctor = info["ctor"]
        ishape = info["input_shape"]
        save = info["save_name"]

        # ---- 1. Load backbone from torch.hub ----
        hub_name = DINO_TORCH_HUB[name]
        print(f"  torch.hub: {hub_name}")
        torch_bb = torch.hub.load(
            "facebookresearch/dino:main", hub_name, pretrained=True
        )
        torch_bb.eval()
        bb_sd = dict(torch_bb.state_dict())

        # ---- 2. Build Keras model (backbone-only), transfer weights ----
        keras_model = ctor(
            include_top=False,
            include_normalization=False,
            input_shape=ishape,
            weights=None,
        )
        _transfer_backbone(keras_model, bb_sd, kind)

        # ---- 3. Verify ----
        _verify_backbone(name, kind, keras_model, bb_sd, ishape)

        # ---- 4. Save ----
        keras_model.save_weights(f"{save}.weights.h5")
        print(f"  saved -> {save}.weights.h5")

        del keras_model, torch_bb, bb_sd
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
