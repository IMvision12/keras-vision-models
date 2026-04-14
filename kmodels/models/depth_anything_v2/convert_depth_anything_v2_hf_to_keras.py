import gc
import re
from typing import Dict, List, Tuple, Type

import keras
import numpy as np
import torch
from tqdm import tqdm
from transformers import DepthAnythingForDepthEstimation

from kmodels.models.depth_anything_v2.depth_anything_v2_model import (
    DepthAnythingV2Base,
    DepthAnythingV2Large,
    DepthAnythingV2MetricIndoorBase,
    DepthAnythingV2MetricIndoorLarge,
    DepthAnythingV2MetricIndoorSmall,
    DepthAnythingV2MetricOutdoorBase,
    DepthAnythingV2MetricOutdoorLarge,
    DepthAnythingV2MetricOutdoorSmall,
    DepthAnythingV2Small,
)
from kmodels.utils.custom_exception import WeightMappingError
from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights


def _transfer_attention(path: str, keras_weight, hf_sd: Dict[str, np.ndarray]) -> None:
    parts = path.split("/")
    m = re.search(r"backbone_block_(\d+)_attn", parts[0])
    if not m:
        raise WeightMappingError(path, "invalid attention path")
    layer_idx = m.group(1)
    suffix = parts[-1]
    torch_suffix = "weight" if suffix == "kernel" else "bias"
    hf_prefix = f"backbone.encoder.layer.{layer_idx}.attention"

    if "qkv" in parts[1]:
        q = hf_sd[f"{hf_prefix}.attention.query.{torch_suffix}"]
        k = hf_sd[f"{hf_prefix}.attention.key.{torch_suffix}"]
        v = hf_sd[f"{hf_prefix}.attention.value.{torch_suffix}"]
        torch_weight = np.concatenate([q, k, v], axis=0)
    elif "proj" in parts[1]:
        torch_weight = hf_sd[f"{hf_prefix}.output.dense.{torch_suffix}"]
    else:
        raise WeightMappingError(path, f"unknown attention sub-layer {parts[1]}")

    transfer_weights(suffix, keras_weight, torch_weight)


weight_name_mapping: Dict[str, str] = {
    "/": ".",
    "_": ".",
    ".ln1.": ".norm1.",
    ".ln2.": ".norm2.",
    "backbone.block.": "backbone.encoder.layer.",
    "backbone.patch.embed.": "backbone.embeddings.patch_embeddings.projection.",
    "backbone.cls.token.cls.token": "backbone.embeddings.cls_token",
    "backbone.pos.embed.pos.embed": "backbone.embeddings.position_embeddings",
    "neck.reassemble.": "neck.reassemble_stage.layers.",
    "neck.fusion.": "neck.fusion_stage.layers.",
    "neck.conv.": "neck.convs.",
    ".res1.conv1.": ".residual_layer1.convolution1.",
    ".res1.conv2.": ".residual_layer1.convolution2.",
    ".res2.conv1.": ".residual_layer2.convolution1.",
    ".res2.conv2.": ".residual_layer2.convolution2.",
    ".kernel": ".weight",
    ".gamma": ".weight",
    ".beta": ".bias",
}

DEPTH_ANYTHING_V2_CONVERSION_CONFIG: List[Tuple[Type[keras.Model], str, str]] = [
    (
        DepthAnythingV2Small,
        "depth-anything/Depth-Anything-V2-Small-hf",
        "depth_anything_v2_small",
    ),
    (
        DepthAnythingV2Base,
        "depth-anything/Depth-Anything-V2-Base-hf",
        "depth_anything_v2_base",
    ),
    (
        DepthAnythingV2Large,
        "depth-anything/Depth-Anything-V2-Large-hf",
        "depth_anything_v2_large",
    ),
    (
        DepthAnythingV2MetricIndoorSmall,
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "depth_anything_v2_metric_indoor_small",
    ),
    (
        DepthAnythingV2MetricIndoorBase,
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "depth_anything_v2_metric_indoor_base",
    ),
    (
        DepthAnythingV2MetricIndoorLarge,
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "depth_anything_v2_metric_indoor_large",
    ),
    (
        DepthAnythingV2MetricOutdoorSmall,
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "depth_anything_v2_metric_outdoor_small",
    ),
    (
        DepthAnythingV2MetricOutdoorBase,
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        "depth_anything_v2_metric_outdoor_base",
    ),
    (
        DepthAnythingV2MetricOutdoorLarge,
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "depth_anything_v2_metric_outdoor_large",
    ),
]

for keras_ctor, hf_id, save_name in DEPTH_ANYTHING_V2_CONVERSION_CONFIG:
    variant = keras_ctor.__name__
    print(f"\n{'=' * 60}")
    print(f"Converting: {variant}  <-  {hf_id}")
    print(f"{'=' * 60}")

    hf_model = DepthAnythingForDepthEstimation.from_pretrained(hf_id).eval()
    hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

    keras_model: keras.Model = keras_ctor(input_shape=(518, 518, 3), weights=None)

    all_weights = [w for layer in keras_model.layers for w in layer.weights]
    for w in tqdm(all_weights, desc="Transferring weights"):
        path = w.path

        if "_attn/" in path:
            _transfer_attention(path, w, hf_sd)
            continue

        m = re.match(r"backbone_block_(\d+)_ls(\d+)/variable(?:_\d+)?$", path)
        if m:
            layer_idx, ls_idx = m.group(1), m.group(2)
            torch_key = (
                f"backbone.encoder.layer.{layer_idx}.layer_scale{ls_idx}.lambda1"
            )
            w.assign(hf_sd[torch_key])
            continue

        torch_key = path
        for old, new in weight_name_mapping.items():
            torch_key = torch_key.replace(old, new)

        if torch_key not in hf_sd:
            raise WeightMappingError(path, torch_key)

        torch_weight = hf_sd[torch_key]
        keras_name = "conv_kernel" if len(w.shape) == 4 else path
        transfer_weights(keras_name, w, torch_weight)

    np.random.seed(42)
    test_image = np.random.rand(1, 518, 518, 3).astype(np.float32)

    keras_depth = keras_model.predict(test_image, verbose=0).squeeze(-1)

    with torch.no_grad():
        hf_input = torch.from_numpy(test_image.transpose(0, 3, 1, 2))
        hf_depth = hf_model(pixel_values=hf_input).predicted_depth.cpu().numpy()

    max_diff = float(np.max(np.abs(keras_depth - hf_depth)))
    mean_diff = float(np.mean(np.abs(keras_depth - hf_depth)))
    print(f"  Max depth diff:  {max_diff:.6f}")
    print(f"  Mean depth diff: {mean_diff:.6f}")
    if max_diff > 25.0:
        raise ValueError(f"{variant}: depth diff {max_diff:.2e} exceeds tolerance")
    print("  Verification OK")

    model_filename = f"{save_name}.weights.h5"
    keras_model.save_weights(model_filename)
    print(f"  Saved -> {model_filename}")

    del keras_model, hf_model, hf_sd
    keras.backend.clear_session()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
