"""Weight conversion script from timm NextViT to Keras NextViT.

Converts weights from timm's nextvit models to the Keras implementation.
Uses the utility functions from kmodels for weight transfer and equivalence testing.
"""

import re

import numpy as np
import timm
import torch
from tqdm import tqdm

from kmodels.models.nextvit.nextvit_model import NextViTBase, NextViTLarge, NextViTSmall
from kmodels.utils.model_equivalence_tester import verify_cls_model_equivalence


def _keras_path_to_torch_key(path):
    """Convert a Keras weight path to the corresponding PyTorch state dict key.

    Keras weight paths look like:
        next_vi_t/stem_0_conv/kernel:0
        next_vi_t/stages_0_blocks_0_mhca/stages_0_blocks_0_mhca_group_conv3x3/kernel:0
        next_vi_t/stages_1_blocks_3_e_mhsa/stages_1_blocks_3_e_mhsa_q/kernel:0
    """
    # Remove :0 suffix if present
    if ":" in path:
        path = path.rsplit(":", 1)[0]

    parts = path.split("/")
    weight_name = parts[-1]

    if len(parts) >= 2:
        layer_name = parts[-2]
    else:
        layer_name = parts[0]
        weight_name = ""

    # Map weight suffixes
    weight_suffix_map = {
        "kernel": "weight",
        "bias": "bias",
        "gamma": "weight",
        "beta": "bias",
        "moving_mean": "running_mean",
        "moving_variance": "running_var",
    }
    torch_suffix = weight_suffix_map.get(weight_name, weight_name)

    torch_prefix = _map_layer_to_torch(layer_name)
    if torch_prefix is None:
        return None

    return f"{torch_prefix}.{torch_suffix}"


def _map_layer_to_torch(layer_name):
    """Map a Keras layer name to its PyTorch module path."""

    # Stem: stem_{i}_conv, stem_{i}_norm
    m = re.match(r"stem_(\d+)_conv$", layer_name)
    if m:
        return f"stem.{m.group(1)}.conv"

    m = re.match(r"stem_(\d+)_norm$", layer_name)
    if m:
        return f"stem.{m.group(1)}.norm"

    # Final norm
    if layer_name == "norm":
        return "norm"

    # Head
    if layer_name == "head_fc":
        return "head.fc"

    # Stage block patterns
    m = re.match(r"stages_(\d+)_blocks_(\d+)_(.*)", layer_name)
    if not m:
        return None

    s, b, rest = m.group(1), m.group(2), m.group(3)
    prefix = f"stages.{s}.blocks.{b}"

    # Patch embed
    if rest == "patch_embed_conv":
        return f"{prefix}.patch_embed.conv"
    if rest == "patch_embed_norm":
        return f"{prefix}.patch_embed.norm"

    # ConvBlock norm (BN after MHCA)
    if rest == "norm":
        return f"{prefix}.norm"

    # TransformerBlock norm1 (BN before E-MHSA)
    if rest == "norm1":
        return f"{prefix}.norm1"

    # TransformerBlock norm2 (BN before MLP)
    if rest == "norm2":
        return f"{prefix}.norm2"

    # MHCA sublayers
    if rest == "mhca_group_conv3x3":
        return f"{prefix}.mhca.group_conv3x3"
    if rest == "mhca_norm":
        return f"{prefix}.mhca.norm"
    if rest == "mhca_projection":
        return f"{prefix}.mhca.projection"

    # E-MHSA sublayers
    if rest == "e_mhsa_q":
        return f"{prefix}.e_mhsa.q"
    if rest == "e_mhsa_k":
        return f"{prefix}.e_mhsa.k"
    if rest == "e_mhsa_v":
        return f"{prefix}.e_mhsa.v"
    if rest == "e_mhsa_proj":
        return f"{prefix}.e_mhsa.proj"
    if rest == "e_mhsa_norm":
        return f"{prefix}.e_mhsa.norm"

    # Projection (in TransformerBlock): conv and norm
    if rest == "projection_conv":
        return f"{prefix}.projection.conv"
    if rest == "projection_norm":
        return f"{prefix}.projection.norm"

    # MLP sublayers
    if rest == "mlp_fc1":
        return f"{prefix}.mlp.fc1"
    if rest == "mlp_fc2":
        return f"{prefix}.mlp.fc2"

    return None


def _transfer_weight(keras_path, keras_var, torch_weight):
    """Transfer a single weight tensor with appropriate transpositions."""
    k_shape = tuple(keras_var.shape)
    t_shape = torch_weight.shape

    if len(k_shape) == 4 and len(t_shape) == 4:
        if "group_conv3x3" in keras_path:
            # Grouped conv: torch (out, in_per_group, kH, kW) -> keras (kH, kW, in_per_group, out)
            # For grouped conv, torch shape is (C_out, C_in/groups, kH, kW)
            # keras shape is (kH, kW, C_in/groups, C_out) -- but wait, Keras groups
            # are handled differently. Let me just do standard conv transpose.
            torch_weight = np.transpose(torch_weight, [2, 3, 1, 0])
        else:
            # Standard conv: torch (out, in, kH, kW) -> keras (kH, kW, in, out)
            torch_weight = np.transpose(torch_weight, [2, 3, 1, 0])
    elif len(k_shape) == 2 and len(t_shape) == 2:
        # Dense: torch (out, in) -> keras (in, out)
        torch_weight = np.transpose(torch_weight)
    # 1D (bias, BN weights) stay as-is

    keras_var.assign(torch_weight)


def convert_nextvit_weights(variant="small"):
    """Run the full weight conversion and verification pipeline.

    Args:
        variant: One of 'small', 'base', 'large'.
    """
    variant_map = {
        "small": {
            "keras_cls": NextViTSmall,
            "torch_name": "nextvit_small.bd_in1k",
        },
        "base": {
            "keras_cls": NextViTBase,
            "torch_name": "nextvit_base.bd_in1k",
        },
        "large": {
            "keras_cls": NextViTLarge,
            "torch_name": "nextvit_large.bd_in1k",
        },
    }

    cfg = variant_map[variant]
    model_config = {
        "torch_model_name": cfg["torch_name"],
        "input_shape": [224, 224, 3],
        "num_classes": 1000,
        "include_top": True,
        "include_normalization": False,
        "classifier_activation": "linear",
    }

    print("Creating Keras model...")
    keras_model = cfg["keras_cls"](
        include_top=model_config["include_top"],
        input_shape=model_config["input_shape"],
        classifier_activation=model_config["classifier_activation"],
        num_classes=model_config["num_classes"],
        include_normalization=model_config["include_normalization"],
        weights=None,
    )
    print(f"Keras model params: {keras_model.count_params():,}")

    print("Creating timm model...")
    torch_model = timm.create_model(
        model_config["torch_model_name"], pretrained=True
    ).eval()
    torch_params = sum(p.numel() for p in torch_model.parameters())
    print(f"Torch model params: {torch_params:,}")

    torch_sd = torch_model.state_dict()

    # Collect all Keras weight variables
    all_keras_weights = []
    for layer in keras_model.layers:
        for w in layer.weights:
            all_keras_weights.append(w)

    transferred = 0
    transferred_torch_keys = set()

    for w in tqdm(all_keras_weights, desc="Transferring weights"):
        path = w.path
        torch_key = _keras_path_to_torch_key(path)

        if torch_key is None:
            print(f"WARNING: No mapping for: {path}")
            continue

        if torch_key not in torch_sd:
            print(f"WARNING: Torch key not found: {torch_key} (from {path})")
            continue

        torch_weight = torch_sd[torch_key].detach().cpu().numpy()
        _transfer_weight(path, w, torch_weight)
        transferred += 1
        transferred_torch_keys.add(torch_key)

    # Check for untransferred torch weights
    all_torch_keys = set(torch_sd.keys())
    not_transferred = all_torch_keys - transferred_torch_keys
    not_transferred = {k for k in not_transferred if "num_batches_tracked" not in k}
    if not_transferred:
        print(f"\nWARNING: {len(not_transferred)} torch keys not transferred:")
        for k in sorted(not_transferred):
            print(f"  {k} {torch_sd[k].shape}")

    print(f"\nTransferred {transferred} weight tensors.")

    # Verify equivalence
    print("\nVerifying model equivalence...")
    results = verify_cls_model_equivalence(
        model_a=torch_model,
        model_b=keras_model,
        input_shape=tuple(model_config["input_shape"]),
        output_specs={"num_classes": model_config["num_classes"]},
        run_performance=False,
        atol=1e-3,
        rtol=1e-3,
    )

    if not results["standard_input"]:
        print("WARNING: Model equivalence test FAILED")
        if "standard_input_diff" in results:
            print(f"  Max diff: {results['standard_input_diff']['max_difference']}")
            print(f"  Mean diff: {results['standard_input_diff']['mean_difference']}")
    else:
        print("Model equivalence test PASSED")

    model_filename = f"{model_config['torch_model_name'].replace('.', '_')}.weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")

    return keras_model, torch_model, results


if __name__ == "__main__":
    for v in ["small", "base", "large"]:
        print(f"\n{'=' * 60}")
        print(f"Converting NextViT {v}...")
        print(f"{'=' * 60}")
        _, tm, _ = convert_nextvit_weights(v)
        del tm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
