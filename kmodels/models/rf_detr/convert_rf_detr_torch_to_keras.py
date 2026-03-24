import argparse
import os

os.environ["KERAS_BACKEND"] = "torch"

from typing import Dict, List, Tuple

import keras
import numpy as np
import torch
from tqdm import tqdm

from kmodels.models.rf_detr.config import RF_DETR_MODEL_CONFIG
from kmodels.models.rf_detr.rf_detr_model import RFDETR
from kmodels.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_nested_layer_weights,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "backbone_encoder_layer_": "backbone.0.encoder.encoder.encoder.layer.",
    "backbone_encoder_layernorm_": "backbone.0.encoder.encoder.layernorm.",
    "_attention_query_": ".attention.attention.query.",
    "_attention_key_": ".attention.attention.key.",
    "_attention_value_": ".attention.attention.value.",
    "_attention_out_proj_": ".attention.output.dense.",
    "_norm1_": ".norm1.",
    "_norm2_": ".norm2.",
    "_layer_scale1_": ".layer_scale1.",
    "_layer_scale2_": ".layer_scale2.",
    "_mlp_fc1_": ".mlp.fc1.",
    "_mlp_fc2_": ".mlp.fc2.",
    "_mlp_weights_in_": ".mlp.fc1.",
    "_mlp_weights_out_": ".mlp.fc2.",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def load_torch_state_dict(variant):
    """Load PyTorch weights using the rfdetr package and convert to numpy."""
    import rfdetr as rfdetr_pkg

    cls = getattr(rfdetr_pkg, variant.replace("RFDETR", "RFDETR"))
    torch_model = cls()
    sd = torch_model.model.model.state_dict()
    return {k: v.cpu().numpy() for k, v in sd.items()}


def build_keras_model(variant):
    """Build a Keras RF-DETR model (no weights)."""
    config = RF_DETR_MODEL_CONFIG[variant]
    res = config["resolution"]
    model = RFDETR(
        hidden_dim=256,
        backbone_hidden_size=384,
        backbone_num_heads=6,
        backbone_num_layers=12,
        backbone_mlp_ratio=4,
        backbone_use_swiglu=False,
        num_register_tokens=0,
        out_feature_indexes=config.get("out_feature_indexes", [3, 6, 9, 12]),
        patch_size=config.get("patch_size", 16),
        num_windows=config.get("num_windows", 2),
        positional_encoding_size=config["positional_encoding_size"],
        resolution=res,
        dec_layers=config["dec_layers"],
        sa_nheads=8,
        ca_nheads=16,
        dec_n_points=2,
        num_queries=300,
        num_classes=91,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        group_detr=13,
        dim_feedforward=2048,
        weights=None,
        input_shape=(res, res, 3),
        name=variant,
    )
    dummy = keras.random.uniform((1, res, res, 3), dtype="float32")
    _ = model(dummy)
    return model


def transfer_all_weights(variant, pytorch_state_dict, keras_model):
    """Map PyTorch state_dict keys to Keras weights and assign values."""
    config = RF_DETR_MODEL_CONFIG[variant]
    out_feature_indexes = config.get("out_feature_indexes", [3, 6, 9, 12])
    num_layers = max(out_feature_indexes)
    dec_layers = config["dec_layers"]

    # ---- Backbone Encoder Layers (name mapping + library utilities) ----
    backbone_encoder_weights: List[Tuple[keras.Variable, str]] = []
    for layer in keras_model.layers:
        if layer.name.startswith("backbone_encoder_layer_") or (
            layer.name == "backbone_encoder_layernorm"
        ):
            for weight in layer.trainable_weights:
                backbone_encoder_weights.append((weight, f"{layer.name}_{weight.name}"))
            for weight in layer.non_trainable_weights:
                backbone_encoder_weights.append((weight, f"{layer.name}_{weight.name}"))

    for keras_weight, keras_weight_name in tqdm(
        backbone_encoder_weights,
        total=len(backbone_encoder_weights),
        desc="Transferring backbone encoder weights",
    ):
        torch_weight_name: str = keras_weight_name
        for keras_name_part, torch_name_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )
            return
        target.assign(tensor)
        assigned.add(keras_path)

    assign(
        "backbone_embeddings/cls_token",
        torch_sd["backbone.0.encoder.encoder.embeddings.cls_token"],
    )
    assign(
        "backbone_embeddings/position_embeddings",
        torch_sd["backbone.0.encoder.encoder.embeddings.position_embeddings"],
    )
    assign(
        "backbone_embeddings/patch_embeddings/projection/kernel",
        torch_sd[
            "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
        ].permute(2, 3, 1, 0),
    )
    assign(
        "backbone_embeddings/patch_embeddings/projection/bias",
        torch_sd[
            "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.bias"
        ],
    )

    for i in range(num_layers):
        pt_prefix = f"backbone.0.encoder.encoder.encoder.layer.{i}"
        k_prefix = f"backbone_encoder_layer_{i}"

        assign(f"{k_prefix}_norm1/gamma", torch_sd[f"{pt_prefix}.norm1.weight"])
        assign(f"{k_prefix}_norm1/beta", torch_sd[f"{pt_prefix}.norm1.bias"])

        assign(
            f"{k_prefix}_attention_query/kernel",
            torch_sd[f"{pt_prefix}.attention.attention.query.weight"],
            transpose=True,
        )
        assign(
            f"{k_prefix}_attention_query/bias",
            torch_sd[f"{pt_prefix}.attention.attention.query.bias"],
        )
        assign(
            f"{k_prefix}_attention_key/kernel",
            torch_sd[f"{pt_prefix}.attention.attention.key.weight"],
            transpose=True,
        )
        assign(
            f"{k_prefix}_attention_key/bias",
            torch_sd[f"{pt_prefix}.attention.attention.key.bias"],
        )
        assign(
            f"{k_prefix}_attention_value/kernel",
            torch_sd[f"{pt_prefix}.attention.attention.value.weight"],
            transpose=True,
        )
        assign(
            f"{k_prefix}_attention_value/bias",
            torch_sd[f"{pt_prefix}.attention.attention.value.bias"],
        )
        assign(
            f"{k_prefix}_attention_out_proj/kernel",
            torch_sd[f"{pt_prefix}.attention.output.dense.weight"],
            transpose=True,
        )
        assign(
            f"{k_prefix}_attention_out_proj/bias",
            torch_sd[f"{pt_prefix}.attention.output.dense.bias"],
        )

        assign(
            f"{k_prefix}_layer_scale1/lambda1",
            torch_sd[f"{pt_prefix}.layer_scale1.lambda1"],
        )

        assign(f"{k_prefix}_norm2/gamma", torch_sd[f"{pt_prefix}.norm2.weight"])
        assign(f"{k_prefix}_norm2/beta", torch_sd[f"{pt_prefix}.norm2.bias"])

        if config.get("backbone_use_swiglu", False):
            assign(
                f"{k_prefix}_mlp_weights_in/kernel",
                torch_sd[f"{pt_prefix}.mlp.fc1.weight"],
                transpose=True,
            )
            assign(
                f"{k_prefix}_mlp_weights_in/bias",
                torch_sd[f"{pt_prefix}.mlp.fc1.bias"],
            )
            assign(
                f"{k_prefix}_mlp_weights_out/kernel",
                torch_sd[f"{pt_prefix}.mlp.fc2.weight"],
                transpose=True,
            )
            assign(
                f"{k_prefix}_mlp_weights_out/bias",
                torch_sd[f"{pt_prefix}.mlp.fc2.bias"],
            )
        else:
            assign(
                f"{k_prefix}_mlp_fc1/kernel",
                torch_sd[f"{pt_prefix}.mlp.fc1.weight"],
                transpose=True,
            )
            assign(
                f"{k_prefix}_mlp_fc1/bias",
                torch_sd[f"{pt_prefix}.mlp.fc1.bias"],
            )
            assign(
                f"{k_prefix}_mlp_fc2/kernel",
                torch_sd[f"{pt_prefix}.mlp.fc2.weight"],
                transpose=True,
            )
            assign(
                f"{k_prefix}_mlp_fc2/bias",
                torch_sd[f"{pt_prefix}.mlp.fc2.bias"],
            )

        assign(
            f"{k_prefix}_layer_scale2/lambda1",
            torch_sd[f"{pt_prefix}.layer_scale2.lambda1"],
        )

    assign(
        "backbone_encoder_layernorm/gamma",
        torch_sd["backbone.0.encoder.encoder.layernorm.weight"],
    )
    assign(
        "backbone_encoder_layernorm/beta",
        torch_sd["backbone.0.encoder.encoder.layernorm.bias"],
    )

    emb_prefix = "backbone.0.encoder.encoder.embeddings"

    assign(
        "projector_c2f_cv1_conv/kernel",
        torch_sd[f"{pt_proj}.0.cv1.conv.weight"].permute(2, 3, 1, 0),
    )
    assign(
        "projector_c2f_cv1_ln/gamma",
        torch_sd[f"{pt_proj}.0.cv1.bn.weight"],
    )
    assign(
        "projector_c2f_cv1_ln/beta",
        torch_sd[f"{pt_proj}.0.cv1.bn.bias"],
    )

    assign(
        "projector_c2f_cv2_conv/kernel",
        torch_sd[f"{pt_proj}.0.cv2.conv.weight"].permute(2, 3, 1, 0),
    )
    assign(
        "projector_c2f_cv2_ln/gamma",
        torch_sd[f"{pt_proj}.0.cv2.bn.weight"],
    )
    assign(
        "projector_c2f_cv2_ln/beta",
        torch_sd[f"{pt_proj}.0.cv2.bn.bias"],
    )

    projector_conv_ln_pairs = [
        ("projector_c2f_cv1_conv", f"{pt_proj}.0.cv1.conv"),
        ("projector_c2f_cv1_ln", f"{pt_proj}.0.cv1.bn"),
        ("projector_c2f_cv2_conv", f"{pt_proj}.0.cv2.conv"),
        ("projector_c2f_cv2_ln", f"{pt_proj}.0.cv2.bn"),
    ]
    for b_idx in range(3):
        pt_bn = f"{pt_proj}.0.m.{b_idx}"
        k_bn = f"projector_c2f_bottleneck_{b_idx}"
        for cv in ["cv1", "cv2"]:
            assign(
                f"{k_bn}_{cv}_conv/kernel",
                torch_sd[f"{pt_bn}.{cv}.conv.weight"].permute(2, 3, 1, 0),
            )
            assign(
                f"{k_bn}_{cv}_ln/gamma",
                torch_sd[f"{pt_bn}.{cv}.bn.weight"],
            )
            assign(
                f"{k_bn}_{cv}_ln/beta",
                torch_sd[f"{pt_bn}.{cv}.bn.bias"],
            )

    for keras_name, pt_name in tqdm(
        projector_conv_ln_pairs,
        desc="Transferring projector weights",
    ):
        layer = keras_model.get_layer(keras_name)
        if keras_name.endswith("_conv"):
            conv_w = pytorch_state_dict[f"{pt_name}.weight"]
            layer.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))
        else:
            layer.weights[0].assign(pytorch_state_dict[f"{pt_name}.weight"])
            layer.weights[1].assign(pytorch_state_dict[f"{pt_name}.bias"])

    proj_ln = keras_model.get_layer("projector_ln")
    proj_ln.weights[0].assign(pytorch_state_dict[f"{pt_proj}.1.weight"])
    proj_ln.weights[1].assign(pytorch_state_dict[f"{pt_proj}.1.bias"])

    enc_output = keras_model.get_layer("enc_output_0")
    enc_output.weights[0].assign(
        pytorch_state_dict["transformer.enc_output.0.weight"].T
    )
    enc_output.weights[1].assign(pytorch_state_dict["transformer.enc_output.0.bias"])

    enc_output_norm = keras_model.get_layer("enc_output_norm_0")
    enc_output_norm.weights[0].assign(
        pytorch_state_dict["transformer.enc_output_norm.0.weight"]
    )
    enc_output_norm.weights[1].assign(
        pytorch_state_dict["transformer.enc_output_norm.0.bias"]
    )

    enc_cls = keras_model.get_layer("enc_out_class_embed_0")
    enc_cls.weights[0].assign(
        pytorch_state_dict["transformer.enc_out_class_embed.0.weight"].T
    )
    enc_cls.weights[1].assign(
        pytorch_state_dict["transformer.enc_out_class_embed.0.bias"]
    )

    for i in range(3):
        bbox_layer = keras_model.get_layer(f"enc_bbox_{i}")
        bbox_layer.weights[0].assign(
            pytorch_state_dict[f"transformer.enc_out_bbox_embed.0.layers.{i}.weight"].T
        )
        bbox_layer.weights[1].assign(
            pytorch_state_dict[f"transformer.enc_out_bbox_embed.0.layers.{i}.bias"]
        )

    for i in range(2):
        rph = keras_model.get_layer(f"ref_point_head_{i}")
        rph.weights[0].assign(
            pytorch_state_dict[
                f"transformer.decoder.ref_point_head.layers.{i}.weight"
            ].T
        )
        rph.weights[1].assign(
            pytorch_state_dict[f"transformer.decoder.ref_point_head.layers.{i}.bias"]
        )

    decoder_name_mapping = {
        "self_attn_out_proj": "self_attn.out_proj",
        "kernel": "weight",
        "gamma": "weight",
        "beta": "bias",
    }

    for i in tqdm(range(dec_layers), desc="Transferring decoder weights"):
        pt_dl = f"transformer.decoder.layers.{i}"
        k_dl = f"decoder_layer_{i}"

        dec_layer = keras_model.get_layer(k_dl)

        # Self-attention uses fused in_proj in PyTorch → split into Q/K/V
        in_proj_w = pytorch_state_dict[f"{pt_dl}.self_attn.in_proj_weight"]
        in_proj_b = pytorch_state_dict[f"{pt_dl}.self_attn.in_proj_bias"]
        q_w, k_w, v_w = np.split(in_proj_w, 3, axis=0)
        q_b, k_b, v_b = np.split(in_proj_b, 3, axis=0)

        weight_dict = {w.path: w for w in dec_layer.weights}
        weight_dict[f"{k_dl}/self_attn_q_proj/kernel"].assign(q_w.T)
        weight_dict[f"{k_dl}/self_attn_q_proj/bias"].assign(q_b)
        weight_dict[f"{k_dl}/self_attn_k_proj/kernel"].assign(k_w.T)
        weight_dict[f"{k_dl}/self_attn_k_proj/bias"].assign(k_b)
        weight_dict[f"{k_dl}/self_attn_v_proj/kernel"].assign(v_w.T)
        weight_dict[f"{k_dl}/self_attn_v_proj/bias"].assign(v_b)

        # Transfer remaining weights (cross_attn, norms, linears) via utility
        transfer_nested_layer_weights(
            dec_layer,
            pytorch_state_dict,
            pt_dl,
            name_mapping=decoder_name_mapping,
            skip_paths=["self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj"],
        )

    dec_norm = keras_model.get_layer("decoder_norm")
    dec_norm.weights[0].assign(pytorch_state_dict["transformer.decoder.norm.weight"])
    dec_norm.weights[1].assign(pytorch_state_dict["transformer.decoder.norm.bias"])

    num_queries = 300

    # --- Learned query embeddings (slice to first num_queries for inference) ---
    num_queries = 300
    assign(
        "refpoint_embed_layer/weight",
        torch_sd["refpoint_embed.weight"][:num_queries],
    )
    assign(
        "query_feat_embed/weight",
        torch_sd["query_feat.weight"][:num_queries],
    )

    query_feat_layer = keras_model.get_layer("query_feat_embed")
    query_feat_layer.weights[0].assign(
        pytorch_state_dict["query_feat.weight"][:num_queries]
    )

    cls_embed = keras_model.get_layer("class_embed")
    cls_embed.weights[0].assign(pytorch_state_dict["class_embed.weight"].T)
    cls_embed.weights[1].assign(pytorch_state_dict["class_embed.bias"])

    for i in range(3):
        bbox_layer = keras_model.get_layer(f"bbox_embed_{i}")
        bbox_layer.weights[0].assign(
            pytorch_state_dict[f"bbox_embed.layers.{i}.weight"].T
        )
        bbox_layer.weights[1].assign(pytorch_state_dict[f"bbox_embed.layers.{i}.bias"])


def verify_equivalence(variant, keras_model, pytorch_state_dict):
    import torchvision.transforms as T

    config = RF_DETR_MODEL_CONFIG[variant]
    res = config["resolution"]

    np.random.seed(42)
    test_input = np.random.rand(1, res, res, 3).astype(np.float32)

    pt_input = torch.tensor(test_input).permute(0, 3, 1, 2)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    pt_input_norm = normalize(pt_input)

    import rfdetr as rfdetr_pkg

    pt_wrapper = getattr(rfdetr_pkg, variant.replace("RFDETR", "RFDETR"))()
    pt_model = pt_wrapper.model.model
    pt_model.eval()

    with torch.no_grad():
        pt_out = pt_model(pt_input_norm)
    pt_logits = pt_out["pred_logits"].cpu().numpy()
    pt_boxes = pt_out["pred_boxes"].cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    keras_input_norm = ((test_input - mean) / std).astype(np.float32)

    keras_out = keras_model(keras_input_norm, training=False)
    keras_logits = keras.ops.convert_to_numpy(keras_out["pred_logits"])
    keras_boxes = keras.ops.convert_to_numpy(keras_out["pred_boxes"])

    logits_diff = np.max(np.abs(pt_logits - keras_logits))
    boxes_diff = np.max(np.abs(pt_boxes - keras_boxes))

    pt_flat = pt_logits.flatten()
    k_flat = keras_logits.flatten()
    logits_cos = float(
        np.dot(pt_flat, k_flat)
        / (np.linalg.norm(pt_flat) * np.linalg.norm(k_flat) + 1e-8)
    )

    print(f"Max logits diff:  {logits_diff:.6f}")
    print(f"Max boxes diff:   {boxes_diff:.6f}")
    print(f"Logits cosine sim: {logits_cos:.6f}")

    if logits_cos < 0.95:
        raise ValueError(
            f"Equivalence test failed: logits cosine similarity {logits_cos:.4f} < 0.95"
        )
    print("Equivalence test passed (logits cosine sim > 0.95)")


def main():
    parser = argparse.ArgumentParser(description="Convert RF-DETR weights to Keras")
    parser.add_argument(
        "--variant",
        type=str,
        default="RFDETRNano",
        choices=list(RF_DETR_MODEL_CONFIG.keys()),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    variant = args.variant
    output = args.output
    if output is None:
        name_map = {
            "RFDETRNano": "rf_detr_nano_coco",
            "RFDETRSmall": "rf_detr_small_coco",
            "RFDETRMedium": "rf_detr_medium_coco",
            "RFDETRBase": "rf_detr_base_coco",
            "RFDETRLarge": "rf_detr_large_coco",
        }
        output = f"{name_map[variant]}.weights.h5"

    print(f"Building Keras {variant} model...")
    keras_model = build_keras_model(variant)
    print(f"  Parameters: {keras_model.count_params():,}")

    print(f"Loading PyTorch weights for {variant}...")
    pytorch_state_dict = load_torch_state_dict(variant)
    print(f"  PyTorch keys: {len(pytorch_state_dict)}")

    print("Transferring weights...")
    transfer_all_weights(variant, pytorch_state_dict, keras_model)

    if not args.skip_verify:
        print("\nVerifying model equivalence...")
        verify_equivalence(variant, keras_model, pytorch_state_dict)

    print(f"\nSaving Keras weights to {output}...")
    keras_model.save_weights(output)
    print("Done!")


if __name__ == "__main__":
    main()
