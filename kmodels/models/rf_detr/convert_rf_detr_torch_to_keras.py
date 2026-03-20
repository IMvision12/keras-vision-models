"""Convert RF-DETR PyTorch weights to Keras 3 format.

Usage:
    python -m kmodels.models.rf_detr.convert_rf_detr_torch_to_keras \
        --variant RFDETRBase \
        --output rf_detr_base_coco.weights.h5
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch

from kmodels.models.rf_detr.config import RF_DETR_MODEL_CONFIG
from kmodels.models.rf_detr.rf_detr_model import RFDETR

VARIANT_TO_RFDETR_CLASS = {
    "RFDETRNano": "RFDETRNano",
    "RFDETRSmall": "RFDETRSmall",
    "RFDETRMedium": "RFDETRMedium",
    "RFDETRBase": "RFDETRBase",
    "RFDETRLarge": "RFDETRLarge",
}

RFDETR_PKG_CLASSES = {
    "RFDETRNano": "RFDETRNano",
    "RFDETRSmall": "RFDETRSmall",
    "RFDETRMedium": "RFDETRMedium",
    "RFDETRBase": "RFDETRBase",
    "RFDETRLarge": "RFDETRLarge",
}


def load_torch_state_dict(variant):
    """Load PyTorch weights using the rfdetr package."""
    import rfdetr as rfdetr_pkg

    cls = getattr(rfdetr_pkg, variant.replace("RFDETR", "RFDETR"))
    torch_model = cls()
    sd = torch_model.model.model.state_dict()
    return sd


def build_keras_model(variant):
    """Build a Keras RF-DETR model (no weights)."""
    config = RF_DETR_MODEL_CONFIG[variant]
    res = config["resolution"]
    model = RFDETR(
        hidden_dim=config["hidden_dim"],
        backbone_hidden_size=config["backbone_hidden_size"],
        backbone_num_heads=config["backbone_num_heads"],
        backbone_num_layers=config["backbone_num_layers"],
        backbone_mlp_ratio=config["backbone_mlp_ratio"],
        backbone_use_swiglu=config["backbone_use_swiglu"],
        num_register_tokens=config["num_register_tokens"],
        out_feature_indexes=config["out_feature_indexes"],
        patch_size=config["patch_size"],
        num_windows=config["num_windows"],
        positional_encoding_size=config["positional_encoding_size"],
        resolution=res,
        dec_layers=config["dec_layers"],
        sa_nheads=config["sa_nheads"],
        ca_nheads=config["ca_nheads"],
        dec_n_points=config["dec_n_points"],
        num_queries=config["num_queries"],
        num_classes=config["num_classes"],
        two_stage=config["two_stage"],
        bbox_reparam=config["bbox_reparam"],
        lite_refpoint_refine=config["lite_refpoint_refine"],
        group_detr=config["group_detr"],
        dim_feedforward=config["dim_feedforward"],
        weights=None,
        input_shape=(res, res, 3),
        name=variant,
    )
    dummy = np.random.rand(1, res, res, 3).astype("float32")
    _ = model(dummy)
    return model


def get_keras_weight_dict(model):
    """Build a dict from weight path -> weight variable."""
    weight_dict = {}
    for layer in model.layers:
        for w in layer.weights:
            weight_dict[w.path] = w
    return weight_dict


def transfer_weights(variant, torch_sd, keras_model):
    """Map PyTorch state_dict keys to Keras weights and assign values."""
    config = RF_DETR_MODEL_CONFIG[variant]
    num_layers = max(config["out_feature_indexes"]) + 1
    dec_layers = config["dec_layers"]

    kw = get_keras_weight_dict(keras_model)

    assigned = set()

    def assign(keras_path, torch_tensor, transpose=False):
        if keras_path not in kw:
            print(f"  WARN: Keras path not found: {keras_path}")
            return
        tensor = torch_tensor.detach().cpu().numpy()
        if transpose:
            tensor = tensor.T
        target = kw[keras_path]
        if tensor.shape != tuple(target.shape):
            print(
                f"  WARN: Shape mismatch for {keras_path}: "
                f"torch={tensor.shape}, keras={tuple(target.shape)}"
            )
            return
        target.assign(tensor)
        assigned.add(keras_path)

    # --- Backbone embeddings ---
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

    # --- Backbone encoder layers ---
    for i in range(num_layers):
        pt_prefix = f"backbone.0.encoder.encoder.encoder.layer.{i}"
        k_prefix = f"backbone_encoder/layer_{i}"

        assign(f"{k_prefix}/norm1/gamma", torch_sd[f"{pt_prefix}.norm1.weight"])
        assign(f"{k_prefix}/norm1/beta", torch_sd[f"{pt_prefix}.norm1.bias"])

        assign(
            f"{k_prefix}/attention/query/kernel",
            torch_sd[f"{pt_prefix}.attention.attention.query.weight"],
            transpose=True,
        )
        assign(
            f"{k_prefix}/attention/query/bias",
            torch_sd[f"{pt_prefix}.attention.attention.query.bias"],
        )
        assign(
            f"{k_prefix}/attention/key/kernel",
            torch_sd[f"{pt_prefix}.attention.attention.key.weight"],
            transpose=True,
        )
        assign(
            f"{k_prefix}/attention/key/bias",
            torch_sd[f"{pt_prefix}.attention.attention.key.bias"],
        )
        assign(
            f"{k_prefix}/attention/value/kernel",
            torch_sd[f"{pt_prefix}.attention.attention.value.weight"],
            transpose=True,
        )
        assign(
            f"{k_prefix}/attention/value/bias",
            torch_sd[f"{pt_prefix}.attention.attention.value.bias"],
        )
        assign(
            f"{k_prefix}/attention/out_proj/kernel",
            torch_sd[f"{pt_prefix}.attention.output.dense.weight"],
            transpose=True,
        )
        assign(
            f"{k_prefix}/attention/out_proj/bias",
            torch_sd[f"{pt_prefix}.attention.output.dense.bias"],
        )

        assign(
            f"{k_prefix}/layer_scale1/lambda1",
            torch_sd[f"{pt_prefix}.layer_scale1.lambda1"],
        )

        assign(f"{k_prefix}/norm2/gamma", torch_sd[f"{pt_prefix}.norm2.weight"])
        assign(f"{k_prefix}/norm2/beta", torch_sd[f"{pt_prefix}.norm2.bias"])

        if config["backbone_use_swiglu"]:
            assign(
                f"{k_prefix}/mlp/weights_in/kernel",
                torch_sd[f"{pt_prefix}.mlp.fc1.weight"],
                transpose=True,
            )
            assign(
                f"{k_prefix}/mlp/weights_in/bias",
                torch_sd[f"{pt_prefix}.mlp.fc1.bias"],
            )
            assign(
                f"{k_prefix}/mlp/weights_out/kernel",
                torch_sd[f"{pt_prefix}.mlp.fc2.weight"],
                transpose=True,
            )
            assign(
                f"{k_prefix}/mlp/weights_out/bias",
                torch_sd[f"{pt_prefix}.mlp.fc2.bias"],
            )
        else:
            assign(
                f"{k_prefix}/mlp/fc1/kernel",
                torch_sd[f"{pt_prefix}.mlp.fc1.weight"],
                transpose=True,
            )
            assign(
                f"{k_prefix}/mlp/fc1/bias",
                torch_sd[f"{pt_prefix}.mlp.fc1.bias"],
            )
            assign(
                f"{k_prefix}/mlp/fc2/kernel",
                torch_sd[f"{pt_prefix}.mlp.fc2.weight"],
                transpose=True,
            )
            assign(
                f"{k_prefix}/mlp/fc2/bias",
                torch_sd[f"{pt_prefix}.mlp.fc2.bias"],
            )

        assign(
            f"{k_prefix}/layer_scale2/lambda1",
            torch_sd[f"{pt_prefix}.layer_scale2.lambda1"],
        )

    assign(
        "backbone_encoder/layernorm/gamma",
        torch_sd["backbone.0.encoder.encoder.layernorm.weight"],
    )
    assign(
        "backbone_encoder/layernorm/beta",
        torch_sd["backbone.0.encoder.encoder.layernorm.bias"],
    )

    # --- Projector (C2f + ChannelLayerNorm) ---
    pt_proj = "backbone.0.projector.stages.0"

    assign(
        "projector_c2f/cv1/conv/kernel",
        torch_sd[f"{pt_proj}.0.cv1.conv.weight"].permute(2, 3, 1, 0),
    )
    assign(
        "projector_c2f/cv1/ln/gamma",
        torch_sd[f"{pt_proj}.0.cv1.bn.weight"],
    )
    assign(
        "projector_c2f/cv1/ln/beta",
        torch_sd[f"{pt_proj}.0.cv1.bn.bias"],
    )

    assign(
        "projector_c2f/cv2/conv/kernel",
        torch_sd[f"{pt_proj}.0.cv2.conv.weight"].permute(2, 3, 1, 0),
    )
    assign(
        "projector_c2f/cv2/ln/gamma",
        torch_sd[f"{pt_proj}.0.cv2.bn.weight"],
    )
    assign(
        "projector_c2f/cv2/ln/beta",
        torch_sd[f"{pt_proj}.0.cv2.bn.bias"],
    )

    for b_idx in range(3):
        pt_bn = f"{pt_proj}.0.m.{b_idx}"
        k_bn = f"projector_c2f/bottleneck_{b_idx}"
        for cv in ["cv1", "cv2"]:
            assign(
                f"{k_bn}/{cv}/conv/kernel",
                torch_sd[f"{pt_bn}.{cv}.conv.weight"].permute(2, 3, 1, 0),
            )
            assign(
                f"{k_bn}/{cv}/ln/gamma",
                torch_sd[f"{pt_bn}.{cv}.bn.weight"],
            )
            assign(
                f"{k_bn}/{cv}/ln/beta",
                torch_sd[f"{pt_bn}.{cv}.bn.bias"],
            )

    assign("projector_ln/gamma", torch_sd[f"{pt_proj}.1.weight"])
    assign("projector_ln/beta", torch_sd[f"{pt_proj}.1.bias"])

    # --- Encoder output (two-stage, group 0 only) ---
    assign(
        "enc_output_0/kernel",
        torch_sd["transformer.enc_output.0.weight"],
        transpose=True,
    )
    assign("enc_output_0/bias", torch_sd["transformer.enc_output.0.bias"])
    assign(
        "enc_output_norm_0/gamma",
        torch_sd["transformer.enc_output_norm.0.weight"],
    )
    assign(
        "enc_output_norm_0/beta",
        torch_sd["transformer.enc_output_norm.0.bias"],
    )

    assign(
        "enc_out_class_embed_0/kernel",
        torch_sd["transformer.enc_out_class_embed.0.weight"],
        transpose=True,
    )
    assign(
        "enc_out_class_embed_0/bias",
        torch_sd["transformer.enc_out_class_embed.0.bias"],
    )

    assign(
        "enc_bbox_0/kernel",
        torch_sd["transformer.enc_out_bbox_embed.0.layers.0.weight"],
        transpose=True,
    )
    assign(
        "enc_bbox_0/bias",
        torch_sd["transformer.enc_out_bbox_embed.0.layers.0.bias"],
    )
    assign(
        "enc_bbox_1/kernel",
        torch_sd["transformer.enc_out_bbox_embed.0.layers.1.weight"],
        transpose=True,
    )
    assign(
        "enc_bbox_1/bias",
        torch_sd["transformer.enc_out_bbox_embed.0.layers.1.bias"],
    )
    assign(
        "enc_bbox_2/kernel",
        torch_sd["transformer.enc_out_bbox_embed.0.layers.2.weight"],
        transpose=True,
    )
    assign(
        "enc_bbox_2/bias",
        torch_sd["transformer.enc_out_bbox_embed.0.layers.2.bias"],
    )

    # --- Ref point head ---
    assign(
        "ref_point_head_0/kernel",
        torch_sd["transformer.decoder.ref_point_head.layers.0.weight"],
        transpose=True,
    )
    assign(
        "ref_point_head_0/bias",
        torch_sd["transformer.decoder.ref_point_head.layers.0.bias"],
    )
    assign(
        "ref_point_head_1/kernel",
        torch_sd["transformer.decoder.ref_point_head.layers.1.weight"],
        transpose=True,
    )
    assign(
        "ref_point_head_1/bias",
        torch_sd["transformer.decoder.ref_point_head.layers.1.bias"],
    )

    # --- Decoder layers ---
    for i in range(dec_layers):
        pt_dl = f"transformer.decoder.layers.{i}"
        k_dl = f"decoder_layer_{i}"

        # Self-attention: fused in_proj_weight -> split into Q, K, V
        in_proj_w = torch_sd[f"{pt_dl}.self_attn.in_proj_weight"]
        in_proj_b = torch_sd[f"{pt_dl}.self_attn.in_proj_bias"]
        q_w, k_w, v_w = torch.chunk(in_proj_w, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(in_proj_b, 3, dim=0)

        assign(f"{k_dl}/self_attn_q_proj/kernel", q_w, transpose=True)
        assign(f"{k_dl}/self_attn_q_proj/bias", q_b)
        assign(f"{k_dl}/self_attn_k_proj/kernel", k_w, transpose=True)
        assign(f"{k_dl}/self_attn_k_proj/bias", k_b)
        assign(f"{k_dl}/self_attn_v_proj/kernel", v_w, transpose=True)
        assign(f"{k_dl}/self_attn_v_proj/bias", v_b)

        assign(
            f"{k_dl}/self_attn_out_proj/kernel",
            torch_sd[f"{pt_dl}.self_attn.out_proj.weight"],
            transpose=True,
        )
        assign(
            f"{k_dl}/self_attn_out_proj/bias",
            torch_sd[f"{pt_dl}.self_attn.out_proj.bias"],
        )

        assign(f"{k_dl}/norm1/gamma", torch_sd[f"{pt_dl}.norm1.weight"])
        assign(f"{k_dl}/norm1/beta", torch_sd[f"{pt_dl}.norm1.bias"])

        # Cross-attention (deformable)
        assign(
            f"{k_dl}/cross_attn/sampling_offsets/kernel",
            torch_sd[f"{pt_dl}.cross_attn.sampling_offsets.weight"],
            transpose=True,
        )
        assign(
            f"{k_dl}/cross_attn/sampling_offsets/bias",
            torch_sd[f"{pt_dl}.cross_attn.sampling_offsets.bias"],
        )
        assign(
            f"{k_dl}/cross_attn/attention_weights/kernel",
            torch_sd[f"{pt_dl}.cross_attn.attention_weights.weight"],
            transpose=True,
        )
        assign(
            f"{k_dl}/cross_attn/attention_weights/bias",
            torch_sd[f"{pt_dl}.cross_attn.attention_weights.bias"],
        )
        assign(
            f"{k_dl}/cross_attn/value_proj/kernel",
            torch_sd[f"{pt_dl}.cross_attn.value_proj.weight"],
            transpose=True,
        )
        assign(
            f"{k_dl}/cross_attn/value_proj/bias",
            torch_sd[f"{pt_dl}.cross_attn.value_proj.bias"],
        )
        assign(
            f"{k_dl}/cross_attn/output_proj/kernel",
            torch_sd[f"{pt_dl}.cross_attn.output_proj.weight"],
            transpose=True,
        )
        assign(
            f"{k_dl}/cross_attn/output_proj/bias",
            torch_sd[f"{pt_dl}.cross_attn.output_proj.bias"],
        )

        assign(f"{k_dl}/norm2/gamma", torch_sd[f"{pt_dl}.norm2.weight"])
        assign(f"{k_dl}/norm2/beta", torch_sd[f"{pt_dl}.norm2.bias"])

        assign(
            f"{k_dl}/linear1/kernel",
            torch_sd[f"{pt_dl}.linear1.weight"],
            transpose=True,
        )
        assign(f"{k_dl}/linear1/bias", torch_sd[f"{pt_dl}.linear1.bias"])
        assign(
            f"{k_dl}/linear2/kernel",
            torch_sd[f"{pt_dl}.linear2.weight"],
            transpose=True,
        )
        assign(f"{k_dl}/linear2/bias", torch_sd[f"{pt_dl}.linear2.bias"])

        assign(f"{k_dl}/norm3/gamma", torch_sd[f"{pt_dl}.norm3.weight"])
        assign(f"{k_dl}/norm3/beta", torch_sd[f"{pt_dl}.norm3.bias"])

    # --- Decoder norm ---
    assign(
        "decoder_norm/gamma",
        torch_sd["transformer.decoder.norm.weight"],
    )
    assign(
        "decoder_norm/beta",
        torch_sd["transformer.decoder.norm.bias"],
    )

    # --- Learned query embeddings (slice to first num_queries for inference) ---
    num_queries = config["num_queries"]
    assign(
        "refpoint_embed_layer/weight",
        torch_sd["refpoint_embed.weight"][:num_queries],
    )
    assign(
        "query_feat_embed/weight",
        torch_sd["query_feat.weight"][:num_queries],
    )

    # --- Output heads ---
    assign("class_embed/kernel", torch_sd["class_embed.weight"], transpose=True)
    assign("class_embed/bias", torch_sd["class_embed.bias"])

    assign(
        "bbox_embed_0/kernel",
        torch_sd["bbox_embed.layers.0.weight"],
        transpose=True,
    )
    assign("bbox_embed_0/bias", torch_sd["bbox_embed.layers.0.bias"])
    assign(
        "bbox_embed_1/kernel",
        torch_sd["bbox_embed.layers.1.weight"],
        transpose=True,
    )
    assign("bbox_embed_1/bias", torch_sd["bbox_embed.layers.1.bias"])
    assign(
        "bbox_embed_2/kernel",
        torch_sd["bbox_embed.layers.2.weight"],
        transpose=True,
    )
    assign("bbox_embed_2/bias", torch_sd["bbox_embed.layers.2.bias"])

    print(f"\nAssigned {len(assigned)} Keras weight tensors")
    total_keras = sum(1 for _ in get_keras_weight_dict(keras_model).keys())
    if len(assigned) < total_keras:
        unassigned = set(get_keras_weight_dict(keras_model).keys()) - assigned
        non_bn_stats = [k for k in unassigned if "moving_" not in k]
        if non_bn_stats:
            print(f"Unassigned non-BN-stat weights: {len(non_bn_stats)}")
            for k in sorted(non_bn_stats):
                print(f"  {k}")


def verify_equivalence(variant, keras_model, torch_sd):
    """Verify Keras model produces similar outputs to PyTorch model."""
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
    keras_logits = keras_out["pred_logits"].detach().cpu().numpy()
    keras_boxes = keras_out["pred_boxes"].detach().cpu().numpy()

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
        default="RFDETRBase",
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
    torch_sd = load_torch_state_dict(variant)
    print(f"  PyTorch keys: {len(torch_sd)}")

    print("Transferring weights...")
    transfer_weights(variant, torch_sd, keras_model)

    if not args.skip_verify:
        print("\nVerifying model equivalence...")
        verify_equivalence(variant, keras_model, torch_sd)

    print(f"\nSaving Keras weights to {output}...")
    keras_model.save_weights(output)
    print("Done!")


if __name__ == "__main__":
    main()
