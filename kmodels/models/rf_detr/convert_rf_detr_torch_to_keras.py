import argparse
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
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
    out_feature_indexes = config.get("out_feature_indexes", [3, 6, 9, 12])
    num_layers = max(out_feature_indexes)
    dec_layers = config["dec_layers"]

    kw = get_keras_weight_dict(keras_model)

    assigned = set()

    def assign(keras_path, torch_tensor, transpose=False):
        if keras_path not in kw:
            print(f"  WARN: Keras path not found: {keras_path}")
            return
        tensor = keras.ops.convert_to_tensor(torch_tensor.detach().cpu())
        if transpose:
            tensor = keras.ops.transpose(tensor)
        target = kw[keras_path]
        if tensor.shape != tuple(target.shape):
            print(
                f"  WARN: Shape mismatch for {keras_path}: "
                f"torch={tensor.shape}, keras={tuple(target.shape)}"
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

    # --- Projector (C2f + ChannelLayerNorm) ---
    pt_proj = "backbone.0.projector.stages.0"

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
    num_queries = 300
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

    test_input = keras.random.uniform((1, res, res, 3), dtype="float32", seed=42)

    pt_input = torch.tensor(keras.ops.convert_to_numpy(test_input)).permute(0, 3, 1, 2)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    pt_input_norm = normalize(pt_input)

    import rfdetr as rfdetr_pkg

    pt_wrapper = getattr(rfdetr_pkg, variant.replace("RFDETR", "RFDETR"))()
    pt_model = pt_wrapper.model.model
    pt_model.eval()

    with torch.no_grad():
        pt_out = pt_model(pt_input_norm)
    pt_logits = keras.ops.convert_to_tensor(pt_out["pred_logits"].cpu())
    pt_boxes = keras.ops.convert_to_tensor(pt_out["pred_boxes"].cpu())

    mean = keras.ops.reshape(
        keras.ops.convert_to_tensor([0.485, 0.456, 0.406], dtype="float32"),
        (1, 1, 1, 3),
    )
    std = keras.ops.reshape(
        keras.ops.convert_to_tensor([0.229, 0.224, 0.225], dtype="float32"),
        (1, 1, 1, 3),
    )
    keras_input_norm = keras.ops.cast((test_input - mean) / std, dtype="float32")

    keras_out = keras_model(keras_input_norm, training=False)
    keras_logits = keras.ops.convert_to_tensor(keras_out["pred_logits"].detach().cpu())
    keras_boxes = keras.ops.convert_to_tensor(keras_out["pred_boxes"].detach().cpu())

    logits_diff = float(keras.ops.max(keras.ops.abs(pt_logits - keras_logits)))
    boxes_diff = float(keras.ops.max(keras.ops.abs(pt_boxes - keras_boxes)))

    pt_flat = keras.ops.reshape(pt_logits, (-1,))
    k_flat = keras.ops.reshape(keras_logits, (-1,))
    logits_cos = float(
        keras.ops.dot(pt_flat, k_flat)
        / (keras.ops.norm(pt_flat) * keras.ops.norm(k_flat) + 1e-8)
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
