"""Convert D-FINE weights from HuggingFace to Keras format."""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForObjectDetection

from kmodels.models.dfine.dfine_model import (
    DFineSmall,
)


def transfer_weights_kernel(keras_var, torch_arr):
    """Transfer conv kernel: torch (out, in, H, W) -> keras (H, W, in, out)."""
    if torch_arr.ndim == 4:
        keras_var.assign(np.transpose(torch_arr, (2, 3, 1, 0)))
    elif torch_arr.ndim == 2:
        keras_var.assign(torch_arr.T)
    elif torch_arr.ndim == 1:
        keras_var.assign(torch_arr)
    else:
        keras_var.assign(torch_arr)


def transfer_dense(keras_layer, sd, hf_prefix):
    """Transfer Dense layer weights."""
    w_key = f"{hf_prefix}.weight"
    b_key = f"{hf_prefix}.bias"
    if w_key in sd:
        keras_layer.kernel.assign(sd[w_key].T)
    if b_key in sd:
        keras_layer.bias.assign(sd[b_key])


def transfer_bn(keras_layer, sd, hf_prefix):
    """Transfer BatchNorm weights."""
    mapping = {
        "gamma": "weight",
        "beta": "bias",
        "moving_mean": "running_mean",
        "moving_variance": "running_var",
    }
    for kvar in keras_layer.weights:
        kname = kvar.name.split("/")[-1]
        # Find the matching torch key
        for k_suffix, t_suffix in mapping.items():
            if k_suffix in kname:
                t_key = f"{hf_prefix}.{t_suffix}"
                if t_key in sd:
                    kvar.assign(sd[t_key])
                break


def transfer_conv_bn(
    keras_model, sd, keras_conv_name, keras_bn_name, hf_conv_key, hf_bn_prefix
):
    """Transfer Conv2D + BatchNorm pair."""
    conv = keras_model.get_layer(keras_conv_name)
    transfer_weights_kernel(conv.kernel, sd[hf_conv_key])
    bn = keras_model.get_layer(keras_bn_name)
    transfer_bn(bn, sd, hf_bn_prefix)


def convert_dfine_small():
    """Convert D-FINE Small from HF to Keras."""
    hf_name = "ustc-community/dfine-small-coco"
    print(f"\nConverting {hf_name}...")

    torch_model = AutoModelForObjectDetection.from_pretrained(
        hf_name,
        trust_remote_code=True,
    ).eval()
    sd = {k: v.cpu().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = DFineSmall(
        weights=None,
        input_shape=(640, 640, 3),
        num_queries=300,
        num_labels=80,
    )
    print(f"  Keras params: {keras_model.count_params():,}")

    # ---- Backbone stem ----
    print("Transferring backbone stem...")
    stem_pairs = [
        ("backbone_stem1", "model.backbone.model.embedder.stem1"),
        ("backbone_stem2a", "model.backbone.model.embedder.stem2a"),
        ("backbone_stem2b", "model.backbone.model.embedder.stem2b"),
        ("backbone_stem3", "model.backbone.model.embedder.stem3"),
        ("backbone_stem4", "model.backbone.model.embedder.stem4"),
    ]
    for k_prefix, hf_prefix in stem_pairs:
        conv = keras_model.get_layer(f"{k_prefix}_conv")
        transfer_weights_kernel(conv.kernel, sd[f"{hf_prefix}.convolution.weight"])
        bn = keras_model.get_layer(f"{k_prefix}_bn")
        transfer_bn(bn, sd, f"{hf_prefix}.normalization")
        # LAB
        lab_scale_key = f"{hf_prefix}.lab.scale"
        if lab_scale_key in sd:
            try:
                lab = keras_model.get_layer(f"{k_prefix}_lab")
                lab.scale.assign(sd[lab_scale_key])
                lab.bias.assign(sd[f"{hf_prefix}.lab.bias"])
            except ValueError:
                pass

    # ---- Backbone stages ----
    print("Transferring backbone stages...")
    bb_cfg = torch_model.config.backbone_config
    if hasattr(bb_cfg, "to_dict"):
        bb_cfg = bb_cfg.to_dict()
    stage_num_blocks = bb_cfg["stage_num_blocks"]
    stage_numb_of_layers = bb_cfg["stage_numb_of_layers"]
    stage_light_block = bb_cfg["stage_light_block"]
    use_lab = bb_cfg["use_learnable_affine_block"]

    for si in range(4):
        # Downsample
        if bb_cfg["stage_downsample"][si]:
            hf_ds = f"model.backbone.model.encoder.stages.{si}.downsample"
            k_ds = f"backbone_stage{si}_downsample"
            ds_conv_key = f"{hf_ds}.convolution.weight"
            if ds_conv_key in sd:
                conv = keras_model.get_layer(f"{k_ds}_conv")
                transfer_weights_kernel(conv.kernel, sd[ds_conv_key])
                bn = keras_model.get_layer(f"{k_ds}_bn")
                transfer_bn(bn, sd, f"{hf_ds}.normalization")

        nb = stage_num_blocks[si]
        nl = stage_numb_of_layers[si]
        light = stage_light_block[si]

        for bi in range(nb):
            hf_blk = f"model.backbone.model.encoder.stages.{si}.blocks.{bi}"
            k_blk = f"backbone_stage{si}_block{bi}"

            for li in range(nl):
                if light:
                    # LightConvBlock: conv1 (1x1 pointwise) + conv2 (depthwise)
                    hf_layer = f"{hf_blk}.layers.{li}"
                    k_layer = f"{k_blk}_layers_{li}"
                    for sub in ["conv1", "conv2"]:
                        hf_sub = f"{hf_layer}.{sub}"
                        k_sub = f"{k_layer}_{sub}"
                        conv = keras_model.get_layer(f"{k_sub}_conv")
                        transfer_weights_kernel(
                            conv.kernel, sd[f"{hf_sub}.convolution.weight"]
                        )
                        bn = keras_model.get_layer(f"{k_sub}_bn")
                        transfer_bn(bn, sd, f"{hf_sub}.normalization")
                        # LAB
                        lab_key = f"{hf_sub}.lab.scale"
                        if lab_key in sd and use_lab:
                            try:
                                lab = keras_model.get_layer(f"{k_sub}_lab")
                                lab.scale.assign(sd[lab_key])
                                lab.bias.assign(sd[f"{hf_sub}.lab.bias"])
                            except ValueError:
                                pass
                else:
                    # Regular conv layer
                    hf_layer = f"{hf_blk}.layers.{li}"
                    k_layer = f"{k_blk}_layers_{li}"
                    conv = keras_model.get_layer(f"{k_layer}_conv")
                    transfer_weights_kernel(
                        conv.kernel, sd[f"{hf_layer}.convolution.weight"]
                    )
                    bn = keras_model.get_layer(f"{k_layer}_bn")
                    transfer_bn(bn, sd, f"{hf_layer}.normalization")
                    lab_key = f"{hf_layer}.lab.scale"
                    if lab_key in sd and use_lab:
                        try:
                            lab = keras_model.get_layer(f"{k_layer}_lab")
                            lab.scale.assign(sd[lab_key])
                            lab.bias.assign(sd[f"{hf_layer}.lab.bias"])
                        except ValueError:
                            pass

            # Aggregation (2 conv layers)
            for ai in range(2):
                hf_agg = f"{hf_blk}.aggregation.{ai}"
                k_agg = f"{k_blk}_agg_{ai}"
                conv = keras_model.get_layer(f"{k_agg}_conv")
                transfer_weights_kernel(conv.kernel, sd[f"{hf_agg}.convolution.weight"])
                bn = keras_model.get_layer(f"{k_agg}_bn")
                transfer_bn(bn, sd, f"{hf_agg}.normalization")
                lab_key = f"{hf_agg}.lab.scale"
                if lab_key in sd and use_lab:
                    try:
                        lab = keras_model.get_layer(f"{k_agg}_lab")
                        lab.scale.assign(sd[lab_key])
                        lab.bias.assign(sd[f"{hf_agg}.lab.bias"])
                    except ValueError:
                        pass

    # ---- Encoder input projections ----
    print("Transferring encoder input projections...")
    n_enc_proj = len(torch_model.config.encoder_in_channels)
    for i in range(n_enc_proj):
        conv = keras_model.get_layer(f"encoder_input_proj_{i}_conv")
        transfer_weights_kernel(
            conv.kernel, sd[f"model.encoder_input_proj.{i}.0.weight"]
        )
        bn = keras_model.get_layer(f"encoder_input_proj_{i}_bn")
        transfer_bn(bn, sd, f"model.encoder_input_proj.{i}.1")

    # ---- AIFI encoder ----
    print("Transferring AIFI encoder...")
    hf_aifi = "model.encoder.aifi.0.layers.0"
    # Self-attention (sub-layers accessed through the MHA layer)
    sa = keras_model.get_layer("aifi_0_layers_0_self_attn")
    for proj_name, hf_name in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", "o_proj"),
    ]:
        dense = getattr(sa, proj_name)
        transfer_dense(dense, sd, f"{hf_aifi}.self_attn.{hf_name}")

    # Layer norms
    ln1 = keras_model.get_layer("aifi_0_layers_0_self_attn_layer_norm")
    transfer_bn(ln1, sd, f"{hf_aifi}.self_attn_layer_norm")
    ln2 = keras_model.get_layer("aifi_0_layers_0_final_layer_norm")
    transfer_bn(ln2, sd, f"{hf_aifi}.final_layer_norm")

    # FFN (note: HF uses mlp.layers.0 / mlp.layers.1)
    fc1 = keras_model.get_layer("aifi_0_layers_0_fc1")
    transfer_dense(fc1, sd, f"{hf_aifi}.mlp.layers.0")
    fc2 = keras_model.get_layer("aifi_0_layers_0_fc2")
    transfer_dense(fc2, sd, f"{hf_aifi}.mlp.layers.1")

    # ---- CCFM ----
    print("Transferring CCFM...")
    num_fpn = torch_model.config.num_feature_levels - 1

    # Lateral convs
    for i in range(num_fpn):
        hf_lc = f"model.encoder.lateral_convs.{i}"
        conv = keras_model.get_layer(f"lateral_convs_{i}_conv")
        transfer_weights_kernel(conv.kernel, sd[f"{hf_lc}.conv.weight"])
        bn = keras_model.get_layer(f"lateral_convs_{i}_norm")
        transfer_bn(bn, sd, f"{hf_lc}.norm")

    # FPN blocks (RepNCSPELAN4)
    for i in range(num_fpn):
        _transfer_rep_ncspelan4(
            keras_model, sd, f"fpn_blocks_{i}", f"model.encoder.fpn_blocks.{i}"
        )

    # Downsample convs (SCDown)
    for i in range(num_fpn):
        for sub in ["conv1", "conv2"]:
            hf_dc = f"model.encoder.downsample_convs.{i}.{sub}"
            k_dc = f"downsample_convs_{i}_{sub}"
            conv = keras_model.get_layer(f"{k_dc}_conv")
            transfer_weights_kernel(conv.kernel, sd[f"{hf_dc}.conv.weight"])
            bn = keras_model.get_layer(f"{k_dc}_norm")
            transfer_bn(bn, sd, f"{hf_dc}.norm")

    # PAN blocks
    for i in range(num_fpn):
        _transfer_rep_ncspelan4(
            keras_model, sd, f"pan_blocks_{i}", f"model.encoder.pan_blocks.{i}"
        )

    # ---- Decoder input projections ----
    print("Transferring decoder input projections...")
    n_dec_proj = torch_model.config.num_feature_levels
    for i in range(n_dec_proj):
        try:
            conv = keras_model.get_layer(f"decoder_input_proj_{i}_conv")
            transfer_weights_kernel(
                conv.kernel, sd[f"model.decoder_input_proj.{i}.0.weight"]
            )
            bn = keras_model.get_layer(f"decoder_input_proj_{i}_bn")
            transfer_bn(bn, sd, f"model.decoder_input_proj.{i}.1")
        except (ValueError, KeyError):
            pass

    # ---- Encoder output heads ----
    print("Transferring encoder output heads...")
    transfer_dense(keras_model.get_layer("enc_output_linear"), sd, "model.enc_output.0")
    ln_enc = keras_model.get_layer("enc_output_layernorm")
    transfer_bn(ln_enc, sd, "model.enc_output.1")
    transfer_dense(keras_model.get_layer("enc_score_head"), sd, "model.enc_score_head")
    for j in range(3):
        transfer_dense(
            keras_model.get_layer(f"enc_bbox_head_{j}"),
            sd,
            f"model.enc_bbox_head.layers.{j}",
        )

    # ---- Decoder query_pos_head ----
    print("Transferring decoder query_pos_head...")
    for j in range(2):
        transfer_dense(
            keras_model.get_layer(f"query_pos_head_{j}"),
            sd,
            f"model.decoder.query_pos_head.layers.{j}",
        )

    # ---- Decoder pre_bbox_head ----
    print("Transferring pre_bbox_head...")
    for j in range(3):
        transfer_dense(
            keras_model.get_layer(f"pre_bbox_head_{j}"),
            sd,
            f"model.decoder.pre_bbox_head.layers.{j}",
        )

    # ---- Decoder up / reg_scale ----
    print("Transferring up / reg_scale...")
    dp = keras_model.get_layer("decoder_params")
    dp.up.assign(sd["model.decoder.up"])
    dp.reg_scale.assign(sd["model.decoder.reg_scale"])

    # ---- Decoder layers ----
    num_dec = torch_model.config.decoder_layers
    for i in tqdm(range(num_dec), desc="Transferring decoder layers"):
        hf_dl = f"model.decoder.layers.{i}"
        k_dl = f"decoder_layers_{i}"
        dec_layer = keras_model.get_layer(k_dl)

        # Self-attention (access sub-layers through the MHA layer object)
        sa = dec_layer.self_attn
        for proj_name, hf_name in [
            ("q_proj", "q_proj"),
            ("k_proj", "k_proj"),
            ("v_proj", "v_proj"),
            ("out_proj", "o_proj"),
        ]:
            dense = getattr(sa, proj_name)
            transfer_dense(dense, sd, f"{hf_dl}.self_attn.{hf_name}")

        # Self-attn layer norm
        transfer_bn(dec_layer.self_attn_layer_norm, sd, f"{hf_dl}.self_attn_layer_norm")

        # Encoder attention (access sub-layers through the deformable attn)
        ea = dec_layer.encoder_attn
        transfer_dense(
            ea.sampling_offsets, sd, f"{hf_dl}.encoder_attn.sampling_offsets"
        )
        transfer_dense(
            ea.attention_weights_proj, sd, f"{hf_dl}.encoder_attn.attention_weights"
        )
        # num_points_scale
        ea.num_points_scale.assign(sd[f"{hf_dl}.encoder_attn.num_points_scale"])

        # Gateway
        transfer_dense(dec_layer.gateway_gate, sd, f"{hf_dl}.gateway.gate")
        transfer_bn(dec_layer.gateway_norm, sd, f"{hf_dl}.gateway.norm")

        # FFN (MLP)
        transfer_dense(dec_layer.fc1, sd, f"{hf_dl}.mlp.layers.0")
        transfer_dense(dec_layer.fc2, sd, f"{hf_dl}.mlp.layers.1")

        # Final layer norm
        transfer_bn(dec_layer.final_layer_norm, sd, f"{hf_dl}.final_layer_norm")

    # ---- Detection heads ----
    for i in tqdm(range(num_dec), desc="Transferring detection heads"):
        transfer_dense(
            keras_model.get_layer(f"class_embed_{i}"),
            sd,
            f"model.decoder.class_embed.{i}",
        )
        for j in range(3):
            transfer_dense(
                keras_model.get_layer(f"bbox_embed_{i}_{j}"),
                sd,
                f"model.decoder.bbox_embed.{i}.layers.{j}",
            )

    # ---- LQE layers ----
    for i in tqdm(range(num_dec), desc="Transferring LQE layers"):
        transfer_dense(
            keras_model.get_layer(f"lqe_{i}_0"),
            sd,
            f"model.decoder.lqe_layers.{i}.reg_conf.layers.0",
        )
        transfer_dense(
            keras_model.get_layer(f"lqe_{i}_1"),
            sd,
            f"model.decoder.lqe_layers.{i}.reg_conf.layers.1",
        )

    # ---- Verify ----
    print("\nVerifying model equivalence...")
    np.random.seed(42)
    test_input = np.random.rand(1, 640, 640, 3).astype(np.float32)

    hf_input = torch.tensor(test_input).permute(0, 3, 1, 2)
    with torch.no_grad():
        hf_output = torch_model(hf_input)
        hf_logits = hf_output.logits.numpy()
        hf_boxes = hf_output.pred_boxes.numpy()

    keras_output = keras_model.predict(test_input, verbose=0)
    keras_logits = np.asarray(keras_output["logits"])
    keras_boxes = np.asarray(keras_output["pred_boxes"])

    logits_diff = np.max(np.abs(hf_logits - keras_logits))
    boxes_diff = np.max(np.abs(hf_boxes - keras_boxes))

    hf_flat = hf_logits.flatten()
    k_flat = keras_logits.flatten()
    logits_cos = float(
        np.dot(hf_flat, k_flat)
        / (np.linalg.norm(hf_flat) * np.linalg.norm(k_flat) + 1e-8)
    )

    print(f"Max logits diff:   {logits_diff:.6f}")
    print(f"Max boxes diff:    {boxes_diff:.6f}")
    print(f"Logits cosine sim: {logits_cos:.6f}")

    output_file = "dfine_small_coco.weights.h5"
    keras_model.save_weights(output_file)
    print(f"Model saved as {output_file}")

    del keras_model, torch_model, sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _transfer_rep_ncspelan4(keras_model, sd, k_prefix, hf_prefix):
    """Transfer RepNCSPELAN4 block weights."""
    # conv1, conv2, conv3, conv4
    for conv_name in ["conv1", "conv2", "conv3", "conv4"]:
        hf_cn = f"{hf_prefix}.{conv_name}"
        k_cn = f"{k_prefix}_{conv_name}"
        try:
            conv = keras_model.get_layer(f"{k_cn}_conv")
            transfer_weights_kernel(conv.kernel, sd[f"{hf_cn}.conv.weight"])
            bn = keras_model.get_layer(f"{k_cn}_norm")
            transfer_bn(bn, sd, f"{hf_cn}.norm")
        except (ValueError, KeyError):
            pass

    # csp_rep1 and csp_rep2
    for csp_name in ["csp_rep1", "csp_rep2"]:
        hf_csp = f"{hf_prefix}.{csp_name}"
        k_csp = f"{k_prefix}_{csp_name}"
        for sub in ["conv1", "conv2", "conv3"]:
            hf_sub = f"{hf_csp}.{sub}"
            k_sub = f"{k_csp}_{sub}"
            conv_key = f"{hf_sub}.conv.weight"
            if conv_key in sd:
                try:
                    conv = keras_model.get_layer(f"{k_sub}_conv")
                    transfer_weights_kernel(conv.kernel, sd[conv_key])
                    bn = keras_model.get_layer(f"{k_sub}_norm")
                    transfer_bn(bn, sd, f"{hf_sub}.norm")
                except ValueError:
                    pass
        # Bottlenecks
        for bi in range(10):  # max bottlenecks
            for sub in ["conv1", "conv2"]:
                hf_bn_key = f"{hf_csp}.bottlenecks.{bi}.{sub}.conv.weight"
                if hf_bn_key not in sd:
                    break
                k_bn = f"{k_csp}_bottlenecks_{bi}_{sub}"
                try:
                    conv = keras_model.get_layer(f"{k_bn}_conv")
                    transfer_weights_kernel(conv.kernel, sd[hf_bn_key])
                    bn = keras_model.get_layer(f"{k_bn}_norm")
                    transfer_bn(bn, sd, f"{hf_csp}.bottlenecks.{bi}.{sub}.norm")
                except ValueError:
                    pass


if __name__ == "__main__":
    convert_dfine_small()
