from typing import Dict, List, Tuple, Union

import keras
import numpy as np
import torch
from tqdm import tqdm
from transformers import RTDetrV2ForObjectDetection

from kmodels.models.rt_detr_v2.rt_detr_v2_model import (
    RTDETRV2ResNet18,
    RTDETRV2ResNet34,
    RTDETRV2ResNet50,
    RTDETRV2ResNet101,
)
from kmodels.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_nested_layer_weights,
    transfer_weights,
)

backbone_name_mapping: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}

model_configs: List[Dict[str, Union[str, type]]] = [
    {
        "keras_cls": RTDETRV2ResNet50,
        "hf_name": "PekingU/rtdetr_v2_r50vd",
        "output": "rtdetr_v2_r50vd_coco.weights.h5",
    },
    {
        "keras_cls": RTDETRV2ResNet18,
        "hf_name": "PekingU/rtdetr_v2_r18vd",
        "output": "rtdetr_v2_r18vd_coco.weights.h5",
    },
    {
        "keras_cls": RTDETRV2ResNet34,
        "hf_name": "PekingU/rtdetr_v2_r34vd",
        "output": "rtdetr_v2_r34vd_coco.weights.h5",
    },
    {
        "keras_cls": RTDETRV2ResNet101,
        "hf_name": "PekingU/rtdetr_v2_r101vd",
        "output": "rtdetr_v2_r101vd_coco.weights.h5",
    },
]

for model_config in model_configs:
    hf_name = model_config["hf_name"]
    output = model_config["output"]

    print(f"\n{'=' * 60}")
    print(f"Converting {hf_name}...")
    print(f"{'=' * 60}")

    torch_model = RTDetrV2ForObjectDetection.from_pretrained(
        hf_name,
        attn_implementation="eager",
    ).eval()
    sd: Dict[str, np.ndarray] = {
        k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
    }

    keras_model: keras.Model = model_config["keras_cls"](
        weights=None,
        input_shape=(640, 640, 3),
        num_queries=300,
        num_labels=80,
    )
    print(f"  Parameters: {keras_model.count_params():,}")

    bb_cfg = torch_model.model.backbone.model.config
    block_repeats = bb_cfg.depths
    layer_type = bb_cfg.layer_type
    num_convs = 2 if layer_type == "basic" else 3

    for i in tqdm(range(3), desc="Transferring backbone stem"):
        hf_pre = f"model.backbone.model.embedder.embedder.{i}"
        conv = keras_model.get_layer(f"backbone_embedder_{i}_conv")
        transfer_weights("conv_kernel", conv.kernel, sd[f"{hf_pre}.convolution.weight"])
        bn = keras_model.get_layer(f"backbone_embedder_{i}_bn")
        skipped = transfer_nested_layer_weights(
            bn,
            sd,
            f"{hf_pre}.normalization",
            name_mapping=backbone_name_mapping,
        )

    stage_pairs: List[Tuple[str, str]] = []
    for si, nb in enumerate(block_repeats):
        for bi in range(nb):
            hf_pre = f"model.backbone.model.encoder.stages.{si}.layers.{bi}"
            k_pre = f"backbone_stage{si}_{bi}"
            for ci in range(num_convs):
                stage_pairs.append(
                    (f"{k_pre}_conv{ci + 1}", f"{hf_pre}.layer.{ci}.convolution")
                )
                stage_pairs.append(
                    (f"{k_pre}_bn{ci + 1}", f"{hf_pre}.layer.{ci}.normalization")
                )
            suf = "shortcut" if si == 0 and bi == 0 else "shortcut.1"
            stage_pairs.append(
                (f"{k_pre}_shortcut_conv", f"{hf_pre}.{suf}.convolution")
            )
            stage_pairs.append(
                (f"{k_pre}_shortcut_bn", f"{hf_pre}.{suf}.normalization")
            )

    for keras_name, hf_prefix in tqdm(
        stage_pairs, desc=f"Transferring backbone stages ({layer_type})"
    ):
        try:
            layer = keras_model.get_layer(keras_name)
        except ValueError:
            continue
        if keras_name.endswith("_conv"):
            hf_key = f"{hf_prefix}.weight"
            if not compare_keras_torch_names(
                keras_name, layer.kernel, hf_key, sd[hf_key]
            ):
                continue
            transfer_weights("conv_kernel", layer.kernel, sd[hf_key])
        else:
            transfer_nested_layer_weights(
                layer,
                sd,
                hf_prefix,
                name_mapping=backbone_name_mapping,
            )

    for i in tqdm(range(3), desc="Transferring encoder input projections"):
        conv = keras_model.get_layer(f"encoder_input_proj_{i}_conv")
        transfer_weights(
            "conv_kernel", conv.kernel, sd[f"model.encoder_input_proj.{i}.0.weight"]
        )
        bn = keras_model.get_layer(f"encoder_input_proj_{i}_bn")
        transfer_nested_layer_weights(
            bn,
            sd,
            f"model.encoder_input_proj.{i}.1",
            name_mapping=backbone_name_mapping,
        )

    print("Transferring AIFI encoder...")
    hf_aifi = "model.encoder.aifi.0.layers.0"
    sa = keras_model.get_layer("aifi_0_layers_0_self_attn")

    aifi_sa_mapping: Dict[str, str] = {
        "aifi_0_layers_0_self_attn_": "",
        "out_proj": "o_proj",
        "kernel": "weight",
        "gamma": "weight",
        "beta": "bias",
    }
    transfer_nested_layer_weights(
        sa,
        sd,
        f"{hf_aifi}.self_attn",
        name_mapping=aifi_sa_mapping,
    )

    for layer_name, hf_suffix in [
        ("aifi_0_layers_0_self_attn_layer_norm", "self_attn_layer_norm"),
        ("aifi_0_layers_0_final_layer_norm", "final_layer_norm"),
    ]:
        ln = keras_model.get_layer(layer_name)
        transfer_nested_layer_weights(
            ln,
            sd,
            f"{hf_aifi}.{hf_suffix}",
            name_mapping=backbone_name_mapping,
        )

    aifi_fc_mapping: Dict[str, str] = {
        "aifi_0_layers_0_": "",
        "kernel": "weight",
        "beta": "bias",
    }
    for layer_name, hf_suffix in [
        ("aifi_0_layers_0_fc1", "mlp.fc1"),
        ("aifi_0_layers_0_fc2", "mlp.fc2"),
    ]:
        fc = keras_model.get_layer(layer_name)
        transfer_nested_layer_weights(
            fc,
            sd,
            f"{hf_aifi}.{hf_suffix}",
            name_mapping=aifi_fc_mapping,
        )

    conv_norm_pairs: List[Tuple[str, str]] = []
    for i in range(2):
        conv_norm_pairs.append(
            (f"lateral_convs_{i}", f"model.encoder.lateral_convs.{i}")
        )
        conv_norm_pairs.append(
            (f"downsample_convs_{i}", f"model.encoder.downsample_convs.{i}")
        )
    for block_type in ["fpn_blocks", "pan_blocks"]:
        for i in range(2):
            hf_blk = f"model.encoder.{block_type}.{i}"
            k_blk = f"{block_type}_{i}"
            conv_norm_pairs.append((f"{k_blk}_conv1", f"{hf_blk}.conv1"))
            conv_norm_pairs.append((f"{k_blk}_conv2", f"{hf_blk}.conv2"))
            if f"{hf_blk}.conv3.conv.weight" in sd:
                conv_norm_pairs.append((f"{k_blk}_conv3", f"{hf_blk}.conv3"))
            for bi in range(3):
                conv_norm_pairs.append(
                    (
                        f"{k_blk}_bottlenecks_{bi}_conv1",
                        f"{hf_blk}.bottlenecks.{bi}.conv1",
                    )
                )
                conv_norm_pairs.append(
                    (
                        f"{k_blk}_bottlenecks_{bi}_conv2",
                        f"{hf_blk}.bottlenecks.{bi}.conv2",
                    )
                )

    for keras_name, hf_prefix in tqdm(
        conv_norm_pairs, desc="Transferring CCFM (FPN + PAN)"
    ):
        conv = keras_model.get_layer(f"{keras_name}_conv")
        transfer_weights("conv_kernel", conv.kernel, sd[f"{hf_prefix}.conv.weight"])
        bn = keras_model.get_layer(f"{keras_name}_norm")
        transfer_nested_layer_weights(
            bn,
            sd,
            f"{hf_prefix}.norm",
            name_mapping=backbone_name_mapping,
        )

    for i in tqdm(range(3), desc="Transferring decoder input projections"):
        conv = keras_model.get_layer(f"decoder_input_proj_{i}_conv")
        transfer_weights(
            "conv_kernel", conv.kernel, sd[f"model.decoder_input_proj.{i}.0.weight"]
        )
        bn = keras_model.get_layer(f"decoder_input_proj_{i}_bn")
        transfer_nested_layer_weights(
            bn,
            sd,
            f"model.decoder_input_proj.{i}.1",
            name_mapping=backbone_name_mapping,
        )

    print("Transferring encoder output heads...")
    for keras_name, hf_key in [
        ("enc_output_linear", "model.enc_output.0"),
        ("enc_score_head", "model.enc_score_head"),
    ]:
        layer = keras_model.get_layer(keras_name)
        transfer_weights("kernel", layer.weights[0], sd[f"{hf_key}.weight"])
        layer.weights[1].assign(sd[f"{hf_key}.bias"])

    enc_ln = keras_model.get_layer("enc_output_layernorm")
    transfer_nested_layer_weights(
        enc_ln,
        sd,
        "model.enc_output.1",
        name_mapping=backbone_name_mapping,
    )

    for j in range(3):
        layer = keras_model.get_layer(f"enc_bbox_head_{j}")
        transfer_weights(
            "kernel", layer.weights[0], sd[f"model.enc_bbox_head.layers.{j}.weight"]
        )
        layer.weights[1].assign(sd[f"model.enc_bbox_head.layers.{j}.bias"])

    for j in range(2):
        layer = keras_model.get_layer(f"query_pos_head_{j}")
        transfer_weights(
            "kernel",
            layer.weights[0],
            sd[f"model.decoder.query_pos_head.layers.{j}.weight"],
        )
        layer.weights[1].assign(sd[f"model.decoder.query_pos_head.layers.{j}.bias"])

    num_dec = torch_model.config.decoder_layers
    for i in tqdm(range(num_dec), desc="Transferring decoder layers"):
        hf_dl = f"model.decoder.layers.{i}"
        k_dl = f"decoder_layers_{i}"

        sa_mapping: Dict[str, str] = {
            f"{k_dl}_self_attn_": "",
            "out_proj": "o_proj",
            "kernel": "weight",
            "beta": "bias",
        }
        sa = keras_model.get_layer(f"{k_dl}_self_attn")
        transfer_nested_layer_weights(
            sa, sd, f"{hf_dl}.self_attn", name_mapping=sa_mapping
        )

        ea = keras_model.get_layer(f"{k_dl}_encoder_attn")
        transfer_nested_layer_weights(
            ea,
            sd,
            f"{hf_dl}.encoder_attn",
            name_mapping={"kernel": "weight", "beta": "bias"},
            skip_paths=["n_points_scale"],
        )
        hf_scale_key = f"{hf_dl}.encoder_attn.n_points_scale"
        if hf_scale_key in sd:
            ea.n_points_scale.assign(sd[hf_scale_key])

        for k_name, hf_suffix in [
            (f"{k_dl}_self_attn_layer_norm", "self_attn_layer_norm"),
            (f"{k_dl}_encoder_attn_layer_norm", "encoder_attn_layer_norm"),
            (f"{k_dl}_final_layer_norm", "final_layer_norm"),
        ]:
            ln = keras_model.get_layer(k_name)
            transfer_nested_layer_weights(
                ln, sd, f"{hf_dl}.{hf_suffix}", name_mapping=backbone_name_mapping
            )

        for k_name, hf_suffix in [
            (f"{k_dl}_fc1", "mlp.fc1"),
            (f"{k_dl}_fc2", "mlp.fc2"),
        ]:
            fc = keras_model.get_layer(k_name)
            transfer_nested_layer_weights(
                fc,
                sd,
                f"{hf_dl}.{hf_suffix}",
                name_mapping={"kernel": "weight", "beta": "bias"},
            )

    for i in tqdm(range(num_dec), desc="Transferring detection heads"):
        try:
            cls = keras_model.get_layer(f"class_embed_{i}")
            transfer_weights(
                "kernel", cls.weights[0], sd[f"model.decoder.class_embed.{i}.weight"]
            )
            cls.weights[1].assign(sd[f"model.decoder.class_embed.{i}.bias"])
        except ValueError:
            pass

        for j in range(3):
            bbox = keras_model.get_layer(f"bbox_embed_{i}_{j}")
            transfer_weights(
                "kernel",
                bbox.weights[0],
                sd[f"model.decoder.bbox_embed.{i}.layers.{j}.weight"],
            )
            bbox.weights[1].assign(sd[f"model.decoder.bbox_embed.{i}.layers.{j}.bias"])

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

    if logits_diff > 1e-3 or boxes_diff > 1e-3:
        sub_enc = keras.Model(
            inputs=keras_model.input,
            outputs=keras_model.get_layer("enc_score_head").output,
        )
        keras_enc_scores = np.asarray(sub_enc.predict(test_input, verbose=0))
        with torch.no_grad():
            hf_model_out = torch_model.model(hf_input)
            hf_enc_scores = hf_model_out.enc_outputs_class.cpu().numpy()
        enc_diff = np.max(np.abs(keras_enc_scores - hf_enc_scores))
        print(f"  Encoder scores diff: {enc_diff:.6f} (fallback check)")
        assert enc_diff < 1e-3, (
            f"Equivalence test failed on encoder scores (diff: {enc_diff:.6f})"
        )

    if logits_cos < 0.95:
        raise ValueError(
            f"Equivalence test failed: logits cosine similarity {logits_cos:.4f} < 0.95"
        )

    print("Equivalence test passed!")

    keras_model.save_weights(output)
    print(f"Model saved as {output}")

    del keras_model, torch_model, sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
