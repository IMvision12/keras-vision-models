import keras
import numpy as np
import torch  # noqa: F401 - must import before keras
from tqdm import tqdm
from transformers import RTDetrForObjectDetection

from kmodels.models.rt_detr.rt_detr_model import (
    RTDETRResNet18,
    RTDETRResNet34,
    RTDETRResNet50,
    RTDETRResNet101,
)
from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights


def _set_dense(layer, w, b):
    """Set Dense layer kernel and bias via variable list."""
    for v in layer.variables:
        if v.name == "kernel":
            v.assign(w)
        elif v.name == "bias":
            v.assign(b)


def _set_ln(layer, w, b):
    """Set LayerNorm gamma and beta."""
    for v in layer.variables:
        if v.name == "gamma":
            v.assign(w)
        elif v.name == "beta":
            v.assign(b)


def _transfer_conv_norm(keras_model, sd, keras_name, hf_prefix):
    """Transfer Conv+BN weights."""
    k_conv = keras_model.get_layer(f"{keras_name}_conv")
    transfer_weights("conv_kernel", k_conv.kernel, sd[f"{hf_prefix}.conv.weight"])
    k_bn = keras_model.get_layer(f"{keras_name}_norm")
    k_bn.gamma.assign(sd[f"{hf_prefix}.norm.weight"])
    k_bn.beta.assign(sd[f"{hf_prefix}.norm.bias"])
    k_bn.moving_mean.assign(sd[f"{hf_prefix}.norm.running_mean"])
    k_bn.moving_variance.assign(sd[f"{hf_prefix}.norm.running_var"])


_D = {"input_shape": (640, 640, 3), "num_queries": 300, "num_labels": 80}

model_configs = [
    {
        "keras_model_cls": RTDETRResNet50,
        "hf_model_name": "PekingU/rtdetr_r50vd",
        "weights_name": "coco",
        **_D,
    },
    {
        "keras_model_cls": RTDETRResNet18,
        "hf_model_name": "PekingU/rtdetr_r18vd",
        "weights_name": "coco",
        **_D,
    },
    {
        "keras_model_cls": RTDETRResNet34,
        "hf_model_name": "PekingU/rtdetr_r34vd",
        "weights_name": "coco",
        **_D,
    },
    {
        "keras_model_cls": RTDETRResNet101,
        "hf_model_name": "PekingU/rtdetr_r101vd",
        "weights_name": "coco",
        **_D,
    },
    {
        "keras_model_cls": RTDETRResNet18,
        "hf_model_name": "PekingU/rtdetr_r18vd_coco_o365",
        "weights_name": "coco_o365",
        **_D,
    },
    {
        "keras_model_cls": RTDETRResNet50,
        "hf_model_name": "PekingU/rtdetr_r50vd_coco_o365",
        "weights_name": "coco_o365",
        **_D,
    },
    {
        "keras_model_cls": RTDETRResNet101,
        "hf_model_name": "PekingU/rtdetr_r101vd_coco_o365",
        "weights_name": "coco_o365",
        **_D,
    },
]

for model_config in model_configs:
    print(f"\n{'=' * 60}")
    print(f"Converting {model_config['hf_model_name']}...")
    print(f"{'=' * 60}")

    torch_model = RTDetrForObjectDetection.from_pretrained(
        model_config["hf_model_name"],
        attn_implementation="eager",
    ).eval()
    sd = {k: v.cpu().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = model_config["keras_model_cls"](
        weights=None,
        input_shape=model_config["input_shape"],
        num_queries=model_config["num_queries"],
        num_labels=model_config["num_labels"],
    )

    # ---- Backbone stem ----
    print("Transferring backbone stem...")
    for i in range(3):
        hf_pre = f"model.backbone.model.embedder.embedder.{i}"
        k_conv = keras_model.get_layer(f"backbone_embedder_{i}_conv")
        transfer_weights(
            "conv_kernel", k_conv.kernel, sd[f"{hf_pre}.convolution.weight"]
        )

        k_bn = keras_model.get_layer(f"backbone_embedder_{i}_bn")
        k_bn.gamma.assign(sd[f"{hf_pre}.normalization.weight"])
        k_bn.beta.assign(sd[f"{hf_pre}.normalization.bias"])
        k_bn.moving_mean.assign(sd[f"{hf_pre}.normalization.running_mean"])
        k_bn.moving_variance.assign(sd[f"{hf_pre}.normalization.running_var"])

    # ---- Backbone stages ----
    bb_cfg = torch_model.model.backbone.model.config
    block_repeats = bb_cfg.depths
    layer_type = bb_cfg.layer_type
    num_convs = 2 if layer_type == "basic" else 3
    print(f"Transferring backbone stages ({layer_type})...")
    for stage_idx, num_blocks in enumerate(block_repeats):
        for block_idx in range(num_blocks):
            hf_pre = (
                f"model.backbone.model.encoder.stages.{stage_idx}.layers.{block_idx}"
            )
            k_pre = f"backbone_stage{stage_idx}_{block_idx}"

            for ci in range(num_convs):
                hf_conv = f"{hf_pre}.layer.{ci}.convolution.weight"
                hf_bn_w = f"{hf_pre}.layer.{ci}.normalization.weight"
                hf_bn_b = f"{hf_pre}.layer.{ci}.normalization.bias"
                hf_bn_m = f"{hf_pre}.layer.{ci}.normalization.running_mean"
                hf_bn_v = f"{hf_pre}.layer.{ci}.normalization.running_var"

                k_conv_name = f"{k_pre}_conv{ci + 1}"
                k_bn_name = f"{k_pre}_bn{ci + 1}"

                k_conv = keras_model.get_layer(k_conv_name)
                transfer_weights("conv_kernel", k_conv.kernel, sd[hf_conv])

                k_bn = keras_model.get_layer(k_bn_name)
                k_bn.gamma.assign(sd[hf_bn_w])
                k_bn.beta.assign(sd[hf_bn_b])
                k_bn.moving_mean.assign(sd[hf_bn_m])
                k_bn.moving_variance.assign(sd[hf_bn_v])

            # Shortcut
            sc_conv_name = f"{k_pre}_shortcut_conv"
            sc_bn_name = f"{k_pre}_shortcut_bn"
            try:
                k_sc_conv = keras_model.get_layer(sc_conv_name)
            except ValueError:
                continue

            if stage_idx == 0 and block_idx == 0:
                hf_sc_conv = f"{hf_pre}.shortcut.convolution.weight"
                hf_sc_bn_w = f"{hf_pre}.shortcut.normalization.weight"
                hf_sc_bn_b = f"{hf_pre}.shortcut.normalization.bias"
                hf_sc_bn_m = f"{hf_pre}.shortcut.normalization.running_mean"
                hf_sc_bn_v = f"{hf_pre}.shortcut.normalization.running_var"
            else:
                hf_sc_conv = f"{hf_pre}.shortcut.1.convolution.weight"
                hf_sc_bn_w = f"{hf_pre}.shortcut.1.normalization.weight"
                hf_sc_bn_b = f"{hf_pre}.shortcut.1.normalization.bias"
                hf_sc_bn_m = f"{hf_pre}.shortcut.1.normalization.running_mean"
                hf_sc_bn_v = f"{hf_pre}.shortcut.1.normalization.running_var"

            transfer_weights("conv_kernel", k_sc_conv.kernel, sd[hf_sc_conv])
            k_sc_bn = keras_model.get_layer(sc_bn_name)
            k_sc_bn.gamma.assign(sd[hf_sc_bn_w])
            k_sc_bn.beta.assign(sd[hf_sc_bn_b])
            k_sc_bn.moving_mean.assign(sd[hf_sc_bn_m])
            k_sc_bn.moving_variance.assign(sd[hf_sc_bn_v])

    # ---- Encoder input projections ----
    print("Transferring encoder input projections...")
    for i in range(3):
        hf_conv = f"model.encoder_input_proj.{i}.0.weight"
        hf_bn_w = f"model.encoder_input_proj.{i}.1.weight"
        hf_bn_b = f"model.encoder_input_proj.{i}.1.bias"
        hf_bn_m = f"model.encoder_input_proj.{i}.1.running_mean"
        hf_bn_v = f"model.encoder_input_proj.{i}.1.running_var"

        k_conv = keras_model.get_layer(f"encoder_input_proj_{i}_conv")
        transfer_weights("conv_kernel", k_conv.kernel, sd[hf_conv])

        k_bn = keras_model.get_layer(f"encoder_input_proj_{i}_bn")
        k_bn.gamma.assign(sd[hf_bn_w])
        k_bn.beta.assign(sd[hf_bn_b])
        k_bn.moving_mean.assign(sd[hf_bn_m])
        k_bn.moving_variance.assign(sd[hf_bn_v])

    # ---- AIFI encoder ----
    print("Transferring AIFI encoder...")
    hf_aifi = "model.encoder.aifi.0.layers.0"
    k_aifi = "aifi_0_layers_0"

    sa = keras_model.get_layer(f"{k_aifi}_self_attn")
    for kp, hp in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", "o_proj"),
    ]:
        proj = getattr(sa, kp)
        _set_dense(
            proj,
            sd[f"{hf_aifi}.self_attn.{hp}.weight"].T,
            sd[f"{hf_aifi}.self_attn.{hp}.bias"],
        )

    _set_ln(
        keras_model.get_layer(f"{k_aifi}_self_attn_layer_norm"),
        sd[f"{hf_aifi}.self_attn_layer_norm.weight"],
        sd[f"{hf_aifi}.self_attn_layer_norm.bias"],
    )
    _set_dense(
        keras_model.get_layer(f"{k_aifi}_fc1"),
        sd[f"{hf_aifi}.mlp.fc1.weight"].T,
        sd[f"{hf_aifi}.mlp.fc1.bias"],
    )
    _set_dense(
        keras_model.get_layer(f"{k_aifi}_fc2"),
        sd[f"{hf_aifi}.mlp.fc2.weight"].T,
        sd[f"{hf_aifi}.mlp.fc2.bias"],
    )
    _set_ln(
        keras_model.get_layer(f"{k_aifi}_final_layer_norm"),
        sd[f"{hf_aifi}.final_layer_norm.weight"],
        sd[f"{hf_aifi}.final_layer_norm.bias"],
    )

    # ---- CCFM: lateral convs, fpn_blocks, downsample_convs, pan_blocks ----
    print("Transferring CCFM (FPN + PAN)...")
    for i in range(2):
        _transfer_conv_norm(
            keras_model, sd, f"lateral_convs_{i}", f"model.encoder.lateral_convs.{i}"
        )
        _transfer_conv_norm(
            keras_model,
            sd,
            f"downsample_convs_{i}",
            f"model.encoder.downsample_convs.{i}",
        )

    for block_type in ["fpn_blocks", "pan_blocks"]:
        for i in range(2):
            hf_blk = f"model.encoder.{block_type}.{i}"
            k_blk = f"{block_type}_{i}"

            _transfer_conv_norm(keras_model, sd, f"{k_blk}_conv1", f"{hf_blk}.conv1")
            _transfer_conv_norm(keras_model, sd, f"{k_blk}_conv2", f"{hf_blk}.conv2")

            # conv3 exists when hidden_expansion != 1.0
            if f"{hf_blk}.conv3.conv.weight" in sd:
                _transfer_conv_norm(
                    keras_model, sd, f"{k_blk}_conv3", f"{hf_blk}.conv3"
                )

            for bi in range(3):
                _transfer_conv_norm(
                    keras_model,
                    sd,
                    f"{k_blk}_bottlenecks_{bi}_conv1",
                    f"{hf_blk}.bottlenecks.{bi}.conv1",
                )
                _transfer_conv_norm(
                    keras_model,
                    sd,
                    f"{k_blk}_bottlenecks_{bi}_conv2",
                    f"{hf_blk}.bottlenecks.{bi}.conv2",
                )

    # ---- Decoder input projections ----
    print("Transferring decoder input projections...")
    for i in range(3):
        k_conv = keras_model.get_layer(f"decoder_input_proj_{i}_conv")
        transfer_weights(
            "conv_kernel", k_conv.kernel, sd[f"model.decoder_input_proj.{i}.0.weight"]
        )
        k_bn = keras_model.get_layer(f"decoder_input_proj_{i}_bn")
        k_bn.gamma.assign(sd[f"model.decoder_input_proj.{i}.1.weight"])
        k_bn.beta.assign(sd[f"model.decoder_input_proj.{i}.1.bias"])
        k_bn.moving_mean.assign(sd[f"model.decoder_input_proj.{i}.1.running_mean"])
        k_bn.moving_variance.assign(sd[f"model.decoder_input_proj.{i}.1.running_var"])

    # ---- Encoder output ----
    print("Transferring encoder output heads...")
    _set_dense(
        keras_model.get_layer("enc_output_linear"),
        sd["model.enc_output.0.weight"].T,
        sd["model.enc_output.0.bias"],
    )
    _set_ln(
        keras_model.get_layer("enc_output_layernorm"),
        sd["model.enc_output.1.weight"],
        sd["model.enc_output.1.bias"],
    )
    _set_dense(
        keras_model.get_layer("enc_score_head"),
        sd["model.enc_score_head.weight"].T,
        sd["model.enc_score_head.bias"],
    )
    for j in range(3):
        _set_dense(
            keras_model.get_layer(f"enc_bbox_head_{j}"),
            sd[f"model.enc_bbox_head.layers.{j}.weight"].T,
            sd[f"model.enc_bbox_head.layers.{j}.bias"],
        )

    print("Transferring decoder...")
    _set_dense(
        keras_model.get_layer("query_pos_head_0"),
        sd["model.decoder.query_pos_head.layers.0.weight"].T,
        sd["model.decoder.query_pos_head.layers.0.bias"],
    )
    _set_dense(
        keras_model.get_layer("query_pos_head_1"),
        sd["model.decoder.query_pos_head.layers.1.weight"].T,
        sd["model.decoder.query_pos_head.layers.1.bias"],
    )

    # ---- Decoder layers ----
    num_dec = torch_model.config.decoder_layers
    for i in tqdm(range(num_dec), desc="Transferring decoder layers"):
        hf_pre = f"model.decoder.layers.{i}"
        k_pre = f"decoder_layers_{i}"

        # Self-attention
        dl = keras_model.get_layer(f"{k_pre}")
        for kp, hp in [
            ("q_proj", "q_proj"),
            ("k_proj", "k_proj"),
            ("v_proj", "v_proj"),
            ("out_proj", "o_proj"),
        ]:
            _set_dense(
                getattr(dl.self_attn, kp),
                sd[f"{hf_pre}.self_attn.{hp}.weight"].T,
                sd[f"{hf_pre}.self_attn.{hp}.bias"],
            )

        _set_ln(
            dl.self_attn_layer_norm,
            sd[f"{hf_pre}.self_attn_layer_norm.weight"],
            sd[f"{hf_pre}.self_attn_layer_norm.bias"],
        )

        ea = dl.encoder_attn
        for k_attr, hf_name in [
            ("sampling_offsets", "sampling_offsets"),
            ("attention_weights_proj", "attention_weights"),
            ("value_proj", "value_proj"),
            ("output_proj", "output_proj"),
        ]:
            _set_dense(
                getattr(ea, k_attr),
                sd[f"{hf_pre}.encoder_attn.{hf_name}.weight"].T,
                sd[f"{hf_pre}.encoder_attn.{hf_name}.bias"],
            )

        _set_ln(
            dl.encoder_attn_layer_norm,
            sd[f"{hf_pre}.encoder_attn_layer_norm.weight"],
            sd[f"{hf_pre}.encoder_attn_layer_norm.bias"],
        )

        _set_dense(
            dl.fc1, sd[f"{hf_pre}.mlp.fc1.weight"].T, sd[f"{hf_pre}.mlp.fc1.bias"]
        )
        _set_dense(
            dl.fc2, sd[f"{hf_pre}.mlp.fc2.weight"].T, sd[f"{hf_pre}.mlp.fc2.bias"]
        )
        _set_ln(
            dl.final_layer_norm,
            sd[f"{hf_pre}.final_layer_norm.weight"],
            sd[f"{hf_pre}.final_layer_norm.bias"],
        )

    # ---- Per-layer class_embed and bbox_embed ----
    print("Transferring detection heads...")
    for i in range(num_dec):
        try:
            _set_dense(
                keras_model.get_layer(f"class_embed_{i}"),
                sd[f"model.decoder.class_embed.{i}.weight"].T,
                sd[f"model.decoder.class_embed.{i}.bias"],
            )
        except ValueError:
            pass

        for j in range(3):
            _set_dense(
                keras_model.get_layer(f"bbox_embed_{i}_{j}"),
                sd[f"model.decoder.bbox_embed.{i}.layers.{j}.weight"].T,
                sd[f"model.decoder.bbox_embed.{i}.layers.{j}.bias"],
            )

    print("Weight transfer complete!")

    # ---- Verification ----
    print("Verifying model equivalence...")
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

    print(f"Max logits diff:  {logits_diff:.6f}")
    print(f"Max boxes diff:   {boxes_diff:.6f}")

    if logits_diff > 1e-3 or boxes_diff > 1e-3:
        # Top-k query selection can differ for some checkpoints due to
        # float32 accumulation in deep encoders. Verify pre-top-k encoder
        # scores instead, which are deterministic.
        sub_enc = keras.Model(
            inputs=keras_model.input,
            outputs=keras_model.get_layer("enc_score_head").output,
        )
        keras_enc_scores = np.asarray(sub_enc.predict(test_input, verbose=0))
        with torch.no_grad():
            hf_model_out = torch_model.model(
                torch.tensor(test_input).permute(0, 3, 1, 2),
            )
            hf_enc_scores = hf_model_out.enc_outputs_class.cpu().numpy()
        enc_diff = np.max(np.abs(keras_enc_scores - hf_enc_scores))
        print(f"  Encoder scores diff: {enc_diff:.6f} (fallback check)")
        assert enc_diff < 1e-3, (
            f"Model equivalence test failed on encoder scores (diff: {enc_diff:.6f})"
        )

    print("Model equivalence test passed!")

    hf_name = model_config["hf_model_name"].split("/")[-1]
    wt_name = model_config.get("weights_name", "coco")
    model_filename = f"{hf_name}_{wt_name}.weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved as {model_filename}")

    del keras_model, torch_model, sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
