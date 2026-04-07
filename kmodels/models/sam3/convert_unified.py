"""Convert HuggingFace facebook/sam3 to a single unified Keras weight file.

Goes directly from HF checkpoint to sam3_unified.weights.h5 (~3.3 GB).
Sam3VideoModel is the canonical format (matches HF's checkpoint structure).

Usage:
    HF_TOKEN=xxx python -m kmodels.models.sam3.convert_unified

Then load:
    sam3 = Sam3(weights="pcs")  # auto-downloads unified file
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT = "sam3_unified.weights.h5"


def _tc(keras_conv, w, b):
    transfer_weights("conv_kernel", keras_conv.kernel, w)
    keras_conv.bias.assign(b)


def _tt(keras_conv, w, b):
    transfer_weights("conv_transpose_kernel", keras_conv.kernel, w)
    keras_conv.bias.assign(b)


def _td(keras_dense, w, b):
    transfer_weights("kernel", keras_dense.kernel, w)
    keras_dense.bias.assign(b)


def _tl(keras_ln, w, b):
    keras_ln.gamma.assign(w)
    keras_ln.beta.assign(b)


def _tff(keras_ff, hf, p):
    _td(keras_ff.proj_in, hf[f"{p}.proj_in.weight"], hf[f"{p}.proj_in.bias"])
    for j, layer in enumerate(keras_ff.hidden_layers):
        _td(layer, hf[f"{p}.layers.{j}.weight"], hf[f"{p}.layers.{j}.bias"])
    _td(keras_ff.proj_out, hf[f"{p}.proj_out.weight"], hf[f"{p}.proj_out.bias"])


def _tfpn(keras_neck, hf, prefix):
    for i, sf in enumerate([4.0, 2.0, 1.0, 0.5]):
        fpn = keras_neck.fpn_layers[i]
        fp = f"{prefix}.fpn_layers.{i}"
        if sf == 4.0:
            _tt(
                fpn._deconv1,
                hf[f"{fp}.scale_layers.0.weight"],
                hf[f"{fp}.scale_layers.0.bias"],
            )
            _tt(
                fpn._deconv2,
                hf[f"{fp}.scale_layers.2.weight"],
                hf[f"{fp}.scale_layers.2.bias"],
            )
        elif sf == 2.0:
            _tt(
                fpn._deconv1,
                hf[f"{fp}.scale_layers.0.weight"],
                hf[f"{fp}.scale_layers.0.bias"],
            )
        _tc(fpn.proj1, hf[f"{fp}.proj1.weight"], hf[f"{fp}.proj1.bias"])
        _tc(fpn.proj2, hf[f"{fp}.proj2.weight"], hf[f"{fp}.proj2.bias"])


def _transfer_detector(sam3_model, hf, prefix=""):
    """Transfer all detector weights from HF state dict."""
    p = f"{prefix}." if prefix else ""
    det = sam3_model.detector

    print("  ViT backbone...")
    patch_conv = det.get_layer("backbone_patch_embed")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf[f"{p}vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"],
    )
    pos_embed = det.get_layer("backbone_position_embedding")
    pos_embed.embeddings.assign(
        hf[f"{p}vision_encoder.backbone.embeddings.position_embeddings"].squeeze(0)
    )

    for i in tqdm(range(det.vit_num_hidden_layers), desc="  ViT layers"):
        layer = det.get_layer(f"backbone_layers_{i}")
        hp = f"{p}vision_encoder.backbone.layers.{i}"
        _tl(
            layer.layer_norm1,
            hf[f"{hp}.layer_norm1.weight"],
            hf[f"{hp}.layer_norm1.bias"],
        )
        _tl(
            layer.layer_norm2,
            hf[f"{hp}.layer_norm2.weight"],
            hf[f"{hp}.layer_norm2.bias"],
        )
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            _td(
                getattr(layer.attn, proj),
                hf[f"{hp}.attention.{proj}.weight"],
                hf[f"{hp}.attention.{proj}.bias"],
            )
        _td(layer.mlp_fc1, hf[f"{hp}.mlp.fc1.weight"], hf[f"{hp}.mlp.fc1.bias"])
        _td(layer.mlp_fc2, hf[f"{hp}.mlp.fc2.weight"], hf[f"{hp}.mlp.fc2.bias"])

    bb_ln = det.get_layer("backbone_layer_norm")
    _tl(
        bb_ln,
        hf[f"{p}vision_encoder.backbone.layer_norm.weight"],
        hf[f"{p}vision_encoder.backbone.layer_norm.bias"],
    )

    print("  Detector FPN...")
    for idx, sf in enumerate(det.fpn_scale_factors):
        fp = f"{p}vision_encoder.neck.fpn_layers.{idx}"
        if sf == 4.0:
            _tc(
                det.get_layer(f"fpn_level_{idx}_deconv1"),
                hf[f"{fp}.scale_layers.0.weight"],
                hf[f"{fp}.scale_layers.0.bias"],
            )
            _tc(
                det.get_layer(f"fpn_level_{idx}_deconv2"),
                hf[f"{fp}.scale_layers.2.weight"],
                hf[f"{fp}.scale_layers.2.bias"],
            )
        elif sf == 2.0:
            _tc(
                det.get_layer(f"fpn_level_{idx}_deconv1"),
                hf[f"{fp}.scale_layers.0.weight"],
                hf[f"{fp}.scale_layers.0.bias"],
            )
        for pn in ["proj1", "proj2"]:
            if f"{fp}.{pn}.weight" in hf:
                _tc(
                    det.get_layer(f"fpn_level_{idx}_{pn}"),
                    hf[f"{fp}.{pn}.weight"],
                    hf[f"{fp}.{pn}.bias"],
                )

    print("  DETR encoder...")
    _td(
        det.get_layer("text_projection"),
        hf[f"{p}text_projection.weight"],
        hf[f"{p}text_projection.bias"],
    )
    for i in tqdm(range(det.detr_encoder_num_layers), desc="  Encoder layers"):
        ep = f"{p}detr_encoder.layers.{i}"
        kp = f"detr_encoder_layers_{i}"
        for an in ["self_attn", "cross_attn"]:
            attn = det.get_layer(f"{kp}_{an}")
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                _td(
                    getattr(attn, proj),
                    hf[f"{ep}.{an}.{proj}.weight"],
                    hf[f"{ep}.{an}.{proj}.bias"],
                )
        for ln in ["layer_norm1", "layer_norm2", "layer_norm3"]:
            _tl(
                det.get_layer(f"{kp}_{ln}"),
                hf[f"{ep}.{ln}.weight"],
                hf[f"{ep}.{ln}.bias"],
            )
        _td(
            det.get_layer(f"{kp}_fc1"),
            hf[f"{ep}.mlp.fc1.weight"],
            hf[f"{ep}.mlp.fc1.bias"],
        )
        _td(
            det.get_layer(f"{kp}_fc2"),
            hf[f"{ep}.mlp.fc2.weight"],
            hf[f"{ep}.mlp.fc2.bias"],
        )

    print("  DETR decoder...")
    det.get_layer("detr_decoder_query_embed").embeddings.assign(
        hf[f"{p}detr_decoder.query_embed.weight"]
    )
    det.get_layer("detr_decoder_reference_points").embeddings.assign(
        hf[f"{p}detr_decoder.reference_points.weight"]
    )
    det.get_layer("detr_decoder_presence_token").embeddings.assign(
        hf[f"{p}detr_decoder.presence_token.weight"]
    )
    for head in ["box_head", "presence_head", "ref_point_head"]:
        h = det.get_layer(f"detr_decoder_{head}")
        for j, d in enumerate(h.dense_layers):
            _td(
                d,
                hf[f"{p}detr_decoder.{head}.layer{j + 1}.weight"],
                hf[f"{p}detr_decoder.{head}.layer{j + 1}.bias"],
            )

    dec_ln = {
        "layer_norm1": "self_attn_layer_norm",
        "layer_norm2": "text_cross_attn_layer_norm",
        "layer_norm3": "vision_cross_attn_layer_norm",
        "layer_norm4": "mlp_layer_norm",
    }
    for i in tqdm(range(det.detr_decoder_num_layers), desc="  Decoder layers"):
        dp = f"{p}detr_decoder.layers.{i}"
        kp = f"detr_decoder_layers_{i}"
        for an in ["self_attn", "text_cross_attn", "vision_cross_attn"]:
            attn = det.get_layer(f"{kp}_{an}")
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                _td(
                    getattr(attn, proj),
                    hf[f"{dp}.{an}.{proj}.weight"],
                    hf[f"{dp}.{an}.{proj}.bias"],
                )
        for kln, hln in dec_ln.items():
            _tl(
                det.get_layer(f"{kp}_{kln}"),
                hf[f"{dp}.{hln}.weight"],
                hf[f"{dp}.{hln}.bias"],
            )
        _td(
            det.get_layer(f"{kp}_fc1"),
            hf[f"{dp}.mlp.fc1.weight"],
            hf[f"{dp}.mlp.fc1.bias"],
        )
        _td(
            det.get_layer(f"{kp}_fc2"),
            hf[f"{dp}.mlp.fc2.weight"],
            hf[f"{dp}.mlp.fc2.bias"],
        )

    _tl(
        det.get_layer("detr_decoder_output_layer_norm"),
        hf[f"{p}detr_decoder.output_layer_norm.weight"],
        hf[f"{p}detr_decoder.output_layer_norm.bias"],
    )
    _tl(
        det.get_layer("detr_decoder_presence_layer_norm"),
        hf[f"{p}detr_decoder.presence_layer_norm.weight"],
        hf[f"{p}detr_decoder.presence_layer_norm.bias"],
    )

    rpb = det.get_layer("detr_decoder_box_rpb")
    for ax in ["x", "y"]:
        mlp = getattr(rpb, f"box_rpb_embed_{ax}")
        for j, d in enumerate(mlp.dense_layers):
            _td(
                d,
                hf[f"{p}detr_decoder.box_rpb_embed_{ax}.layer{j + 1}.weight"],
                hf[f"{p}detr_decoder.box_rpb_embed_{ax}.layer{j + 1}.bias"],
            )

    print("  Scoring + mask decoder...")
    sp = f"{p}dot_product_scoring"
    for n, hn in [
        ("text_mlp_fc1", "text_mlp.layer1"),
        ("text_mlp_fc2", "text_mlp.layer2"),
    ]:
        _td(
            det.get_layer(f"dot_product_scoring_{n}"),
            hf[f"{sp}.{hn}.weight"],
            hf[f"{sp}.{hn}.bias"],
        )
    _tl(
        det.get_layer("dot_product_scoring_text_mlp_out_norm"),
        hf[f"{sp}.text_mlp_out_norm.weight"],
        hf[f"{sp}.text_mlp_out_norm.bias"],
    )
    for n in ["text_proj", "query_proj"]:
        _td(
            det.get_layer(f"dot_product_scoring_{n}"),
            hf[f"{sp}.{n}.weight"],
            hf[f"{sp}.{n}.bias"],
        )

    for s in range(len(det.fpn_scale_factors) - 2):
        _tc(
            det.get_layer(f"pixel_decoder_stage_{s}_conv"),
            hf[f"{p}mask_decoder.pixel_decoder.conv_layers.{s}.weight"],
            hf[f"{p}mask_decoder.pixel_decoder.conv_layers.{s}.bias"],
        )
        gn = det.get_layer(f"pixel_decoder_stage_{s}_gn")
        gn.gamma.assign(hf[f"{p}mask_decoder.pixel_decoder.norms.{s}.weight"])
        gn.beta.assign(hf[f"{p}mask_decoder.pixel_decoder.norms.{s}.bias"])
    _tc(
        det.get_layer("mask_decoder_instance_proj"),
        hf[f"{p}mask_decoder.instance_projection.weight"],
        hf[f"{p}mask_decoder.instance_projection.bias"],
    )
    _tc(
        det.get_layer("mask_decoder_semantic_proj"),
        hf[f"{p}mask_decoder.semantic_projection.weight"],
        hf[f"{p}mask_decoder.semantic_projection.bias"],
    )
    for j in range(3):
        _td(
            det.get_layer(f"mask_embedder_linear{j + 1}"),
            hf[f"{p}mask_decoder.mask_embedder.layers.{j}.weight"],
            hf[f"{p}mask_decoder.mask_embedder.layers.{j}.bias"],
        )
    pca = det.get_layer("mask_decoder_prompt_cross_attn")
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        _td(
            getattr(pca, proj),
            hf[f"{p}mask_decoder.prompt_cross_attn.{proj}.weight"],
            hf[f"{p}mask_decoder.prompt_cross_attn.{proj}.bias"],
        )
    _tl(
        det.get_layer("mask_decoder_prompt_cross_attn_norm"),
        hf[f"{p}mask_decoder.prompt_cross_attn_norm.weight"],
        hf[f"{p}mask_decoder.prompt_cross_attn_norm.bias"],
    )

    print("  CLIP text encoder...")
    te = sam3_model.text_encoder
    te.token_embedding.weights[0].assign(
        hf[f"{p}text_encoder.text_model.embeddings.token_embedding.weight"]
    )
    te.position_embedding.weights[0].assign(
        hf[f"{p}text_encoder.text_model.embeddings.position_embedding.weight"]
    )
    for i in tqdm(range(te.num_hidden_layers), desc="  CLIP layers"):
        layer = te.encoder_layers[i]
        hp = f"{p}text_encoder.text_model.encoder.layers.{i}"
        _tl(
            layer.layer_norm1,
            hf[f"{hp}.layer_norm1.weight"],
            hf[f"{hp}.layer_norm1.bias"],
        )
        _tl(
            layer.layer_norm2,
            hf[f"{hp}.layer_norm2.weight"],
            hf[f"{hp}.layer_norm2.bias"],
        )
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            hf_proj = "out_proj" if proj == "o_proj" else proj
            _td(
                getattr(layer.self_attn, proj),
                hf[f"{hp}.self_attn.{hf_proj}.weight"],
                hf[f"{hp}.self_attn.{hf_proj}.bias"],
            )
        _td(layer.fc1, hf[f"{hp}.mlp.fc1.weight"], hf[f"{hp}.mlp.fc1.bias"])
        _td(layer.fc2, hf[f"{hp}.mlp.fc2.weight"], hf[f"{hp}.mlp.fc2.bias"])
    _tl(
        te.final_layer_norm,
        hf[f"{p}text_encoder.text_model.final_layer_norm.weight"],
        hf[f"{p}text_encoder.text_model.final_layer_norm.bias"],
    )

    print("  Geometry encoder...")
    geo = sam3_model.geometry_encoder
    _td(
        geo.boxes_direct_project,
        hf[f"{p}geometry_encoder.boxes_direct_project.weight"],
        hf[f"{p}geometry_encoder.boxes_direct_project.bias"],
    )
    _td(
        geo.boxes_pos_enc_project,
        hf[f"{p}geometry_encoder.boxes_pos_enc_project.weight"],
        hf[f"{p}geometry_encoder.boxes_pos_enc_project.bias"],
    )
    _tc(
        geo.boxes_pool_project,
        hf[f"{p}geometry_encoder.boxes_pool_project.weight"],
        hf[f"{p}geometry_encoder.boxes_pool_project.bias"],
    )
    geo.label_embed.weights[0].assign(hf[f"{p}geometry_encoder.label_embed.weight"])
    geo.cls_embed.weights[0].assign(hf[f"{p}geometry_encoder.cls_embed.weight"])
    _tl(
        geo.vision_layer_norm,
        hf[f"{p}geometry_encoder.vision_layer_norm.weight"],
        hf[f"{p}geometry_encoder.vision_layer_norm.bias"],
    )
    _tl(
        geo.prompt_layer_norm,
        hf[f"{p}geometry_encoder.prompt_layer_norm.weight"],
        hf[f"{p}geometry_encoder.prompt_layer_norm.bias"],
    )
    _tl(
        geo.output_layer_norm,
        hf[f"{p}geometry_encoder.output_layer_norm.weight"],
        hf[f"{p}geometry_encoder.output_layer_norm.bias"],
    )
    _td(
        geo.final_proj,
        hf[f"{p}geometry_encoder.final_proj.weight"],
        hf[f"{p}geometry_encoder.final_proj.bias"],
    )
    for i in range(len(geo.transformer_layers)):
        layer = geo.transformer_layers[i]
        gp = f"{p}geometry_encoder.layers.{i}"
        for an in ["self_attn", "cross_attn"]:
            attn = getattr(layer, an)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                _td(
                    getattr(attn, proj),
                    hf[f"{gp}.{an}.{proj}.weight"],
                    hf[f"{gp}.{an}.{proj}.bias"],
                )
        _tl(
            layer.layer_norm1,
            hf[f"{gp}.layer_norm1.weight"],
            hf[f"{gp}.layer_norm1.bias"],
        )
        _tl(
            layer.layer_norm2,
            hf[f"{gp}.layer_norm2.weight"],
            hf[f"{gp}.layer_norm2.bias"],
        )
        _tl(
            layer.layer_norm3,
            hf[f"{gp}.layer_norm3.weight"],
            hf[f"{gp}.layer_norm3.bias"],
        )
        _td(layer.fc1, hf[f"{gp}.mlp.fc1.weight"], hf[f"{gp}.mlp.fc1.bias"])
        _td(layer.fc2, hf[f"{gp}.mlp.fc2.weight"], hf[f"{gp}.mlp.fc2.bias"])


def _transfer_tracker(tv, hf, prefix=""):
    """Transfer all tracker-video weights from HF state dict."""
    p = f"{prefix}." if prefix else ""

    tv.shared_image_embedding.positional_embedding.assign(
        hf[f"{p}shared_image_embedding.positional_embedding"]
    )
    pe = tv.prompt_encoder
    pe.shared_embedding.positional_embedding.assign(
        hf[f"{p}prompt_encoder.shared_embedding.positional_embedding"]
    )
    pe.point_embed.weights[0].assign(hf[f"{p}prompt_encoder.point_embed.weight"])
    pe.not_a_point_embed.weights[0].assign(
        hf[f"{p}prompt_encoder.not_a_point_embed.weight"]
    )
    pe.no_mask_embed.weights[0].assign(hf[f"{p}prompt_encoder.no_mask_embed.weight"])
    me = pe.mask_embed
    for n in ["conv1", "conv2", "conv3"]:
        _tc(
            getattr(me, n),
            hf[f"{p}prompt_encoder.mask_embed.{n}.weight"],
            hf[f"{p}prompt_encoder.mask_embed.{n}.bias"],
        )
    _tl(
        me.layer_norm1,
        hf[f"{p}prompt_encoder.mask_embed.layer_norm1.weight"],
        hf[f"{p}prompt_encoder.mask_embed.layer_norm1.bias"],
    )
    _tl(
        me.layer_norm2,
        hf[f"{p}prompt_encoder.mask_embed.layer_norm2.weight"],
        hf[f"{p}prompt_encoder.mask_embed.layer_norm2.bias"],
    )

    md = tv.mask_decoder
    md.iou_token.weights[0].assign(hf[f"{p}mask_decoder.iou_token.weight"])
    md.mask_tokens.weights[0].assign(hf[f"{p}mask_decoder.mask_tokens.weight"])
    md.obj_score_token.weights[0].assign(hf[f"{p}mask_decoder.obj_score_token.weight"])
    _tc(
        md.upscale_conv1,
        hf[f"{p}mask_decoder.upscale_conv1.weight"],
        hf[f"{p}mask_decoder.upscale_conv1.bias"],
    )
    _tc(
        md.upscale_conv2,
        hf[f"{p}mask_decoder.upscale_conv2.weight"],
        hf[f"{p}mask_decoder.upscale_conv2.bias"],
    )
    _tl(
        md.upscale_layer_norm,
        hf[f"{p}mask_decoder.upscale_layer_norm.weight"],
        hf[f"{p}mask_decoder.upscale_layer_norm.bias"],
    )
    _tc(
        md.conv_s0,
        hf[f"{p}mask_decoder.conv_s0.weight"],
        hf[f"{p}mask_decoder.conv_s0.bias"],
    )
    _tc(
        md.conv_s1,
        hf[f"{p}mask_decoder.conv_s1.weight"],
        hf[f"{p}mask_decoder.conv_s1.bias"],
    )
    for i in range(4):
        _tff(
            md.output_hypernetworks_mlps[i],
            hf,
            f"{p}mask_decoder.output_hypernetworks_mlps.{i}",
        )
    _tff(md.iou_prediction_head, hf, f"{p}mask_decoder.iou_prediction_head")
    _tff(md.pred_obj_score_head, hf, f"{p}mask_decoder.pred_obj_score_head")

    tw = md.transformer
    for i in range(2):
        block = tw.transformer_layers[i]
        bp = f"{p}mask_decoder.transformer.layers.{i}"
        for an in [
            "self_attn",
            "cross_attn_token_to_image",
            "cross_attn_image_to_token",
        ]:
            attn = getattr(block, an)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                _td(
                    getattr(attn, proj),
                    hf[f"{bp}.{an}.{proj}.weight"],
                    hf[f"{bp}.{an}.{proj}.bias"],
                )
        for ln in ["layer_norm1", "layer_norm2", "layer_norm3", "layer_norm4"]:
            _tl(getattr(block, ln), hf[f"{bp}.{ln}.weight"], hf[f"{bp}.{ln}.bias"])
        _td(
            block.mlp.proj_in,
            hf[f"{bp}.mlp.proj_in.weight"],
            hf[f"{bp}.mlp.proj_in.bias"],
        )
        _td(
            block.mlp.proj_out,
            hf[f"{bp}.mlp.proj_out.weight"],
            hf[f"{bp}.mlp.proj_out.bias"],
        )

    fa = tw.final_attn_token_to_image
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        _td(
            getattr(fa, proj),
            hf[f"{p}mask_decoder.transformer.final_attn_token_to_image.{proj}.weight"],
            hf[f"{p}mask_decoder.transformer.final_attn_token_to_image.{proj}.bias"],
        )
    _tl(
        tw.layer_norm_final_attn,
        hf[f"{p}mask_decoder.transformer.layer_norm_final_attn.weight"],
        hf[f"{p}mask_decoder.transformer.layer_norm_final_attn.bias"],
    )

    ma = tv.memory_attention
    for i in range(4):
        layer = ma.attention_layers[i]
        lp = f"{p}memory_attention.layers.{i}"
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            _td(
                getattr(layer.self_attn, proj),
                hf[f"{lp}.self_attn.{proj}.weight"],
                hf[f"{lp}.self_attn.{proj}.bias"],
            )
            _td(
                getattr(layer.cross_attn_image, proj),
                hf[f"{lp}.cross_attn_image.{proj}.weight"],
                hf[f"{lp}.cross_attn_image.{proj}.bias"],
            )
        _td(layer.linear1, hf[f"{lp}.linear1.weight"], hf[f"{lp}.linear1.bias"])
        _td(layer.linear2, hf[f"{lp}.linear2.weight"], hf[f"{lp}.linear2.bias"])
        for ln in ["layer_norm1", "layer_norm2", "layer_norm3"]:
            _tl(getattr(layer, ln), hf[f"{lp}.{ln}.weight"], hf[f"{lp}.{ln}.bias"])
    _tl(
        ma.layer_norm,
        hf[f"{p}memory_attention.layer_norm.weight"],
        hf[f"{p}memory_attention.layer_norm.bias"],
    )

    mem = tv.memory_encoder
    mds = mem.mask_downsampler
    for i in range(len(mds.downsample_layers)):
        layer = mds.downsample_layers[i]
        _tc(
            layer.conv,
            hf[f"{p}memory_encoder.mask_downsampler.layers.{i}.conv.weight"],
            hf[f"{p}memory_encoder.mask_downsampler.layers.{i}.conv.bias"],
        )
        _tl(
            layer.layer_norm,
            hf[f"{p}memory_encoder.mask_downsampler.layers.{i}.layer_norm.weight"],
            hf[f"{p}memory_encoder.mask_downsampler.layers.{i}.layer_norm.bias"],
        )
    _tc(
        mds.final_conv,
        hf[f"{p}memory_encoder.mask_downsampler.final_conv.weight"],
        hf[f"{p}memory_encoder.mask_downsampler.final_conv.bias"],
    )
    _tc(
        mem.feature_projection,
        hf[f"{p}memory_encoder.feature_projection.weight"],
        hf[f"{p}memory_encoder.feature_projection.bias"],
    )
    fuser = mem.memory_fuser
    for i in range(len(fuser.fuser_layers)):
        block = fuser.fuser_layers[i]
        bp = f"{p}memory_encoder.memory_fuser.layers.{i}"
        transfer_weights(
            "depthwise_conv_kernel",
            block.depthwise_conv.kernel,
            hf[f"{bp}.depthwise_conv.weight"],
        )
        block.depthwise_conv.bias.assign(hf[f"{bp}.depthwise_conv.bias"])
        _tl(
            block.layer_norm, hf[f"{bp}.layer_norm.weight"], hf[f"{bp}.layer_norm.bias"]
        )
        _td(
            block.pointwise_conv1,
            hf[f"{bp}.pointwise_conv1.weight"],
            hf[f"{bp}.pointwise_conv1.bias"],
        )
        _td(
            block.pointwise_conv2,
            hf[f"{bp}.pointwise_conv2.weight"],
            hf[f"{bp}.pointwise_conv2.bias"],
        )
        block.scale.assign(hf[f"{bp}.scale"])
    _tc(
        mem.projection,
        hf[f"{p}memory_encoder.projection.weight"],
        hf[f"{p}memory_encoder.projection.bias"],
    )

    _tff(tv.object_pointer_proj, hf, f"{p}object_pointer_proj")
    _tc(
        tv.mask_downsample,
        hf[f"{p}mask_downsample.weight"],
        hf[f"{p}mask_downsample.bias"],
    )
    tv.no_memory_embedding.assign(hf[f"{p}no_memory_embedding"])
    tv.no_memory_positional_encoding.assign(hf[f"{p}no_memory_positional_encoding"])
    tv.no_object_pointer.assign(hf[f"{p}no_object_pointer"])
    tv.memory_temporal_positional_encoding.assign(
        hf[f"{p}memory_temporal_positional_encoding"]
    )
    if hasattr(tv, "temporal_positional_encoding_projection_layer"):
        _td(
            tv.temporal_positional_encoding_projection_layer,
            hf[f"{p}temporal_positional_encoding_projection_layer.weight"],
            hf[f"{p}temporal_positional_encoding_projection_layer.bias"],
        )
    if hasattr(tv, "occlusion_spatial_embedding_parameter"):
        tv.occlusion_spatial_embedding_parameter.assign(
            hf[f"{p}occlusion_spatial_embedding_parameter"]
        )

    _tfpn(tv.vision_neck, hf, f"{p}vision_encoder.neck")


def convert():
    from transformers.models.sam3_video.modeling_sam3_video import (
        Sam3VideoModel as HFVideoModel,
    )

    print("Loading HF Sam3VideoModel (canonical checkpoint)...")
    hf_model = HFVideoModel.from_pretrained(
        "facebook/sam3", attn_implementation="eager", token=HF_TOKEN
    ).eval()
    hf = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
    print(f"HF: {len(hf)} keys")

    print("\nBuilding Keras models...")
    from kmodels.models.sam3.sam3_model import Sam3
    from kmodels.models.sam3_tracker_video.sam3_tracker_video_model import (
        Sam3TrackerVideo,
    )
    from kmodels.models.sam3_video.sam3_video_model import Sam3Video

    sam3 = Sam3(input_shape=(1008, 1008, 3), weights=None)
    tv = Sam3TrackerVideo(sam3_model=sam3, weights=None)
    vm = Sam3Video(sam3_model=sam3, tracker_video_model=tv, weights=None)

    print("\nTransferring detector weights (detector_model.*)...")
    _transfer_detector(sam3, hf, prefix="detector_model")

    print("\nTransferring tracker weights (tracker_model.* + vision_encoder.neck.*)...")
    tracker_hf = {}
    for k, v in hf.items():
        if k.startswith("tracker_model."):
            tracker_hf[k[len("tracker_model.") :]] = v
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoModel as HFTVModel,
    )

    hf_tv_full = HFTVModel.from_pretrained(
        "facebook/sam3", attn_implementation="eager", token=HF_TOKEN
    ).eval()
    for k, v in hf_tv_full.state_dict().items():
        if k.startswith("vision_encoder.neck."):
            tracker_hf[k] = v.cpu().numpy()
    del hf_tv_full

    _transfer_tracker(tv, tracker_hf)

    print("\nTransferring video neck (tracker_neck.*)...")
    _tfpn(vm.tracker_neck, hf, "tracker_neck")

    print(f"\nSaving {OUTPUT}...")
    vm_params = sum(w.numpy().size for w in vm.weights)
    print(f"  Total params: {vm_params:,}")
    vm.save_weights(OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"  Saved: {OUTPUT} ({size_mb:.0f} MB)")

    print("\nVerifying...")
    sam3_v = Sam3(input_shape=(1008, 1008, 3), weights=None)
    tv_v = Sam3TrackerVideo(sam3_model=sam3_v, weights=None)
    vm_v = Sam3Video(sam3_model=sam3_v, tracker_video_model=tv_v, weights=None)
    vm_v.load_weights(OUTPUT)

    from keras import ops
    from PIL import Image
    from transformers.models.sam3_tracker_video.processing_sam3_tracker_video import (
        Sam3TrackerVideoProcessor,
    )

    hf_proc = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3", token=HF_TOKEN)
    img = Image.open("real_video_frames/00000.jpg").convert("RGB")
    inputs = hf_proc(
        images=img,
        input_points=[[[[480, 260]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )

    with torch.no_grad():
        out = tv_v(
            pixel_values=inputs["pixel_values"].permute(0, 2, 3, 1).numpy(),
            input_points=inputs["input_points"].numpy(),
            input_labels=inputs["input_labels"].numpy().astype(np.int32),
            multimask_output=False,
        )
    obj = ops.convert_to_numpy(out["object_score_logits"]).flatten()[0]
    print(f"  obj_score: {obj:.3f} (expected: 18.089)")
    print(f"  PASS: {abs(obj - 18.089) < 0.01}")
    print("\nDone!")


if __name__ == "__main__":
    convert()
