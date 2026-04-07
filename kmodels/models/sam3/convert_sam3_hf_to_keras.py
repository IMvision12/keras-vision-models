import os

os.environ["KERAS_BACKEND"] = "torch"

from tqdm import tqdm  # noqa: E402

from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT = "sam3.weights.h5"


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

    # Extract detector-only keys
    det_hf = {}
    for k, v in hf.items():
        if k.startswith("detector_model."):
            det_hf[k[len("detector_model.") :]] = v
    print(f"Detector keys: {len(det_hf)}")
    del hf_model, hf

    print("\nBuilding Keras Sam3...")
    from kmodels.models.sam3.sam3_model import Sam3

    sam3 = Sam3(input_shape=(1008, 1008, 3), weights=None)

    print("\nTransferring detector weights...")
    _transfer_detector(sam3, det_hf)

    print(f"\nSaving {OUTPUT}...")
    total_params = sum(w.numpy().size for w in sam3.weights)
    print(f"  Total params: {total_params:,}")
    sam3.save_weights(OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"  Saved: {OUTPUT} ({size_mb:.0f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    convert()
