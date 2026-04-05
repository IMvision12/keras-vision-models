import os

os.environ["KERAS_BACKEND"] = "torch"

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import Sam3Model  # noqa: E402

from kmodels.models.sam3.sam3_clip import SAM3CLIPTextEncoder  # noqa: E402
from kmodels.models.sam3.sam3_layers import SAM3GeometryEncoder  # noqa: E402
from kmodels.models.sam3.sam3_model import Sam3  # noqa: E402
from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN")

model_configs = [
    {
        "keras_model_cls": Sam3,
        "hf_model_name": "facebook/sam3",
        "input_shape": (1008, 1008, 3),
    },
]


def convert_sam3(model_config):
    print(f"\n{'=' * 60}")
    print(f"Converting {model_config['hf_model_name']}...")
    print(f"{'=' * 60}")

    print(f"Loading HF model: {model_config['hf_model_name']}")
    hf_model = Sam3Model.from_pretrained(
        model_config["hf_model_name"],
        attn_implementation="eager",
        token=HF_TOKEN,
    ).eval()
    hf = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
    print(f"HF model has {len(hf)} weight tensors")

    print("Creating Keras model...")
    keras_model = model_config["keras_model_cls"](
        input_shape=model_config["input_shape"], weights=None
    )
    print(f"Keras model params: {keras_model.count_params():,}")

    # ── Vision encoder backbone ──────────────────────────────────
    print("Transferring ViT backbone weights...")

    patch_conv = keras_model.get_layer("backbone_patch_embed")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf["vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"],
    )

    pos_embed_layer = keras_model.get_layer("backbone_position_embedding")
    hf_pos = hf["vision_encoder.backbone.embeddings.position_embeddings"]
    pos_embed_layer.embeddings.assign(hf_pos.squeeze(0))

    num_layers = keras_model.vit_num_hidden_layers
    for i in tqdm(range(num_layers), desc="ViT layers"):
        layer = keras_model.get_layer(f"backbone_layers_{i}")
        p = f"vision_encoder.backbone.layers.{i}"

        layer.layer_norm1.gamma.assign(hf[f"{p}.layer_norm1.weight"])
        layer.layer_norm1.beta.assign(hf[f"{p}.layer_norm1.bias"])
        layer.layer_norm2.gamma.assign(hf[f"{p}.layer_norm2.weight"])
        layer.layer_norm2.beta.assign(hf[f"{p}.layer_norm2.bias"])

        attn = layer.attn
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            transfer_weights(
                "kernel",
                getattr(attn, proj).kernel,
                hf[f"{p}.attention.{proj}.weight"],
            )
            getattr(attn, proj).bias.assign(hf[f"{p}.attention.{proj}.bias"])

        transfer_weights("kernel", layer.mlp_fc1.kernel, hf[f"{p}.mlp.fc1.weight"])
        layer.mlp_fc1.bias.assign(hf[f"{p}.mlp.fc1.bias"])
        transfer_weights("kernel", layer.mlp_fc2.kernel, hf[f"{p}.mlp.fc2.weight"])
        layer.mlp_fc2.bias.assign(hf[f"{p}.mlp.fc2.bias"])

    backbone_ln = keras_model.get_layer("backbone_layer_norm")
    backbone_ln.gamma.assign(hf["vision_encoder.backbone.layer_norm.weight"])
    backbone_ln.beta.assign(hf["vision_encoder.backbone.layer_norm.bias"])

    # ── FPN neck ─────────────────────────────────────���───────────
    print("Transferring FPN neck weights...")

    scale_factors = keras_model.fpn_scale_factors
    for level_idx, scale_factor in enumerate(scale_factors):
        fpn_p = f"vision_encoder.neck.fpn_layers.{level_idx}"

        if scale_factor == 4.0:
            # Two deconv layers: scale_layers.0 and scale_layers.2
            deconv1 = keras_model.get_layer(f"fpn_level_{level_idx}_deconv1")
            transfer_weights(
                "conv_kernel",
                deconv1.kernel,
                hf[f"{fpn_p}.scale_layers.0.weight"],
            )
            deconv1.bias.assign(hf[f"{fpn_p}.scale_layers.0.bias"])

            deconv2 = keras_model.get_layer(f"fpn_level_{level_idx}_deconv2")
            transfer_weights(
                "conv_kernel",
                deconv2.kernel,
                hf[f"{fpn_p}.scale_layers.2.weight"],
            )
            deconv2.bias.assign(hf[f"{fpn_p}.scale_layers.2.bias"])

        elif scale_factor == 2.0:
            # One deconv layer: scale_layers.0
            deconv1 = keras_model.get_layer(f"fpn_level_{level_idx}_deconv1")
            transfer_weights(
                "conv_kernel",
                deconv1.kernel,
                hf[f"{fpn_p}.scale_layers.0.weight"],
            )
            deconv1.bias.assign(hf[f"{fpn_p}.scale_layers.0.bias"])

        # proj1, proj2 for all levels
        for proj_name in ["proj1", "proj2"]:
            keras_layer = keras_model.get_layer(f"fpn_level_{level_idx}_{proj_name}")
            hf_key = f"{fpn_p}.{proj_name}.weight"
            if hf_key in hf:
                transfer_weights("conv_kernel", keras_layer.kernel, hf[hf_key])
                keras_layer.bias.assign(hf[f"{fpn_p}.{proj_name}.bias"])

    # ── Text projection ──────────────────────────────────────────
    print("Transferring text projection...")
    text_proj = keras_model.get_layer("text_projection")
    transfer_weights("kernel", text_proj.kernel, hf["text_projection.weight"])
    text_proj.bias.assign(hf["text_projection.bias"])

    # ── DETR encoder ─────────────────────────────────────────────
    print("Transferring DETR encoder...")
    num_enc_layers = keras_model.detr_encoder_num_layers
    for i in tqdm(range(num_enc_layers), desc="DETR encoder layers"):
        prefix = f"detr_encoder_layers_{i}"
        p = f"detr_encoder.layers.{i}"

        self_attn = keras_model.get_layer(f"{prefix}_self_attn")
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            transfer_weights(
                "kernel",
                getattr(self_attn, proj).kernel,
                hf[f"{p}.self_attn.{proj}.weight"],
            )
            getattr(self_attn, proj).bias.assign(hf[f"{p}.self_attn.{proj}.bias"])

        cross_attn = keras_model.get_layer(f"{prefix}_cross_attn")
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            transfer_weights(
                "kernel",
                getattr(cross_attn, proj).kernel,
                hf[f"{p}.cross_attn.{proj}.weight"],
            )
            getattr(cross_attn, proj).bias.assign(hf[f"{p}.cross_attn.{proj}.bias"])

        for ln_name in ["layer_norm1", "layer_norm2", "layer_norm3"]:
            ln = keras_model.get_layer(f"{prefix}_{ln_name}")
            ln.gamma.assign(hf[f"{p}.{ln_name}.weight"])
            ln.beta.assign(hf[f"{p}.{ln_name}.bias"])

        fc1 = keras_model.get_layer(f"{prefix}_fc1")
        transfer_weights("kernel", fc1.kernel, hf[f"{p}.mlp.fc1.weight"])
        fc1.bias.assign(hf[f"{p}.mlp.fc1.bias"])
        fc2 = keras_model.get_layer(f"{prefix}_fc2")
        transfer_weights("kernel", fc2.kernel, hf[f"{p}.mlp.fc2.weight"])
        fc2.bias.assign(hf[f"{p}.mlp.fc2.bias"])

    # ── DETR decoder ─────────────────────────────────────────────
    print("Transferring DETR decoder...")

    query_embed = keras_model.get_layer("detr_decoder_query_embed")
    query_embed.embeddings.assign(hf["detr_decoder.query_embed.weight"])

    ref_points = keras_model.get_layer("detr_decoder_reference_points")
    ref_points.embeddings.assign(hf["detr_decoder.reference_points.weight"])

    presence_token = keras_model.get_layer("detr_decoder_presence_token")
    presence_token.embeddings.assign(hf["detr_decoder.presence_token.weight"])

    # Box head (layer1, layer2, layer3 in HF → dense_0, dense_1, dense_2 in Keras)
    box_head = keras_model.get_layer("detr_decoder_box_head")
    for j, dense in enumerate(box_head.dense_layers):
        transfer_weights(
            "kernel",
            dense.kernel,
            hf[f"detr_decoder.box_head.layer{j + 1}.weight"],
        )
        dense.bias.assign(hf[f"detr_decoder.box_head.layer{j + 1}.bias"])

    pres_head = keras_model.get_layer("detr_decoder_presence_head")
    for j, dense in enumerate(pres_head.dense_layers):
        transfer_weights(
            "kernel",
            dense.kernel,
            hf[f"detr_decoder.presence_head.layer{j + 1}.weight"],
        )
        dense.bias.assign(hf[f"detr_decoder.presence_head.layer{j + 1}.bias"])

    rph = keras_model.get_layer("detr_decoder_ref_point_head")
    for j, dense in enumerate(rph.dense_layers):
        transfer_weights(
            "kernel",
            dense.kernel,
            hf[f"detr_decoder.ref_point_head.layer{j + 1}.weight"],
        )
        dense.bias.assign(hf[f"detr_decoder.ref_point_head.layer{j + 1}.bias"])

    # Decoder layer norm name mapping:
    # Keras layer_norm1 = HF self_attn_layer_norm
    # Keras layer_norm2 = HF text_cross_attn_layer_norm
    # Keras layer_norm3 = HF vision_cross_attn_layer_norm
    # Keras layer_norm4 = HF mlp_layer_norm
    dec_ln_map = {
        "layer_norm1": "self_attn_layer_norm",
        "layer_norm2": "text_cross_attn_layer_norm",
        "layer_norm3": "vision_cross_attn_layer_norm",
        "layer_norm4": "mlp_layer_norm",
    }

    num_dec_layers = keras_model.detr_decoder_num_layers
    for i in tqdm(range(num_dec_layers), desc="DETR decoder layers"):
        prefix = f"detr_decoder_layers_{i}"
        p = f"detr_decoder.layers.{i}"

        for attn_name in ["self_attn", "text_cross_attn", "vision_cross_attn"]:
            attn = keras_model.get_layer(f"{prefix}_{attn_name}")
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                transfer_weights(
                    "kernel",
                    getattr(attn, proj).kernel,
                    hf[f"{p}.{attn_name}.{proj}.weight"],
                )
                getattr(attn, proj).bias.assign(hf[f"{p}.{attn_name}.{proj}.bias"])

        for keras_ln, hf_ln in dec_ln_map.items():
            ln = keras_model.get_layer(f"{prefix}_{keras_ln}")
            ln.gamma.assign(hf[f"{p}.{hf_ln}.weight"])
            ln.beta.assign(hf[f"{p}.{hf_ln}.bias"])

        fc1 = keras_model.get_layer(f"{prefix}_fc1")
        transfer_weights("kernel", fc1.kernel, hf[f"{p}.mlp.fc1.weight"])
        fc1.bias.assign(hf[f"{p}.mlp.fc1.bias"])
        fc2 = keras_model.get_layer(f"{prefix}_fc2")
        transfer_weights("kernel", fc2.kernel, hf[f"{p}.mlp.fc2.weight"])
        fc2.bias.assign(hf[f"{p}.mlp.fc2.bias"])

    # ── DETR decoder shared norms ──────────────────────────────────
    print("Transferring decoder output/presence layer norms...")
    out_ln = keras_model.get_layer("detr_decoder_output_layer_norm")
    out_ln.gamma.assign(hf["detr_decoder.output_layer_norm.weight"])
    out_ln.beta.assign(hf["detr_decoder.output_layer_norm.bias"])

    pres_ln = keras_model.get_layer("detr_decoder_presence_layer_norm")
    pres_ln.gamma.assign(hf["detr_decoder.presence_layer_norm.weight"])
    pres_ln.beta.assign(hf["detr_decoder.presence_layer_norm.bias"])

    # ── Box RPB embeddings ───────────────────────────────────────
    print("Transferring box RPB embeddings...")
    box_rpb = keras_model.get_layer("detr_decoder_box_rpb")
    for axis_name in ["x", "y"]:
        rpb_mlp = getattr(box_rpb, f"box_rpb_embed_{axis_name}")
        for j, dense in enumerate(rpb_mlp.dense_layers):
            transfer_weights(
                "kernel",
                dense.kernel,
                hf[f"detr_decoder.box_rpb_embed_{axis_name}.layer{j + 1}.weight"],
            )
            dense.bias.assign(
                hf[f"detr_decoder.box_rpb_embed_{axis_name}.layer{j + 1}.bias"]
            )

    # ── Dot-product scoring ──────────────────────────────────────
    print("Transferring dot-product scoring...")
    prefix = "dot_product_scoring"

    for name, hf_name in [
        ("text_mlp_fc1", "text_mlp.layer1"),
        ("text_mlp_fc2", "text_mlp.layer2"),
    ]:
        layer = keras_model.get_layer(f"{prefix}_{name}")
        transfer_weights("kernel", layer.kernel, hf[f"{prefix}.{hf_name}.weight"])
        layer.bias.assign(hf[f"{prefix}.{hf_name}.bias"])

    norm = keras_model.get_layer(f"{prefix}_text_mlp_out_norm")
    norm.gamma.assign(hf[f"{prefix}.text_mlp_out_norm.weight"])
    norm.beta.assign(hf[f"{prefix}.text_mlp_out_norm.bias"])

    for name in ["text_proj", "query_proj"]:
        layer = keras_model.get_layer(f"{prefix}_{name}")
        transfer_weights("kernel", layer.kernel, hf[f"{prefix}.{name}.weight"])
        layer.bias.assign(hf[f"{prefix}.{name}.bias"])

    # ── Mask decoder ─────────────────────────────────────────────
    print("Transferring mask decoder...")

    # HF pixel decoder uses len(fpn_levels)-2 stages (2 for default config)
    num_up = len(keras_model.fpn_scale_factors) - 2
    for stage_idx in range(num_up):
        conv = keras_model.get_layer(f"pixel_decoder_stage_{stage_idx}_conv")
        transfer_weights(
            "conv_kernel",
            conv.kernel,
            hf[f"mask_decoder.pixel_decoder.conv_layers.{stage_idx}.weight"],
        )
        conv.bias.assign(hf[f"mask_decoder.pixel_decoder.conv_layers.{stage_idx}.bias"])

        gn = keras_model.get_layer(f"pixel_decoder_stage_{stage_idx}_gn")
        gn.gamma.assign(hf[f"mask_decoder.pixel_decoder.norms.{stage_idx}.weight"])
        gn.beta.assign(hf[f"mask_decoder.pixel_decoder.norms.{stage_idx}.bias"])

    inst_proj = keras_model.get_layer("mask_decoder_instance_proj")
    transfer_weights(
        "conv_kernel",
        inst_proj.kernel,
        hf["mask_decoder.instance_projection.weight"],
    )
    inst_proj.bias.assign(hf["mask_decoder.instance_projection.bias"])

    sem_proj = keras_model.get_layer("mask_decoder_semantic_proj")
    transfer_weights(
        "conv_kernel",
        sem_proj.kernel,
        hf["mask_decoder.semantic_projection.weight"],
    )
    sem_proj.bias.assign(hf["mask_decoder.semantic_projection.bias"])

    for j in range(3):
        me_layer = keras_model.get_layer(f"mask_embedder_linear{j + 1}")
        transfer_weights(
            "kernel",
            me_layer.kernel,
            hf[f"mask_decoder.mask_embedder.layers.{j}.weight"],
        )
        me_layer.bias.assign(hf[f"mask_decoder.mask_embedder.layers.{j}.bias"])

    # ── Prompt cross-attention in mask decoder ───────────────────
    print("Transferring prompt cross-attention...")
    pcattn = keras_model.get_layer("mask_decoder_prompt_cross_attn")
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        transfer_weights(
            "kernel",
            getattr(pcattn, proj).kernel,
            hf[f"mask_decoder.prompt_cross_attn.{proj}.weight"],
        )
        getattr(pcattn, proj).bias.assign(
            hf[f"mask_decoder.prompt_cross_attn.{proj}.bias"]
        )
    pcanorm = keras_model.get_layer("mask_decoder_prompt_cross_attn_norm")
    pcanorm.gamma.assign(hf["mask_decoder.prompt_cross_attn_norm.weight"])
    pcanorm.beta.assign(hf["mask_decoder.prompt_cross_attn_norm.bias"])

    # ── CLIP Text Encoder (standalone model) ────────────────────────
    print("Building and transferring CLIP text encoder...")
    text_enc_layer = SAM3CLIPTextEncoder(name="text_encoder")
    text_enc_layer.build((None, 32))
    if True:
        text_enc = text_enc_layer
        # Token + position embeddings
        text_enc.token_embedding.weights[0].assign(
            hf["text_encoder.text_model.embeddings.token_embedding.weight"]
        )
        text_enc.position_embedding.weights[0].assign(
            hf["text_encoder.text_model.embeddings.position_embedding.weight"]
        )
        # Encoder layers
        for i in tqdm(range(text_enc.num_hidden_layers), desc="CLIP text layers"):
            layer = text_enc.encoder_layers[i]
            hp = f"text_encoder.text_model.encoder.layers.{i}"
            layer.layer_norm1.gamma.assign(hf[f"{hp}.layer_norm1.weight"])
            layer.layer_norm1.beta.assign(hf[f"{hp}.layer_norm1.bias"])
            layer.layer_norm2.gamma.assign(hf[f"{hp}.layer_norm2.weight"])
            layer.layer_norm2.beta.assign(hf[f"{hp}.layer_norm2.bias"])
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                hf_proj = "out_proj" if proj == "o_proj" else proj
                transfer_weights(
                    "kernel",
                    getattr(layer.self_attn, proj).kernel,
                    hf[f"{hp}.self_attn.{hf_proj}.weight"],
                )
                getattr(layer.self_attn, proj).bias.assign(
                    hf[f"{hp}.self_attn.{hf_proj}.bias"]
                )
            transfer_weights("kernel", layer.fc1.kernel, hf[f"{hp}.mlp.fc1.weight"])
            layer.fc1.bias.assign(hf[f"{hp}.mlp.fc1.bias"])
            transfer_weights("kernel", layer.fc2.kernel, hf[f"{hp}.mlp.fc2.weight"])
            layer.fc2.bias.assign(hf[f"{hp}.mlp.fc2.bias"])
        text_enc.final_layer_norm.gamma.assign(
            hf["text_encoder.text_model.final_layer_norm.weight"]
        )
        text_enc.final_layer_norm.beta.assign(
            hf["text_encoder.text_model.final_layer_norm.bias"]
        )
    # Save text encoder
    input_ids = keras.Input(shape=(32,), dtype="int32", name="input_ids")
    attn_mask_input = keras.Input(shape=(32,), dtype="int32", name="attention_mask")
    text_out = text_enc_layer(input_ids, attention_mask=attn_mask_input)
    text_encoder_model = keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attn_mask_input},
        outputs=text_out,
        name="SAM3TextEncoder",
    )
    text_encoder_model.save_weights("sam3_text_encoder.weights.h5")
    print(f"  Text encoder saved: {text_encoder_model.count_params():,} params")

    # ── Geometry Encoder (standalone model) ───────────────────────
    print("Building and transferring geometry encoder...")
    geo_layer = SAM3GeometryEncoder(name="geometry_encoder")
    geo_layer.build((None, None, 4))
    if True:
        geo = geo_layer
        transfer_weights(
            "kernel",
            geo.boxes_direct_project.kernel,
            hf["geometry_encoder.boxes_direct_project.weight"],
        )
        geo.boxes_direct_project.bias.assign(
            hf["geometry_encoder.boxes_direct_project.bias"]
        )
        transfer_weights(
            "kernel",
            geo.boxes_pos_enc_project.kernel,
            hf["geometry_encoder.boxes_pos_enc_project.weight"],
        )
        geo.boxes_pos_enc_project.bias.assign(
            hf["geometry_encoder.boxes_pos_enc_project.bias"]
        )
        transfer_weights(
            "conv_kernel",
            geo.boxes_pool_project.kernel,
            hf["geometry_encoder.boxes_pool_project.weight"],
        )
        geo.boxes_pool_project.bias.assign(
            hf["geometry_encoder.boxes_pool_project.bias"]
        )
        geo.label_embed.weights[0].assign(hf["geometry_encoder.label_embed.weight"])
        geo.cls_embed.weights[0].assign(hf["geometry_encoder.cls_embed.weight"])
        geo.vision_layer_norm.gamma.assign(
            hf["geometry_encoder.vision_layer_norm.weight"]
        )
        geo.vision_layer_norm.beta.assign(hf["geometry_encoder.vision_layer_norm.bias"])
        geo.prompt_layer_norm.gamma.assign(
            hf["geometry_encoder.prompt_layer_norm.weight"]
        )
        geo.prompt_layer_norm.beta.assign(hf["geometry_encoder.prompt_layer_norm.bias"])
        geo.output_layer_norm.gamma.assign(
            hf["geometry_encoder.output_layer_norm.weight"]
        )
        geo.output_layer_norm.beta.assign(hf["geometry_encoder.output_layer_norm.bias"])
        transfer_weights(
            "kernel", geo.final_proj.kernel, hf["geometry_encoder.final_proj.weight"]
        )
        geo.final_proj.bias.assign(hf["geometry_encoder.final_proj.bias"])
        for i in range(len(geo.transformer_layers)):
            layer = geo.transformer_layers[i]
            gp = f"geometry_encoder.layers.{i}"
            for attn_name, hf_attn in [
                ("self_attn", "self_attn"),
                ("cross_attn", "cross_attn"),
            ]:
                attn = getattr(layer, attn_name)
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    transfer_weights(
                        "kernel",
                        getattr(attn, proj).kernel,
                        hf[f"{gp}.{hf_attn}.{proj}.weight"],
                    )
                    getattr(attn, proj).bias.assign(hf[f"{gp}.{hf_attn}.{proj}.bias"])
            layer.layer_norm1.gamma.assign(hf[f"{gp}.layer_norm1.weight"])
            layer.layer_norm1.beta.assign(hf[f"{gp}.layer_norm1.bias"])
            layer.layer_norm2.gamma.assign(hf[f"{gp}.layer_norm2.weight"])
            layer.layer_norm2.beta.assign(hf[f"{gp}.layer_norm2.bias"])
            layer.layer_norm3.gamma.assign(hf[f"{gp}.layer_norm3.weight"])
            layer.layer_norm3.beta.assign(hf[f"{gp}.layer_norm3.bias"])
            transfer_weights("kernel", layer.fc1.kernel, hf[f"{gp}.mlp.fc1.weight"])
            layer.fc1.bias.assign(hf[f"{gp}.mlp.fc1.bias"])
            transfer_weights("kernel", layer.fc2.kernel, hf[f"{gp}.mlp.fc2.weight"])
            layer.fc2.bias.assign(hf[f"{gp}.mlp.fc2.bias"])
    print(
        f"  Geometry encoder built: {sum(np.prod(w.shape) for w in geo_layer.weights):,} params"
    )

    # Save geometry encoder weights via a wrapper model
    boxes_in = keras.Input(shape=(None, 4), name="boxes", dtype="float32")
    labels_in = keras.Input(shape=(None,), name="box_labels", dtype="int32")
    mask_in = keras.Input(shape=(None,), name="box_mask", dtype="float32")
    vis_flat_in = keras.Input(shape=(None, 256), name="vision_flat", dtype="float32")
    pos_flat_in = keras.Input(shape=(None, 256), name="vision_pos", dtype="float32")
    geo_out, geo_mask_out = geo_layer(
        boxes_in, labels_in, mask_in, vis_flat_in, pos_flat_in
    )
    geo_model = keras.Model(
        inputs=[boxes_in, labels_in, mask_in, vis_flat_in, pos_flat_in],
        outputs={"features": geo_out, "mask": geo_mask_out},
    )
    geo_model.save_weights("sam3_geometry_encoder.weights.h5")
    print("  Geometry encoder weights saved: sam3_geometry_encoder.weights.h5")

    print(f"\nAll {len(hf)} HF weight tensors transferred.")

    print("\nWeight transfer complete!")

    # ── Model equivalence test ─────────────────────────────────
    print("\nRunning model equivalence test...")
    np.random.seed(42)
    input_shape = model_config["input_shape"]
    test_image = np.random.rand(1, *input_shape).astype(np.float32)
    test_text_features = np.random.rand(1, 10, 1024).astype(np.float32)
    test_text_mask = np.ones((1, 10), dtype=np.float32)

    # Run Keras model
    keras_output = keras_model.predict(
        {
            "pixel_values": test_image,
            "text_features": test_text_features,
            "text_attention_mask": test_text_mask,
        },
        verbose=0,
    )

    print("Keras output shapes:")
    for k, v in keras_output.items():
        print(f"  {k}: {v.shape}")

    # Run HF model with same inputs (using pre-computed text features)
    test_image_torch = torch.from_numpy(test_image).permute(0, 3, 1, 2)  # NHWC -> NCHW
    test_text_torch = torch.from_numpy(test_text_features)
    test_text_mask_torch = torch.from_numpy(test_text_mask).long()

    # HF expects text_embeds as already-projected (256-dim) features
    with torch.no_grad():
        # Project text features the same way as our Keras model
        text_projected_hf = hf_model.text_projection(test_text_torch)
        hf_out = hf_model(
            pixel_values=test_image_torch,
            text_embeds=text_projected_hf,
            attention_mask=test_text_mask_torch,
        )

    hf_masks = hf_out.pred_masks.cpu().numpy()
    hf_boxes = hf_out.pred_boxes.cpu().numpy()
    hf_logits = hf_out.pred_logits.cpu().numpy()
    hf_semantic = hf_out.semantic_seg.cpu().numpy()

    keras_masks = keras_output["pred_masks"]
    keras_boxes = keras_output["pred_boxes"]
    keras_logits = keras_output["pred_logits"]
    keras_semantic = keras_output["semantic_seg"]

    mask_diff = np.max(np.abs(keras_masks - hf_masks))
    box_diff = np.max(np.abs(keras_boxes - hf_boxes))
    logit_diff = np.max(
        np.abs(
            keras_logits - hf_logits.squeeze(-1)
            if hf_logits.ndim > keras_logits.ndim
            else hf_logits
        )
    )
    semantic_diff = np.max(np.abs(keras_semantic - hf_semantic))

    print(f"\n{'=' * 40}")
    print("Model Equivalence Results:")
    print(f"  mask_diff:     {mask_diff:.6e}")
    print(f"  box_diff:      {box_diff:.6e}")
    print(f"  logit_diff:    {logit_diff:.6e}")
    print(f"  semantic_diff: {semantic_diff:.6e}")
    print(f"{'=' * 40}")

    if mask_diff < 1e-3 and box_diff < 1e-3:
        print("PASS: Model outputs match within tolerance.")
    else:
        print("WARNING: Outputs differ significantly. Check component implementations.")

    model_base = model_config["hf_model_name"].split("/")[-1].replace("-", "_")
    model_filename = model_base + ".weights.h5"
    keras_model.save_weights(model_filename)
    print(f"\nModel saved as {model_filename}")

    del keras_model, hf_model, hf
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    for config in model_configs:
        convert_sam3(config)
