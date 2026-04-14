import gc

import keras
import numpy as np
import torch
from transformers import Sam2Model, Sam2VideoModel

from kmodels.models.sam2_video.sam2_video_model import (
    Sam2VideoBasePlus,
    Sam2VideoLarge,
    Sam2VideoSmall,
    Sam2VideoTiny,
)
from kmodels.utils.weight_transfer_torch_to_keras import (
    transfer_nested_layer_weights,
    transfer_weights,
)

SAM2_VIDEO_HF_MODEL_IDS = {
    "Sam2VideoTiny": "facebook/sam2-hiera-tiny",
    "Sam2VideoSmall": "facebook/sam2-hiera-small",
    "Sam2VideoBasePlus": "facebook/sam2-hiera-base-plus",
    "Sam2VideoLarge": "facebook/sam2-hiera-large",
}

VARIANTS = [
    (
        "Sam2VideoTiny",
        Sam2VideoTiny,
        SAM2_VIDEO_HF_MODEL_IDS["Sam2VideoTiny"],
        "sam2_video_hiera_tiny",
    ),
    (
        "Sam2VideoSmall",
        Sam2VideoSmall,
        SAM2_VIDEO_HF_MODEL_IDS["Sam2VideoSmall"],
        "sam2_video_hiera_small",
    ),
    (
        "Sam2VideoBasePlus",
        Sam2VideoBasePlus,
        SAM2_VIDEO_HF_MODEL_IDS["Sam2VideoBasePlus"],
        "sam2_video_hiera_base_plus",
    ),
    (
        "Sam2VideoLarge",
        Sam2VideoLarge,
        SAM2_VIDEO_HF_MODEL_IDS["Sam2VideoLarge"],
        "sam2_video_hiera_large",
    ),
]

BACKBONE_NAME_MAPPING = {
    "mlp_proj_in": "mlp.proj_in",
    "mlp_proj_out": "mlp.proj_out",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def _transfer_backbone(keras_model, hf_sd):
    patch_conv = keras_model.get_layer("backbone_patch_embed_projection")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf_sd["vision_encoder.backbone.patch_embed.projection.weight"],
    )
    patch_conv.bias.assign(hf_sd["vision_encoder.backbone.patch_embed.projection.bias"])

    pos_layer = keras_model.get_layer("backbone_pos_embed")
    pos_layer.pos_embed.assign(
        np.transpose(hf_sd["vision_encoder.backbone.pos_embed"], (0, 2, 3, 1))
    )
    pos_layer.pos_embed_window.assign(
        np.transpose(hf_sd["vision_encoder.backbone.pos_embed_window"], (0, 2, 3, 1))
    )
    pos_layer._recompute_full_pos()

    total_blocks = sum(keras_model.blocks_per_stage)
    for i in range(total_blocks):
        layer = keras_model.get_layer(f"backbone_blocks_{i}")
        transfer_nested_layer_weights(
            layer,
            hf_sd,
            f"vision_encoder.backbone.blocks.{i}",
            name_mapping=BACKBONE_NAME_MAPPING,
        )

    n_fpn = len(keras_model.backbone_channel_list)
    for i in range(n_fpn):
        neck_conv = keras_model.get_layer(f"neck_convs_{i}")
        transfer_weights(
            "conv_kernel",
            neck_conv.kernel,
            hf_sd[f"vision_encoder.neck.convs.{i}.weight"],
        )
        neck_conv.bias.assign(hf_sd[f"vision_encoder.neck.convs.{i}.bias"])


def _transfer_no_memory_embedding(keras_model, hf_sd):
    no_mem = keras_model.get_layer("no_memory_embedding")
    no_mem.embedding.assign(hf_sd["no_memory_embedding"].reshape(1, 1, 1, -1))


def _transfer_prompt_encoder(keras_model, hf_sd):
    prompt_enc = keras_model.get_layer("prompt_encoder")
    prompt_enc.shared_embedding.positional_embedding.assign(
        hf_sd["shared_image_embedding.positional_embedding"]
    )
    hf_point = hf_sd["prompt_encoder.point_embed.weight"]
    for i in range(prompt_enc.num_point_embeddings):
        prompt_enc.point_embeddings[i].assign(hf_point[i : i + 1])
    prompt_enc.not_a_point_embed.assign(
        hf_sd["prompt_encoder.not_a_point_embed.weight"]
    )
    prompt_enc.no_mask_embed.assign(hf_sd["prompt_encoder.no_mask_embed.weight"])


def _transfer_mask_decoder(keras_model, hf_sd):
    mask_dec = keras_model.get_layer("mask_decoder")
    mask_dec.obj_score_token.assign(hf_sd["mask_decoder.obj_score_token.weight"])
    mask_dec.iou_token.assign(hf_sd["mask_decoder.iou_token.weight"])
    mask_dec.mask_tokens.assign(hf_sd["mask_decoder.mask_tokens.weight"])

    for i in range(mask_dec.num_hidden_layers):
        hf_pfx = f"mask_decoder.transformer.layers.{i}"
        for attn_layer, attn_suffix in [
            (mask_dec.transformer_self_attns[i], "self_attn"),
            (
                mask_dec.transformer_cross_attn_token_to_images[i],
                "cross_attn_token_to_image",
            ),
            (
                mask_dec.transformer_cross_attn_image_to_tokens[i],
                "cross_attn_image_to_token",
            ),
        ]:
            for proj, hf_proj in [
                ("q_proj", "q_proj"),
                ("k_proj", "k_proj"),
                ("v_proj", "v_proj"),
                ("out_proj", "o_proj"),
            ]:
                p = getattr(attn_layer, proj)
                transfer_weights(
                    "kernel",
                    p.kernel,
                    hf_sd[f"{hf_pfx}.{attn_suffix}.{hf_proj}.weight"],
                )
                p.bias.assign(hf_sd[f"{hf_pfx}.{attn_suffix}.{hf_proj}.bias"])

        for ln_idx in range(1, 5):
            ln = getattr(mask_dec, f"transformer_layer_norm{ln_idx}s")[i]
            ln.gamma.assign(hf_sd[f"{hf_pfx}.layer_norm{ln_idx}.weight"])
            ln.beta.assign(hf_sd[f"{hf_pfx}.layer_norm{ln_idx}.bias"])

        mlp1 = mask_dec.transformer_mlp_lin1s[i]
        transfer_weights("kernel", mlp1.kernel, hf_sd[f"{hf_pfx}.mlp.proj_in.weight"])
        mlp1.bias.assign(hf_sd[f"{hf_pfx}.mlp.proj_in.bias"])
        mlp2 = mask_dec.transformer_mlp_lin2s[i]
        transfer_weights("kernel", mlp2.kernel, hf_sd[f"{hf_pfx}.mlp.proj_out.weight"])
        mlp2.bias.assign(hf_sd[f"{hf_pfx}.mlp.proj_out.bias"])

    hf_final = "mask_decoder.transformer.final_attn_token_to_image"
    for proj, hf_proj in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", "o_proj"),
    ]:
        p = getattr(mask_dec.final_attn_token_to_image, proj)
        transfer_weights("kernel", p.kernel, hf_sd[f"{hf_final}.{hf_proj}.weight"])
        p.bias.assign(hf_sd[f"{hf_final}.{hf_proj}.bias"])

    mask_dec.layer_norm_final_attn.gamma.assign(
        hf_sd["mask_decoder.transformer.layer_norm_final_attn.weight"]
    )
    mask_dec.layer_norm_final_attn.beta.assign(
        hf_sd["mask_decoder.transformer.layer_norm_final_attn.bias"]
    )

    transfer_weights(
        "conv_kernel",
        mask_dec.upscale_conv1.kernel,
        hf_sd["mask_decoder.upscale_conv1.weight"],
    )
    mask_dec.upscale_conv1.bias.assign(hf_sd["mask_decoder.upscale_conv1.bias"])
    mask_dec.upscale_layer_norm.gamma.assign(
        hf_sd["mask_decoder.upscale_layer_norm.weight"]
    )
    mask_dec.upscale_layer_norm.beta.assign(
        hf_sd["mask_decoder.upscale_layer_norm.bias"]
    )
    transfer_weights(
        "conv_kernel",
        mask_dec.upscale_conv2.kernel,
        hf_sd["mask_decoder.upscale_conv2.weight"],
    )
    mask_dec.upscale_conv2.bias.assign(hf_sd["mask_decoder.upscale_conv2.bias"])

    transfer_weights(
        "conv_kernel", mask_dec.conv_s0.kernel, hf_sd["mask_decoder.conv_s0.weight"]
    )
    mask_dec.conv_s0.bias.assign(hf_sd["mask_decoder.conv_s0.bias"])
    transfer_weights(
        "conv_kernel", mask_dec.conv_s1.kernel, hf_sd["mask_decoder.conv_s1.weight"]
    )
    mask_dec.conv_s1.bias.assign(hf_sd["mask_decoder.conv_s1.bias"])

    for i in range(mask_dec.num_mask_tokens):
        hf_pfx = f"mask_decoder.output_hypernetworks_mlps.{i}"
        transfer_weights(
            "kernel",
            mask_dec.output_hypernetworks_mlps_proj_ins[i].kernel,
            hf_sd[f"{hf_pfx}.proj_in.weight"],
        )
        mask_dec.output_hypernetworks_mlps_proj_ins[i].bias.assign(
            hf_sd[f"{hf_pfx}.proj_in.bias"]
        )
        transfer_weights(
            "kernel",
            mask_dec.output_hypernetworks_mlps_hidden_layers[i].kernel,
            hf_sd[f"{hf_pfx}.layers.0.weight"],
        )
        mask_dec.output_hypernetworks_mlps_hidden_layers[i].bias.assign(
            hf_sd[f"{hf_pfx}.layers.0.bias"]
        )
        transfer_weights(
            "kernel",
            mask_dec.output_hypernetworks_mlps_proj_outs[i].kernel,
            hf_sd[f"{hf_pfx}.proj_out.weight"],
        )
        mask_dec.output_hypernetworks_mlps_proj_outs[i].bias.assign(
            hf_sd[f"{hf_pfx}.proj_out.bias"]
        )

    for head_prefix, proj_in, hiddens, proj_out in [
        (
            "mask_decoder.iou_prediction_head",
            mask_dec.iou_head_proj_in,
            mask_dec.iou_head_hidden_layers,
            mask_dec.iou_head_proj_out,
        ),
        (
            "mask_decoder.pred_obj_score_head",
            mask_dec.obj_score_proj_in,
            mask_dec.obj_score_hidden_layers,
            mask_dec.obj_score_proj_out,
        ),
    ]:
        transfer_weights(
            "kernel", proj_in.kernel, hf_sd[f"{head_prefix}.proj_in.weight"]
        )
        proj_in.bias.assign(hf_sd[f"{head_prefix}.proj_in.bias"])
        for j, hl in enumerate(hiddens):
            transfer_weights(
                "kernel", hl.kernel, hf_sd[f"{head_prefix}.layers.{j}.weight"]
            )
            hl.bias.assign(hf_sd[f"{head_prefix}.layers.{j}.bias"])
        transfer_weights(
            "kernel", proj_out.kernel, hf_sd[f"{head_prefix}.proj_out.weight"]
        )
        proj_out.bias.assign(hf_sd[f"{head_prefix}.proj_out.bias"])


def _transfer_memory_attention(keras_model, hf_sd):
    mem_attn = keras_model.memory_attention
    num_layers = mem_attn.num_layers

    for i in range(num_layers):
        layer = mem_attn.attn_layers[i]
        hf_pfx = f"memory_attention.layers.{i}"

        for attn, attn_name in [
            (layer.self_attn, "self_attn"),
            (layer.cross_attn_image, "cross_attn_image"),
        ]:
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                p = getattr(attn, proj)
                transfer_weights(
                    "kernel", p.kernel, hf_sd[f"{hf_pfx}.{attn_name}.{proj}.weight"]
                )
                p.bias.assign(hf_sd[f"{hf_pfx}.{attn_name}.{proj}.bias"])

        transfer_weights(
            "kernel", layer.linear1.kernel, hf_sd[f"{hf_pfx}.linear1.weight"]
        )
        layer.linear1.bias.assign(hf_sd[f"{hf_pfx}.linear1.bias"])
        transfer_weights(
            "kernel", layer.linear2.kernel, hf_sd[f"{hf_pfx}.linear2.weight"]
        )
        layer.linear2.bias.assign(hf_sd[f"{hf_pfx}.linear2.bias"])

        for ln_idx in range(1, 4):
            ln = getattr(layer, f"layer_norm{ln_idx}")
            ln.gamma.assign(hf_sd[f"{hf_pfx}.layer_norm{ln_idx}.weight"])
            ln.beta.assign(hf_sd[f"{hf_pfx}.layer_norm{ln_idx}.bias"])

    mem_attn.layer_norm.gamma.assign(hf_sd["memory_attention.layer_norm.weight"])
    mem_attn.layer_norm.beta.assign(hf_sd["memory_attention.layer_norm.bias"])


def _transfer_memory_encoder(keras_model, hf_sd):
    sub = keras_model.memory_encoder_submodel

    for i in range(4):
        hf_pfx = f"memory_encoder.mask_downsampler.layers.{i}"
        conv = sub.get_layer(f"mem_enc_mask_ds_conv_{i}")
        ln = sub.get_layer(f"mem_enc_mask_ds_ln_{i}")
        transfer_weights("conv_kernel", conv.kernel, hf_sd[f"{hf_pfx}.conv.weight"])
        conv.bias.assign(hf_sd[f"{hf_pfx}.conv.bias"])
        ln.gamma.assign(hf_sd[f"{hf_pfx}.layer_norm.weight"])
        ln.beta.assign(hf_sd[f"{hf_pfx}.layer_norm.bias"])

    final_conv = sub.get_layer("mem_enc_mask_ds_final_conv")
    transfer_weights(
        "conv_kernel",
        final_conv.kernel,
        hf_sd["memory_encoder.mask_downsampler.final_conv.weight"],
    )
    final_conv.bias.assign(hf_sd["memory_encoder.mask_downsampler.final_conv.bias"])

    feat_proj = sub.get_layer("mem_enc_feature_proj")
    transfer_weights(
        "conv_kernel",
        feat_proj.kernel,
        hf_sd["memory_encoder.feature_projection.weight"],
    )
    feat_proj.bias.assign(hf_sd["memory_encoder.feature_projection.bias"])

    for i in range(2):
        hf_pfx = f"memory_encoder.memory_fuser.layers.{i}"
        dw_conv = sub.get_layer(f"mem_enc_fuser_{i}_dw_conv")
        ln = sub.get_layer(f"mem_enc_fuser_{i}_ln")
        pw1 = sub.get_layer(f"mem_enc_fuser_{i}_pw1")
        pw2 = sub.get_layer(f"mem_enc_fuser_{i}_pw2")
        scale_layer = sub.get_layer(f"mem_enc_fuser_{i}_scale")
        scale_layer.scale.assign(hf_sd[f"{hf_pfx}.scale"])
        dw_w = hf_sd[f"{hf_pfx}.depthwise_conv.weight"]
        dw_w = np.transpose(dw_w, (2, 3, 0, 1))
        dw_conv.kernel.assign(dw_w)
        dw_conv.bias.assign(hf_sd[f"{hf_pfx}.depthwise_conv.bias"])
        ln.gamma.assign(hf_sd[f"{hf_pfx}.layer_norm.weight"])
        ln.beta.assign(hf_sd[f"{hf_pfx}.layer_norm.bias"])
        transfer_weights(
            "kernel", pw1.kernel, hf_sd[f"{hf_pfx}.pointwise_conv1.weight"]
        )
        pw1.bias.assign(hf_sd[f"{hf_pfx}.pointwise_conv1.bias"])
        transfer_weights(
            "kernel", pw2.kernel, hf_sd[f"{hf_pfx}.pointwise_conv2.weight"]
        )
        pw2.bias.assign(hf_sd[f"{hf_pfx}.pointwise_conv2.bias"])

    projection = sub.get_layer("mem_enc_projection")
    transfer_weights(
        "conv_kernel",
        projection.kernel,
        hf_sd["memory_encoder.projection.weight"],
    )
    projection.bias.assign(hf_sd["memory_encoder.projection.bias"])


def _transfer_video_params(keras_model, hf_sd):
    keras_model.no_memory_positional_encoding.assign(
        hf_sd["no_memory_positional_encoding"]
    )
    keras_model.memory_temporal_positional_encoding.assign(
        hf_sd["memory_temporal_positional_encoding"]
    )
    keras_model.no_object_pointer.assign(hf_sd["no_object_pointer"])
    if "occlusion_spatial_embedding_parameter" in hf_sd:
        keras_model.occlusion_spatial_embedding_parameter.assign(
            hf_sd["occlusion_spatial_embedding_parameter"]
        )

    ptr_sub = keras_model.obj_ptr_proj_submodel
    proj_in = ptr_sub.get_layer("obj_ptr_proj_proj_in")
    layer_0 = ptr_sub.get_layer("obj_ptr_proj_layers_0")
    proj_out = ptr_sub.get_layer("obj_ptr_proj_proj_out")

    transfer_weights(
        "kernel", proj_in.kernel, hf_sd["object_pointer_proj.proj_in.weight"]
    )
    proj_in.bias.assign(hf_sd["object_pointer_proj.proj_in.bias"])
    transfer_weights(
        "kernel", layer_0.kernel, hf_sd["object_pointer_proj.layers.0.weight"]
    )
    layer_0.bias.assign(hf_sd["object_pointer_proj.layers.0.bias"])
    transfer_weights(
        "kernel", proj_out.kernel, hf_sd["object_pointer_proj.proj_out.weight"]
    )
    proj_out.bias.assign(hf_sd["object_pointer_proj.proj_out.bias"])

    transfer_weights(
        "conv_kernel",
        keras_model.mask_downsample_layer.kernel,
        hf_sd["mask_downsample.weight"],
    )
    keras_model.mask_downsample_layer.bias.assign(hf_sd["mask_downsample.bias"])

    if "temporal_positional_encoding_projection_layer.weight" in hf_sd:
        transfer_weights(
            "kernel",
            keras_model.temporal_pos_enc_proj.kernel,
            hf_sd["temporal_positional_encoding_projection_layer.weight"],
        )
        keras_model.temporal_pos_enc_proj.bias.assign(
            hf_sd["temporal_positional_encoding_projection_layer.bias"]
        )
    else:
        keras_model.temporal_pos_enc_proj.kernel.assign(
            np.zeros_like(keras_model.temporal_pos_enc_proj.kernel.numpy())
        )
        keras_model.temporal_pos_enc_proj.bias.assign(
            np.zeros_like(keras_model.temporal_pos_enc_proj.bias.numpy())
        )


def transfer_sam2_video_weights(keras_model, hf_state_dict):
    _transfer_backbone(keras_model, hf_state_dict)
    _transfer_no_memory_embedding(keras_model, hf_state_dict)
    _transfer_prompt_encoder(keras_model, hf_state_dict)
    _transfer_mask_decoder(keras_model, hf_state_dict)
    _transfer_memory_attention(keras_model, hf_state_dict)
    _transfer_memory_encoder(keras_model, hf_state_dict)
    _transfer_video_params(keras_model, hf_state_dict)


if __name__ == "__main__":
    for name, ctor, hf_id, save_name in VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Converting: {name}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_video_model = Sam2VideoModel.from_pretrained(
            hf_id, attn_implementation="eager"
        ).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_video_model.state_dict().items()}

        hf_image_model = Sam2Model.from_pretrained(
            hf_id, attn_implementation="eager"
        ).eval()

        keras_model = ctor(input_shape=(1024, 1024, 3), weights=None)
        transfer_sam2_video_weights(keras_model, hf_sd)

        np.random.seed(42)
        test_image = np.random.rand(1, 1024, 1024, 3).astype(np.float32)
        test_points = np.array([[[[500.0, 500.0]]]], dtype=np.float32)
        test_labels = np.array([[[1]]], dtype=np.int32)

        keras_output = keras_model.predict(
            {
                "pixel_values": test_image,
                "input_points": test_points,
                "input_labels": test_labels,
            },
            verbose=0,
        )
        keras_masks = keras_output["pred_masks"]
        keras_iou = keras_output["iou_scores"]

        with torch.no_grad():
            hf_input = {
                "pixel_values": torch.from_numpy(test_image.transpose(0, 3, 1, 2)),
                "input_points": torch.from_numpy(test_points),
                "input_labels": torch.from_numpy(test_labels),
                "multimask_output": True,
            }
            hf_output = hf_image_model(**hf_input)
            hf_masks = hf_output.pred_masks.cpu().numpy()
            hf_iou = hf_output.iou_scores.cpu().numpy()

        mask_diff = float(np.max(np.abs(keras_masks - hf_masks)))
        iou_diff = float(np.max(np.abs(keras_iou - hf_iou)))
        print(f"  Max mask diff: {mask_diff:.6f}")
        print(f"  Max IoU diff:  {iou_diff:.6f}")
        assert mask_diff < 0.5, f"{name}: mask diff {mask_diff:.2e}"
        assert iou_diff < 0.05, f"{name}: IoU diff {iou_diff:.2e}"
        print("  Verification OK")

        out = f"{save_name}.weights.h5"
        keras_model.save_weights(out)
        print(f"  Saved -> {out}")

        del keras_model, hf_video_model, hf_image_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
