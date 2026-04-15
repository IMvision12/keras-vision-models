import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from tqdm import tqdm
from transformers import Sam2Model

from kmodels.models.sam2.sam2_model import (
    Sam2BasePlus,
    Sam2Large,
    Sam2Small,
    Sam2Tiny,
)
from kmodels.utils.weight_transfer_torch_to_keras import (
    transfer_nested_layer_weights,
    transfer_weights,
)

backbone_name_mapping = {
    "mlp_proj_in": "mlp.proj_in",
    "mlp_proj_out": "mlp.proj_out",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

model_configs = [
    {
        "keras_model_cls": Sam2Tiny,
        "hf_model_name": "facebook/sam2-hiera-tiny",
        "input_shape": (1024, 1024, 3),
    },
    {
        "keras_model_cls": Sam2Small,
        "hf_model_name": "facebook/sam2-hiera-small",
        "input_shape": (1024, 1024, 3),
    },
    {
        "keras_model_cls": Sam2BasePlus,
        "hf_model_name": "facebook/sam2-hiera-base-plus",
        "input_shape": (1024, 1024, 3),
    },
    {
        "keras_model_cls": Sam2Large,
        "hf_model_name": "facebook/sam2-hiera-large",
        "input_shape": (1024, 1024, 3),
    },
]

for model_config in model_configs:
    print(f"\n{'=' * 60}")
    print(f"Converting {model_config['hf_model_name']}...")
    print(f"{'=' * 60}")

    print(f"Loading HF model: {model_config['hf_model_name']}")
    hf_model = Sam2Model.from_pretrained(
        model_config["hf_model_name"], attn_implementation="eager"
    ).eval()
    hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

    print("Creating Keras model...")
    keras_model = model_config["keras_model_cls"](
        input_shape=model_config["input_shape"], weights=None
    )

    print("Transferring patch embedding...")
    patch_conv = keras_model.get_layer("backbone_patch_embed_projection")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf_state_dict["vision_encoder.backbone.patch_embed.projection.weight"],
    )
    patch_conv.bias.assign(
        hf_state_dict["vision_encoder.backbone.patch_embed.projection.bias"]
    )

    print("Transferring positional embedding...")
    pos_layer = keras_model.get_layer("backbone_pos_embed")
    pos_embed_hf = hf_state_dict["vision_encoder.backbone.pos_embed"]
    pos_layer.pos_embed.assign(np.transpose(pos_embed_hf, (0, 2, 3, 1)))
    pos_embed_window_hf = hf_state_dict["vision_encoder.backbone.pos_embed_window"]
    pos_layer.pos_embed_window.assign(np.transpose(pos_embed_window_hf, (0, 2, 3, 1)))
    pos_layer._recompute_full_pos()

    total_blocks = sum(keras_model.blocks_per_stage)
    for i in tqdm(range(total_blocks), desc="Transferring backbone blocks"):
        layer = keras_model.get_layer(f"backbone_blocks_{i}")
        skipped = transfer_nested_layer_weights(
            layer,
            hf_state_dict,
            f"vision_encoder.backbone.blocks.{i}",
            name_mapping=backbone_name_mapping,
        )
        if skipped:
            for w, path in skipped:
                print(f"  WARNING: Skipped {path}")

    print("Transferring FPN neck...")
    n_fpn = len(keras_model.backbone_channel_list)
    for i in range(n_fpn):
        neck_conv = keras_model.get_layer(f"neck_convs_{i}")
        hf_key = f"vision_encoder.neck.convs.{i}.weight"
        transfer_weights("conv_kernel", neck_conv.kernel, hf_state_dict[hf_key])
        neck_conv.bias.assign(hf_state_dict[f"vision_encoder.neck.convs.{i}.bias"])

    print("Transferring no-memory embedding...")
    no_mem_layer = keras_model.get_layer("no_memory_embedding")
    hf_no_mem = hf_state_dict["no_memory_embedding"]
    no_mem_layer.embedding.assign(hf_no_mem.reshape(1, 1, 1, -1))

    print("Transferring prompt encoder...")
    prompt_enc = keras_model.get_layer("prompt_encoder")
    prompt_enc.shared_embedding.positional_embedding.assign(
        hf_state_dict["shared_image_embedding.positional_embedding"]
    )

    hf_point_embed = hf_state_dict["prompt_encoder.point_embed.weight"]
    for i in range(prompt_enc.num_point_embeddings):
        prompt_enc.point_embeddings[i].assign(hf_point_embed[i : i + 1])

    prompt_enc.not_a_point_embed.assign(
        hf_state_dict["prompt_encoder.not_a_point_embed.weight"]
    )
    prompt_enc.no_mask_embed.assign(
        hf_state_dict["prompt_encoder.no_mask_embed.weight"]
    )

    print("Transferring mask decoder...")
    mask_dec = keras_model.get_layer("mask_decoder")

    mask_dec.obj_score_token.assign(
        hf_state_dict["mask_decoder.obj_score_token.weight"]
    )
    mask_dec.iou_token.assign(hf_state_dict["mask_decoder.iou_token.weight"])
    mask_dec.mask_tokens.assign(hf_state_dict["mask_decoder.mask_tokens.weight"])

    for i in range(mask_dec.num_hidden_layers):
        hf_prefix = f"mask_decoder.transformer.layers.{i}"

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
            for proj_name, hf_proj_name in [
                ("q_proj", "q_proj"),
                ("k_proj", "k_proj"),
                ("v_proj", "v_proj"),
                ("out_proj", "o_proj"),
            ]:
                proj = getattr(attn_layer, proj_name)
                transfer_weights(
                    "kernel",
                    proj.kernel,
                    hf_state_dict[f"{hf_prefix}.{attn_suffix}.{hf_proj_name}.weight"],
                )
                proj.bias.assign(
                    hf_state_dict[f"{hf_prefix}.{attn_suffix}.{hf_proj_name}.bias"]
                )

        ln1 = mask_dec.transformer_layer_norm1s[i]
        ln1.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.weight"])
        ln1.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.bias"])

        ln2 = mask_dec.transformer_layer_norm2s[i]
        ln2.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.weight"])
        ln2.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.bias"])

        mlp_lin1 = mask_dec.transformer_mlp_lin1s[i]
        transfer_weights(
            "kernel",
            mlp_lin1.kernel,
            hf_state_dict[f"{hf_prefix}.mlp.proj_in.weight"],
        )
        mlp_lin1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.proj_in.bias"])

        mlp_lin2 = mask_dec.transformer_mlp_lin2s[i]
        transfer_weights(
            "kernel",
            mlp_lin2.kernel,
            hf_state_dict[f"{hf_prefix}.mlp.proj_out.weight"],
        )
        mlp_lin2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.proj_out.bias"])

        ln3 = mask_dec.transformer_layer_norm3s[i]
        ln3.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm3.weight"])
        ln3.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm3.bias"])

        ln4 = mask_dec.transformer_layer_norm4s[i]
        ln4.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm4.weight"])
        ln4.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm4.bias"])

    hf_final_attn = "mask_decoder.transformer.final_attn_token_to_image"
    for proj_name, hf_proj_name in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", "o_proj"),
    ]:
        proj = getattr(mask_dec.final_attn_token_to_image, proj_name)
        transfer_weights(
            "kernel",
            proj.kernel,
            hf_state_dict[f"{hf_final_attn}.{hf_proj_name}.weight"],
        )
        proj.bias.assign(hf_state_dict[f"{hf_final_attn}.{hf_proj_name}.bias"])

    final_ln = mask_dec.layer_norm_final_attn
    final_ln.gamma.assign(
        hf_state_dict["mask_decoder.transformer.layer_norm_final_attn.weight"]
    )
    final_ln.beta.assign(
        hf_state_dict["mask_decoder.transformer.layer_norm_final_attn.bias"]
    )

    transfer_weights(
        "conv_kernel",
        mask_dec.upscale_conv1.kernel,
        hf_state_dict["mask_decoder.upscale_conv1.weight"],
    )
    mask_dec.upscale_conv1.bias.assign(hf_state_dict["mask_decoder.upscale_conv1.bias"])

    mask_dec.upscale_layer_norm.gamma.assign(
        hf_state_dict["mask_decoder.upscale_layer_norm.weight"]
    )
    mask_dec.upscale_layer_norm.beta.assign(
        hf_state_dict["mask_decoder.upscale_layer_norm.bias"]
    )

    transfer_weights(
        "conv_kernel",
        mask_dec.upscale_conv2.kernel,
        hf_state_dict["mask_decoder.upscale_conv2.weight"],
    )
    mask_dec.upscale_conv2.bias.assign(hf_state_dict["mask_decoder.upscale_conv2.bias"])

    transfer_weights(
        "conv_kernel",
        mask_dec.conv_s0.kernel,
        hf_state_dict["mask_decoder.conv_s0.weight"],
    )
    mask_dec.conv_s0.bias.assign(hf_state_dict["mask_decoder.conv_s0.bias"])

    transfer_weights(
        "conv_kernel",
        mask_dec.conv_s1.kernel,
        hf_state_dict["mask_decoder.conv_s1.weight"],
    )
    mask_dec.conv_s1.bias.assign(hf_state_dict["mask_decoder.conv_s1.bias"])

    num_mask_tokens = mask_dec.num_mask_tokens
    for i in range(num_mask_tokens):
        hf_prefix = f"mask_decoder.output_hypernetworks_mlps.{i}"

        proj_in = mask_dec.output_hypernetworks_mlps_proj_ins[i]
        transfer_weights(
            "kernel",
            proj_in.kernel,
            hf_state_dict[f"{hf_prefix}.proj_in.weight"],
        )
        proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])

        hidden = mask_dec.output_hypernetworks_mlps_hidden_layers[i]
        transfer_weights(
            "kernel",
            hidden.kernel,
            hf_state_dict[f"{hf_prefix}.layers.0.weight"],
        )
        hidden.bias.assign(hf_state_dict[f"{hf_prefix}.layers.0.bias"])

        proj_out = mask_dec.output_hypernetworks_mlps_proj_outs[i]
        transfer_weights(
            "kernel",
            proj_out.kernel,
            hf_state_dict[f"{hf_prefix}.proj_out.weight"],
        )
        proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])

    hf_prefix = "mask_decoder.iou_prediction_head"
    transfer_weights(
        "kernel",
        mask_dec.iou_head_proj_in.kernel,
        hf_state_dict[f"{hf_prefix}.proj_in.weight"],
    )
    mask_dec.iou_head_proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])
    for j, hidden_layer in enumerate(mask_dec.iou_head_hidden_layers):
        transfer_weights(
            "kernel",
            hidden_layer.kernel,
            hf_state_dict[f"{hf_prefix}.layers.{j}.weight"],
        )
        hidden_layer.bias.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.bias"])
    transfer_weights(
        "kernel",
        mask_dec.iou_head_proj_out.kernel,
        hf_state_dict[f"{hf_prefix}.proj_out.weight"],
    )
    mask_dec.iou_head_proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])

    hf_prefix = "mask_decoder.pred_obj_score_head"
    transfer_weights(
        "kernel",
        mask_dec.obj_score_proj_in.kernel,
        hf_state_dict[f"{hf_prefix}.proj_in.weight"],
    )
    mask_dec.obj_score_proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])
    for j, hidden_layer in enumerate(mask_dec.obj_score_hidden_layers):
        transfer_weights(
            "kernel",
            hidden_layer.kernel,
            hf_state_dict[f"{hf_prefix}.layers.{j}.weight"],
        )
        hidden_layer.bias.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.bias"])
    transfer_weights(
        "kernel",
        mask_dec.obj_score_proj_out.kernel,
        hf_state_dict[f"{hf_prefix}.proj_out.weight"],
    )
    mask_dec.obj_score_proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])

    print("Weight transfer complete!")

    print("Verifying model equivalence...")
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
    keras_obj = keras_output["object_score_logits"]

    with torch.no_grad():
        hf_input = {
            "pixel_values": torch.from_numpy(test_image.transpose(0, 3, 1, 2)),
            "input_points": torch.from_numpy(test_points),
            "input_labels": torch.from_numpy(test_labels),
            "multimask_output": True,
        }
        hf_output = hf_model(**hf_input)
        hf_masks = hf_output.pred_masks.cpu().numpy()
        hf_iou = hf_output.iou_scores.cpu().numpy()
        hf_obj = hf_output.object_score_logits.cpu().numpy()

    mask_diff = np.max(np.abs(keras_masks - hf_masks))
    iou_diff = np.max(np.abs(keras_iou - hf_iou))
    obj_diff = np.max(np.abs(keras_obj - hf_obj))
    print(f"Max mask diff:          {mask_diff:.6f}")
    print(f"Max IoU diff:           {iou_diff:.6f}")
    print(f"Max object-logit diff:  {obj_diff:.6f}")

    assert mask_diff < 0.5, f"Mask diff too large: {mask_diff}"
    assert iou_diff < 0.05, f"IoU diff too large: {iou_diff}"
    assert obj_diff < 0.05, f"Object-score logit diff too large: {obj_diff}"
    print("Model equivalence verified!")

    model_base = model_config["hf_model_name"].split("/")[-1].replace("-", "_")
    model_filename = model_base + ".weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved as {model_filename}")

    del keras_model, hf_model, hf_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
