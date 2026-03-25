import numpy as np
import torch
from tqdm import tqdm
from transformers import SamModel

from kmodels.models.sam.sam_model import SAM_ViT_Base, SAM_ViT_Huge, SAM_ViT_Large
from kmodels.utils.weight_transfer_torch_to_keras import (
    transfer_nested_layer_weights,
    transfer_weights,
)

vision_encoder_name_mapping = {
    "mlp_lin1": "mlp.lin1",
    "mlp_lin2": "mlp.lin2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

attention_name_mapping = {
    "kernel": "weight",
}

model_configs = [
    {
        "keras_model_cls": SAM_ViT_Base,
        "hf_model_name": "facebook/sam-vit-base",
        "input_shape": (1024, 1024, 3),
    },
    {
        "keras_model_cls": SAM_ViT_Large,
        "hf_model_name": "facebook/sam-vit-large",
        "input_shape": (1024, 1024, 3),
    },
    {
        "keras_model_cls": SAM_ViT_Huge,
        "hf_model_name": "facebook/sam-vit-huge",
        "input_shape": (1024, 1024, 3),
    },
]

for model_config in model_configs:
    print(f"\n{'=' * 60}")
    print(f"Converting {model_config['hf_model_name']}...")
    print(f"{'=' * 60}")

    print(f"Loading HF model: {model_config['hf_model_name']}")
    hf_model = SamModel.from_pretrained(model_config["hf_model_name"]).eval()
    hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

    print("Creating Keras model...")
    keras_model = model_config["keras_model_cls"](
        input_shape=model_config["input_shape"], weights=None
    )

    print("Transferring patch embeddings...")
    patch_conv = keras_model.get_layer("vision_encoder_patch_embed_projection")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf_state_dict["vision_encoder.patch_embed.projection.weight"],
    )
    patch_conv.bias.assign(hf_state_dict["vision_encoder.patch_embed.projection.bias"])

    print("Transferring position embedding...")
    pos_layer = keras_model.get_layer("vision_encoder_pos_embed")
    pos_layer.pos_embed.assign(hf_state_dict["vision_encoder.pos_embed"])

    num_layers = keras_model.vision_num_hidden_layers
    for i in tqdm(range(num_layers), desc="Transferring vision encoder layers"):
        layer = keras_model.get_layer(f"vision_encoder_layers_{i}")
        skipped = transfer_nested_layer_weights(
            layer,
            hf_state_dict,
            f"vision_encoder.layers.{i}",
            name_mapping=vision_encoder_name_mapping,
            skip_paths=["rel_pos"],
        )
        for keras_weight, _ in skipped:
            w_name = keras_weight.path.split("/")[-1]
            hf_key = f"vision_encoder.layers.{i}.attn.{w_name}"
            keras_weight.assign(hf_state_dict[hf_key])

    print("Transferring vision neck...")
    neck_conv1 = keras_model.get_layer("vision_encoder_neck_conv1")
    transfer_weights(
        "conv_kernel",
        neck_conv1.kernel,
        hf_state_dict["vision_encoder.neck.conv1.weight"],
    )

    neck_ln1 = keras_model.get_layer("vision_encoder_neck_layer_norm1")
    neck_ln1.gamma.assign(hf_state_dict["vision_encoder.neck.layer_norm1.weight"])
    neck_ln1.beta.assign(hf_state_dict["vision_encoder.neck.layer_norm1.bias"])

    neck_conv2 = keras_model.get_layer("vision_encoder_neck_conv2")
    transfer_weights(
        "conv_kernel",
        neck_conv2.kernel,
        hf_state_dict["vision_encoder.neck.conv2.weight"],
    )

    neck_ln2 = keras_model.get_layer("vision_encoder_neck_layer_norm2")
    neck_ln2.gamma.assign(hf_state_dict["vision_encoder.neck.layer_norm2.weight"])
    neck_ln2.beta.assign(hf_state_dict["vision_encoder.neck.layer_norm2.bias"])

    print("Transferring shared image embedding...")
    image_pe_layer = keras_model.get_layer("image_positional_embeddings")
    image_pe_layer.shared_embedding.positional_embedding.assign(
        hf_state_dict["shared_image_embedding.positional_embedding"]
    )

    print("Transferring prompt encoder...")
    prompt_enc = keras_model.get_layer("prompt_encoder")
    for i in range(prompt_enc.num_point_embeddings):
        prompt_enc.point_embeddings[i].assign(
            hf_state_dict[f"prompt_encoder.point_embed.{i}.weight"]
        )

    prompt_enc.not_a_point_embed.assign(
        hf_state_dict["prompt_encoder.not_a_point_embed.weight"]
    )
    prompt_enc.no_mask_embed.assign(
        hf_state_dict["prompt_encoder.no_mask_embed.weight"]
    )

    print("Transferring mask decoder...")
    mask_dec = keras_model.get_layer("mask_decoder")

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
            for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                proj = getattr(attn_layer, proj_name)
                transfer_weights(
                    "kernel",
                    proj.kernel,
                    hf_state_dict[f"{hf_prefix}.{attn_suffix}.{proj_name}.weight"],
                )
                proj.bias.assign(
                    hf_state_dict[f"{hf_prefix}.{attn_suffix}.{proj_name}.bias"]
                )

        ln1 = mask_dec.transformer_layer_norm1s[i]
        ln1.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.weight"])
        ln1.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.bias"])

        ln2 = mask_dec.transformer_layer_norm2s[i]
        ln2.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.weight"])
        ln2.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.bias"])

        mlp_lin1 = mask_dec.transformer_mlp_lin1s[i]
        transfer_weights(
            "kernel", mlp_lin1.kernel, hf_state_dict[f"{hf_prefix}.mlp.lin1.weight"]
        )
        mlp_lin1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin1.bias"])

        mlp_lin2 = mask_dec.transformer_mlp_lin2s[i]
        transfer_weights(
            "kernel", mlp_lin2.kernel, hf_state_dict[f"{hf_prefix}.mlp.lin2.weight"]
        )
        mlp_lin2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin2.bias"])

        ln3 = mask_dec.transformer_layer_norm3s[i]
        ln3.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm3.weight"])
        ln3.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm3.bias"])

        ln4 = mask_dec.transformer_layer_norm4s[i]
        ln4.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm4.weight"])
        ln4.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm4.bias"])

    hf_final_attn = "mask_decoder.transformer.final_attn_token_to_image"
    for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        proj = getattr(mask_dec.final_attn_token_to_image, proj_name)
        transfer_weights(
            "kernel",
            proj.kernel,
            hf_state_dict[f"{hf_final_attn}.{proj_name}.weight"],
        )
        proj.bias.assign(hf_state_dict[f"{hf_final_attn}.{proj_name}.bias"])

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

    num_mask_tokens = mask_dec.num_mask_tokens
    for i in range(num_mask_tokens):
        hf_prefix = f"mask_decoder.output_hypernetworks_mlps.{i}"

        proj_in = mask_dec.output_hypernetworks_mlps_proj_ins[i]
        transfer_weights(
            "kernel", proj_in.kernel, hf_state_dict[f"{hf_prefix}.proj_in.weight"]
        )
        proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])

        for j in range(mask_dec._hyper_num_hidden):
            idx = i * mask_dec._hyper_num_hidden + j
            hidden = mask_dec.output_hypernetworks_mlps_hidden_layers[idx]
            transfer_weights(
                "kernel", hidden.kernel, hf_state_dict[f"{hf_prefix}.layers.{j}.weight"]
            )
            hidden.bias.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.bias"])

        proj_out = mask_dec.output_hypernetworks_mlps_proj_outs[i]
        transfer_weights(
            "kernel", proj_out.kernel, hf_state_dict[f"{hf_prefix}.proj_out.weight"]
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
    keras_masks = keras_output["pred_masks"][:, :, 1:]
    keras_iou = keras_output["iou_scores"][:, :, 1:]

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

    mask_diff = np.max(np.abs(keras_masks - hf_masks))
    iou_diff = np.max(np.abs(keras_iou - hf_iou))
    print(f"Max mask diff: {mask_diff:.6f}")
    print(f"Max IoU diff:  {iou_diff:.6f}")

    assert mask_diff < 0.8, f"Mask diff too large: {mask_diff}"
    assert iou_diff < 1e-2, f"IoU diff too large: {iou_diff}"
    print("Model equivalence verified!")

    model_base = model_config["hf_model_name"].split("/")[-1].replace("-", "_")
    if "huge" in model_config["hf_model_name"]:
        model_filename = model_base + ".weights.json"
        keras_model.save_weights(model_filename, max_shard_size=1.5)
    else:
        model_filename = model_base + ".weights.h5"
        keras_model.save_weights(model_filename)
    print(f"Model saved as {model_filename}")

    del keras_model, hf_model, hf_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
