import numpy as np
import torch
from tqdm import tqdm
from transformers import SamModel

from kmodels.models.sam.sam_model import SAM_ViT_Base, SAM_ViT_Huge, SAM_ViT_Large

VARIANT_MAP = {
    "base": SAM_ViT_Base,
    "large": SAM_ViT_Large,
    "huge": SAM_ViT_Huge,
}


def convert_model(
    hf_model_name="facebook/sam-vit-huge",
    input_shape=(1024, 1024, 3),
    variant="huge",
):
    """Convert HuggingFace SAM weights to Keras format.

    Args:
        hf_model_name: HuggingFace model ID.
        input_shape: Input shape (H, W, C).
        variant: Model variant ("base", "large", or "huge").
    """
    print(f"Loading HF model: {hf_model_name}")
    hf_model = SamModel.from_pretrained(hf_model_name).eval()
    hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

    keras_model_cls = VARIANT_MAP[variant]
    print(f"Creating Keras model ({variant})...")
    keras_model = keras_model_cls(input_shape=input_shape, weights=None)

    # ─── Vision Encoder: Patch Embeddings ───
    print("Transferring patch embeddings...")
    patch_conv = keras_model.get_layer("vision_encoder_patch_embed_projection")
    conv_w = hf_state_dict["vision_encoder.patch_embed.projection.weight"]
    patch_conv.kernel.assign(np.transpose(conv_w, (2, 3, 1, 0)))
    patch_conv.bias.assign(hf_state_dict["vision_encoder.patch_embed.projection.bias"])

    # ─── Vision Encoder: Absolute Position Embedding ───
    print("Transferring position embedding...")
    pos_layer = keras_model.get_layer("vision_encoder_pos_embed")
    pos_emb = hf_state_dict["vision_encoder.pos_embed"]
    pos_layer.pos_embed.assign(pos_emb)

    # ─── Vision Encoder: Transformer Layers ───
    num_layers = keras_model.vision_num_hidden_layers
    for i in tqdm(range(num_layers), desc="Transferring vision encoder layers"):
        hf_prefix = f"vision_encoder.layers.{i}"
        k_prefix = f"vision_encoder_layers_{i}"

        layer = keras_model.get_layer(k_prefix)

        # Layer norm 1
        layer.layer_norm1.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.weight"])
        layer.layer_norm1.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.bias"])

        # Attention QKV
        attn = layer.attn
        qkv_w = hf_state_dict[f"{hf_prefix}.attn.qkv.weight"]
        qkv_b = hf_state_dict[f"{hf_prefix}.attn.qkv.bias"]
        attn.qkv.kernel.assign(qkv_w.T)
        attn.qkv.bias.assign(qkv_b)

        # Attention proj
        proj_w = hf_state_dict[f"{hf_prefix}.attn.proj.weight"]
        proj_b = hf_state_dict[f"{hf_prefix}.attn.proj.bias"]
        attn.proj.kernel.assign(proj_w.T)
        attn.proj.bias.assign(proj_b)

        # Relative position embeddings
        if attn.use_rel_pos:
            attn.rel_pos_h.assign(hf_state_dict[f"{hf_prefix}.attn.rel_pos_h"])
            attn.rel_pos_w.assign(hf_state_dict[f"{hf_prefix}.attn.rel_pos_w"])

        # Layer norm 2
        layer.layer_norm2.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.weight"])
        layer.layer_norm2.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.bias"])

        # MLP (inlined: mlp_lin1, mlp_lin2 are now direct attributes)
        layer.mlp_lin1.kernel.assign(hf_state_dict[f"{hf_prefix}.mlp.lin1.weight"].T)
        layer.mlp_lin1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin1.bias"])
        layer.mlp_lin2.kernel.assign(hf_state_dict[f"{hf_prefix}.mlp.lin2.weight"].T)
        layer.mlp_lin2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin2.bias"])

    # ─── Vision Encoder: Neck ───
    print("Transferring vision neck...")
    neck_conv1 = keras_model.get_layer("vision_encoder_neck_conv1")
    conv1_w = hf_state_dict["vision_encoder.neck.conv1.weight"]
    neck_conv1.kernel.assign(np.transpose(conv1_w, (2, 3, 1, 0)))

    neck_ln1 = keras_model.get_layer("vision_encoder_neck_layer_norm1")
    neck_ln1.gamma.assign(hf_state_dict["vision_encoder.neck.layer_norm1.weight"])
    neck_ln1.beta.assign(hf_state_dict["vision_encoder.neck.layer_norm1.bias"])

    neck_conv2 = keras_model.get_layer("vision_encoder_neck_conv2")
    conv2_w = hf_state_dict["vision_encoder.neck.conv2.weight"]
    neck_conv2.kernel.assign(np.transpose(conv2_w, (2, 3, 1, 0)))

    neck_ln2 = keras_model.get_layer("vision_encoder_neck_layer_norm2")
    neck_ln2.gamma.assign(hf_state_dict["vision_encoder.neck.layer_norm2.weight"])
    neck_ln2.beta.assign(hf_state_dict["vision_encoder.neck.layer_norm2.bias"])

    # ─── Shared Image Embedding ───
    print("Transferring shared image embedding...")
    image_pe_layer = keras_model.get_layer("image_positional_embeddings")
    shared_emb = image_pe_layer.shared_embedding
    shared_emb.positional_embedding.assign(
        hf_state_dict["shared_image_embedding.positional_embedding"]
    )

    # ─── Prompt Encoder ───
    print("Transferring prompt encoder...")
    prompt_enc = keras_model.get_layer("prompt_encoder")

    for i in range(prompt_enc.num_point_embeddings):
        hf_key = f"prompt_encoder.point_embed.{i}.weight"
        prompt_enc.point_embeddings[i].assign(hf_state_dict[hf_key])

    prompt_enc.not_a_point_embed.assign(
        hf_state_dict["prompt_encoder.not_a_point_embed.weight"]
    )
    prompt_enc.no_mask_embed.assign(
        hf_state_dict["prompt_encoder.no_mask_embed.weight"]
    )

    # ─── Mask Decoder ───
    print("Transferring mask decoder...")
    mask_dec = keras_model.get_layer("mask_decoder")

    mask_dec.iou_token.assign(hf_state_dict["mask_decoder.iou_token.weight"])
    mask_dec.mask_tokens.assign(hf_state_dict["mask_decoder.mask_tokens.weight"])

    # Two-way transformer layers (inlined)
    for i in range(mask_dec.num_hidden_layers):
        hf_prefix = f"mask_decoder.transformer.layers.{i}"

        # Self attention
        self_attn = mask_dec.transformer_self_attns[i]
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            hf_key = f"{hf_prefix}.self_attn.{proj_name}"
            dense = getattr(self_attn, proj_name)
            dense.kernel.assign(hf_state_dict[f"{hf_key}.weight"].T)
            dense.bias.assign(hf_state_dict[f"{hf_key}.bias"])

        mask_dec.transformer_layer_norm1s[i].gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm1.weight"]
        )
        mask_dec.transformer_layer_norm1s[i].beta.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm1.bias"]
        )

        # Cross attention token to image
        cross_t2i = mask_dec.transformer_cross_attn_token_to_images[i]
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            hf_key = f"{hf_prefix}.cross_attn_token_to_image.{proj_name}"
            dense = getattr(cross_t2i, proj_name)
            dense.kernel.assign(hf_state_dict[f"{hf_key}.weight"].T)
            dense.bias.assign(hf_state_dict[f"{hf_key}.bias"])

        mask_dec.transformer_layer_norm2s[i].gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm2.weight"]
        )
        mask_dec.transformer_layer_norm2s[i].beta.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm2.bias"]
        )

        # MLP (inlined)
        mask_dec.transformer_mlp_lin1s[i].kernel.assign(
            hf_state_dict[f"{hf_prefix}.mlp.lin1.weight"].T
        )
        mask_dec.transformer_mlp_lin1s[i].bias.assign(
            hf_state_dict[f"{hf_prefix}.mlp.lin1.bias"]
        )
        mask_dec.transformer_mlp_lin2s[i].kernel.assign(
            hf_state_dict[f"{hf_prefix}.mlp.lin2.weight"].T
        )
        mask_dec.transformer_mlp_lin2s[i].bias.assign(
            hf_state_dict[f"{hf_prefix}.mlp.lin2.bias"]
        )

        mask_dec.transformer_layer_norm3s[i].gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm3.weight"]
        )
        mask_dec.transformer_layer_norm3s[i].beta.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm3.bias"]
        )

        # Cross attention image to token
        cross_i2t = mask_dec.transformer_cross_attn_image_to_tokens[i]
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            hf_key = f"{hf_prefix}.cross_attn_image_to_token.{proj_name}"
            dense = getattr(cross_i2t, proj_name)
            dense.kernel.assign(hf_state_dict[f"{hf_key}.weight"].T)
            dense.bias.assign(hf_state_dict[f"{hf_key}.bias"])

        mask_dec.transformer_layer_norm4s[i].gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm4.weight"]
        )
        mask_dec.transformer_layer_norm4s[i].beta.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm4.bias"]
        )

    # Final attention token to image
    final_attn = mask_dec.final_attn_token_to_image
    hf_prefix = "mask_decoder.transformer.final_attn_token_to_image"
    for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        dense = getattr(final_attn, proj_name)
        dense.kernel.assign(hf_state_dict[f"{hf_prefix}.{proj_name}.weight"].T)
        dense.bias.assign(hf_state_dict[f"{hf_prefix}.{proj_name}.bias"])

    mask_dec.layer_norm_final_attn.gamma.assign(
        hf_state_dict["mask_decoder.transformer.layer_norm_final_attn.weight"]
    )
    mask_dec.layer_norm_final_attn.beta.assign(
        hf_state_dict["mask_decoder.transformer.layer_norm_final_attn.bias"]
    )

    # Upscale convs
    upscale_conv1_w = hf_state_dict["mask_decoder.upscale_conv1.weight"]
    mask_dec.upscale_conv1.kernel.assign(np.transpose(upscale_conv1_w, (2, 3, 1, 0)))
    mask_dec.upscale_conv1.bias.assign(hf_state_dict["mask_decoder.upscale_conv1.bias"])

    mask_dec.upscale_layer_norm.gamma.assign(
        hf_state_dict["mask_decoder.upscale_layer_norm.weight"]
    )
    mask_dec.upscale_layer_norm.beta.assign(
        hf_state_dict["mask_decoder.upscale_layer_norm.bias"]
    )

    upscale_conv2_w = hf_state_dict["mask_decoder.upscale_conv2.weight"]
    mask_dec.upscale_conv2.kernel.assign(np.transpose(upscale_conv2_w, (2, 3, 1, 0)))
    mask_dec.upscale_conv2.bias.assign(hf_state_dict["mask_decoder.upscale_conv2.bias"])

    # Output hypernetworks MLPs (inlined)
    num_mask_tokens = mask_dec.num_mask_tokens
    for i in range(num_mask_tokens):
        hf_prefix = f"mask_decoder.output_hypernetworks_mlps.{i}"

        mask_dec.output_hypernetworks_mlps_proj_ins[i].kernel.assign(
            hf_state_dict[f"{hf_prefix}.proj_in.weight"].T
        )
        mask_dec.output_hypernetworks_mlps_proj_ins[i].bias.assign(
            hf_state_dict[f"{hf_prefix}.proj_in.bias"]
        )

        for j in range(mask_dec._hyper_num_hidden):
            idx = i * mask_dec._hyper_num_hidden + j
            mask_dec.output_hypernetworks_mlps_hidden_layers[idx].kernel.assign(
                hf_state_dict[f"{hf_prefix}.layers.{j}.weight"].T
            )
            mask_dec.output_hypernetworks_mlps_hidden_layers[idx].bias.assign(
                hf_state_dict[f"{hf_prefix}.layers.{j}.bias"]
            )

        mask_dec.output_hypernetworks_mlps_proj_outs[i].kernel.assign(
            hf_state_dict[f"{hf_prefix}.proj_out.weight"].T
        )
        mask_dec.output_hypernetworks_mlps_proj_outs[i].bias.assign(
            hf_state_dict[f"{hf_prefix}.proj_out.bias"]
        )

    # IoU prediction head (inlined)
    hf_prefix = "mask_decoder.iou_prediction_head"

    mask_dec.iou_head_proj_in.kernel.assign(
        hf_state_dict[f"{hf_prefix}.proj_in.weight"].T
    )
    mask_dec.iou_head_proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])

    for j, hidden_layer in enumerate(mask_dec.iou_head_hidden_layers):
        hidden_layer.kernel.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.weight"].T)
        hidden_layer.bias.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.bias"])

    mask_dec.iou_head_proj_out.kernel.assign(
        hf_state_dict[f"{hf_prefix}.proj_out.weight"].T
    )
    mask_dec.iou_head_proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])

    print("Weight transfer complete!")

    # ─── Verify Model Equivalence ───
    print("Verifying model equivalence...")
    np.random.seed(42)
    test_image = np.random.rand(1, 1024, 1024, 3).astype(np.float32)
    test_points = np.array([[[[500.0, 500.0]]]], dtype=np.float32)
    test_labels = np.array([[[1]]], dtype=np.int32)

    # Keras forward pass — outputs all 4 masks (1 single + 3 multi)
    keras_output = keras_model.predict(
        {
            "pixel_values": test_image,
            "input_points": test_points,
            "input_labels": test_labels,
        },
        verbose=0,
    )
    # Slice to match HF multimask_output=True: masks[:, :, 1:], iou[:, :, 1:]
    keras_masks = keras_output["pred_masks"][:, :, 1:]
    keras_iou = keras_output["iou_scores"][:, :, 1:]

    # HuggingFace forward pass (multimask_output=True returns 3 masks)
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

    # Save
    model_filename = hf_model_name.split("/")[-1].replace("-", "_") + ".weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved as {model_filename}")

    del keras_model, hf_model, hf_state_dict
    torch.cuda.empty_cache()

    return None


if __name__ == "__main__":
    configs = [
        {
            "hf_model_name": "facebook/sam-vit-base",
            "input_shape": (1024, 1024, 3),
            "variant": "base",
        },
        {
            "hf_model_name": "facebook/sam-vit-large",
            "input_shape": (1024, 1024, 3),
            "variant": "large",
        },
        {
            "hf_model_name": "facebook/sam-vit-huge",
            "input_shape": (1024, 1024, 3),
            "variant": "huge",
        },
    ]

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"Converting {cfg['hf_model_name']}...")
        print(f"{'=' * 60}")
        convert_model(**cfg)
        print()
