import numpy as np
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
    patch_embed = keras_model.get_layer("vision_encoder_patch_embed")
    conv_w = hf_state_dict["vision_encoder.patch_embed.projection.weight"]
    patch_embed.projection.kernel.assign(np.transpose(conv_w, (2, 3, 1, 0)))
    patch_embed.projection.bias.assign(
        hf_state_dict["vision_encoder.patch_embed.projection.bias"]
    )

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

        # MLP
        mlp = layer.mlp
        mlp.lin1.kernel.assign(hf_state_dict[f"{hf_prefix}.mlp.lin1.weight"].T)
        mlp.lin1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin1.bias"])
        mlp.lin2.kernel.assign(hf_state_dict[f"{hf_prefix}.mlp.lin2.weight"].T)
        mlp.lin2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin2.bias"])

    # ─── Vision Encoder: Neck ───
    print("Transferring vision neck...")
    neck = keras_model.get_layer("vision_encoder_neck")

    conv1_w = hf_state_dict["vision_encoder.neck.conv1.weight"]
    neck.conv1.kernel.assign(np.transpose(conv1_w, (2, 3, 1, 0)))

    neck.layer_norm1.gamma.assign(
        hf_state_dict["vision_encoder.neck.layer_norm1.weight"]
    )
    neck.layer_norm1.beta.assign(hf_state_dict["vision_encoder.neck.layer_norm1.bias"])

    conv2_w = hf_state_dict["vision_encoder.neck.conv2.weight"]
    neck.conv2.kernel.assign(np.transpose(conv2_w, (2, 3, 1, 0)))

    neck.layer_norm2.gamma.assign(
        hf_state_dict["vision_encoder.neck.layer_norm2.weight"]
    )
    neck.layer_norm2.beta.assign(hf_state_dict["vision_encoder.neck.layer_norm2.bias"])

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

    # Two-way transformer layers
    for i in range(mask_dec.num_hidden_layers):
        hf_prefix = f"mask_decoder.transformer.layers.{i}"
        tf_layer = mask_dec.transformer_layers[i]

        # Self attention
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            hf_key = f"{hf_prefix}.self_attn.{proj_name}"
            dense = getattr(tf_layer.self_attn, proj_name)
            dense.kernel.assign(hf_state_dict[f"{hf_key}.weight"].T)
            dense.bias.assign(hf_state_dict[f"{hf_key}.bias"])

        tf_layer.layer_norm1.gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm1.weight"]
        )
        tf_layer.layer_norm1.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.bias"])

        # Cross attention token to image
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            hf_key = f"{hf_prefix}.cross_attn_token_to_image.{proj_name}"
            dense = getattr(tf_layer.cross_attn_token_to_image, proj_name)
            dense.kernel.assign(hf_state_dict[f"{hf_key}.weight"].T)
            dense.bias.assign(hf_state_dict[f"{hf_key}.bias"])

        tf_layer.layer_norm2.gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm2.weight"]
        )
        tf_layer.layer_norm2.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.bias"])

        # MLP
        tf_layer.mlp_lin1.kernel.assign(hf_state_dict[f"{hf_prefix}.mlp.lin1.weight"].T)
        tf_layer.mlp_lin1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin1.bias"])
        tf_layer.mlp_lin2.kernel.assign(hf_state_dict[f"{hf_prefix}.mlp.lin2.weight"].T)
        tf_layer.mlp_lin2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin2.bias"])

        tf_layer.layer_norm3.gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm3.weight"]
        )
        tf_layer.layer_norm3.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm3.bias"])

        # Cross attention image to token
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            hf_key = f"{hf_prefix}.cross_attn_image_to_token.{proj_name}"
            dense = getattr(tf_layer.cross_attn_image_to_token, proj_name)
            dense.kernel.assign(hf_state_dict[f"{hf_key}.weight"].T)
            dense.bias.assign(hf_state_dict[f"{hf_key}.bias"])

        tf_layer.layer_norm4.gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_norm4.weight"]
        )
        tf_layer.layer_norm4.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm4.bias"])

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

    # Output hypernetworks MLPs
    num_mask_tokens = mask_dec.num_mask_tokens
    for i in range(num_mask_tokens):
        mlp = mask_dec.output_hypernetworks_mlps[i]
        hf_prefix = f"mask_decoder.output_hypernetworks_mlps.{i}"

        mlp.proj_in.kernel.assign(hf_state_dict[f"{hf_prefix}.proj_in.weight"].T)
        mlp.proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])

        for j, hidden_layer in enumerate(mlp.hidden_layers):
            hidden_layer.kernel.assign(
                hf_state_dict[f"{hf_prefix}.layers.{j}.weight"].T
            )
            hidden_layer.bias.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.bias"])

        mlp.proj_out.kernel.assign(hf_state_dict[f"{hf_prefix}.proj_out.weight"].T)
        mlp.proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])

    # IoU prediction head
    iou_head = mask_dec.iou_prediction_head
    hf_prefix = "mask_decoder.iou_prediction_head"

    iou_head.proj_in.kernel.assign(hf_state_dict[f"{hf_prefix}.proj_in.weight"].T)
    iou_head.proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])

    for j, hidden_layer in enumerate(iou_head.hidden_layers):
        hidden_layer.kernel.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.weight"].T)
        hidden_layer.bias.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.bias"])

    iou_head.proj_out.kernel.assign(hf_state_dict[f"{hf_prefix}.proj_out.weight"].T)
    iou_head.proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])

    print("Weight transfer complete!")

    # Save
    model_filename = hf_model_name.split("/")[-1].replace("-", "_") + ".weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved as {model_filename}")

    return keras_model


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
