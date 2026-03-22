import numpy as np
import torch
from tqdm import tqdm
from transformers import SamModel

from kmodels.models.sam.sam_model import SAM_ViT_Base, SAM_ViT_Huge, SAM_ViT_Large
from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights

VARIANT_MAP = {
    "base": SAM_ViT_Base,
    "large": SAM_ViT_Large,
    "huge": SAM_ViT_Huge,
}


def _transfer(keras_name, keras_weight, torch_weight):
    """Transfer a single weight using the shared utility."""
    transfer_weights(keras_name, keras_weight, torch_weight)


def _transfer_dense(dense_layer, hf_state_dict, hf_key, keras_name):
    """Transfer kernel and bias for a Dense layer."""
    _transfer(
        f"{keras_name}_kernel",
        dense_layer.kernel,
        hf_state_dict[f"{hf_key}.weight"],
    )
    _transfer(
        f"{keras_name}_bias",
        dense_layer.bias,
        hf_state_dict[f"{hf_key}.bias"],
    )


def _transfer_layernorm(ln_layer, hf_state_dict, hf_key, keras_name):
    """Transfer gamma and beta for a LayerNormalization layer."""
    ln_layer.gamma.assign(hf_state_dict[f"{hf_key}.weight"])
    ln_layer.beta.assign(hf_state_dict[f"{hf_key}.bias"])


def _transfer_conv(conv_layer, hf_state_dict, hf_key, keras_name):
    """Transfer kernel (and optional bias) for a Conv2D/Conv2DTranspose layer."""
    _transfer(
        f"{keras_name}_conv_kernel",
        conv_layer.kernel,
        hf_state_dict[f"{hf_key}.weight"],
    )
    bias_key = f"{hf_key}.bias"
    if bias_key in hf_state_dict:
        _transfer(
            f"{keras_name}_bias",
            conv_layer.bias,
            hf_state_dict[bias_key],
        )


def _transfer_attention(attn_layer, hf_state_dict, hf_prefix, keras_name):
    """Transfer q/k/v/out_proj weights for a SAMTwoWayAttention layer."""
    for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        _transfer_dense(
            getattr(attn_layer, proj_name),
            hf_state_dict,
            f"{hf_prefix}.{proj_name}",
            f"{keras_name}_{proj_name}",
        )


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
    _transfer_conv(
        keras_model.get_layer("vision_encoder_patch_embed_projection"),
        hf_state_dict,
        "vision_encoder.patch_embed.projection",
        "patch_embed",
    )

    # ─── Vision Encoder: Absolute Position Embedding ───
    print("Transferring position embedding...")
    pos_layer = keras_model.get_layer("vision_encoder_pos_embed")
    pos_layer.pos_embed.assign(hf_state_dict["vision_encoder.pos_embed"])

    # ─── Vision Encoder: Transformer Layers ───
    num_layers = keras_model.vision_num_hidden_layers
    for i in tqdm(range(num_layers), desc="Transferring vision encoder layers"):
        hf_prefix = f"vision_encoder.layers.{i}"
        layer = keras_model.get_layer(f"vision_encoder_layers_{i}")

        _transfer_layernorm(
            layer.layer_norm1, hf_state_dict, f"{hf_prefix}.layer_norm1", "ln1"
        )

        # QKV (combined projection)
        _transfer_dense(
            layer.attn.qkv, hf_state_dict, f"{hf_prefix}.attn.qkv", "attn_qkv"
        )
        # Attention output projection
        _transfer_dense(
            layer.attn.proj, hf_state_dict, f"{hf_prefix}.attn.proj", "attn_proj"
        )

        # Relative position embeddings (direct assign, not transposed)
        if layer.attn.use_rel_pos:
            layer.attn.rel_pos_h.assign(hf_state_dict[f"{hf_prefix}.attn.rel_pos_h"])
            layer.attn.rel_pos_w.assign(hf_state_dict[f"{hf_prefix}.attn.rel_pos_w"])

        _transfer_layernorm(
            layer.layer_norm2, hf_state_dict, f"{hf_prefix}.layer_norm2", "ln2"
        )

        # MLP
        _transfer_dense(
            layer.mlp_lin1, hf_state_dict, f"{hf_prefix}.mlp.lin1", "mlp_lin1"
        )
        _transfer_dense(
            layer.mlp_lin2, hf_state_dict, f"{hf_prefix}.mlp.lin2", "mlp_lin2"
        )

    # ─── Vision Encoder: Neck ───
    print("Transferring vision neck...")
    _transfer_conv(
        keras_model.get_layer("vision_encoder_neck_conv1"),
        hf_state_dict,
        "vision_encoder.neck.conv1",
        "neck_conv1",
    )
    _transfer_layernorm(
        keras_model.get_layer("vision_encoder_neck_layer_norm1"),
        hf_state_dict,
        "vision_encoder.neck.layer_norm1",
        "neck_ln1",
    )
    _transfer_conv(
        keras_model.get_layer("vision_encoder_neck_conv2"),
        hf_state_dict,
        "vision_encoder.neck.conv2",
        "neck_conv2",
    )
    _transfer_layernorm(
        keras_model.get_layer("vision_encoder_neck_layer_norm2"),
        hf_state_dict,
        "vision_encoder.neck.layer_norm2",
        "neck_ln2",
    )

    # ─── Shared Image Embedding ───
    print("Transferring shared image embedding...")
    image_pe_layer = keras_model.get_layer("image_positional_embeddings")
    image_pe_layer.shared_embedding.positional_embedding.assign(
        hf_state_dict["shared_image_embedding.positional_embedding"]
    )

    # ─── Prompt Encoder ───
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

    # ─── Mask Decoder ───
    print("Transferring mask decoder...")
    mask_dec = keras_model.get_layer("mask_decoder")

    # Token embeddings (direct assign)
    mask_dec.iou_token.assign(hf_state_dict["mask_decoder.iou_token.weight"])
    mask_dec.mask_tokens.assign(hf_state_dict["mask_decoder.mask_tokens.weight"])

    # Two-way transformer layers
    for i in range(mask_dec.num_hidden_layers):
        hf_prefix = f"mask_decoder.transformer.layers.{i}"

        _transfer_attention(
            mask_dec.transformer_self_attns[i],
            hf_state_dict,
            f"{hf_prefix}.self_attn",
            f"dec_self_attn_{i}",
        )
        _transfer_layernorm(
            mask_dec.transformer_layer_norm1s[i],
            hf_state_dict,
            f"{hf_prefix}.layer_norm1",
            f"dec_ln1_{i}",
        )

        _transfer_attention(
            mask_dec.transformer_cross_attn_token_to_images[i],
            hf_state_dict,
            f"{hf_prefix}.cross_attn_token_to_image",
            f"dec_cross_t2i_{i}",
        )
        _transfer_layernorm(
            mask_dec.transformer_layer_norm2s[i],
            hf_state_dict,
            f"{hf_prefix}.layer_norm2",
            f"dec_ln2_{i}",
        )

        _transfer_dense(
            mask_dec.transformer_mlp_lin1s[i],
            hf_state_dict,
            f"{hf_prefix}.mlp.lin1",
            f"dec_mlp_lin1_{i}",
        )
        _transfer_dense(
            mask_dec.transformer_mlp_lin2s[i],
            hf_state_dict,
            f"{hf_prefix}.mlp.lin2",
            f"dec_mlp_lin2_{i}",
        )
        _transfer_layernorm(
            mask_dec.transformer_layer_norm3s[i],
            hf_state_dict,
            f"{hf_prefix}.layer_norm3",
            f"dec_ln3_{i}",
        )

        _transfer_attention(
            mask_dec.transformer_cross_attn_image_to_tokens[i],
            hf_state_dict,
            f"{hf_prefix}.cross_attn_image_to_token",
            f"dec_cross_i2t_{i}",
        )
        _transfer_layernorm(
            mask_dec.transformer_layer_norm4s[i],
            hf_state_dict,
            f"{hf_prefix}.layer_norm4",
            f"dec_ln4_{i}",
        )

    # Final attention token to image
    _transfer_attention(
        mask_dec.final_attn_token_to_image,
        hf_state_dict,
        "mask_decoder.transformer.final_attn_token_to_image",
        "dec_final_attn",
    )
    _transfer_layernorm(
        mask_dec.layer_norm_final_attn,
        hf_state_dict,
        "mask_decoder.transformer.layer_norm_final_attn",
        "dec_final_ln",
    )

    # Upscale convolutions
    _transfer_conv(
        mask_dec.upscale_conv1,
        hf_state_dict,
        "mask_decoder.upscale_conv1",
        "upscale_conv1",
    )
    _transfer_layernorm(
        mask_dec.upscale_layer_norm,
        hf_state_dict,
        "mask_decoder.upscale_layer_norm",
        "upscale_ln",
    )
    _transfer_conv(
        mask_dec.upscale_conv2,
        hf_state_dict,
        "mask_decoder.upscale_conv2",
        "upscale_conv2",
    )

    # Output hypernetworks MLPs
    num_mask_tokens = mask_dec.num_mask_tokens
    for i in range(num_mask_tokens):
        hf_prefix = f"mask_decoder.output_hypernetworks_mlps.{i}"

        _transfer_dense(
            mask_dec.output_hypernetworks_mlps_proj_ins[i],
            hf_state_dict,
            f"{hf_prefix}.proj_in",
            f"hyper_{i}_proj_in",
        )

        for j in range(mask_dec._hyper_num_hidden):
            idx = i * mask_dec._hyper_num_hidden + j
            _transfer_dense(
                mask_dec.output_hypernetworks_mlps_hidden_layers[idx],
                hf_state_dict,
                f"{hf_prefix}.layers.{j}",
                f"hyper_{i}_hidden_{j}",
            )

        _transfer_dense(
            mask_dec.output_hypernetworks_mlps_proj_outs[i],
            hf_state_dict,
            f"{hf_prefix}.proj_out",
            f"hyper_{i}_proj_out",
        )

    # IoU prediction head
    hf_prefix = "mask_decoder.iou_prediction_head"
    _transfer_dense(
        mask_dec.iou_head_proj_in,
        hf_state_dict,
        f"{hf_prefix}.proj_in",
        "iou_proj_in",
    )
    for j, hidden_layer in enumerate(mask_dec.iou_head_hidden_layers):
        _transfer_dense(
            hidden_layer,
            hf_state_dict,
            f"{hf_prefix}.layers.{j}",
            f"iou_hidden_{j}",
        )
    _transfer_dense(
        mask_dec.iou_head_proj_out,
        hf_state_dict,
        f"{hf_prefix}.proj_out",
        "iou_proj_out",
    )

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
