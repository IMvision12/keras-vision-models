"""Convert HuggingFace Sam3TrackerModel weights to Keras."""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.models.sam3_tracker.modeling_sam3_tracker import (  # noqa: E402
    Sam3TrackerModel as HFSam3TrackerModel,
)

from kmodels.models.sam3.sam3_model import Sam3  # noqa: E402
from kmodels.models.sam3_tracker.sam3_tracker_model import Sam3Tracker  # noqa: E402
from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN")


def convert():
    print("Loading HF Sam3TrackerModel...")
    hf_model = HFSam3TrackerModel.from_pretrained(
        "facebook/sam3", attn_implementation="eager", token=HF_TOKEN
    ).eval()
    hf = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
    print(f"HF model has {len(hf)} weight tensors")

    # Filter to tracker-specific weights (exclude vision encoder)
    tracker_keys = [k for k in hf if not k.startswith("vision_encoder.")]
    print(f"Tracker-specific weights: {len(tracker_keys)}")

    print("\nCreating Keras models...")
    sam3 = Sam3(input_shape=(1008, 1008, 3), weights=None)
    sam3.load_weights("sam3.weights.h5")
    tracker = Sam3Tracker(sam3_model=sam3, weights=None)

    # ── Shared image embedding ──────────────────────────────────
    print("Transferring shared_image_embedding...")
    tracker.shared_image_embedding.positional_embedding.assign(
        hf["shared_image_embedding.positional_embedding"]
    )

    # ── Prompt encoder ──────────────────────────────────────────
    print("Transferring prompt encoder...")
    pe = tracker.prompt_encoder

    pe.shared_embedding.positional_embedding.assign(
        hf["prompt_encoder.shared_embedding.positional_embedding"]
    )
    pe.point_embed.weights[0].assign(hf["prompt_encoder.point_embed.weight"])
    pe.not_a_point_embed.weights[0].assign(
        hf["prompt_encoder.not_a_point_embed.weight"]
    )
    pe.no_mask_embed.weights[0].assign(hf["prompt_encoder.no_mask_embed.weight"])

    # Mask embedding (downsampler)
    me = pe.mask_embed
    for i, name in enumerate(["conv1", "conv2", "conv3"]):
        conv = getattr(me, name)
        transfer_weights(
            "conv_kernel", conv.kernel, hf[f"prompt_encoder.mask_embed.{name}.weight"]
        )
        conv.bias.assign(hf[f"prompt_encoder.mask_embed.{name}.bias"])
    me.layer_norm1.gamma.assign(hf["prompt_encoder.mask_embed.layer_norm1.weight"])
    me.layer_norm1.beta.assign(hf["prompt_encoder.mask_embed.layer_norm1.bias"])
    me.layer_norm2.gamma.assign(hf["prompt_encoder.mask_embed.layer_norm2.weight"])
    me.layer_norm2.beta.assign(hf["prompt_encoder.mask_embed.layer_norm2.bias"])

    # ── Mask decoder ────────────────────────────────────────────
    print("Transferring mask decoder...")
    md = tracker.mask_decoder

    md.iou_token.weights[0].assign(hf["mask_decoder.iou_token.weight"])
    md.mask_tokens.weights[0].assign(hf["mask_decoder.mask_tokens.weight"])
    md.obj_score_token.weights[0].assign(hf["mask_decoder.obj_score_token.weight"])

    # Upscaling convolutions
    transfer_weights(
        "conv_kernel", md.upscale_conv1.kernel, hf["mask_decoder.upscale_conv1.weight"]
    )
    md.upscale_conv1.bias.assign(hf["mask_decoder.upscale_conv1.bias"])
    transfer_weights(
        "conv_kernel", md.upscale_conv2.kernel, hf["mask_decoder.upscale_conv2.weight"]
    )
    md.upscale_conv2.bias.assign(hf["mask_decoder.upscale_conv2.bias"])
    md.upscale_layer_norm.gamma.assign(hf["mask_decoder.upscale_layer_norm.weight"])
    md.upscale_layer_norm.beta.assign(hf["mask_decoder.upscale_layer_norm.bias"])

    # conv_s0, conv_s1
    transfer_weights(
        "conv_kernel", md.conv_s0.kernel, hf["mask_decoder.conv_s0.weight"]
    )
    md.conv_s0.bias.assign(hf["mask_decoder.conv_s0.bias"])
    transfer_weights(
        "conv_kernel", md.conv_s1.kernel, hf["mask_decoder.conv_s1.weight"]
    )
    md.conv_s1.bias.assign(hf["mask_decoder.conv_s1.bias"])

    # Output hypernetwork MLPs
    for i in range(4):
        mlp = md.output_hypernetworks_mlps[i]
        hp = f"mask_decoder.output_hypernetworks_mlps.{i}"
        transfer_weights("kernel", mlp.proj_in.kernel, hf[f"{hp}.proj_in.weight"])
        mlp.proj_in.bias.assign(hf[f"{hp}.proj_in.bias"])
        for j, layer in enumerate(mlp.hidden_layers):
            transfer_weights("kernel", layer.kernel, hf[f"{hp}.layers.{j}.weight"])
            layer.bias.assign(hf[f"{hp}.layers.{j}.bias"])
        transfer_weights("kernel", mlp.proj_out.kernel, hf[f"{hp}.proj_out.weight"])
        mlp.proj_out.bias.assign(hf[f"{hp}.proj_out.bias"])

    # IoU prediction head
    iou = md.iou_prediction_head
    transfer_weights(
        "kernel",
        iou.proj_in.kernel,
        hf["mask_decoder.iou_prediction_head.proj_in.weight"],
    )
    iou.proj_in.bias.assign(hf["mask_decoder.iou_prediction_head.proj_in.bias"])
    for j, layer in enumerate(iou.hidden_layers):
        transfer_weights(
            "kernel",
            layer.kernel,
            hf[f"mask_decoder.iou_prediction_head.layers.{j}.weight"],
        )
        layer.bias.assign(hf[f"mask_decoder.iou_prediction_head.layers.{j}.bias"])
    transfer_weights(
        "kernel",
        iou.proj_out.kernel,
        hf["mask_decoder.iou_prediction_head.proj_out.weight"],
    )
    iou.proj_out.bias.assign(hf["mask_decoder.iou_prediction_head.proj_out.bias"])

    # Object score head
    obj = md.pred_obj_score_head
    transfer_weights(
        "kernel",
        obj.proj_in.kernel,
        hf["mask_decoder.pred_obj_score_head.proj_in.weight"],
    )
    obj.proj_in.bias.assign(hf["mask_decoder.pred_obj_score_head.proj_in.bias"])
    for j, layer in enumerate(obj.hidden_layers):
        transfer_weights(
            "kernel",
            layer.kernel,
            hf[f"mask_decoder.pred_obj_score_head.layers.{j}.weight"],
        )
        layer.bias.assign(hf[f"mask_decoder.pred_obj_score_head.layers.{j}.bias"])
    transfer_weights(
        "kernel",
        obj.proj_out.kernel,
        hf["mask_decoder.pred_obj_score_head.proj_out.weight"],
    )
    obj.proj_out.bias.assign(hf["mask_decoder.pred_obj_score_head.proj_out.bias"])

    # Two-way transformer
    tw = md.transformer
    for i in range(2):
        block = tw.transformer_layers[i]
        bp = f"mask_decoder.transformer.layers.{i}"

        for attn_name in [
            "self_attn",
            "cross_attn_token_to_image",
            "cross_attn_image_to_token",
        ]:
            attn = getattr(block, attn_name)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                transfer_weights(
                    "kernel",
                    getattr(attn, proj).kernel,
                    hf[f"{bp}.{attn_name}.{proj}.weight"],
                )
                getattr(attn, proj).bias.assign(hf[f"{bp}.{attn_name}.{proj}.bias"])

        for ln_name in ["layer_norm1", "layer_norm2", "layer_norm3", "layer_norm4"]:
            ln = getattr(block, ln_name)
            ln.gamma.assign(hf[f"{bp}.{ln_name}.weight"])
            ln.beta.assign(hf[f"{bp}.{ln_name}.bias"])

        # MLP
        mlp = block.mlp
        transfer_weights("kernel", mlp.proj_in.kernel, hf[f"{bp}.mlp.proj_in.weight"])
        mlp.proj_in.bias.assign(hf[f"{bp}.mlp.proj_in.bias"])
        transfer_weights("kernel", mlp.proj_out.kernel, hf[f"{bp}.mlp.proj_out.weight"])
        mlp.proj_out.bias.assign(hf[f"{bp}.mlp.proj_out.bias"])

    # Final attention
    fa = tw.final_attn_token_to_image
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        transfer_weights(
            "kernel",
            getattr(fa, proj).kernel,
            hf[f"mask_decoder.transformer.final_attn_token_to_image.{proj}.weight"],
        )
        getattr(fa, proj).bias.assign(
            hf[f"mask_decoder.transformer.final_attn_token_to_image.{proj}.bias"]
        )
    tw.layer_norm_final_attn.gamma.assign(
        hf["mask_decoder.transformer.layer_norm_final_attn.weight"]
    )
    tw.layer_norm_final_attn.beta.assign(
        hf["mask_decoder.transformer.layer_norm_final_attn.bias"]
    )

    # ── No memory embedding ─────────────────────────────────────
    tracker.no_memory_embedding.assign(hf["no_memory_embedding"])

    print(f"\nTracker params: {sum(w.numpy().size for w in tracker.weights):,}")

    # ── Equivalence test ────────────────────────────────────────
    print("\nRunning equivalence test...")
    import urllib.request

    from PIL import Image

    urllib.request.urlretrieve(
        "http://images.cocodataset.org/val2017/000000039769.jpg", "test_image.jpg"
    )
    img = Image.open("test_image.jpg").convert("RGB")
    from kmodels.models.sam3.sam3_processor import preprocess_image

    pixel_values_keras, _ = preprocess_image(img)
    # Use HF pixel values for fair comparison
    from transformers.models.sam3_tracker.processing_sam3_tracker import (
        Sam3TrackerProcessor,
    )

    hf_proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3", token=HF_TOKEN)
    hf_inputs = hf_proc(
        images=img,
        input_points=[[[[300, 250]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    pv_hf = hf_inputs["pixel_values"]
    pv_keras = pv_hf.permute(0, 2, 3, 1).numpy()
    points_hf = hf_inputs["input_points"]
    labels_hf = hf_inputs["input_labels"]

    with torch.no_grad():
        hf_out = hf_model(
            pixel_values=pv_hf,
            input_points=points_hf,
            input_labels=labels_hf,
            multimask_output=True,
        )

    # Keras
    from keras import ops

    with torch.no_grad():
        k_out = tracker(
            pixel_values=pv_keras,
            input_points=points_hf.numpy(),
            input_labels=labels_hf.numpy().astype(np.int32),
            multimask_output=True,
        )

    hf_masks = hf_out.pred_masks.cpu().numpy()
    k_masks = ops.convert_to_numpy(k_out["pred_masks"])
    hf_iou = hf_out.iou_scores.cpu().numpy()
    k_iou = ops.convert_to_numpy(k_out["iou_scores"])
    hf_obj = hf_out.object_score_logits.cpu().numpy()
    k_obj = ops.convert_to_numpy(k_out["object_score_logits"])

    print(f"mask_diff:   {np.max(np.abs(hf_masks - k_masks)):.6e}")
    print(f"iou_diff:    {np.max(np.abs(hf_iou - k_iou)):.6e}")
    print(f"obj_diff:    {np.max(np.abs(hf_obj - k_obj)):.6e}")
    print(f"HF iou:      {hf_iou.flatten()}")
    print(f"Keras iou:   {k_iou.flatten()}")

    # Save tracker weights (tracker-specific only, not vision encoder)
    # We save by creating a temporary model without the sam3_model reference
    tracker.save_weights("sam3_tracker.weights.h5")
    print("\nTracker weights saved: sam3_tracker.weights.h5")


if __name__ == "__main__":
    convert()
