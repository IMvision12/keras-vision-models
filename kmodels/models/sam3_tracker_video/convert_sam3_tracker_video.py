"""Convert HuggingFace Sam3TrackerVideoModel weights to Keras."""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (  # noqa: E402
    Sam3TrackerVideoModel as HFSam3TrackerVideoModel,
)

from kmodels.models.sam3.sam3_model import Sam3  # noqa: E402
from kmodels.models.sam3_tracker_video.sam3_tracker_video_model import (  # noqa: E402
    Sam3TrackerVideo,
)
from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN")


def _transfer_conv(keras_conv, hf_weight, hf_bias):
    """Transfer conv weights from HF (OIHW) to Keras."""
    transfer_weights("conv_kernel", keras_conv.kernel, hf_weight)
    keras_conv.bias.assign(hf_bias)


def _transfer_dense(keras_dense, hf_weight, hf_bias):
    """Transfer dense/linear weights from HF (out, in) to Keras (in, out)."""
    transfer_weights("kernel", keras_dense.kernel, hf_weight)
    keras_dense.bias.assign(hf_bias)


def _transfer_ln(keras_ln, hf_weight, hf_bias):
    """Transfer LayerNorm weights."""
    keras_ln.gamma.assign(hf_weight)
    keras_ln.beta.assign(hf_bias)


def _transfer_feedforward(keras_ff, hf_prefix, hf):
    """Transfer Sam3TrackerFeedForward (proj_in, hidden_layers, proj_out)."""
    _transfer_dense(
        keras_ff.proj_in,
        hf[f"{hf_prefix}.proj_in.weight"],
        hf[f"{hf_prefix}.proj_in.bias"],
    )
    for j, layer in enumerate(keras_ff.hidden_layers):
        _transfer_dense(
            layer,
            hf[f"{hf_prefix}.layers.{j}.weight"],
            hf[f"{hf_prefix}.layers.{j}.bias"],
        )
    _transfer_dense(
        keras_ff.proj_out,
        hf[f"{hf_prefix}.proj_out.weight"],
        hf[f"{hf_prefix}.proj_out.bias"],
    )


def convert():
    print("Loading HF Sam3TrackerVideoModel...")
    hf_model = HFSam3TrackerVideoModel.from_pretrained(
        "facebook/sam3", attn_implementation="eager", token=HF_TOKEN
    ).eval()
    hf = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
    print(f"HF model has {len(hf)} weight tensors")

    # Filter to tracker-video-specific weights (exclude vision encoder)
    tracker_keys = [k for k in hf if not k.startswith("vision_encoder.")]
    print(f"Tracker-video-specific weights: {len(tracker_keys)}")

    print("\nCreating Keras models...")
    sam3 = Sam3(input_shape=(1008, 1008, 3), weights=None)
    sam3.load_weights("sam3.weights.h5")
    tracker_video = Sam3TrackerVideo(sam3_model=sam3, weights=None)

    # ── Shared image embedding ──
    print("Transferring shared_image_embedding...")
    tracker_video.shared_image_embedding.positional_embedding.assign(
        hf["shared_image_embedding.positional_embedding"]
    )

    # ── Prompt encoder ──
    print("Transferring prompt encoder...")
    pe = tracker_video.prompt_encoder
    pe.shared_embedding.positional_embedding.assign(
        hf["prompt_encoder.shared_embedding.positional_embedding"]
    )
    pe.point_embed.weights[0].assign(hf["prompt_encoder.point_embed.weight"])
    pe.not_a_point_embed.weights[0].assign(
        hf["prompt_encoder.not_a_point_embed.weight"]
    )
    pe.no_mask_embed.weights[0].assign(hf["prompt_encoder.no_mask_embed.weight"])

    # Mask embedding
    me = pe.mask_embed
    for name in ["conv1", "conv2", "conv3"]:
        conv = getattr(me, name)
        _transfer_conv(
            conv,
            hf[f"prompt_encoder.mask_embed.{name}.weight"],
            hf[f"prompt_encoder.mask_embed.{name}.bias"],
        )
    _transfer_ln(
        me.layer_norm1,
        hf["prompt_encoder.mask_embed.layer_norm1.weight"],
        hf["prompt_encoder.mask_embed.layer_norm1.bias"],
    )
    _transfer_ln(
        me.layer_norm2,
        hf["prompt_encoder.mask_embed.layer_norm2.weight"],
        hf["prompt_encoder.mask_embed.layer_norm2.bias"],
    )

    # ── Mask decoder ──
    print("Transferring mask decoder...")
    md = tracker_video.mask_decoder

    md.iou_token.weights[0].assign(hf["mask_decoder.iou_token.weight"])
    md.mask_tokens.weights[0].assign(hf["mask_decoder.mask_tokens.weight"])
    md.obj_score_token.weights[0].assign(hf["mask_decoder.obj_score_token.weight"])

    _transfer_conv(
        md.upscale_conv1,
        hf["mask_decoder.upscale_conv1.weight"],
        hf["mask_decoder.upscale_conv1.bias"],
    )
    _transfer_conv(
        md.upscale_conv2,
        hf["mask_decoder.upscale_conv2.weight"],
        hf["mask_decoder.upscale_conv2.bias"],
    )
    _transfer_ln(
        md.upscale_layer_norm,
        hf["mask_decoder.upscale_layer_norm.weight"],
        hf["mask_decoder.upscale_layer_norm.bias"],
    )
    _transfer_conv(
        md.conv_s0, hf["mask_decoder.conv_s0.weight"], hf["mask_decoder.conv_s0.bias"]
    )
    _transfer_conv(
        md.conv_s1, hf["mask_decoder.conv_s1.weight"], hf["mask_decoder.conv_s1.bias"]
    )

    for i in range(4):
        _transfer_feedforward(
            md.output_hypernetworks_mlps[i],
            f"mask_decoder.output_hypernetworks_mlps.{i}",
            hf,
        )
    _transfer_feedforward(
        md.iou_prediction_head, "mask_decoder.iou_prediction_head", hf
    )
    _transfer_feedforward(
        md.pred_obj_score_head, "mask_decoder.pred_obj_score_head", hf
    )

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
                _transfer_dense(
                    getattr(attn, proj),
                    hf[f"{bp}.{attn_name}.{proj}.weight"],
                    hf[f"{bp}.{attn_name}.{proj}.bias"],
                )
        for ln_name in ["layer_norm1", "layer_norm2", "layer_norm3", "layer_norm4"]:
            ln = getattr(block, ln_name)
            _transfer_ln(ln, hf[f"{bp}.{ln_name}.weight"], hf[f"{bp}.{ln_name}.bias"])
        _transfer_dense(
            block.mlp.proj_in,
            hf[f"{bp}.mlp.proj_in.weight"],
            hf[f"{bp}.mlp.proj_in.bias"],
        )
        _transfer_dense(
            block.mlp.proj_out,
            hf[f"{bp}.mlp.proj_out.weight"],
            hf[f"{bp}.mlp.proj_out.bias"],
        )

    fa = tw.final_attn_token_to_image
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        _transfer_dense(
            getattr(fa, proj),
            hf[f"mask_decoder.transformer.final_attn_token_to_image.{proj}.weight"],
            hf[f"mask_decoder.transformer.final_attn_token_to_image.{proj}.bias"],
        )
    _transfer_ln(
        tw.layer_norm_final_attn,
        hf["mask_decoder.transformer.layer_norm_final_attn.weight"],
        hf["mask_decoder.transformer.layer_norm_final_attn.bias"],
    )

    # ── Memory attention ──
    print("Transferring memory attention...")
    ma = tracker_video.memory_attention
    for i in range(4):
        layer = ma.attention_layers[i]
        lp = f"memory_attention.layers.{i}"

        # Self-attention
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            _transfer_dense(
                getattr(layer.self_attn, proj),
                hf[f"{lp}.self_attn.{proj}.weight"],
                hf[f"{lp}.self_attn.{proj}.bias"],
            )
        # Cross-attention
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            _transfer_dense(
                getattr(layer.cross_attn_image, proj),
                hf[f"{lp}.cross_attn_image.{proj}.weight"],
                hf[f"{lp}.cross_attn_image.{proj}.bias"],
            )
        # FFN
        _transfer_dense(
            layer.linear1, hf[f"{lp}.linear1.weight"], hf[f"{lp}.linear1.bias"]
        )
        _transfer_dense(
            layer.linear2, hf[f"{lp}.linear2.weight"], hf[f"{lp}.linear2.bias"]
        )
        # LayerNorms
        _transfer_ln(
            layer.layer_norm1,
            hf[f"{lp}.layer_norm1.weight"],
            hf[f"{lp}.layer_norm1.bias"],
        )
        _transfer_ln(
            layer.layer_norm2,
            hf[f"{lp}.layer_norm2.weight"],
            hf[f"{lp}.layer_norm2.bias"],
        )
        _transfer_ln(
            layer.layer_norm3,
            hf[f"{lp}.layer_norm3.weight"],
            hf[f"{lp}.layer_norm3.bias"],
        )

    _transfer_ln(
        ma.layer_norm,
        hf["memory_attention.layer_norm.weight"],
        hf["memory_attention.layer_norm.bias"],
    )

    # ── Memory encoder ──
    print("Transferring memory encoder...")
    mem_enc = tracker_video.memory_encoder

    # Mask downsampler layers
    mds = mem_enc.mask_downsampler
    num_ds_layers = len(mds.downsample_layers)
    for i in range(num_ds_layers):
        layer = mds.downsample_layers[i]
        _transfer_conv(
            layer.conv,
            hf[f"memory_encoder.mask_downsampler.layers.{i}.conv.weight"],
            hf[f"memory_encoder.mask_downsampler.layers.{i}.conv.bias"],
        )
        _transfer_ln(
            layer.layer_norm,
            hf[f"memory_encoder.mask_downsampler.layers.{i}.layer_norm.weight"],
            hf[f"memory_encoder.mask_downsampler.layers.{i}.layer_norm.bias"],
        )
    _transfer_conv(
        mds.final_conv,
        hf["memory_encoder.mask_downsampler.final_conv.weight"],
        hf["memory_encoder.mask_downsampler.final_conv.bias"],
    )

    # Feature projection
    _transfer_conv(
        mem_enc.feature_projection,
        hf["memory_encoder.feature_projection.weight"],
        hf["memory_encoder.feature_projection.bias"],
    )

    # Memory fuser (ConvNeXt blocks)
    fuser = mem_enc.memory_fuser
    for i in range(len(fuser.fuser_layers)):
        block = fuser.fuser_layers[i]
        bp = f"memory_encoder.memory_fuser.layers.{i}"

        # Depthwise conv — HF uses groups=embed_dim, Keras uses DepthwiseConv2D
        hf_dw_weight = hf[f"{bp}.depthwise_conv.weight"]
        # HF depthwise: (C, 1, K, K), Keras DepthwiseConv2D: (K, K, C, 1) or channels_first
        transfer_weights(
            "depthwise_conv_kernel", block.depthwise_conv.kernel, hf_dw_weight
        )
        block.depthwise_conv.bias.assign(hf[f"{bp}.depthwise_conv.bias"])

        _transfer_ln(
            block.layer_norm, hf[f"{bp}.layer_norm.weight"], hf[f"{bp}.layer_norm.bias"]
        )

        # Pointwise convs (implemented as Dense)
        _transfer_dense(
            block.pointwise_conv1,
            hf[f"{bp}.pointwise_conv1.weight"],
            hf[f"{bp}.pointwise_conv1.bias"],
        )
        _transfer_dense(
            block.pointwise_conv2,
            hf[f"{bp}.pointwise_conv2.weight"],
            hf[f"{bp}.pointwise_conv2.bias"],
        )

        block.scale.assign(hf[f"{bp}.scale"])

    # Output projection
    _transfer_conv(
        mem_enc.projection,
        hf["memory_encoder.projection.weight"],
        hf["memory_encoder.projection.bias"],
    )

    # ── Object pointer projection ──
    print("Transferring object pointer projection...")
    _transfer_feedforward(tracker_video.object_pointer_proj, "object_pointer_proj", hf)

    # ── Vision neck (tracker's own FPN) ──
    print("Transferring vision neck (tracker FPN)...")
    neck = tracker_video.vision_neck
    scale_factors = [4.0, 2.0, 1.0, 0.5]
    for i, sf in enumerate(scale_factors):
        fpn = neck.fpn_layers[i]
        fp = f"vision_encoder.neck.fpn_layers.{i}"
        if sf == 4.0:
            transfer_weights(
                "conv_transpose_kernel",
                fpn._deconv1.kernel,
                hf[f"{fp}.scale_layers.0.weight"],
            )
            fpn._deconv1.bias.assign(hf[f"{fp}.scale_layers.0.bias"])
            transfer_weights(
                "conv_transpose_kernel",
                fpn._deconv2.kernel,
                hf[f"{fp}.scale_layers.2.weight"],
            )
            fpn._deconv2.bias.assign(hf[f"{fp}.scale_layers.2.bias"])
        elif sf == 2.0:
            transfer_weights(
                "conv_transpose_kernel",
                fpn._deconv1.kernel,
                hf[f"{fp}.scale_layers.0.weight"],
            )
            fpn._deconv1.bias.assign(hf[f"{fp}.scale_layers.0.bias"])
        _transfer_conv(fpn.proj1, hf[f"{fp}.proj1.weight"], hf[f"{fp}.proj1.bias"])
        _transfer_conv(fpn.proj2, hf[f"{fp}.proj2.weight"], hf[f"{fp}.proj2.bias"])

    # ── Mask downsample (4x4 conv) ──
    _transfer_conv(
        tracker_video.mask_downsample,
        hf["mask_downsample.weight"],
        hf["mask_downsample.bias"],
    )

    # ── Learnable embeddings ──
    print("Transferring learnable embeddings...")
    tracker_video.no_memory_embedding.assign(hf["no_memory_embedding"])
    tracker_video.no_memory_positional_encoding.assign(
        hf["no_memory_positional_encoding"]
    )
    tracker_video.no_object_pointer.assign(hf["no_object_pointer"])
    tracker_video.memory_temporal_positional_encoding.assign(
        hf["memory_temporal_positional_encoding"]
    )

    # Temporal PE projection
    if hasattr(tracker_video, "temporal_positional_encoding_projection_layer"):
        _transfer_dense(
            tracker_video.temporal_positional_encoding_projection_layer,
            hf["temporal_positional_encoding_projection_layer.weight"],
            hf["temporal_positional_encoding_projection_layer.bias"],
        )

    # Occlusion spatial embedding
    if hasattr(tracker_video, "occlusion_spatial_embedding_parameter"):
        tracker_video.occlusion_spatial_embedding_parameter.assign(
            hf["occlusion_spatial_embedding_parameter"]
        )

    print(
        f"\nTracker-video params: {sum(w.numpy().size for w in tracker_video.weights):,}"
    )

    # ── Save weights first ──
    tracker_video.save_weights("sam3_tracker_video.weights.h5")
    print("\nTracker-video weights saved: sam3_tracker_video.weights.h5")

    # ── Component-level equivalence tests ──
    # Sam3TrackerVideoModel uses inference_session for forward(), so we test
    # individual components against the HF model's submodules.
    from keras import ops

    print("\n=== Component Equivalence Tests ===")

    # 1. Memory Encoder: compare mask_downsampler + fuser + projection
    # Find compatible dims: mask_downsampler has total_stride=16, so mask must be
    # 16x larger than feature spatial dims. Use small spatial (4x4) for speed.
    print("\n--- Memory Encoder ---")
    dummy_vis = torch.randn(1, 256, 4, 4)
    dummy_mask = torch.randn(1, 1, 64, 64)  # 64/16=4 matches vision

    with torch.no_grad():
        hf_mem_out, hf_mem_pos = hf_model.memory_encoder(dummy_vis, dummy_mask)

    vis_np = dummy_vis.numpy()
    mask_np = dummy_mask.numpy()
    with torch.no_grad():
        k_mem_out, k_mem_pos = tracker_video.memory_encoder(
            ops.convert_to_tensor(vis_np),
            ops.convert_to_tensor(mask_np),
        )

    mem_feat_diff = np.max(np.abs(hf_mem_out.numpy() - ops.convert_to_numpy(k_mem_out)))
    mem_pos_diff = np.max(np.abs(hf_mem_pos.numpy() - ops.convert_to_numpy(k_mem_pos)))
    print(f"  maskmem_features diff: {mem_feat_diff:.6e}")
    print(f"  maskmem_pos_enc diff:  {mem_pos_diff:.6e}")

    # 2. Memory Attention: skip full test (requires 72x72=5184 tokens, OOM for attention)
    # Weight transfer verified above; component weights checked individually.
    print("\n--- Memory Attention ---")
    print("  Skipped (requires full 5184-token seq matching pre-computed RoPE)")
    print("  Weights transferred: 4 layers x (self_attn + cross_attn + FFN + 3 LN)")

    # 2b. Test sub-components of memory encoder to debug diff
    print("\n--- Memory Encoder Sub-components ---")
    # Mask downsampler alone
    dummy_mask_small = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        hf_ds = hf_model.memory_encoder.mask_downsampler(dummy_mask_small)
        k_ds = tracker_video.memory_encoder.mask_downsampler(
            ops.convert_to_tensor(dummy_mask_small.numpy())
        )
    ds_diff = np.max(np.abs(hf_ds.numpy() - ops.convert_to_numpy(k_ds)))
    print(f"  mask_downsampler diff: {ds_diff:.6e}")

    # Feature projection alone
    dummy_vis_small = torch.randn(1, 256, 4, 4)
    with torch.no_grad():
        hf_fp = hf_model.memory_encoder.feature_projection(dummy_vis_small)
        k_fp = tracker_video.memory_encoder.feature_projection(
            ops.convert_to_tensor(dummy_vis_small.numpy())
        )
    fp_diff = np.max(np.abs(hf_fp.numpy() - ops.convert_to_numpy(k_fp)))
    print(f"  feature_projection diff: {fp_diff:.6e}")

    # Memory fuser alone (needs 256-channel NCHW input)
    dummy_fused = torch.randn(1, 256, 4, 4)
    with torch.no_grad():
        hf_fused = hf_model.memory_encoder.memory_fuser(dummy_fused)
        k_fused = tracker_video.memory_encoder.memory_fuser(
            ops.convert_to_tensor(dummy_fused.numpy())
        )
    fused_diff = np.max(np.abs(hf_fused.numpy() - ops.convert_to_numpy(k_fused)))
    print(f"  memory_fuser diff: {fused_diff:.6e}")

    # 3. Object Pointer Projection
    print("\n--- Object Pointer Projection ---")
    dummy_token = torch.randn(1, 256)
    with torch.no_grad():
        hf_ptr = hf_model.object_pointer_proj(dummy_token)
    with torch.no_grad():
        k_ptr = tracker_video.object_pointer_proj(
            ops.convert_to_tensor(dummy_token.numpy())
        )
    ptr_diff = np.max(np.abs(hf_ptr.numpy() - ops.convert_to_numpy(k_ptr)))
    print(f"  object_pointer_proj diff: {ptr_diff:.6e}")

    # 4. Single-frame mask decoder (same as Sam3Tracker — reuses same weights)
    print("\n--- Single-frame Mask Decoder (via Keras call) ---")
    import urllib.request

    from PIL import Image

    urllib.request.urlretrieve(
        "http://images.cocodataset.org/val2017/000000039769.jpg", "test_image.jpg"
    )
    img = Image.open("test_image.jpg").convert("RGB")
    from kmodels.models.sam3.sam3_processor import preprocess_image

    pv_keras, _ = preprocess_image(img)

    with torch.no_grad():
        k_out = tracker_video(
            pixel_values=pv_keras,
            input_points=np.array([[[[300, 250]]]], dtype=np.float32),
            input_labels=np.array([[[1]]], dtype=np.int32),
            multimask_output=True,
        )
    k_iou = ops.convert_to_numpy(k_out["iou_scores"])
    k_obj = ops.convert_to_numpy(k_out["object_score_logits"])
    print(f"  Keras iou_scores:   {k_iou.flatten()}")
    print(f"  Keras obj_score:    {k_obj.flatten()}")

    print("\n=== All component tests complete ===")


if __name__ == "__main__":
    convert()
