import numpy as np

from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights


def _transfer_backbone(keras_model, hf_sd, depth):
    # Patch embedding
    patch_conv = keras_model.get_layer("backbone_patch_embed")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf_sd["backbone.embeddings.patch_embeddings.projection.weight"],
    )
    patch_conv.bias.assign(
        hf_sd["backbone.embeddings.patch_embeddings.projection.bias"]
    )

    # CLS token
    cls_layer = keras_model.get_layer("backbone_cls_token")
    cls_layer.cls.assign(hf_sd["backbone.embeddings.cls_token"])

    # Position embeddings
    pos_layer = keras_model.get_layer("backbone_pos_embed")
    pos_layer.position_embedding.assign(
        hf_sd["backbone.embeddings.position_embeddings"]
    )

    # Transformer blocks
    for i in range(depth):
        hf_pfx = f"backbone.encoder.layer.{i}"

        # LN1
        ln1 = keras_model.get_layer(f"backbone_block_{i}_ln1")
        ln1.gamma.assign(hf_sd[f"{hf_pfx}.norm1.weight"])
        ln1.beta.assign(hf_sd[f"{hf_pfx}.norm1.bias"])

        # Attention: HF has separate Q, K, V; Keras has combined QKV
        attn = keras_model.get_layer(f"backbone_block_{i}_attn")
        q_w = hf_sd[f"{hf_pfx}.attention.attention.query.weight"]
        k_w = hf_sd[f"{hf_pfx}.attention.attention.key.weight"]
        v_w = hf_sd[f"{hf_pfx}.attention.attention.value.weight"]
        qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
        transfer_weights("kernel", attn.qkv.kernel, qkv_w)

        q_b = hf_sd[f"{hf_pfx}.attention.attention.query.bias"]
        k_b = hf_sd[f"{hf_pfx}.attention.attention.key.bias"]
        v_b = hf_sd[f"{hf_pfx}.attention.attention.value.bias"]
        qkv_b = np.concatenate([q_b, k_b, v_b], axis=0)
        attn.qkv.bias.assign(qkv_b)

        # Output projection
        transfer_weights(
            "kernel",
            attn.proj.kernel,
            hf_sd[f"{hf_pfx}.attention.output.dense.weight"],
        )
        attn.proj.bias.assign(hf_sd[f"{hf_pfx}.attention.output.dense.bias"])

        # LayerScale 1
        ls1 = keras_model.get_layer(f"backbone_block_{i}_ls1")
        ls1.gamma.assign(hf_sd[f"{hf_pfx}.layer_scale1.lambda1"])

        # LN2
        ln2 = keras_model.get_layer(f"backbone_block_{i}_ln2")
        ln2.gamma.assign(hf_sd[f"{hf_pfx}.norm2.weight"])
        ln2.beta.assign(hf_sd[f"{hf_pfx}.norm2.bias"])

        # MLP
        mlp_fc1 = keras_model.get_layer(f"backbone_block_{i}_mlp_fc1")
        transfer_weights("kernel", mlp_fc1.kernel, hf_sd[f"{hf_pfx}.mlp.fc1.weight"])
        mlp_fc1.bias.assign(hf_sd[f"{hf_pfx}.mlp.fc1.bias"])

        mlp_fc2 = keras_model.get_layer(f"backbone_block_{i}_mlp_fc2")
        transfer_weights("kernel", mlp_fc2.kernel, hf_sd[f"{hf_pfx}.mlp.fc2.weight"])
        mlp_fc2.bias.assign(hf_sd[f"{hf_pfx}.mlp.fc2.bias"])

        # LayerScale 2
        ls2 = keras_model.get_layer(f"backbone_block_{i}_ls2")
        ls2.gamma.assign(hf_sd[f"{hf_pfx}.layer_scale2.lambda1"])

    # Shared backbone LayerNorm
    backbone_ln = keras_model.get_layer("backbone_layernorm")
    backbone_ln.gamma.assign(hf_sd["backbone.layernorm.weight"])
    backbone_ln.beta.assign(hf_sd["backbone.layernorm.bias"])


def _transfer_neck(keras_model, hf_sd, reassemble_factors):
    # Reassemble stage
    for i in range(4):
        proj = keras_model.get_layer(f"neck_reassemble_{i}_projection")
        transfer_weights(
            "conv_kernel",
            proj.kernel,
            hf_sd[f"neck.reassemble_stage.layers.{i}.projection.weight"],
        )
        proj.bias.assign(hf_sd[f"neck.reassemble_stage.layers.{i}.projection.bias"])

        factor = reassemble_factors[i]
        if factor == 1:
            continue

        resize = keras_model.get_layer(f"neck_reassemble_{i}_resize")
        if factor > 1:
            # ConvTranspose2d
            hf_w = hf_sd[f"neck.reassemble_stage.layers.{i}.resize.weight"]
            # HF: (C_in, C_out, kH, kW) -> Keras: (kH, kW, C_out, C_in)
            keras_w = np.transpose(hf_w, (2, 3, 1, 0))
            resize.kernel.assign(keras_w)
        else:
            # Conv2d for downsampling
            transfer_weights(
                "conv_kernel",
                resize.kernel,
                hf_sd[f"neck.reassemble_stage.layers.{i}.resize.weight"],
            )
        resize.bias.assign(hf_sd[f"neck.reassemble_stage.layers.{i}.resize.bias"])

    # Neck convs (no bias)
    for i in range(4):
        conv = keras_model.get_layer(f"neck_conv_{i}")
        transfer_weights("conv_kernel", conv.kernel, hf_sd[f"neck.convs.{i}.weight"])

    # Fusion stage
    for i in range(4):
        hf_pfx = f"neck.fusion_stage.layers.{i}"

        proj = keras_model.get_layer(f"neck_fusion_{i}_projection")
        transfer_weights(
            "conv_kernel",
            proj.kernel,
            hf_sd[f"{hf_pfx}.projection.weight"],
        )
        proj.bias.assign(hf_sd[f"{hf_pfx}.projection.bias"])

        # Residual layer 1 (only exists for fusion layers 1, 2, 3 that have residual)
        for res_idx in [1, 2]:
            res_name = f"res{res_idx}"
            hf_res_name = f"residual_layer{res_idx}"
            for conv_idx in [1, 2]:
                try:
                    conv = keras_model.get_layer(
                        f"neck_fusion_{i}_{res_name}_conv{conv_idx}"
                    )
                except ValueError:
                    continue
                transfer_weights(
                    "conv_kernel",
                    conv.kernel,
                    hf_sd[f"{hf_pfx}.{hf_res_name}.convolution{conv_idx}.weight"],
                )
                conv.bias.assign(
                    hf_sd[f"{hf_pfx}.{hf_res_name}.convolution{conv_idx}.bias"]
                )


def _transfer_head(keras_model, hf_sd):
    for conv_name in ["head_conv1", "head_conv2", "head_conv3"]:
        hf_name = conv_name.replace("head_", "head.")
        conv = keras_model.get_layer(conv_name)
        transfer_weights("conv_kernel", conv.kernel, hf_sd[f"{hf_name}.weight"])
        conv.bias.assign(hf_sd[f"{hf_name}.bias"])


def transfer_depth_anything_v1_weights(keras_model, hf_sd, config):
    _transfer_backbone(keras_model, hf_sd, config["backbone_depth"])
    _transfer_neck(keras_model, hf_sd, config["reassemble_factors"])
    _transfer_head(keras_model, hf_sd)


if __name__ == "__main__":
    import gc

    import keras
    import torch
    from transformers import DepthAnythingForDepthEstimation

    from kmodels.models.depth_anything_v1.config import (
        DEPTH_ANYTHING_V1_HF_MODEL_IDS,
        DEPTH_ANYTHING_V1_MODEL_CONFIG,
    )
    from kmodels.models.depth_anything_v1.depth_anything_v1_model import (
        DepthAnythingV1Base,
        DepthAnythingV1Large,
        DepthAnythingV1Small,
    )

    VARIANTS = [
        ("DepthAnythingV1Small", DepthAnythingV1Small, "depth_anything_v1_small"),
        ("DepthAnythingV1Base", DepthAnythingV1Base, "depth_anything_v1_base"),
        ("DepthAnythingV1Large", DepthAnythingV1Large, "depth_anything_v1_large"),
    ]

    for name, ctor, save_name in VARIANTS:
        hf_id = DEPTH_ANYTHING_V1_HF_MODEL_IDS[name]
        config = DEPTH_ANYTHING_V1_MODEL_CONFIG[name]
        print(f"\n{'=' * 60}")
        print(f"Converting: {name}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = DepthAnythingForDepthEstimation.from_pretrained(hf_id).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = ctor(input_shape=(518, 518, 3), weights=None)
        transfer_depth_anything_v1_weights(keras_model, hf_sd, config)

        np.random.seed(42)
        test_image = np.random.rand(1, 518, 518, 3).astype(np.float32)

        keras_output = keras_model.predict(
            test_image,
            verbose=0,
        )
        keras_depth = keras_output

        with torch.no_grad():
            hf_input = torch.from_numpy(test_image.transpose(0, 3, 1, 2))
            hf_output = hf_model(pixel_values=hf_input)
            hf_depth = hf_output.predicted_depth.cpu().numpy()

        # Keras output: (B, H, W, 1), HF output: (B, H, W)
        keras_depth_squeezed = keras_depth.squeeze(-1)

        depth_diff = float(np.max(np.abs(keras_depth_squeezed - hf_depth)))
        mean_diff = float(np.mean(np.abs(keras_depth_squeezed - hf_depth)))
        print(f"  Max depth diff: {depth_diff:.6f}")
        print(f"  Mean depth diff: {mean_diff:.6f}")
        assert depth_diff < 25.0, f"{name}: depth diff {depth_diff:.2e}"
        print("  Verification OK")

        out = f"{save_name}.weights.h5"
        keras_model.save_weights(out)
        print(f"  Saved -> {out}")

        del keras_model, hf_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
