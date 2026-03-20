import keras
import numpy as np
import torch
from tqdm import tqdm
from transformers import EomtForUniversalSegmentation

from kmodels.models.eomt.eomt_model import EoMT_Base, EoMT_Large, EoMT_Small

VARIANT_MAP = {
    "small": EoMT_Small,
    "base": EoMT_Base,
    "large": EoMT_Large,
}


def convert_model(
    hf_model_name="tue-mps/coco_panoptic_eomt_large_640",
    input_shape=(640, 640, 3),
    num_queries=200,
    num_labels=133,
    variant="large",
):
    """Convert HuggingFace EoMT weights to Keras format.

    Args:
        hf_model_name: HuggingFace model ID.
        input_shape: Input shape (H, W, C).
        num_queries: Number of object queries.
        num_labels: Number of segmentation classes.
        variant: Model variant ("small", "base", or "large").
    """
    print(f"Loading HF model: {hf_model_name}")
    hf_model = EomtForUniversalSegmentation.from_pretrained(hf_model_name).eval()
    hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

    keras_model_cls = VARIANT_MAP[variant]
    print(f"Creating Keras model ({variant})...")
    keras_model = keras_model_cls(
        num_queries=num_queries,
        num_labels=num_labels,
        input_shape=input_shape,
        weights=None,
    )

    # Transfer embeddings
    print("Transferring embeddings...")

    # Patch embeddings projection (Conv2D)
    patch_proj = keras_model.get_layer("embeddings")
    conv_layer = patch_proj.patch_embeddings.projection

    # HF: (out_channels, in_channels, kH, kW) -> Keras: (kH, kW, in_channels, out_channels)
    conv_w = hf_state_dict["embeddings.patch_embeddings.projection.weight"]
    conv_layer.kernel.assign(np.transpose(conv_w, (2, 3, 1, 0)))
    conv_layer.bias.assign(hf_state_dict["embeddings.patch_embeddings.projection.bias"])

    # CLS token
    patch_proj.cls_token.assign(hf_state_dict["embeddings.cls_token"])

    # Register tokens
    patch_proj.register_tokens.assign(hf_state_dict["embeddings.register_tokens"])

    # Position embeddings (HF uses nn.Embedding, stored as (num_patches, hidden_size))
    pos_emb = hf_state_dict["embeddings.position_embeddings.weight"]
    patch_proj.position_embeddings.assign(np.expand_dims(pos_emb, axis=0))

    # Transfer query weights
    print("Transferring query weights...")
    query_layer = keras_model.get_layer("query")
    query_layer.query_weight.assign(hf_state_dict["query.weight"])

    # Transfer transformer layers
    num_layers = keras_model.num_hidden_layers
    for i in tqdm(range(num_layers), desc="Transferring transformer layers"):
        hf_prefix = f"layers.{i}"
        keras_layer = keras_model.get_layer(f"layers_{i}")

        # Layer norm 1
        keras_layer.norm1.gamma.assign(hf_state_dict[f"{hf_prefix}.norm1.weight"])
        keras_layer.norm1.beta.assign(hf_state_dict[f"{hf_prefix}.norm1.bias"])

        # Attention: q_proj, k_proj, v_proj, out_proj
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            dense = getattr(keras_layer.attention, proj)
            w = hf_state_dict[f"{hf_prefix}.attention.{proj}.weight"]
            dense.kernel.assign(w.T)
            dense.bias.assign(hf_state_dict[f"{hf_prefix}.attention.{proj}.bias"])

        # Layer scale 1
        keras_layer.layer_scale1.gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_scale1.lambda1"]
        )

        # Layer norm 2
        keras_layer.norm2.gamma.assign(hf_state_dict[f"{hf_prefix}.norm2.weight"])
        keras_layer.norm2.beta.assign(hf_state_dict[f"{hf_prefix}.norm2.bias"])

        # MLP
        if hasattr(keras_layer.mlp, "fc1"):
            # Standard MLP
            keras_layer.mlp.fc1.kernel.assign(
                hf_state_dict[f"{hf_prefix}.mlp.fc1.weight"].T
            )
            keras_layer.mlp.fc1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.fc1.bias"])
            keras_layer.mlp.fc2.kernel.assign(
                hf_state_dict[f"{hf_prefix}.mlp.fc2.weight"].T
            )
            keras_layer.mlp.fc2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.fc2.bias"])
        else:
            # SwiGLU FFN
            keras_layer.mlp.weights_in.kernel.assign(
                hf_state_dict[f"{hf_prefix}.mlp.weights_in.weight"].T
            )
            keras_layer.mlp.weights_in.bias.assign(
                hf_state_dict[f"{hf_prefix}.mlp.weights_in.bias"]
            )
            keras_layer.mlp.weights_out.kernel.assign(
                hf_state_dict[f"{hf_prefix}.mlp.weights_out.weight"].T
            )
            keras_layer.mlp.weights_out.bias.assign(
                hf_state_dict[f"{hf_prefix}.mlp.weights_out.bias"]
            )

        # Layer scale 2
        keras_layer.layer_scale2.gamma.assign(
            hf_state_dict[f"{hf_prefix}.layer_scale2.lambda1"]
        )

    # Final layer norm
    print("Transferring final layer norm...")
    layernorm = keras_model.get_layer("layernorm")
    layernorm.gamma.assign(hf_state_dict["layernorm.weight"])
    layernorm.beta.assign(hf_state_dict["layernorm.bias"])

    # Class predictor
    print("Transferring class predictor...")
    class_pred = keras_model.get_layer("class_predictor")
    class_pred.kernel.assign(hf_state_dict["class_predictor.weight"].T)
    class_pred.bias.assign(hf_state_dict["class_predictor.bias"])

    # Mask head
    print("Transferring mask head...")
    mask_head = keras_model.get_layer("mask_head")
    for fc_name in ["fc1", "fc2", "fc3"]:
        fc = getattr(mask_head, fc_name)
        fc.kernel.assign(hf_state_dict[f"mask_head.{fc_name}.weight"].T)
        fc.bias.assign(hf_state_dict[f"mask_head.{fc_name}.bias"])

    # Upscale block
    print("Transferring upscale block...")
    upscale = keras_model.get_layer("upscale_block")
    for block_idx in range(keras_model.num_upscale_blocks):
        hf_block_prefix = f"upscale_block.block.{block_idx}"
        keras_block = upscale.blocks[block_idx]

        # ConvTranspose2d: PyTorch (in, out, kH, kW) -> Keras (kH, kW, out, in)
        conv1_w = hf_state_dict[f"{hf_block_prefix}.conv1.weight"]
        keras_block.conv1.kernel.assign(np.transpose(conv1_w, (2, 3, 1, 0)))
        keras_block.conv1.bias.assign(hf_state_dict[f"{hf_block_prefix}.conv1.bias"])

        # DepthwiseConv2d: PyTorch (channels, 1, kH, kW) -> Keras (kH, kW, channels, 1)
        conv2_w = hf_state_dict[f"{hf_block_prefix}.conv2.weight"]
        keras_block.conv2.kernel.assign(np.transpose(conv2_w, (2, 3, 0, 1)))

        # LayerNorm2d
        keras_block.layernorm2d.norm.gamma.assign(
            hf_state_dict[f"{hf_block_prefix}.layernorm2d.weight"]
        )
        keras_block.layernorm2d.norm.beta.assign(
            hf_state_dict[f"{hf_block_prefix}.layernorm2d.bias"]
        )

    # Verify equivalence
    print("\nVerifying model equivalence...")
    np.random.seed(42)
    test_input = np.random.rand(1, *input_shape).astype(np.float32)

    # Normalize for HF model
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    normalized_input = (test_input - mean) / std

    # HF inference (channels first)
    hf_input = torch.tensor(normalized_input).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        hf_output = hf_model(pixel_values=hf_input)
        hf_class_logits = hf_output.class_queries_logits.numpy()
        hf_mask_logits = hf_output.masks_queries_logits.numpy()

    # Keras inference
    keras_output = keras_model(normalized_input.astype(np.float32), training=False)
    keras_class_logits = keras.ops.convert_to_numpy(keras_output["class_logits"])
    keras_mask_logits = keras.ops.convert_to_numpy(keras_output["mask_logits"])

    class_diff = np.max(np.abs(hf_class_logits - keras_class_logits))
    mask_diff = np.max(np.abs(hf_mask_logits - keras_mask_logits))

    print(f"Max class logits diff: {class_diff:.6f}")
    print(f"Max mask logits diff:  {mask_diff:.6f}")

    if class_diff > 1e-3 or mask_diff > 2e-3:
        raise ValueError(
            f"Model equivalence test failed "
            f"(class: {class_diff:.6f}, mask: {mask_diff:.6f})"
        )

    print("Model equivalence test passed!")

    # Save
    model_filename = hf_model_name.split("/")[-1].replace("-", "_") + ".weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved as {model_filename}")

    return keras_model


if __name__ == "__main__":
    configs = [
        {
            "hf_model_name": "tue-mps/coco_panoptic_eomt_small_640_2x",
            "input_shape": (640, 640, 3),
            "num_queries": 200,
            "num_labels": 133,
            "variant": "small",
        },
        {
            "hf_model_name": "tue-mps/coco_panoptic_eomt_base_640_2x",
            "input_shape": (640, 640, 3),
            "num_queries": 200,
            "num_labels": 133,
            "variant": "base",
        },
        {
            "hf_model_name": "tue-mps/coco_panoptic_eomt_large_640",
            "input_shape": (640, 640, 3),
            "num_queries": 200,
            "num_labels": 133,
            "variant": "large",
        },
    ]

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"Converting {cfg['hf_model_name']}...")
        print(f"{'=' * 60}")
        convert_model(**cfg)
        print()
