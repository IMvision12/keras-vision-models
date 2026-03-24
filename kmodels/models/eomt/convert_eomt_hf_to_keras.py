from typing import Dict, List, Union

import keras
import numpy as np
import torch
from tqdm import tqdm
from transformers import EomtForUniversalSegmentation

from kmodels.models.eomt.eomt_model import EoMT_Base, EoMT_Large, EoMT_Small
from kmodels.utils.weight_transfer_torch_to_keras import (
    transfer_nested_layer_weights,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

model_configs: List[Dict[str, Union[type, str, int]]] = [
    {
        "keras_model_cls": EoMT_Small,
        "hf_model_name": "tue-mps/coco_panoptic_eomt_small_640_2x",
        "input_shape": (640, 640, 3),
        "num_queries": 200,
        "num_labels": 133,
    },
    {
        "keras_model_cls": EoMT_Base,
        "hf_model_name": "tue-mps/coco_panoptic_eomt_base_640_2x",
        "input_shape": (640, 640, 3),
        "num_queries": 200,
        "num_labels": 133,
    },
    {
        "keras_model_cls": EoMT_Large,
        "hf_model_name": "tue-mps/coco_panoptic_eomt_large_640",
        "input_shape": (640, 640, 3),
        "num_queries": 200,
        "num_labels": 133,
    },
    {
        "keras_model_cls": EoMT_Large,
        "hf_model_name": "tue-mps/coco_instance_eomt_large_640",
        "input_shape": (640, 640, 3),
        "num_queries": 200,
        "num_labels": 80,
    },
    {
        "keras_model_cls": EoMT_Large,
        "hf_model_name": "tue-mps/ade20k_semantic_eomt_large_512",
        "input_shape": (512, 512, 3),
        "num_queries": 100,
        "num_labels": 150,
    },
]

for model_config in model_configs:
    print(f"\n{'=' * 60}")
    print(f"Converting {model_config['hf_model_name']}...")
    print(f"{'=' * 60}")

    keras_model: keras.Model = model_config["keras_model_cls"](
        num_queries=model_config["num_queries"],
        num_labels=model_config["num_labels"],
        input_shape=model_config["input_shape"],
        weights=None,
    )

    hf_model: torch.nn.Module = EomtForUniversalSegmentation.from_pretrained(
        model_config["hf_model_name"]
    ).eval()

    hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

    emb = keras_model.get_layer("embeddings")

    transfer_weights(
        "conv_kernel",
        emb.patch_embeddings.projection.weights[0],
        hf_state_dict["embeddings.patch_embeddings.projection.weight"],
    )
    emb.patch_embeddings.projection.weights[1].assign(
        hf_state_dict["embeddings.patch_embeddings.projection.bias"]
    )

    emb.cls_token.assign(hf_state_dict["embeddings.cls_token"])
    emb.register_tokens.assign(hf_state_dict["embeddings.register_tokens"])

    pos_emb = hf_state_dict["embeddings.position_embeddings.weight"]
    emb.position_embeddings.assign(np.expand_dims(pos_emb, axis=0))

    query_layer = keras_model.get_layer("query")
    query_layer.query_weight.assign(hf_state_dict["query.weight"])

    num_layers = keras_model.num_hidden_layers
    use_swiglu = keras_model.use_swiglu_ffn
    for i in tqdm(range(num_layers), desc="Transferring transformer layers"):
        hf_prefix = f"layers.{i}"
        k_prefix = f"layers_{i}"

        norm1 = keras_model.get_layer(f"{k_prefix}_norm1")
        norm1.weights[0].assign(hf_state_dict[f"{hf_prefix}.norm1.weight"])
        norm1.weights[1].assign(hf_state_dict[f"{hf_prefix}.norm1.bias"])

        attn = keras_model.get_layer(f"{k_prefix}_attention")
        transfer_nested_layer_weights(
            attn,
            hf_state_dict,
            f"{hf_prefix}.attention",
            name_mapping=weight_name_mapping,
        )

        ls1 = keras_model.get_layer(f"{k_prefix}_layer_scale1")
        ls1.weights[0].assign(hf_state_dict[f"{hf_prefix}.layer_scale1.lambda1"])

        norm2 = keras_model.get_layer(f"{k_prefix}_norm2")
        norm2.weights[0].assign(hf_state_dict[f"{hf_prefix}.norm2.weight"])
        norm2.weights[1].assign(hf_state_dict[f"{hf_prefix}.norm2.bias"])

        if not use_swiglu:
            fc1 = keras_model.get_layer(f"{k_prefix}_mlp_fc1")
            transfer_weights(
                "kernel",
                fc1.weights[0],
                hf_state_dict[f"{hf_prefix}.mlp.fc1.weight"],
            )
            fc1.weights[1].assign(hf_state_dict[f"{hf_prefix}.mlp.fc1.bias"])
            fc2 = keras_model.get_layer(f"{k_prefix}_mlp_fc2")
            transfer_weights(
                "kernel",
                fc2.weights[0],
                hf_state_dict[f"{hf_prefix}.mlp.fc2.weight"],
            )
            fc2.weights[1].assign(hf_state_dict[f"{hf_prefix}.mlp.fc2.bias"])
        else:
            w_in = keras_model.get_layer(f"{k_prefix}_mlp_weights_in")
            transfer_weights(
                "kernel",
                w_in.weights[0],
                hf_state_dict[f"{hf_prefix}.mlp.weights_in.weight"],
            )
            w_in.weights[1].assign(hf_state_dict[f"{hf_prefix}.mlp.weights_in.bias"])
            w_out = keras_model.get_layer(f"{k_prefix}_mlp_weights_out")
            transfer_weights(
                "kernel",
                w_out.weights[0],
                hf_state_dict[f"{hf_prefix}.mlp.weights_out.weight"],
            )
            w_out.weights[1].assign(hf_state_dict[f"{hf_prefix}.mlp.weights_out.bias"])

        ls2 = keras_model.get_layer(f"{k_prefix}_layer_scale2")
        ls2.weights[0].assign(hf_state_dict[f"{hf_prefix}.layer_scale2.lambda1"])

    layernorm = keras_model.get_layer("layernorm")
    layernorm.weights[0].assign(hf_state_dict["layernorm.weight"])
    layernorm.weights[1].assign(hf_state_dict["layernorm.bias"])

    class_pred = keras_model.get_layer("class_predictor")
    transfer_weights(
        "kernel",
        class_pred.weights[0],
        hf_state_dict["class_predictor.weight"],
    )
    class_pred.weights[1].assign(hf_state_dict["class_predictor.bias"])

    for fc_name in ["fc1", "fc2", "fc3"]:
        fc = keras_model.get_layer(f"mask_head_{fc_name}")
        transfer_weights(
            "kernel",
            fc.weights[0],
            hf_state_dict[f"mask_head.{fc_name}.weight"],
        )
        fc.weights[1].assign(hf_state_dict[f"mask_head.{fc_name}.bias"])

    for block_idx in range(keras_model.num_upscale_blocks):
        hf_block_prefix = f"upscale_block.block.{block_idx}"
        k_block_prefix = f"upscale_block_{block_idx}"

        conv1 = keras_model.get_layer(f"{k_block_prefix}_conv1")
        transfer_weights(
            "conv_kernel",
            conv1.weights[0],
            hf_state_dict[f"{hf_block_prefix}.conv1.weight"],
        )
        conv1.weights[1].assign(hf_state_dict[f"{hf_block_prefix}.conv1.bias"])

        conv2 = keras_model.get_layer(f"{k_block_prefix}_conv2")
        transfer_weights(
            "dwconv_kernel",
            conv2.weights[0],
            hf_state_dict[f"{hf_block_prefix}.conv2.weight"],
        )

        ln = keras_model.get_layer(f"{k_block_prefix}_layernorm")
        ln.weights[0].assign(hf_state_dict[f"{hf_block_prefix}.layernorm2d.weight"])
        ln.weights[1].assign(hf_state_dict[f"{hf_block_prefix}.layernorm2d.bias"])

    print("\nVerifying model equivalence...")
    np.random.seed(42)
    input_shape = model_config["input_shape"]
    test_input = np.random.rand(1, *input_shape).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    normalized_input = (test_input - mean) / std

    hf_input = torch.tensor(normalized_input).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        hf_output = hf_model(pixel_values=hf_input)
        hf_class_logits = hf_output.class_queries_logits.numpy()
        hf_mask_logits = hf_output.masks_queries_logits.numpy()

    keras_output = keras_model(normalized_input.astype(np.float32), training=False)
    keras_class_logits = keras.ops.convert_to_numpy(keras_output["class_logits"])
    keras_mask_logits = keras.ops.convert_to_numpy(keras_output["mask_logits"])

    class_diff = np.max(np.abs(hf_class_logits - keras_class_logits))
    mask_diff = np.max(np.abs(hf_mask_logits - keras_mask_logits))

    print(f"Max class logits diff: {class_diff:.6f}")
    print(f"Max mask logits diff:  {mask_diff:.6f}")

    if class_diff > 1e-3 or mask_diff > 5e-3:
        raise ValueError(
            f"Model equivalence test failed "
            f"(class: {class_diff:.6f}, mask: {mask_diff:.6f})"
        )

    print("Model equivalence test passed!")

    model_filename = (
        model_config["hf_model_name"].split("/")[-1].replace("-", "_") + ".weights.h5"
    )
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")

    del keras_model, hf_model, hf_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
