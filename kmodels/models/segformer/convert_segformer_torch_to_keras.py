from typing import Dict, List

import keras
import numpy as np
import torch
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from kmodels.models import segformer
from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "block": "segformer.encoder.block",
    "patch.embed": "segformer.encoder.patch_embeddings",
    "layernorm": "layer_norm",
    "layer_norm.1": "layer_norm_1",
    "layer_norm.2": "layer_norm_2",
    "conv.proj": "proj",
    "dense.1": "dense1",
    "dense.2": "dense2",
    "dwconv": "dwconv.dwconv",
    "final": "segformer.encoder",
    "segformer.encoder.layer_norm_1": "segformer.encoder.layer_norm.1",
    "segformer.encoder.layer_norm_2": "segformer.encoder.layer_norm.2",
    "segformer.encoder.layer_norm_3": "segformer.encoder.layer_norm.3",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "predictions": "classifier",
}

attn_name_replace = {
    "block": "segformer.encoder.block",
    "attn.q": "attention.self.query",
    "attn.k": "attention.self.key",
    "attn.v": "attention.self.value",
    "attn.proj": "attention.output.dense",
    "attn.sr": "attention.self.sr",
    "attn.norm": "attention.self.layer_norm",
}

SEGFORMER_WEIGHTS_CONFIG: List[Dict] = [
    # SegFormerB0
    {
        "keras_cls": segformer.SegFormerB0,
        "hf_name": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        "variant_name": "SegFormerB0",
        "input_shape": (1024, 1024, 3),
        "num_classes": 19,
        "output": "SegFormer_B0_city_1024.weights.h5",
    },
    {
        "keras_cls": segformer.SegFormerB0,
        "hf_name": "nvidia/segformer-b0-finetuned-cityscapes-768-768",
        "variant_name": "SegFormerB0",
        "input_shape": (768, 768, 3),
        "num_classes": 19,
        "output": "SegFormer_B0_city_768.weights.h5",
    },
    {
        "keras_cls": segformer.SegFormerB0,
        "hf_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "variant_name": "SegFormerB0",
        "input_shape": (512, 512, 3),
        "num_classes": 150,
        "output": "SegFormer_B0_ade.weights.h5",
    },
    # SegFormerB1
    {
        "keras_cls": segformer.SegFormerB1,
        "hf_name": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        "variant_name": "SegFormerB1",
        "input_shape": (1024, 1024, 3),
        "num_classes": 19,
        "output": "SegFormer_B1_city_1024.weights.h5",
    },
    {
        "keras_cls": segformer.SegFormerB1,
        "hf_name": "nvidia/segformer-b1-finetuned-ade-512-512",
        "variant_name": "SegFormerB1",
        "input_shape": (512, 512, 3),
        "num_classes": 150,
        "output": "SegFormer_B1_ade.weights.h5",
    },
    # SegFormerB2
    {
        "keras_cls": segformer.SegFormerB2,
        "hf_name": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        "variant_name": "SegFormerB2",
        "input_shape": (1024, 1024, 3),
        "num_classes": 19,
        "output": "SegFormer_B2_city_1024.weights.h5",
    },
    {
        "keras_cls": segformer.SegFormerB2,
        "hf_name": "nvidia/segformer-b2-finetuned-ade-512-512",
        "variant_name": "SegFormerB2",
        "input_shape": (512, 512, 3),
        "num_classes": 150,
        "output": "SegFormer_B2_ade.weights.h5",
    },
    # SegFormerB3
    {
        "keras_cls": segformer.SegFormerB3,
        "hf_name": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
        "variant_name": "SegFormerB3",
        "input_shape": (1024, 1024, 3),
        "num_classes": 19,
        "output": "SegFormer_B3_city_1024.weights.h5",
    },
    {
        "keras_cls": segformer.SegFormerB3,
        "hf_name": "nvidia/segformer-b3-finetuned-ade-512-512",
        "variant_name": "SegFormerB3",
        "input_shape": (512, 512, 3),
        "num_classes": 150,
        "output": "SegFormer_B3_ade.weights.h5",
    },
    # SegFormerB4
    {
        "keras_cls": segformer.SegFormerB4,
        "hf_name": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
        "variant_name": "SegFormerB4",
        "input_shape": (1024, 1024, 3),
        "num_classes": 19,
        "output": "SegFormer_B4_city_1024.weights.h5",
    },
    {
        "keras_cls": segformer.SegFormerB4,
        "hf_name": "nvidia/segformer-b4-finetuned-ade-512-512",
        "variant_name": "SegFormerB4",
        "input_shape": (512, 512, 3),
        "num_classes": 150,
        "output": "SegFormer_B4_ade.weights.h5",
    },
    # SegFormerB5
    {
        "keras_cls": segformer.SegFormerB5,
        "hf_name": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "variant_name": "SegFormerB5",
        "input_shape": (1024, 1024, 3),
        "num_classes": 19,
        "output": "SegFormer_B5_city_1024.weights.h5",
    },
    {
        "keras_cls": segformer.SegFormerB5,
        "hf_name": "nvidia/segformer-b5-finetuned-ade-640-640",
        "variant_name": "SegFormerB5",
        "input_shape": (640, 640, 3),
        "num_classes": 150,
        "output": "SegFormer_B5_ade.weights.h5",
    },
]

for config in SEGFORMER_WEIGHTS_CONFIG:
    keras_cls = config["keras_cls"]
    hf_name = config["hf_name"]
    variant_name = config["variant_name"]
    input_shape = config["input_shape"]
    num_classes = config["num_classes"]
    output_file = config["output"]

    print(f"\n{'=' * 60}")
    print(f"Converting {variant_name} from {hf_name}")
    print(f"  input_shape={input_shape}, num_classes={num_classes}")
    print(f"  output={output_file}")
    print(f"{'=' * 60}")

    keras_model: keras.Model = keras_cls(
        weights=None,
        num_classes=num_classes,
        input_shape=input_shape,
        backbone=None,
    )
    torch_model: torch.nn.Module = SegformerForSemanticSegmentation.from_pretrained(
        hf_name
    ).eval()
    trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
        torch_model
    )
    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model.backbone
    )

    for keras_weight, keras_weight_name in tqdm(
        trainable_keras_weights + non_trainable_keras_weights,
        total=len(trainable_keras_weights + non_trainable_keras_weights),
        desc=f"Transferring backbone weights ({variant_name})",
    ):
        torch_weight_name: str = keras_weight_name
        for keras_name_part, torch_name_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )

        torch_weights_dict: Dict[str, torch.Tensor] = {
            **trainable_torch_weights,
            **non_trainable_torch_weights,
        }

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, torch_weights_dict, attn_name_replace
            )
            continue

        if torch_weight_name not in torch_weights_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]

        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name,
                keras_weight.shape,
                torch_weight_name,
                torch_weight.shape,
            )

        transfer_weights(keras_weight_name, keras_weight, torch_weight)

    pytorch_state_dict = torch_model.state_dict()

    # Linear C1 projection
    keras_model.get_layer(f"{variant_name}_head_linear_c1").weights[0].assign(
        pytorch_state_dict["decode_head.linear_c.0.proj.weight"].cpu().numpy().T
    )
    keras_model.get_layer(f"{variant_name}_head_linear_c1").weights[1].assign(
        pytorch_state_dict["decode_head.linear_c.0.proj.bias"].cpu().numpy()
    )

    # Linear C2 projection
    keras_model.get_layer(f"{variant_name}_head_linear_c2").weights[0].assign(
        pytorch_state_dict["decode_head.linear_c.1.proj.weight"].cpu().numpy().T
    )
    keras_model.get_layer(f"{variant_name}_head_linear_c2").weights[1].assign(
        pytorch_state_dict["decode_head.linear_c.1.proj.bias"].cpu().numpy()
    )

    # Linear C3 projection
    keras_model.get_layer(f"{variant_name}_head_linear_c3").weights[0].assign(
        pytorch_state_dict["decode_head.linear_c.2.proj.weight"].cpu().numpy().T
    )
    keras_model.get_layer(f"{variant_name}_head_linear_c3").weights[1].assign(
        pytorch_state_dict["decode_head.linear_c.2.proj.bias"].cpu().numpy()
    )

    # Linear C4 projection
    keras_model.get_layer(f"{variant_name}_head_linear_c4").weights[0].assign(
        pytorch_state_dict["decode_head.linear_c.3.proj.weight"].cpu().numpy().T
    )
    keras_model.get_layer(f"{variant_name}_head_linear_c4").weights[1].assign(
        pytorch_state_dict["decode_head.linear_c.3.proj.bias"].cpu().numpy()
    )

    # Conv2D (linear fuse conv)
    conv_weight = pytorch_state_dict["decode_head.linear_fuse.weight"].cpu().numpy()
    conv_weight = np.transpose(conv_weight, (2, 3, 1, 0))
    keras_model.get_layer(f"{variant_name}_head_fusion_conv").weights[0].assign(
        conv_weight
    )

    # Batch Normalization
    bn_layer = keras_model.get_layer(f"{variant_name}_head_fusion_bn")
    bn_layer.weights[0].assign(
        pytorch_state_dict["decode_head.batch_norm.weight"].cpu().numpy()
    )
    bn_layer.weights[1].assign(
        pytorch_state_dict["decode_head.batch_norm.bias"].cpu().numpy()
    )
    bn_layer.weights[2].assign(
        pytorch_state_dict["decode_head.batch_norm.running_mean"].cpu().numpy()
    )
    bn_layer.weights[3].assign(
        pytorch_state_dict["decode_head.batch_norm.running_var"].cpu().numpy()
    )

    # Final Conv Layer
    final_conv_weight = (
        pytorch_state_dict["decode_head.classifier.weight"].cpu().numpy()
    )
    final_conv_weight = np.transpose(final_conv_weight, (2, 3, 1, 0))
    keras_model.get_layer(f"{variant_name}_head_classifier").weights[0].assign(
        final_conv_weight
    )
    keras_model.get_layer(f"{variant_name}_head_classifier").weights[1].assign(
        pytorch_state_dict["decode_head.classifier.bias"].cpu().numpy()
    )

    # Verify equivalence (compare at classifier level before final upsample,
    # since Keras upsamples to input size but HF outputs at 1/4 resolution)
    print("Verifying model equivalence...")
    np.random.seed(42)
    test_input = np.random.rand(1, *input_shape).astype(np.float32)
    hf_input = torch.tensor(test_input).permute(0, 3, 1, 2)

    with torch.no_grad():
        hf_output = torch_model(pixel_values=hf_input).logits.numpy()
    hf_output = np.transpose(hf_output, (0, 2, 3, 1))

    classifier_layer = keras_model.get_layer(f"{variant_name}_head_classifier")
    sub_model = keras.Model(keras_model.input, classifier_layer.output)
    keras_output = np.array(sub_model.predict(test_input, verbose=0))

    max_diff = np.max(np.abs(hf_output - keras_output))
    print(f"Max logits diff: {max_diff:.6f}")

    if max_diff > 1e-3:
        raise ValueError(
            f"Equivalence test failed for {variant_name} - max diff {max_diff:.6f} > 1e-3"
        )
    print("Model equivalence test passed!")

    # Save the model
    keras_model.save_weights(output_file)
    print(f"Saved {output_file}")

    del keras_model, torch_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
