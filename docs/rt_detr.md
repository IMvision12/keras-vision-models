# RT-DETR

**Paper**: [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)

RT-DETR (Real-Time DEtection TRansformer) is the first real-time end-to-end object detection model based on the DETR framework. It combines a ResNet-vd backbone with a hybrid encoder (AIFI transformer + CCFM feature pyramid) and a deformable-attention decoder with iterative bounding box refinement. RT-DETR achieves state-of-the-art speed-accuracy trade-offs, outperforming YOLO detectors at comparable latency.

## Architecture Highlights

- **ResNet-vd Backbone:** Uses a 3-convolution stem variant of ResNet with average-pooling shortcuts for improved feature extraction.
- **Hybrid Encoder (AIFI + CCFM):** Attention-based Intra-scale Feature Interaction on the highest-level features, combined with Cross-scale Feature Fusion via FPN top-down and PAN bottom-up paths with CSPRepLayer blocks.
- **Deformable Decoder:** Multi-scale deformable attention enables efficient cross-attention across feature pyramid levels with learned sampling locations.
- **Two-Stage Query Init:** Encoder proposals initialize decoder queries, eliminating the need for learned query embeddings.
- **Iterative Box Refinement:** Each decoder layer refines bounding box predictions from the previous layer.

## Available Models

| Model | Backbone | Params | Weights |
|-------|----------|--------|---------|
| `RTDETRResNet18` | ResNet-18-vd | 20M | `coco`, `coco_o365` |
| `RTDETRResNet34` | ResNet-34-vd | 31M | `coco` |
| `RTDETRResNet50` | ResNet-50-vd | 43M | `coco`, `coco_o365` |
| `RTDETRResNet101` | ResNet-101-vd | 77M | `coco`, `coco_o365` |

## Basic Usage

```python
import kmodels

# RT-DETR with ResNet-50 backbone (COCO pre-trained)
model = kmodels.models.rt_detr.RTDETRResNet50(weights="coco")

# Available variants
model = kmodels.models.rt_detr.RTDETRResNet18(weights="coco")
model = kmodels.models.rt_detr.RTDETRResNet34(weights="coco")
model = kmodels.models.rt_detr.RTDETRResNet101(weights="coco")

# COCO + Objects365 pre-trained weights
model = kmodels.models.rt_detr.RTDETRResNet50(weights="coco_o365")

# Without pre-trained weights
model = kmodels.models.rt_detr.RTDETRResNet50(weights=None)
```

## Example Inference

```python
import kmodels
from kmodels.models.rt_detr import RTDETRImageProcessor, RTDETRPostProcessor
from PIL import Image

model = kmodels.models.rt_detr.RTDETRResNet50(weights="coco")

image = Image.open("image.jpg")
original_size = image.size[::-1]  # (H, W)

# Preprocess: resize to 640x640, rescale to [0, 1] (no ImageNet normalization)
processed = RTDETRImageProcessor(image)

# Inference
output = model(processed, training=False)
# output["logits"]:     (1, 300, 80) — class logits per query
# output["pred_boxes"]: (1, 300, 4)  — normalized (cx, cy, w, h)

# Post-process: sigmoid, top-K selection, convert boxes to pixel coords
results = RTDETRPostProcessor(output, threshold=0.5, target_sizes=[original_size])
for score, label, box in zip(results[0]["scores"], results[0]["label_names"], results[0]["boxes"]):
    print(f"{label}: {score:.2f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

# Output:
# cat: 0.96 at [7, 55, 318, 472]
# cat: 0.96 at [343, 24, 639, 372]
# remote: 0.95 at [41, 73, 176, 118]
# remote: 0.92 at [334, 77, 370, 187]
# couch: 0.97 at [1, 2, 640, 475]
```

## Preprocessing Notes

Unlike DETR and RF-DETR, RT-DETR does **not** apply ImageNet normalization. The model expects input images rescaled to `[0, 1]` (divide by 255) and resized to `640x640`. The `RTDETRImageProcessor` handles this automatically.

## Weight Conversion

To convert weights from HuggingFace checkpoints:

```bash
KERAS_BACKEND=torch python kmodels/models/rt_detr/convert_rt_detr_hf_to_keras.py
```

This converts all 7 checkpoints from the [PekingU](https://huggingface.co/PekingU) organization:
- `PekingU/rtdetr_r18vd`, `PekingU/rtdetr_r34vd`, `PekingU/rtdetr_r50vd`, `PekingU/rtdetr_r101vd`
- `PekingU/rtdetr_r18vd_coco_o365`, `PekingU/rtdetr_r50vd_coco_o365`, `PekingU/rtdetr_r101vd_coco_o365`
