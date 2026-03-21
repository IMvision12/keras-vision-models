# DETR

**Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

DETR (DEtection TRansformer) is an end-to-end object detection model that combines a convolutional backbone with a Transformer encoder-decoder architecture. It eliminates the need for hand-designed components like non-maximum suppression and anchor generation by using a set-based global loss via bipartite matching.

## Basic Usage

```python
import kmodels

# Load DETR with ResNet-50 backbone (COCO pre-trained)
model = kmodels.models.detr.DETRResNet50(
    weights="detr_resnet50_coco.weights.h5",
    input_shape=(800, 800, 3),
    include_normalization=False,
)

# Without pre-trained weights
model = kmodels.models.detr.DETRResNet50(weights=None, input_shape=(800, 800, 3))

# ResNet-101 variant
model = kmodels.models.detr.DETRResNet101(weights=None, input_shape=(800, 800, 3))
```

## Example Inference

```python
import kmodels
from kmodels.models.detr import DETRImageProcessor, DETRPostProcessor
from PIL import Image

model = kmodels.models.detr.DETRResNet50(
    weights="detr_resnet50_coco.weights.h5",
    input_shape=(800, 800, 3),
    include_normalization=False,
)

image = Image.open("image.jpg")
original_size = image.size[::-1]  # (H, W)

# Preprocess: resize, rescale, ImageNet normalize
processed = DETRImageProcessor(image, size={"height": 800, "width": 800})

# Inference
output = model(processed, training=False)
# output["logits"]:     (1, 100, 92) — class logits per query
# output["pred_boxes"]: (1, 100, 4)  — normalized (cx, cy, w, h)

# Post-process: filter by confidence, convert boxes to pixel coords
results = DETRPostProcessor(output, threshold=0.7, target_sizes=[original_size])
for score, label, box in zip(results[0]["scores"], results[0]["label_names"], results[0]["boxes"]):
    print(f"{label}: {score:.2f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

# Output:
# bird: 0.97 at [209, 44, 430, 393]
```
