# Object Detection Models

## DETR

### Basic Usage

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

### Example Inference

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

## RF-DETR

### Basic Usage

```python
import kmodels

# RF-DETR Base (29M params, 560px, COCO pre-trained)
model = kmodels.models.rf_detr.RFDETRBase(weights="coco")

# Available variants: Nano (384px), Small (512px), Medium (576px), Base (560px), Large (704px)
model = kmodels.models.rf_detr.RFDETRNano(weights="coco")
model = kmodels.models.rf_detr.RFDETRSmall(weights="coco")
model = kmodels.models.rf_detr.RFDETRMedium(weights="coco")
model = kmodels.models.rf_detr.RFDETRLarge(weights="coco")

# Without pre-trained weights
model = kmodels.models.rf_detr.RFDETRBase(weights=None)
```

### Example Inference

```python
import kmodels
from kmodels.models.rf_detr import RFDETRImageProcessor, RFDETRPostProcessor
from PIL import Image

model = kmodels.models.rf_detr.RFDETRBase(weights="coco")

image = Image.open("image.jpg")
original_size = image.size[::-1]  # (H, W)

# Preprocess: rescale, ImageNet normalize, resize to model resolution
processed = RFDETRImageProcessor(image, size={"height": 560, "width": 560})

# Inference
output = model(processed, training=False)
# output["pred_logits"]: (1, 300, 91) — class logits per query
# output["pred_boxes"]:  (1, 300, 4)  — normalized (cx, cy, w, h)

# Post-process: sigmoid, top-K selection, convert boxes to pixel coords
results = RFDETRPostProcessor(output, threshold=0.5, target_sizes=[original_size])
for score, label, box in zip(results[0]["scores"], results[0]["label_names"], results[0]["boxes"]):
    print(f"{label}: {score:.2f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

# Output:
# cat: 0.96 at [7, 55, 318, 472]
# cat: 0.93 at [343, 24, 639, 372]
# remote: 0.90 at [41, 73, 176, 118]
# remote: 0.73 at [334, 77, 370, 187]
# couch: 0.67 at [1, 2, 640, 475]
```

## Available Object Detection Models

| Model Name | Reference Paper | Source of Weights |
|------------|----------------|-------------------|
| DETR | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | `transformers` |
| RF-DETR | [RF-DETR: Real-Time Detection Transformer](https://arxiv.org/abs/2502.18860) | `rfdetr` |
