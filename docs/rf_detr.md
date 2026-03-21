# RF-DETR

**Paper**: [RF-DETR: Real-Time Detection Transformer](https://arxiv.org/abs/2502.18860)

RF-DETR is a real-time object detection model based on the DETR framework, designed for high-speed inference while maintaining strong detection accuracy. It comes in multiple size variants to balance speed and accuracy for different deployment scenarios.

## Model Variants

- **RFDETRNano** — 384px resolution
- **RFDETRSmall** — 512px resolution
- **RFDETRMedium** — 576px resolution
- **RFDETRBase** — 560px resolution, 29M params
- **RFDETRLarge** — 704px resolution

## Basic Usage

```python
import kmodels

# RF-DETR Base (29M params, 560px, COCO pre-trained)
model = kmodels.models.rf_detr.RFDETRBase(weights="coco")

# Available variants
model = kmodels.models.rf_detr.RFDETRNano(weights="coco")
model = kmodels.models.rf_detr.RFDETRSmall(weights="coco")
model = kmodels.models.rf_detr.RFDETRMedium(weights="coco")
model = kmodels.models.rf_detr.RFDETRLarge(weights="coco")

# Without pre-trained weights
model = kmodels.models.rf_detr.RFDETRBase(weights=None)
```

## Example Inference

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
