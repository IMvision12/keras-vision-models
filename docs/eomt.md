# EoMT

**Paper**: [Encoder-only Mask Transformer for Panoptic Segmentation](https://arxiv.org/abs/2504.07957)

EoMT (Encoder-only Mask Transformer) is a panoptic segmentation model that simplifies the standard encoder-decoder mask transformer pipeline by using only an encoder architecture. This approach reduces complexity while maintaining competitive segmentation performance.

## Architecture Highlights

- **Encoder-Only Design:** Simplifies the standard encode-decode vision pipeline by utilizing an highly efficient encoder-only design for panoptic segmentation.
- **Unified Segmentation Modeling:** Simultaneously excels at semantic, instance, and panoptic segmentation under one streamlined framework.
- **High Efficiency:** Eliminates the heavyweight decoder yielding drastically improved computational overhead and low latency.
- **Top-Tier Panoptic Processing:** Delivers strong dense prediction accuracy across varied visual domains and complex overlapping imagery.

## Available Models

| Model Variant | Supported Pre-trained Weights | Description |
|---------------|-------------------------------|-------------|
| `EoMT_Small`  | `coco_panoptic_640`           | Lightweight variant for panoptic segmentation |
| `EoMT_Base`   | `coco_panoptic_640`           | Standard base variant for panoptic segmentation |
| `EoMT_Large`  | `coco_panoptic_640`, `coco_instance_640`, `ade20k_semantic_512` | Large variant supporting Panoptic, Instance, and Semantic segmentation |

*Note: `coco_panoptic_640` weights are pretrained on the COCO panoptic dataset at 640x640 resolution. `coco_instance_640` brings COCO instance segmentation capability. `ade20k_semantic_512` provides rigorous semantic segmentation trained on the ADE20K dataset natively working at 512x512 resolution.*

## Basic Usage

```python
import kmodels

# Small Variant
model = kmodels.models.eomt.EoMT_Small(weights="coco_panoptic_640", input_shape=(640, 640, 3))

# Large Variant with Instance Weights
model_large = kmodels.models.eomt.EoMT_Large(weights="coco_instance_640", input_shape=(640, 640, 3))

# Without pre-trained weights
model_custom = kmodels.models.eomt.EoMT_Base(weights=None, input_shape=(640, 640, 3))
```

## Inference Example

Below is a complete example of loading an image, running it through the model, and extracting predictions.

```python
import keras
import numpy as np
from PIL import Image
import kmodels

# Load base model with pre-trained weights
model = kmodels.models.eomt.EoMT_Base(weights="coco_panoptic_640", input_shape=(640, 640, 3))

# Load and preprocess image
image = Image.open("scene.jpg").convert("RGB")
original_size = image.size
# Resize to the model's expected shape
image = image.resize((640, 640))

# Convert to tensor and add batch dimension
input_tensor = keras.ops.convert_to_tensor(np.array(image).astype("float32") / 255.0)
input_tensor = keras.ops.expand_dims(input_tensor, axis=0) # Shape: (1, 640, 640, 3)

# Run Inference
output = model(input_tensor)

# Output generally provides fused semantic mappings. Process accordingly!
print(output.keys() if isinstance(output, dict) else output.shape)
```
