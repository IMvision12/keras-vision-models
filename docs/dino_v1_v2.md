# DINO & DINOv2

**DINO Paper**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
**DINOv2 Paper**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

DINO (self-**DI**stillation with **NO** labels) is a self-supervised learning method for Vision Transformers. It produces rich visual features without any labeled data, making the models excellent general-purpose feature extractors for downstream tasks like segmentation, detection, retrieval, and depth estimation.

DINOv2 improves on DINO with a larger curated dataset (LVD-142M), LayerScale, and stronger training recipe, producing state-of-the-art visual features.

## Model Variants

### DINO

- **DinoViTSmall16** -- ViT-S/16 (~21 M params, 16x16 patches)
- **DinoViTSmall8** -- ViT-S/8 (~21 M params, 8x8 patches)
- **DinoViTBase16** -- ViT-B/16 (~85 M params, 16x16 patches)
- **DinoViTBase8** -- ViT-B/8 (~85 M params, 8x8 patches)
- **DinoResNet50** -- ResNet-50 (~23 M params)

### DINOv2

- **DinoV2Small14** -- ViT-S/14 (~22 M params, 14x14 patches)
- **DinoV2Base14** -- ViT-B/14 (~86 M params, 14x14 patches)
- **DinoV2Large14** -- ViT-L/14 (~300 M params, 14x14 patches)

## Available Weights

### DINO

| Variant | dino |
|---------|:----:|
| DinoViTSmall16 | ✅ |
| DinoViTSmall8 | ✅ |
| DinoViTBase16 | ✅ |
| DinoViTBase8 | ✅ |
| DinoResNet50 | ✅ |

### DINOv2

| Variant | dinov2 |
|---------|:------:|
| DinoV2Small14 | ✅ |
| DinoV2Base14 | ✅ |
| DinoV2Large14 | ✅ |

## Features and Capabilities

- **Self-Supervised Features:** Trained without labels, producing general-purpose visual representations.
- **Backbone Mode:** `as_backbone=True` returns intermediate feature maps for dense prediction tasks (segmentation, detection).
- **Fine-Tuning Support:** `include_top=True` adds a classification head (randomly initialized) for fine-tuning on custom datasets.
- **Flexible Input:** Supports variable input resolutions (position embeddings are interpolated automatically for DINOv2).

## Basic Usage

### Feature Extraction (default)

```python
import numpy as np
from kmodels.models.dino import DinoViTSmall16
from kmodels.models.dino_v2 import DinoV2Small14

# DINO v1 -- returns token sequence (B, N, 384)
model = DinoViTSmall16(weights="dino")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(features.shape)  # (1, 197, 384)

# DINOv2 -- returns token sequence (B, N, 384)
model = DinoV2Small14(weights="dinov2")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(features.shape)  # (1, 257, 384)
```

### Intermediate Features for Dense Prediction

```python
from kmodels.models.dino_v2 import DinoV2Base14

model = DinoV2Base14(weights="dinov2", as_backbone=True)
feature_maps = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(feature_maps))  # 13 feature maps (1 after embedding + 12 blocks)
```

### Fine-Tuning with Classification Head

```python
from kmodels.models.dino_v2 import DinoV2Small14

model = DinoV2Small14(
    weights="dinov2",
    include_top=True,
    num_classes=10,
)
# Freeze backbone, train head
for layer in model.layers[:-1]:
    layer.trainable = False
```

### Using ResNet Backbone (DINO v1)

```python
from kmodels.models.dino import DinoResNet50

# Spatial feature map (B, H, W, 2048)
model = DinoResNet50(weights="dino")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(features.shape)  # (1, 7, 7, 2048)

# Pooled features (B, 2048)
model = DinoResNet50(weights="dino", pooling="avg")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(features.shape)  # (1, 2048)
```
