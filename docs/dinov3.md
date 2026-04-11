# DINOv3

**Paper**: [DINOv3: Self-Supervised Visual Representation Learning at Scale](https://arxiv.org/abs/2508.10104)

DINOv3 is the third generation of the DINO self-supervised learning framework. It introduces 2D Rotary Position Embeddings (RoPE) and register tokens to the ViT backbone, and distills features into ConvNeXt-v2 student networks. Trained on the large-scale LVD-1689M dataset, DINOv3 produces state-of-the-art visual features for downstream tasks.

## Weights License

DINOv3 weights are gated on HuggingFace. Before using `weights="dinov3"`:

1. Accept the license at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
2. Authenticate: `huggingface-cli login` or set `export HF_TOKEN=<your_token>`

On first use, weights are downloaded, converted to Keras, and cached at `~/.cache/kmodels/`.

## Model Variants

### ViT

- **DinoV3ViTSmall16** -- ViT-S/16 (~21 M params, 384-dim, 6 heads, 12 blocks)
- **DinoV3ViTBase16** -- ViT-B/16 (~86 M params, 768-dim, 12 heads, 12 blocks)
- **DinoV3ViTLarge16** -- ViT-L/16 (~300 M params, 1024-dim, 16 heads, 24 blocks)

### ConvNeXt

- **DinoV3ConvNeXtTiny** -- ConvNeXt-v2-Tiny (~29 M params, depths [3,3,9,3])
- **DinoV3ConvNeXtSmall** -- ConvNeXt-v2-Small (~50 M params, depths [3,3,27,3])
- **DinoV3ConvNeXtBase** -- ConvNeXt-v2-Base (~89 M params, depths [3,3,27,3])
- **DinoV3ConvNeXtLarge** -- ConvNeXt-v2-Large (~198 M params, depths [3,3,27,3])

## Features and Capabilities

- **Gated Weight Loading:** Weights are automatically downloaded from HuggingFace, converted from PyTorch, and cached locally at `~/.cache/kmodels/`.
- **2D RoPE:** ViT variants use 2D Rotary Position Embeddings applied to patch tokens only (CLS and register tokens are excluded).
- **Register Tokens:** 4 learnable register tokens inserted between CLS and patch tokens improve attention map quality.
- **Backbone Mode:** `as_backbone=True` returns intermediate feature maps for dense prediction tasks.
- **Fine-Tuning Support:** `include_top=True` adds a classification head for fine-tuning on custom datasets.

## Basic Usage

### Setup

```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token"
```

### Feature Extraction (default)

```python
import numpy as np
from kmodels.models.dino_v3 import DinoV3ViTSmall16, DinoV3ConvNeXtTiny

# ViT -- returns token sequence (B, N, 384)
model = DinoV3ViTSmall16(weights="dinov3")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(features.shape)  # (1, 201, 384)  -- 1 CLS + 4 register + 196 patches

# ConvNeXt -- returns spatial feature map (B, H, W, C)
model = DinoV3ConvNeXtTiny(weights="dinov3")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(features.shape)  # (1, 7, 7, 768)
```

### Intermediate Features for Dense Prediction

```python
from kmodels.models.dino_v3 import DinoV3ViTBase16

model = DinoV3ViTBase16(weights="dinov3", as_backbone=True)
feature_maps = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(feature_maps))  # 13 feature maps (1 after embedding + 12 blocks)
```

### Fine-Tuning with Classification Head

```python
from kmodels.models.dino_v3 import DinoV3ViTSmall16

model = DinoV3ViTSmall16(
    weights="dinov3",
    include_top=True,
    num_classes=10,
)
# Freeze backbone, train head
for layer in model.layers[:-1]:
    layer.trainable = False
```

### ConvNeXt with Pooling

```python
from kmodels.models.dino_v3 import DinoV3ConvNeXtBase

# Pooled features (B, 1024)
model = DinoV3ConvNeXtBase(weights="dinov3", pooling="avg")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(features.shape)  # (1, 1024)
```
