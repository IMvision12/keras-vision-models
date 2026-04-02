# SegFormer

**Paper**: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

SegFormer is a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perceptron (MLP) decoders. It comprises a hierarchically structured Transformer encoder which outputs multiscale features, and a lightweight All-MLP decoder which aggregates information from different layers.

## Model Variants

- **SegFormerB0** — Lightweight (embed_dim: 256, dropout_rate: 0.1)
- **SegFormerB1** — Small (embed_dim: 256, dropout_rate: 0.1)
- **SegFormerB2** — Medium (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB3** — Large (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB4** — Extra Large (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB5** — XXL (embed_dim: 768, dropout_rate: 0.1)

## Available Weights

| Variant | cityscapes_1024 | cityscapes_768 | ade20k_512 |
|---------|:-:|:-:|:-:|
| SegFormerB0 | ✅ | ✅ | ✅ |
| SegFormerB1 | ✅ | | ✅ |
| SegFormerB2 | ✅ | | ✅ |
| SegFormerB3 | ✅ | | ✅ |
| SegFormerB4 | ✅ | | ✅ |
| SegFormerB5 | ✅ | | ✅ |

## Basic Usage

```python
import kmodels

# Pre-Trained weights (cityscapes or ade20k or mit(in1k))
# ade20k and cityscapes can be used for fine-tuning by giving custom `num_classes`
# If `num_classes` is not specified by default for ade20k it will be 150 and for cityscapes it will be 19
model = kmodels.models.segformer.SegFormerB0(weights="ade20k", input_shape=(512,512,3))
model = kmodels.models.segformer.SegFormerB0(weights="cityscapes", input_shape=(512,512,3))

# Fine-Tune using `MiT` backbone (This will load `in1k` weights)
model = kmodels.models.segformer.SegFormerB0(weights="mit", input_shape=(512,512,3))
```

## Custom Backbone Support

```python
import kmodels

# With no backbone weights
backbone = kmodels.models.resnet.ResNet50(as_backbone=True, weights=None, include_top=False, input_shape=(224,224,3))
segformer = kmodels.models.segformer.SegFormerB0(weights=None, backbone=backbone, num_classes=10, input_shape=(224,224,3))

# With backbone weights
backbone = kmodels.models.resnet.ResNet50(as_backbone=True, weights="tv_in1k", include_top=False, input_shape=(224,224,3))
segformer = kmodels.models.segformer.SegFormerB0(weights=None, backbone=backbone, num_classes=10, input_shape=(224,224,3))
```

## Example Inference

```python
import kmodels
from kmodels.models.segformer import SegFormerImageProcessor
from PIL import Image
import numpy as np
import keras

model = kmodels.models.segformer.SegFormerB0(weights="ade20k_512", input_shape=(512, 512, 3))

image = Image.open("image.jpg").convert("RGB")

processed = SegFormerImageProcessor(image=image, do_resize=True,
    size={"height": 512, "width": 512}, do_rescale=True, do_normalize=True)

output = model(processed, training=False)
pred_mask = np.argmax(keras.ops.convert_to_numpy(output[0]), axis=-1)

# ADE20K class names (150 classes)
ADE20K_CLASSES = ["wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth",
    "door", "table", "mountain", "plant", "curtain", "chair", ...]

unique = np.unique(pred_mask)
print(f"Detected classes: {[ADE20K_CLASSES[c] for c in unique if c < len(ADE20K_CLASSES)]}")

# Output:
# Detected classes: ['wall', 'floor', 'ceiling', 'cabinet', 'person', 'door',
#   'table', 'plant', 'shelf', 'mirror', 'lamp', 'counter', 'sink', 'stove', ...]
```

## Full Inference with Visualization

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmodels.models.segformer import SegFormerB0, SegFormerImageProcessor

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
    "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]

model = SegFormerB0(weights="ade20k_512", input_shape=(512, 512, 3))

img = Image.open("image.jpg").convert("RGB")
original_size = img.size  # (W, H)

processed = SegFormerImageProcessor(image=img, do_resize=True,
    size={"height": 512, "width": 512}, do_rescale=True, do_normalize=True)

output = model(processed, training=False)
pred_mask = np.argmax(keras.ops.convert_to_numpy(output[0]), axis=-1)

# Resize mask to original size
mask_resized = np.array(
    Image.fromarray(pred_mask.astype(np.uint8)).resize(original_size, Image.NEAREST)
)

# Generate colors per class
np.random.seed(42)
colors = np.random.randint(50, 220, size=(150, 3), dtype=np.uint8)

colored_mask = colors[mask_resized % 150]
overlay = np.array(img).copy()
alpha = 0.55
overlay = (overlay * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.imshow(overlay)

# Legend for top classes by area
unique_classes = np.unique(mask_resized)
class_areas = [(c, (mask_resized == c).sum()) for c in unique_classes]
class_areas.sort(key=lambda x: -x[1])
top_classes = [c for c, _ in class_areas[:8]]

legend_patches = []
legend_names = []
for c in top_classes:
    color = colors[c % 150] / 255.0
    patch = plt.Rectangle((0, 0), 1, 1, fc=color)
    legend_patches.append(patch)
    legend_names.append(ADE20K_CLASSES[c] if c < len(ADE20K_CLASSES) else f"class_{c}")
ax.legend(legend_patches, legend_names, loc="upper right", fontsize=10)

ax.set_title("SegFormer Semantic Segmentation (ADE20K)", fontsize=16)
ax.axis("off")
plt.tight_layout()
fig.savefig("segformer_output.jpg", bbox_inches="tight", dpi=120)
plt.close(fig)
```

![SegFormer Semantic Segmentation Output](../assets/segformer_output.jpg)
