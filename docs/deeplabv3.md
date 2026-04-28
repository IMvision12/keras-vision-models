# DeepLabV3

**Paper**: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

DeepLabV3 is a highly accurate semantic segmentation model that employs atrous (dilated) convolution to capture multi-scale spatial context without losing spatial resolution. It features an Atrous Spatial Pyramid Pooling (ASPP) module that probes convolutional features at multiple scales, making it highly robust for segmenting objects of varying sizes.

## Architecture Highlights

- **ResNet Backbone:** Leverages deep residual networks (ResNet-50 or ResNet-101) for robust feature extraction.
- **Atrous Convolution:** Controls the resolution of features computed by Deep CNNs and effectively enlarges the field of view of filters without increasing the number of parameters or the amount of computation.
- **ASPP Module:** Captures multi-scale information by applying parallel atrous convolutions with different dilation rates.

## Available Models

| Model | Description | Weights |
|-------|-------------|---------|
| `DeepLabV3ResNet50` | DeepLabV3 with a ResNet-50 backbone | `coco_voc` |
| `DeepLabV3ResNet101` | DeepLabV3 with a ResNet-101 backbone | `coco_voc` |

*Note: The `coco_voc` weights are pre-trained on the COCO dataset and fine-tuned on the PASCAL VOC segmentation dataset. They output predictions across 21 classes (20 objects + 1 background).*

## Basic Usage

```python
import kmodels

# Load model with pre-trained weights
model = kmodels.models.deeplabv3.DeepLabV3ResNet50(weights="coco_voc", input_shape=(512, 512, 3))

# Using ResNet101 backbone
model_large = kmodels.models.deeplabv3.DeepLabV3ResNet101(weights="coco_voc", input_shape=(512, 512, 3))

# Initialize a model without pre-trained weights for custom training
custom_model = kmodels.models.deeplabv3.DeepLabV3ResNet50(
    weights=None,
    input_shape=(512, 512, 3),
    num_classes=21
)
```

## Inference Example

```python
import kmodels
from kmodels.models.deeplabv3 import DeepLabV3ImageProcessor, DeepLabV3PostProcessor

model = kmodels.models.deeplabv3.DeepLabV3ResNet50(weights="coco_voc", input_shape=(512, 512, 3))

processor = DeepLabV3ImageProcessor(size={"height": 512, "width": 512})
image = processor("image.jpg")

output = model(image, training=False)  # Output shape: (1, 512, 512, 21)

result = DeepLabV3PostProcessor(output)
print(f"Detected: {[c for c in result['class_names'] if c != 'background']}")

# Output:
# Detected: ['person']
```

### Data format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
# follow the global config (the default)
processor = DeepLabV3ImageProcessor()
inputs = processor("photo.jpg")

# force channels_first for this call only
processor = DeepLabV3ImageProcessor(data_format="channels_first")
inputs = processor("photo.jpg")
```

Image processors return tensors in the requested layout; post-processors accept tensors in either layout and read the flag to pick the channel axis. See `docs/utils.md` for which families have format-sensitive post-processors.

## Full Inference with Visualization

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmodels.models.deeplabv3 import DeepLabV3ResNet50, DeepLabV3ImageProcessor, DeepLabV3PostProcessor

VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

model = DeepLabV3ResNet50(weights="coco_voc", input_shape=(512, 512, 3))

img = Image.open("image.jpg").convert("RGB")
original_size = img.size[::-1]  # (H, W)

processor = DeepLabV3ImageProcessor(size={"height": 512, "width": 512})
processed = processor(img)
output = model(processed, training=False)

result = DeepLabV3PostProcessor(output, target_size=original_size)
mask_resized = result["segmentation"]

colored_mask = VOC_COLORMAP[mask_resized]
overlay = np.array(img).copy()
alpha = 0.5
mask_pixels = mask_resized > 0
overlay[mask_pixels] = (overlay[mask_pixels] * (1 - alpha) + colored_mask[mask_pixels] * alpha).astype(np.uint8)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.imshow(overlay)

# Add legend
unique_classes = [c for c in result["unique_classes"] if c > 0]
legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=VOC_COLORMAP[c] / 255.0) for c in unique_classes]
if legend_patches:
    ax.legend(legend_patches, [n for c, n in zip(result["unique_classes"], result["class_names"]) if c > 0],
              loc="upper right", fontsize=11)

ax.set_title("DeepLabV3 Semantic Segmentation", fontsize=16)
ax.axis("off")
plt.tight_layout()
fig.savefig("deeplabv3_output.jpg", bbox_inches="tight", dpi=120)
plt.close(fig)
```

![DeepLabV3 Semantic Segmentation Output](../assets/deeplabv3_output.jpg)

## Custom Dataset Usage

When using a model fine-tuned on a custom dataset, pass your class names to the post-processor via `label_names`:

```python
MY_CLASSES = ["background", "crack", "pothole", "patch"]

result = DeepLabV3PostProcessor(output, target_size=original_size, label_names=MY_CLASSES)
```

If `label_names` is not provided, Pascal VOC class names (21 classes) are used by default.
