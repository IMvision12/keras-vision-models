# SAM2 (Segment Anything Model 2)

## Overview

SAM2 (Segment Anything Model 2) is the next generation of the Segment Anything Model, designed for promptable segmentation in both images and videos. It features a Hiera hierarchical vision transformer backbone with improved efficiency and performance. SAM2 introduces object score prediction and high-resolution feature skip connections for enhanced mask quality.

**Reference:** [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) (Ravi et al., 2024)

## Available Models

| Model | Parameters | Backbone | Description | Weights |
|-------|-----------|----------|-------------|---------|
| `Sam2Tiny` | ~38M | Hiera-Tiny | Smallest and fastest variant | `sav` |
| `Sam2Small` | ~46M | Hiera-Small | Balanced speed and accuracy | `sav` |
| `Sam2BasePlus` | ~80M | Hiera-Base+ | Enhanced base model | `sav` |
| `Sam2Large` | ~224M | Hiera-Large | Largest and most accurate | `sav` |

All models use a 1024×1024 input resolution and are trained on the SA-V dataset (Segment Anything in Videos).

## Basic Usage

```python
import kmodels

# List available SAM2 models
print(kmodels.list_models("sam2"))

# Create a SAM2 model (default 1024x1024 input)
model = kmodels.models.sam2.Sam2Tiny(
    input_shape=(1024, 1024, 3),
    weights="sav",
)
```

## Inference with Point Prompts

```python
import numpy as np
import keras

# Load model
model = kmodels.models.sam2.Sam2Large(
    input_shape=(1024, 1024, 3),
    weights="sav",
)

# Prepare inputs
# Note: Image should be preprocessed to 1024x1024
pixel_values = keras.ops.convert_to_tensor(image)  # Shape: (1, 1024, 1024, 3)

# Point prompts: (batch, num_prompts, num_points, 2)
# Coordinates should be in the resized image space (0-1023)
input_points = np.array([[[[450, 600]]]])  # (x, y) pixel coordinates

# Labels: 1 = foreground, 0 = background, -1 = padding
input_labels = np.array([[[1]]])

# Run inference
outputs = model({
    "pixel_values": pixel_values,
    "input_points": input_points,
    "input_labels": input_labels,
})

# Extract outputs
masks = outputs["pred_masks"]  # Shape: (1, 1, 3, 256, 256)
iou_scores = outputs["iou_scores"]  # Shape: (1, 1, 3)
object_scores = outputs["object_score_logits"]  # Shape: (1, 1, 1)

print(f"Masks shape: {masks.shape}")
print(f"IoU scores: {iou_scores}")
print(f"Object score logits: {object_scores}")
```

## Inference with Box Prompts

```python
import numpy as np

model = kmodels.models.sam2.Sam2Small(
    input_shape=(1024, 1024, 3),
    weights="sav",
)

# Box prompts are encoded as two corner points
# Box format: [x1, y1, x2, y2] -> converted to points
box = np.array([100, 200, 400, 500])

# Convert box to point format: top-left and bottom-right corners
input_points = np.array([[
    [box[0], box[1]],  # top-left
    [box[2], box[3]],  # bottom-right
]])
input_points = np.expand_dims(input_points, axis=0)  # Add batch dim

# Labels: 2 = top-left corner, 3 = bottom-right corner
input_labels = np.array([[[2, 3]]])

outputs = model({
    "pixel_values": pixel_values,
    "input_points": input_points,
    "input_labels": input_labels,
})
```

## Multiple Point Prompts

```python
import numpy as np

# Multiple points for a single object
input_points = np.array([[
    [[450, 600], [500, 650], [400, 550]]  # 3 points
]])

# 1 = foreground, 0 = background
input_labels = np.array([[[1, 1, 0]]])  # 2 foreground, 1 background

outputs = model({
    "pixel_values": pixel_values,
    "input_points": input_points,
    "input_labels": input_labels,
})
```

## Batch Processing Multiple Prompts

```python
import numpy as np

# Process multiple prompts in parallel
# Shape: (batch=1, num_prompts=3, num_points=1, 2)
input_points = np.array([[
    [[100, 200]],  # First prompt
    [[300, 400]],  # Second prompt
    [[500, 600]],  # Third prompt
]])

input_labels = np.array([[[1], [1], [1]]])

outputs = model({
    "pixel_values": pixel_values,
    "input_points": input_points,
    "input_labels": input_labels,
})

# Output shapes:
# masks: (1, 3, 3, 256, 256) - 3 prompts, 3 masks each
# iou_scores: (1, 3, 3) - 3 prompts, 3 scores each
```

## Architecture

SAM2 consists of three main components:

1. **Hiera Backbone (Image Encoder):** A hierarchical vision transformer with:
   - Multi-scale blocks with windowed and global attention
   - Query pooling at stage transitions for efficiency
   - Windowed positional embeddings
   - FPN (Feature Pyramid Network) neck with sine-cosine positional encodings
   - Processes 1024×1024 input into multi-scale features (64×64, 128×128, 256×256)

2. **Prompt Encoder:** Encodes sparse prompts (points, boxes) via random Fourier feature positional encoding with learned type embeddings, and dense prompts (masks) via a small CNN. Shared positional embedding layer with image encoder.

3. **Mask Decoder:** A lightweight two-way transformer (2 layers) that:
   - Jointly attends between prompt tokens and image embeddings
   - Uses high-resolution feature skip connections from FPN
   - Generates mask predictions via hypernetwork MLPs
   - Predicts IoU confidence scores (with sigmoid activation)
   - Predicts object-presence scores (logits)

## Model Outputs

The model returns a dictionary with:
- `pred_masks`: Predicted masks of shape `(batch, num_prompts, num_multimask_outputs, 256, 256)`
  - By default, `num_multimask_outputs=3` (excludes the single-mask output)
  - Masks are at 4× the image embedding resolution
- `iou_scores`: Predicted IoU scores (0-1) for each mask of shape `(batch, num_prompts, num_multimask_outputs)`
  - Sigmoid-activated confidence scores
- `object_score_logits`: Object presence score logits of shape `(batch, num_prompts, 1)`
  - Raw logits indicating whether a valid object is present

## Key Improvements over SAM v1

1. **Hiera Backbone:** More efficient hierarchical architecture with better speed/accuracy tradeoff
2. **Multi-scale Features:** FPN provides features at multiple resolutions with skip connections
3. **Object Score Prediction:** Additional head to predict object presence
4. **Improved Mask Quality:** High-resolution skip connections enhance fine details
5. **Video Support:** Architecture designed for temporal consistency (video mode not yet implemented in this release)

## Prompt Label Convention

- `1`: Foreground point
- `0`: Background point
- `2`: Box top-left corner
- `3`: Box bottom-right corner
- `-1`: Padding (ignored)
- `-10`: Zero embedding (special case)

## Performance Tips

1. **Model Selection:**
   - Use `Sam2Tiny` for real-time applications
   - Use `Sam2Small` for balanced performance
   - Use `Sam2Large` for highest quality

2. **Prompt Strategy:**
   - Start with a single foreground point
   - Add background points to refine boundaries
   - Use boxes for objects with clear rectangular bounds
   - Combine multiple points for complex shapes

3. **Mask Selection:**
   - The model outputs 3 masks by default
   - Use `iou_scores` to select the best mask
   - Higher IoU score indicates better mask quality

## Example: Complete Workflow

```python
import numpy as np
import keras
from PIL import Image

# Load and preprocess image
image = Image.open("photo.jpg").convert("RGB")
original_size = image.size  # (width, height)

# Resize to 1024x1024, rescale, and apply ImageNet normalization
resized = image.resize((1024, 1024))
pixel_values = np.array(resized).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
pixel_values = (pixel_values - mean) / std
pixel_values = keras.ops.convert_to_tensor(
    np.expand_dims(pixel_values, axis=0), dtype="float32"
)

# Load model
model = kmodels.models.sam2.Sam2Tiny(
    input_shape=(1024, 1024, 3),
    weights="sav",
)

# Define prompts (single foreground point in 1024x1024 space)
input_points = keras.ops.convert_to_tensor(
    np.array([[[[512, 512]]]], dtype=np.float32)
)
input_labels = keras.ops.convert_to_tensor(
    np.array([[[1]]], dtype=np.int32)
)

# Run inference
outputs = model({
    "pixel_values": pixel_values,
    "input_points": input_points,
    "input_labels": input_labels,
})

# Select best mask based on IoU score
masks = keras.ops.convert_to_numpy(outputs["pred_masks"])[0, 0]  # Shape: (3, 256, 256)
iou_scores = keras.ops.convert_to_numpy(outputs["iou_scores"])[0, 0]  # Shape: (3,)
best_mask_idx = np.argmax(iou_scores)
best_mask = masks[best_mask_idx]

# Resize mask to original image size
best_mask_resized = np.array(
    Image.fromarray((best_mask > 0).astype(np.uint8) * 255).resize(
        original_size, Image.BILINEAR
    )
) > 128

print(f"Best mask IoU score: {iou_scores[best_mask_idx]:.3f}")
print(f"Object score: {keras.ops.sigmoid(outputs['object_score_logits'][0, 0, 0]):.3f}")
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

from kmodels.models.sam2 import Sam2Tiny

COLORS = [
    np.array([0, 180, 255, 128]) / 255.0,    # cyan
    np.array([255, 100, 50, 128]) / 255.0,    # orange
    np.array([50, 220, 100, 128]) / 255.0,    # green
]


def show_mask(mask, ax, color):
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, ax, color, marker_size=375):
    ax.scatter(coords[0], coords[1], color=color, marker="*",
               s=marker_size, edgecolors="white", linewidths=1.25, zorder=5)


model = Sam2Tiny(input_shape=(1024, 1024, 3), weights="sav")
img = Image.open("cyclist.jpg").convert("RGB")
original_size = img.size  # (W, H)

# Preprocess: resize to 1024x1024, rescale, ImageNet normalize
resized = img.resize((1024, 1024))
pixel_values = np.array(resized).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
pixel_values = (pixel_values - mean) / std
pixel_values = keras.ops.convert_to_tensor(
    np.expand_dims(pixel_values, axis=0), dtype="float32"
)

# Point prompts for 3 objects (coordinates in 1024x1024 space)
prompts = [
    {"points": np.array([[[[426, 560]]]], dtype=np.float32), "labels": np.array([[[1]]], dtype=np.int32), "name": "cyclist"},
    {"points": np.array([[[[832, 736]]]], dtype=np.float32), "labels": np.array([[[1]]], dtype=np.int32), "name": "dog"},
    {"points": np.array([[[[387, 928]]]], dtype=np.float32), "labels": np.array([[[1]]], dtype=np.int32), "name": "bench"},
]

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.imshow(np.array(img))

for i, prompt in enumerate(prompts):
    input_points = keras.ops.convert_to_tensor(prompt["points"], dtype="float32")
    input_labels = keras.ops.convert_to_tensor(prompt["labels"], dtype="int32")

    outputs = model({
        "pixel_values": pixel_values,
        "input_points": input_points,
        "input_labels": input_labels,
    })

    masks = keras.ops.convert_to_numpy(outputs["pred_masks"])[0, 0]
    iou_scores = keras.ops.convert_to_numpy(outputs["iou_scores"])[0, 0]
    best_idx = np.argmax(iou_scores)
    best_mask = masks[best_idx]

    mask_pil = Image.fromarray((best_mask > 0).astype(np.uint8) * 255).resize(
        original_size, Image.BILINEAR
    )
    best_mask_full = np.array(mask_pil) > 128

    color = COLORS[i]
    show_mask(best_mask_full, ax, color)

    pt = prompt["points"][0, 0, 0]
    pt_orig = (pt[0] * original_size[0] / 1024, pt[1] * original_size[1] / 1024)
    show_points(pt_orig, ax, color=color[:3])
    print(f"  {prompt['name']}: IoU={iou_scores[best_idx]:.3f}")

ax.set_title("SAM2 — Multiple Point Prompts", fontsize=16)
ax.axis("off")
plt.tight_layout()
fig.savefig("sam2_output.jpg", bbox_inches="tight", dpi=120)
plt.close(fig)
```

![SAM2 Multiple Point Prompts Output](../assets/sam2_output.jpg)

## Citation

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```
