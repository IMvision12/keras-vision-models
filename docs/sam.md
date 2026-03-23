# SAM (Segment Anything Model)

## Overview

SAM (Segment Anything Model) is a promptable segmentation model that can generate high-quality segmentation masks for any object in an image, given input prompts such as points, bounding boxes, or masks. It was trained on the SA-1B dataset containing over 1 billion masks on 11 million images.

**Reference:** [Segment Anything](https://arxiv.org/abs/2304.02643) (Kirillov et al., 2023)

## Available Models

| Model | Parameters | Description | Weights |
|-------|-----------|-------------|---------|
| `SAM_ViT_Base` | ~93M | ViT-B/16 backbone | `sa1b` |
| `SAM_ViT_Large` | ~308M | ViT-L/16 backbone | `sa1b` |
| `SAM_ViT_Huge` | ~636M | ViT-H/16 backbone | `sa1b` |

## Basic Usage

```python
import kmodels

# List available SAM models
print(kmodels.list_models("sam"))

# Create a SAM model (default 1024x1024 input)
model = kmodels.models.sam.SAM_ViT_Base(
    input_shape=(1024, 1024, 3),
    weights="sa1b",
)
```

## Inference with Point Prompts

```python
import numpy as np
from kmodels.models.sam import SAM_ViT_Huge, SAMImageProcessorWithPrompts, SAMPostProcessMasks

# Load model
model = SAM_ViT_Huge(input_shape=(1024, 1024, 3), weights="sa1b")

# Preprocess image with point prompts
inputs = SAMImageProcessorWithPrompts(
    "photo.jpg",
    input_points=np.array([[[450, 600]]]),  # (x, y) pixel coordinates
    input_labels=np.array([[1]]),            # 1 = foreground
)

# Run inference
outputs = model({
    "pixel_values": inputs["pixel_values"],
    "input_points": inputs["input_points"],
    "input_labels": inputs["input_labels"],
    "input_boxes": inputs["input_boxes"],
    "input_masks": inputs["input_masks"],
})

# Post-process masks to original resolution
masks = SAMPostProcessMasks(
    outputs["pred_masks"],
    original_size=inputs["original_size"],
    reshaped_size=inputs["reshaped_size"],
)

# masks shape: (1, point_batch, num_masks, orig_h, orig_w)
# iou_scores shape: (1, point_batch, num_masks)
print(f"Masks shape: {masks.shape}")
print(f"IoU scores: {outputs['iou_scores']}")
```

## Inference with Box Prompts

```python
import numpy as np
from kmodels.models.sam import SAM_ViT_Huge, SAMImageProcessorWithPrompts, SAMPostProcessMasks

model = SAM_ViT_Huge(input_shape=(1024, 1024, 3), weights="sa1b")

inputs = SAMImageProcessorWithPrompts(
    "photo.jpg",
    input_boxes=np.array([[100, 200, 400, 500]]),  # (x1, y1, x2, y2)
)

outputs = model({
    "pixel_values": inputs["pixel_values"],
    "input_points": inputs["input_points"],
    "input_labels": inputs["input_labels"],
    "input_boxes": inputs["input_boxes"],
    "input_masks": inputs["input_masks"],
})

masks = SAMPostProcessMasks(
    outputs["pred_masks"],
    original_size=inputs["original_size"],
    reshaped_size=inputs["reshaped_size"],
)
```

## Architecture

SAM consists of three main components:

1. **Vision Encoder (Image Encoder):** A ViT backbone with windowed attention and relative positional embeddings. Processes the input image (1024×1024) into a dense feature map (64×64×256).

2. **Prompt Encoder:** Encodes sparse prompts (points, boxes) via positional encoding + learned type embeddings, and dense prompts (masks) via a small CNN.

3. **Mask Decoder:** A lightweight two-way transformer (2 layers) that attends between prompt tokens and image embeddings, then generates mask predictions and IoU confidence scores.

## Model Outputs

The model returns a dictionary with:
- `pred_masks`: Low-resolution predicted masks of shape `(batch, point_batch, num_masks, 256, 256)`
- `iou_scores`: Predicted IoU scores for each mask of shape `(batch, point_batch, num_masks)`

Use `SAMPostProcessMasks` to upscale masks to the original image resolution.
