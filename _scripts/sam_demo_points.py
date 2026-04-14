import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmodels.models.sam import (
    SAM_ViT_Large,
    SAMImageProcessorWithPrompts,
    SAMPostProcessMasks,
)

COLORS = [
    np.array([0, 180, 255, 150]) / 255.0,  # cyan — left cat
    np.array([255, 90, 60, 150]) / 255.0,  # orange — right cat
]


def show_mask(mask, ax, color):
    h, w = mask.shape
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_points(coords, ax, color, marker_size=340):
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        color=color,
        marker="*",
        s=marker_size,
        edgecolors="white",
        linewidths=1.25,
        zorder=5,
    )


print("Loading SAM ViT-Large with local sa1b weights...")
model = SAM_ViT_Large(
    input_shape=(1024, 1024, 3),
    weights="sam_vit_large.weights.h5",
)

image_path = "assets/coco_cats.jpg"
img = Image.open(image_path).convert("RGB")
print(f"Image size: {img.size}")

prompts = [
    {"points": np.array([[[150, 200]]]), "labels": np.array([[1]]), "name": "left cat"},
    {
        "points": np.array([[[440, 180]]]),
        "labels": np.array([[1]]),
        "name": "right cat",
    },
]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(np.array(img))

for i, prompt in enumerate(prompts):
    inputs = SAMImageProcessorWithPrompts(
        img,
        input_points=prompt["points"],
        input_labels=prompt["labels"],
    )
    outputs = model.predict(
        {
            "pixel_values": inputs["pixel_values"],
            "input_points": inputs["input_points"],
            "input_labels": inputs["input_labels"],
        },
        verbose=0,
    )

    masks = SAMPostProcessMasks(
        outputs["pred_masks"],
        original_size=inputs["original_size"],
        reshaped_size=inputs["reshaped_size"],
    )

    masks_np = keras.ops.convert_to_numpy(masks)[0, 0]
    iou_scores = keras.ops.convert_to_numpy(outputs["iou_scores"])[0, 0]
    best_idx = int(np.argmax(iou_scores))
    best_mask = masks_np[best_idx] > 0.0

    color = COLORS[i]
    show_mask(best_mask, ax, color)
    show_points(prompt["points"][0], ax, color=color[:3])
    print(f"  {prompt['name']:14s} IoU={iou_scores[best_idx]:.3f}")

ax.set_title("SAM ViT-Large — Point Prompts (COCO cats)", fontsize=14)
ax.axis("off")
plt.tight_layout()
out_path = "assets/sam_coco_cats_output.jpg"
fig.savefig(out_path, bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"Saved to {out_path}")
