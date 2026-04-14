import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmodels.models.sam import SAM_ViT_Large, SAMGenerateMasks


def overlay_masks(ax, masks_list):
    """Draw every mask as a semi-transparent random-colored overlay."""
    if not masks_list:
        return
    rng = np.random.default_rng(7)
    ordered = sorted(
        [np.asarray(keras.ops.convert_to_numpy(m)).astype(bool) for m in masks_list],
        key=lambda m: -int(m.sum()),
    )
    h, w = ordered[0].shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for mask in ordered:
        color = np.concatenate([rng.random(3), [0.55]])
        overlay[mask] = color
    ax.imshow(overlay)


print("Loading SAM ViT-Large with local sa1b weights...")
model = SAM_ViT_Large(
    input_shape=(1024, 1024, 3),
    weights="sam_vit_large.weights.h5",
)

image_path = "assets/coco_cats.jpg"
img = Image.open(image_path).convert("RGB")
print(f"Image size: {img.size}")
img_np = np.array(img, dtype=np.float32)

print("Running SAMGenerateMasks...")
result = SAMGenerateMasks(
    model,
    img_np,
    points_per_side=16,
    points_per_batch=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crops_nms_thresh=0.7,
    crop_n_layers=0,
)
print(f"Generated {len(result['masks'])} masks")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(np.array(img))
axes[0].set_title("Input", fontsize=13)
axes[0].axis("off")

axes[1].imshow(np.array(img))
overlay_masks(axes[1], result["masks"])
axes[1].set_title(f"SAM ViT-Large — AMG ({len(result['masks'])} masks)", fontsize=13)
axes[1].axis("off")

plt.tight_layout()
out_path = "assets/sam_coco_cats_amg_output.jpg"
fig.savefig(out_path, bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"Saved to {out_path}")
