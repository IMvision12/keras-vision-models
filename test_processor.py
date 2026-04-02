"""Test dfine_preprocess/dfine_postprocess against HF pipeline."""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from kmodels.models.dfine.dfine_image_processor import (
    dfine_postprocess,
    dfine_preprocess,
)
from kmodels.models.dfine.dfine_model import DFineXLarge

# Load models
print("Loading models...")
hf_model = AutoModelForObjectDetection.from_pretrained(
    "ustc-community/dfine-xlarge-coco", trust_remote_code=True
).eval()
hf_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")

km_model = DFineXLarge(
    weights=None, input_shape=(640, 640, 3), num_queries=300, num_labels=80
)
km_model.load_weights("dfine_xlarge_coco.weights.h5")

# Test images
test_images = [
    "images/cat.jpg",
    "images/train.png",
    "images/bird.png",
    "images/space.png",
    "images/valley.png",
]

for img_path in test_images:
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"\n{'=' * 70}")
    print(f"Image: {img_path} ({orig_w}x{orig_h})")

    # ===== HF pipeline =====
    hf_inputs = hf_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        hf_out = hf_model(**hf_inputs)

    hf_results = hf_processor.post_process_object_detection(
        hf_out, threshold=0.5, target_sizes=[(orig_h, orig_w)]
    )
    hf_res = hf_results[0]

    # ===== kmodels pipeline =====
    km_input = dfine_preprocess(img)

    # Verify preprocessing matches HF
    hf_pixel = hf_inputs["pixel_values"].numpy()  # (1, 3, H, W)
    hf_pixel_nhwc = np.transpose(hf_pixel, (0, 2, 3, 1))
    km_input_np = np.array(km_input)
    preproc_diff = np.max(np.abs(hf_pixel_nhwc - km_input_np))
    print(f"  Preprocessing diff (vs HF): {preproc_diff:.8f}")

    with torch.no_grad():
        km_out = km_model(torch.from_numpy(km_input_np))

    km_results = dfine_postprocess(
        km_out, threshold=0.5, target_sizes=[(orig_h, orig_w)]
    )
    km_res = km_results[0]

    # ===== Compare =====
    print(f"\n  {'Source':10s} | {'Class':15s} | {'Score':>8s} | {'Box (xyxy)':>40s}")
    print(f"  {'-' * 10}-+-{'-' * 15}-+-{'-' * 8}-+-{'-' * 40}")

    # HF detections
    for j in range(len(hf_res["scores"])):
        s = hf_res["scores"][j].item()
        l = hf_res["labels"][j].item()
        b = hf_res["boxes"][j].tolist()
        from kmodels.models.dfine.dfine_image_processor import COCO_CLASSES

        cls = COCO_CLASSES[l] if l < len(COCO_CLASSES) else f"cls{l}"
        print(
            f"  {'HF':10s} | {cls:15s} | {s:8.4f} | [{b[0]:8.1f}, {b[1]:8.1f}, {b[2]:8.1f}, {b[3]:8.1f}]"
        )

    # kmodels detections
    km_scores_np = np.array(km_res["scores"])
    km_boxes_np = np.array(km_res["boxes"])
    for j in range(len(km_res["label_names"])):
        s = float(km_scores_np[j])
        cls = km_res["label_names"][j]
        b = km_boxes_np[j].tolist()
        print(
            f"  {'kmodels':10s} | {cls:15s} | {s:8.4f} | [{b[0]:8.1f}, {b[1]:8.1f}, {b[2]:8.1f}, {b[3]:8.1f}]"
        )

    # Match summary
    hf_classes = set()
    for j in range(len(hf_res["scores"])):
        hf_classes.add(COCO_CLASSES[hf_res["labels"][j].item()])
    km_classes = set(km_res["label_names"])

    if hf_classes == km_classes:
        print(f"\n  MATCH: same {len(hf_classes)} class(es) detected")
    else:
        print(f"\n  DIFF: HF={hf_classes}, kmodels={km_classes}")

print(f"\n{'=' * 70}")
print("Done.")
