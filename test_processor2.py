"""Test preprocessor and postprocessor against HF (using HF model for inference)."""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from keras import ops
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from kmodels.models.dfine.dfine_image_processor import (
    COCO_CLASSES,
    dfine_postprocess,
    dfine_preprocess,
)

# Load HF model and processor
print("Loading HF model...")
hf_model = AutoModelForObjectDetection.from_pretrained(
    "ustc-community/dfine-small-coco", trust_remote_code=True
).eval()
hf_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-small-coco")

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

    # ===== 1. Compare preprocessing =====
    hf_inputs = hf_processor(images=img, return_tensors="pt")
    hf_pixel = hf_inputs["pixel_values"].numpy()  # (1, 3, H, W)
    hf_pixel_nhwc = np.transpose(hf_pixel, (0, 2, 3, 1))  # (1, H, W, 3)

    km_input = dfine_preprocess(img)
    km_input_np = ops.convert_to_numpy(km_input)

    preproc_diff = np.max(np.abs(hf_pixel_nhwc - km_input_np))
    print(f"  Preprocess diff: {preproc_diff:.8f}")
    print(f"  HF shape: {hf_pixel.shape}, ours: {km_input_np.shape}")
    print(f"  HF range: [{hf_pixel.min():.4f}, {hf_pixel.max():.4f}]")
    print(f"  Ours range: [{km_input_np.min():.4f}, {km_input_np.max():.4f}]")

    # ===== 2. Run HF model, compare postprocessing =====
    with torch.no_grad():
        hf_out = hf_model(**hf_inputs)

    # HF postprocess
    hf_results = hf_processor.post_process_object_detection(
        hf_out, threshold=0.5, target_sizes=[(orig_h, orig_w)]
    )
    hf_res = hf_results[0]

    # Our postprocess on the SAME model outputs
    outputs_for_ours = {
        "logits": hf_out.logits,
        "pred_boxes": hf_out.pred_boxes,
    }
    km_results = dfine_postprocess(
        outputs_for_ours, threshold=0.5, target_sizes=[(orig_h, orig_w)]
    )
    km_res = km_results[0]

    # Compare
    n_hf = len(hf_res["scores"])
    n_km = len(km_res["scores"])

    print(f"\n  HF detections: {n_hf}, Ours: {n_km}")
    print(f"\n  {'Src':6s} | {'Class':15s} | {'Score':>8s} | {'Box':>45s}")
    print(f"  {'-' * 6}-+-{'-' * 15}-+-{'-' * 8}-+-{'-' * 45}")

    for j in range(n_hf):
        s = hf_res["scores"][j].item()
        l = hf_res["labels"][j].item()
        b = hf_res["boxes"][j].tolist()
        cls = COCO_CLASSES[l] if l < len(COCO_CLASSES) else f"cls{l}"
        print(
            f"  {'HF':6s} | {cls:15s} | {s:8.4f} | [{b[0]:9.2f}, {b[1]:9.2f}, {b[2]:9.2f}, {b[3]:9.2f}]"
        )

    km_scores_np = ops.convert_to_numpy(km_res["scores"])
    km_boxes_np = ops.convert_to_numpy(km_res["boxes"])
    km_labels_np = ops.convert_to_numpy(km_res["labels"])
    for j in range(n_km):
        s = float(km_scores_np[j])
        cls = km_res["label_names"][j]
        b = km_boxes_np[j].tolist()
        print(
            f"  {'Ours':6s} | {cls:15s} | {s:8.4f} | [{b[0]:9.2f}, {b[1]:9.2f}, {b[2]:9.2f}, {b[3]:9.2f}]"
        )

    # Quantitative comparison
    if n_hf > 0 and n_km > 0:
        min_n = min(n_hf, n_km)
        hf_s = np.array([hf_res["scores"][j].item() for j in range(min_n)])
        km_s = np.array([float(km_scores_np[j]) for j in range(min_n)])
        hf_b = np.array([hf_res["boxes"][j].tolist() for j in range(min_n)])
        km_b = np.array([km_boxes_np[j].tolist() for j in range(min_n)])
        hf_l = np.array([hf_res["labels"][j].item() for j in range(min_n)])
        km_l = np.array([int(km_labels_np[j]) for j in range(min_n)])

        print(f"\n  Score diff (max):  {np.max(np.abs(hf_s - km_s)):.8f}")
        print(f"  Box diff (max):    {np.max(np.abs(hf_b - km_b)):.4f}")
        print(f"  Labels match:      {np.all(hf_l == km_l)}")

print(f"\n{'=' * 70}")
print("Done.")
