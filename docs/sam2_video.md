# SAM2 Video (Segment Anything Model 2 — Video)

## Overview

SAM2 Video extends the Segment Anything Model 2 image architecture with the components needed for promptable **video** segmentation: a memory bank of past frames, a memory attention stack that conditions the current frame on previously stored features, a memory encoder that fuses the predicted mask back into a memory representation, and an object pointer projection that lets the model track an object across frames. The image encoder, prompt encoder and mask decoder are shared with the SAM2 image model, so a single set of pretrained weights drives both.

**Reference:** [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) (Ravi et al., 2024)

## Available Models

| Model              | Parameters | Backbone     | Description                       | Weights |
|--------------------|-----------|--------------|-----------------------------------|---------|
| `Sam2VideoTiny`    | ~38M      | Hiera-Tiny   | Smallest and fastest variant      | `sav`   |
| `Sam2VideoSmall`   | ~46M      | Hiera-Small  | Balanced speed and accuracy       | `sav`   |
| `Sam2VideoBasePlus`| ~80M      | Hiera-Base+  | Enhanced base model               | `sav`   |
| `Sam2VideoLarge`   | ~224M     | Hiera-Large  | Largest and most accurate         | `sav`   |

All models use a 1024×1024 input resolution and are trained on the SA-V dataset (Segment Anything in Videos).

## Architecture

`Sam2Video` extends the SAM2 image model with the following extra components:

1. **Hiera Backbone + FPN** — identical to the image model, produces multi-scale image embeddings for each frame.
2. **Prompt Encoder** — encodes sparse (points, boxes) and dense (masks) prompts.
3. **Mask Decoder** — two-way transformer that predicts masks, IoU scores, object-presence logits, and the mask tokens used to derive the object pointer.
4. **Memory Attention** — `Sam2VideoMemoryAttention`, a stack of 4 self-attention + cross-attention + FFN blocks with 2D axial RoPE. Conditions the current frame's features on memory tokens (past frame features) and object pointer tokens for non-cond frames.
5. **Memory Encoder Sub-Model** — `self.memory_encoder_submodel`, built from the functional `sam2_video_memory_encoder` helper. Takes `(vision_features, predicted_mask)` and produces compact memory features stored in the per-object memory bank.
6. **Object Pointer Projection Sub-Model** — `self.obj_ptr_proj_submodel`, built from the functional `sam2_video_ffn` helper. Projects the best mask token from the decoder into a 256-dim object pointer that the memory attention cross-attention consumes.

The main functional graph runs the per-frame encode → prompt → decode path. The dynamic video inference loop in `sam2_video_inference.py` orchestrates the per-frame memory bank state updates and calls the sub-models on demand.

## Basic Usage

```python
import kmodels

# List available SAM2 Video models
print(kmodels.list_models("sam2_video"))

# Create a SAM2 Video model
model = kmodels.models.sam2_video.Sam2VideoTiny(
    input_shape=(1024, 1024, 3),
    weights="sav",
)
```

The model exposes:

- The functional graph `model({"pixel_values": ..., "input_points": ..., "input_labels": ...})` for single-frame image-style inference.
- `model.memory_attention` — `Sam2VideoMemoryAttention` Layer.
- `model.memory_encoder_submodel` — `keras.Model` for `(vision_features, mask) → memory_features`.
- `model.obj_ptr_proj_submodel` — `keras.Model` for `mask_token → object_pointer`.
- `model.no_memory_positional_encoding`, `model.memory_temporal_positional_encoding`, `model.no_object_pointer`, `model.occlusion_spatial_embedding_parameter` — free weights used by the inference loop.
- `model.mask_downsample_layer`, `model.temporal_pos_enc_proj` — single-layer helpers used by the inference loop.

## Image Processor

`kmodels.models.sam2_video` ships a pure-Keras frame processor with the same interface as `kmodels.models.sam2.Sam2ImageProcessor`:

- `Sam2VideoImageProcessor(image)` — preprocess one frame and return default empty prompt placeholders.
- `Sam2VideoImageProcessorWithPrompts(image, input_points, input_labels)` — same as above plus encoded point prompts (per-axis stretched into 1024-space).
- `Sam2VideoPostProcessMasks(pred_masks, original_size)` — bilinear-resize predicted masks back to the original frame resolution.

```python
import numpy as np
from kmodels.models.sam2_video import (
    Sam2VideoSmall,
    Sam2VideoImageProcessorWithPrompts,
    Sam2VideoPostProcessMasks,
)

model = Sam2VideoSmall(input_shape=(1024, 1024, 3), weights="sav")
inputs = Sam2VideoImageProcessorWithPrompts(
    "frame.jpg",
    input_points=np.array([[[450, 600]]], dtype=np.float32),
    input_labels=np.array([[1]], dtype=np.int32),
)
outputs = model({
    "pixel_values": inputs["pixel_values"],
    "input_points": inputs["input_points"],
    "input_labels": inputs["input_labels"],
})
masks = Sam2VideoPostProcessMasks(
    outputs["pred_masks"], original_size=inputs["original_size"]
)
```

## Single-Frame Inference (Image-Style)

When you only need a single frame's masks (no temporal tracking), call `Sam2Video` exactly the same way as the SAM2 image model:

```python
import numpy as np
from kmodels.models.sam2_video import (
    Sam2VideoLarge,
    Sam2VideoImageProcessorWithPrompts,
)

model = Sam2VideoLarge(input_shape=(1024, 1024, 3), weights="sav")

inputs = Sam2VideoImageProcessorWithPrompts(
    "frame.jpg",
    input_points=np.array([[[450, 600]]], dtype=np.float32),
    input_labels=np.array([[1]], dtype=np.int32),
)
outputs = model({
    "pixel_values": inputs["pixel_values"],
    "input_points": inputs["input_points"],
    "input_labels": inputs["input_labels"],
})

masks = outputs["pred_masks"]                   # (1, 1, 3, 256, 256)
iou_scores = outputs["iou_scores"]              # (1, 1, 3)
object_score_logits = outputs["object_score_logits"]  # (1, 1, 1)
```

The functional graph also exposes intermediate tensors that the video inference loop needs: `image_embeddings_raw`, `image_embeddings`, `high_res_feat_s0`, `high_res_feat_s1`, `image_pe`, `sparse_embeddings`, `dense_embeddings`, `mask_tokens_out_all`, `pred_masks_all`, `iou_scores_all`.

## Video Inference (Memory-Conditioned Tracking)

For full video tracking with memory attention, use `Sam2VideoInferenceSession` + `Sam2VideoPredictor` from `kmodels.models.sam2_video.sam2_video_inference`. The session holds the per-object memory bank, the predictor wraps a `Sam2Video` model and runs the dynamic forward loop. Frames are loaded with `cv2.VideoCapture` and preprocessed with `Sam2VideoImageProcessor`.

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import keras
import torch
from tqdm import tqdm

from kmodels.models.sam2_video import (
    Sam2VideoSmall,
    Sam2VideoImageProcessor,
)
from kmodels.models.sam2_video.sam2_video_inference import (
    Sam2VideoInferenceSession,
    Sam2VideoPredictor,
)

VIDEO_PATH = "input.mp4"
OUT_PATH = "output.mp4"
NUM_FRAMES = 60
OUT_FPS = 24
MASK_COLOR = np.array([255, 221, 102], dtype=np.float32)  # BGR light blue
MASK_ALPHA = 0.5


def load_video_frames(path, num_frames):
    """Sample `num_frames` evenly across the video and return RGB arrays."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps else 0.0

    indices = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, height, width, duration


# Load and uniformly sample frames
frames, vh, vw, duration = load_video_frames(VIDEO_PATH, NUM_FRAMES)

# Preprocess each frame with the pure-Keras processor
processed_frames = {}
for i, frame in enumerate(frames):
    proc = Sam2VideoImageProcessor(frame)
    processed_frames[i] = keras.ops.convert_to_numpy(proc["pixel_values"])

# Build the Keras model and predictor
model = Sam2VideoSmall(input_shape=(1024, 1024, 3), weights="sav")
predictor = Sam2VideoPredictor(model)

# Create a session and add a single point prompt at frame 0.
# Coordinates passed to the session are in 1024-space (the model input space),
# so stretch the original-pixel point per-axis.
session = Sam2VideoInferenceSession(
    processed_frames=processed_frames,
    video_height=vh,
    video_width=vw,
)
px_orig, py_orig = vw // 2, vh // 2
px_1024 = px_orig * (1024.0 / vw)
py_1024 = py_orig * (1024.0 / vh)
session.add_point_inputs(
    obj_id=1,
    frame_idx=0,
    point_coords=[[px_1024, py_1024]],
    point_labels=[1],
)

# Propagate through every frame and collect 1024-space masks
masks_by_frame = {}
with torch.no_grad():
    for frame_idx, results in tqdm(
        predictor.propagate_in_video(session),
        total=len(frames),
        desc="Tracking",
    ):
        masks_by_frame[frame_idx] = results[0]["high_res_masks"][0, :, :, 0].copy()

# Overlay masks on the source frames and save as mp4
repeat_per_frame = max(1, int(round(OUT_FPS * duration / max(len(frames), 1))))
writer = cv2.VideoWriter(
    OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (vw, vh)
)

for i, rgb in enumerate(frames):
    mask_1024 = masks_by_frame[i]
    mask_vid = cv2.resize(
        mask_1024.astype(np.float32), (vw, vh), interpolation=cv2.INTER_LINEAR
    )
    mask_bool = (mask_vid > 0).astype(np.float32)[..., None]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    color = np.broadcast_to(MASK_COLOR, bgr.shape)
    blended = bgr * (1 - mask_bool * MASK_ALPHA) + color * (mask_bool * MASK_ALPHA)
    frame_out = np.clip(blended, 0, 255).astype(np.uint8)
    for _ in range(repeat_per_frame):
        writer.write(frame_out)

writer.release()
```

### Inference API Summary

- **`Sam2VideoInferenceSession(processed_frames, video_height, video_width)`** — holds the preprocessed frames and the per-object memory bank. `processed_frames` is a `dict[int, np.ndarray]` of `(1, 1024, 1024, 3)` channels-last (or `(1, 3, 1024, 1024)` channels-first) preprocessed tensors.
- **`session.add_point_inputs(obj_id, frame_idx, point_coords, point_labels)`** — register a point prompt for an object. Coordinates are in 1024-space (the model input space). Internally creates an entry in `point_inputs_per_obj` and assigns an integer `obj_idx` to the supplied `obj_id`.
- **`Sam2VideoPredictor(sam2_video_model)`** — wraps the `Sam2Video` model and exposes `run_frame(session, frame_idx, is_init_cond_frame)` for one-shot frame processing and `propagate_in_video(session, start_frame_idx=0)` for an end-to-end iterator that yields `(frame_idx, results)` tuples.

### Per-Frame Output Dictionary

Each frame's per-object result returned by the predictor (and stored in `session.output_dict_per_obj[obj_idx]`) contains:

- `pred_masks` — best mask logits at decoder resolution `(1, 1, 1, 256, 256)`.
- `high_res_masks` — best mask logits upsampled to 1024×1024, in the model's data format.
- `object_pointer` — `(1, 1, 256)` projection of the best mask token, used as a memory-attention cross-attention key on later frames.
- `object_score_logits` — `(1, 1, 1)` object-presence logits.
- `maskmem_features` — output of `model.memory_encoder_submodel`, the memory bank tensor for this frame.
- `maskmem_pos_enc` — sine positional encoding paired with `maskmem_features`.

## Memory Bank Layout

`session.output_dict_per_obj[obj_idx]` is split into two dicts indexed by `frame_idx`:

- `cond_frame_outputs` — frames that were prompt frames (received explicit point/box/mask inputs). Memory attention always attends to these regardless of how far back they are.
- `non_cond_frame_outputs` — propagated frames. Memory attention attends to up to the `num_maskmem - 1` (default 6) most-recent non-cond frames.

The predictor maintains up to `MAX_OBJECT_POINTERS_IN_ENCODER = 16` past object pointers, prioritizing cond pointers and then filling the remaining slots with the most-recent non-cond pointers.

## Channels-First vs Channels-Last

`Sam2Video` follows `keras.config.image_data_format()` end-to-end:

- `keras.config.set_image_data_format("channels_last")` (Keras default) — the main graph runs `(B, H, W, C)`, the memory encoder sub-model accepts `(B, 64, 64, 256)` features and `(B, 1024, 1024, 1)` masks.
- `keras.config.set_image_data_format("channels_first")` — the main graph runs `(B, C, H, W)`, the memory encoder sub-model accepts `(B, 256, 64, 64)` features and `(B, 1, 1024, 1024)` masks.

The video inference loop in `sam2_video_inference.py` reads the global format on every frame and routes tensors accordingly, so no manual transposes are required when you switch formats. **Weight conversion always runs in channels-last + torch backend.**

## Performance Tips

1. **Frame sampling** — the predictor can run on every native frame, but on CPU each step is dominated by the encoder + memory attention. For demo/preview videos, sample 30–100 frames evenly with `cv2.VideoCapture` (see `load_video_frames` in the example above) and use `torch.no_grad()` plus `gc.collect()` per frame to keep memory bounded on long clips.
2. **Output playback** — if your sampled frame count is much smaller than the source frame rate, the output video will play as a slideshow. Encoding at a normal output fps (e.g. 24) and repeating each tracked frame N times keeps playback smooth in any player.
3. **Init-cond placement** — the cond frame (the one with the prompt) is the only frame the rest of the bank is anchored on for the early steps, so pick a frame where the target is clearly visible.
4. **Multi-object tracking** — call `session.add_point_inputs(obj_id=...)` multiple times before propagating; each `obj_id` gets its own memory bank entry and propagation runs them all per frame in `predictor.run_frame`.

## Citation

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```
