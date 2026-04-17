# Depth Anything V1 & V2

**V1 Paper**: [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891)
**V2 Paper**: [Depth Anything V2](https://arxiv.org/abs/2406.09414)

Depth Anything is a monocular depth-estimation model family that pairs a
DINOv2 ViT backbone with a DPT-style neck and head. V1 trains on a mix of
labeled images and very large-scale pseudo-labeled images; V2 keeps the same
architecture but replaces the labeled real images with synthetic data and
scales up the teacher model to produce noticeably sharper and more robust
depth maps. The V2 release also ships metric-depth variants fine-tuned for
indoor and outdoor scenes.

Both versions share the same Keras implementation — the `DepthAnythingV1`
class hosts the architecture, and `DepthAnythingV2*` factories instantiate it
with the V2 per-variant hyperparameters and weights.

## Architecture

1. **DINOv2 backbone** (`depth_anything_v1_dino_backbone`) — patch embed +
   CLS token + position embeddings + `backbone_depth` pre-norm transformer
   blocks with LayerScale on both branches. Returns four intermediate
   feature maps at the block indices listed in `out_indices`.
2. **DPT neck** (`depth_anything_v1_neck`) — reassemble (1x1 projection +
   per-factor up/down sampling), project to `fusion_hidden_size` with 3x3
   convs, and walk the pyramid bottom-up through four fusion stages.
3. **Depth head** (`depth_anything_v1_head`) — three convs with an
   aligned-corners bilinear upsample to the input resolution between the
   first and second conv. Relative variants end in a `ReLU`, metric
   variants end in a `sigmoid` scaled by `max_depth`.

The fusion and head upsamples use a pure-Keras
`depth_anything_v1_aligned_bilinear_resize` that matches
`torch.nn.functional.interpolate(..., align_corners=True)` via explicit
gather + lerp, so the model is numerically consistent across `torch`,
`jax`, and `tensorflow` backends and respects
`keras.config.image_data_format()` end-to-end.

## Available Models

### Relative Depth (V1 and V2)

| Variant                | Parameters | Backbone       | Description                              |
|------------------------|-----------:|----------------|------------------------------------------|
| `DepthAnythingV1Small` |     ~24 M  | DINOv2 ViT-S/14 | Smallest / fastest relative-depth model |
| `DepthAnythingV1Base`  |     ~97 M  | DINOv2 ViT-B/14 | Balanced speed / accuracy                |
| `DepthAnythingV1Large` |    ~335 M  | DINOv2 ViT-L/14 | Most accurate V1 relative-depth model    |
| `DepthAnythingV2Small` |     ~24 M  | DINOv2 ViT-S/14 | V2 retrained small variant               |
| `DepthAnythingV2Base`  |     ~97 M  | DINOv2 ViT-B/14 | V2 retrained base variant                |
| `DepthAnythingV2Large` |    ~335 M  | DINOv2 ViT-L/14 | V2 retrained large variant               |

### Metric Depth (V2 only)

| Variant                              | Max depth | Description                            |
|--------------------------------------|----------:|----------------------------------------|
| `DepthAnythingV2MetricIndoorSmall`   |    20 m   | Indoor metric depth (NYUv2 fine-tuned) |
| `DepthAnythingV2MetricIndoorBase`    |    20 m   | Indoor metric depth                    |
| `DepthAnythingV2MetricIndoorLarge`   |    20 m   | Indoor metric depth                    |
| `DepthAnythingV2MetricOutdoorSmall`  |    80 m   | Outdoor metric depth (KITTI-style)     |
| `DepthAnythingV2MetricOutdoorBase`   |    80 m   | Outdoor metric depth                   |
| `DepthAnythingV2MetricOutdoorLarge`  |    80 m   | Outdoor metric depth                   |

All variants default to a 518×518 input (37x37 DINOv2 patch grid).

## Available Weights

| Variant                             | da_v1 | da_v2 |
|-------------------------------------|:-----:|:-----:|
| `DepthAnythingV1Small`              |  ✅   |       |
| `DepthAnythingV1Base`               |  ✅   |       |
| `DepthAnythingV1Large`              |  ✅   |       |
| `DepthAnythingV2Small`              |       |  ✅   |
| `DepthAnythingV2Base`               |       |  ✅   |
| `DepthAnythingV2Large`              |       |  ✅   |
| `DepthAnythingV2MetricIndoorSmall`  |       |  ✅   |
| `DepthAnythingV2MetricIndoorBase`   |       |  ✅   |
| `DepthAnythingV2MetricIndoorLarge`  |       |  ✅   |
| `DepthAnythingV2MetricOutdoorSmall` |       |  ✅   |
| `DepthAnythingV2MetricOutdoorBase`  |       |  ✅   |
| `DepthAnythingV2MetricOutdoorLarge` |       |  ✅   |

## Image Processor

Both `kmodels.models.depth_anything_v1` and
`kmodels.models.depth_anything_v2` ship a pure-Keras image processor that
resizes an input image with bicubic interpolation, rescales to `[0, 1]`,
and applies ImageNet normalization. Unlike HF `DPTImageProcessor` — which
preserves the aspect ratio and produces a variable-shape output — this
processor stretches the image directly to the target size so the shape
matches what the Keras model was built with.

- `DepthAnythingV1ImageProcessor(image, target_size=518)` / `DepthAnythingV2ImageProcessor(...)`
- `DepthAnythingV1PostProcessDepth(predicted_depth, original_size)` /
  `DepthAnythingV2PostProcessDepth(...)`

`target_size` accepts either a single `int` (square output) or a
`(height, width)` tuple. Both dimensions should be multiples of the
DINOv2 patch size (14). The pretrained 518×518 position embeddings are
bilinearly interpolated to the new grid when weights are loaded, so
non-518 inputs work as long as the model was built with the same shape.

```python
import numpy as np
from kmodels.models.depth_anything_v1 import (
    DepthAnythingV1Small,
    DepthAnythingV1ImageProcessor,
    DepthAnythingV1PostProcessDepth,
)

model = DepthAnythingV1Small(weights="da_v1")
inputs = DepthAnythingV1ImageProcessor("photo.jpg")
depth = model(inputs["pixel_values"])
depth_full = DepthAnythingV1PostProcessDepth(
    depth, original_size=inputs["original_size"]
)
print(depth_full.shape)  # (1, orig_h, orig_w)
```

## Basic Usage

### Relative Depth with V1

```python
import numpy as np
from kmodels.models.depth_anything_v1 import (
    DepthAnythingV1Small,
    DepthAnythingV1ImageProcessor,
    DepthAnythingV1PostProcessDepth,
)

model = DepthAnythingV1Small(weights="da_v1")
inputs = DepthAnythingV1ImageProcessor("photo.jpg")
depth = model(inputs["pixel_values"])
depth_full = DepthAnythingV1PostProcessDepth(
    depth, original_size=inputs["original_size"]
)
```

### Relative Depth with V2

```python
import numpy as np
from kmodels.models.depth_anything_v2 import (
    DepthAnythingV2Base,
    DepthAnythingV2ImageProcessor,
    DepthAnythingV2PostProcessDepth,
)

model = DepthAnythingV2Base(weights="da_v2")
inputs = DepthAnythingV2ImageProcessor("photo.jpg")
depth = model(inputs["pixel_values"])
depth_full = DepthAnythingV2PostProcessDepth(
    depth, original_size=inputs["original_size"]
)
```

### Metric Indoor Depth (V2)

```python
from kmodels.models.depth_anything_v2 import (
    DepthAnythingV2MetricIndoorLarge,
    DepthAnythingV2ImageProcessor,
    DepthAnythingV2PostProcessDepth,
)

model = DepthAnythingV2MetricIndoorLarge(weights="da_v2")
inputs = DepthAnythingV2ImageProcessor("room.jpg")
depth = model(inputs["pixel_values"])
depth_full = DepthAnythingV2PostProcessDepth(
    depth, original_size=inputs["original_size"]
)
# depth_full values are in meters, bounded to [0, 20]
```

### Metric Outdoor Depth (V2)

```python
from kmodels.models.depth_anything_v2 import (
    DepthAnythingV2MetricOutdoorLarge,
)

model = DepthAnythingV2MetricOutdoorLarge(weights="da_v2")
# ... same processor + post-process flow, depth bounded to [0, 80]
```

## Non-518 Input Shapes

Both versions accept any input shape whose height and width are
multiples of 14. The DINOv2 position embeddings are resampled to the new
patch grid when the pretrained weights are loaded (via
`AddPositionEmbs.load_own_variables`), and the fusion-block upsample
targets are derived from the model's construction-time `input_shape`, so
each instance is locked to the shape you pick at build time.

```python
from kmodels.models.depth_anything_v2 import (
    DepthAnythingV2Small,
    DepthAnythingV2ImageProcessor,
)

# Non-square 392x784 (28x56 patch grid) with pretrained weights
model = DepthAnythingV2Small(
    input_shape=(392, 784, 3),
    weights="da_v2",
)
inputs = DepthAnythingV2ImageProcessor(
    "photo.jpg", target_size=(392, 784)
)
depth = model(inputs["pixel_values"])
```

## Channels-First vs Channels-Last

Both versions follow `keras.config.image_data_format()` end-to-end. The
backbone patch embed, neck convs, head convs, and the aligned-corners
bilinear upsample all dispatch on the global data format, so switching
between `channels_last` and `channels_first` requires no manual
transposes.

```python
import keras
keras.config.set_image_data_format("channels_first")

from kmodels.models.depth_anything_v1 import DepthAnythingV1Small
model = DepthAnythingV1Small(weights="da_v1")
# model input: (B, 3, 518, 518)  /  output: (B, 1, 518, 518)
```

Weight conversion always runs in `channels_last` + torch backend.

### Data format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
# follow the global config (the default)
inputs = DepthAnythingV1ImageProcessor("photo.jpg")

# force channels_first for this call only
inputs = DepthAnythingV1ImageProcessor("photo.jpg", data_format="channels_first")
```

Image processors return tensors in the requested layout; post-processors accept tensors in either layout and read the flag to pick the channel axis. See `docs/utils.md` for which families have format-sensitive post-processors.

## Model Outputs

- **Relative variants** return non-negative disparity-style depth. Useful
  for depth ordering, monocular SLAM initialization, and cases where only
  relative depth is needed.
- **Metric indoor variants** return metric depth in meters bounded to
  `[0, 20]`.
- **Metric outdoor variants** return metric depth in meters bounded to
  `[0, 80]`.

Output shape is `(batch, height, width, 1)` in `channels_last` or
`(batch, 1, height, width)` in `channels_first`. Use
`Depth*PostProcessDepth` to resample back to the original image size and
squeeze the channel dimension.

## Citations

```bibtex
@inproceedings{yang2024depth,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```

```bibtex
@article{yang2024depthv2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:2406.09414},
  year={2024}
}
```
