# EoMT

**Paper**: [Encoder-only Mask Transformer for Panoptic Segmentation](https://arxiv.org/abs/2504.07957)

EoMT (Encoder-only Mask Transformer) is a panoptic segmentation model that simplifies the standard encoder-decoder mask transformer pipeline by using only an encoder architecture. This approach reduces complexity while maintaining competitive segmentation performance.

## Basic Usage

```python
import kmodels

model = kmodels.models.eomt.EoMT(weights="coco_panoptic", input_shape=(640, 640, 3))

# Without pre-trained weights
model = kmodels.models.eomt.EoMT(weights=None, input_shape=(640, 640, 3))
```
