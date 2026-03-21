# SegFormer

**Paper**: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

SegFormer is a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perceptron (MLP) decoders. It comprises a hierarchically structured Transformer encoder which outputs multiscale features, and a lightweight All-MLP decoder which aggregates information from different layers.

## Model Variants

- **SegFormerB0** — Lightweight (embed_dim: 256, dropout_rate: 0.1)
- **SegFormerB1** — Small (embed_dim: 256, dropout_rate: 0.1)
- **SegFormerB2** — Medium (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB3** — Large (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB4** — Extra Large (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB5** — XXL (embed_dim: 768, dropout_rate: 0.1)

## Available Weights

| Variant | cityscapes_1024 | cityscapes_768 | ade20k_512 |
|---------|:-:|:-:|:-:|
| SegFormerB0 | ✅ | ✅ | ✅ |
| SegFormerB1 | ✅ | | ✅ |
| SegFormerB2 | ✅ | | ✅ |
| SegFormerB3 | ✅ | | ✅ |
| SegFormerB4 | ✅ | | ✅ |
| SegFormerB5 | ✅ | | ✅ |

## Basic Usage

```python
import kmodels

# Pre-Trained weights (cityscapes or ade20k or mit(in1k))
# ade20k and cityscapes can be used for fine-tuning by giving custom `num_classes`
# If `num_classes` is not specified by default for ade20k it will be 150 and for cityscapes it will be 19
model = kmodels.models.segformer.SegFormerB0(weights="ade20k", input_shape=(512,512,3))
model = kmodels.models.segformer.SegFormerB0(weights="cityscapes", input_shape=(512,512,3))

# Fine-Tune using `MiT` backbone (This will load `in1k` weights)
model = kmodels.models.segformer.SegFormerB0(weights="mit", input_shape=(512,512,3))
```

## Custom Backbone Support

```python
import kmodels

# With no backbone weights
backbone = kmodels.models.resnet.ResNet50(as_backbone=True, weights=None, include_top=False, input_shape=(224,224,3))
segformer = kmodels.models.segformer.SegFormerB0(weights=None, backbone=backbone, num_classes=10, input_shape=(224,224,3))

# With backbone weights
backbone = kmodels.models.resnet.ResNet50(as_backbone=True, weights="tv_in1k", include_top=False, input_shape=(224,224,3))
segformer = kmodels.models.segformer.SegFormerB0(weights=None, backbone=backbone, num_classes=10, input_shape=(224,224,3))
```

## Example Inference

```python
import kmodels
from PIL import Image
import numpy as np

model = kmodels.models.segformer.SegFormerB0(weights="ade20k_512")

image = Image.open("ADE_train_00000586.jpg")
processed_img = kmodels.models.segformer.SegFormerImageProcessor(image=image,
    do_resize=True,
    size={"height": 512, "width": 512},
    do_rescale=True,
    do_normalize=True)
outs = model.predict(processed_img)
outs = np.argmax(outs[0], axis=-1)
visualize_segmentation(outs, image)
```
![output](../images/seg_output.png)
