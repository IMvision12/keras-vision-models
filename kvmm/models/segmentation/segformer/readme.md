# SegFormer Model

## Overview

**Paper**: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

SegFormer is a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perceptron (MLP) decoders. SegFormer comprises two main components: (1) a novel hierarchically structured Transformer encoder which outputs multiscale features, and (2) a lightweight All-MLP decoder which aggregates information from different layers to render the final segmentation prediction.

This implementation provides Keras/TensorFlow models for SegFormer with various architectures, enabling semantic segmentation tasks on popular datasets like Cityscapes and ADE20K.

## 🏗️ Model Variants

- **SegFormerB0** - Lightweight model (embed_dim: 256, dropout_rate: 0.1)
- **SegFormerB1** - Small model (embed_dim: 256, dropout_rate: 0.1)
- **SegFormerB2** - Medium model (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB3** - Large model (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB4** - Extra Large model (embed_dim: 768, dropout_rate: 0.1)
- **SegFormerB5** - XXL model (embed_dim: 768, dropout_rate: 0.1)

## 📊 Available Weights

### SegFormerB0
- **cityscapes_1024**: Trained on Cityscapes dataset for 1024×1024 resolution
- **cityscapes_768**: Trained on Cityscapes dataset for 768×768 resolution
- **ade20k_512**: Trained on ADE20K dataset for 512×512 resolution

### SegFormerB1
- **cityscapes_1024**: Trained on Cityscapes dataset for 1024×1024 resolution
- **ade20k_512**: Trained on ADE20K dataset for 512×512 resolution

### SegFormerB2
- **cityscapes_1024**: Trained on Cityscapes dataset for 1024×1024 resolution
- **ade20k_512**: Trained on ADE20K dataset for 512×512 resolution

### SegFormerB3
- **cityscapes_1024**: Trained on Cityscapes dataset for 1024×1024 resolution
- **ade20k_512**: Trained on ADE20K dataset for 512×512 resolution

### SegFormerB4
- **cityscapes_1024**: Trained on Cityscapes dataset for 1024×1024 resolution
- **ade20k_512**: Trained on ADE20K dataset for 512×512 resolution

### SegFormerB5
- **cityscapes_1024**: Trained on Cityscapes dataset for 1024×1024 resolution
- **ade20k_512**: Trained on ADE20K dataset for 640×640 resolution

## 🛠️ Basic Usage

```python
import kvmm
from PIL import Image
import numpy as np

# Load model with pre-trained weights
model = kvmm.models.segformer.SegFormerB0(weights="ade20k_512")

# Load and preprocess image
image = Image.open("ADE_train_00000586.jpg")
processed_img = kvmm.models.segformer.SegFormerImageProcessor(
   image=image,
   do_resize=True,
   size={"height": 512, "width": 512},
   do_rescale=True,
   do_normalize=True
)

# Run inference
outputs = model.predict(processed_img)
segmentation_map = np.argmax(outputs[0], axis=-1)

# Visualize results
visualize_segmentation(segmentation_map, image)