# KVMM: Keras Vision Models 🚀

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Keras](https://img.shields.io/badge/keras-v3.5.0+-success.svg)](https://github.com/keras-team/keras)
![Python](https://img.shields.io/badge/python-v3.10.0+-success.svg)

## 📌 Table of Contents

- 📖 [Introduction](#introduction)
- ⚡ [Installation](#installation)
- 🛠️ [Usage](#usage)
- 📑 [Models](#models)
- 📜 [License](#license)
- 🌟 [Credits](#Credits)

## 📖 Introduction

Keras Vision Models (KVMM) is a collection of vision models with pretrained weights, built entirely with Keras 3. It supports a range of tasks, including segmentation, object detection, vision-language modeling (VLMs), and classification. KVMM includes custom layers and backbone support, providing flexibility and efficiency across various vision applications. For backbones, there are various weight variants like `in1k`, `in21k`, `fb_dist_in1k`, `ms_in22k`, `fb_in22k_ft_in1k`, `ns_jft_in1k`, `aa_in1k`, `cvnets_in1k`, `augreg_in21k_ft_in1k`, `augreg_in21k`, and many more.

## ⚡Installation 

From PyPI (recommended)

```shell
pip install -U kvmm
```

From Source

```shell
pip install -U git+https://github.com/IMvision12/keras-vision-models
```

## 🛠️ Usage

<h3><b>🔎 Listing Available Models</b></h3>

Shows all available models, including backbones, segmentation models, object detection models, and vision-language models (VLMs). It also includes the names of the weights available for each specific model variant.
    
```python
import kvmm
print(kvmm.list_models())

## Output:
"""
CaiTM36 : fb_dist_in1k_384
CaiTM48 : fb_dist_in1k_448
CaiTS24 : fb_dist_in1k_224, fb_dist_in1k_384
...
ConvMixer1024D20 : in1k
ConvMixer1536D20 : in1k
...
ConvNeXtAtto : d2_in1k
ConvNeXtBase : fb_in1k, fb_in22k, fb_in22k_ft_in1k, fb_in22k_ft_in1k_384
...
"""
```
<h3><b>🔎 List Specific Model Variant</b></h3>

```python
import kvmm
print(kvmm.list_models("swin"))

# Output:
"""
SwinBaseP4W12 : ms_in1k, ms_in22k, ms_in22k_ft_in1k
SwinBaseP4W7 : ms_in1k, ms_in22k, ms_in22k_ft_in1k
SwinLargeP4W12 : ms_in22k, ms_in22k_ft_in1k
SwinLargeP4W7 : ms_in22k, ms_in22k_ft_in1k
SwinSmallP4W7 : ms_in1k, ms_in22k, ms_in22k_ft_in1k
SwinTinyP4W7 : ms_in1k, ms_in22k
"""
```

<h3><b>⚙️ Layers </b></h3>
KVMM provides various custom layers like StochasticDepth, LayerScale, EfficientMultiheadSelfAttention, and more. These layers can be seamlessly integrated into your custom models and workflows 🚀

```python
import kvmm

# Example 1
layer = kvmm.layers.StochasticDepth(drop_path_rate=0.1)
output = layer(input_tensor, training=True)

# Example 2
window_partition = WindowPartition(window_size=7)
windowed_features = window_partition(features, height=28, width=28)
```

<h3><b>🏗️ Backbone Usage (Classification) </b></h3>

#### 🛠️ Basic Usage
```python
import kvmm
import numpy as np

# default configuration
model = kvmm.models.vit.ViTTiny16()

# For Fine-Tuning (default weight)
model = kvmm.models.vit.ViTTiny16(include_top=False, input_shape=(224,224,3))
# Custom Weight
model = kvmm.models.vit.ViTTiny16(include_top=False, input_shape=(224,224,3), weights="augreg_in21k_224")

# Backbone Support
model = kvmm.models.vit.ViTTiny16(include_top=False, as_backbone=True, input_shape=(224,224,3), weights="augreg_in21k_224")
random_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
features = model(random_input)
print(f"Number of feature maps: {len(features)}")
for i, feature in enumerate(features):
    print(f"Feature {i} shape: {feature.shape}")

"""
Output:

Number of feature maps: 13
Feature 0 shape: (1, 197, 192)
Feature 1 shape: (1, 197, 192)
Feature 2 shape: (1, 197, 192)
...
"""    
```

#### Example Inference

```python
from keras import ops
from keras.applications.imagenet_utils import decode_predictions
import kvmm
from PIL import Image

model = kvmm.models.swin.SwinTinyP4W7(input_shape=[224, 224, 3])

image = Image.open("bird.png").resize((224, 224))
x = ops.convert_to_tensor(image)
x = ops.expand_dims(x, axis=0)

# Predict
preds = model.predict(x)
print("Predicted:", decode_predictions(preds, top=3)[0])

#output:
Predicted: [('n01537544', 'indigo_bunting', np.float32(0.9135666)), ('n01806143', 'peacock', np.float32(0.0003379386)), ('n02017213', 'European_gallinule', np.float32(0.00027174334))]
```

<h3><b>🧩 Segmentation </b></h3>

#### 🛠️ Basic Usage
 
```python
import kvmm

# Pre-Trained weights (cityscapes or ade20kor mit(in1k))
# ade20k and cityscapes can be used for fine-tuning by giving custom `num_classes`
# If `num_classes` is not specified by default for ade20k it will be 150 and for cityscapes it will be 19
model = kvmm.models.segformer.SegFormerB0(weights="ade20k", input_shape=(512,512,3))
model = kvmm.models.segformer.SegFormerB0(weights="cityscapes", input_shape=(512,512,3))

# Fine-Tune using `MiT` backbone (This will load `in1k` weights)
model = kvmm.models.segformer.SegFormerB0(weights="mit", input_shape=(512,512,3))
```

#### 🚀 Custom Backbone Support

```python
import kvmm

# With no backbone weights
backbone = kvmm.models.resnet.ResNet50(as_backbone=True, weights=None, include_top=False, input_shape=(224,224,3))
segformer = kvmm.models.segformer.SegFormerB0(weights=None, backbone=backbone, num_classes=10, input_shape=(224,224,3))

# With backbone weights
import kvmm
backbone = kvmm.models.resnet.ResNet50(as_backbone=True, weights="tv_in1k", include_top=False, input_shape=(224,224,3))
segformer = kvmm.models.segformer.SegFormerB0(weights=None, backbone=backbone, num_classes=10, input_shape=(224,224,3))
```

#### 🚀 Example Inference

```python
import kvmm
from PIL import Image
import numpy as np

model = kvmm.models.segformer.SegFormerB0(weights="ade20k_512")

image = Image.open("ADE_train_00000586.jpg")
processed_img = kvmm.models.segformer.SegFormerImageProcessor(image=image,
    do_resize=True,
    size={"height": 512, "width": 512},
    do_rescale=True,
    do_normalize=True)
outs = model.predict(processed_img)
outs = np.argmax(outs[0], axis=-1)
visualize_segmentation(outs, image)
```
![output](images/seg_output.png)


<h3><b>VLMS 🚧 </b></h3>

#### 🛠️ Basic Usage

```python
import keras

import kvmm

processor = kvmm.models.clip.CLIPProcessor()
model = kvmm.models.clip.ClipVitBase16(
    weights="openai_224",
    input_shape=(224, 224, 3),
)
inputs = processor(text=["mountains", "tortoise", "cat"], image_paths="cat1.jpg")
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["input_ids"],
        "padding_mask": inputs["attention_mask"],
    }
)

print("Raw Model Output:")
print(output)

preds = keras.ops.softmax(output["image_logits"]).numpy().squeeze()
result = dict(zip(["mountains", "tortoise", "cat"], preds))
print("\nPrediction probabilities:")
print(result)

#output:
"""{'image_logits': <tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[11.042501, 10.388493, 18.414747]], dtype=float32)>, 'text_logits': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
array([[11.042501],
       [10.388493],
       [18.414747]], dtype=float32)>}

Prediction probabilities:
{'mountains': np.float32(0.0006278555), 'tortoise': np.float32(0.000326458), 'cat': np.float32(0.99904567)}"""
```
## 📑 Models

- Backbones:

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | CaiT | [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239) | `timm` |
    | ConvMixer | [Patches Are All You Need?](https://arxiv.org/abs/2201.09792) | `timm` |
    | ConvNeXt | [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) | `timm` |
    | ConvNeXt V2 | [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) | `timm` |
    | DeiT | [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) | `timm` |
    | DenseNet | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | `timm` |
    | EfficientNet | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | `timm` |
    | EfficientNet-Lite | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | `timm` |
    | EfficientNetV2 | [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) | `timm` |
    | FlexiViT | [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013) | `timm` |
    | InceptionNeXt | [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900) | `timm` |
    | Inception-ResNet-v2 | [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) | `timm` |
    | Inception-v3 | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | `timm` |
    | Inception-v4 | [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) | `timm` |
    | MiT | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | `transformers` |
    | MLP-Mixer | [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) | `timm` |
    | MobileNetV2 | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | `timm` |
    | MobileNetV3 | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) | `keras` |
    | MobileViT | [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) | `timm` |
    | MobileViTV2 | [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680) | `timm` |
    | PiT | [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302) | `timm` |
    | PoolFormer | [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) | `timm` |
    | Res2Net | [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169) | `timm` |
    | ResMLP | [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404) | `timm` |
    | ResNet | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | `timm` |
    | ResNetV2 | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | `timm` |
    | ResNeXt | [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) | `timm` |
    | SENet | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | `timm` |
    | Swin Transformer | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | `timm` |
    | VGG | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | `timm` |
    | ViT | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | `timm` |
    | Xception | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) | `keras` |

<br>

- Segmentation

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | SegFormer | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | `transformers`|

<br>

- Vision-Language-Models (VLMs)

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | CLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | `transformers`|
  
## 📜 License

This project leverages [timm](https://github.com/huggingface/pytorch-image-models#licenses) and [transformers](https://github.com/huggingface/transformers#license) for converting pretrained weights from PyTorch to Keras. For licensing details, please refer to the respective repositories.

- 🔖 **kvmm Code**: This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).


## 🌟 Credits

- The [Keras](https://github.com/keras-team/keras) team for their powerful and user-friendly deep learning framework
- The [Transformers](https://github.com/huggingface/transformers) library for its robust tools for loading and adapting pretrained models  
- The [pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) project for pioneering many computer vision model implementations
- All contributors to the original papers and architectures implemented in this library

## Citing

### BibTeX

```bash
@misc{gc2025kvmm,
  author = {Gitesh Chawda},
  title = {Keras Vision Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/IMvision12/keras-vision-models}}
}
```
