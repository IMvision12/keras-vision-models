# Keras Models 🚀

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Keras](https://img.shields.io/badge/keras-v3.5.0+-success.svg)](https://github.com/keras-team/keras)
![Python](https://img.shields.io/badge/python-v3.10.0+-success.svg)

## 📖 Introduction

Keras Models (kmodels) is a collection of models with pretrained weights, built entirely with Keras 3. It supports a range of tasks, including classification, object detection (DETR, RT-DETR, RF-DETR), segmentation (SAM, SAM2, SegFormer, DeepLabV3, EoMT), vision-language modeling (CLIP, SigLIP, SigLIP2), and more. kmodels includes custom layers and backbone support, providing flexibility and efficiency across various applications. For backbones, there are various weight variants like `in1k`, `in21k`, `fb_dist_in1k`, `ms_in22k`, `fb_in22k_ft_in1k`, `ns_jft_in1k`, `aa_in1k`, `cvnets_in1k`, `augreg_in21k_ft_in1k`, `augreg_in21k`, and many more.

## ⚡ Installation

From PyPI (recommended)

```shell
pip install -U kmodels
```

From Source

```shell
pip install -U git+https://github.com/IMvision12/keras-models
```

## 📑 Documentation

| Topic | Description |
|-------|-------------|
| [Backbone Models](docs/backbones.md) | Classification backbones (ViT, ResNet, Swin, ConvNeXt, EfficientNet, and more) with usage examples and model listing |

**Segmentation**

| Model | Description |
|-------|-------------|
| [SAM](docs/sam.md) | Segment Anything Model — promptable segmentation with points, boxes, or masks (ViT-B/L/H) |
| [SAM2](docs/sam2.md) | Segment Anything Model 2 — next generation of promptable visual segmentation (Hiera Tiny/Small/Base+/Large) |
| [SegFormer](docs/segformer.md) | Transformer-based semantic segmentation with MLP decoder, Cityscapes & ADE20K weights |
| [DeepLabV3](docs/deeplabv3.md) | Atrous convolution-based semantic segmentation |
| [EoMT](docs/eomt.md) | Encoder-only Mask Transformer for panoptic segmentation |

**Object Detection**

| Model | Description |
|-------|-------------|
| [DETR](docs/detr.md) | End-to-end object detection with Transformers (ResNet-50/101 backbones) |
| [RT-DETR](docs/rt_detr.md) | Real-time DETR with ResNet-vd backbone and hybrid encoder (ResNet-18/34/50/101 variants) |
| [RF-DETR](docs/rf_detr.md) | Real-time detection transformer (Nano, Small, Medium, Base, Large variants) |

**Vision-Language Models**

| Model | Description |
|-------|-------------|
| [CLIP](docs/clip.md) | Contrastive Language-Image Pre-training for zero-shot classification |
| [SigLIP](docs/siglip.md) | Sigmoid loss-based language-image pre-training with multilingual support |
| [SigLIP2](docs/siglip2.md) | Next-gen SigLIP with improved semantic understanding and 256K vocabulary |

## 📑 Models

- Backbones

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

- Object Detection

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | DETR | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | `transformers`|
    | RT-DETR | [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) | `transformers` |
    | RF-DETR | [RF-DETR: Real-Time Detection Transformer](https://arxiv.org/abs/2502.18860) | `rfdetr` |

<br>

- Segmentation

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | DeepLabV3 | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | `torchvision` |
    | EoMT | [Encoder-only Mask Transformer for Panoptic Segmentation](https://arxiv.org/abs/2504.07957) | `transformers` |
    | SAM | [Segment Anything](https://arxiv.org/abs/2304.02643) | `transformers` |
    | SAM2 | [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) | `transformers` |
    | SegFormer | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | `transformers`|

<br>

- Vision-Language-Models (VLMs)

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | CLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | `transformers`|
    | SigLIP | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | `transformers`|
    | SigLIP2 | [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786) | `transformers`|

<br>

## 📜 License

This project leverages [timm](https://github.com/huggingface/pytorch-image-models#licenses) and [transformers](https://github.com/huggingface/transformers#license) for converting pretrained weights from PyTorch to Keras. For licensing details, please refer to the respective repositories.

- 🔖 **kmodels Code**: This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).


## 🌟 Credits

- The [Keras](https://github.com/keras-team/keras) team for their powerful and user-friendly deep learning framework
- The [Transformers](https://github.com/huggingface/transformers) library for its robust tools for loading and adapting pretrained models
- The [pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) project for pioneering many computer vision model implementations
- All contributors to the original papers and architectures implemented in this library

## Citing

### BibTeX

```bash
@misc{gc2025kmodels,
  author = {Gitesh Chawda},
  title = {Keras Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/IMvision12/keras-models}}
```
