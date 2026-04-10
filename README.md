# Keras Models 🚀

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Keras](https://img.shields.io/badge/keras-v3.5.0+-success.svg)](https://github.com/keras-team/keras)
![Python](https://img.shields.io/badge/python-v3.10.0+-success.svg)

## 📖 Introduction

Keras Models (kmodels) is a collection of models with pretrained weights, built entirely with Keras 3. It supports a range of tasks, including classification, object detection (DETR, RT-DETR, RT-DETRv2, RF-DETR, D-FINE), segmentation (SAM, SAM2, SAM3, SegFormer, DeepLabV3, EoMT), vision-language modeling (CLIP, SigLIP, SigLIP2), and more. It includes hybrid architectures like MaxViT alongside traditional CNNs and pure transformers. kmodels includes custom layers and backbone support, providing flexibility and efficiency across various applications. For backbones, there are various weight variants like `in1k`, `in21k`, `fb_dist_in1k`, `ms_in22k`, `fb_in22k_ft_in1k`, `ns_jft_in1k`, `aa_in1k`, `cvnets_in1k`, `augreg_in21k_ft_in1k`, `augreg_in21k`, and many more.

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

Per-model usage guides — covering loading pre-trained weights, preprocessing, inference, post-processing, and full visualization examples — live in the [`docs/`](docs/) folder. Each model in the tables below has a dedicated page there (e.g. [`docs/rt_detr_v2.md`](docs/rt_detr_v2.md), [`docs/sam2.md`](docs/sam2.md), [`docs/clip.md`](docs/clip.md), [`docs/backbones.md`](docs/backbones.md)).

## 📊 Benchmarks

Per-variant paper-reported metrics and parameter counts for every model family — for picking a model that fits your accuracy/size budget — live in [`benchmarks/`](benchmarks/). The metric depends on the task: ImageNet-1K **Top-1** for backbones, COCO val2017 **box AP** for object detection, **mIoU / PQ / J&F** for segmentation (depending on the family's evaluation protocol), and parameter counts for vision-language models. See [`benchmarks/backbones.md`](benchmarks/backbones.md), [`benchmarks/object_detection.md`](benchmarks/object_detection.md), [`benchmarks/segmentation.md`](benchmarks/segmentation.md), and [`benchmarks/vlm.md`](benchmarks/vlm.md).

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
    | MaxViT | [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697) | `timm` |
    | MiT | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | `transformers` |
    | MLP-Mixer | [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) | `timm` |
    | MobileNetV2 | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | `timm` |
    | MobileNetV3 | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) | `keras` |
    | MobileViT | [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) | `timm` |
    | MobileViTV2 | [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680) | `timm` |
    | NextViT | [Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios](https://arxiv.org/abs/2207.05501) | `timm` |
    | PiT | [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302) | `timm` |
    | PoolFormer | [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) | `timm` |
    | Res2Net | [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169) | `timm` |
    | ResMLP | [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404) | `timm` |
    | ResNet | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | `timm` |
    | ResNetV2 | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | `timm` |
    | ResNeXt | [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) | `timm` |
    | SENet | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | `timm` |
    | Swin Transformer | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | `timm` |
    | Swin Transformer V2 | [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883) | `timm` |
    | VGG | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | `timm` |
    | ViT | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | `timm` |
    | Xception | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) | `keras` |

<br>

- Object Detection

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | D-FINE | [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842) | `transformers` |
    | DETR | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | `transformers`|
    | RT-DETR | [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) | `transformers` |
    | RT-DETRv2 | [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformers](https://arxiv.org/abs/2407.17140) | `transformers` |
    | RF-DETR | [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554) | `rfdetr` |

<br>

- Segmentation

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | DeepLabV3 | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | `torchvision` |
    | EoMT | [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108) | `transformers` |
    | SAM | [Segment Anything](https://arxiv.org/abs/2304.02643) | `transformers` |
    | SAM2 | [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) | `transformers` |
    | SAM3 | [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719) | `transformers` |
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
