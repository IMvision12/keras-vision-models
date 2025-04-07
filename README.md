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

- 🔎 List All Models
Shows all available models, including backbones, segmentation models, object detection models, and vision-language models (VLMs). It also includes the names of the weights available for each specific model variant.

    ```python
    import kvmm
    print(kvmm.utils.list_models())

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

- 🔎 List Specific Model Variant:

    ```python
    import kvmm
    print(kvmm.utils.list_models("swin"))

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

- ⚙️ Layers : KVMM provides various custom layers like StochasticDepth, LayerScale, EfficientMultiheadSelfAttention, and more. These layers can be seamlessly integrated into your custom models and workflows 🚀

    ```python
    import kvmm

    # Example 1
    layer = kvmm.layers.StochasticDepth(drop_path_rate=0.1)
    output = layer(input_tensor, training=True)

    # Example 2
    window_partition = WindowPartition(window_size=7)
    windowed_features = window_partition(features, height=28, width=28)
    ```

- 🏗️ Backbone Usage (Classification)

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

- Segmentation

    ### 🛠️ Usage
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

    ### 🚀 Custom Backbone Suppport 
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

- Object Detection 🚧
- VLMS 🚧

## 📑 Models

- Backbones:

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | cait | [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239) | `timm` |
    | convmixer | [Patches Are All You Need?](https://arxiv.org/abs/2201.09792) | `timm` |
    | convnext | [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) | `timm` |
    | convnextv2 | [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) | `timm` |
    | deit | [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) | `timm` |
    | densenet | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | `timm` |
    | efficientnet | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | `timm` |
    | efficientnet_lite | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | `timm` |
    | efficientnetv2 | [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) | `timm` |
    | flexivit | [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013) | `timm` |
    | inception_next | [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900) | `timm` |
    | inception_resnetv2 | [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) | `timm` |
    | inceptionv3 | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | `timm` |
    | inceptionv4 | [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) | `timm` |
    | mit | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | `transformers` |
    | mlp_mixer | [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) | `timm` |
    | mobilenetv2 | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | `timm` |
    | mobilenetv3 | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) | `keras` |
    | mobilevit | [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) | `timm` |
    | mobilevitv2 | [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680) | `timm` |
    | pit | [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302) | `timm` |
    | poolformer | [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) | `timm` |
    | res2net | [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169) | `timm` |
    | resmlp | [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404) | `timm` |
    | resnet | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | `timm` |
    | resnetv2 | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | `timm` |
    | resnext | [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) | `timm` |
    | senet | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | `timm` |
    | swin | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | `timm` |
    | vgg | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | `timm` |
    | vit | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | `timm` |
    | xception | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) | `keras` |

<br>

- Segmentation

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | SegFormer | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | `transformers`|

## 📜 License

This project **uses** [timm](https://github.com/huggingface/pytorch-image-models#licenses), [transformers](https://github.com/huggingface/transformers#license). These libraries were used for porting weights from PyTorch to Keras.

- 🔖 **kvmm Code**: The code in this repository is available under the **Apache 2.0 license**.

## 🌟 Credits

- The [Keras](https://github.com/keras-team/keras) team for their powerful and user-friendly deep learning framework
- The [Transformers](https://github.com/huggingface/transformers) library for its robust tools for loading and adapting pretrained models  
- The [pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) project for pioneering many computer vision model implementations
- All contributors to the original papers and architectures implemented in this library

## Citing

### BibTeX

```bash
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

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
