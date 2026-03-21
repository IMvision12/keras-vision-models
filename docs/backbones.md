# Backbone Models (Classification)

## Listing Available Models

Shows all available models, including backbones, segmentation models, object detection models, and vision-language models (VLMs). It also includes the names of the weights available for each specific model variant.

```python
import kmodels
print(kmodels.list_models())

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

## List Specific Model Variant

```python
import kmodels
print(kmodels.list_models("swin"))

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

## Basic Usage

```python
import kmodels
import numpy as np

# default configuration
model = kmodels.models.vit.ViTTiny16()

# For Fine-Tuning (default weight)
model = kmodels.models.vit.ViTTiny16(include_top=False, input_shape=(224,224,3))
# Custom Weight
model = kmodels.models.vit.ViTTiny16(include_top=False, input_shape=(224,224,3), weights="augreg_in21k_224")

# Backbone Support
model = kmodels.models.vit.ViTTiny16(include_top=False, as_backbone=True, input_shape=(224,224,3), weights="augreg_in21k_224")
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

## Example Inference

```python
from keras import ops
from keras.applications.imagenet_utils import decode_predictions
import kmodels
from PIL import Image

model = kmodels.models.swin.SwinTinyP4W7(input_shape=[224, 224, 3])

image = Image.open("bird.png").resize((224, 224))
x = ops.convert_to_tensor(image)
x = ops.expand_dims(x, axis=0)

# Predict
preds = model.predict(x)
print("Predicted:", decode_predictions(preds, top=3)[0])

#output:
Predicted: [('n01537544', 'indigo_bunting', np.float32(0.9135666)), ('n01806143', 'peacock', np.float32(0.0003379386)), ('n02017213', 'European_gallinule', np.float32(0.00027174334))]
```

## Available Backbone Models

| Model Name | Reference Paper | Source of Weights |
|------------|----------------|-------------------|
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
