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

## Features and Capabilities

- **Wide Variety of Architectures:** Extensive collection covering CNNs (ResNet, ConvNeXt), Vision Transformers (ViT, Swin), and hybrid models (MaxViT, MobileViT).
- **Pretrained Weights Checkpoints:** Provides weights pretrained on diverse large scale datasets like ImageNet-1K, ImageNet-21K, and JFT datasets.
- **Easy Feature Extraction:** Robust backbone API that yields multi-scale feature maps from intermediate network stages for down-stream tasks.
- **Classification & fine-tuning Support:** Seamless integration for transfer learning and image classification customization out of the box.

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
