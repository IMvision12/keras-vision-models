# DeepLabV3

**Paper**: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

DeepLabV3 is a highly accurate semantic segmentation model that employs atrous (dilated) convolution to capture multi-scale spatial context without losing spatial resolution. It features an Atrous Spatial Pyramid Pooling (ASPP) module that probes convolutional features at multiple scales, making it highly robust for segmenting objects of varying sizes.

## Architecture Highlights

- **ResNet Backbone:** Leverages deep residual networks (ResNet-50 or ResNet-101) for robust feature extraction.
- **Atrous Convolution:** Controls the resolution of features computed by Deep CNNs and effectively enlarges the field of view of filters without increasing the number of parameters or the amount of computation.
- **ASPP Module:** Captures multi-scale information by applying parallel atrous convolutions with different dilation rates.

## Available Models

| Model | Description | Weights |
|-------|-------------|---------|
| `DeepLabV3ResNet50` | DeepLabV3 with a ResNet-50 backbone | `coco_voc` |
| `DeepLabV3ResNet101` | DeepLabV3 with a ResNet-101 backbone | `coco_voc` |

*Note: The `coco_voc` weights are pre-trained on the COCO dataset and fine-tuned on the PASCAL VOC segmentation dataset. They output predictions across 21 classes (20 objects + 1 background).*

## Basic Usage

```python
import kmodels

# Load model with pre-trained weights
model = kmodels.models.deeplabv3.DeepLabV3ResNet50(weights="coco_voc", input_shape=(512, 512, 3))

# Using ResNet101 backbone
model_large = kmodels.models.deeplabv3.DeepLabV3ResNet101(weights="coco_voc", input_shape=(512, 512, 3))

# Initialize a model without pre-trained weights for custom training
custom_model = kmodels.models.deeplabv3.DeepLabV3ResNet50(
    weights=None,
    input_shape=(512, 512, 3),
    num_classes=21
)
```

## Inference Example

Below is a complete example of loading an image, running it through DeepLabV3, and decoding the resulting segmentation mask using `numpy` and `PIL`.

```python
import keras
import numpy as np
from PIL import Image
import kmodels

# Load model
model = kmodels.models.deeplabv3.DeepLabV3ResNet50(weights="coco_voc", input_shape=(512, 512, 3))

# Load and preprocess image
image = Image.open("street.jpg").convert("RGB")
original_size = image.size
image = image.resize((512, 512))

# Convert to tensor and add batch dimension
input_tensor = keras.ops.convert_to_tensor(np.array(image).astype("float32") / 255.0)
input_tensor = keras.ops.expand_dims(input_tensor, axis=0) # Shape: (1, 512, 512, 3)

# Run Inference
output = model(input_tensor) # Output shape: (1, 512, 512, 21)

# Get the class index with the highest probability per pixel
pred_mask = keras.ops.argmax(output, axis=-1)[0].numpy()

# pred_mask contains pixel-wise class indices (0 to 20).
# Class 0 is generally the background.
print(f"Mask shape: {pred_mask.shape}")
print(f"Predicted classes present in the image: {np.unique(pred_mask)}")
```
