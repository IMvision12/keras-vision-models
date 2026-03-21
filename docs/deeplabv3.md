# DeepLabV3

**Paper**: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

DeepLabV3 is a semantic segmentation model that employs atrous (dilated) convolution to capture multi-scale context. It uses Atrous Spatial Pyramid Pooling (ASPP) to probe features at multiple scales for robust segmentation.

## Basic Usage

```python
import kmodels

model = kmodels.models.deeplabv3.DeepLabV3ResNet50(weights="coco_voc", input_shape=(512, 512, 3))
model = kmodels.models.deeplabv3.DeepLabV3ResNet101(weights="coco_voc", input_shape=(512, 512, 3))

# Without pre-trained weights
model = kmodels.models.deeplabv3.DeepLabV3ResNet50(weights=None, input_shape=(512, 512, 3), num_classes=21)
```
