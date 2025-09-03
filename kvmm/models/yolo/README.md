# YOLO Models

## Overview

YOLO (You Only Look Once) is a family of real-time object detection models that revolutionized computer vision by treating object detection as a single regression problem. Unlike traditional two-stage detectors, YOLO models predict bounding boxes and class probabilities directly from full images in one evaluation, making them extremely fast and suitable for real-time applications.

This implementation provides Keras/TensorFlow models for various YOLO architectures, enabling object detection, instance segmentation, and other computer vision tasks with state-of-the-art performance.

## 🏗️ Available YOLO Versions

- **YOLOv8** 
- **YOLOv5**
- More YOLO versions coming soon...

## 🔧 Core Components

### Preprocessing
- **YoloPreProcessor** - Handles image preprocessing with letterbox resizing, normalization, and format conversion
- Maintains aspect ratio while resizing to target dimensions
- Uses standard YOLO padding color (114, 114, 114) for letterbox
- Supports batch processing and file path inputs

### Postprocessing  
- **YoloPostProcessor** - Handles model output processing including bbox decoding, class predictions, and NMS
- Converts raw model outputs to final detection results
- Configurable confidence and IoU thresholds
- Supports multiple detection heads and anchor-free architectures

### Architecture Components
- **Blocks** - Core building blocks (C3, Conv, SPPF, Bottleneck, c2f)
- **Head** - Detection head for multi-scale feature processing
- **Layers** - Specialized layers like DFL (Distribution Focal Loss)
- **Utils** - Helper functions for bbox operations, anchor generation, and scaling

## 🛠️ Basic Usage

```python
import keras
import kvmm

# Create preprocessor and model
preprocessor = kvmm.models.yolo.YoloPreProcessor(image_size=640)
model = kvmm.models.yolo.YoloV5s(weights="coco", training=False)

# Process single image
image_path = "path/to/image.jpg"
processed = preprocessor(image_paths=image_path)
detections = model(processed["images"])

print("YOLOv5 detections shape:", detections.shape)
```

## Batch Processing Multiple Images

```python
import keras
import kvmm

preprocessor = kvmm.models.yolo.YoloPreProcessor(image_size=640)
model = kvmm.models.yolo.YoloV5s(weights="coco", training=False)

# Process multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
processed = preprocessor(image_paths=image_paths)
detections = model(processed["images"])

print(f"Batch detections shape: {detections.shape}")
print(f"Processed {len(image_paths)} images")
```

## Custom Preprocessing Configuration

```python
import kvmm

# Custom preprocessor settings
custom_preprocessor = kvmm.models.yolo.YoloPreProcessor(
    image_size=1024,  # Higher resolution
    letterbox_color=[128, 128, 128],  # Different padding color
    do_normalize=True,  # Enable normalization
    letterbox_auto=True,  # Auto-adjust padding for stride
    letterbox_stride=32,  # Stride for padding adjustment
)

# Process with custom settings
processed = custom_preprocessor(image_paths="high_res_image.jpg")
```

## Training Mode

🚧 **Training functionality is currently under construction** 🚧

Training support for YOLO models is being developed and will include:
- Loss functions optimized for object detection
- Training loops with proper data loading
- Support for custom datasets
- Transfer learning from pre-trained weights
- Data augmentation integration

Stay tuned for updates!