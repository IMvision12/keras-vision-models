# SigLIP2 Model

## Overview

paper : [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features
](https://arxiv.org/abs/2502.14786)

SigLIP2 is the next generation of the Sigmoid Loss for Language Image Pre-training model, building upon the success of the original SigLIP architecture. Like its predecessor, SigLIP2 uses a simple pairwise sigmoid loss for Language-Image Pre-training which operates solely on image-text pairs without requiring a global view of pairwise similarities for normalization. This approach continues to achieve superior zero-shot performance compared to CLIP on classification tasks while maintaining training efficiency.

SigLIP2 features an enhanced Vision Transformer (ViT) as the image encoder and an improved text transformer as the text encoder, with optimized architectures and expanded vocabulary support (256K vocabulary size) for better multilingual understanding and performance.

This implementation provides Keras/TensorFlow models for SigLIP2 with various Vision Transformer architectures, enabling zero-shot image classification, image-text similarity, and advanced multimodal understanding tasks.

## 🏗️ Model Variants

- **SigLIP2BaseP16** - Base model with 16×16 patch size (768 hidden dimensions)
- **SigLIP2BaseP32** - Base model with 32×32 patch size (768 hidden dimensions)
- **SigLIP2LargeP16** - Large model with 16×16 patch size (1024 hidden dimensions)
- **SigLIP2So400mP14** - So400m model with 14×14 patch size (1152 hidden dimensions, trained on 400M samples)
- **SigLIP2So400mP16** - So400m model with 16×16 patch size (1152 hidden dimensions, trained on 400M samples)

## 📊 Available Weights

### SigLIP2BaseP16
- **google_224**: Google weights for 224×224 resolution
- **google_256**: Google weights for 256×256 resolution
- **google_384**: Google weights for 384×384 resolution
- **google_512**: Google weights for 512×512 resolution

### SigLIP2BaseP32
- **google_256**: Google weights for 256×256 resolution

### SigLIP2LargeP16
- **google_256**: Google weights for 256×256 resolution
- **google_384**: Google weights for 384×384 resolution
- **google_512**: Google weights for 512×512 resolution

### SigLIP2So400mP14
- **google_224**: Google weights for 224×224 resolution (trained on 400M samples)
- **google_384**: Google weights for 384×384 resolution (trained on 400M samples)

### SigLIP2So400mP16
- **google_256**: Google weights for 256×256 resolution (trained on 400M samples)
- **google_384**: Google weights for 384×384 resolution (trained on 400M samples)
- **google_512**: Google weights for 512×512 resolution (trained on 400M samples)

## 🛠️ Basic Usage

```python
import keras
import kvmm

processor = kvmm.models.siglip2.SigLIP2Processor()
model = kvmm.models.siglip2.SigLIP2BaseP16(
   weights="google_224",
   input_shape=(224, 224, 3), # You can fine-tune or infer with variable size 
)
inputs = processor(text=["mountains", "tortoise", "cat"], image_paths="cat1.jpg")
output = model(
   {
       "images": inputs["images"],
       "token_ids": inputs["input_ids"],
   }
)

print("Raw Model Output:")
print(output)

preds = keras.ops.softmax(output["image_logits"]).numpy().squeeze()
result = dict(zip(["mountains", "tortoise", "cat"], preds))
print("\nPrediction probabilities:")
print(result)
```

## Batch Processing Multiple Images

```python
import keras
import kvmm
import numpy as np

processor = kvmm.models.siglip2.SigLIP2Processor()
model = kvmm.models.siglip2.SigLIP2BaseP16(weights="google_224")

# Process multiple images at once
image_paths = ["dog.jpg", "cat1.jpg"]
labels = ["a photo of a dog", "a photo of a car", "a photo of a flower", "a photo of a cat"]

inputs = processor(text=labels, image_paths=image_paths)
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
})

# Get probabilities for each image using sigmoid activation
probs = keras.ops.softmax(output["image_logits"]).numpy()

for i, img_path in enumerate(image_paths):
    print(f"\nPredictions for {img_path}:")
    img_probs = probs[i]
    for j, label in enumerate(labels):
        print(f"  {label}: {img_probs[j]:.4f}")
```

## Enhanced Multilingual Support

```python
import keras
import kvmm

# Enhanced multilingual support with larger vocabulary
processor = kvmm.models.siglip2.SigLIP2Processor()
model = kvmm.models.siglip2.SigLIP2BaseP16(weights="google_256", input_shape=(224, 224, 3))

# Example with multiple languages
multilingual_labels = [
    "un gato",      # Spanish: a cat
    "ein Vogel",    # German: a bird
    "कुत्ता",        # Hindi: dog
]

inputs = processor(text=multilingual_labels, image_paths="pet_image.jpg")
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
})

probs = keras.ops.softmax(output["image_logits"]).numpy().squeeze()
result = dict(zip(multilingual_labels, probs))
print("Enhanced multilingual predictions:")
print(result)
```

## Using So400m Models for Superior Performance

The So400m variants are trained on 400 million image-text pairs, offering enhanced performance for demanding applications:


```python
import keras
import kvmm

# Use So400m model for best performance
processor = kvmm.models.siglip2.SigLIP2Processor(image_resolution=384)
model = kvmm.models.siglip2.SigLIP2So400mP16(
    weights="google_384",
    input_shape=(384, 384, 3)
)

# Complex scene understanding
complex_labels = [
    "crowded marketplace with people shopping",
    "peaceful countryside landscape",
    "busy urban intersection with traffic",
    "serene beach at sunset",
    "industrial warehouse facility"
]

inputs = processor(text=complex_labels, image_paths="complex_scene.jpg")
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
})

probs = keras.ops.softmax(output["image_logits"]).numpy().squeeze()
scene_analysis = dict(zip(complex_labels, probs))

print("Complex scene understanding:")
for description, confidence in scene_analysis.items():
    print(f"  {description}: {confidence:.4f}")
```