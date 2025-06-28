# SigLIP Model

## Overview

Paper: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

SigLIP (Sigmoid Loss for Language Image Pre-training) is a simple pairwise sigmoid loss for Language-Image Pre-training which operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization. This approach achieves better zero-shot performance compared to CLIP on classification tasks while being more efficient during training.

SigLIP uses a Vision Transformer (ViT) as the image encoder and a text transformer as the text encoder, similar to CLIP but with improved training efficiency through the sigmoid loss function instead of contrastive loss.

This implementation provides Keras/TensorFlow models for SigLIP with various Vision Transformer architectures, enabling zero-shot image classification, image-text similarity, and multimodal understanding tasks.

## 🏗️ Model Variants

- **SigLIPBaseP16** - Base model with 16×16 patch size (768 hidden dimensions)
- **SigLIPLargeP16** - Large model with 16×16 patch size (1024 hidden dimensions)
- **SigLIPSo400mP14** - So400m model with 14×14 patch size (1152 hidden dimensions, trained on 400M samples)

## 📊 Available Weights

### SigLIPBaseP16
- **google_224**: Google weights for 224×224 resolution
- **google_256**: Google weights for 256×256 resolution  
- **google_multilingual_256**: Google multilingual weights for 256×256 resolution
- **google_384**: Google weights for 384×384 resolution
- **google_512**: Google weights for 512×512 resolution

### SigLIPLargeP16
- **google_256**: Google weights for 256×256 resolution
- **google_384**: Google weights for 384×384 resolution

### SigLIPSo400mP14
- **google_224**: Google weights for 224×224 resolution (trained on 400M samples)
- **google_384**: Google weights for 384×384 resolution (trained on 400M samples)

## 🛠️ Basic Usage

```python
import keras
import kvmm

processor = kvmm.models.siglip.SigLIPProcessor()
model = kvmm.models.siglip.SigLIPBaseP16(
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

processor = kvmm.models.siglip.SigLIPProcessor()
model = kvmm.models.siglip.SigLIPBaseP16(weights="google_224")

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

## Multilingual Support

```python
import keras
import kvmm

# Use multilingual weights for non-English text
processor = kvmm.models.siglip.SigLIPProcessor(multilingual=True)
model = kvmm.models.siglip.SigLIPBaseP16(weights="google_multilingual_256")

# Example with multiple languages
multilingual_labels = [
    "un gato",  # Spanish: a cat
    "un chien", # French: a dog
    "ein Vogel", # German: a bird
    "cat"       # English: cat
]

inputs = processor(text=multilingual_labels, image_paths="pet_image.jpg")
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
})

probs = keras.ops.softmax(output["image_logits"]).numpy().squeeze()
result = dict(zip(multilingual_labels, probs))
print("Multilingual predictions:")
print(result)
```