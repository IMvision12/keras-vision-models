# CLIP Model

## Overview

Paper: [Learning Transferable Visual Representations from Natural Language Supervision](https://arxiv.org/abs/2103.00020)

CLIP (Contrastive Language-Image Pre-training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task. CLIP uses a Vision Transformer (ViT) as the image encoder and a causal language model as the text encoder.

This implementation provides Keras/TensorFlow models for CLIP with various Vision Transformer architectures, enabling zero-shot image classification, image-text similarity, and multimodal understanding tasks.

## 🏗️ Model Variants

- **ClipVitBase16** - Base model with 16×16 patch size
- **ClipVitBase32** - Base model with 32×32 patch size  
- **ClipVitLarge14** - Large model with 14×14 patch size
- **ClipVitG14** - Giant model with 14×14 patch size
- **ClipVitBigG14** - Biggest Giant model with 14×14 patch size

## 📊 Available Weights

### ClipVitBase16
- **openai_224**: Original OpenAI weights trained on 400M image-text pairs (224×224 resolution)

### ClipVitBase32
- **openai_224**: Original OpenAI weights (224×224 resolution)
- **laion2b_s34B_b79K_224**: LAION-2B dataset weights with 34B samples (224×224 resolution)

### ClipVitLarge14
- **openai_224**: Original OpenAI weights (224×224 resolution)
- **openai_336**: OpenAI weights fine-tuned for higher resolution (336×336)
- **laion2b_s32B_b82K_224**: LAION-2B dataset weights with 32B samples (224×224 resolution)

### ClipVitG14
- **laion2b_s12B_b42K_224**: LAION-2B dataset weights with 12B samples (224×224 resolution)

### ClipVitBigG14
- **laion2b_39B_b160k_224**: LAION-2B dataset weights with 39B samples (224×224 resolution)

## 🛠️ Basic Usage

```python
import keras
import kvmm

processor = kvmm.models.clip.CLIPProcessor()
model = kvmm.models.clip.ClipVitBase16(
    weights="openai_224",
    input_shape=(224, 224, 3), # You can fine-tune or infer with variable size 
)
inputs = processor(text=["mountains", "tortoise", "cat"], image_paths="cat1.jpg")
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["input_ids"],
        "padding_mask": inputs["attention_mask"],
    }
)

print("Raw Model Output:")
print(output)

preds = keras.ops.softmax(output["image_logits"]).numpy().squeeze()
result = dict(zip(["mountains", "tortoise", "cat"], preds))
print("\nPrediction probabilities:")
print(result)

#output:
"""{'image_logits': <tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[11.042501, 10.388493, 18.414747]], dtype=float32)>, 'text_logits': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
array([[11.042501],
       [10.388493],
       [18.414747]], dtype=float32)>}

Prediction probabilities:
{'mountains': np.float32(0.0006278555), 'tortoise': np.float32(0.000326458), 'cat': np.float32(0.99904567)}"""
```

## Batch Processing Multiple Images

```python
import keras
import kvmm
import numpy as np

processor = kvmm.models.clip.CLIPProcessor()
model = kvmm.models.clip.ClipVitBase16(weights="openai_224")

# Process multiple images at once
image_paths = ["dog.jpg", "cat1.jpg"]
labels = ["a photo of a dog", "a photo of a car", "a photo of a flower", "a photo of a cat"]

inputs = processor(text=labels, image_paths=image_paths)
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
    "padding_mask": inputs["attention_mask"],
})

# Get probabilities for each image
probs = keras.ops.softmax(output["image_logits"], axis=-1).numpy()

for i, img_path in enumerate(image_paths):
    print(f"\nPredictions for {img_path}:")
    img_probs = probs[i]
    for j, label in enumerate(labels):
        print(f"  {label}: {img_probs[j]:.4f}")
```