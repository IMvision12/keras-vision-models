# CLIP

**Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

CLIP (Contrastive Language-Image Pre-training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task. CLIP uses a Vision Transformer (ViT) as the image encoder and a causal language model as the text encoder.

## Model Variants

- **ClipVitBase16** — Base model with 16x16 patch size
- **ClipVitBase32** — Base model with 32x32 patch size
- **ClipVitLarge14** — Large model with 14x14 patch size
- **ClipVitG14** — Giant model with 14x14 patch size
- **ClipVitBigG14** — Biggest Giant model with 14x14 patch size

## Available Weights

| Variant | openai_224 | openai_336 | laion2b |
|---------|:-:|:-:|:-:|
| ClipVitBase16 | ✅ | | |
| ClipVitBase32 | ✅ | | ✅ (s34B_b79K_224) |
| ClipVitLarge14 | ✅ | ✅ | ✅ (s32B_b82K_224) |
| ClipVitG14 | | | ✅ (s12B_b42K_224) |
| ClipVitBigG14 | | | ✅ (39B_b160k_224) |

## Features and Capabilities

- **Zero-Shot Classification:** Capable of classifying images into arbitrary categories without targeted finetuning.
- **Cross-Modal Retrieval:** Aligned text and image embeddings space enables semantic image search via natural language queries.
- **Robust Representations:** Extracted vision and text embeddings provide powerful foundations for multiple downstream tasks.
- **Flexible Scalability:** Offers various Vision Transformer (ViT) sizes scaling from Base to Large.

## Basic Usage

```python
import keras
import kmodels

processor = kmodels.models.clip.CLIPProcessor()
model = kmodels.models.clip.ClipVitBase16(
    weights="openai_224",
    input_shape=(224, 224, 3),
)
inputs = processor(text=["mountains", "tortoise", "cat"], image_paths="cat1.jpg")
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["input_ids"],
        "padding_mask": inputs["attention_mask"],
    }
)

preds = keras.ops.softmax(output["image_logits"]).numpy().squeeze()
result = dict(zip(["mountains", "tortoise", "cat"], preds))
print("\nPrediction probabilities:")
print(result)

#output:
"""
Prediction probabilities:
{'mountains': np.float32(0.0006278555), 'tortoise': np.float32(0.000326458), 'cat': np.float32(0.99904567)}
"""
```

## Batch Processing Multiple Images

```python
import keras
import kmodels

processor = kmodels.models.clip.CLIPProcessor()
model = kmodels.models.clip.ClipVitBase16(weights="openai_224")

image_paths = ["dog.jpg", "cat1.jpg"]
labels = ["a photo of a dog", "a photo of a car", "a photo of a flower", "a photo of a cat"]

inputs = processor(text=labels, image_paths=image_paths)
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
    "padding_mask": inputs["attention_mask"],
})

probs = keras.ops.softmax(output["image_logits"], axis=-1).numpy()

for i, img_path in enumerate(image_paths):
    print(f"\nPredictions for {img_path}:")
    img_probs = probs[i]
    for j, label in enumerate(labels):
        print(f"  {label}: {img_probs[j]:.4f}")
```
