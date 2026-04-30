# SigLIP2

**Paper**: [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)

SigLIP2 is the next generation of the SigLIP model, building upon the original architecture with enhanced Vision Transformer encoders, an improved text transformer, and expanded vocabulary support (256K vocabulary size) for better multilingual understanding and performance.

## Model Variants

- **SigLIP2BaseP16** — Base model with 16x16 patch size (768 hidden dimensions)
- **SigLIP2BaseP32** — Base model with 32x32 patch size (768 hidden dimensions)
- **SigLIP2LargeP16** — Large model with 16x16 patch size (1024 hidden dimensions)
- **SigLIP2So400mP14** — So400m model with 14x14 patch size (1152 hidden dimensions)
- **SigLIP2So400mP16** — So400m model with 16x16 patch size (1152 hidden dimensions)

## Available Weights

| Variant | google_224 | google_256 | google_384 | google_512 |
|---------|:-:|:-:|:-:|:-:|
| SigLIP2BaseP16 | ✅ | ✅ | ✅ | ✅ |
| SigLIP2BaseP32 | | ✅ | | |
| SigLIP2LargeP16 | | ✅ | ✅ | ✅ |
| SigLIP2So400mP14 | ✅ | | ✅ | |
| SigLIP2So400mP16 | | ✅ | ✅ | ✅ |

## Basic Usage

```python
import keras
import kmodels

processor = kmodels.models.siglip2.SigLIP2Processor()
model = kmodels.models.siglip2.SigLIP2BaseP16(
   weights="google_224",
   input_shape=(224, 224, 3),
)
inputs = processor(text=["mountains", "tortoise", "cat"], image_paths="cat1.jpg")
output = model(
   {
       "images": inputs["images"],
       "token_ids": inputs["input_ids"],
   }
)

preds = keras.ops.softmax(output["image_logits"]).numpy().squeeze()
result = dict(zip(["mountains", "tortoise", "cat"], preds))
print("\nPrediction probabilities:")
print(result)
```

### Data format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
processor = SigLIP2ImageProcessor(data_format="channels_first")
inputs = processor("photo.jpg")
```

Image processors return tensors in the requested layout; post-processors accept tensors in either layout and read the flag to pick the channel axis. See `docs/utils.md` for which families have format-sensitive post-processors.

## Batch Processing Multiple Images

```python
import keras
import kmodels

processor = kmodels.models.siglip2.SigLIP2Processor()
model = kmodels.models.siglip2.SigLIP2BaseP16(weights="google_224")

image_paths = ["dog.jpg", "cat1.jpg"]
labels = ["a photo of a dog", "a photo of a car", "a photo of a flower", "a photo of a cat"]

inputs = processor(text=labels, image_paths=image_paths)
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
})

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
import kmodels

processor = kmodels.models.siglip2.SigLIP2Processor()
model = kmodels.models.siglip2.SigLIP2BaseP16(weights="google_256", input_shape=(224, 224, 3))

multilingual_labels = [
    "un gato",   # Spanish: a cat
    "ein Vogel", # German: a bird
    "कुत्ता",     # Hindi: dog
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

```python
import keras
import kmodels

processor = kmodels.models.siglip2.SigLIP2Processor(image_resolution=384)
model = kmodels.models.siglip2.SigLIP2So400mP16(
    weights="google_384",
    input_shape=(384, 384, 3)
)

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
