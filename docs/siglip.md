# SigLIP

**Paper**: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

SigLIP (Sigmoid Loss for Language Image Pre-training) uses a simple pairwise sigmoid loss for Language-Image Pre-training which operates solely on image-text pairs without requiring a global view of pairwise similarities for normalization. It achieves better zero-shot performance compared to CLIP on classification tasks while being more efficient during training.

## Model Variants

- **SigLIPBaseP16** — Base model with 16x16 patch size (768 hidden dimensions)
- **SigLIPLargeP16** — Large model with 16x16 patch size (1024 hidden dimensions)
- **SigLIPSo400mP14** — So400m model with 14x14 patch size (1152 hidden dimensions)

## Available Weights

| Variant | google_224 | google_256 | google_multilingual_256 | google_384 | google_512 |
|---------|:-:|:-:|:-:|:-:|:-:|
| SigLIPBaseP16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| SigLIPLargeP16 | | ✅ | | ✅ | |
| SigLIPSo400mP14 | ✅ | | | ✅ | |

## Basic Usage

```python
import keras
import kmodels

processor = kmodels.models.siglip.SigLIPProcessor()
model = kmodels.models.siglip.SigLIPBaseP16(
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
processor = SigLIPImageProcessor(data_format="channels_first")
inputs = processor("photo.jpg")
```

Image processors return tensors in the requested layout; post-processors accept tensors in either layout and read the flag to pick the channel axis. See `docs/utils.md` for which families have format-sensitive post-processors.

## Batch Processing Multiple Images

```python
import keras
import kmodels

processor = kmodels.models.siglip.SigLIPProcessor()
model = kmodels.models.siglip.SigLIPBaseP16(weights="google_224")

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

## Multilingual Support

```python
import keras
import kmodels

processor = kmodels.models.siglip.SigLIPProcessor(multilingual=True)
model = kmodels.models.siglip.SigLIPBaseP16(weights="google_multilingual_256", input_shape=(224, 224, 3))

multilingual_labels = [
    "un gato",   # Spanish: a cat
    "un chien",  # French: a dog
    "ein Vogel", # German: a bird
    "cat"        # English: cat
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
