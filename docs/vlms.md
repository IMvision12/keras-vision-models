# Vision-Language Models (VLMs)

## CLIP

### Basic Usage

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

For more details, see the [CLIP model documentation](../kmodels/models/clip/readme.md).

## SigLIP

### Basic Usage

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

For more details, see the [SigLIP model documentation](../kmodels/models/siglip/readme.md).

## SigLIP2

### Basic Usage

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

For more details, see the [SigLIP2 model documentation](../kmodels/models/siglip2/readme.md).

## Available VLM Models

| Model Name | Reference Paper | Source of Weights |
|------------|----------------|-------------------|
| CLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | `transformers` |
| SigLIP | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | `transformers` |
| SigLIP2 | [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786) | `transformers` |
