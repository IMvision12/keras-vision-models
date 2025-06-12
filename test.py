import keras

import kvmm

processor = kvmm.models.clip.CLIPProcessor()
model = kvmm.models.clip.ClipVitBase16(
    weights="res_224px",
    input_shape=(224, 224, 3),
)
inputs = processor(text=["mountains", "tortoise", "bird"], image_paths="images/bird.png")
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