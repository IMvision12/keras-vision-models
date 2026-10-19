# Diffusion Models

<div class="kf-note kf-note--weights">
<b>Weights:</b> every checkpoint is hosted under the
<a href="https://huggingface.co/zeromodels">zeromodels organization</a> as
<code>zm_config.json</code> + Keras weights + <code>tokenizer.json</code>. Load one with
<code>ZMDiffusion.from_weights("zeromodels/&lt;variant&gt;")</code>; no conversion is needed.
On-the-fly <code>hf:</code> conversion is not supported for diffusion models.
</div>

Text-to-image generation (and image-to-image with the SDXL refiner) in pure Keras 3, on
JAX, PyTorch or TensorFlow. Every family loads through two classes from
`zeromodels.diffusers`:

- `ZMDiffusion`: the model and its `generate`.
- `ZMDiffusionTokenizer`: the prompt tokenizer.

The checkpoint's `model_type` (in its `zm_config.json`) selects the family, so switching
from Stable Diffusion to Qwen-Image only changes the repo id.

## Supported models

| Family | `model_type` | Hosted checkpoints (`zeromodels/...`) |
|---|---|---|
| <span id="stable-diffusion"></span>Stable Diffusion 1.x | `stable_diffusion` | `stable-diffusion-v1-1` to `stable-diffusion-v1-5` |
| <span id="stable-diffusion-2"></span>Stable Diffusion 2.x | `stable_diffusion_2` | `stable-diffusion-2`, `stable-diffusion-2-base`, `stable-diffusion-2-1`, `stable-diffusion-2-1-base`, `sd-turbo` |
| <span id="stable-diffusion-xl"></span>Stable Diffusion XL | `stable_diffusion_xl` | `stable-diffusion-xl-base-1.0`, `sdxl-turbo` |
| Stable Diffusion XL refiner | `stable_diffusion_xl_refiner` | `stable-diffusion-xl-refiner-1.0` (image-to-image) |
| <span id="stable-diffusion-3"></span>Stable Diffusion 3 | `stable_diffusion_3` | `stable-diffusion-3-medium` |
| <span id="stable-diffusion-3-5"></span>Stable Diffusion 3.5 | `stable_diffusion_3_5` | `stable-diffusion-3.5-large`, `stable-diffusion-3.5-large-turbo`, `stable-diffusion-3.5-medium` |
| <span id="qwen-image"></span>Qwen-Image | `qwen_image` | `qwen-image` |
| <span id="qwen-image-2-1"></span>Qwen-Image-2.1 | `qwen_image_21` | `qwen-image-2.1` |

Stable Diffusion 3 and 3.5 can also attach the separately hosted T5-XXL text encoder,
[`zeromodels/t5-v1_1-xxl-encoder`](https://huggingface.co/zeromodels/t5-v1_1-xxl-encoder),
with `text_encoder_3=`.

## Example: Stable Diffusion 3

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

model = ZMDiffusion.from_weights(
    "zeromodels/stable-diffusion-3-medium",
    text_encoder_3="zeromodels/t5-v1_1-xxl-encoder",  # optional: omit to save memory
)
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-3-medium")

inputs = tokenizer(
    "a fluffy corgi wearing round sunglasses, sitting on a surfboard at a sunny beach, photo"
)
images = model.generate(**inputs, num_inference_steps=28, guidance_scale=7.0, seed=4)

print(images.shape, images.dtype)  # (1, 1024, 1024, 3) uint8
Image.fromarray(images[0]).save("corgi.png")
```

<img src="../assets/stable_diffusion_3_corgi.jpg" alt="Stable Diffusion 3 medium: a corgi wearing sunglasses on a surfboard at a sunny beach, 1024px" width="420">

## Example: Qwen-Image

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

model = ZMDiffusion.from_weights("zeromodels/qwen-image")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/qwen-image")

inputs = tokenizer(
    "a coffee shop entrance with a chalkboard sign reading 'Open', warm afternoon light"
)
images = model.generate(
    **inputs,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=4.0,
    seed=1,
)

print(images.shape, images.dtype)  # (1, 1024, 1024, 3) uint8
Image.fromarray(images[0]).save("coffee.png")
```

## Outputs

Generated with the hosted checkpoints, same code as above with the repo, prompt and
settings shown.

| | |
|---|---|
| <img src="../assets/stable_diffusion_fox.jpg" alt="Stable Diffusion 1.5: a red fox in a field of lavender" width="300"><br>**`stable-diffusion-v1-5`**, 50 steps, guidance 7.5, seed 1<br>*a red fox sitting in a field of lavender, golden hour, wildlife photography, sharp focus* | <img src="../assets/stable_diffusion_2_ramen.jpg" alt="Stable Diffusion 2.1-base: a bowl of ramen" width="300"><br>**`stable-diffusion-2-1-base`**, 50 steps, guidance 7.5, seed 2<br>*a steaming bowl of ramen on a wooden table, food photography, shallow depth of field* |
| <img src="../assets/stable_diffusion_2_sd_turbo_raccoon.jpg" alt="SD-Turbo, one step: a baby raccoon in a priest robe" width="300"><br>**`sd-turbo`**, 1 step, no guidance<br>*a cinematic shot of a baby raccoon wearing an intricate italian priest robe* | <img src="../assets/stable_diffusion_xl_turbo_raccoon.jpg" alt="SDXL-Turbo, one step: a baby raccoon in a priest robe" width="300"><br>**`sdxl-turbo`**, 1 step, no guidance<br>*a cinematic shot of a baby raccoon wearing an intricate italian priest robe* |
| <img src="../assets/stable_diffusion_xl_lighthouse.jpg" alt="Stable Diffusion XL base: a lighthouse at dawn" width="300"><br>**`stable-diffusion-xl-base-1.0`**, 30 steps, guidance 5.0, seed 3<br>*a lighthouse on a rocky coast at dawn, dramatic clouds, cinematic, highly detailed* | <img src="../assets/stable_diffusion_xl_refiner_lighthouse.jpg" alt="Stable Diffusion XL base + refiner: the lighthouse, refined" width="300"><br>**base + `stable-diffusion-xl-refiner-1.0`**, base to 80% then refiner (`denoising_start=0.8`)<br>*the same lighthouse prompt* |
| <img src="../assets/stable_diffusion_3_corgi.jpg" alt="Stable Diffusion 3 medium: a corgi with sunglasses on a surfboard" width="300"><br>**`stable-diffusion-3-medium`** + T5, 28 steps, guidance 7.0, seed 4<br>*a fluffy corgi wearing round sunglasses, sitting on a surfboard at a sunny beach, photo* | <img src="../assets/stable_diffusion_3_5_treehouse.jpg" alt="Stable Diffusion 3.5 medium: a treehouse village at twilight" width="300"><br>**`stable-diffusion-3.5-medium`** + T5, 40 steps, guidance 4.5, seed 5<br>*a whimsical treehouse village at twilight, lanterns glowing, watercolor illustration* |

### Verified against diffusers

The same prompt and initial latent through ZeroModels (left) and 🤗 diffusers (right).
Stable Diffusion 1.5 matches to within one pixel level (99.3% of pixels identical,
PSNR 74.7 dB).

<img src="../assets/stable_diffusion_text_to_image_output.jpg" alt="ZeroModels (left) and diffusers (right): Stable Diffusion 1.5, red fox in lavender" width="640">

<img src="../assets/stable_diffusion_2_text_to_image_output.jpg" alt="ZeroModels (left) and diffusers (right): Stable Diffusion 2.1-base, bowl of ramen" width="640">

<img src="../assets/stable_diffusion_xl_text_to_image_output.jpg" alt="ZeroModels (left) and diffusers (right): Stable Diffusion XL base, lighthouse" width="640">

## `generate`

Pass the tokenizer's output, plus any of:

| Argument | Meaning |
|---|---|
| `negative_input_ids` | a tokenized negative prompt (`tokenizer("...")["input_ids"]`) |
| `num_inference_steps`, `guidance_scale` | sampler steps and guidance strength; default to the repo's own settings |
| `seed` | reproducible noise on one backend |
| `latents` | explicit initial noise, for identical images on every backend |
| `image`, `strength` | image-to-image: start from an image instead of noise |
| `denoising_end`, `denoising_start`, `output_type="latent"` | split a run between the SDXL base and refiner |
| `height`, `width` | output size (Qwen-Image families) |

It returns `(batch, height, width, 3)` uint8 NumPy images. Prompts batch through the
tokenizer: `tokenizer(["a cat", "a dog"])`.

## Loading options

`from_weights` also takes `load_dtype="float16"` / `"bfloat16"` / `"float32"`,
`quantization="int8"` / `"int4"` for lower memory, and size overrides such as
`unet_sample_size=96, vae_sample_size=768` to build for another resolution. The sampler can
be swapped at any time with `model.scheduler = ...` (`zeromodels.base.base_scheduler`), and
the trained components are attributes (`.unet` or `.transformer`, `.vae`,
`.text_encoder`), each callable on its own.

The same repos also load through the Auto API: `AutoZModel.from_weights(repo)` returns a
`ZMDiffusion` and `AutoZMTokenizer.from_weights(repo)` a `ZMDiffusionTokenizer`.
