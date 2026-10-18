# Diffusers

<div class="kf-note kf-note--weights">
<b>Weights:</b> every diffusion checkpoint is hosted under the
<a href="https://huggingface.co/zeromodels">zeromodels organization</a> as one
<code>zm_config.json</code> + <code>model.weights.h5</code> (or sharded
<code>model.weights.json</code>) + <code>tokenizer.json</code>. Load any of them with
<code>ZMDiffusion.from_weights("zeromodels/&lt;variant&gt;")</code>. On-the-fly
<code>hf:</code> conversion is not supported for diffusion models.
</div>

`zeromodels.diffusers` holds every text-to-image model of ZeroModels behind **one
pipeline class and one tokenizer**, the way `zeromodels.sentence_transformers` holds the
sentence-embedding models:

- `ZMDiffusion`: the model and its `generate` (diffusers' `DiffusionPipeline`).
- `ZMDiffusionTokenizer`: the prompt tokenizer.

These two are the whole public API (`zeromodels.diffusers.__all__`); everything else in the
package (the family recipes, configs and components) is internal.

| | |
|---|---|
| Class | `ZMDiffusion` |
| Tokenizer | `ZMDiffusionTokenizer` |
| Selector | `model_type` (the architecture differs per family) |
| Recipes | Family classes in `diffusers/<family>/`, registered in `DIFFUSION_FAMILIES` |
| Hub files | `zm_config.json` + `zm_preprocessor.json` + `tokenizer.json` |

A checkpoint's `model_type` (recorded in its `zm_config.json`) selects the family; the
family decides which components are built and how a prompt becomes an image, so loading
and generating look the same for Stable Diffusion 1.x and Qwen-Image alike. The public API
is `zeromodels.diffusers`, separate from the upstream `diffusers` library (which is only
needed to convert checkpoints).

## Usage

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

repo = "zeromodels/stable-diffusion-xl-base-1.0"
pipe = ZMDiffusion.from_weights(repo)
tokenizer = ZMDiffusionTokenizer.from_weights(repo)

images = pipe.generate(**tokenizer("a lighthouse on a rocky coast at dawn"), seed=0)
Image.fromarray(images[0]).save("lighthouse.png")  # (1024, 1024, 3) uint8
```

Use the same repo for the model and the tokenizer. Swapping the repo id is all it takes to
switch family; the defaults (`num_inference_steps`, `guidance_scale`, the sampler) come
from each repo.

## Families

| `model_type` | Hosted checkpoints | Page |
|---|---|---|
| `stable_diffusion` | `stable-diffusion-v1-1` to `stable-diffusion-v1-5` | [Stable Diffusion](stable_diffusion.md) |
| `stable_diffusion_2` | `stable-diffusion-2`, `stable-diffusion-2-base`, `stable-diffusion-2-1`, `stable-diffusion-2-1-base`, `sd-turbo` | [Stable Diffusion 2](stable_diffusion_2.md) |
| `stable_diffusion_xl` | `stable-diffusion-xl-base-1.0`, `sdxl-turbo` | [Stable Diffusion XL](stable_diffusion_xl.md) |
| `stable_diffusion_xl_refiner` | `stable-diffusion-xl-refiner-1.0` (image-to-image) | [Stable Diffusion XL](stable_diffusion_xl.md#the-refiner) |
| `stable_diffusion_3` | `stable-diffusion-3-medium` | [Stable Diffusion 3](stable_diffusion_3.md) |
| `stable_diffusion_3_5` | `stable-diffusion-3.5-large`, `stable-diffusion-3.5-large-turbo`, `stable-diffusion-3.5-medium` | [Stable Diffusion 3.5](stable_diffusion_3_5.md) |
| `qwen_image` | `qwen-image` | [Qwen-Image](qwen_image.md) |
| `qwen_image_21` | `qwen-image-2.1` | [Qwen-Image-2.1](qwen_image_21.md) |

The family pages cover each architecture, its generation arguments, the verification
against diffusers and the memory needed.

## `ZMDiffusion`

```python
ZMDiffusion(model_type, scheduler=None, **config_fields)
ZMDiffusion(config)  # or a family config positionally (zeromodels.diffusers.<family>)
ZMDiffusion.from_weights("zeromodels/<variant>", **kwargs)
```

One functional `keras.Model` holds every trained component as disconnected paths: the
denoiser (`{"sample", "timestep", "encoder_hidden_states", ...} -> "noise_pred"`), the VAE
(`{"image", "latent"} -> "moments" / "image"`) and the text towers (token ids to prompt
embeddings). The components are attributes that `generate` runs step by step and that can
be called on their own:

| Family | Denoiser | VAE | Text towers |
|---|---|---|---|
| SD 1.x / 2.x | `.unet` | `.vae` | `.text_encoder` |
| SDXL | `.unet` | `.vae` | `.text_encoder`, `.text_encoder_2` |
| SDXL refiner | `.unet` | `.vae` | `.text_encoder_2` |
| SD 3 / 3.5 | `.transformer` | `.vae` | `.text_encoder`, `.text_encoder_2` (+ optional `.text_encoder_3`) |
| Qwen-Image / 2.1 | `.transformer` | `.vae` | `.text_encoder` |

`model.config` is the typed family config (`StableDiffusionXLConfig` for an SDXL
checkpoint), `model.model_type` the family, `model.scheduler` the sampler (built from the
repo's `scheduler_config`; assign another at any time) and `model.generate_args` the repo's
default generation settings.

`from_weights` accepts everything [`BaseModel.from_weights`](loading_weights.md) does
(`load_dtype`, `quantization`, `attn_implementation`, `cache_converted`, size overrides such
as `unet_sample_size=96, vae_sample_size=768`) plus `text_encoder_3` for SD 3 / 3.5: a
hosted `SD3T5EncoderModel` repo id (`"zeromodels/t5-v1_1-xxl-encoder"`) or a built
encoder, attached outside the checkpoint's weights.

### `generate`

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
    image=None,
    strength=None,
    denoising_start=None,
    denoising_end=None,
    output_type="image",
    **family_arguments,
)
```

The shared loop is [`BaseDiffusion.generate`](main_classes.md#basediffusion): tokenized
prompts in, `(batch, H, W, 3)` uint8 images (or the final latent) out, with classifier-free
guidance, image-to-image (`image` + `strength`) and the base / refiner split
(`denoising_end` / `denoising_start`). Family-specific arguments:

| Family | Arguments |
|---|---|
| SDXL / refiner | `original_size`, `crops_coords_top_left`, `target_size` (`(height, width)` pairs), `aesthetic_score` / `negative_aesthetic_score` on the refiner, `negative_*` variants |
| SD 3 / 3.5 | `input_ids_3` / `negative_input_ids_3` (the T5 tokens the tokenizer returns) |
| Qwen-Image / 2.1 | `height`, `width`, `negative_attention_mask` |

## `ZMDiffusionTokenizer`

```python
ZMDiffusionTokenizer(
    model_type,
    hf_id=None,
    tokenizer_file=None,
    tokenizer_file_3=None,
    max_seq_len=None,
    max_sequence_length=None,
    pad_token=None,
)
ZMDiffusionTokenizer.from_weights("zeromodels/<variant>")
```

Calling it on a string or a list of strings returns what `generate` takes:

| `model_type` | Encoding | Returns |
|---|---|---|
| `stable_diffusion`, `stable_diffusion_xl`, `stable_diffusion_xl_refiner` | CLIP BPE, `<|endoftext|>`-padded to 77 | `input_ids`, `attention_mask` |
| `stable_diffusion_2` | CLIP BPE, `!`-padded to 77 | `input_ids`, `attention_mask` |
| `stable_diffusion_3`, `stable_diffusion_3_5` | CLIP BPE (77) + T5 SentencePiece (`tokenizer_3.json`, 256) | `input_ids`, `attention_mask`, `input_ids_3` |
| `qwen_image`, `qwen_image_21` | the pipeline's ChatML template, Qwen2 / Qwen3 BPE | `input_ids`, `attention_mask` |

`decode(ids)` turns ids back into text. `from_weights` reads the family and its settings
from the repo's `zm_preprocessor.json`, or from the `model_type` of its `zm_config.json`
when the repo has no `zm_preprocessor.json`, and the vocabulary from the repo's
`tokenizer.json`.

## Auto loading

The same repos work with the Auto API:

```python
from zeromodels import AutoZModel, AutoZMTokenizer

repo = "zeromodels/qwen-image"
pipe = AutoZModel.from_weights(repo)  # ZMDiffusion
tokenizer = AutoZMTokenizer.from_weights(repo)  # ZMDiffusionTokenizer
```

`AutoZMTextToImage` (and `AutoZMImageToImage` for the SDXL refiner) resolve to
`ZMDiffusion` as well. See [Auto](auto.md).

## Hub metadata

A repo's `zm_config.json` names the class and the family:

```json
{
  "library_name": "zeromodels",
  "model_module": "zeromodels.diffusers",
  "model_class": "ZMDiffusion",
  "variant": "stable-diffusion-v1-5",
  "model_type": "stable_diffusion",
  "unet_config": {"...": "..."},
  "vae_config": {"...": "..."},
  "text_config": {"...": "..."},
  "scheduler_config": {"_class_name": "PNDMScheduler", "...": "..."},
  "generate_args": {"num_inference_steps": 50, "guidance_scale": 7.5}
}
```

and `zm_preprocessor.json` the tokenizer recipe (the files themselves are read from the
same repo):

```json
{
  "library_name": "zeromodels",
  "preprocessor_module": "zeromodels.diffusers",
  "preprocessor_class": "ZMDiffusionTokenizer",
  "variant": "stable-diffusion-v1-5",
  "model_type": "stable_diffusion",
  "max_seq_len": 77,
  "pad_token": "<|endoftext|>"
}
```

## Migrating from the per-family classes

The per-family container and pipeline classes were replaced by `ZMDiffusion`, and the
per-family tokenizers by `ZMDiffusionTokenizer`, the only two public names. The
components and configs moved from `zeromodels.models.<family>` into the internal
`zeromodels.diffusers.<family>` packages.
The hosted weights are unchanged, and `zm_config.json` files that still name an old class
load into `ZMDiffusion`.

| Before | Now |
|---|---|
| `StableDiffusionTextToImage`, `StableDiffusionModel` (and the `2`, `XL`, `3`, `3_5` variants) | `ZMDiffusion` |
| `StableDiffusionXLRefinerImageToImage`, `StableDiffusionXLRefinerModel` | `ZMDiffusion` |
| `QwenImageTextToImage`, `QwenImageModel`, `QwenImage21TextToImage`, `QwenImage21Model` | `ZMDiffusion` |
| `StableDiffusionTokenizer` (and every family tokenizer) | `ZMDiffusionTokenizer` |
| `from zeromodels.models.stable_diffusion_3 import SD3T5EncoderModel` | `ZMDiffusion.from_weights(repo, text_encoder_3="zeromodels/t5-v1_1-xxl-encoder")` |

## Adding a family

A family is a plain recipe class (for example `StableDiffusionFamily` in
`zeromodels/diffusers/stable_diffusion/stable_diffusion_model.py`) registered in
`DIFFUSION_FAMILIES` (`diffusion_model.py`) under its config's `model_type`. It builds the
components and the graph (`build_components`, `build_graph`) and implements the
`BaseDiffusion` hooks (`encode_prompt`, `unconditional_ids`, `predict_noise`,
`decode_latents`, `latent_shape`, `default_scheduler`, and `generate` / `denoise` /
`prepare_latents` where it differs from the shared loop), each taking the `ZMDiffusion`
instance; a hook the family does not define falls back to `BaseDiffusion`'s. Its config
is a plain `BaseConfig` whose `model_type` names the family. Add the tokenizer settings to
`DIFFUSION_TOKENIZERS` (`diffusion_tokenizer.py`) and the `model_type` to the Auto tables.
