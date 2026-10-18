# Diffusion Models

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
| `stable_diffusion` | `stable-diffusion-v1-1` to `stable-diffusion-v1-5` | [Stable Diffusion](#stable-diffusion) |
| `stable_diffusion_2` | `stable-diffusion-2`, `stable-diffusion-2-base`, `stable-diffusion-2-1`, `stable-diffusion-2-1-base`, `sd-turbo` | [Stable Diffusion 2](#stable-diffusion-2) |
| `stable_diffusion_xl` | `stable-diffusion-xl-base-1.0`, `sdxl-turbo` | [Stable Diffusion XL](#stable-diffusion-xl) |
| `stable_diffusion_xl_refiner` | `stable-diffusion-xl-refiner-1.0` (image-to-image) | [Stable Diffusion XL](#stable-diffusion-xl-the-refiner) |
| `stable_diffusion_3` | `stable-diffusion-3-medium` | [Stable Diffusion 3](#stable-diffusion-3) |
| `stable_diffusion_3_5` | `stable-diffusion-3.5-large`, `stable-diffusion-3.5-large-turbo`, `stable-diffusion-3.5-medium` | [Stable Diffusion 3.5](#stable-diffusion-3-5) |
| `qwen_image` | `qwen-image` | [Qwen-Image](#qwen-image) |
| `qwen_image_21` | `qwen-image-2.1` | [Qwen-Image-2.1](#qwen-image-2-1) |

The family sections below cover each architecture, its generation arguments, the
verification against diffusers and the memory needed.

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

## Stable Diffusion 1.x {#stable-diffusion}

Stable Diffusion 1.x, ported to pure Keras 3: latent text-to-image diffusion with a
UNet denoiser, a KL-regularized autoencoder and the CLIP ViT-L/14 text encoder. It loads
through [`ZMDiffusion`](#zmdiffusion), the one diffusion pipeline class of zeromodels,
with `model_type="stable_diffusion"`: a single functional graph with the three towers as
disconnected sub-graphs (the way `CLIPModel` holds its two), so a hosted repo is one
`model.weights.h5` and one `zm_config.json`, plus `generate`, the classifier-free
guidance denoising loop of `BaseDiffusion` (the diffusion counterpart of
`BaseGeneration`). Prompts are tokenized by `ZMDiffusionTokenizer`.

All five SD 1.x checkpoints (v1-1 to v1-5) share this architecture; only the weights
differ. The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion
is deliberately **not supported** for diffusion models.

Key facts of the port:

- **Both data formats**: the UNet and VAE follow `keras.config.image_data_format()`
  (every conv takes it, GroupNorm normalizes the layout's channel axis, and the two
  token <-> grid boundaries transpose around their flatten under `channels_first`).
  Weights are layout-independent, so one hosted checkpoint serves both, and the
  converter is a `(O, I, H, W) -> (H, W, I, O)` kernel transpose and nothing else.
- **Block-level layers**: the UNet and VAE are built from composite Keras layers
  (`StableDiffusionResnetBlock2D`, `StableDiffusionTransformer2DModel`,
  `StableDiffusionCrossAttention`, ...). A functional graph keeps every node's output
  alive until the forward ends, and at 512px the UNet's per-op intermediates (4096x4096
  attention maps above all) would need several GB; a layer's internals are freed when
  its call returns.
- **Schedulers match diffusers to the bit**: the training noise schedule is built in
  float32 the way torch does it, so `alphas_cumprod` and the Euler sigmas are identical
  and a 50-step run stays within float rounding of the reference.
- **Verified end to end**: at 512px, fp32, PNDM 50 steps and the same initial latent, the
  generated image matches diffusers' to within **1 uint8 level** (99.3% of pixels
  identical, PSNR 75 dB); the UNet, VAE and text encoder outputs agree to ~1e-6.

Links:

- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models (arXiv:2112.10752)](https://arxiv.org/abs/2112.10752)
- Reference implementation: [diffusers `StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
- License: [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

See also [clip.md](clip.md) (the text encoder is `CLIPTextModel`).

### Variants {#stable-diffusion-variants}

Preconverted, float32 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. Each repo is one container: UNet (860M) + VAE
(84M) + CLIP text encoder (123M), 1.07B parameters, 3.97 GB. All checkpoints are released
under the CreativeML OpenRAIL-M license, which carries use-based restrictions.

| Variant | Hub | Training |
|---|---|---|
| `stable-diffusion-v1-1` | [`zeromodels/stable-diffusion-v1-1`](https://huggingface.co/zeromodels/stable-diffusion-v1-1) | 237k steps at 256px on laion2B-en, then 194k steps at 512px on laion-high-resolution |
| `stable-diffusion-v1-2` | [`zeromodels/stable-diffusion-v1-2`](https://huggingface.co/zeromodels/stable-diffusion-v1-2) | v1-1 + 515k steps at 512px on laion-aesthetics v2 5+ |
| `stable-diffusion-v1-3` | [`zeromodels/stable-diffusion-v1-3`](https://huggingface.co/zeromodels/stable-diffusion-v1-3) | v1-2 + 195k steps at 512px, 10% text-conditioning dropout |
| `stable-diffusion-v1-4` | [`zeromodels/stable-diffusion-v1-4`](https://huggingface.co/zeromodels/stable-diffusion-v1-4) | v1-2 + 225k steps at 512px, 10% text-conditioning dropout |
| `stable-diffusion-v1-5` | [`zeromodels/stable-diffusion-v1-5`](https://huggingface.co/zeromodels/stable-diffusion-v1-5) | v1-2 + 595k steps at 512px, 10% text-conditioning dropout |

The 10% text-conditioning dropout of v1-3 onwards is what makes classifier-free guidance
work well; v1-5 is the usual default.

### API {#stable-diffusion-api}

Configs are typed: `StableDiffusionConfig` (composite) over `UNet2DConditionConfig`,
`AutoencoderKLConfig` and `StableDiffusionTextConfig` (a `CLIPTextConfig` with the
ViT-L/14 defaults), plus the checkpoint's `scheduler_config` and the CLIP special token
ids. Each repo's `zm_config.json` parses through it; the constructor stays flat, with the
sub-config fields prefixed `unet_` / `vae_` / `text_`.

#### `ZMDiffusion` {#stable-diffusion-zmdiffusion}

The pipeline class of every diffusion family (see [Diffusers](#zmdiffusion)); a repo's
`model_type` (`"stable_diffusion"` here) selects the SD 1.x family, which supplies the
hooks `BaseDiffusion`'s `generate` needs (`encode_prompt` on the CLIP tower, the empty
prompt's `unconditional_ids`, `predict_noise` on the UNet, `decode_latents` on the VAE)
and a `scheduler`, built from the config's `scheduler_config` (PNDM for every SD 1.x
checkpoint).

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | `(batch, 77)` token ids, `**tokenizer(prompts)` |
| `attention_mask` | `None` | accepted alongside the tokenizer's output; SD attends its padding tokens, so it is unused |
| `negative_input_ids` | `None` | `(batch, 77)` tokenized negative prompt for guidance; the empty prompt when unset |
| `num_inference_steps` | `None` | scheduler steps; the repo's `generate_args` (50) when unset |
| `guidance_scale` | `None` | classifier-free guidance strength; `generate_args` (7.5) when unset, `<= 1` disables it |
| `seed` | `None` | seed for the initial latent, reproducible per backend |
| `latents` | `None` | explicit `(batch, 64, 64, 4)` initial latent; identical results across backends |

Returns `(batch, 512, 512, 3)` uint8 numpy images. The denoiser call (the UNet plus the
guidance combine) runs as a per-backend compiled function cached on the instance,
`jax.jit` on JAX and `tf.function(jit_compile=True)` on TensorFlow, eager on Torch; its
shapes are the same at every step, so one compile serves the whole loop.

| Constructor arg | Default | Meaning |
|---|---|---|
| `model_type` | required | `"stable_diffusion"` (read from the repo's `zm_config.json` on `from_weights`) |
| `scheduler` | `None` | a `BaseScheduler`; built from `scheduler_config` when unset |
| `unet_sample_size` | `64` | latent size the UNet graph is built for (image size / 8) |
| `vae_sample_size` | `512` | image size the VAE graphs are built for |
| `unet_*` / `vae_*` / `text_*` | SD 1.x | the flat sub-config fields (see [Configuration](configuration.md)) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | `49406` / `49407` / `49407` | CLIP's special tokens, used to build the empty prompt |
| `scheduler_config` | `None` | the diffusers scheduler dict (`_class_name`, betas, `steps_offset`, ...) |

The graph is three disconnected paths, one per component. Inputs
`{"sample", "timestep", "encoder_hidden_states"}` (UNet), `{"image", "latent"}` (VAE
encode / decode) and `{"token_ids", "padding_mask"}` (text); outputs
`{"noise_pred", "moments", "image", "text_embeds"}`. The components are exposed as
`.unet`, `.vae` and `.text_encoder` and share the model's weights, so one load fills all
three, and each can be called on its own.

#### `UNet2DConditionModel` {#stable-diffusion-unet2dconditionmodel}

The denoiser (diffusers' `UNet2DConditionModel`): a symmetric down / mid / up UNet whose
ResNet blocks are conditioned on the sinusoidal timestep embedding and whose
`CrossAttn` blocks attend the latent to the text context. Inputs
`{"sample": (B, h, w, 4), "timestep": (B,), "encoder_hidden_states": (B, 77, 768)}`,
output `{"sample": (B, h, w, 4)}`.

| Arg | Default | Meaning |
|---|---|---|
| `sample_size` | `64` | latent spatial size the graph is built for |
| `in_channels` / `out_channels` | `4` | latent channels |
| `down_block_types` / `up_block_types` | SD 1.x | `CrossAttnDownBlock2D` x3 + `DownBlock2D`, mirrored |
| `block_out_channels` | `(320, 640, 1280, 1280)` | width per level |
| `layers_per_block` | `2` | ResNet blocks per level |
| `cross_attention_dim` | `768` | width of the text context |
| `num_attention_heads` | `8` | heads in the cross-attention blocks |
| `norm_num_groups` | `32` | GroupNorm groups |
| `text_seq_len` | `77` | static text length |

#### `AutoencoderKL` {#stable-diffusion-autoencoderkl}

The VAE (diffusers' `AutoencoderKL`): `encode(image, sample=False)` returns the posterior
mean (or a sample) of a `(B, H/8, W/8, 4)` latent and `decode(latent)` the `(B, H, W, 3)`
image in `[-1, 1]`; multiply / divide by `scaling_factor` (0.18215) around the UNet, as
`generate` does. One functional graph with the encoder and decoder as two disconnected
paths.

| Arg | Default | Meaning |
|---|---|---|
| `in_channels` / `out_channels` | `3` | image channels |
| `latent_channels` | `4` | latent channels |
| `block_out_channels` | `(128, 256, 512, 512)` | width per level, x8 spatial compression |
| `layers_per_block` | `2` | ResNet blocks per encoder level |
| `norm_num_groups` | `32` | GroupNorm groups |
| `sample_size` | `512` | image size the graphs are built for |
| `scaling_factor` | `0.18215` | latent scaling |

#### Schedulers {#stable-diffusion-schedulers}

`zeromodels.base.base_scheduler` has the SD samplers as weightless classes with the
diffusers interface (`set_timesteps`, `scale_model_input`, `step`,
`init_noise_sigma`), configured the diffusers way:

| Scheduler | Notes |
|---|---|
| `PNDMScheduler` | the SD 1.x default (PLMS, `skip_prk_steps=True`) |
| `DDIMScheduler` | deterministic, `eta=0` |
| `EulerDiscreteScheduler` | Karras-style sigma sampler, `linspace` spacing |
| `EulerAncestralDiscreteScheduler` | stochastic Euler (`seed` for reproducibility) |

`get_scheduler(config)` builds the one named by a diffusers scheduler config dict, which
is what the model does with its `scheduler_config`; swap at any time:

```python
from zeromodels.base.base_scheduler import EulerDiscreteScheduler

model.scheduler = EulerDiscreteScheduler.from_config(model.config.scheduler_config)
```

### Preprocessing {#stable-diffusion-preprocessing}

#### `ZMDiffusionTokenizer` {#stable-diffusion-zmdiffusiontokenizer}

The tokenizer of every diffusion family; for `model_type="stable_diffusion"` it is the
CLIP ViT-L/14 BPE tokenizer: `<|startoftext|>` / `<|endoftext|>` framing, truncated and
`<|endoftext|>`-padded to 77 tokens, the fixed context length the UNet's cross-attention
is built for. Calling it returns `{"input_ids", "attention_mask"}` for a string or a list
of strings. `from_weights("zeromodels/<variant>")` reads the family from the repo.

```python
ZMDiffusionTokenizer(
    model_type="stable_diffusion", hf_id=None, tokenizer_file=None, max_seq_len=77
)
```

There is no image processor and no processor class: text-to-image takes token ids in
and hands uint8 images out.

### End-to-end example {#stable-diffusion-end-to-end-example}

#### Single prompt {#stable-diffusion-single-prompt}

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

model = ZMDiffusion.from_weights("zeromodels/stable-diffusion-v1-5")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-v1-5")

inputs = tokenizer(
    "a red fox sitting in a field of lavender, golden hour, wildlife photography, sharp focus"
)
images = model.generate(**inputs, num_inference_steps=50, guidance_scale=7.5, seed=1)

Image.fromarray(images[0]).save("fox.png")  # (512, 512, 3) uint8
```

<img src="../assets/stable_diffusion_fox.jpg" alt="Stable Diffusion v1-5: a red fox sitting in a field of lavender at golden hour, 512px" width="380">

#### Negative prompt and batching {#stable-diffusion-negative-prompt-and-batching}

Prompts batch through the tokenizer; a negative prompt is tokenized the same way and
passed as `negative_input_ids` (one per prompt, or one broadcast row).

```python
prompts = [
    "a watercolor painting of a lighthouse at sunset",
    "a red vintage car on a coastal road",
]
inputs = tokenizer(prompts)  # {"input_ids": (2, 77), "attention_mask": (2, 77)}
negative = tokenizer(["blurry, low quality"] * len(prompts))["input_ids"]

images = model.generate(**inputs, negative_input_ids=negative, num_inference_steps=30)
for i, image in enumerate(images):
    Image.fromarray(image).save(f"out_{i}.png")
```

#### Reproducible across backends {#stable-diffusion-reproducible-across-backends}

`seed` reproduces a run on a given backend, but `keras.random` differs between torch,
jax and tensorflow. For an image that is identical everywhere, pass the initial latent
yourself (`(batch, 64, 64, 4)` at 512px):

```python
import numpy as np

latents = np.random.default_rng(0).standard_normal((1, 64, 64, 4)).astype("float32")
images = model.generate(**tokenizer("a bowl of ramen"), latents=latents)
```

#### Verified against diffusers {#stable-diffusion-verified-against-diffusers}

The fox prompt above with the same initial latent, PNDM 50 steps and guidance 7.5
through `ZMDiffusion` (left) and diffusers' `StableDiffusionPipeline`
(right), both fp32:

<img src="../assets/stable_diffusion_text_to_image_output.jpg" alt="zeromodels (left) and diffusers (right) generations of a red fox in lavender from the same latent" width="760">

```
tokenizer input_ids           identical
text_encoder                  max|d|=9.5e-06
unet noise_pred (t=981)       max|d|=9.5e-07
vae decode                    max|d|=3.5e-06
final image (uint8)           max|d|=1  mean|d|=0.0022  PSNR=74.7 dB  identical_pixels=99.34%
```

#### Components on their own {#stable-diffusion-components-on-their-own}

```python
from zeromodels.diffusers import ZMDiffusion

sd = ZMDiffusion.from_weights("zeromodels/stable-diffusion-v1-5")
latent = sd.vae.encode(image)  # (1, 512, 512, 3) in [-1, 1] -> (1, 64, 64, 4)
text = sd.text_encoder({"token_ids": ids, "padding_mask": mask})["last_hidden_state"]
```

#### Other resolutions {#stable-diffusion-other-resolutions}

The graphs are built for a fixed size, the weights are not. Rebuild at another size with
the constructor overrides (multiples of 64px; SD 1.x was trained at 512px):

```python
model = ZMDiffusion.from_weights(
    "zeromodels/stable-diffusion-v1-5", unet_sample_size=96, vae_sample_size=768
)
images = model.generate(**tokenizer("a mountain lake at dawn"))  # (1, 768, 768, 3)
```

### Data Format {#stable-diffusion-data-format}

**Both `channels_last` and `channels_first` are supported.** The models read
`keras.config.image_data_format()` when they are constructed (there is no
`data_format` argument), so set it once, before `from_weights`, and every tensor that
crosses the model boundary follows it:

| | `channels_last` (default) | `channels_first` |
|---|---|---|
| `latents` passed to `generate` | `(batch, 64, 64, 4)` | `(batch, 4, 64, 64)` |
| `AutoencoderKL.encode` input / `decode` output | `(batch, 512, 512, 3)` | `(batch, 3, 512, 512)` |
| UNet `sample` | `(batch, 64, 64, 4)` | `(batch, 4, 64, 64)` |

`generate` itself always hands back `(batch, height, width, 3)` uint8 images, whatever
the layout, so the PIL step is the same in both. The same hosted `model.weights.h5`
loads under either layout (conv kernels are `(kh, kw, in, out)` in both), and the two
runs agree to within 1 uint8 level (99.7% of pixels identical, the rest being the
NCHW / NHWC convolution kernels' rounding).

```python
import keras

keras.config.set_image_data_format("channels_first")

model = ZMDiffusion.from_weights("zeromodels/stable-diffusion-v1-5")
images = model.generate(
    **tokenizer("a bowl of ramen"), seed=0
)  # still (1, 512, 512, 3)
```

Note that `keras.config.set_image_data_format` is global state. Set it once at the top
of a script rather than toggling it between calls, since already-built models keep the
layout they were constructed with.

### Memory and speed {#stable-diffusion-memory-and-speed}

The fp32 container is 3.97 GB; a 512px, batch-1, guided run peaks at 6.5 GB of GPU
memory (the guided UNet call runs batch 2). On an RTX 4060 laptop the 50-step run takes
~25 s in eager Torch, about 1.5x diffusers' eager time. On JAX the compiled denoiser step
measured 2-3x faster than an op-by-op run; TensorFlow uses the same mechanism.

### Loading Fine-tuned Weights {#stable-diffusion-loading-fine-tuned-weights}

The five hosted checkpoints are the supported weights; any repo laid out like them
(`zm_config.json` declaring `ZMDiffusion` with `model_type: "stable_diffusion"`,
`model.weights.h5`, `tokenizer.json`) loads with `from_weights("<org>/<repo>")`; configs
written before `ZMDiffusion` (declaring `StableDiffusionModel`) load the same way. The
`hf:` prefix raises for diffusion models:
convert a diffusers-format checkpoint once with
`zeromodels/diffusers/stable_diffusion/convert_stable_diffusion_diffusers_to_keras.py`
(`transfer_stable_diffusion(repo)`, `pip install zeromodels[conversion]`) and host the result.

## Stable Diffusion 2.x {#stable-diffusion-2}

Stable Diffusion 2.x, ported to pure Keras 3: the [Stable Diffusion](#stable-diffusion)
latent-diffusion architecture with the second-generation conditioning. It loads
through [`ZMDiffusion`](#zmdiffusion) with `model_type="stable_diffusion_2"`, the SD 1.x
family with the SD 2 configuration, so everything on that page applies: one functional
model and one weights file per repo, `generate` from `BaseDiffusion`, the schedulers,
both data formats. What changes is the config:

- **Text encoder**: OpenCLIP ViT-H/14's text tower, 1024 wide, 16 heads, `gelu`, shipped
  truncated to its 23rd (penultimate) layer, which SD 2 conditions on.
- **UNet**: 1024-d cross-attention, one head count per level (`(5, 10, 20, 20)`, a 64-wide
  head everywhere, where SD 1.x used 8 heads at every level) and a linear token
  projection in the Transformer2D blocks instead of the 1x1 conv.
- **Tokenizer**: the same CLIP BPE, padded with `!` (id 0) the OpenCLIP way rather than
  `<|endoftext|>`, so the empty prompt used for classifier-free guidance is
  `[<|startoftext|>, <|endoftext|>, !, !, ...]`.
- **768px checkpoints** (`stable-diffusion-2`, `stable-diffusion-2-1`): built at a 96x96
  latent and trained with the **v-prediction** objective; their repos carry a
  v-prediction DDIM scheduler, which `generate` picks up from `scheduler_config`.
- **SD-Turbo** (`sd-turbo`): SD 2.1 distilled with Adversarial Diffusion Distillation to
  generate in 1 to 4 steps **without guidance**; its repo carries an Euler scheduler with
  `trailing` timestep spacing and `generate_args` of 1 step, `guidance_scale=0.0`.

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion is
deliberately **not supported** for diffusion models. The original `stabilityai/*` repos are
no longer on the Hub; the conversion sources are the `sd2-community` mirrors.

Links:

- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models (arXiv:2112.10752)](https://arxiv.org/abs/2112.10752)
- Reference implementation: [diffusers `StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
- License: [CreativeML Open RAIL++-M](https://huggingface.co/sd2-community/stable-diffusion-2-1/blob/main/LICENSE-MODEL)

See also [Stable Diffusion 1.x](#stable-diffusion), [clip.md](clip.md).

### Variants {#stable-diffusion-2-variants}

Preconverted, float32 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. Each repo is one container: UNet (866M) + VAE
(84M) + OpenCLIP text encoder (340M), 1.29B parameters, 5.16 GB (4.81 GiB, one
`model.weights.h5`). The SD 2 checkpoints are released under the CreativeML Open RAIL++-M
license; SD-Turbo under the Stability AI Non-Commercial Research Community License.

| Variant | Hub | Resolution | Objective | Training |
|---|---|---|---|---|
| `stable-diffusion-2-base` | [`zeromodels/stable-diffusion-2-base`](https://huggingface.co/zeromodels/stable-diffusion-2-base) | 512 | epsilon | from scratch: 550k steps at 256px on LAION-5B (aesthetics >= 4.5), then 850k steps at 512px |
| `stable-diffusion-2` | [`zeromodels/stable-diffusion-2`](https://huggingface.co/zeromodels/stable-diffusion-2) | 768 | v-prediction | 2-base + 150k steps at 768px |
| `stable-diffusion-2-1-base` | [`zeromodels/stable-diffusion-2-1-base`](https://huggingface.co/zeromodels/stable-diffusion-2-1-base) | 512 | epsilon | 2-base + 220k steps at 512px (punsafe 0.98) |
| `stable-diffusion-2-1` | [`zeromodels/stable-diffusion-2-1`](https://huggingface.co/zeromodels/stable-diffusion-2-1) | 768 | v-prediction | 2 + 55k steps (punsafe 0.1) + 155k steps (punsafe 0.98) at 768px |
| `sd-turbo` | [`zeromodels/sd-turbo`](https://huggingface.co/zeromodels/sd-turbo) | 512 | epsilon, 1 to 4 steps, no guidance | SD 2.1 distilled with Adversarial Diffusion Distillation (non-commercial) |

Use the `-base` checkpoints for 512px images and the others for 768px; each repo's
`zm_config.json` builds the graph at its native size.

### API {#stable-diffusion-2-api}

Configs are typed: `StableDiffusion2Config` (composite, `model_type`
`"stable_diffusion_2"`) over `StableDiffusion2UNetConfig` (a `UNet2DConditionConfig`
with the SD 2 widths), `AutoencoderKLConfig` and `StableDiffusion2TextConfig` (a
`CLIPTextConfig` with the ViT-H/14 sizes), plus the scheduler config and the token ids
(`pad_token_id` 0). Flat constructor, `unet_` / `vae_` / `text_` prefixes, like SD 1.x.

#### `ZMDiffusion` {#stable-diffusion-2-zmdiffusion}

`ZMDiffusion` with `model_type="stable_diffusion_2"` (read from the repo); `generate` is
the SD 1.x one:

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
)
```

Returns `(batch, H, W, 3)` uint8 images at the checkpoint's resolution; `latents` is
`(batch, 64, 64, 4)` for the 512px checkpoints and `(batch, 96, 96, 4)` for the 768px
ones (`channels_first`: channels second). Defaults come from the repo's `generate_args`
(50 steps, guidance 7.5). See [Stable Diffusion](#stable-diffusion-zmdiffusion)
for the argument table.

The graph is SD 1.x's three disconnected paths (`.unet`, `.vae`, `.text_encoder`) with
the same inputs / outputs, `encoder_hidden_states` 1024 wide.

The components are the SD 1.x classes, `UNet2DConditionModel` (built with
`num_attention_heads=(5, 10, 20, 20)`, `use_linear_projection=True`,
`cross_attention_dim=1024`) and `AutoencoderKL`; see their tables on the
[Stable Diffusion](#stable-diffusion-api) page.

### Preprocessing {#stable-diffusion-2-preprocessing}

#### `ZMDiffusionTokenizer` {#stable-diffusion-2-zmdiffusiontokenizer}

`ZMDiffusionTokenizer` with `model_type="stable_diffusion_2"`, which pads with `!`:
CLIP BPE, `<|startoftext|>` / `<|endoftext|>` framing, truncated and `!`-padded to 77
tokens. Returns `{"input_ids", "attention_mask"}`.

```python
ZMDiffusionTokenizer(
    model_type="stable_diffusion_2", hf_id=None, tokenizer_file=None, max_seq_len=77
)  # pad_token defaults to "!"
```

### End-to-end example {#stable-diffusion-2-end-to-end-example}

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

model = ZMDiffusion.from_weights("zeromodels/stable-diffusion-2-1-base")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-2-1-base")

inputs = tokenizer(
    "a steaming bowl of ramen on a wooden table, food photography, shallow depth of field"
)
images = model.generate(**inputs, num_inference_steps=50, guidance_scale=7.5, seed=2)

Image.fromarray(images[0]).save("ramen.png")  # (512, 512, 3) uint8
```

<img src="../assets/stable_diffusion_2_ramen.jpg" alt="Stable Diffusion 2.1-base: a steaming bowl of ramen on a wooden table, 512px" width="380">

#### 768px, v-prediction {#stable-diffusion-2-768px-v-prediction}

The 768px checkpoints need no extra arguments; the v-prediction DDIM scheduler and the
96x96 latent come from the repo:

```python
model = ZMDiffusion.from_weights("zeromodels/stable-diffusion-2-1")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-2-1")
images = model.generate(
    **tokenizer("a lighthouse on a cliff at dusk, oil painting")
)  # (1, 768, 768, 3)
```

#### SD-Turbo {#stable-diffusion-2-sd-turbo}

One step, no guidance (the repo's defaults; up to 4 steps sharpen a little):

```python
model = ZMDiffusion.from_weights("zeromodels/sd-turbo")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/sd-turbo")
images = model.generate(
    **tokenizer(
        "a cinematic shot of a baby raccoon wearing an intricate italian priest robe"
    )
)
```

<img src="../assets/stable_diffusion_2_sd_turbo_raccoon.jpg" alt="SD-Turbo, one step: a baby raccoon in an italian priest robe, 512px" width="380">

The same prompt and latent through diffusers' `StableDiffusionPipeline` (fp32) match to
1 uint8 level: 99.9% of pixels identical at 1 step (PSNR 82 dB), 99.8% at 4 steps.

Negative prompts, batching, explicit `latents` for cross-backend reproducibility, other
resolutions and the components on their own work exactly as on the
[Stable Diffusion](#stable-diffusion-end-to-end-example) page. Image-to-image
(`generate(..., image=..., strength=...)`) works as described for
[`BaseDiffusion`](main_classes.md#basediffusion).

#### Verified against diffusers {#stable-diffusion-2-verified-against-diffusers}

The ramen prompt above with the same initial latent, PNDM 50 steps and guidance 7.5
through `ZMDiffusion` (left) and diffusers' `StableDiffusionPipeline`
(right) on `stable-diffusion-2-1-base`, both fp32:

<img src="../assets/stable_diffusion_2_text_to_image_output.jpg" alt="zeromodels (left) and diffusers (right) generations of a bowl of ramen from stable-diffusion-2-1-base with the same latent" width="760">

```
tokenizer input_ids           identical
text_encoder                  max|d|=1.5e-05
unet noise_pred (t=981)       max|d|=7.2e-07
vae decode                    max|d|=2.6e-06
final image (uint8)           max|d|=1  mean|d|=0.0095  PSNR=68.3 dB  identical_pixels=97.22%
```

The 768px v-prediction checkpoint (`stable-diffusion-2-1`, DDIM 20 steps, same latent)
matches the same way: UNet 8.3e-6, VAE 7.9e-5, final image max 1 uint8 level, 99.3% of
pixels identical, PSNR 74.6 dB.

### Data Format {#stable-diffusion-2-data-format}

Both `channels_last` and `channels_first` are supported, as for
[Stable Diffusion](#stable-diffusion-data-format); `generate` always returns
`(batch, H, W, 3)` uint8.

### Memory and speed {#stable-diffusion-2-memory-and-speed}

The fp32 container is 5.16 GB. At 512px a guided batch-1 run needs a little over 7 GB of
GPU memory in eager Torch (about 1.5x diffusers' eager time); at 768px the self-attention
runs over 9216 tokens, so the guided run needs roughly 12 GB, or the CPU.

### Loading Fine-tuned Weights {#stable-diffusion-2-loading-fine-tuned-weights}

Any repo laid out like the hosted ones (`zm_config.json` declaring
`ZMDiffusion` with `model_type: "stable_diffusion_2"`, the weights, `tokenizer.json`)
loads with `from_weights("<org>/<repo>")`. The `hf:` prefix raises for diffusion models: convert a
diffusers-format SD 2 checkpoint once with
`zeromodels/diffusers/stable_diffusion_2/convert_stable_diffusion_2_diffusers_to_keras.py`
(`transfer_stable_diffusion_2(repo)`, `pip install zeromodels[conversion]`) and host the result.

## Stable Diffusion XL {#stable-diffusion-xl}

Stable Diffusion XL, ported to pure Keras 3: the [Stable Diffusion](#stable-diffusion)
latent-diffusion recipe scaled up to 1024px with a 2.6B-parameter UNet and two text
encoders, plus the **refiner**, an image-to-image expert that finishes the base model's
latents, and **SDXL-Turbo**, the base distilled to a few steps. It loads
through [`ZMDiffusion`](#zmdiffusion) with `model_type="stable_diffusion_xl"` (the
refiner: `"stable_diffusion_xl_refiner"`), a family built on SD 1.x's with a fourth
component and the SDXL conditioning hooks, so everything on that page applies: one
functional model and one set of weights per repo, `generate` from `BaseDiffusion`, the
schedulers, both data formats.
What changes:

- **Two text encoders**: CLIP ViT-L/14's text tower (768-d, the SD 1.x one) and OpenCLIP
  ViT-bigG/14's (1280-d, 32 layers, `gelu`, with a 1280-d `text_projection`). The prompt
  goes through both; their **penultimate** hidden states (no final LayerNorm) are
  concatenated into the UNet's 2048-d cross-attention context, and the second tower's
  projected pooled (EOT) state is the `text_embeds` micro-conditioning.
- **One tokenizer call**: both towers use the same CLIP BPE and differ only in the pad
  token (`<|endoftext|>` vs `!`), so `ZMDiffusionTokenizer` returns the first one's ids
  and the model derives the second tower's ids from the returned `attention_mask`.
- **Micro-conditioning**: the UNet's timestep embedding also receives the pooled text
  embedding and six size / crop values (`original_size`, `crops_coords_top_left`,
  `target_size`, each `(height, width)`), sinusoidally embedded and projected by the
  `add_embedding` MLP (`addition_embed_type="text_time"`).
- **UNet**: three levels (320, 640, 1280) with no attention at the first, (1, 2, 10)
  transformer blocks per Transformer2D, (5, 10, 20) heads (64 wide), linear token
  projection, built at a 128x128 latent (1024px).
- **VAE**: the SDXL autoencoder (`scaling_factor` 0.13025), built in **float32 whatever
  the load dtype** (`force_upcast`: it overflows in float16, as in diffusers).
- **Classifier-free guidance** with no negative prompt uses **zero** embeddings
  (`force_zeros_for_empty_prompt`), not the encoded empty prompt. Default guidance is 5.0.
- **Scheduler**: Euler with `leading` timestep spacing (`steps_offset` 1) for the base
  models and the refiners, ancestral Euler with `trailing` spacing for SDXL-Turbo; all
  come from the repo's `scheduler_config`.
- **Refiner**: a second checkpoint (a `ZMDiffusion` with
  `model_type="stable_diffusion_xl_refiner"`) with the OpenCLIP tower alone (1280-d
  context), a four-level UNet ((384, 768, 1536, 1536), attention at the middle levels, 4
  blocks per Transformer2D, 2.3B parameters) and an **aesthetic score** as the fifth
  micro-conditioning id. It refines a latent the base left partially denoised
  (`denoising_start`) or any image (`strength`); its empty negative prompt is encoded, not
  zeroed.
- **float16 weights**: SDXL was released in float16, so the repos store that (the VAE in
  float32) and `from_weights` builds the model in float16 by default (6.6 GB of weights).
  Pass `load_dtype="float32"` for a float32 model (13.9 GB).

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion is
deliberately **not supported** for diffusion models.

Links:

- Paper: [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis (arXiv:2307.01952)](https://arxiv.org/abs/2307.01952)
- Reference implementation: [diffusers `StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)
- Licenses: [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
  (1.0 base and refiner), [Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md)
  (Turbo); the 0.9 research previews are not hosted

See also [Stable Diffusion 1.x](#stable-diffusion), [Stable Diffusion 2.x](#stable-diffusion-2), [clip.md](clip.md).

### Variants {#stable-diffusion-xl-variants}

Preconverted weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. A base repo is one container: UNet (2.57B) + VAE
(84M) + CLIP ViT-L/14 text encoder (123M) + OpenCLIP ViT-bigG/14 text encoder (695M),
3.47B parameters, 6.6 GB in float16 (two `model.weights.json` shards); a refiner repo is
UNet (2.26B) + VAE + OpenCLIP text encoder, 3.04B parameters, 5.8 GB.

| Variant | Hub | `model_type` | Resolution | Sampler | License |
|---|---|---|---|---|---|
| `stable-diffusion-xl-base-1.0` | [`zeromodels/stable-diffusion-xl-base-1.0`](https://huggingface.co/zeromodels/stable-diffusion-xl-base-1.0) | `stable_diffusion_xl` | 1024 | Euler, 50 steps, guidance 5.0 | CreativeML Open RAIL++-M |
| `stable-diffusion-xl-refiner-1.0` | [`zeromodels/stable-diffusion-xl-refiner-1.0`](https://huggingface.co/zeromodels/stable-diffusion-xl-refiner-1.0) | `stable_diffusion_xl_refiner` | 1024 | Euler, 50 steps, guidance 5.0, strength 0.3 | CreativeML Open RAIL++-M |
| `sdxl-turbo` | [`zeromodels/sdxl-turbo`](https://huggingface.co/zeromodels/sdxl-turbo) | `stable_diffusion_xl` | 512 | Euler ancestral, 1 step, no guidance | Stability AI Non-Commercial Research Community |

The SDXL 0.9 research previews (`stabilityai/stable-diffusion-xl-base-0.9` and
`-refiner-0.9`, the same architecture with different weights, gated behind the research
license) are not hosted; with an authorized token the converter turns them into the same
checkpoints (`transfer_stable_diffusion_xl(repo, token=...)`, see
[Loading Fine-tuned Weights](#stable-diffusion-xl-loading-fine-tuned-weights)). SDXL-Turbo is a
research-only, non-commercial release; its repo defaults (`generate_args`) are 1 step and
`guidance_scale=0.0`, which is how it was trained (do not add guidance).

### API {#stable-diffusion-xl-api}

Configs are typed: `StableDiffusionXLConfig` (composite, `model_type`
`"stable_diffusion_xl"`) over `StableDiffusionXLUNetConfig`, `StableDiffusionXLVAEConfig`,
`StableDiffusionTextConfig` (the CLIP ViT-L/14 tower, as SD 1.x) and
`StableDiffusionXLTextConfig2` (the OpenCLIP ViT-bigG/14 tower, with `projection_dim` and
its own `hidden_act`), plus the scheduler config, the token ids (`pad_token_id_2` for the
second tokenizer's `!`), `force_zeros_for_empty_prompt` and `requires_aesthetics_score`.
Flat constructor with `unet_` / `vae_` / `text_` / `text_2_` prefixes.
`StableDiffusionXLRefinerConfig` (`model_type` `"stable_diffusion_xl_refiner"`) is the
same with `text_config=None`, the `StableDiffusionXLRefinerUNetConfig` denoiser,
`requires_aesthetics_score=True` and `force_zeros_for_empty_prompt=False`.

#### `ZMDiffusion` {#stable-diffusion-xl-zmdiffusion}

`ZMDiffusion` with `model_type="stable_diffusion_xl"` (read from the repo); `generate`
gains the micro-conditioning arguments:

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
    original_size=None,
    crops_coords_top_left=(0, 0),
    target_size=None,
    aesthetic_score=6.0,
    negative_original_size=None,
    negative_crops_coords_top_left=None,
    negative_target_size=None,
    negative_aesthetic_score=2.5,
)
```

| Argument | Description |
|---|---|
| `input_ids`, `attention_mask` | `**tokenizer(prompts)`; the mask tells the model which positions are padding, so the second tower's `!`-padded ids can be derived (without it, everything after the first `<|endoftext|>` counts as padding). |
| `negative_input_ids` | Tokenized negative prompt, one row per prompt or a single row shared by the batch. Without it the unconditional branch is zero embeddings (`force_zeros_for_empty_prompt`), as in diffusers. |
| `num_inference_steps`, `guidance_scale`, `seed`, `latents` | As for [Stable Diffusion](#stable-diffusion-zmdiffusion); defaults from the repo's `generate_args` (50 steps, guidance 5.0 for the base model). |
| `image`, `strength`, `denoising_start`, `denoising_end`, `output_type` | The image-to-image and ensemble controls of [`BaseDiffusion.generate`](main_classes.md#basediffusion): refine an `image` (noised to `strength`), stop early (`denoising_end`) and hand the latent over (`output_type="latent"`), or resume one (`denoising_start`). |
| `original_size`, `crops_coords_top_left`, `target_size` | SDXL's size / crop conditioning, `(height, width)` in pixels; the sizes default to the generated image's size, the crop to `(0, 0)`. |
| `aesthetic_score` | The refiner's fifth conditioning value (6.0), in place of `target_size`; ignored by the base models. |
| `negative_*` | The same for the negative branch; each defaults to its positive value, except `negative_aesthetic_score` (2.5). |

Returns `(batch, H, W, 3)` uint8 images (or the latent); `latents` is
`(batch, 128, 128, 4)` at 1024px (`channels_first`: channels second).

The hooks, for anyone composing their own loop: `encode_prompt(input_ids, attention_mask,
original_size, crops_coords_top_left, target_size)` returns the dict the UNet takes
(`encoder_hidden_states` (batch, 77, 2048), `text_embeds` (batch, 1280), `time_ids`
(batch, 6)); `encode_negative_prompt` zeroes the first two when no negative ids are given;
`predict_noise(latents, timesteps, embeddings)` runs the UNet.

##### The refiner {#stable-diffusion-xl-the-refiner}

The same class with `model_type="stable_diffusion_xl_refiner"` (`text_encoder_2` alone,
`aesthetic_score` in the conditioning, `generate_args` with `strength` 0.3). Its
`generate` is the one above; with no `image` / `latents` it runs text-to-image, which the
refiner was not trained for. The two intended uses:

```python
base = ZMDiffusion.from_weights("zeromodels/stable-diffusion-xl-base-1.0")
refiner = ZMDiffusion.from_weights("zeromodels/stable-diffusion-xl-refiner-1.0")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-xl-base-1.0")
inputs = tokenizer("a lighthouse on a rocky coast at dawn, dramatic clouds, cinematic")

# ensemble of experts: the base denoises 80% of the schedule, the refiner the rest
latent = base.generate(
    **inputs, num_inference_steps=50, denoising_end=0.8, output_type="latent"
)
images = refiner.generate(
    **inputs, latents=latent, num_inference_steps=50, denoising_start=0.8
)

# image-to-image: refine any image (uint8 or [0, 1] float, (batch, H, W, 3))
images = refiner.generate(**inputs, image=images, strength=0.3, seed=0)
```

##### The graph {#stable-diffusion-xl-the-graph}

Four disconnected paths in one functional model, `.unet` / `.vae` /
`.text_encoder` / `.text_encoder_2`.

| Inputs | Outputs |
|---|---|
| `sample` (B, 128, 128, 4), `timestep` (B,), `encoder_hidden_states` (B, 77, 2048), `text_embeds` (B, 1280), `time_ids` (B, 6) | `noise_pred` (B, 128, 128, 4) |
| `image` (B, 1024, 1024, 3), `latent` (B, 128, 128, 4) | `moments` (B, 128, 128, 8), `image` (B, 1024, 1024, 3) |
| `token_ids` (B, 77), `token_ids_2` (B, 77), `padding_mask` (B, 77) | `prompt_embeds` (B, 77, 2048), `pooled_prompt_embeds` (B, 1280) |

Each text tower is a functional CLIP text model (the `clip` family's layers) whose outputs
are `penultimate_hidden_state`, `last_hidden_state`, `pooler_output` and, for the second,
`text_embeds`. The UNet is the SD 1.x `UNet2DConditionModel` with the SDXL config
(`transformer_layers_per_block=(1, 2, 10)`, `addition_embed_type="text_time"`); the VAE
is `AutoencoderKL` with `force_upcast=True`. The refiner is the same
graph without the first tower (`token_ids_2` + `padding_mask` in, `prompt_embeds`
(B, 77, 1280) out) and with `time_ids` (B, 5).

### Preprocessing {#stable-diffusion-xl-preprocessing}

#### `ZMDiffusionTokenizer` {#stable-diffusion-xl-zmdiffusiontokenizer}

`ZMDiffusionTokenizer` with `model_type="stable_diffusion_xl"` (CLIP BPE,
`<|startoftext|>` / `<|endoftext|>` framing,
`<|endoftext|>`-padded to 77 tokens). Returns `{"input_ids", "attention_mask"}`; pass both
to `generate`.

```python
ZMDiffusionTokenizer(
    model_type="stable_diffusion_xl", hf_id=None, tokenizer_file=None, max_seq_len=77
)  # pad_token defaults to "<|endoftext|>"; the refiner repos use the same settings
```

A literal `!` in a prompt is encoded as OpenCLIP encodes it (the BPE token `!</w>`, id
256) in the second tower's derived ids; diffusers' `tokenizer_2`, whose pad token is `!`,
turns it into the pad id 0 instead.

### End-to-end example {#stable-diffusion-xl-end-to-end-example}

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

model = ZMDiffusion.from_weights("zeromodels/stable-diffusion-xl-base-1.0")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-xl-base-1.0")

inputs = tokenizer(
    "a lighthouse on a rocky coast at dawn, dramatic clouds, cinematic, highly detailed"
)
images = model.generate(**inputs, num_inference_steps=30, guidance_scale=5.0, seed=3)

Image.fromarray(images[0]).save("lighthouse.png")  # (1024, 1024, 3) uint8
```

<img src="../assets/stable_diffusion_xl_lighthouse.jpg" alt="Stable Diffusion XL base 1.0: a lighthouse on a rocky coast at dawn, 1024px" width="380">

#### Base + refiner {#stable-diffusion-xl-base-refiner}

The SDXL "ensemble of experts": the base runs the first 80% of the schedule and hands its
latent to the refiner, which finishes it. Same prompt and initial latent as above:

```python
refiner = ZMDiffusion.from_weights("zeromodels/stable-diffusion-xl-refiner-1.0")
latent = model.generate(
    **inputs, num_inference_steps=30, denoising_end=0.8, output_type="latent", seed=3
)
images = refiner.generate(
    **inputs, latents=latent, num_inference_steps=30, denoising_start=0.8
)
```

<img src="../assets/stable_diffusion_xl_refiner_lighthouse.jpg" alt="Stable Diffusion XL base + refiner: the lighthouse, refined" width="380">

#### SDXL-Turbo {#stable-diffusion-xl-sdxl-turbo}

One step, no guidance, 512px (the repo's defaults):

```python
model = ZMDiffusion.from_weights("zeromodels/sdxl-turbo")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/sdxl-turbo")
images = model.generate(
    **tokenizer(
        "a cinematic shot of a baby raccoon wearing an intricate italian priest robe"
    )
)
```

<img src="../assets/stable_diffusion_xl_turbo_raccoon.jpg" alt="SDXL-Turbo, one step: a baby raccoon in an italian priest robe, 512px" width="380">

#### Negative prompts, micro-conditioning {#stable-diffusion-xl-negative-prompts-micro-conditioning}

```python
inputs = tokenizer(["a red bicycle", "a blue bicycle"])
negative = tokenizer("blurry, low quality")  # one row, shared by the batch
images = model.generate(
    **inputs,
    negative_input_ids=negative["input_ids"],
    original_size=(2048, 2048),  # "a crop of a larger image": sharper detail
    crops_coords_top_left=(0, 0),
    target_size=(1024, 1024),
    seed=0,
)
```

#### Other resolutions {#stable-diffusion-xl-other-resolutions}

The graphs are built for the repo's resolution; pass overrides to build for another
(the weights are resolution-independent; multiples of 64px):

```python
model = ZMDiffusion.from_weights(
    "zeromodels/stable-diffusion-xl-base-1.0",
    unet_sample_size=(96, 128),
    vae_sample_size=(768, 1024),
)
images = model.generate(**tokenizer("a wide mountain valley"))  # (1, 768, 1024, 3)
```

#### float32 {#stable-diffusion-xl-float32}

```python
model = ZMDiffusion.from_weights(
    "zeromodels/stable-diffusion-xl-base-1.0", load_dtype="float32"
)
```

#### Verified against diffusers {#stable-diffusion-xl-verified-against-diffusers}

The components of `stable-diffusion-xl-base-1.0` and `stable-diffusion-xl-refiner-1.0`
were checked one by one against diffusers 0.39 / transformers in float32 on the same
inputs (the converted float16 weights computed in float32):

```
base     text_encoder penultimate      max|d|=6.1e-05   (values up to 854)
base     text_encoder_2 penultimate    max|d|=3.1e-05
base     text_encoder_2 pooled         max|d|=3.6e-06
base     unet noise_pred (t=981)       max|d|=9.3e-06
base     vae decode (512px)            max|d|=3.5e-05
refiner  text_encoder_2 penultimate    max|d|=3.1e-05
refiner  unet noise_pred (t=181)       max|d|=1.2e-05
```

The pipelines (Euler `leading` spacing, classifier-free guidance with the zeroed or
encoded negative branch, the micro-conditioning values, negative prompts, batches,
`strength`, `denoising_end` / `denoising_start`) were checked against
`StableDiffusionXLPipeline` and `StableDiffusionXLImg2ImgPipeline` on small
random-weight models, where the generated images are pixel-identical on torch and jax.
At full size in float16 (the default load; diffusers in float16 with its float32 VAE),
the lighthouse above matches diffusers to PSNR 49.7 dB (98.5% of pixels within 2 uint8
levels), the base + refiner ensemble to PSNR 51.9 dB (98.8% within 2) and SDXL-Turbo
(1 step, 512px) to PSNR 44.9 dB (92% within 2; a single step amplifies float16 rounding).
The lighthouse, zeromodels (left) and diffusers (right):

<img src="../assets/stable_diffusion_xl_text_to_image_output.jpg" alt="zeromodels (left) and diffusers (right) generations of the lighthouse from stable-diffusion-xl-base-1.0 with the same latent" width="760">

### Data Format {#stable-diffusion-xl-data-format}

Both `channels_last` and `channels_first` are supported, as for
[Stable Diffusion](#stable-diffusion-data-format); `generate` always returns
`(batch, H, W, 3)` uint8.

### Memory and speed {#stable-diffusion-xl-memory-and-speed}

Measured on the torch backend with the default float16 load (weights resident: 6.8 GB for
a base model, 6.3 GB for the refiner, the VAE in float32):

| Resolution | Denoising step (guided, batch 1) | VAE decode (float32) |
|---|---|---|
| 512px | peak 7.1 GB, 1.2 s/step on an RTX 4060 Laptop | peak 8.5 GB |
| 768px | peak 8.0 GB | peak 10.5 GB |
| 1024px | peak 10.4 GB | about 14 GB |

Keras' functional executor keeps every layer output of a graph alive until the graph
returns, so the decoder's feature maps at 1024px add several GB on top of the weights: a
16 GB GPU runs 1024px comfortably, a 12 GB GPU 768px; on 8 GB, generate at 512px, or
denoise on the GPU (`output_type="latent"`) and decode the latent on the CPU (about 100 s
at 1024px) with a second, CPU-only process. In float32 the weights alone take 13.9 GB.
The UNet's attention runs the portable `"sdpa"` math by default, which materializes the
4096 x 4096 float32 logits of the 64 x 64 level at 1024px; loading with
`attn_implementation="fused"` (the backend's fused kernel, see
[`fused_attention`](main_classes.md#fused_attention)) cuts the 1024px step from a 10.3 GB
peak to 7.6 GB on the same GPU. On JAX and TensorFlow the denoiser step is compiled once
per run (see [`BaseDiffusion`](main_classes.md#basediffusion)).

### Loading Fine-tuned Weights {#stable-diffusion-xl-loading-fine-tuned-weights}

Any repo laid out like the hosted ones (`zm_config.json` declaring
`ZMDiffusion` with `model_type: "stable_diffusion_xl"` or
`"stable_diffusion_xl_refiner"`, the weights, `tokenizer.json`) loads with
`from_weights("<org>/<repo>")`. The `hf:` prefix raises for diffusion models: convert a
diffusers-format SDXL checkpoint once with
`zeromodels/diffusers/stable_diffusion_xl/convert_stable_diffusion_xl_diffusers_to_keras.py`
(`transfer_stable_diffusion_xl(repo)`, `pip install zeromodels[conversion]`) and host the result.

## Stable Diffusion 3 {#stable-diffusion-3}

Stable Diffusion 3, ported to pure Keras 3: a rectified-flow model whose denoiser is the
**MMDiT**, a multimodal diffusion transformer in which the latent tokens and the text tokens
attend to each other jointly, conditioned on the timestep and the pooled text embeddings
through adaptive layer norms. It loads through [`ZMDiffusion`](#zmdiffusion) with
`model_type="stable_diffusion_3"` and keeps the SD family's shape (one functional model
and one set of weights per repo, `generate` from `BaseDiffusion`, the schedulers, both
data formats) with new parts:

- **MMDiT** (`SD3Transformer2DModel`): the 16-channel latent is patchified (2x2) into 1536-d
  tokens with a sinusoidal position table cropped from a 192x192 grid; 24 joint blocks of
  24 heads, each with AdaLN-Zero modulation of both streams, joint attention over the
  concatenated latent + text tokens, and gated GELU feed-forwards; a final adaptive norm
  and projection unpatchify the velocity. 2B parameters.
- **Three text encoders**: CLIP ViT-L/14 (768-d) and OpenCLIP ViT-bigG/14 (1280-d), each
  with a projection, and **T5-XXL** (4.7B, 4096-d, 256 tokens). The two CLIP penultimate
  states are concatenated (2048-d), zero-padded to 4096 and followed along the sequence by
  the T5 states (77 + 256 tokens); the projected pooled CLIP states (2048-d) feed the
  conditioning embedding.
- **T5 is optional and separate**: the checkpoint holds the MMDiT, the VAE and the two CLIP
  towers (2.9B, 5.8 GB float16); the T5-XXL encoder is hosted once for every SD 3 / 3.5
  checkpoint (`zeromodels/t5-v1_1-xxl-encoder`, an
  [`SD3T5EncoderModel`](#stable-diffusion-3-sd3t5encodermodel), 9.5 GB float16) and attached with
  `text_encoder_3=`. Without it the T5 features are zeros, SD 3's documented
  memory-saving mode (prompt adherence drops, the images stay good).
- **VAE**: 16 latent channels, no quant convolutions, `scaling_factor` 1.5305 and
  `shift_factor` 0.0609 (`z = (x - shift) * scale`), built in float32 (`force_upcast`).
- **Sampler**: `FlowMatchEulerDiscreteScheduler` with `shift` 3.0 (rectified flow: the
  timesteps are `sigma * 1000`, a step is `x + (sigma_next - sigma) * v`); 28 steps at
  guidance 7.0 by default. No negative prompt means the encoded empty prompt.
- **One tokenizer call**: `ZMDiffusionTokenizer` runs the CLIP BPE (the second
  tower's `!`-padded ids are derived from the mask, as SDXL) and the T5 SentencePiece
  tokenizer, returning `{"input_ids", "attention_mask", "input_ids_3"}`.
- **float16 weights**, the release precision; `from_weights` builds the model in float16
  by default (`load_dtype="float32"` for float32).

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion is
deliberately **not supported** for diffusion models. The upstream repo is gated behind the
Stability AI Non-Commercial Research Community License.

Links:

- Paper: [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- Reference implementation: [diffusers `StableDiffusion3Pipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)
- License: [Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/blob/main/LICENSE.md)

See also [Stable Diffusion 3.5](#stable-diffusion-3-5), [Stable Diffusion XL](#stable-diffusion-xl), [clip.md](clip.md).

### Variants {#stable-diffusion-3-variants}

| Variant | Hub | Resolution | Sampler | License |
|---|---|---|---|---|
| `stable-diffusion-3-medium` | [`zeromodels/stable-diffusion-3-medium`](https://huggingface.co/zeromodels/stable-diffusion-3-medium) | 1024 | flow-match Euler (shift 3), 28 steps, guidance 7.0 | Stability AI Non-Commercial Research Community |
| `t5-v1_1-xxl-encoder` | [`zeromodels/t5-v1_1-xxl-encoder`](https://huggingface.co/zeromodels/t5-v1_1-xxl-encoder) | text encoder (shared with SD 3.5) | | Apache 2.0 (Google T5 v1.1) |

### API {#stable-diffusion-3-api}

Configs are typed: `StableDiffusion3Config` (composite, `model_type` `"stable_diffusion_3"`)
over `StableDiffusion3TransformerConfig`, `StableDiffusion3VAEConfig`,
`StableDiffusion3TextConfig` (CLIP ViT-L/14 with its 768-d projection) and
`StableDiffusionXLTextConfig2` (OpenCLIP ViT-bigG/14 with its 1280-d projection), plus
the scheduler config, the CLIP and T5 token ids and `max_sequence_length` (256). Flat
constructor with `transformer_` / `vae_` / `text_` / `text_2_` prefixes.

#### `ZMDiffusion` {#stable-diffusion-3-zmdiffusion}

`ZMDiffusion` with `model_type="stable_diffusion_3"` (read from the repo); `generate`
over the SD 3 graph:

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
    input_ids_3=None,
    negative_input_ids_3=None,
)
```

| Argument | Description |
|---|---|
| `input_ids`, `attention_mask`, `input_ids_3` | `**tokenizer(prompts)`: the CLIP ids and mask (the second tower's ids are derived from the mask) and the T5 ids (`(batch, 256)`). Without `input_ids_3`, or without an attached T5, the T5 features are zeros. |
| `negative_input_ids`, `negative_input_ids_3` | The tokenized negative prompt (`neg = tokenizer("..."); negative_input_ids=neg["input_ids"], negative_input_ids_3=neg["input_ids_3"]`); the empty prompt when omitted. |
| `num_inference_steps`, `guidance_scale`, `seed`, `latents`, `image`, `strength`, ... | As for [`BaseDiffusion.generate`](main_classes.md#basediffusion); defaults from the repo's `generate_args` (28 steps, guidance 7.0). `latents` is `(batch, 128, 128, 16)` at 1024px. |

Returns `(batch, H, W, 3)` uint8 images.

`from_weights` takes one extra argument, `text_encoder_3`: a hosted `SD3T5EncoderModel`
repo id (loaded at the same `load_dtype`) or a built `SD3T5EncoderModel`;
`sd.text_encoder_3` can also be assigned later (or set to `None`). It lives outside the
checkpoint's weights; passing `text_encoder_3` for a family other than SD 3 / 3.5 raises.

The graph is four disconnected paths in one functional model, `.transformer` / `.vae` /
`.text_encoder` / `.text_encoder_2`.

| Inputs | Outputs |
|---|---|
| `sample` (B, 128, 128, 16), `timestep` (B,), `encoder_hidden_states` (B, 333, 4096), `pooled_projections` (B, 2048) | `noise_pred` (B, 128, 128, 16) |
| `image` (B, 1024, 1024, 3), `latent` (B, 128, 128, 16) | `moments` (B, 128, 128, 32), `image` (B, 1024, 1024, 3) |
| `token_ids` (B, 77), `token_ids_2` (B, 77), `padding_mask` (B, 77) | `prompt_embeds` (B, 77, 2048), `pooled_prompt_embeds` (B, 2048) |

#### `SD3Transformer2DModel` {#stable-diffusion-3-sd3transformer2dmodel}

The MMDiT on its own (`{"sample", "timestep", "encoder_hidden_states",
"pooled_projections"} -> {"sample"}`), built from `StableDiffusion3TransformerConfig`
(`sample_size` 128, `patch_size` 2, `num_layers` 24, `num_attention_heads` 24 x
`attention_head_dim` 64, `joint_attention_dim` 4096, `caption_projection_dim` 1536,
`pooled_projection_dim` 2048, `pos_embed_max_size` 192, `qk_norm`,
`dual_attention_layers`). Its layers (`StableDiffusion3PatchEmbed`,
`StableDiffusion3AdaLayerNorm`, `StableDiffusion3JointAttention`,
`StableDiffusion3JointTransformerBlock`, `StableDiffusion3GELUFeedForward`) are named
with the diffusers module paths.

The joint attention runs over 4429 tokens at 1024px, so `StableDiffusion3JointAttention`
defaults to the `"fused"` attention implementation (`keras.ops.dot_product_attention`,
the backend's own kernel: torch's flash / memory-efficient kernels never materialize the
4429 x 4429 logits; see [`fused_attention`](main_classes.md#fused_attention)) instead of
the library-wide `"sdpa"` math. `from_weights(..., attn_implementation="sdpa")` switches
it back.

#### `SD3T5EncoderModel` {#stable-diffusion-3-sd3t5encodermodel}

The third text encoder, the encoder half of T5 v1.1 XXL (transformers `T5EncoderModel` on
`google/t5-v1_1-xxl`), as its own model in this family: `{"input_ids", "attention_mask"}`
(`(batch, 256)` int32) `-> {"last_hidden_state": (batch, 256, 4096)}`, built from
`StableDiffusion3T5EncoderConfig` (`vocab_size` 32128, `embed_dim` 4096, `key_value_dim`
64, `mlp_dim` 10240, `num_layers` 24, `num_heads` 64, `relative_attention_num_buckets` 32,
`relative_attention_max_distance` 128, `layer_norm_eps` 1e-6; 4.7B parameters). The blocks
reuse the [T5](t5.md) family's attention, RMSNorm and relative position bias, and add the
gated-GELU feed-forward of T5 v1.1 (`wo(gelu_tanh(wi_0(x)) * wi_1(x))`,
`StableDiffusion3T5GatedFeedForward`), which the original T5 does not have. Loads on its own with
`SD3T5EncoderModel.from_weights("zeromodels/t5-v1_1-xxl-encoder")` (`load_dtype`,
`quantization="int8"` as needed); its inputs are the `input_ids_3` of the SD 3 tokenizer.

### Preprocessing {#stable-diffusion-3-preprocessing}

#### `ZMDiffusionTokenizer` {#stable-diffusion-3-zmdiffusiontokenizer}

```python
ZMDiffusionTokenizer(
    model_type="stable_diffusion_3",
    hf_id=None,
    tokenizer_file=None,
    tokenizer_file_3=None,
    max_seq_len=77,
    max_sequence_length=256,
)
```

Returns `{"input_ids", "attention_mask", "input_ids_3"}`; pass all three to `generate`.

### End-to-end example {#stable-diffusion-3-end-to-end-example}

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

model = ZMDiffusion.from_weights(
    "zeromodels/stable-diffusion-3-medium",
    text_encoder_3="zeromodels/t5-v1_1-xxl-encoder",  # omit on small machines: T5 features zeroed
)
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-3-medium")

inputs = tokenizer(
    "a fluffy corgi wearing round sunglasses, sitting on a surfboard at a sunny beach, photo"
)
images = model.generate(**inputs, num_inference_steps=28, guidance_scale=7.0, seed=4)

Image.fromarray(images[0]).save("corgi.png")  # (1024, 1024, 3) uint8
```

<img src="../assets/stable_diffusion_3_corgi.jpg" alt="Stable Diffusion 3 medium: a corgi with sunglasses on a surfboard, 1024px" width="380">

#### The T5 encoder on the CPU {#stable-diffusion-3-the-t5-encoder-on-the-cpu}

The 9.5 GB T5-XXL rarely fits next to the MMDiT on a consumer GPU. Encode the prompt
separately (an `SD3T5EncoderModel` in another process, or int8) and hand the ids to a
model without T5 for the CLIP part only, or attach it in full on a large machine. The
`encode_prompt` hook returns the conditioning dict the transformer takes, so any
precomputed T5 features can be substituted there.

#### Verified against diffusers {#stable-diffusion-3-verified-against-diffusers}

The components of `stable-diffusion-3-medium` were checked one by one against diffusers
0.39 / transformers 5 in float32 on the same inputs (the converted float16 weights
computed in float32):

```
text_encoder penultimate       max|d|=6.1e-05   (values up to 854)
text_encoder_2 penultimate     max|d|=3.1e-05
pooled (L + G, projected)      max|d|=3.8e-06
t5-v1_1-xxl encoder (256 tok)  max|d|=1.1e-04   (values up to 6.3)
mmdit velocity (t=1000)        max|d|=9.9e-06
vae decode (512px)             max|d|=6.3e-05
vae encode                     max|d|=1.4e-05
```

The pipeline (flow-match Euler with shift 3, classifier-free guidance with the encoded
empty prompt, the three tokenizers, batched negative prompts, the no-T5 mode) was checked
against `StableDiffusion3Pipeline` on a small random-weight SD 3, where the generated images
are pixel-identical on torch and jax. End to end, `stable-diffusion-3-medium` without the
T5 encoder (28 steps, guidance 7, the same initial latent) reproduces the diffusers
pipeline in float32 at 512px: 92.8% identical pixels, max |d| 2 / 255, PSNR 63.9 dB (the
final latent differs by at most 2.5e-2 with values up to 5.1). In float16 on the GPU the
two implementations round differently through the 28 guided steps: the pictures agree in
composition but not pixel for pixel (PSNR 26 dB at 1024px, 29 dB at 768px, float16 MMDiT
on both sides), the usual float16 sampling noise.

### Data Format {#stable-diffusion-3-data-format}

Both `channels_last` and `channels_first` are supported; `generate` always returns
`(batch, H, W, 3)` uint8.

### Memory and speed {#stable-diffusion-3-memory-and-speed}

The checkpoint is 5.8 GB in float16 (the T5 encoder adds 9.5 GB). Measured on the torch
backend with the default float16 load (RTX 4060 Laptop, 8 GB), T5 features precomputed:

| Resolution | Denoising step (guided, batch 1) | VAE decode (float32) |
|---|---|---|
| 512px | peak 6.1 GB, 0.5 s/step | peak 7.6 GB, 4 s |
| 768px | peak 6.4 GB, 0.8 s/step | peak 9.5 GB |
| 1024px | peak 6.9 GB, 1.0 s/step | peak 12.4 GB |

The MMDiT itself fits an 8 GB GPU at the native 1024px thanks to the fused attention
kernel (the `"sdpa"` math needs 3.8 GB per block for the 4429 x 4429 float32 logits and
runs out of memory). The VAE decoder is the SDXL one (16 latent channels aside) and Keras'
functional executor keeps its feature maps alive, see the
[SDXL notes](#stable-diffusion-xl-memory-and-speed): on 8 GB, denoise on the GPU
(`output_type="latent"`) and decode on the CPU (75 to 90 s at 1024px). The T5-XXL encoder
encodes a prompt and its negative on the CPU in about 10 to 25 s (float32 compute over
the float16 weights); a 24 GB GPU holds everything. On JAX and TensorFlow the denoiser
step is compiled once per run.

### Loading Fine-tuned Weights {#stable-diffusion-3-loading-fine-tuned-weights}

Any repo laid out like the hosted ones (`zm_config.json` declaring
`ZMDiffusion` with `model_type: "stable_diffusion_3"`, the weights, the two tokenizer
files) loads with
`from_weights("<org>/<repo>")`. The `hf:` prefix raises for diffusion models: convert a
diffusers-format SD 3 checkpoint once with
`zeromodels/diffusers/stable_diffusion_3/convert_stable_diffusion_3_diffusers_to_keras.py`
(`transfer_stable_diffusion_3(repo)`; `transfer_t5_encoder(repo)` for the T5 tower) and host the
result.

## Stable Diffusion 3.5 {#stable-diffusion-3-5}

Stable Diffusion 3.5, ported to pure Keras 3: the [Stable Diffusion 3](#stable-diffusion-3)
architecture, retrained and scaled. It loads through
[`ZMDiffusion`](#zmdiffusion) with `model_type="stable_diffusion_3_5"`, the SD 3 family
with the SD 3.5 configuration, so everything on that page applies: the MMDiT, the three text encoders with the separately hosted T5-XXL, the
16-channel VAE, the flow-match sampler, one tokenizer call. What changes is the config:

- **QK normalization**: every attention block RMS-normalizes its queries and keys per head
  (`qk_norm="rms_norm"`), the training stabilizer of SD 3.5.
- **Large** (8B): 38 blocks of 38 heads (2432-d tokens), 28 steps at guidance 3.5.
- **Large Turbo**: the large model distilled with Adversarial Diffusion Distillation to 4
  steps without guidance (`guidance_scale=0.0`).
- **Medium** (2.5B, MMDiT-X): 24 blocks of 24 heads with **dual attention** in blocks 0 to
  12 (a second, latent-only attention with its own modulation next to the joint one) and a
  384x384 position grid (up to 2 MP); 40 steps at guidance 4.5.

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion is
deliberately **not supported** for diffusion models. The upstream repos are gated behind
the Stability AI Community License.

Links:

- Paper: [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- Reference implementation: [diffusers `StableDiffusion3Pipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)
- License: [Stability AI Community License](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md)

See also [Stable Diffusion 3](#stable-diffusion-3).

### Variants {#stable-diffusion-3-5-variants}

Each repo is one checkpoint: MMDiT + VAE (84M) + CLIP ViT-L/14 (123M) + OpenCLIP
ViT-bigG/14 (695M); the T5-XXL encoder (`zeromodels/t5-v1_1-xxl-encoder`, the SD 3
family's `SD3T5EncoderModel`, 9.5 GB) is shared and optional.

| Variant | Hub | MMDiT | Sampler | Size |
|---|---|---|---|---|
| `stable-diffusion-3.5-large` | [`zeromodels/stable-diffusion-3.5-large`](https://huggingface.co/zeromodels/stable-diffusion-3.5-large) | 38 x 38 heads, 8B | flow-match Euler (shift 3), 28 steps, guidance 3.5 | 17.8 GB float16 |
| `stable-diffusion-3.5-large-turbo` | [`zeromodels/stable-diffusion-3.5-large-turbo`](https://huggingface.co/zeromodels/stable-diffusion-3.5-large-turbo) | 38 x 38 heads, 8B | 4 steps, no guidance | 17.8 GB float16 |
| `stable-diffusion-3.5-medium` | [`zeromodels/stable-diffusion-3.5-medium`](https://huggingface.co/zeromodels/stable-diffusion-3.5-medium) | 24 x 24 heads + dual attention, 2.5B | flow-match Euler (shift 3), 40 steps, guidance 4.5 | 6.8 GB float16 |

### API {#stable-diffusion-3-5-api}

`StableDiffusion3_5Config` (`model_type` `"stable_diffusion_3_5"`) is
`StableDiffusion3Config` over `StableDiffusion3_5TransformerConfig` (the SD 3.5 large
defaults: `num_layers` 38, `num_attention_heads` 38, `caption_projection_dim` 2432,
`qk_norm="rms_norm"`; medium's repo carries `dual_attention_layers=(0, ..., 12)` and
`pos_embed_max_size=384`). `ZMDiffusion.generate` and
`ZMDiffusionTokenizer` (`model_type="stable_diffusion_3_5"`) behave as for SD 3; see
[Stable Diffusion 3](#stable-diffusion-3-api).

### End-to-end example {#stable-diffusion-3-5-end-to-end-example}

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

model = ZMDiffusion.from_weights(
    "zeromodels/stable-diffusion-3.5-medium",
    text_encoder_3="zeromodels/t5-v1_1-xxl-encoder",  # optional
)
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-3.5-medium")

inputs = tokenizer(
    "a whimsical treehouse village at twilight, lanterns glowing, watercolor illustration"
)
images = model.generate(**inputs, num_inference_steps=40, guidance_scale=4.5, seed=5)

Image.fromarray(images[0]).save("treehouse.png")  # (1024, 1024, 3) uint8
```

<img src="../assets/stable_diffusion_3_5_treehouse.jpg" alt="Stable Diffusion 3.5 medium: a treehouse village at twilight, 1024px" width="380">

#### Large Turbo {#stable-diffusion-3-5-large-turbo}

```python
model = ZMDiffusion.from_weights("zeromodels/stable-diffusion-3.5-large-turbo")
images = model.generate(
    **tokenizer("a red panda reading a book")
)  # 4 steps, guidance 0 (the repo defaults)
```

#### Verified against diffusers {#stable-diffusion-3-5-verified-against-diffusers}

The components of `stable-diffusion-3.5-medium` (the MMDiT-X with dual attention and RMS
qk norms) were checked against diffusers 0.39 in float32 on the same inputs (the converted
float16 weights computed in float32): MMDiT velocity max|d| 1.6e-5 (values up to 5.2), the
text encoders and the VAE as for [SD 3](#stable-diffusion-3-verified-against-diffusers).
The SD 3.5 block variants (dual attention, `qk_norm`) are also covered by the small
random-weight pipeline test, pixel-identical to `StableDiffusion3Pipeline` on torch and
jax, and by the SD 3 float32 end-to-end check (92.8% identical pixels, PSNR 63.9 dB at
512px), which the two families share down to the sampler. The picture above is the
1024px float16 output of `stable-diffusion-3.5-medium` with the T5 encoder.

### Memory and speed {#stable-diffusion-3-5-memory-and-speed}

The medium checkpoint is 6.8 GB in float16, the large ones 17.8 GB (a 24 GB GPU); the
T5-XXL encoder adds 9.5 GB. See [Stable Diffusion 3](#stable-diffusion-3-memory-and-speed):
the medium MMDiT-X at 1024px peaks at 7.5 GB during the guided denoising step on the torch
backend (fused attention, float16), which an 8 GB GPU runs with the VAE decode on the CPU.

### Loading Fine-tuned Weights {#stable-diffusion-3-5-loading-fine-tuned-weights}

Any repo laid out like the hosted ones (`zm_config.json` declaring
`ZMDiffusion` with `model_type: "stable_diffusion_3_5"`, the weights, the two tokenizer
files) loads with
`from_weights("<org>/<repo>")`. Convert a diffusers-format checkpoint once with
`zeromodels/diffusers/stable_diffusion_3_5/convert_stable_diffusion_3_5_diffusers_to_keras.py`
and host the result.

## Qwen-Image {#qwen-image}

Qwen-Image, ported to pure Keras 3: latent text-to-image flow-matching with a
60-layer double-stream MMDiT denoiser, a Wan-derived KL autoencoder and the
Qwen2.5-VL-7B Instruct text tower. It loads through [`ZMDiffusion`](#zmdiffusion), the one diffusion
pipeline class of zeromodels, with `model_type="qwen_image"`: a single functional graph with the three towers as disconnected
sub-graphs (the way `CLIPModel` holds its two), so a hosted repo is one set of
sharded weights and one `zm_config.json`. plus
`generate`: the true-CFG denoising loop of `BaseDiffusion`, the
diffusion counterpart of `BaseGeneration`.

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion
is deliberately **not supported** for diffusion models.

Key facts of the port:

- **Packed latents**: the denoiser sees `(B, H/2 · W/2, 64)` tokens; pack / unpack
  match Diffusers' 2×2 patching around the 16-channel VAE latent grid
  (`AutoencoderKLQwenImage`, 8× spatial, channels-last NDHWC with `T=1`).
- **True CFG**: when `guidance_scale > 1`, two transformer forwards and Diffusers'
  prediction-norm renormalization (`true_cfg_scale`); a space negative prompt is
  the default when none is passed.
- **bf16-safe dual-stream clip**: Diffusers clips dual-stream activations to the
  fp16 finite range only under float16; the port matches that, so bf16 runs keep
  the ~1e7 magnitudes Qwen-Image produces instead of collapsing to noise.
- **Schedulers match Diffusers**: `FlowMatchEulerDiscreteScheduler` with dynamic
  resolution shifting (`mu` from packed sequence length) so timesteps agree with
  the reference within float rounding.
- **Verified against Diffusers**: transformer, text encoder, VAE and scheduler
  outputs agree with the reference within float noise (see
  [Verified against Diffusers](#qwen-image-verified-against-diffusers)).

Links:

- Paper: [Qwen-Image Technical Report (arXiv:2508.02324)](https://arxiv.org/abs/2508.02324)
- Reference implementation: [diffusers `QwenImagePipeline`](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
- License: [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image/blob/main/LICENSE)

See also [qwen2_5_vl.md](qwen2_5_vl.md) (the text tower),
[Qwen-Image-2.1](#qwen-image-2-1) (2.1 single-stream / unpatched),
[Stable Diffusion 3](#stable-diffusion-3) (flow-match / MMDiT-style diffusion
in ZeroModels).

### Variants {#qwen-image-variants}

Preconverted, bfloat16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. Each repo is one checkpoint: MMDiT
transformer + VAE + Qwen2.5-VL text tower, ~53 GiB at 16-bit. Released under
Apache-2.0.

| Variant | Hub | Source |
|---|---|---|
| `qwen-image` | [`zeromodels/qwen-image`](https://huggingface.co/zeromodels/qwen-image) | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |

Default `generate_args`: 50 flow-match steps, `guidance_scale=4.0`, 1024×1024.

### API {#qwen-image-api}

Configs are typed: `QwenImageConfig` (composite, `model_type` `"qwen_image"`) over
`QwenImageTransformerConfig`, `QwenImageVAEConfig` and `QwenImageTextConfig`, plus
the checkpoint's `scheduler_config` and the Qwen special token ids. Each repo's
`zm_config.json` parses through it; the constructor stays flat, with the
sub-config fields prefixed `transformer_` / `vae_` / `text_`.

#### `ZMDiffusion` {#qwen-image-zmdiffusion}

The pipeline class of every diffusion family (see [Diffusers](#zmdiffusion)); the
repo's `model_type` (`"qwen_image"` here) selects this family, which supplies the hooks
`BaseDiffusion`'s `generate` needs the mixin needs (`encode_prompt` on the text
tower with ChatML template drop, `unconditional_ids`, `predict_noise` on the
transformer, `decode_latents` on the VAE with mean/std un-normalization) and a
`scheduler`, built from the config's `scheduler_config`
(`FlowMatchEulerDiscreteScheduler` with dynamic shifting).

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    negative_attention_mask=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
    height=None,
    width=None,
    output_type="image",
    **conditioning,
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | ChatML-templated token ids, `**tokenizer(prompts)` |
| `attention_mask` | `None` | padding mask from the tokenizer |
| `negative_input_ids` | `None` | tokenized negative prompt for true CFG; a templated space prompt when unset |
| `negative_attention_mask` | `None` | mask for the negative ids |
| `negative_prompt` | `None` | optional text (or one string per image) passed via `**conditioning`; tokenized when negative ids are unset |
| `num_inference_steps` | `None` | scheduler steps; the repo's `generate_args` (50) when unset |
| `guidance_scale` | `None` | true CFG strength; `generate_args` (4.0) when unset, `<= 1` disables it |
| `seed` | `None` | seed for the initial latent, reproducible per backend |
| `latents` | `None` | explicit packed initial latent; identical results across backends |
| `height` / `width` | `None` | output pixel size; `default_sample_size * 8` (1024) when unset; rounded to VAE×pack multiples |
| `output_type` | `"image"` | `"image"` for uint8 RGB, `"latent"` for packed latents |

Returns `(batch, height, width, 3)` uint8 numpy images.

| Constructor arg | Default | Meaning |
|---|---|---|
| `model_type` | required | `"qwen_image"` (read from the repo's `zm_config.json` on `from_weights`) |
| `scheduler` | `None` | a `BaseScheduler`; built from `scheduler_config` when unset |
| `transformer_sample_size` | `128` | latent side the transformer graph is built for (image / 8) |
| `vae_sample_size` | `1024` | image size the VAE graphs are built for |
| `transformer_*` / `vae_*` / `text_*` | Qwen-Image | the flat sub-config fields (see [Configuration](configuration.md)) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | `151643` / `151645` / `151643` | Qwen special tokens |
| `scheduler_config` | `None` | the Diffusers scheduler dict (`_class_name`, shift, dynamic shifting, ...) |
| `prompt_template_encode_start_idx` | `34` | ChatML template tokens dropped after the text encoder |
| `max_sequence_length` | `512` | prompt embed length after the template drop |

The graph is three disconnected paths, one per component. Inputs cover the transformer (`sample`, `timestep`,
`encoder_hidden_states`, `encoder_hidden_states_mask`), the VAE (`image`, `latent`)
and the text tower (`token_ids`, `padding_mask`); outputs include the packed
velocity, VAE moments / sample, and prompt embeds. The components are exposed as
`.transformer`, `.vae` and `.text_encoder` and share the model's weights, so one load
fills all three, and each can be called on its own.

#### `QwenImageTransformer2DModel` {#qwen-image-qwenimagetransformer2dmodel}

The denoiser (Diffusers' `QwenImageTransformer2DModel`): a 60-layer double-stream
MMDiT over **packed** latents and text features, with MS-RoPE
(`axes_dims_rope=(16, 56, 56)`). Inputs
`{"sample": (B, seq, 64), "timestep": (B,), "encoder_hidden_states": (B, text_seq, 3584)}`
(optional `encoder_hidden_states_mask`), output `{"sample": (B, seq, 64)}` packed
velocity.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `2` | latent pack side |
| `in_channels` | `64` | packed token width (`16 × 2 × 2`) |
| `out_channels` | `16` | unpacked latent channels |
| `num_layers` | `60` | dual-stream DiT blocks |
| `attention_head_dim` / `num_attention_heads` | `128` / `24` | head geometry (inner dim 3072) |
| `joint_attention_dim` | `3584` | text feature width from Qwen2.5-VL |
| `axes_dims_rope` | `(16, 56, 56)` | MS-RoPE axis splits |
| `sample_size` | `128` | latent spatial side the graph is built for |
| `text_seq_len` | `512` | static text length after the template drop |

#### `AutoencoderKLQwenImage` {#qwen-image-autoencoderklqwenimage}

The VAE (Diffusers' `AutoencoderKLQwenImage`): `encode(image, sample=False)` returns
the posterior mean (or a sample) of a `(B, H/8, W/8, 16)` latent and
`decode(latent)` the `(B, H, W, 3)` image in `[-1, 1]`. Latent normalisation uses
`latents_mean` / `latents_std` around the denoiser, as `generate` does. One
functional graph with the encoder and decoder as two disconnected paths
(channels-last NDHWC, `T=1` for still images).

| Arg | Default | Meaning |
|---|---|---|
| `base_dim` | `96` | base channel width |
| `z_dim` | `16` | latent channels |
| `dim_mult` | `(1, 2, 4, 4)` | width multipliers per level |
| `num_res_blocks` | `2` | residual blocks per level |
| `temperal_downsample` | `(False, True, True)` | which levels downsample time (unused at `T=1`) |
| `sample_size` | `1024` | image size the graphs are built for |
| `latents_mean` / `latents_std` | Qwen-Image | per-channel latent normalisation |

#### Schedulers {#qwen-image-schedulers}

`zeromodels.base.base_scheduler` has the flow-match sampler with the Diffusers
interface (`set_timesteps`, `step`, `init_noise_sigma`), configured the Diffusers
way:

| Scheduler | Notes |
|---|---|
| `FlowMatchEulerDiscreteScheduler` | Qwen-Image default; `use_dynamic_shifting=True`, exponential time shift |

`get_scheduler(config)` builds the one named by a Diffusers scheduler config dict,
which is what the model does with its `scheduler_config`. At generate time the
task sets timesteps with resolution-dependent `mu` and a linspace sigma schedule,
matching Diffusers.

### Preprocessing {#qwen-image-preprocessing}

#### `ZMDiffusionTokenizer` {#qwen-image-zmdiffusiontokenizer}

The Qwen2 BPE tokenizer with the Diffusers ChatML prompt template wrapped around
each string. Calling it returns `{"input_ids", "attention_mask"}` for a string or
a list of strings. `generate` / `encode_prompt` drop the template prefix
(`prompt_template_encode_start_idx=34`) after the text encoder and keep at most
`max_sequence_length` (512) tokens.

```python
ZMDiffusionTokenizer(
    model_type="qwen_image", hf_id=None, tokenizer_file=None, max_seq_len=1024
)
```

There is no image processor and no processor class: text-to-image takes token ids
in and hands uint8 images out.

### End-to-end example {#qwen-image-end-to-end-example}

#### Single prompt {#qwen-image-single-prompt}

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

Image.fromarray(images[0]).save("coffee.png")  # (1024, 1024, 3) uint8
```

#### Negative prompt and batching {#qwen-image-negative-prompt-and-batching}

Prompts batch through the tokenizer; a negative prompt is tokenized the same way
and passed as `negative_input_ids` (one per prompt, or one broadcast row).

```python
prompts = [
    "a watercolor painting of a lighthouse at sunset",
    "a red vintage car on a coastal road",
]
inputs = tokenizer(prompts)
negative = tokenizer(["blurry, low quality"] * len(prompts))["input_ids"]

images = model.generate(
    **inputs,
    negative_input_ids=negative,
    height=512,
    width=512,
    num_inference_steps=30,
)
for i, image in enumerate(images):
    Image.fromarray(image).save(f"out_{i}.png")
```

#### Reproducible across backends {#qwen-image-reproducible-across-backends}

`seed` reproduces a run on a given backend, but `keras.random` differs between
torch, jax and tensorflow. For an image that is identical everywhere, pass the
initial packed latent yourself. At 1024px the spatial noise before packing is
`(batch, 128, 128, 16)`; after packing it is `(batch, 4096, 64)`:

```python
import numpy as np

from zeromodels.diffusers.qwen_image.qwen_image_model import pack_latents

h, w, channels = 128, 128, 16
noise = np.random.default_rng(0).standard_normal((1, h, w, channels)).astype("float32")
latents = pack_latents(noise, h, w)
images = model.generate(**tokenizer("a bowl of ramen"), latents=latents)
```

#### Verified against Diffusers {#qwen-image-verified-against-diffusers}

Component checks against Diffusers / Transformers with the same weights (fp32
unless noted):

```
scheduler timesteps / mu     within 1e-4 / identical
transformer (fp32, N blocks) relative error ~8e-6
transformer (bf16, N blocks) cosine similarity 0.99998
text encoder (fp32)          relative error ~1.7e-6
VAE encode / decode          matches Diffusers (NCDHW <-> NDHWC transpose)
```

End-to-end image parity depends on dtype and resolution; prefer bf16 (or fp32)
over fp16 — Qwen-Image activations overflow the fp16 range without the Diffusers
clip path.

#### Components on their own {#qwen-image-components-on-their-own}

```python
from zeromodels.diffusers import ZMDiffusion

qi = ZMDiffusion.from_weights("zeromodels/qwen-image")
latent = qi.vae.encode(image)  # (1, H, W, 3) in [-1, 1] -> (1, H/8, W/8, 16)
text = qi.text_encoder({"input_ids": ids, "attention_mask": mask})["last_hidden_state"]
```

#### Other resolutions {#qwen-image-other-resolutions}

The graphs are built for a fixed size, the weights are not. Rebuild at another
size with the constructor overrides (multiples of 16px after VAE×pack; the
checkpoint targets 1024px):

```python
model = ZMDiffusion.from_weights(
    "zeromodels/qwen-image",
    transformer_sample_size=64,
    vae_sample_size=512,
)
images = model.generate(
    **tokenizer("a mountain lake at dawn"), height=512, width=512
)  # (1, 512, 512, 3)
```

### Data Format {#qwen-image-data-format}

**Channels-last only for the VAE.** `AutoencoderKLQwenImage` is built as NDHWC
`Conv3D` (with `T=1` for still images); the transformer works on packed sequence
tokens and has no spatial layout. `generate` always hands back
`(batch, height, width, 3)` uint8 images.

| | Shape |
|---|---|
| `latents` passed to `generate` (packed) | `(batch, (H/16)·(W/16), 64)` |
| VAE encode input / decode output | `(batch, H, W, 3)` |
| VAE latent | `(batch, H/8, W/8, 16)` |
| Transformer `sample` | `(batch, seq, 64)` |

### Memory and speed {#qwen-image-memory-and-speed}

The bf16 / float16 checkpoint is about 53 GiB. Building the full graph on a ~40 GB
GPU OOMs; build on CPU (or load with component offload) and move one tower at a
time. A guided 1024px run needs substantial activation headroom on top of the
~38 GiB transformer; 512px is the practical default on 40 GB cards. Prefer the
fused attention path (`keras.ops.dot_product_attention` / torch SDPA) over the
plain matmul softmax default.

### Loading Fine-tuned Weights {#qwen-image-loading-fine-tuned-weights}

The hosted checkpoint is the supported weight; any repo laid out like it
(`zm_config.json` declaring `ZMDiffusion` with `model_type: "qwen_image"`, sharded `*.weights.json` /
`*.weights.h5`, `tokenizer.json`) loads with `from_weights("<org>/<repo>")`. The
`hf:` prefix raises for diffusion models: convert a Diffusers-format checkpoint
once with
`zeromodels/diffusers/qwen_image/convert_qwen_image_diffusers_to_keras.py`
(`transfer_qwen_image(repo)`, `pip install zeromodels[conversion]`) and host the
result.

## Qwen-Image-2.1 {#qwen-image-2-1}

Qwen-Image-2.1, ported to pure Keras 3: latent text-to-image flow-matching with a
32-layer **single-stream** block-causal DiT, a residual 64-channel KL autoencoder
(16× spatial, RGBA), and the Qwen3-VL text tower. It loads through
[`ZMDiffusion`](#zmdiffusion), the one diffusion pipeline class of zeromodels, with
`model_type="qwen_image_21"`: one functional graph holding the three towers, plus
`generate`. Prompts are tokenized by `ZMDiffusionTokenizer`.

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion
is deliberately **not supported** for diffusion models.

Key facts of the port:

- **Unpatched latents**: the denoiser sees `(B, H · W, 64)` tokens (VAE scale 16;
  no 2×2 packing). At 1024px that is a `(B, 4096, 64)` sequence.
- **Block-causal attention**: text is causal; the target image block is
  bidirectional and can attend to all preceding text.
- **`causal_condition`**: text tokens modulate from `t = 0` (timestep-independent),
  matching Diffusers' KV-cache-ready conditioning.
- **True CFG optional**: Diffusers defaults to `true_cfg_scale=1.0` (no guidance).
  Pass `guidance_scale > 1` with a negative prompt to enable dual forwards.
- **Pre-norm text features**: the text tower returns decoder outputs *before* the
  final RMSNorm, matching Diffusers' forward hook on the language-model norm.
- **Schedulers match Diffusers**: `FlowMatchEulerDiscreteScheduler` with dynamic
  resolution shifting (`mu` from image sequence length) and `shift_terminal`.

Links:

- Source: [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1)
- Reference: [diffusers `QwenImage21Pipeline`](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
- License: [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1)
- See also [Qwen-Image](#qwen-image) (1.0 double-stream / packed),
  [qwen3_vl.md](qwen3_vl.md) (text tower)

### Variants {#qwen-image-2-1-variants}

Preconverted, bfloat16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. Each repo is one checkpoint: DiT + VAE +
Qwen3-VL text tower, ~28 GiB at 16-bit.

| Variant | Hub | Source |
|---|---|---|
| `qwen-image-2.1` | [`zeromodels/qwen-image-2.1`](https://huggingface.co/zeromodels/qwen-image-2.1) | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) |

Default `generate_args`: 40 flow-match steps, `guidance_scale=1.0`, 1024×1024.

### API {#qwen-image-2-1-api}

Configs are typed: `QwenImage21Config` (composite, `model_type` `"qwen_image_21"`)
over `QwenImage21TransformerConfig`, `QwenImage21VAEConfig` and
`QwenImage21TextConfig`, plus the checkpoint's `scheduler_config` and Qwen special
token ids. Each repo's `zm_config.json` parses through it; the constructor stays
flat, with the sub-config fields prefixed `transformer_` / `vae_` / `text_`.

#### `ZMDiffusion` {#qwen-image-2-1-zmdiffusion}

The pipeline class of every diffusion family (see [Diffusers](#zmdiffusion)); the
repo's `model_type` (`"qwen_image_21"` here) selects this family, which supplies the hooks
`BaseDiffusion`'s `generate` needs: `encode_prompt` (ChatML template drop after the text
encoder), `predict_noise` on the transformer, `decode_latents` on the VAE with
mean/std un-normalization, and a `FlowMatchEulerDiscreteScheduler` from
`scheduler_config`.

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    negative_attention_mask=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
    height=None,
    width=None,
    output_type="image",
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | ChatML-templated token ids, `**tokenizer(prompts)` |
| `attention_mask` | `None` | padding mask from the tokenizer |
| `negative_input_ids` | `None` | tokenized negative prompt; CFG only when set **and** `guidance_scale > 1` |
| `negative_attention_mask` | `None` | mask for the negative ids |
| `num_inference_steps` | `None` | scheduler steps; `generate_args` (40) when unset |
| `guidance_scale` | `None` | true CFG strength; `generate_args` (1.0) when unset |
| `seed` | `None` | seed for the initial latent |
| `latents` | `None` | explicit unpatched initial latent `(B, H·W, 64)` |
| `height` / `width` | `None` | output pixel size; `default_sample_size * 16` (1024) when unset |
| `output_type` | `"image"` | `"image"` for uint8 RGB, `"latent"` for sequence latents |

Returns `(batch, height, width, 3)` uint8 numpy images (VAE RGBA is cropped to RGB).

| Constructor arg | Default | Meaning |
|---|---|---|
| `model_type` | required | `"qwen_image_21"` (read from the repo's `zm_config.json` on `from_weights`) |
| `scheduler` | `None` | a `BaseScheduler`; built from `scheduler_config` when unset |
| `transformer_sample_size` | `64` | latent side the DiT graph is built for (image / 16) |
| `vae_sample_size` | `1024` | image size the VAE graphs are built for |
| `transformer_*` / `vae_*` / `text_*` | Qwen-Image-2.1 | flat sub-config fields (see [Configuration](configuration.md)) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | `151643` / `151645` / `151643` | Qwen special tokens |
| `scheduler_config` | `None` | Diffusers scheduler dict (dynamic shifting, `max_shift=0.9`, ...) |
| `prompt_template_encode_start_idx` | `14` | ChatML system-prefix tokens dropped after the text encoder |
| `max_sequence_length` | `512` | prompt embed length after the template drop |

The graph is three disconnected paths, one per component.
Inputs cover the transformer (`sample`, `timestep`, `encoder_hidden_states`,
`encoder_hidden_states_mask`), the VAE (`image`, `latent`) and the text tower
(`token_ids`, `padding_mask`). Components are exposed as `.transformer`, `.vae`
and `.text_encoder`.

#### `QwenImage21Transformer2DModel` {#qwen-image-2-1-qwenimage21transformer2dmodel}

The denoiser (Diffusers' `QwenImage21Transformer2DModel`): a 32-layer
single-stream DiT over **unpatched** latents and Qwen3-VL text features, with
3-axis RoPE (`axes_dims_rope=(16, 56, 56)`). Inputs
`{"sample": (B, seq, 64), "timestep": (B,), "encoder_hidden_states": (B, text_seq, 4096)}`
(plus mask), output `{"sample": (B, seq, 64)}` target velocity.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `1` | unpatched (asserted) |
| `in_channels` / `out_channels` | `64` / `64` | latent token width |
| `num_layers` | `32` | single-stream blocks |
| `attention_head_dim` / `num_attention_heads` | `128` / `32` | head geometry (inner dim 4096) |
| `context_in_dim` | `4096` | text feature width from Qwen3-VL |
| `mlp_ratio` | `3` | SwiGLU expansion |
| `causal_condition` | `True` | text modulates from `t=0` |
| `sample_size` | `64` | latent spatial side the graph is built for |
| `text_seq_len` | `512` | static text length after the template drop |

#### `AutoencoderKLQwenImage21` {#qwen-image-2-1-autoencoderklqwenimage21}

The VAE (Diffusers' `AutoencoderKLQwenImage21`): residual encoder/decoder,
`encode(image)` → `(B, H/16, W/16, 64)` latents, `decode(latent)` →
`(B, H, W, 4)` RGBA in `[-1, 1]`. Latent normalisation uses `latents_mean` /
`latents_std` around the denoiser.

| Arg | Default | Meaning |
|---|---|---|
| `base_dim` / `decoder_base_dim` | `96` / `144` | channel widths |
| `z_dim` | `64` | latent channels |
| `dim_mult` | `(1, 2, 4, 8, 8)` | width multipliers per level |
| `scale_factor_spatial` | `16` | pixel → latent downscale |
| `input_channels` / `out_channels` | `4` / `4` | RGBA |
| `is_residual` | `True` | residual blocks |
| `sample_size` | `1024` | image size the graphs are built for |
| `latents_mean` / `latents_std` | Qwen-Image-2.1 | per-channel latent normalisation |

#### `QwenImage21TextEncoderModel` {#qwen-image-2-1-qwenimage21textencodermodel}

Qwen3-VL text tower only (no vision / LM head). Returns
`{"last_hidden_state"}` as **pre-final-norm** hidden states.

#### Schedulers {#qwen-image-2-1-schedulers}

| Scheduler | Notes |
|---|---|
| `FlowMatchEulerDiscreteScheduler` | `use_dynamic_shifting=True`, exponential time shift, `shift_terminal=0.02`, `max_shift=0.9` |

At generate time the task sets timesteps with resolution-dependent `mu` and a
linspace sigma schedule, matching Diffusers.

### Preprocessing {#qwen-image-2-1-preprocessing}

#### `ZMDiffusionTokenizer` {#qwen-image-2-1-zmdiffusiontokenizer}

Qwen3 BPE with Diffusers' T2I ChatML template (`Comprehend and analyze the
provided prompt.`). Returns `{"input_ids", "attention_mask"}`. `encode_prompt`
drops `prompt_template_encode_start_idx` (14) tokens and keeps at most
`max_sequence_length` (512).

```python
ZMDiffusionTokenizer(
    model_type="qwen_image_21", hf_id=None, tokenizer_file=None, max_seq_len=1024
)
```

### End-to-end example {#qwen-image-2-1-end-to-end-example}

#### Single prompt {#qwen-image-2-1-single-prompt}

```python
import os

os.environ["KERAS_BACKEND"] = "torch"

from PIL import Image
from zeromodels.diffusers import ZMDiffusion, ZMDiffusionTokenizer

model = ZMDiffusion.from_weights("zeromodels/qwen-image-2.1")
tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/qwen-image-2.1")

inputs = tokenizer("a photo of a capybara wearing a wizard hat, soft window light")
images = model.generate(
    **inputs,
    height=1024,
    width=1024,
    num_inference_steps=40,
    guidance_scale=1.0,
    seed=0,
)

Image.fromarray(images[0]).save("capybara.png")  # (1024, 1024, 3) uint8
```

#### Optional true CFG {#qwen-image-2-1-optional-true-cfg}

```python
inputs = tokenizer("a watercolor lighthouse at sunset")
negative = tokenizer("blurry, low quality")["input_ids"]

images = model.generate(
    **inputs,
    negative_input_ids=negative,
    guidance_scale=4.0,
    height=1024,
    width=1024,
)
```

#### Reproducible latents {#qwen-image-2-1-reproducible-latents}

At 1024px the unpatched noise is `(batch, 64, 64, 64)` → `(batch, 4096, 64)`:

```python
import numpy as np
from zeromodels.diffusers.qwen_image_21.qwen_image_21_model import pack_latents

h, w, channels = 64, 64, 64
noise = np.random.default_rng(0).standard_normal((1, h, w, channels)).astype("float32")
latents = pack_latents(noise, h, w)
images = model.generate(**tokenizer("a bowl of ramen"), latents=latents)
```

#### Other resolutions {#qwen-image-2-1-other-resolutions}

Graphs are built for a fixed size; weights are not. Rebuild with constructor
overrides (multiples of 32px; checkpoint targets 1024):

```python
model = ZMDiffusion.from_weights(
    "zeromodels/qwen-image-2.1",
    transformer_sample_size=32,
    vae_sample_size=512,
)
images = model.generate(**tokenizer("a mountain lake at dawn"), height=512, width=512)
```

#### Offline conversion {#qwen-image-2-1-offline-conversion}

```bash
pip install zeromodels[conversion]
# KERAS_BACKEND=torch; prefer CPU for the full ~28 GiB bf16 build
python -m zeromodels.diffusers.qwen_image_21.convert_qwen_image_21_diffusers_to_keras
```

`transfer_qwen_image_21(repo)` streams Diffusers shards into a `ZMDiffusion` and
writes sharded `*.weights.json`. Building the full text tower on GPU can OOM
during Functional shape tracing; convert on CPU, then load for CUDA inference.

### Data Format {#qwen-image-2-1-data-format}

**Channels-last for the VAE.** Transformer works on sequence tokens.

| | Shape |
|---|---|
| `latents` passed to `generate` (unpatched) | `(batch, (H/16)·(W/16), 64)` |
| VAE encode input | `(batch, H, W, 4)` RGBA in `[-1, 1]` |
| VAE decode output | `(batch, H, W, 4)`; `generate` returns RGB uint8 |
| VAE latent | `(batch, H/16, W/16, 64)` |
| Transformer `sample` | `(batch, seq, 64)` |

### Memory and speed {#qwen-image-2-1-memory-and-speed}

The bf16 checkpoint is about 28 GiB (DiT ~14 GiB + text ~14 GiB + VAE ~1.4 GiB).
Building the full Functional graph at 1024px on GPU can OOM during attention shape
tracing (layers use `compute_output_spec` to avoid materializing seq² temps). Prefer
CPU convert / build, then CUDA generate. A 1024px run needs substantial activation
headroom on an 80 GB card; prefer fused attention
(`keras.ops.dot_product_attention` / torch SDPA).

### Loading Fine-tuned Weights {#qwen-image-2-1-loading-fine-tuned-weights}

The hosted checkpoint is the supported weight; any repo laid out like it
(`zm_config.json` declaring `ZMDiffusion` with `model_type: "qwen_image_21"`, sharded `*.weights.json` /
`*.weights.h5`, `tokenizer.json`) loads with `from_weights("<org>/<repo>")`. The
`hf:` prefix raises for diffusion models: convert Diffusers format once with
`convert_qwen_image_21_diffusers_to_keras.py` and host the result.

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
