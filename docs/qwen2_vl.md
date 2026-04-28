# Qwen2-VL ⚠️ Under Progress

> **⚠️ This model is under active development.** The architecture and
> conversion path are in place, but end-to-end multimodal generation
> parity with HuggingFace has only been spot-checked on the 2B-Instruct
> variant. Expect API changes; do not rely on it for production.

**Paper**: [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)

Qwen2-VL is Alibaba's second-generation vision-language model. It pairs
a Qwen2 LLM (decoder-only transformer with grouped-query attention and
M-RoPE) with a ViT-based vision tower whose output is merged into the
LLM token stream at `<|image_pad|>` placeholder positions. It supports
multi-image input, video frames (via temporal-patch grouping), and
arbitrary aspect ratios through dynamic patch counts.

kmodels ships a **pure Keras 3** port of all six official Qwen2-VL
checkpoints. Weights are not redistributed: the first call to a
factory function downloads the HF checkpoint, converts it on the fly,
and caches the result under `~/.cache/kmodels/<variant>/`.

## Model Variants

| Variant | Params | LM Layers | Hidden | Heads | KV Heads | LM head | HF source |
|---|---|---|---|---|---|---|---|
| `Qwen2VL2B` | 1.5 B | 28 | 1536 | 12 | 2 | tied | `Qwen/Qwen2-VL-2B` |
| `Qwen2VL2BInstruct` | 1.5 B | 28 | 1536 | 12 | 2 | tied | `Qwen/Qwen2-VL-2B-Instruct` |
| `Qwen2VL7B` | 8 B | 28 | 3584 | 28 | 4 | untied | `Qwen/Qwen2-VL-7B` |
| `Qwen2VL7BInstruct` | 8 B | 28 | 3584 | 28 | 4 | untied | `Qwen/Qwen2-VL-7B-Instruct` |
| `Qwen2VL72B` | 72 B | 80 | 8192 | 64 | 8 | untied | `Qwen/Qwen2-VL-72B` |
| `Qwen2VL72BInstruct` | 72 B | 80 | 8192 | 64 | 8 | untied | `Qwen/Qwen2-VL-72B-Instruct` |

All variants share the same vision tower (32 layers, `embed_dim=1280`,
16 heads, patch size 14, temporal-patch 2, spatial-merge 2) and M-RoPE
configuration (`mrope_section=[16, 24, 24]`, `rope_theta=1e6`). Only
the LLM size and `tie_word_embeddings` differ — the 7B / 72B variants
carry a separate `lm_head` Dense, while 2B reuses the input embedding
table for the output projection.

## Available Weights

Each factory's default `weights="qwen"` triggers an on-the-fly download
+ convert from HuggingFace on first use. There is no kmodels-hosted
copy. Subsequent calls load instantly from the local cache.

| Variant | `qwen` | RAM at conversion |
|---|:-:|---|
| `Qwen2VL2B` | ✅ | ~6 GB |
| `Qwen2VL2BInstruct` | ✅ | ~6 GB |
| `Qwen2VL7B` | ✅ | ~32 GB |
| `Qwen2VL7BInstruct` | ✅ | ~32 GB |
| `Qwen2VL72B` | ✅ | ~150 GB |
| `Qwen2VL72BInstruct` | ✅ | ~150 GB |

The base (non-Instruct) variants depend on Qwen having published that
specific tag — if the HF repo doesn't exist, the call fails with a
clear 404. The Instruct variants are the recommended starting point.

## Basic Usage

```python
from kmodels.models.qwen2_vl import (
    Qwen2VL2BInstruct,
    Qwen2VLImageProcessor,
    Qwen2VLTokenizer,
    qwen2_vl_generate,
)

bundle = Qwen2VL2BInstruct(weights="qwen")
tokenizer = Qwen2VLTokenizer()
image_processor = Qwen2VLImageProcessor()

ids, text = qwen2_vl_generate(
    bundle,
    tokenizer,
    image_processor,
    prompt_text="Describe this image in one sentence.",
    images=["photo.jpg"],
    max_new_tokens=128,
)
print(text)
```

`bundle` is a dict containing `embed_tokens`, `llm`, `vision`,
`lm_head` (only for 7B / 72B), `llm_inv_freq`, `vision_inv_freq`, and
the resolved config. Generation handles the tied vs untied LM head
distinction automatically.

## Multi-Image / Text-Only Prompts

`qwen2_vl_generate` accepts a list of images (Python paths or PIL
images) and a chat-style prompt. Pass `images=None` for text-only
generation:

```python
ids, text = qwen2_vl_generate(
    bundle, tokenizer, image_processor,
    prompt_text="Write a short haiku about the ocean.",
    images=None,
    max_new_tokens=64,
)
```

For multi-image inputs, list every image explicitly; the prompt should
contain the same number of `<|image_pad|>` placeholder positions
(handled automatically by `tokenizer.build_chat_prompt`).

## Architecture Notes

### Vision tower

The visual encoder takes pre-flattened patches of shape
`(N_patches, 3 * temporal_patch_size * patch_size * patch_size) =
(N, 1176)` and produces `(N_merged, hidden_size)` LLM-space features
after a 2×2 spatial merge and a two-layer GELU MLP merger. Position
information is provided as pre-computed
`(vision_cos, vision_sin)` matching each patch's `(t, h, w)` triplet.

### LLM

The text decoder accepts pre-embedded inputs (`inputs_embeds`) so the
multimodal path can substitute vision features at `<|image_pad|>`
positions before the first transformer block. Inside each block, the
self-attention applies M-RoPE — three independent RoPE rotations for
the temporal / height / width axes, concatenated along the head
dimension per `mrope_section`.

### Untied LM head (7B / 72B)

Qwen ties word embeddings only for the 2B variants. The 7B and 72B
checkpoints carry a separate `lm_head.weight`; kmodels mirrors this
with a `keras.layers.Dense(vocab_size, use_bias=False)` stored as
`bundle["lm_head"]`. Generation picks `lm_head` when present, falls
back to `embed_tokens.T` otherwise.

## Cache Layout

Converted weights are saved as separate `.weights.h5` files per
sub-component under `~/.cache/kmodels/<variant_lowercase>/`:

```
~/.cache/kmodels/qwen2vl2binstruct/
  qwen2vl2binstruct_embed.weights.h5
  qwen2vl2binstruct_llm.weights.h5
  qwen2vl2binstruct_vision.weights.h5
  qwen2vl2binstruct_lm_head.weights.h5   # only for 7B / 72B
```

Delete the directory to force a re-conversion from HF.

## Citation

```bibtex
@article{wang2024qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the
         World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and
          Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing
          and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai
          and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu,
          Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```
