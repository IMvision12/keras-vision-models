# MetaCLIP 2

**Paper**: [MetaCLIP 2: A Worldwide Scaling Recipe](https://arxiv.org/abs/2507.22062)

MetaCLIP 2 is Meta's second-generation contrastive vision-language model,
trained on multilingual image-text pairs. Architecturally it is a direct
descendant of OpenAI CLIP — same ViT vision tower, same causal-attention
text tower, same logit-scale head — differing only in:

- **Multilingual text encoder** in two tokenizer families:
  - **Worldwide** — XLM-R SentencePiece tokenizer, vocab `901 629`, 300+
    languages (13 variants).
  - **mT5** — SigLIP-style SentencePiece tokenizer, vocab `250 100`
    (3 variants).
- **Configurable MLP activation** (`gelu` for most variants, `quick_gelu`
  for the `huge-quickgelu` checkpoint).
- **EOS pooling** via explicit `eos_token_id` lookup instead of the
  `argmax-over-token-ids` trick CLIP uses (XLM-R's mask token
  `901628` would break that trick; mT5's `eos=1` also wouldn't be the
  argmax).

kmodels reuses CLIP's attention, position-embedding, and logit-scale
layers directly; only the model wiring and tokenizers are MetaCLIP
2-specific.

## Available Models

### Worldwide (XLM-R tokenizer, vocab 901 629, eos = 2)

| Variant | Image | Patch | Vision hidden / L | Text hidden / L | Proj | Activation |
|---|---|---|---|---|---|---|
| `MetaClip2WorldwideS16` | 224 | 16 | 384 / 12 | 384 / 12 | 384 | gelu |
| `MetaClip2WorldwideS16_384` | 384 | 16 | 384 / 12 | 384 / 12 | 384 | gelu |
| `MetaClip2WorldwideM16` | 224 | 16 | 512 / 12 | 512 / 12 | 512 | gelu |
| `MetaClip2WorldwideM16_384` | 384 | 16 | 512 / 12 | 512 / 12 | 512 | gelu |
| `MetaClip2WorldwideB16` | 224 | 16 | 768 / 12 | 512 / 12 | 512 | gelu |
| `MetaClip2WorldwideB16_384` | 384 | 16 | 768 / 12 | 512 / 12 | 512 | gelu |
| `MetaClip2WorldwideB32` | 224 | 32 | 768 / 12 | 512 / 12 | 512 | gelu |
| `MetaClip2WorldwideB32_384` | 384 | 32 | 768 / 12 | 512 / 12 | 512 | gelu |
| `MetaClip2WorldwideL14` | 224 | 14 | 1024 / 24 | 768 / 12 | 768 | gelu |
| `MetaClip2WorldwideHugeQuickgelu` | 224 | 14 | 1280 / 32 | 1024 / 24 | 1024 | quick_gelu |
| `MetaClip2WorldwideHuge378` | 378 | 14 | 1280 / 32 | 1024 / 24 | 1024 | gelu |
| `MetaClip2WorldwideGiant` | 224 | 14 | 1664 / 48 | 1280 / 32 | 1280 | gelu |
| `MetaClip2WorldwideGiant378` | 378 | 14 | 1664 / 48 | 1280 / 32 | 1280 | gelu |

### mT5 (mT5 tokenizer, vocab 250 100, eos = 1)

Same CLIP-style transformer architecture as the Worldwide variants; only
the tokenizer vocab and EOS id differ.

| Variant | Image | Patch | Vision hidden / L | Text hidden / L | Proj | Activation |
|---|---|---|---|---|---|---|
| `MetaClip2Mt5WorldwideS16` | 224 | 16 | 384 / 12 | 384 / 12 | 384 | gelu |
| `MetaClip2Mt5WorldwideM16` | 224 | 16 | 512 / 12 | 512 / 12 | 512 | gelu |
| `MetaClip2Mt5WorldwideB32` | 224 | 32 | 768 / 12 | 512 / 12 | 512 | gelu |

## Basic Usage

```python
from kmodels.models.metaclip2 import (
    MetaClip2WorldwideB32,
    MetaClip2Processor,
)

model = MetaClip2WorldwideB32(weights="worldwide_224")
processor = MetaClip2Processor(image_resolution=224)

inputs = processor(
    text=["un chat", "a dog", "ein Auto"],
    image_paths="photo.jpg",
)
outputs = model(inputs, training=False)
print(outputs["image_logits"].shape)
```

## Tokenizers

Both tokenizers are **pure-Python wrappers around `sentencepiece`** — no
`transformers` dependency at runtime. The SentencePiece model files are
hosted on the kmodels [`metaclip-2`
release](https://github.com/IMvision12/keras-models/releases/tag/metaclip-2)
and downloaded + cached on first use.

### `MetaClip2Tokenizer` — for Worldwide (XLM-R) variants

Wraps `sentencepiece.SentencePieceProcessor` with the XLM-R convention:
fairseq-style offset `+1` on SentencePiece ids, BOS/EOS wrapping, EOS
padding. Hardcoded token ids: `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`,
`<mask>=901628`. Context length defaults to `77`.

```python
from kmodels.models.metaclip2 import MetaClip2Tokenizer
tok = MetaClip2Tokenizer(context_length=77)
out = tok(inputs=["hello world", "bonjour le monde"])
print(out["token_ids"].shape)  # (2, 77)
```

### `MetaClip2Mt5Tokenizer` — for mT5 variants

SigLIP-style tokenizer: **lowercase text**, SP encode, append `eos=1`,
pad with `eos` (no bos). Hardcoded token ids: `eos=1`, `pad=1`, `unk=2`.

```python
from kmodels.models.metaclip2 import MetaClip2Mt5Tokenizer
tok = MetaClip2Mt5Tokenizer(context_length=77)
out = tok(inputs=["hello world", "una foto de un gato"])
print(out["token_ids"].shape)  # (2, 77)
```

### Tokenizer files (release assets)

Two SentencePiece model files — no tokenizer config JSON needed. Special
token ids are hardcoded in Python constants since the XLM-R / SigLIP
conventions are fixed.

| File | Used by | Vocab |
|---|---|---|
| [`sentencepiece.bpe.model`](https://github.com/IMvision12/keras-models/releases/download/metaclip-2/sentencepiece.bpe.model) | `MetaClip2Tokenizer` (13 Worldwide variants) | 901 629 |
| [`spiece.model`](https://github.com/IMvision12/keras-models/releases/download/metaclip-2/spiece.model) | `MetaClip2Mt5Tokenizer` (3 mT5 variants) | 250 100 |

Both are ~5 MB. Token-id parity with HF's `XLMRobertaTokenizerFast` and
`SiglipTokenizer` is verified — kmodels produces **bit-identical** token
ids across English / German / French / Chinese / Japanese inputs.

## Image processor

`MetaClip2ImageProcessor` inherits from `CLIPImageProcessor` but uses
the HF MetaCLIP 2 default config: **direct square resize** to
`image_resolution` with `PIL.BICUBIC` (no shortest-edge scaling + center
crop, which is CLIP's default). Rescale to `[0, 1]` and OpenAI-CLIP
normalization follow.

```python
from kmodels.models.metaclip2 import MetaClip2ImageProcessor
proc = MetaClip2ImageProcessor(image_resolution=224)
pixel_values = proc(image_paths="photo.jpg")["images"]
# (1, 224, 224, 3) float32, normalized
```

## Data format

`MetaClip2ImageProcessor` accepts the same `data_format=None` kwarg as
every other kmodels processor — `None` resolves to
`keras.config.image_data_format()`; pass `"channels_first"` /
`"channels_last"` to override per-call.

## Citation

```bibtex
@article{chuang2025metaclip2,
  title={MetaCLIP 2: A Worldwide Scaling Recipe},
  author={Chuang, Yung-Sung and Xie, Saining and others},
  journal={arXiv preprint arXiv:2507.22062},
  year={2025}
}
```
