# MetaCLIP 2

**Paper**: [MetaCLIP 2: A Worldwide Scaling Recipe](https://arxiv.org/abs/2507.22062)

MetaCLIP 2 is Meta's second-generation contrastive vision-language model,
trained on multilingual image-text pairs with the XLM-RoBERTa tokenizer.
Architecturally it is a direct descendant of OpenAI CLIP — same
ViT vision tower, same causal-attention text tower, same logit-scale head —
differing only in:

- **Multilingual text encoder** with the XLM-R SentencePiece tokenizer
  (vocab size `901629`, covering 300+ languages).
- **Configurable MLP activation** (`gelu` for most variants,
  `quick_gelu` for the `huge-quickgelu` checkpoint).
- **EOS pooling** via explicit `eos_token_id == 2` lookup (the
  `argmax-over-token-ids` trick CLIP uses would fail because the XLM-R
  mask token `901628` is larger than the EOS id).

kmodels reuses CLIP's attention, position-embedding, and logit-scale
layers directly; only the model wiring and tokenizer are MetaCLIP 2-specific.

## Available Models

### Worldwide (XLM-R tokenizer, vocab 901 629)

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

### mT5 (mT5 tokenizer, vocab 250 000, eos_token_id = 1)

Same CLIP-style transformer architecture as the Worldwide variants; the
only differences are the smaller multilingual vocabulary and a different
EOS id.

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

## Tokenizer

`MetaClip2Tokenizer` wraps `sentencepiece.SentencePieceProcessor` with the
XLM-R model file (downloaded once from the HF hub). Special token ids
match the XLM-R convention: `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`,
`<mask>=901628`. Context length defaults to 77 (same as CLIP).

```python
from kmodels.models.metaclip2 import MetaClip2Tokenizer
tok = MetaClip2Tokenizer(context_length=77)
ids = tok(["hello world", "bonjour le monde"])
print(ids["token_ids"].shape)  # (2, 77)
```

## Data format

`MetaClip2ImageProcessor` accepts the same `data_format=None` kwarg as
every other kmodels processor — `None` resolves to
`keras.config.image_data_format()`; pass `"channels_first"` /
`"channels_last"` to override per-call.

## Weight conversion

Convert any HF MetaCLIP 2 checkpoint to Keras weights with:

```bash
python -m kmodels.models.metaclip2.convert_metaclip2_hf_to_keras MetaClip2WorldwideB32
```

The script downloads the HF model (via `transformers.AutoModel`), walks
the Keras weight tree, and writes a sharded `.weights.h5` file. The
conversion is identical to CLIP's name-mapping scheme except it handles
the larger text-tower (up to 32 layers, 1280 hidden) and the 901629-token
embedding table.

## Citation

```bibtex
@article{chuang2025metaclip2,
  title={MetaCLIP 2: A Worldwide Scaling Recipe},
  author={Chuang, Yung-Sung and Xie, Saining and others},
  journal={arXiv preprint arXiv:2507.22062},
  year={2025}
}
```
