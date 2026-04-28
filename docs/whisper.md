# Whisper

**Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)

Whisper is OpenAI's encoder-decoder transformer for multilingual speech
recognition and translation. The encoder ingests an 80- or 128-channel
log-mel spectrogram (30 s, 16 kHz, padded) through a 2-layer conv stem
(stride 1 + stride 2) and a stack of pre-LN transformer blocks; the
decoder generates byte-level BPE token ids autoregressively, attending
to the encoder output via cross-attention. Token embeddings are tied
with the LM head.

kmodels ships a **pure Keras 3** port of all eight official OpenAI
checkpoints with bit-close parity to HuggingFace's reference
implementation. The processor, encoder, decoder, and greedy `generate`
loop run unmodified on TensorFlow / Torch / JAX backends — no
`transformers` or `torch` runtime dependency.

## Model Variants

| Variant | Params | Layers (enc / dec) | d_model | Heads | Mel bins | Vocab | Tokenizer |
|---|---|---|---|---|---|---|---|
| `WhisperTiny` | 39 M | 4 / 4 | 384 | 6 | 80 | 51 865 | v1 |
| `WhisperBase` | 74 M | 6 / 6 | 512 | 8 | 80 | 51 865 | v1 |
| `WhisperSmall` | 244 M | 12 / 12 | 768 | 12 | 80 | 51 865 | v1 |
| `WhisperMedium` | 769 M | 24 / 24 | 1024 | 16 | 80 | 51 865 | v1 |
| `WhisperLarge` | 1 550 M | 32 / 32 | 1280 | 20 | 80 | 51 865 | v1 |
| `WhisperLargeV2` | 1 550 M | 32 / 32 | 1280 | 20 | 80 | 51 865 | v1 |
| `WhisperLargeV3` | 1 550 M | 32 / 32 | 1280 | 20 | **128** | 51 866 | v3 |
| `WhisperLargeV3Turbo` | 809 M | 32 / **4** | 1280 | 20 | 128 | 51 866 | v3 |

**v1** (51 865) covers tiny → large-v2; **v3** (51 866) covers
large-v3 + large-v3-turbo (one extra Cantonese language id and 128 mel
bins instead of 80).

## Available Weights

Every variant ships a single `"openai"` weights preset converted from
the official OpenAI checkpoints on HuggingFace. One combined
`.weights.h5` file per variant (encoder + decoder together) is hosted
under the kmodels
[`whisper`](https://github.com/IMvision12/keras-models/releases/tag/whisper)
release tag and downloaded on first use, then cached locally.

| Variant | `openai` |
|---|:-:|
| `WhisperTiny` | ✅ |
| `WhisperBase` | ✅ |
| `WhisperSmall` | ✅ |
| `WhisperMedium` | ✅ |
| `WhisperLarge` | ✅ |
| `WhisperLargeV2` | ✅ |
| `WhisperLargeV3` | ✅ |
| `WhisperLargeV3Turbo` | ✅ |

## Model

Every variant factory (`WhisperTiny`, `WhisperBase`, ...) returns a
`Whisper` — a `keras.Model` Functional subclass that wires the
encoder and decoder into a single graph. The encoder and decoder are
exposed as attributes for inference / generation paths:

```python
from kmodels.models.whisper import WhisperTiny

model = WhisperTiny(weights="openai")        # Whisper instance

model.encoder        # keras.Model: input_features -> (B, T, d_model)
model.decoder        # keras.Model: {decoder_input_ids, encoder_hidden_states} -> logits
model.d_model        # 384
model.vocab_size     # 51865

# Joint forward pass (teacher-forced training):
out = model({
    "input_features":   mel,    # (B, n_mels, 3000)
    "decoder_input_ids": ids,   # (B, L)
})
out["encoder_hidden_states"]    # (B, T, d_model)
out["logits"]                   # (B, L, vocab_size)
```

The class is also constructable directly with custom hyperparameters
for from-scratch training:

```python
from kmodels.models.whisper import Whisper

model = Whisper(
    d_model=384,
    encoder_layers=4, decoder_layers=4,
    encoder_attention_heads=6, decoder_attention_heads=6,
    encoder_ffn_dim=1536, decoder_ffn_dim=1536,
    num_mel_bins=80, vocab_size=51865,
)
```

## Features and Capabilities

- **Multilingual ASR**: 99 languages via the v1 tokenizer, 100 via v3
  (adds Cantonese).
- **Speech-to-text translation**: any supported language to English via
  the `<|translate|>` task token.
- **High-level wrapper**: `WhisperGenerate` for one-call ASR /
  translation (audio in, text out).
- **Single processor entry point**: `WhisperProcessor` bundles the
  feature extractor, tokenizer, and `forced_decoder_ids` builder —
  matches the HF API surface so port code is one-liner equivalent.
- **Pure Keras 3**: feature extractor uses `keras.ops.stft` and runs on
  any backend; tokenizer is a Rust-backed
  `tokenizers.Tokenizer` (no `transformers`).
- **Fine-tunable**: every variable in the encoder + decoder is
  trainable; gradients flow through the tied LM head.

## Basic Usage

The shortest path is `WhisperGenerate` — one callable that wraps
encoder + decoder + processor + greedy decoding.

```python
from kmodels.models.whisper import (
    WhisperTiny, WhisperProcessor, WhisperGenerate,
)

model = WhisperTiny(weights="openai")
processor = WhisperProcessor(variant="v1")    # 51865 vocab, 80 mels

generator = WhisperGenerate(model, processor)

# raw_audio: 1-D float32 in [-1, 1] at 16 kHz
text = generator(raw_audio, language="en", task="transcribe")
print(text)        # ['hello world']
```

## Real-World End-to-End Example

End-to-end transcription on a real audio sample (5.86 s clip from
LibriSpeech). The asset ships with the repo at
``assets/librispeech_sample.wav`` (16 kHz mono, 16-bit PCM).

**Input audio:**

<audio controls>
  <source src="../assets/librispeech_sample.wav" type="audio/wav">
  Your browser does not support the audio element. Download the file
  directly: <a href="../assets/librispeech_sample.wav">librispeech_sample.wav</a>
</audio>

**Reference transcript (LibriSpeech ground truth):**

> MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD
> TO WELCOME HIS GOSPEL

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import soundfile as sf

from kmodels.models.whisper import (
    WhisperBase, WhisperProcessor, WhisperGenerate,
)

# Build model + processor + greedy generator
model = WhisperBase(weights="openai")
processor = WhisperProcessor(variant="v1")
generator = WhisperGenerate(model, processor)

# Load a 16 kHz mono float32 waveform
audio, sr = sf.read("assets/librispeech_sample.wav")
assert sr == 16000

# Transcribe — `WhisperGenerate` runs feature extraction, encoder,
# greedy decoding, and detokenization in one call.
text = generator(audio, language="en", task="transcribe", max_new_tokens=224)
print(text[0])
```

**Output:**

```
 Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
```

Whisper's training data includes capitalization and punctuation, so the
output reads more naturally than the LibriSpeech reference (which is
all-caps and punctuation-stripped). Both are correct — the words match
once you normalize.

### Translation to English

Swap `task="transcribe"` for `task="translate"` — `WhisperGenerate`
looks up the right token ids internally:

```python
text = generator(wave, language="fr", task="translate", max_new_tokens=224)
```

### Auto language detection

Pass `language=None` to drop the language slot from the prompt — the
decoder picks the language from its own logits at position 1.

```python
text = generator(wave, language=None, task="transcribe")
```

### Returning token ids instead of text

```python
ids = generator(wave, language="en", return_ids=True)   # List[List[int]]
```

### Using the lower-level API directly

If you need custom decoding (beam search, prefix scoring, KV-cache,
etc.), call ``model.encoder`` and ``model.decoder`` directly:

```python
inputs = processor(audio=wave, sampling_rate=16000)
forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")

# Encode once.
enc_out = model.encoder(inputs["input_features"])

# Then drive decoding however you like — call model.decoder per step
# with {"decoder_input_ids": ids, "encoder_hidden_states": enc_out}
# and apply your own logits processors / search strategy.
```

### Large-v3 / large-v3-turbo

`WhisperLargeV3` and `WhisperLargeV3Turbo` use **128 mel bins** and the
**v3 tokenizer** (vocab 51 866 — adds Cantonese `yue`). The processor
handles both via constructor kwargs:

```python
from kmodels.models.whisper import WhisperLargeV3Turbo, WhisperProcessor

model = WhisperLargeV3Turbo(weights="openai")
processor = WhisperProcessor(variant="v3", n_mels=128)

# Cantonese transcription
forced = processor.get_decoder_prompt_ids(language="yue", task="transcribe")
```

Token ids for `<|transcribe|>` / `<|translate|>` / `<|notimestamps|>`
shift by `+1` between v1 and v3 (because of the extra Cantonese
language id) — `get_decoder_prompt_ids` resolves them dynamically from
each variant's `added_tokens.json`, so the same calling code works
across all eight variants.

## Processor

`WhisperProcessor` is the recommended top-level entry point — it
bundles the feature extractor, tokenizer, and `forced_decoder_ids`
builder behind a single object that mirrors HuggingFace's
`transformers.WhisperProcessor` API.

```python
from kmodels.models.whisper import WhisperProcessor

processor = WhisperProcessor(
    variant="v1",        # or "v3"
    n_mels=80,           # 128 for large-v3 / large-v3-turbo
    sampling_rate=16000,
    chunk_length=30,
)

# audio path
out = processor(audio=wave, sampling_rate=16000)
# {"input_features": (B, n_mels, T)}

# label path (fine-tuning)
out = processor(text=["hello world", "foo bar"])
# {"input_ids": (B, L), "attention_mask": (B, L)}

# both at once
out = processor(audio=wave, text="hello world", sampling_rate=16000)
# {"input_features": ..., "input_ids": ..., "attention_mask": ...}

# decoder prompt
forced = processor.get_decoder_prompt_ids(
    language="en",       # ISO code or "<|en|>"; None for autodetect
    task="transcribe",   # or "translate"
    no_timestamps=True,  # set False to keep <|notimestamps|> off
)

# decoded text
text = processor.decode(ids, skip_special_tokens=True)
texts = processor.batch_decode(ids_batch, skip_special_tokens=True)

# the underlying components are still accessible
processor.feature_extractor   # WhisperFeatureExtractor
processor.tokenizer           # WhisperTokenizer
```

The two sub-components are documented below in case you need fine
control over them — but for normal use, `WhisperProcessor` is enough.

## Feature Extractor

`WhisperFeatureExtractor` is a **pure Keras 3** mel-spectrogram
extractor — every numeric op (`stft`, `matmul`, `log`, `maximum`) goes
through `keras.ops`, so the same code runs on TF / Torch / JAX. Only
the input-list normalization is numpy plumbing.

Pipeline (matches HF / OpenAI exactly):

1. Pad or truncate raw waveform to 30 s @ 16 kHz (`n_samples = 480 000`).
2. STFT: `n_fft = 400`, `hop = 160`, Hann window, centered reflect pad.
3. Power spectrogram → 80- (or 128-) channel Slaney mel bank, 0–8 kHz.
4. `log10(max(mel, 1e-10))`, clamp to `max - 8.0`, then `(x + 4) / 4`.

```python
feat = WhisperFeatureExtractor(
    sampling_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80,        # use 128 for large-v3 / large-v3-turbo
    chunk_length=30,
)
mel = feat(raw_audio_or_list_of_waves)   # (B, n_mels, 3000)
```

Verified against HF `WhisperFeatureExtractor` to **max diff ~2.4e-5**
on real audio.

## Tokenizer

`WhisperTokenizer` wraps a Rust-backed `tokenizers.Tokenizer` (BPE +
ByteLevel pre/post) with the ~1607 Whisper special tokens (languages,
timestamps, task tokens) registered via `add_special_tokens`. No
runtime `transformers` dependency.

```python
from kmodels.models.whisper import WhisperTokenizer

tok = WhisperTokenizer(variant="v1")    # tiny..large-v2 (51865 vocab)
ids = tok.encode("Hello, world!")
text = tok.decode(ids, skip_special_tokens=True)

tok_v3 = WhisperTokenizer(variant="v3") # large-v3 / large-v3-turbo (51866 vocab)
```

Vocab files (`vocab.json`, `merges.txt`, `added_tokens.json`) are
hosted on the kmodels `whisper` release tag and downloaded on first
use.

| File | Variant | Vocab |
|---|---|---|
| `vocab.json` + `merges.txt` + `added_tokens.json` | `v1` (tiny / base / small / medium / large / large-v2) | 51 865 |
| `vocab_v3.json` + `merges_v3.txt` + `added_tokens_v3.json` | `v3` (large-v3 / large-v3-turbo) | 51 866 |

## Generation

The greedy decoding loop is bundled into :class:`WhisperGenerate`
(matches HF's default Whisper generate). It supports the same logit
processors:

- `forced_decoder_ids` (built by `processor.get_decoder_prompt_ids`):
  at decoded position `k`, force a specific token. This is how OpenAI
  / HF inject the language + task + no-timestamps prefix at positions
  1 / 2 / 3.
- `suppress_tokens`: permanently masked token ids. Defaults to OpenAI's
  hard-coded list of 88 ids (punctuation-only / non-speech).
- `begin_suppress_tokens`: masked only at the first decoded step.
  Defaults to `[220, 50257]` (space + `<|endoftext|>`).

```python
from kmodels.models.whisper.config import WHISPER_SUPPRESS_TOKENS

generator = WhisperGenerate(model, processor)
text = generator(
    wave,
    language="en", task="transcribe",
    max_new_tokens=224,
    suppress_tokens=WHISPER_SUPPRESS_TOKENS,   # default when None
    begin_suppress_tokens=[220, 50257],        # default when None
)
```

`model.encoder(mel)` and `model.decoder({"decoder_input_ids": ids,
"encoder_hidden_states": enc_out})` are also exposed directly for
custom decoding loops (beam search, prefix scoring, KV-cache
implementations, etc.).

## Fine-tuning

All variables in the encoder + decoder are trainable, and the tied LM
head produces gradients into the embedding table. The processor's
text path is what feeds the label tensor:

```python
import keras
from kmodels.models.whisper import WhisperTiny, WhisperProcessor

model = WhisperTiny(weights="openai")
encoder, decoder = model.encoder, model.decoder
processor = WhisperProcessor(variant="v1")

# audio batch -> input features
inputs = processor(audio=audio_batch, sampling_rate=16000)
# text batch -> label ids
labels = processor(text=text_batch)["input_ids"]

optimizer = keras.optimizers.AdamW(1e-5)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train_step(input_features, decoder_input_ids, labels):
    with keras.backend.GradientTape() as tape:    # framework-specific
        enc_out = encoder(input_features, training=True)
        logits = decoder(
            {"decoder_input_ids": decoder_input_ids,
             "encoder_hidden_states": enc_out},
            training=True,
        )
        loss = loss_fn(labels, logits)
    grads = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, encoder.trainable_variables + decoder.trainable_variables)
    )
    return loss
```

Verified on a 10-step overfit run: loss drops from **5.21 → 0.12**,
166/166 trainable variables receive gradients.

## Citation

```bibtex
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg
          and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```
