# Sentence Transformer

<div class="kf-note kf-note--weights">
<b>Weights:</b> Pre-converted Keras weights for all 13 checkpoints are hosted under the
<a href="https://huggingface.co/zeromodels">zeromodels organization</a>. Load one with
<code>from_weights("zeromodels/all-MiniLM-L6-v2")</code>; no conversion is needed.
Original <code>sentence-transformers</code> checkpoints can also be converted on the fly
with the <code>hf:</code> prefix, e.g.
<code>from_weights("hf:sentence-transformers/all-MiniLM-L6-v2")</code>.
The backbone and embedding recipe are read from the checkpoint metadata.
</div>

ZeroModels loads 13 original dense sentence-embedding checkpoints. See the
[Sentence Transformer models](sentence_transformers.md) page for the available
ZeroModels-hosted checkpoint IDs and Hub links.
It builds a Keras 3 encoder and applies the checkpoint's declared pooling,
optional Dense projection, and optional L2 normalization. Parity has been
verified on the Keras Torch backend; TensorFlow and JAX parity have not yet
been checked for all 13 checkpoints.

The public API is `zeromodels.sentence_transformers`, separate from the upstream
`sentence_transformers` reference library, and it is exactly two classes:
`ZMSentenceTransformer` and `ZMSentenceTransformerTokenizer`. Hub repos whose
`zm_config.json` or `zm_preprocessor.json` still name the former classes
(`SentenceTransformerModel`, `SentenceTransformer`, `SentenceTransformerTextEmbed`,
`SentenceTransformerTokenizer`) load into these two unchanged.

| | |
|---|---|
| Class | `ZMSentenceTransformer` |
| Tokenizer | `ZMSentenceTransformerTokenizer` |
| Selector | `backbone_type` (one architecture, pluggable encoder) |
| Recipes | `SENTENCE_TRANSFORMER_BACKBONES` maps each `backbone_type` to its encoder builder: the `zeromodels.models` MPNet / BERT / RoBERTa / XLM-R encoders, and DistilBERT / ALBERT builders in `sentence_transformer_backbones.py` |
| Hub files | `zm_config.json` + `zm_preprocessor.json` + `tokenizer.json` |

Supported `backbone_type` values: `mpnet`, `bert`, `roberta`, `xlm-roberta`,
`distilbert`, `albert`.

## Usage

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.sentence_transformers import (
    ZMSentenceTransformer,
    ZMSentenceTransformerTokenizer,
)

repo = "zeromodels/all-MiniLM-L6-v2"
model = ZMSentenceTransformer.from_weights(repo)
tokenizer = ZMSentenceTransformerTokenizer.from_weights(repo)

embeddings = model.encode(
    ["A man is eating food.", "A man is eating pasta."],
    tokenizer=tokenizer,
)
print(embeddings.shape)  # (2, 384)
print(model.similarity(embeddings[:1], embeddings[1:]))
```

`model(tokenizer(texts))` returns `embeddings` and `token_embeddings` tensors.
`encode()` batches raw strings and returns a NumPy array. Use the same repo for
the model and tokenizer. `model.similarity()` uses the checkpoint's declared
cosine or dot-product scoring function.

## Auto loading

The same ZeroModels-hosted repositories work with the Auto API:

```python
from zeromodels import AutoZModel, AutoZMTokenizer

repo = "zeromodels/all-MiniLM-L6-v2"
model = AutoZModel.from_weights(repo)  # ZMSentenceTransformer
tokenizer = AutoZMTokenizer.from_weights(repo)  # ZMSentenceTransformerTokenizer
```

Detection uses `model_type: "sentence_transformer"` in `zm_config.json`, so this
works for all 13 hosted checkpoints and compatible fine-tunes. The model's full
embedding recipe and the tokenizer's checkpoint-specific settings are preserved.

## Tokenizer preprocessing metadata

Each checkpoint uses its own `tokenizer.json` for the vocabulary, normalization,
token splitting, and special-token insertion. Optional `zm_preprocessor.json`
records the surrounding official configuration in two separate blocks:

```json
{
  "library_name": "zeromodels",
  "preprocessor_module": "zeromodels.sentence_transformers",
  "preprocessor_class": "ZMSentenceTransformerTokenizer",
  "tokenizer_file": "tokenizer.json",
  "backbone_type": "bert",
  "tokenizer_config": {
    "do_lower_case": true,
    "model_max_length": 512,
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]"
  },
  "sentence_transformer_config": {
    "max_seq_length": 256,
    "do_lower_case": false
  }
}
```

This abbreviated example represents `all-MiniLM-L6-v2`. The tokenizer's own
lowercasing is already encoded in `tokenizer.json`; the Sentence Transformers
flag requests an additional normalization step only when true. The effective
sequence limit is 256, including special tokens. Padding and truncation sides
come from `tokenizer_config`, defaulting to right; batches pad to their longest
sequence and pairs use longest-first truncation.

Both metadata and tokenizer assets load from one resolved Hub commit, available
as `tokenizer.source_revision`. Pass `revision=` to select a particular snapshot.
An explicit `max_seq_len=` overrides the stored limit. Legacy repos without
`zm_preprocessor.json` continue to read the backbone and sequence limit from
`zm_config.json`, including its nested `text_config` form.

`write_zm_preprocessor(directory, tokenizer, variant)` exports a portable recipe
referencing `tokenizer.json` in the same repo, rather than a local cache path.
New exports set both `model_module` in `zm_config.json` and `preprocessor_module`
in `zm_preprocessor.json` to `zeromodels.sentence_transformers`.
Text preprocessing uses the `tokenizers` library; `transformers` is not needed
to load these converted checkpoints.

## Original Sentence Transformers checkpoints

To load directly from an original Sentence Transformers repo instead, use the
`hf:` prefix for both components and pass the model's resolved revision to the
tokenizer:

```python
upstream = "hf:sentence-transformers/all-MiniLM-L6-v2"
model = ZMSentenceTransformer.from_weights(upstream)
tokenizer = ZMSentenceTransformerTokenizer.from_weights(
    upstream, revision=model.source_revision
)
```

An explicit commit SHA can be supplied as `revision=` to pin the upstream
snapshot.

`AutoZMTextEmbed` resolves ZeroModels `sentence_transformer` configs to
`ZMSentenceTransformer` as well. For upstream `hf:` repos, load the
concrete class as shown above: their root `model_type` describes the backbone,
not the sentence-embedding task.

Unsupported upstream module types, combinations, or pooling modes raise a
`ValueError` rather than silently changing the embedding recipe. Cross-encoders,
sparse encoders, and multi-vector encoders are separate tasks and are not
supported by this model.

## Parity check

The optional `sentence-transformers` dependency is needed only for comparison
against the original implementation. After the Usage example above, compare
the published ZeroModels weights with the matching original checkpoint:

```python
import numpy as np
from sentence_transformers import SentenceTransformer as ReferenceSentenceTransformer

texts = ["A man is eating food.", "A man is eating pasta."]
reference = ReferenceSentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
expected = reference.encode(texts, normalize_embeddings=False)
actual = model.encode(texts, tokenizer=tokenizer)
print("Maximum absolute difference:", np.max(np.abs(actual - expected)))
```

Use matching checkpoint names from the models table, compare in float32, and
record Hub commit SHAs for reproducible results. This downloads original
weights for comparison; it does not convert or publish a new checkpoint.
