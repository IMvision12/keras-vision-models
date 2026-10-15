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

## Usage

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.st_models import (
    SentenceTransformer,
    SentenceTransformerTokenizer,
)

repo = "zeromodels/all-MiniLM-L6-v2"
model = SentenceTransformer.from_weights(repo)
tokenizer = SentenceTransformerTokenizer.from_weights(repo)

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

To load directly from an original Sentence Transformers repo instead, use the
`hf:` prefix for both components and pass the model's resolved revision to the
tokenizer:

```python
upstream = "hf:sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer.from_weights(upstream)
tokenizer = SentenceTransformerTokenizer.from_weights(
    upstream, revision=model.source_revision
)
```

An explicit commit SHA can be supplied as `revision=` to pin the upstream
snapshot.

`AutoZMTextEmbed` resolves ZeroModels `sentence_transformer` configs to the
`SentenceTransformerTextEmbed` task view. For upstream `hf:` repos, load the
concrete class as shown above: their root `model_type` describes the backbone,
not the sentence-embedding task.

Unsupported upstream module types, combinations, or pooling modes raise a
`ValueError` rather than silently changing the embedding recipe. Cross-encoders,
sparse encoders, and multi-vector encoders are separate tasks and are not
supported by this model.

## Parity check

The optional `sentence-transformers` dependency is needed only for comparison
against the original implementation:

```bash
python scripts/verify_sentence_transformer_parity.py --model all
```

The script compares embeddings at pinned upstream commits for all 13 rows.
Use `--model <name>` to check one checkpoint. It downloads upstream weights
into the standard Hugging Face cache but does not create or publish converted
releases.
