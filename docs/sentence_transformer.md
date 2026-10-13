# Sentence Transformer

ZeroModels loads the 13 original dense sentence-embedding checkpoints shown in
the [Sentence Transformers pretrained-model table](https://sbert.net/docs/sentence_transformer/pretrained_models.html).
It builds a Keras 3 encoder and applies the checkpoint's declared pooling,
optional Dense projection, and optional L2 normalization. Parity has been
verified on the Keras Torch backend; TensorFlow and JAX parity have not yet
been checked for all 13 checkpoints.

These are **upstream** Hugging Face repos: use the `hf:` prefix below. No
conversion or upload is needed for upstream loading.

## Supported upstream checkpoints

| Checkpoint (`sentence-transformers/<name>`) | Actual backbone | Embedding recipe | Dim |
|---|---|---|---:|
| `all-mpnet-base-v2` | MPNet | mean → normalize | 768 |
| `multi-qa-mpnet-base-dot-v1` | MPNet | CLS, dot-product similarity | 768 |
| `all-distilroberta-v1` | RoBERTa | mean → normalize | 768 |
| `all-MiniLM-L12-v2` | BERT | mean → normalize | 384 |
| `multi-qa-distilbert-cos-v1` | DistilBERT | mean → normalize | 768 |
| `all-MiniLM-L6-v2` | BERT | mean → normalize | 384 |
| `multi-qa-MiniLM-L6-cos-v1` | BERT | mean → normalize | 384 |
| `paraphrase-multilingual-mpnet-base-v2` | XLM-RoBERTa | mean | 768 |
| `paraphrase-albert-small-v2` | ALBERT | mean | 768 |
| `paraphrase-multilingual-MiniLM-L12-v2` | BERT | mean | 384 |
| `paraphrase-MiniLM-L3-v2` | BERT | mean | 384 |
| `distiluse-base-multilingual-cased-v1` | DistilBERT | mean → Tanh Dense | 512 |
| `distiluse-base-multilingual-cased-v2` | DistilBERT | mean → Tanh Dense | 512 |

The name `paraphrase-multilingual-mpnet-base-v2` is historical: its actual
checkpoint `model_type` is `xlm-roberta`, and ZeroModels uses the XLM-R backbone.

## Usage

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.sentence_transformer import (
    SentenceTransformerModel,
    SentenceTransformerTokenizer,
)

repo = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformerModel.from_weights(f"hf:{repo}")
tokenizer = SentenceTransformerTokenizer.from_weights(
    f"hf:{repo}", revision=model.source_revision
)

embeddings = model.encode(
    ["A man is eating food.", "A man is eating pasta."],
    tokenizer=tokenizer,
)
print(embeddings.shape)  # (2, 384)
print(model.similarity(embeddings[:1], embeddings[1:]))
```

`model(tokenizer(texts))` returns `embeddings` and `token_embeddings` tensors.
`encode()` batches raw strings and returns a NumPy array. Use the same
checkpoint and revision for the model and tokenizer. An explicit commit SHA can
be supplied as `revision=` to pin the upstream snapshot. `model.similarity()`
uses the checkpoint's declared cosine or dot-product scoring function.

`AutoZMTextEmbed` resolves ZeroModels `sentence_transformer` configs to the
`SentenceTransformerTextEmbed` task view. For upstream `hf:` repos, load the
concrete class as shown above: their
root `model_type` describes the backbone, not the sentence-embedding task.

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
