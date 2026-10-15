# Sentence Transformer models

ZeroModels provides 13 converted dense sentence-embedding checkpoints. Each
checkpoint combines a text encoder with its original pooling, optional Dense
projection, and optional L2 normalization. Load them through the
[`zeromodels.st_models` API](sentence_transformer.md); no conversion is needed
for the ZeroModels-hosted weights.

| Checkpoint | Backbone | Embedding recipe | Dimensions |
|---|---|---|---:|
| [`all-mpnet-base-v2`](https://huggingface.co/zeromodels/all-mpnet-base-v2) | MPNet | Mean pooling → L2 normalization | 768 |
| [`multi-qa-mpnet-base-dot-v1`](https://huggingface.co/zeromodels/multi-qa-mpnet-base-dot-v1) | MPNet | CLS pooling; dot-product similarity | 768 |
| [`all-distilroberta-v1`](https://huggingface.co/zeromodels/all-distilroberta-v1) | RoBERTa | Mean pooling → L2 normalization | 768 |
| [`all-MiniLM-L12-v2`](https://huggingface.co/zeromodels/all-MiniLM-L12-v2) | BERT | Mean pooling → L2 normalization | 384 |
| [`multi-qa-distilbert-cos-v1`](https://huggingface.co/zeromodels/multi-qa-distilbert-cos-v1) | DistilBERT | Mean pooling → L2 normalization | 768 |
| [`all-MiniLM-L6-v2`](https://huggingface.co/zeromodels/all-MiniLM-L6-v2) | BERT | Mean pooling → L2 normalization | 384 |
| [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/zeromodels/multi-qa-MiniLM-L6-cos-v1) | BERT | Mean pooling → L2 normalization | 384 |
| [`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/zeromodels/paraphrase-multilingual-mpnet-base-v2) | XLM-RoBERTa | Mean pooling | 768 |
| [`paraphrase-albert-small-v2`](https://huggingface.co/zeromodels/paraphrase-albert-small-v2) | ALBERT | Mean pooling | 768 |
| [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/zeromodels/paraphrase-multilingual-MiniLM-L12-v2) | BERT | Mean pooling | 384 |
| [`paraphrase-MiniLM-L3-v2`](https://huggingface.co/zeromodels/paraphrase-MiniLM-L3-v2) | BERT | Mean pooling | 384 |
| [`distiluse-base-multilingual-cased-v1`](https://huggingface.co/zeromodels/distiluse-base-multilingual-cased-v1) | DistilBERT | Mean pooling → Tanh Dense | 512 |
| [`distiluse-base-multilingual-cased-v2`](https://huggingface.co/zeromodels/distiluse-base-multilingual-cased-v2) | DistilBERT | Mean pooling → Tanh Dense | 512 |

The `paraphrase-multilingual-mpnet-base-v2` name is historical; the checkpoint
actually uses an XLM-RoBERTa backbone.

## Quick start

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.st_models import SentenceTransformer, SentenceTransformerTokenizer

repo = "zeromodels/all-MiniLM-L6-v2"
model = SentenceTransformer.from_weights(repo)
tokenizer = SentenceTransformerTokenizer.from_weights(repo)

embeddings = model.encode(["A cat on a mat.", "A dog on a mat."], tokenizer=tokenizer)
print(embeddings.shape)  # (2, 384)
```

See [Sentence Transformer](sentence_transformer.md) for the complete API,
upstream `hf:` loading, similarity scoring, and parity-check instructions.
