import json
import re

import keras
import numpy as np
from keras import layers, ops

VOCAB_URL = (
    "https://github.com/IMvision12/keras-models/releases/download/clip/vocab.json"
)
MERGES_URL = (
    "https://github.com/IMvision12/keras-models/releases/download/clip/merges.txt"
)

SAM3_CONTEXT_LENGTH = 32
SAM3_VOCAB_SIZE = 49408
SAM3_BOS_TOKEN_ID = 49406
SAM3_EOS_TOKEN_ID = 49407
SAM3_PAD_TOKEN_ID = 49407


def _download_file(url):
    from kmodels.utils import download_file

    return download_file(url)


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BPE_PAT = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s\w]|[\w]+|\s+""",
    re.IGNORECASE,
)


class SAM3CLIPTokenizer:
    """BPE tokenizer for SAM3's CLIP text encoder (context_length=32).

    Pure Python — no HuggingFace dependency. Auto-downloads vocab and merges
    on first use.

    Args:
        vocab_file: Path to vocab.json. If None, auto-downloads.
        merges_file: Path to merges.txt. If None, auto-downloads.
        context_length: Max sequence length (default 32 for SAM3).

    Usage:
        tokenizer = SAM3CLIPTokenizer()
        input_ids, attention_mask = tokenizer.encode("a cat")
        # input_ids: (1, 32) int32, attention_mask: (1, 32) float32

        # Batch:
        input_ids, mask = tokenizer.encode(["cat", "dog"])
        # input_ids: (2, 32), mask: (2, 32)
    """

    def __init__(
        self, vocab_file=None, merges_file=None, context_length=SAM3_CONTEXT_LENGTH
    ):
        self.context_length = context_length
        self.bos_token_id = SAM3_BOS_TOKEN_ID
        self.eos_token_id = SAM3_EOS_TOKEN_ID
        self.pad_token_id = SAM3_PAD_TOKEN_ID

        if vocab_file is None:
            vocab_file = _download_file(VOCAB_URL)
        if merges_file is None:
            merges_file = _download_file(MERGES_URL)

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            if lines[0].startswith("#version:"):
                lines = lines[1:]
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(lines)}

        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }

    def _get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        pairs = self._get_pairs(word)
        if not pairs:
            result = token + "</w>"
            self.cache[token] = result
            return result
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def _tokenize_text(self, text):
        text = re.sub(r"\s+", " ", text.strip())
        ids = []
        for token in re.findall(_BPE_PAT, text):
            if not token.strip():
                continue
            encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            for bpe_tok in self._bpe(encoded).split(" "):
                if bpe_tok and bpe_tok in self.encoder:
                    ids.append(self.encoder[bpe_tok])
        return ids

    def encode(self, text):
        """Tokenize text and return padded input_ids + attention_mask.

        Args:
            text: str or list of str.

        Returns:
            input_ids: numpy array (batch, context_length) int32.
            attention_mask: numpy array (batch, context_length) float32.
        """
        if isinstance(text, str):
            text = [text]

        all_ids = []
        all_masks = []
        for t in text:
            token_ids = (
                [self.bos_token_id] + self._tokenize_text(t) + [self.eos_token_id]
            )
            if len(token_ids) > self.context_length:
                token_ids = token_ids[: self.context_length]
            n_valid = len(token_ids)
            pad_len = self.context_length - n_valid
            token_ids = token_ids + [self.pad_token_id] * pad_len
            mask = [1.0] * n_valid + [0.0] * pad_len
            all_ids.append(token_ids)
            all_masks.append(mask)

        return (
            np.array(all_ids, dtype=np.int32),
            np.array(all_masks, dtype=np.float32),
        )

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = [
            self.decoder.get(tid, "")
            for tid in token_ids
            if tid not in (self.bos_token_id, self.eos_token_id, self.pad_token_id)
        ]
        text = "".join(tokens)
        try:
            raw = bytearray(self.byte_decoder.get(c, ord(c)) for c in text)
            return raw.decode("utf-8", errors="replace").replace("</w>", " ").strip()
        except Exception:
            return ""


SAM3_CLIP_HIDDEN_SIZE = 1024
SAM3_CLIP_NUM_LAYERS = 24
SAM3_CLIP_NUM_HEADS = 16
SAM3_CLIP_INTERMEDIATE_SIZE = 4096
SAM3_CLIP_MAX_POSITION = 32


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3CLIPAttention(layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        dim = self.hidden_size
        self.q_proj = layers.Dense(dim, name="q_proj")
        self.q_proj.build((None, None, dim))
        self.k_proj = layers.Dense(dim, name="k_proj")
        self.k_proj.build((None, None, dim))
        self.v_proj = layers.Dense(dim, name="v_proj")
        self.v_proj.build((None, None, dim))
        self.o_proj = layers.Dense(dim, name="o_proj")
        self.o_proj.build((None, None, dim))
        self.built = True

    def call(self, hidden_states, attention_mask=None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape = ops.shape(q)
        k_shape = ops.shape(k)
        q = ops.reshape(
            q, (q_shape[0], q_shape[1], self.num_attention_heads, self.head_dim)
        )
        k = ops.reshape(
            k, (k_shape[0], k_shape[1], self.num_attention_heads, self.head_dim)
        )
        v = ops.reshape(
            v, (k_shape[0], k_shape[1], self.num_attention_heads, self.head_dim)
        )

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = ops.softmax(attn_weights, axis=-1)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (q_shape[0], q_shape[1], self.hidden_size)
        )
        return self.o_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3CLIPEncoderLayer(layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

    def build(self, input_shape):
        dim = self.hidden_size
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm1")
        self.layer_norm1.build((None, None, dim))
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm2")
        self.layer_norm2.build((None, None, dim))
        self.self_attn = SAM3CLIPAttention(
            dim, self.num_attention_heads, name="self_attn"
        )
        self.self_attn.build((None, None, dim))
        self.fc1 = layers.Dense(self.intermediate_size, name="fc1")
        self.fc1.build((None, None, dim))
        self.fc2 = layers.Dense(dim, name="fc2")
        self.fc2.build((None, None, self.intermediate_size))
        self.built = True

    def call(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = ops.nn.gelu(hidden_states, approximate=False)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3CLIPTextEncoder(layers.Layer):
    """Full CLIP text encoder: token_embed + pos_embed + 24 layers + LN.

    Returns last_hidden_state (batch, seq_len, 1024).
    """

    def __init__(
        self,
        vocab_size=SAM3_VOCAB_SIZE,
        hidden_size=SAM3_CLIP_HIDDEN_SIZE,
        num_hidden_layers=SAM3_CLIP_NUM_LAYERS,
        num_attention_heads=SAM3_CLIP_NUM_HEADS,
        intermediate_size=SAM3_CLIP_INTERMEDIATE_SIZE,
        max_position_embeddings=SAM3_CLIP_MAX_POSITION,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

    def build(self, input_shape):
        self.token_embedding = layers.Embedding(
            self.vocab_size, self.hidden_size, name="token_embedding"
        )
        self.token_embedding.build((None, self.max_position_embeddings))
        self.position_embedding = layers.Embedding(
            self.max_position_embeddings, self.hidden_size, name="position_embedding"
        )
        self.position_embedding.build((None, self.max_position_embeddings))
        self.encoder_layers = []
        for i in range(self.num_hidden_layers):
            layer = SAM3CLIPEncoderLayer(
                self.hidden_size,
                self.num_attention_heads,
                self.intermediate_size,
                name=f"layers_{i}",
            )
            layer.build((None, None, self.hidden_size))
            self.encoder_layers.append(layer)
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="final_layer_norm"
        )
        self.final_layer_norm.build((None, None, self.hidden_size))
        self.built = True

    def call(self, input_ids, attention_mask=None):
        seq_len = ops.shape(input_ids)[1]
        hidden_states = self.token_embedding(input_ids)
        position_ids = ops.arange(seq_len, dtype="int32")
        hidden_states = hidden_states + self.position_embedding(position_ids)

        causal_mask = ops.triu(ops.full((seq_len, seq_len), -1e9), k=1)
        if attention_mask is not None:
            pad_mask = ops.cast(
                1.0 - ops.cast(attention_mask, "float32"), "float32"
            ) * (-1e9)
            pad_mask = ops.reshape(pad_mask, (-1, 1, 1, seq_len))
            combined_mask = causal_mask + pad_mask
        else:
            combined_mask = causal_mask

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=combined_mask)

        return self.final_layer_norm(hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "max_position_embeddings": self.max_position_embeddings,
            }
        )
        return config


def build_text_encoder(weights_path=None):
    """Build a standalone Keras model for SAM3's CLIP text encoder.

    Args:
        weights_path: Optional path to .weights.h5 file.

    Returns:
        keras.Model with inputs {input_ids, attention_mask} and output (B, 32, 1024).
    """
    input_ids = keras.Input(
        shape=(SAM3_CONTEXT_LENGTH,), dtype="int32", name="input_ids"
    )
    attention_mask = keras.Input(
        shape=(SAM3_CONTEXT_LENGTH,), dtype="int32", name="attention_mask"
    )

    encoder = SAM3CLIPTextEncoder(name="text_encoder")
    encoder.build((None, SAM3_CONTEXT_LENGTH))
    output = encoder(input_ids, attention_mask=attention_mask)

    model = keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=output,
        name="SAM3TextEncoder",
    )

    if weights_path is not None:
        model.load_weights(weights_path)

    return model
