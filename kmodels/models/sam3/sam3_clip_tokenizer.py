import json
import re

import numpy as np

from kmodels.weight_utils import download_file

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
