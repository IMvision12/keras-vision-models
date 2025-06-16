import json
import os
import re
import unicodedata

import keras


@keras.saving.register_keras_serializable(package="kvmm")
class CLIPTokenizer(keras.Layer):
    """
    CLIP Tokenizer Implementation for Keras

    This module provides a Keras-based implementation of the CLIP tokenizer used in OpenAI's
    CLIP (Contrastive Language-Image Pre-training) model. The tokenizer converts text into
    token IDs that can be processed by the CLIP text encoder.

    The tokenizer implements BPE (Byte-Pair Encoding) tokenization with special handling for
    various text preprocessing steps including Unicode normalization, whitespace cleaning,
    and special token handling.

    Args:
        vocab_file (str): Path to the vocabulary JSON file
        merges_file (str): Path to the BPE merges file
        context_length (int, optional): Maximum context length for padding/truncation. Defaults to 77.
        errors (str, optional): Error handling strategy for decoding. Defaults to "replace".
        unk_token (str, optional): Token for unknown words. Defaults to "<|endoftext|>".
        bos_token (str, optional): Beginning of sequence token. Defaults to "<|startoftext|>".
        eos_token (str, optional): End of sequence token. Defaults to "<|endoftext|>".
        pad_token (str, optional): Padding token. Defaults to "<|endoftext|>".

    Key features:
    - Byte-level BPE tokenization
    - Support for special tokens (BOS, EOS, PAD, UNK)
    - Text preprocessing including Unicode normalization and whitespace cleaning
    - Integration with Keras as a layer for seamless use in neural network pipelines

    Example usage:

        # Initialize the tokenizer with vocabulary and merges files
        tokenizer = CLIPTokenizer(
            vocab_file="path/to/vocab.json",
            merges_file="path/to/merges.txt",
            context_length=77
        )

        # Tokenize and encode a single text
        text = "A photo of a cat"
        encoded = tokenizer(text)

        # Tokenize a batch of texts
        texts = ["A photo of a cat", "A painting of a dog"]
        batch_encoded = tokenizer(texts)

        # Decode token IDs back to text
        token_ids = encoded["input_ids"][0]
        decoded_text = tokenizer.decode(token_ids)
    """

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        context_length: int = 77,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_length = context_length
        self.errors = errors

        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.special_tokens = {
            "<|startoftext|>": 49406,
            "<|endoftext|>": 49407,
            "<|pad|>": 0,
        }

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        if merges_file and os.path.exists(merges_file):
            with open(merges_file, "r", encoding="utf-8") as f:
                bpe_merges = f.read().strip().split("\n")
                if bpe_merges[0].startswith("#version:"):
                    bpe_merges = bpe_merges[1:]
            self.bpe_ranks = dict(
                zip(
                    [tuple(merge.split()) for merge in bpe_merges],
                    range(len(bpe_merges)),
                )
            )
        else:
            self.bpe_ranks = {}

        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }

        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^a-zA-Z0-9\s]+|\s+""",
            re.IGNORECASE,
        )

        self.fix_text = None
        self.use_ftfy = False

        self.bos_token_id = self.special_tokens.get(self.bos_token, 0)
        self.eos_token_id = self.special_tokens.get(self.eos_token, 0)
        self.pad_token_id = self.special_tokens.get(self.pad_token, 0)

    def _bytes_to_unicode(self):
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
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _whitespace_clean(self, text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def _is_punctuation(self, char):
        cp = ord(char)
        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _is_whitespace(self, char):
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _is_chinese_char(self, cp):
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True
        return False

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _whitespace_tokenize(self, text):
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _basic_tokenize(self, text, never_split=None):
        never_split = set(never_split) if never_split else set()
        text = self._clean_text(text)

        text = unicodedata.normalize("NFC", text)

        orig_tokens = self._whitespace_tokenize(text)
        split_tokens = []

        for token in orig_tokens:
            if token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)

            split_tokens.append(token)

        output_tokens = self._whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self._get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
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

            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        if self.use_ftfy:
            text = self._whitespace_clean(self.fix_text(text)).lower()
        else:
            text = " ".join(self._basic_tokenize(text))
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def encode(self, text):
        bpe_tokens = self.tokenize(text)
        token_ids = [
            self.encoder.get(token, self.encoder.get(self.unk_token, 0))
            for token in bpe_tokens
        ]
        return token_ids

    def decode(self, token_ids):
        text = "".join([self.decoder.get(token_id, "") for token_id in token_ids])
        byte_array = bytearray([self.byte_decoder.get(c, ord(c)) for c in text])
        text = (
            byte_array.decode("utf-8", errors=self.errors).replace("</w>", " ").strip()
        )
        return text

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]
        if token_ids_1 is None:
            return bos_token + token_ids_0 + eos_token
        return bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token

    def create_attention_mask(self, token_ids):
        mask = [True] * len(token_ids)
        padding_length = self.context_length - len(mask)
        if padding_length > 0:
            mask = mask + [False] * padding_length
        else:
            mask = mask[: self.context_length]
        return mask

    def prepare_for_model(self, text):
        if isinstance(text, str):
            token_ids = self.encode(text)
        else:
            token_ids = text
        token_ids = self.build_inputs_with_special_tokens(token_ids)
        if len(token_ids) > self.context_length:
            token_ids = token_ids[: self.context_length]
        attention_mask = self.create_attention_mask(token_ids)
        padding_length = self.context_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length
        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def call(self, inputs=None):
        if inputs is None:
            raise ValueError("No text inputs provided to CLIPTokenizer")

        if isinstance(inputs, str):
            inputs = [inputs]

        all_token_ids = []
        all_masks = []

        for text in inputs:
            prepared_input = self.prepare_for_model(text)
            all_token_ids.append(prepared_input["input_ids"])
            all_masks.append(prepared_input["attention_mask"])

        return {
            "input_ids": keras.ops.convert_to_tensor(all_token_ids, dtype="int32"),
            "attention_mask": keras.ops.convert_to_tensor(all_masks, dtype="bool"),
        }
