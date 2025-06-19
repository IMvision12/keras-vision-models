import json
import re
import string
from typing import Dict, List, Optional, Union

import keras
from keras import ops

@keras.saving.register_keras_serializable(package="kvmm")
class SigLIPTokenizer(keras.Layer):
    """
    SigLIP Tokenizer Implementation for Keras
    
    This module provides a Keras-based implementation of the SigLIP tokenizer used in Google's
    SigLIP (Sigmoid Loss for Language Image Pre-training) model. The tokenizer converts text into
    token IDs that can be processed by the SigLIP text encoder.
    
    The tokenizer implements greedy subword tokenization with SentencePiece-style preprocessing,
    including text canonicalization, punctuation removal, and special token handling. It uses
    a longest-match-first approach for tokenization.
    
    Args:
        vocab_file (str): Path to the vocabulary JSON file containing token-to-ID mappings
        context_length (int, optional): Maximum context length for padding/truncation. Defaults to 64.
        do_lower_case (bool, optional): Whether to convert text to lowercase during preprocessing. Defaults to True.
        unk_token (str, optional): Token for unknown/out-of-vocabulary words. Defaults to "<unk>".
        pad_token (str, optional): Padding token used for sequence padding. Defaults to "</s>".
        eos_token (str, optional): End of sequence token. Defaults to "</s>".
    
    Key features:
    - Greedy longest-match-first subword tokenization
    - SentencePiece-style text preprocessing with underline prefix
    - Text canonicalization including punctuation removal and whitespace normalization
    - Support for special tokens (UNK, PAD, EOS)
    - Configurable case sensitivity
    - Integration with Keras as a layer for seamless use in neural network pipelines
    - Tensor-based operations for efficient batch processing
    
    Text preprocessing pipeline:
    1. Canonicalize text (remove punctuation, normalize whitespace)
    2. Apply lowercase conversion if enabled
    3. Add SentencePiece underline prefix
    4. Perform greedy tokenization using longest-match-first strategy
    5. Handle unknown characters with UNK token
    
    Example usage:
        # Initialize the tokenizer with vocabulary file
        tokenizer = SigLIPTokenizer(
            vocab_file="path/to/vocab.json",
            context_length=64,
            do_lower_case=True
        )
        
        # Tokenize and encode a single text
        text = "A photo of a cat"
        encoded = tokenizer(text)
        
        # Tokenize a batch of texts
        texts = ["A photo of a cat", "A painting of a dog"]
        batch_encoded = tokenizer(texts)
        
        # Decode token IDs back to text
        token_ids = encoded["input_ids"][0]
        decoded_text = tokenizer.decode(token_ids.numpy())
        
        # Get sequence lengths (excluding padding)
        lengths = tokenizer.get_sequence_length(encoded["input_ids"])
        
        # Batch decode multiple sequences
        decoded_texts = tokenizer.batch_decode(encoded["input_ids"])
    
    Note:
        This tokenizer is specifically designed for SigLIP models and may not be compatible
        with other vision-language models. The greedy tokenization approach differs from
        BPE-based tokenizers used in models like CLIP.
    """
    def __init__(
        self,
        vocab_file: str,
        context_length: int = 64,
        do_lower_case: bool = True,
        unk_token: str = "<unk>",
        pad_token: str = "</s>",
        eos_token: str = "</s>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_length = context_length
        self.do_lower_case = do_lower_case

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.unk_token_id = self.encoder.get(self.unk_token, 0)
        self.pad_token_id = self.encoder.get(self.pad_token, 1)
        self.eos_token_id = self.encoder.get(self.eos_token, 1)

        self.spiece_underline = "▁"
        self._build_subword_vocab()

        self._build_token_lookup_tensors()

    def _build_subword_vocab(self):
        self.sorted_tokens = sorted(
            [token for token in self.encoder.keys() if not token.startswith("<")],
            key=len,
            reverse=True
        )
        self.token_set = set(self.encoder.keys())

    def _build_token_lookup_tensors(self):
        vocab_keys = list(self.encoder.keys())
        vocab_values = list(self.encoder.values())

        self.vocab_keys_tensor = ops.convert_to_tensor(vocab_keys, dtype="string")
        self.vocab_values_tensor = ops.convert_to_tensor(vocab_values, dtype="int32")

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def canonicalize_text(self, text: str, keep_punctuation_exact_string: Optional[str] = None) -> str:
        if keep_punctuation_exact_string:
            text = keep_punctuation_exact_string.join(
                self.remove_punctuation(part) for part in text.split(keep_punctuation_exact_string)
            )
        else:
            text = self.remove_punctuation(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def _preprocess_text(self, text: str) -> str:
        text = self.canonicalize_text(text)
        if self.do_lower_case:
            text = text.lower()
        return text

    def _tokenize_greedy(self, text: str) -> List[str]:
        if not text:
            return []
        text = self.spiece_underline + text.replace(self.spiece_underline, " ")
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for token in self.sorted_tokens:
                if text[i:].startswith(token):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                char = text[i]
                if char in self.token_set:
                    tokens.append(char)
                else:
                    if self.unk_token in self.token_set:
                        tokens.append(self.unk_token)
                i += 1

        return tokens

    def tokenize(self, text: str) -> List[str]:
        text = self._preprocess_text(text)
        tokens = self._tokenize_greedy(text)
        tokens = [token for token in tokens if token]
        return tokens

    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            tokens = self.tokenize(text)
            token_ids = [
                self.encoder.get(token, self.unk_token_id)
                for token in tokens
            ]
            return token_ids
        else:
            all_token_ids = []
            for single_text in text:
                tokens = self.tokenize(single_text)
                token_ids = [
                    self.encoder.get(token, self.unk_token_id)
                    for token in tokens
                ]
                all_token_ids.append(token_ids)
            return all_token_ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.decoder.get(token_id, self.unk_token) for token_id in token_ids]
        text = "".join(tokens)
        text = text.replace(self.spiece_underline, " ")
        text = text.strip()
        for special_token in [self.unk_token, self.pad_token, self.eos_token]:
            text = text.replace(special_token, "")

        return text.strip()

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        return token_ids + [self.eos_token_id]

    def prepare_for_model_tensor(self, token_ids_list: List[List[int]]) -> Dict[str, keras.KerasTensor]:
        processed_sequences = []

        for token_ids in token_ids_list:
            token_ids_with_eos = token_ids + [self.eos_token_id]

            if len(token_ids_with_eos) > self.context_length:
                token_ids_with_eos = token_ids_with_eos[:self.context_length]

            processed_sequences.append(token_ids_with_eos)

        max_len = self.context_length
        padded_sequences = []

        for seq in processed_sequences:
            padding_length = max_len - len(seq)
            if padding_length > 0:
                seq_tensor = ops.convert_to_tensor(seq, dtype="int32")
                padded_seq = ops.pad(
                    seq_tensor,
                    [[0, padding_length]],
                    constant_values=self.pad_token_id
                )
                padded_sequences.append(padded_seq)
            else:
                padded_sequences.append(ops.convert_to_tensor(seq, dtype="int32"))

        input_ids = ops.stack(padded_sequences, axis=0)

        return {"input_ids": input_ids}

    def prepare_for_model(self, text: Union[str, List[int]]) -> Dict[str, List[int]]:
        if isinstance(text, str):
            token_ids = self.encode(text)
        else:
            token_ids = text

        token_ids = self.build_inputs_with_special_tokens(token_ids)

        if len(token_ids) > self.context_length:
            token_ids = token_ids[:self.context_length]

        padding_length = self.context_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length

        return {
            "input_ids": token_ids
        }

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def call(self, inputs):
        if inputs is None:
            raise ValueError("No text inputs provided to SigLIPTokenizer")

        if isinstance(inputs, str):
            inputs = [inputs]

        all_token_ids = self.encode(inputs)
        result = self.prepare_for_model_tensor(all_token_ids)

        return result

    def batch_decode(self, token_ids_batch: keras.KerasTensor, skip_special_tokens: bool = True) -> List[str]:
        if hasattr(token_ids_batch, 'numpy'):
            token_ids_batch = token_ids_batch.numpy()

        decoded_texts = []
        for token_ids in token_ids_batch:
            token_ids_list = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)

            if skip_special_tokens:
                token_ids_list = [tid for tid in token_ids_list if tid != self.pad_token_id]

            decoded_text = self.decode(token_ids_list)
            decoded_texts.append(decoded_text)

        return decoded_texts

    def get_sequence_length(self, input_ids: keras.KerasTensor) -> keras.KerasTensor:
        pad_token_tensor = ops.convert_to_tensor(self.pad_token_id, dtype="int32")
        mask = ops.not_equal(input_ids, pad_token_tensor)
        lengths = ops.sum(ops.cast(mask, dtype="int32"), axis=1)
        return lengths

    def truncate_sequences(self, input_ids: keras.KerasTensor, max_length: int) -> keras.KerasTensor:
        if max_length >= input_ids.shape[1]:
            return input_ids
        return input_ids[:, :max_length]

    def get_config(self):
        config = super().get_config()
        config.update({
            "context_length": self.context_length,
            "do_lower_case": self.do_lower_case,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
        })
        return config