import re
import string
from typing import Dict, List, Optional, Union

import keras
import sentencepiece as spm
from keras import ops


@keras.saving.register_keras_serializable(package="kvmm")
class SigLIPTokenizer(keras.Layer):
    """
    SigLIP Tokenizer Implementation for Keras

    This module provides a Keras-based implementation of the SigLIP tokenizer used in Google's
    SigLIP (Sigmoid Loss for Language Image Pre-training) model. The tokenizer converts text into
    token IDs that can be processed by the SigLIP text encoder.

    The tokenizer uses SentencePiece for subword tokenization with minimal text preprocessing
    to match the official SigLIP implementation.

    Args:
        vocab_file (str): Path to the SentencePiece model file (.model format)
        context_length (int, optional): Maximum context length for padding/truncation. Defaults to 64.
        do_lower_case (bool, optional): Whether to convert text to lowercase during preprocessing. Defaults to False.
        unk_token (str, optional): Token for unknown/out-of-vocabulary words. Defaults to "<unk>".
        pad_token (str, optional): Padding token used for sequence padding. Defaults to "</s>".
        eos_token (str, optional): End of sequence token. Defaults to "</s>".

    Key features:
    - SentencePiece-based subword tokenization
    - Minimal text preprocessing (only whitespace normalization)
    - Preserves capitalization and punctuation by default
    - Support for special tokens (UNK, PAD, EOS)
    - Integration with Keras as a layer for seamless use in neural network pipelines
    - Tensor-based operations for efficient batch processing

    Text preprocessing pipeline:
    1. Normalize whitespace (multiple spaces → single space, strip)
    2. Apply lowercase conversion if enabled (disabled by default)
    3. Use SentencePiece encoding directly
    4. Add EOS token

    Example usage:
        # Initialize the tokenizer with SentencePiece model file
        tokenizer = SigLIPTokenizer(
            vocab_file="path/to/vocab.model",
            context_length=64,
            do_lower_case=False  # Default behavior preserves case and punctuation
        )

        # Tokenize and encode a single text
        text = "A photo of a cat"
        encoded = tokenizer(text)

        # Tokenize a batch of texts
        texts = ["A photo of a cat", "A painting of a dog"]
        batch_encoded = tokenizer(texts)

        # Decode token IDs back to text
        token_ids = encoded["input_ids"][0]
        decoded_text = tokenizer.detokenize(token_ids.numpy())

        # Get sequence lengths (excluding padding)
        lengths = tokenizer.get_sequence_length(encoded["input_ids"])

        # Batch decode multiple sequences
        decoded_texts = tokenizer.batch_detokenize(encoded["input_ids"])

    Note:
        This tokenizer is specifically designed for SigLIP models and matches the official
        SigLIP tokenization behavior. It preserves capitalization and punctuation by default,
        unlike some other vision-language model tokenizers.
    """

    def __init__(
        self,
        vocab_file: str,
        context_length: int = 64,
        do_lower_case: bool = False,
        unk_token: str = "<unk>",
        pad_token: str = "</s>",
        eos_token: str = "</s>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file
        self.context_length = context_length
        self.do_lower_case = do_lower_case

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(vocab_file)

        self.encoder = {
            self.sp_model.id_to_piece(i): i
            for i in range(self.sp_model.get_piece_size())
        }
        self.decoder = {
            i: self.sp_model.id_to_piece(i)
            for i in range(self.sp_model.get_piece_size())
        }

        self.unk_token_id = self.sp_model.piece_to_id(self.unk_token)
        self.pad_token_id = self.sp_model.piece_to_id(self.pad_token)
        self.eos_token_id = self.sp_model.piece_to_id(self.eos_token)

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def canonicalize_text(
        self, text: str, keep_punctuation_exact_string: Optional[str] = None
    ) -> str:
        if keep_punctuation_exact_string:
            text = keep_punctuation_exact_string.join(
                self.remove_punctuation(part)
                for part in text.split(keep_punctuation_exact_string)
            )
        else:
            text = self.remove_punctuation(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            processed_text = self._preprocess_text(text)
            token_ids = self.sp_model.encode_as_ids(processed_text)
            token_ids.append(self.eos_token_id)

            return token_ids
        else:
            all_token_ids = []
            for single_text in text:
                processed_text = self._preprocess_text(single_text)
                token_ids = self.sp_model.encode_as_ids(processed_text)
                token_ids.append(self.eos_token_id)
                all_token_ids.append(token_ids)
            return all_token_ids

    def detokenize(
        self,
        token_ids: Union[List[int], List[List[int]], keras.KerasTensor],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        if hasattr(token_ids, "numpy"):
            token_ids = token_ids.numpy()

        if (
            isinstance(token_ids, list)
            and len(token_ids) > 0
            and isinstance(token_ids[0], list)
        ):
            decoded_texts = []
            for seq_token_ids in token_ids:
                if hasattr(seq_token_ids, "tolist"):
                    seq_token_ids = seq_token_ids.tolist()

                if skip_special_tokens:
                    special_token_ids = {
                        self.pad_token_id,
                        self.eos_token_id,
                        self.unk_token_id,
                    }
                    seq_token_ids = [
                        tid for tid in seq_token_ids if tid not in special_token_ids
                    ]

                decoded_text = self.sp_model.decode_ids(seq_token_ids)
                decoded_texts.append(decoded_text.strip())

            return decoded_texts
        elif hasattr(token_ids, "ndim") and token_ids.ndim == 2:
            decoded_texts = []
            for seq_token_ids in token_ids:
                if hasattr(seq_token_ids, "tolist"):
                    seq_token_ids = seq_token_ids.tolist()

                if skip_special_tokens:
                    special_token_ids = {
                        self.pad_token_id,
                        self.eos_token_id,
                        self.unk_token_id,
                    }
                    seq_token_ids = [
                        tid for tid in seq_token_ids if tid not in special_token_ids
                    ]

                decoded_text = self.sp_model.decode_ids(seq_token_ids)
                decoded_texts.append(decoded_text.strip())

            return decoded_texts
        else:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()

            if skip_special_tokens:
                special_token_ids = {
                    self.pad_token_id,
                    self.eos_token_id,
                    self.unk_token_id,
                }
                token_ids = [tid for tid in token_ids if tid not in special_token_ids]

            decoded_text = self.sp_model.decode_ids(token_ids)
            return decoded_text.strip()

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        return token_ids + [self.eos_token_id]

    def prepare_for_model_tensor(
        self, token_ids_list: List[List[int]]
    ) -> Dict[str, keras.KerasTensor]:
        processed_sequences = []

        for token_ids in token_ids_list:
            if len(token_ids) >= self.context_length:
                token_ids_processed = token_ids[: self.context_length]
            else:
                token_ids_processed = token_ids

            processed_sequences.append(token_ids_processed)

        max_len = self.context_length
        padded_sequences = []

        for seq in processed_sequences:
            padding_length = max_len - len(seq)
            if padding_length > 0:
                seq_tensor = ops.convert_to_tensor(seq, dtype="int32")
                padded_seq = ops.pad(
                    seq_tensor, [[0, padding_length]], constant_values=self.pad_token_id
                )
                padded_sequences.append(padded_seq)
            else:
                padded_sequences.append(ops.convert_to_tensor(seq, dtype="int32"))

        input_ids = ops.stack(padded_sequences, axis=0)

        return {"input_ids": input_ids}

    def prepare_for_model(self, text: Union[str, List[int]]) -> Dict[str, List[int]]:
        if isinstance(text, str):
            token_ids = self.tokenize(text)
        else:
            token_ids = text

        if len(token_ids) >= self.context_length:
            token_ids = token_ids[: self.context_length]
        else:
            if len(token_ids) > self.context_length:
                token_ids = token_ids[: self.context_length]

        padding_length = self.context_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length

        return {"input_ids": token_ids}

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def call(self, inputs):
        if inputs is None:
            raise ValueError("No text inputs provided to SigLIPTokenizer")

        if isinstance(inputs, str):
            inputs = [inputs]

        all_token_ids = self.tokenize(inputs)
        result = self.prepare_for_model_tensor(all_token_ids)

        return result

    def batch_detokenize(
        self, token_ids_batch: keras.KerasTensor, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()

        decoded_texts = []
        for token_ids in token_ids_batch:
            token_ids_list = (
                token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
            )

            if skip_special_tokens:
                token_ids_list = [
                    tid for tid in token_ids_list if tid != self.pad_token_id
                ]

            decoded_text = self.detokenize(token_ids_list)
            decoded_texts.append(decoded_text)

        return decoded_texts

    def get_sequence_length(self, input_ids: keras.KerasTensor) -> keras.KerasTensor:
        pad_token_tensor = ops.convert_to_tensor(self.pad_token_id, dtype="int32")
        mask = ops.not_equal(input_ids, pad_token_tensor)
        lengths = ops.sum(ops.cast(mask, dtype="int32"), axis=1)
        return lengths

    def truncate_sequences(
        self, input_ids: keras.KerasTensor, max_length: int
    ) -> keras.KerasTensor:
        if max_length >= input_ids.shape[1]:
            return input_ids
        return input_ids[:, :max_length]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "context_length": self.context_length,
                "do_lower_case": self.do_lower_case,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "eos_token": self.eos_token,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
