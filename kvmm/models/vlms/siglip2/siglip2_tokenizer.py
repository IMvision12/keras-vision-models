from typing import Dict, List, Union

import keras
import sentencepiece as spm
from keras import ops


@keras.saving.register_keras_serializable(package="kvmm")
class SigLIP2Tokenizer(keras.Layer):
    """
    SigLIP2 Tokenizer Implementation for Keras with SentencePiece Model Support

    This module provides a pure Keras 3 implementation of the SigLIP2 tokenizer that uses
    a SentencePiece model (.spm file) for tokenization. The tokenizer converts text into
    token IDs that can be processed by the SigLIP2 text encoder.

    This tokenizer uses the SentencePiece library to load and process the .spm model file,
    providing native SentencePiece tokenization with special token handling for beginning
    of sentence (BOS), end of sentence (EOS), and padding (PAD) tokens.

    Args:
        vocab_file (str): Path to the SentencePiece model file (.spm)
        context_length (int, optional): Maximum context length for padding/truncation. Defaults to 64.
        add_bos (bool, optional): Add beginning of sentence token to the result. Defaults to True.
        add_eos (bool, optional): Add end of sentence token to the result. Defaults to True.
        pad_token (str, optional): Padding token used for sequence padding. Defaults to "<pad>".
        bos_token (str, optional): Beginning of sequence token. Defaults to "<bos>".
        eos_token (str, optional): End of sequence token. Defaults to "<eos>".
        unk_token (str, optional): Unknown token for out-of-vocabulary words. Defaults to "<unk>".

    Key features:
    - Native SentencePiece model support (.spm files)
    - Pure Keras 3 integration for neural network pipelines
    - Support for special tokens (BOS, EOS, PAD, UNK)
    - Configurable sequence length with padding/truncation
    - Tensor-based operations for efficient batch processing
    - Automatic vocabulary extraction from SentencePiece model

    Example usage:
        # Initialize the tokenizer with SentencePiece model file
        tokenizer = SigLIP2Tokenizer(
            vocab_file="path/to/model.spm",
            context_length=64,
            add_bos=True,
            add_eos=True
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

    Note:
        This tokenizer requires the `sentencepiece` library to be installed:
        pip install sentencepiece
    """

    def __init__(
        self,
        vocab_file: str,
        context_length: int = 64,
        add_bos: bool = False,
        add_eos: bool = False,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_file = vocab_file
        self.context_length = context_length
        self.add_bos = add_bos
        self.add_eos = add_eos

        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(vocab_file)

        self._build_vocabulary()

        self._get_special_token_ids()

    def _build_vocabulary(self):
        vocab_size = self.sp_model.get_piece_size()

        self.encoder = {}
        self.decoder = {}

        for i in range(vocab_size):
            token = self.sp_model.id_to_piece(i)
            self.encoder[token] = i
            self.decoder[i] = token

    def _get_special_token_ids(self):
        self.pad_token_id = self.encoder.get(self.pad_token, self.sp_model.pad_id())
        self.bos_token_id = self.encoder.get(self.bos_token, self.sp_model.bos_id())
        self.eos_token_id = self.encoder.get(self.eos_token, self.sp_model.eos_id())
        self.unk_token_id = self.encoder.get(self.unk_token, self.sp_model.unk_id())

        if self.pad_token_id == -1:
            self.pad_token_id = self.encoder.get(self.pad_token, 0)
        if self.bos_token_id == -1:
            self.bos_token_id = self.encoder.get(self.bos_token, 1)
        if self.eos_token_id == -1:
            self.eos_token_id = self.encoder.get(self.eos_token, 2)
        if self.unk_token_id == -1:
            self.unk_token_id = self.encoder.get(self.unk_token, 3)

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            if not text:
                return []
            token_ids = self.sp_model.encode_as_ids(text)
            return token_ids
        else:
            all_token_ids = []
            for single_text in text:
                if not single_text:
                    all_token_ids.append([])
                else:
                    token_ids = self.sp_model.encode_as_ids(single_text)
                    all_token_ids.append(token_ids)
            return all_token_ids

    def detokenize(self, token_ids: List[int]) -> str:
        filtered_ids = []
        for token_id in token_ids:
            if token_id not in [
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            ]:
                filtered_ids.append(token_id)

        if filtered_ids:
            text = self.sp_model.decode_ids(filtered_ids)
        else:
            text = ""

        return text

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        result = token_ids[:]

        if self.add_bos and self.bos_token_id is not None and self.bos_token_id != -1:
            if not result or result[0] != self.bos_token_id:
                result = [self.bos_token_id] + result

        if self.add_eos and self.eos_token_id is not None and self.eos_token_id != -1:
            if not result or result[-1] != self.eos_token_id:
                result = result + [self.eos_token_id]

        return result

    def prepare_for_model_tensor(
        self, token_ids_list: List[List[int]]
    ) -> Dict[str, keras.KerasTensor]:
        processed_sequences = []

        for token_ids in token_ids_list:
            token_ids_with_special = self.build_inputs_with_special_tokens(token_ids)

            if len(token_ids_with_special) > self.context_length:
                token_ids_with_special = token_ids_with_special[: self.context_length]

            processed_sequences.append(token_ids_with_special)

        padded_sequences = []
        for seq in processed_sequences:
            padding_length = self.context_length - len(seq)
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

        token_ids = self.build_inputs_with_special_tokens(token_ids)

        if len(token_ids) > self.context_length:
            token_ids = token_ids[: self.context_length]

        padding_length = self.context_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length

        return {"input_ids": token_ids}

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocabulary(self) -> List[str]:
        vocab = []
        for i in range(self.vocab_size):
            vocab.append(self.sp_model.id_to_piece(i))
        return vocab

    def id_to_token(self, id: int) -> str:
        if id >= self.vocab_size or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocab_size - 1}]. Received: {id}"
            )
        return self.sp_model.id_to_piece(id)

    def token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def batch_decode(
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

    def call(self, inputs):
        if inputs is None:
            raise ValueError("No text inputs provided to SigLIP2Tokenizer")

        if isinstance(inputs, str):
            inputs = [inputs]

        if hasattr(inputs, "numpy"):
            inputs = inputs.numpy()
            if inputs.ndim == 0:
                inputs = [inputs.item()]
            else:
                inputs = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in inputs.flatten()
                ]

        all_token_ids = self.tokenize(inputs)
        result = self.prepare_for_model_tensor(all_token_ids)

        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "context_length": self.context_length,
                "add_bos": self.add_bos,
                "add_eos": self.add_eos,
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
