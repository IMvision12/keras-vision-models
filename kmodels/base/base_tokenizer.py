from typing import Optional

import keras


class BaseTokenizer(keras.layers.Layer):
    """Base class for kmodels tokenizers.

    Wraps a Rust / SentencePiece / BPE backend in a ``keras.layers.Layer``
    so it slots into the Keras 3 serialization registry alongside the
    rest of kmodels. Subclasses must implement ``call`` (text -> ids)
    and ``decode`` (ids -> text); ``batch_decode`` is provided as a
    pure-Python loop over ``decode``.

    Special-token ids (``bos_token_id``, ``eos_token_id``,
    ``pad_token_id``) live on the base so any caller can read them
    without knowing the concrete tokenizer class.
    """

    def __init__(
        self,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def call(self, inputs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(inputs)`."
        )

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement `decode(ids, skip_special_tokens)`."
        )

    def batch_decode(self, ids_batch, skip_special_tokens: bool = True):
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
        )
        return config
