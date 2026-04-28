import keras


class BaseTokenizer(keras.layers.Layer):
    """Abstract base for kmodels tokenizers.

    Subclasses must implement ``call`` (text -> ids) and ``decode``
    (ids -> text). ``batch_decode`` is provided as a pure-Python loop
    over ``decode``.

    Concrete tokenizers add their own state (vocab path, merges,
    special-token ids, BPE / SentencePiece backend) and their own
    ``get_config`` payload — the base intentionally bakes in no
    defaults.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
