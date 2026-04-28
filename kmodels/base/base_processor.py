import keras


class BaseProcessor(keras.layers.Layer):
    """Base class for kmodels multi-modal processors.

    Multi-modal processors compose a :class:`BaseTokenizer` and a
    :class:`BaseImageProcessor` / :class:`BaseFeatureExtractor` into
    one callable. Subclasses set ``self.tokenizer`` /
    ``self.image_processor`` / ``self.feature_extractor`` in
    ``__init__`` and implement ``call`` to dispatch over their
    component(s). ``decode`` / ``batch_decode`` are wired through to
    the tokenizer.
    """

    tokenizer = None
    image_processor = None
    feature_extractor = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement `call`.")

    def decode(self, *args, **kwargs) -> str:
        if self.tokenizer is None:
            raise AttributeError(
                f"{type(self).__name__}.decode() requires `self.tokenizer` to be set."
            )
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        if self.tokenizer is None:
            raise AttributeError(
                f"{type(self).__name__}.batch_decode() requires "
                "`self.tokenizer` to be set."
            )
        return self.tokenizer.batch_decode(*args, **kwargs)
