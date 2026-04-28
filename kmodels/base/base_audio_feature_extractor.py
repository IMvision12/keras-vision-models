import keras


class BaseAudioFeatureExtractor(keras.layers.Layer):
    """Abstract base for kmodels audio feature extractors.

    Subclasses implement ``call(raw_speech, ...)`` returning the
    spectrogram / feature tensor. Concrete subclasses define their own
    constructor kwargs (sampling rate, FFT size, mel bin count, chunk
    length, etc.) and ``get_config`` payload — the base intentionally
    bakes in no defaults.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, raw_speech, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(raw_speech, ...)`."
        )
