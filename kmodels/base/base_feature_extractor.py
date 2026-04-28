from typing import Optional

import keras


class BaseFeatureExtractor(keras.layers.Layer):
    """Base class for kmodels audio feature extractors.

    Provides the standard log-mel-style kwargs (sampling rate, FFT /
    hop / mel bin count, chunk length) and derived quantities
    (``n_samples``, ``nb_max_frames``). Subclasses implement
    ``call(raw_speech, sampling_rate=...)`` returning the spectrogram
    tensor.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        chunk_length: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chunk_length = chunk_length
        self.n_samples = sampling_rate * chunk_length
        self.nb_max_frames = self.n_samples // hop_length

    def call(self, raw_speech, sampling_rate: Optional[int] = None):
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "`call(raw_speech, sampling_rate=None)`."
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "chunk_length": self.chunk_length,
            }
        )
        return config
