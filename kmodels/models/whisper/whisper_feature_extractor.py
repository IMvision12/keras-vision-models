import numpy as np


def _hann_window(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _mel_filter_bank(
    n_fft: int,
    n_mels: int,
    sample_rate: int = 16000,
    min_hz: float = 0.0,
    max_hz: float = 8000.0,
) -> np.ndarray:
    """Slaney-style mel filter bank matching HF WhisperFeatureExtractor.

    Produces a ``(n_mels, n_fft // 2 + 1)`` matrix. This reproduces the same
    filter bank the original Whisper codebase ships via ``mel_filters.npz``.
    """

    def hz_to_mel(f):
        f_min, f_sp = 0.0, 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        mels = np.where(
            f >= min_log_hz,
            min_log_mel + np.log(f / min_log_hz) / logstep,
            (f - f_min) / f_sp,
        )
        return mels

    def mel_to_hz(m):
        f_min, f_sp = 0.0, 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        freqs = np.where(
            m >= min_log_mel,
            min_log_hz * np.exp(logstep * (m - min_log_mel)),
            f_min + f_sp * m,
        )
        return freqs

    mel_min, mel_max = hz_to_mel(min_hz), hz_to_mel(max_hz)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lower = (fft_freqs - hz_points[i]) / (hz_points[i + 1] - hz_points[i])
        upper = (hz_points[i + 2] - fft_freqs) / (hz_points[i + 2] - hz_points[i + 1])
        filters[i] = np.maximum(0, np.minimum(lower, upper))
        # Slaney normalization: divide by the area of the triangle
        enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
        filters[i] *= enorm
    return filters


class WhisperFeatureExtractor:
    """Mel spectrogram extractor matching HF ``WhisperFeatureExtractor``.

    The defaults reproduce Whisper's original pipeline:
      * sample rate 16 kHz
      * 30-second chunks (480,000 samples), zero-padded
      * STFT: n_fft=400, hop=160, centered reflect padding (librosa-style)
      * Hann window
      * 80-channel Slaney mel filter bank, 0 to 8 kHz
      * log10 magnitude, clamped at ``max - 8.0``, then ``(x + 4) / 4``
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        chunk_length: int = 30,
    ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chunk_length = chunk_length
        self.n_samples = sampling_rate * chunk_length
        self.nb_max_frames = self.n_samples // hop_length
        self.window = _hann_window(n_fft).astype(np.float32)
        self.mel_filters = _mel_filter_bank(n_fft, n_mels, sampling_rate)

    def _stft(self, waveform: np.ndarray) -> np.ndarray:
        # Center-reflect pad (librosa default) then frame + window + rfft.
        pad = self.n_fft // 2
        wav = np.pad(waveform, pad_width=pad, mode="reflect")
        n_frames = 1 + (len(wav) - self.n_fft) // self.hop_length
        frames = np.lib.stride_tricks.as_strided(
            wav,
            shape=(n_frames, self.n_fft),
            strides=(wav.strides[0] * self.hop_length, wav.strides[0]),
        ).copy()
        frames *= self.window
        spec = np.fft.rfft(frames, n=self.n_fft, axis=-1)
        return spec  # shape: (n_frames, n_fft//2+1)

    def _log_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        spec = self._stft(waveform)
        power = (np.abs(spec) ** 2).astype(np.float32)
        mel = power @ self.mel_filters.T  # (n_frames, n_mels)
        mel = mel.T  # (n_mels, n_frames)
        # drop last frame (HF convention: n_frames - 1 = n_samples / hop)
        mel = mel[:, :-1]

        log_spec = np.log10(np.maximum(mel, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.astype(np.float32)

    def __call__(self, raw_speech, sampling_rate: int = 16000):
        """Takes a numpy array or list of numpy arrays (mono, float32 in [-1, 1]).

        Returns a ``(batch, n_mels, nb_max_frames)`` numpy array.
        """
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"WhisperFeatureExtractor expects {self.sampling_rate} Hz input; "
                f"got {sampling_rate} Hz."
            )
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        elif isinstance(raw_speech, (list, tuple)):
            raw_speech = [np.asarray(w, dtype=np.float32) for w in raw_speech]
        else:
            raw_speech = [np.asarray(raw_speech, dtype=np.float32).squeeze()]

        batch = []
        for wav in raw_speech:
            wav = np.asarray(wav, dtype=np.float32)
            if len(wav) < self.n_samples:
                wav = np.pad(wav, (0, self.n_samples - len(wav)))
            else:
                wav = wav[: self.n_samples]
            batch.append(self._log_mel_spectrogram(wav))
        return np.stack(batch, axis=0)
