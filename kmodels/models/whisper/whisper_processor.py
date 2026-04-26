from typing import List, Optional, Union

import keras

from .whisper_feature_extractor import WhisperFeatureExtractor
from .whisper_tokenizer import WhisperTokenizer


@keras.saving.register_keras_serializable(package="kmodels")
class WhisperProcessor(keras.layers.Layer):
    """Combined audio + text processor for Whisper.

    Wraps :class:`WhisperFeatureExtractor` and :class:`WhisperTokenizer`,
    matching HuggingFace's ``WhisperProcessor`` API surface. Use it for
    every input/output transform you need around a Whisper model:

    * ``processor(audio=..., sampling_rate=16000)`` — log-mel features.
    * ``processor(text=...)`` — BPE token ids + attention mask
      (label path during fine-tuning).
    * ``processor.get_decoder_prompt_ids(language, task, no_timestamps)``
      — turns ``("en", "transcribe")`` into the
      ``forced_decoder_ids`` list that
      :class:`kmodels.models.whisper.WhisperGenerate` consumes.
    * ``processor.decode`` / ``processor.batch_decode`` — proxy to the
      tokenizer.

    Token ids for languages, tasks and ``<|notimestamps|>`` are looked
    up from the tokenizer's loaded ``added_tokens.json`` rather than
    hard-coded, so the same processor works for both v1 (51 865 vocab)
    and v3 (51 866 vocab) — those vocabularies disagree on the offsets
    of every special token after the extra Cantonese language id.

    Args:
        variant: Tokenizer variant — ``"v1"`` (tiny / base / small /
            medium / large / large-v2) or ``"v3"`` (large-v3 /
            large-v3-turbo).
        n_mels: Number of mel bins. ``80`` for v1 variants, ``128`` for
            large-v3 / large-v3-turbo.
        sampling_rate / n_fft / hop_length / chunk_length: Forwarded to
            :class:`WhisperFeatureExtractor`.
        vocab_file / merges_file / added_tokens_file: Forwarded to
            :class:`WhisperTokenizer`.
    """

    def __init__(
        self,
        variant: str = "v1",
        n_mels: int = 80,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_length: int = 30,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        added_tokens_file: Optional[str] = None,
        bos_token_id: int = 50257,
        eos_token_id: int = 50257,
        pad_token_id: int = 50257,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.feature_extractor = WhisperFeatureExtractor(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            chunk_length=chunk_length,
        )
        self.tokenizer = WhisperTokenizer(
            variant=variant,
            vocab_file=vocab_file,
            merges_file=merges_file,
            added_tokens_file=added_tokens_file,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    @property
    def added_tokens(self) -> dict:
        return self.tokenizer.added_tokens

    def _special_id(self, token: str) -> int:
        try:
            return int(self.added_tokens[token])
        except KeyError as e:
            raise KeyError(
                f"Special token {token!r} not found in Whisper "
                f"{self.variant!r} added_tokens.json"
            ) from e

    @property
    def decoder_start_token_id(self) -> int:
        return self._special_id("<|startoftranscript|>")

    def get_decoder_prompt_ids(
        self,
        language: Optional[str] = "en",
        task: str = "transcribe",
        no_timestamps: bool = True,
    ) -> List[tuple]:
        """Build the ``forced_decoder_ids`` list for ``WhisperGenerate``.

        Mirrors HF ``WhisperProcessor.get_decoder_prompt_ids``. The
        returned list is keyed on the **decoded position** (1, 2, 3) —
        position 0 is always ``<|startoftranscript|>`` and is set
        separately via ``decoder_start_token_id``.

        Args:
            language: Either a 2-3 char ISO code (``"en"``, ``"zh"``,
                ``"yue"``) or the full special token (``"<|en|>"``).
                Pass ``None`` to skip the language slot (e.g. for the
                auto-detect path).
            task: ``"transcribe"`` or ``"translate"``.
            no_timestamps: When ``True``, append the
                ``<|notimestamps|>`` token at position 3 so the decoder
                emits raw text without timestamp tokens.
        """
        if task not in ("transcribe", "translate"):
            raise ValueError(f"task must be 'transcribe' or 'translate', got {task!r}")

        prompt = []
        pos = 1
        if language is not None:
            tok = language if language.startswith("<|") else f"<|{language}|>"
            prompt.append((pos, self._special_id(tok)))
            pos += 1
        prompt.append((pos, self._special_id(f"<|{task}|>")))
        pos += 1
        if no_timestamps:
            prompt.append((pos, self._special_id("<|notimestamps|>")))
        return prompt

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )

    def call(
        self,
        audio=None,
        text: Union[str, List[str], None] = None,
        sampling_rate: int = 16000,
    ):
        if audio is None and text is None:
            raise ValueError(
                "At least one of 'audio' or 'text' must be provided to WhisperProcessor"
            )

        out = {}
        if audio is not None:
            out["input_features"] = self.feature_extractor(
                audio, sampling_rate=sampling_rate
            )
        if text is not None:
            tok_out = self.tokenizer(inputs=text)
            out["input_ids"] = tok_out["input_ids"]
            out["attention_mask"] = tok_out["attention_mask"]
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "n_mels": self.feature_extractor.n_mels,
                "sampling_rate": self.feature_extractor.sampling_rate,
                "n_fft": self.feature_extractor.n_fft,
                "hop_length": self.feature_extractor.hop_length,
                "chunk_length": self.feature_extractor.chunk_length,
                "vocab_file": self.tokenizer.vocab_file,
                "merges_file": self.tokenizer.merges_file,
                "added_tokens_file": self.tokenizer.added_tokens_file,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
        )
        return config
