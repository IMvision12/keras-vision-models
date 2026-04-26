"""Downstream task wrappers for Whisper.

* :class:`WhisperGenerate` — one-call ASR / translation. Bundles the
  Whisper encoder + decoder + :class:`WhisperProcessor` and runs greedy
  decoding through :func:`whisper_generate`.
* :class:`WhisperClassify` — encoder + mean pool + linear head for
  audio classification (language id, intent, keyword spotting,
  emotion, ...). Mirrors HuggingFace's
  ``WhisperForAudioClassification``.
"""

from typing import List, Optional, Union

import keras
from keras import layers

from .whisper_model import whisper_generate
from .whisper_processor import WhisperProcessor


class WhisperGenerate:
    """Convenience wrapper around a Whisper bundle + processor.

    Replaces the 6-line ``feature_extractor → get_decoder_prompt_ids →
    whisper_generate → batch_decode`` chain with a single callable:

    >>> model = WhisperBase(weights="openai")
    >>> generator = WhisperGenerate(model, processor)
    >>> generator(wave, language="en", task="transcribe")
    ['hello world']

    The wrapper holds no trainable state of its own — it just routes
    arguments to the processor and the underlying greedy decoder.

    Args:
        model: The bundle dict returned by ``Whisper{Tiny,Base,...}()``,
            i.e. ``{"encoder": ..., "decoder": ..., "config": ...}``.
        processor: A :class:`WhisperProcessor` matching the model's
            tokenizer variant + mel bin count.
    """

    def __init__(self, model: dict, processor: WhisperProcessor):
        self.encoder = model["encoder"]
        self.decoder = model["decoder"]
        self.processor = processor

    def __call__(
        self,
        audio,
        language: Optional[str] = "en",
        task: str = "transcribe",
        no_timestamps: bool = True,
        max_new_tokens: int = 224,
        sampling_rate: int = 16000,
        return_ids: bool = False,
        suppress_tokens: Optional[list] = None,
        begin_suppress_tokens: Optional[list] = None,
    ) -> Union[List[str], List[List[int]]]:
        """Run audio through the full transcription / translation pipeline.

        Args:
            audio: 1-D waveform or list / batched array of waveforms at
                ``sampling_rate`` Hz.
            language: Either a 2-3 char ISO code (``"en"``, ``"fr"``,
                ``"yue"``), the full special token (``"<|en|>"``), or
                ``None`` to let the decoder auto-detect.
            task: ``"transcribe"`` (same-language) or ``"translate"``
                (any language to English).
            no_timestamps: When ``True`` (default), forces the
                ``<|notimestamps|>`` token so the output is raw text.
            max_new_tokens: Maximum decoded tokens after the prompt.
            sampling_rate: Must match the processor's configured rate
                (default ``16000``).
            return_ids: When ``True``, return the raw token-id lists
                instead of decoded strings.
            suppress_tokens / begin_suppress_tokens: Forwarded to
                :func:`whisper_generate`. ``None`` keeps OpenAI's
                defaults.
        """
        inputs = self.processor(audio=audio, sampling_rate=sampling_rate)
        forced = self.processor.get_decoder_prompt_ids(
            language=language, task=task, no_timestamps=no_timestamps
        )
        ids = whisper_generate(
            self.encoder,
            self.decoder,
            inputs["input_features"],
            forced_decoder_ids=forced,
            decoder_start_token_id=self.processor.decoder_start_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
        )
        if return_ids:
            return ids
        return self.processor.batch_decode(ids, skip_special_tokens=True)


def WhisperClassify(
    model: dict,
    num_classes: int,
    projector_dim: Optional[int] = None,
    pooling: str = "mean",
    classifier_dropout: float = 0.0,
    freeze_encoder: bool = False,
    name: str = "whisper_classifier",
) -> keras.Model:
    """Whisper encoder + projector + temporal pool + linear classifier.

    Drop-in replacement for HuggingFace's
    ``WhisperForAudioClassification``. Use it for any task that maps a
    fixed-length audio chunk to a class label — language id, intent,
    keyword spotting, emotion, speaker id, etc. The decoder is
    discarded; only the pretrained encoder is reused.

    The architecture is:

    1. ``encoder(input_features)``  →  ``(B, T, d_model)``
    2. Optional ``Dense(projector_dim)`` projector
    3. Pool over time (``mean`` / ``max`` / ``first``)
    4. Optional dropout
    5. ``Dense(num_classes)`` head → logits

    Returned as a Functional :class:`keras.Model` — supports
    ``compile`` / ``fit`` / ``save_weights`` / ``load_weights`` like
    any other Keras model.

    Args:
        model: The bundle dict returned by ``Whisper{Tiny,Base,...}()``,
            i.e. ``{"encoder": ..., "decoder": ..., "config": ...}``.
            Only the encoder is used; the decoder is discarded.
        num_classes: Output class count.
        projector_dim: When set, insert a ``Dense(projector_dim)``
            layer between the encoder and the pool. ``None`` (default)
            keeps ``d_model`` straight through.
        pooling: ``"mean"`` (default), ``"max"``, or ``"first"``.
        classifier_dropout: Dropout applied to the pooled vector before
            the classification head.
        freeze_encoder: When ``True``, freeze the encoder weights —
            useful for linear-probe baselines.
        name: Model name.
    """
    if pooling not in ("mean", "max", "first"):
        raise ValueError(f"pooling must be 'mean', 'max', or 'first'; got {pooling!r}")

    encoder = model["encoder"]
    if freeze_encoder:
        encoder.trainable = False

    inp = encoder.input
    h = encoder.output  # (B, T, d_model)

    if projector_dim is not None:
        h = layers.Dense(projector_dim, name="projector")(h)

    if pooling == "mean":
        pooled = layers.GlobalAveragePooling1D(
            data_format="channels_last", name="pool"
        )(h)
    elif pooling == "max":
        pooled = layers.GlobalMaxPooling1D(data_format="channels_last", name="pool")(h)
    else:
        pooled = layers.Lambda(lambda x: x[:, 0, :], name="pool")(h)

    if classifier_dropout > 0:
        pooled = layers.Dropout(classifier_dropout, name="classifier_dropout")(pooled)

    logits = layers.Dense(num_classes, name="classifier")(pooled)
    return keras.Model(inputs=inp, outputs=logits, name=name)
