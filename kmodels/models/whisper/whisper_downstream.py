"""Downstream task wrappers for Whisper.

* :class:`WhisperGenerate` — one-call ASR / translation. Bundles a
  :class:`Whisper` + :class:`WhisperProcessor` and runs greedy
  decoding to text in one shot.
"""

from typing import List, Optional, Union

import numpy as np
from keras import ops

from .config import WHISPER_BEGIN_SUPPRESS_TOKENS, WHISPER_SUPPRESS_TOKENS
from .whisper_model import Whisper
from .whisper_processor import WhisperProcessor


class WhisperGenerate:
    """Convenience wrapper around a :class:`Whisper` + processor.

    Replaces the 6-line ``feature_extractor → get_decoder_prompt_ids →
    greedy decode → batch_decode`` chain with a single callable:

    >>> model = WhisperBase(weights="openai")
    >>> generator = WhisperGenerate(model, processor)
    >>> generator(wave, language="en", task="transcribe")
    ['hello world']

    The wrapper holds no trainable state of its own — it just routes
    arguments to the processor and runs the inlined greedy decoding
    loop against ``model.encoder`` / ``model.decoder``.

    Args:
        model: A :class:`Whisper` instance (typically returned by
            ``Whisper{Tiny,Base,...}()``).
        processor: A :class:`WhisperProcessor` matching the model's
            tokenizer variant + mel bin count.
    """

    def __init__(self, model: Whisper, processor: WhisperProcessor):
        self.encoder = model.encoder
        self.decoder = model.decoder
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

        Mirrors the key logit processors used by HF Whisper generate:

        * ``forced_decoder_ids`` (built by the processor): at decoded
          position ``k``, force the output to a specific id — typically
          ``[(1, lang_id), (2, task_id), (3, 50363)]`` for English
          no-timestamps transcription.
        * ``suppress_tokens``: permanently forbid this set of token ids.
        * ``begin_suppress_tokens``: suppress these only at the very
          first generated step (e.g. blank/silent tokens).

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
            suppress_tokens / begin_suppress_tokens: Lists of token ids
                to mask out. ``None`` keeps OpenAI's defaults.
        """
        inputs = self.processor(audio=audio, sampling_rate=sampling_rate)
        forced = dict(
            self.processor.get_decoder_prompt_ids(
                language=language, task=task, no_timestamps=no_timestamps
            )
        )
        decoder_start_token_id = self.processor.decoder_start_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id

        suppress_set = set(
            suppress_tokens if suppress_tokens is not None else WHISPER_SUPPRESS_TOKENS
        )
        begin_suppress_set = set(
            begin_suppress_tokens
            if begin_suppress_tokens is not None
            else WHISPER_BEGIN_SUPPRESS_TOKENS
        )

        enc_out = self.encoder(inputs["input_features"])
        enc_np = (
            ops.convert_to_numpy(enc_out)
            if not isinstance(enc_out, np.ndarray)
            else enc_out
        )
        batch = enc_np.shape[0]

        generated = np.full((batch, 1), decoder_start_token_id, dtype=np.int32)
        done = np.zeros(batch, dtype=bool)

        for step in range(max_new_tokens):
            cur_pos = generated.shape[1]
            if cur_pos in forced:
                next_ids = np.full((batch,), forced[cur_pos], dtype=np.int32)
            else:
                logits = self.decoder(
                    {
                        "decoder_input_ids": generated,
                        "encoder_hidden_states": enc_np,
                    }
                )
                next_logits = ops.convert_to_numpy(logits)[:, -1, :].copy()
                if suppress_set:
                    next_logits[:, list(suppress_set)] = -1e9
                if step == 0 and begin_suppress_set:
                    next_logits[:, list(begin_suppress_set)] = -1e9
                next_ids = np.argmax(next_logits, axis=-1).astype(np.int32)

            next_ids = np.where(done, eos_token_id, next_ids)
            generated = np.concatenate([generated, next_ids[:, None]], axis=1)
            done = done | (next_ids == eos_token_id)
            if done.all():
                break

        ids = [list(row) for row in generated]
        if return_ids:
            return ids
        return self.processor.batch_decode(ids, skip_special_tokens=True)
