"""
stt.py - Speech-to-text using faster-whisper.

Loads the model once in a background thread; transcription is thread-safe
via an internal lock so only one segment is processed at a time.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "base.en") -> None:
        self._model_size = model_size
        self._model: Optional[WhisperModel] = None
        self._lock = threading.Lock()
        self.loaded = False
        self.load_error: Optional[str] = None

    def load(self) -> None:
        """Download (if needed) and load the Whisper model.
        Call this from a background thread to avoid blocking the UI."""
        try:
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )
            self.loaded = True
        except Exception as exc:
            self.load_error = str(exc)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono array sampled at 16 kHz.

        Returns an empty string if the model is not loaded yet or if nothing
        meaningful was detected.
        """
        if not self.loaded or self._model is None:
            return ""

        with self._lock:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=1,
                best_of=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            text = " ".join(s.text.strip() for s in segments).strip()

        return text
