"""
audio.py - Audio capture with energy-based VAD.

Supports:
  - Microphone input (any device)
  - WASAPI loopback (system audio from any output device)
    Requires pyaudiowpatch; falls back to plain pyaudio (mic only) if absent.

Emits float32 numpy arrays at 16 kHz when a speech segment is detected.
"""

from __future__ import annotations

import threading
from collections import deque
from math import gcd
from typing import Callable, List, Optional

import numpy as np
import scipy.signal

try:
    import pyaudiowpatch as pyaudio
    HAS_WASAPI = True
except ImportError:
    import pyaudio  # type: ignore
    HAS_WASAPI = False

# ── constants ────────────────────────────────────────────────────────────────
TARGET_SR = 16_000          # Whisper expects 16 kHz
CHUNK_SIZE = 512            # samples at the device's native sample rate
PRE_ROLL_CHUNKS = 8         # 16 kHz chunks kept before speech starts (~512 ms)
TRAILING_SILENCE = 22       # 16 kHz silent chunks before segment ends (~1.4 s)
MIN_SPEECH_CHUNKS = 5       # minimum speech chunks to emit a segment (~320 ms)
DEFAULT_THRESHOLD = 0.008   # RMS energy threshold (float32 normalised to [-1, 1])


class AudioDevice:
    def __init__(
        self,
        index: int,
        name: str,
        sample_rate: float,
        channels: int,
        is_loopback: bool = False,
    ) -> None:
        self.index = index
        self.name = name
        self.sample_rate = int(sample_rate)
        self.channels = min(int(channels), 2)
        self.is_loopback = is_loopback

    def __str__(self) -> str:
        tag = "[Loopback] " if self.is_loopback else "[Mic] "
        return tag + self.name

    def __repr__(self) -> str:
        return f"AudioDevice(index={self.index}, name={self.name!r}, loopback={self.is_loopback})"


class AudioCapture:
    """Captures audio from a device and fires a callback with speech segments."""

    def __init__(self) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.threshold: float = DEFAULT_THRESHOLD

    # ── device enumeration ───────────────────────────────────────────────────

    def list_devices(self) -> List[AudioDevice]:
        devices: List[AudioDevice] = []
        p = pyaudio.PyAudio()
        try:
            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                    if info["maxInputChannels"] > 0:
                        devices.append(
                            AudioDevice(
                                index=i,
                                name=info["name"],
                                sample_rate=info["defaultSampleRate"],
                                channels=info["maxInputChannels"],
                                is_loopback=False,
                            )
                        )
                except Exception:
                    continue

            if HAS_WASAPI:
                try:
                    for info in p.get_loopback_device_info_generator():  # type: ignore[attr-defined]
                        if info["maxInputChannels"] > 0:
                            devices.append(
                                AudioDevice(
                                    index=info["index"],
                                    name=info["name"],
                                    sample_rate=info["defaultSampleRate"],
                                    channels=info["maxInputChannels"],
                                    is_loopback=True,
                                )
                            )
                except Exception:
                    pass
        finally:
            p.terminate()
        return devices

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(
        self,
        device: AudioDevice,
        callback: Callable[[np.ndarray], None],
    ) -> None:
        """Start capture.  *callback* is called from a background thread."""
        self.stop()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, args=(device, callback), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    # ── capture loop ─────────────────────────────────────────────────────────

    def _run(
        self, device: AudioDevice, callback: Callable[[np.ndarray], None]
    ) -> None:
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=device.channels,
                rate=device.sample_rate,
                input=True,
                input_device_index=device.index,
                frames_per_buffer=CHUNK_SIZE,
            )
        except Exception as exc:
            p.terminate()
            return

        pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_CHUNKS)
        speech_buf: List[np.ndarray] = []
        trailing_count = 0
        in_speech = False

        try:
            while self._running:
                raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16)

                # Mix to mono
                if device.channels > 1:
                    chunk = chunk.reshape(-1, device.channels).mean(axis=1).astype(np.int16)

                # Resample to 16 kHz
                chunk_16k = _resample(chunk, device.sample_rate, TARGET_SR)
                chunk_f = chunk_16k.astype(np.float32) / 32768.0

                energy = float(np.sqrt(np.mean(chunk_f ** 2)))

                if energy > self.threshold:
                    if not in_speech:
                        speech_buf = list(pre_roll) + [chunk_f]
                        in_speech = True
                    else:
                        speech_buf.append(chunk_f)
                    trailing_count = 0
                else:
                    if in_speech:
                        speech_buf.append(chunk_f)
                        trailing_count += 1
                        if trailing_count >= TRAILING_SILENCE:
                            net = len(speech_buf) - trailing_count - len(pre_roll)
                            if net >= MIN_SPEECH_CHUNKS:
                                callback(np.concatenate(speech_buf))
                            speech_buf = []
                            trailing_count = 0
                            in_speech = False
                    else:
                        pre_roll.append(chunk_f)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            p.terminate()


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    g = gcd(from_sr, to_sr)
    resampled = scipy.signal.resample_poly(audio, to_sr // g, from_sr // g)
    return resampled.astype(np.int16)
