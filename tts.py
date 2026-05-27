"""tts.py — Text-to-speech via edge-tts (Microsoft Neural voices) + pygame.mixer."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time

# ── optional dependencies ──────────────────────────────────────────────────────
try:
    import edge_tts as _edge
    import pygame.mixer as _mixer
    _mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

# Authoritative ATC voice — clear, neutral US male
ATC_VOICE = "en-US-GuyNeural"

_lock = threading.Lock()        # Serialise playback; prevents audio overlap
_stop_flag = threading.Event()


def speak(text: str) -> None:
    """Speak *text* asynchronously.

    Returns immediately; actual synthesis runs in a daemon thread.
    Calls are serialised by a lock so overlapping speech is queued, not cut off.
    Safe to call from any thread.
    """
    if not HAS_TTS or not text.strip():
        return
    threading.Thread(target=_speak_sync, args=(text,), daemon=True).start()


def stop() -> None:
    """Interrupt the currently playing speech immediately."""
    _stop_flag.set()
    if HAS_TTS:
        try:
            _mixer.music.stop()
        except Exception:
            pass


def _speak_sync(text: str) -> None:
    """Blocking synthesis + playback; should only run inside a daemon thread."""
    with _lock:
        _stop_flag.clear()
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            communicate = _edge.Communicate(text, ATC_VOICE)
            asyncio.run(communicate.save(path))
            if _stop_flag.is_set():
                return
            _mixer.music.load(path)
            _mixer.music.play()
            while _mixer.music.get_busy() and not _stop_flag.is_set():
                time.sleep(0.05)
            _mixer.music.stop()
        except Exception:
            pass
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
