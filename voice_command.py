"""Hands-free voice control for the SyntoniaPro live dashboard.

Adds two voice commands so the subject can start and stop a take without
clicking the sidebar buttons (which introduces hand / head / face
movement that contaminates the very take being recorded):

    "record", "start", "begin"   → start recording
    "stop", "halt", "end", "done" → stop recording

Multi-language keyword sets are included for English, Spanish,
Portuguese, French and Italian.

Implementation
--------------
A background daemon thread cycles through:
    1) capture ~1.5 s of mono 16 kHz audio via sounddevice
    2) skip if amplitude is too low (no speech)
    3) transcribe with faster-whisper (the local Whisper-small model
       already shipped with the project; no extra download)
    4) lower-case scan for command keywords
    5) when one matches, set self.pending to "record" or "stop"

The Streamlit fragment that already polls every 100 ms calls
consume_pending() and toggles state on the main thread. The listener
self-suspends while a take is recording so it doesn't fight the
AudioRecorder for the mic.

Falls back silently when sounddevice or faster-whisper aren't
available (same posture as audio_capture.py).
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd  # type: ignore
    _HAS_SOUND = True
except Exception:  # pragma: no cover
    sd = None
    _HAS_SOUND = False


SAMPLE_RATE = 16000
CHUNK_DURATION_S = 1.5
SILENCE_PEAK = 0.012
COMMAND_COOLDOWN_S = 1.0


START_KEYWORDS = {
    "record", "recording", "start", "begin", "go",
    "grabar", "graba", "comenzar", "comienza", "iniciar", "inicia",
    "gravar", "começar", "iniciar",
    "enregistrer", "commencer",
    "registra", "iniziare",
}

STOP_KEYWORDS = {
    "stop", "halt", "end", "done", "finish", "finished", "cease",
    "parar", "para", "detener", "detén", "alto", "fin",
    "parar", "fim",
    "arrêter", "arrête",
    "ferma", "fine",
}


class VoiceCommandListener:
    """Background mic listener that turns the words 'record' or 'stop'
    into commands the dashboard can act on. Held as a Streamlit
    @st.cache_resource singleton.
    """

    def __init__(self) -> None:
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending: Optional[str] = None
        self._last_transcript: str = ""
        self._last_command_time: float = 0.0
        self._model = None
        self._enabled = False

    @property
    def available(self) -> bool:
        return _HAS_SOUND

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active(self) -> bool:
        return self._active

    @property
    def last_transcript(self) -> str:
        with self._lock:
            return self._last_transcript

    def enable(self) -> bool:
        """Turn the listener on. Background thread starts when not
        already running. Returns whether the listener became enabled.
        """
        if not _HAS_SOUND:
            return False
        self._enabled = True
        self._start_thread_if_idle()
        return True

    def disable(self) -> None:
        self._enabled = False
        self._stop_thread()

    def suspend(self) -> None:
        """Stop listening — used while a take is actually recording so
        sounddevice doesn't fight the AudioRecorder for the mic.
        """
        self._stop_thread()

    def resume(self) -> None:
        """Resume listening after a take ends, if the user had enabled it."""
        if self._enabled:
            self._start_thread_if_idle()

    def consume_pending(self) -> Optional[str]:
        with self._lock:
            cmd = self._pending
            self._pending = None
            return cmd

    # ── internals ────────────────────────────────────────────────────

    def _start_thread_if_idle(self) -> None:
        if self._active:
            return
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _stop_thread(self) -> None:
        self._active = False
        # Don't join — daemon thread will exit on next loop iteration.

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(
                "models/whisper-small",
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
            return True
        except Exception as e:
            with self._lock:
                self._last_transcript = f"(model load failed: {e})"
            return False

    def _loop(self) -> None:
        if not self._ensure_model():
            self._active = False
            return
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_S)
        while self._active:
            try:
                # Record a short chunk on the default input device.
                audio = sd.rec(
                    chunk_samples,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                if not self._active:
                    break
                wav = audio.flatten()
                if float(np.max(np.abs(wav))) < SILENCE_PEAK:
                    continue
                # Write to temp wav, transcribe, delete.
                fd, path = tempfile.mkstemp(prefix="cmd_", suffix=".wav")
                os.close(fd)
                try:
                    clipped = np.clip(wav, -1.0, 1.0)
                    pcm = (clipped * 32767).astype(np.int16)
                    with wave.open(path, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(SAMPLE_RATE)
                        w.writeframes(pcm.tobytes())
                    segments, _ = self._model.transcribe(
                        path,
                        vad_filter=True,
                    )
                    text = " ".join(seg.text for seg in segments).strip().lower()
                finally:
                    try:
                        Path(path).unlink()
                    except Exception:
                        pass
                if not text:
                    continue
                with self._lock:
                    self._last_transcript = text
                # Tokenise loosely and check against keyword sets.
                tokens = {tok.strip(".,!?;:") for tok in text.split()}
                cmd: Optional[str] = None
                if tokens & STOP_KEYWORDS:
                    cmd = "stop"
                elif tokens & START_KEYWORDS:
                    cmd = "record"
                if cmd is None:
                    continue
                now = time.time()
                if now - self._last_command_time < COMMAND_COOLDOWN_S:
                    continue
                with self._lock:
                    self._pending = cmd
                self._last_command_time = now
            except Exception as e:
                with self._lock:
                    self._last_transcript = f"(error: {e})"
                time.sleep(0.4)
