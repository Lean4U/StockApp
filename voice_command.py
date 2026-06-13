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
CHUNK_DURATION_S = 1.0
POLL_INTERVAL_S = 0.25
BUFFER_DURATION_S = 2.5
SILENCE_PEAK = 0.012
COMMAND_COOLDOWN_S = 0.8
HEARD_DISPLAY_S = 1.8


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

# Whisper notoriously hallucinates 'thank you' / 'thanks' on short
# low-energy or silent chunks (over-represented in YouTube training
# data). Ignore these outputs outright. Also captures other common
# hallucinations on noise.
WHISPER_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks", "thanks.", "thank", "you",
    "you.", "yeah", "yeah.", "yep", "uh", "um", "mhm", "mm-hmm",
    "okay", "ok",
    "subtitles by", "subtitled by", "translated by",
    "♪",
    "gracias", "gracias.", "sí", "ah", "eh",
}

# Initial prompt biases Whisper toward our actual command vocabulary so
# it preferentially transcribes ambiguous syllables as 'record' / 'stop'
# rather than reaching for 'thank you'.
_WHISPER_BIAS_PROMPT = (
    "Voice command: record, start, begin, go, stop, halt, end, done, "
    "grabar, comenzar, parar, detener."
)


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
        # Diagnostics surfaced to the sidebar.
        self._last_error: str = ""
        self._chunks_processed: int = 0
        self._commands_detected: int = 0
        self._model_loaded: bool = False
        self._last_chunk_peak: float = 0.0
        # Continuous-capture state — replaces the chunked sd.rec() loop
        # so words can't fall in the gap between consecutive captures.
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._stream = None
        # Used by the dashboard to show a toast on every command detected.
        self._last_command: Optional[str] = None
        self._last_command_announce_time: float = 0.0

    @property
    def last_event_command(self) -> Optional[str]:
        return self._last_command

    @property
    def last_event_time(self) -> float:
        return self._last_command_announce_time

    def take_announcement(self) -> Optional[str]:
        """Return the most recently fired command exactly once, so the
        dashboard can show a toast that announces it. Returns None on
        subsequent calls until another command is detected."""
        with self._lock:
            cmd = self._last_command
            self._last_command = None
            return cmd

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def chunks_processed(self) -> int:
        return self._chunks_processed

    @property
    def commands_detected(self) -> int:
        return self._commands_detected

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def last_chunk_peak(self) -> float:
        return self._last_chunk_peak

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
            # Try the same path layout audio_capture.transcribe uses; that
            # works after scripts/fetch_models.py has run.
            tried = []
            for cand in (
                "models/whisper-small",
                "Systran/faster-whisper-small",  # fall back to HF id
            ):
                try:
                    self._model = WhisperModel(
                        cand,
                        device="cpu",
                        compute_type="int8",
                        local_files_only=(cand.startswith("models/")),
                    )
                    self._model_loaded = True
                    return True
                except Exception as inner:
                    tried.append(f"{cand}: {inner}")
                    continue
            raise RuntimeError(
                "Could not load Whisper for voice commands: "
                + " | ".join(tried)
            )
        except Exception as e:
            with self._lock:
                self._last_error = f"model load failed: {e}"
            return False

    def _audio_callback(self, indata, frames, t, status) -> None:
        """Continuous capture callback — append to the circular buffer."""
        if not self._active:
            return
        flat = indata.copy().flatten()
        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, flat])
            max_samples = int(SAMPLE_RATE * BUFFER_DURATION_S)
            if len(self._buffer) > max_samples:
                self._buffer = self._buffer[-max_samples:]

    def _ensure_stream(self) -> bool:
        if self._stream is not None:
            return True
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            return True
        except Exception as e:
            with self._lock:
                self._last_error = f"stream open failed: {e}"
            self._stream = None
            return False

    def _close_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None

    def _loop(self) -> None:
        if not self._ensure_model():
            self._active = False
            return
        if not self._ensure_stream():
            self._active = False
            return
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_S)
        try:
            while self._active:
                # Sleep first so the first tick has audio in the buffer.
                time.sleep(POLL_INTERVAL_S)
                if not self._active:
                    break
                # Take the most-recent 1.5 s slice from the rolling buffer.
                with self._buffer_lock:
                    if len(self._buffer) < chunk_samples:
                        continue
                    wav = self._buffer[-chunk_samples:].copy()
                peak = float(np.max(np.abs(wav)))
                self._last_chunk_peak = peak
                self._chunks_processed += 1
                if peak < SILENCE_PEAK:
                    continue
                try:
                    cmd = self._classify(wav)
                except Exception as e:
                    with self._lock:
                        self._last_error = f"classify error: {e}"
                    continue
                if cmd is None:
                    continue
                now = time.time()
                if now - self._last_command_time < COMMAND_COOLDOWN_S:
                    continue
                with self._lock:
                    self._pending = cmd
                    self._last_command = cmd
                    self._last_command_announce_time = now
                self._commands_detected += 1
                self._last_command_time = now
                # Drop the buffer after a successful command so the same
                # utterance can't trigger again in the next slice.
                with self._buffer_lock:
                    self._buffer = np.zeros(0, dtype=np.float32)
        finally:
            self._close_stream()

    def _classify(self, wav: np.ndarray) -> Optional[str]:
        """Transcribe a single audio chunk, update last_transcript, return
        the matched command or None. Shared by the background loop and
        the explicit 'Test mic' button.
        """
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
            # Three anti-hallucination knobs:
            #   initial_prompt: bias Whisper toward our keyword vocabulary
            #   no_speech_threshold: reject chunks Whisper thinks are silence
            #   condition_on_previous_text=False: don't drift across chunks
            segments, _ = self._model.transcribe(
                path,
                vad_filter=True,
                initial_prompt=_WHISPER_BIAS_PROMPT,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text for seg in segments).strip().lower()
        finally:
            try:
                Path(path).unlink()
            except Exception:
                pass
        with self._lock:
            self._last_transcript = text
        if not text:
            return None
        # Reject known Whisper hallucinations outright. These almost
        # always mean the user didn't actually say anything.
        if text in WHISPER_HALLUCINATIONS:
            return None
        # Token-level exact match.
        tokens = {tok.strip(".,!?;:¡¿") for tok in text.split()}
        if tokens & STOP_KEYWORDS:
            return "stop"
        if tokens & START_KEYWORDS:
            return "record"
        # Substring fall-back: 'recording', 'rec', 'gracaí' etc. should
        # still trigger 'record'. Search against the lowered transcript.
        for kw in STOP_KEYWORDS:
            if kw in text:
                return "stop"
        for kw in START_KEYWORDS:
            if kw in text:
                return "record"
        return None

    def test_one_shot(self) -> Optional[str]:
        """Synchronous one-shot test: capture one chunk, transcribe, return
        the matched command (or None). Used by the sidebar diagnostic
        button. Loads the model if not yet loaded.
        """
        if not _HAS_SOUND:
            with self._lock:
                self._last_error = "sounddevice not installed"
            return None
        if not self._ensure_model():
            return None
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_S)
        try:
            audio = sd.rec(
                chunk_samples,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            wav = audio.flatten()
            self._last_chunk_peak = float(np.max(np.abs(wav)))
            return self._classify(wav)
        except Exception as e:
            with self._lock:
                self._last_error = f"capture error: {e}"
            return None
