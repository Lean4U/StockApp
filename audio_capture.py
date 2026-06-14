"""Live audio capture for SyntoniaPro live dashboard.

Captures mono 16 kHz audio in the background while the user records, then
hands the buffer to faster-whisper for transcription. Each transcript
segment carries a start/end timestamp aligned to the video timeline, so
the dashboard can show "what you said" at each behavioural inflection.

Privacy posture matches the rest of the app: audio buffer lives in
RAM; nothing is sent off-device; the wav is written to a temp path
that's deleted after transcription.

Falls back gracefully when sounddevice or faster-whisper aren't
installed — the dashboard treats transcription as optional.
"""
from __future__ import annotations

import os
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

SAMPLE_RATE = 16000

try:
    import sounddevice as sd  # type: ignore
    _HAS_SOUND = True
except Exception:  # pragma: no cover
    sd = None
    _HAS_SOUND = False


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class AudioRecorder:
    """Background mic recorder with start/stop control.

    Held as a Streamlit @st.cache_resource singleton so it survives reruns.
    """

    def __init__(self) -> None:
        self._frames: List[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self._recording = False

    @property
    def available(self) -> bool:
        return _HAS_SOUND

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def n_samples(self) -> int:
        with self._lock:
            return sum(f.shape[0] for f in self._frames)

    @property
    def duration_s(self) -> float:
        return self.n_samples / float(SAMPLE_RATE)

    def _callback(self, indata, frames, t, status) -> None:  # noqa: D401
        if not self._recording:
            return
        with self._lock:
            self._frames.append(indata.copy())

    def start(self, anchor_t: float) -> bool:
        """Start capture. ``anchor_t`` is the video-pipeline timestamp at
        which the recorder began so transcript segments can be aligned.
        """
        if not _HAS_SOUND or self._recording:
            return False
        with self._lock:
            self._frames = []
        self._t0 = anchor_t
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self._recording = True
            return True
        except Exception:
            self._stream = None
            self._recording = False
            return False

    def stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None

    def clear(self) -> None:
        self.stop()
        with self._lock:
            self._frames = []
        self._t0 = None

    def get_audio(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._frames:
                return None
            return np.concatenate(self._frames, axis=0).flatten()

    def save_wav(self, path: Path) -> bool:
        audio = self.get_audio()
        if audio is None:
            return False
        # Convert float32 [-1, 1] to int16 PCM.
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return True


def transcribe(
    audio_path: Path,
    model_root: Path = Path("models/whisper-small"),
    language: Optional[str] = None,
) -> List[TranscriptSegment]:
    """Run faster-whisper on a local wav file and return aligned segments.

    Returns an empty list on any failure (model missing, lib missing, etc.) —
    transcription is treated as a best-effort, optional layer.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return []
    if not audio_path.exists():
        return []
    try:
        model = WhisperModel(
            str(model_root),
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )
        segments, _ = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
        )
        out: List[TranscriptSegment] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            out.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                )
            )
        return out
    except Exception:
        return []


def segment_at(
    segments: List[TranscriptSegment], t: float
) -> Optional[TranscriptSegment]:
    """Return the transcript segment that brackets time ``t``."""
    if not segments:
        return None
    for s in segments:
        if s.start <= t <= s.end:
            return s
    # Nearest fallback
    nearest = min(segments, key=lambda s: abs((s.start + s.end) / 2 - t))
    if abs((nearest.start + nearest.end) / 2 - t) <= 5.0:
        return nearest
    return None


def text_within(
    segments: List[TranscriptSegment], start_t: float, end_t: float
) -> str:
    """Concatenate transcript text overlapping the window [start_t, end_t]."""
    if not segments:
        return ""
    out: List[str] = []
    for s in segments:
        if s.end < start_t or s.start > end_t:
            continue
        out.append(s.text)
    return " ".join(out)


def ephemeral_wav_path() -> Path:
    """Return a unique temp path for a wav file."""
    fd, p = tempfile.mkstemp(prefix="syntonia_audio_", suffix=".wav")
    os.close(fd)
    return Path(p)
