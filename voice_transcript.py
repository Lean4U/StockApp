"""Local-only speech-to-text using faster-whisper.

PRIVACY GUARANTEE
-----------------
This module never sends audio over the network. The Whisper model files are
downloaded *once* from huggingface.co into ``models/whisper-small/`` and all
inference runs on-device via CTranslate2. After the one-time model fetch, the
host can be fully firewalled and transcription continues to work.

USAGE
-----
After running ``scripts/fetch_models.sh`` (or otherwise placing the model under
``models/whisper-small``), call ``transcribe_audio(video_path)`` to get a
``TranscriptResult`` with time-aligned segments and word-level timestamps.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


DEFAULT_MODEL_DIR = "models/whisper-small"
DEFAULT_MODEL_SIZE = "small"

# In-process cache so the dashboard's repeated dashboard reruns don't reload
# the model each time.
_MODEL_CACHE: dict = {}


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start": float(self.start),
            "end": float(self.end),
            "text": self.text,
            "words": [
                {"word": w.word, "start": float(w.start), "end": float(w.end)}
                for w in self.words
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptSegment":
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            text=str(d["text"]),
            words=[
                WordTiming(word=str(w["word"]),
                           start=float(w["start"]),
                           end=float(w["end"]))
                for w in d.get("words", [])
            ],
        )


@dataclass
class TranscriptResult:
    language: str
    language_probability: float
    duration_s: float
    segments: List[TranscriptSegment]

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": float(self.language_probability),
            "duration_s": float(self.duration_s),
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptResult":
        return cls(
            language=str(d.get("language", "?")),
            language_probability=float(d.get("language_probability", 0.0)),
            duration_s=float(d.get("duration_s", 0.0)),
            segments=[TranscriptSegment.from_dict(s) for s in d.get("segments", [])],
        )

    def text_within(self, start_t: float, end_t: float) -> str:
        """Return the concatenated transcript text whose words overlap the
        ``[start_t, end_t]`` window. Falls back to segment-level overlap when
        word-level timings are missing."""
        chunks: List[str] = []
        for seg in self.segments:
            if seg.end < start_t or seg.start > end_t:
                continue
            if seg.words:
                kept = [w.word for w in seg.words
                        if w.start <= end_t and w.end >= start_t]
                if kept:
                    chunks.append("".join(kept).strip())
            else:
                chunks.append(seg.text.strip())
        return " ".join(c for c in chunks if c)

    def segment_at(self, t: float) -> Optional[TranscriptSegment]:
        """Return the segment containing time ``t`` (or the nearest preceding
        one within 0.5 s) — used to populate the live caption strip."""
        for seg in self.segments:
            if seg.start <= t <= seg.end:
                return seg
        # Snap to nearest recently-ended segment so the caption hangs in view
        # for short pauses.
        prev = None
        for seg in self.segments:
            if seg.end < t:
                prev = seg
            else:
                break
        if prev is not None and (t - prev.end) < 0.5:
            return prev
        return None


def is_available(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    """Quick check whether the local Whisper model has been pre-staged."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception:
        return False
    p = Path(model_dir)
    if not p.exists():
        return False
    # Either the canonical layout (models--Systran--faster-whisper-small) or a
    # directory holding the snapshot files counts.
    if (p / "model.bin").exists() or (p / "tokenizer.json").exists():
        return True
    for child in p.rglob("model.bin"):
        return True
    return False


def _load_model(model_dir: str, model_size: str, compute_type: str):
    cache_key = (model_dir, model_size, compute_type)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    from faster_whisper import WhisperModel
    # ``download_root`` keeps the model purely under our control. If the
    # snapshot is already there CTranslate2 reuses it without touching the
    # network. ``local_files_only=True`` is enforced once the snapshot exists
    # so a misconfigured network won't trigger a re-fetch mid-session.
    has_local = Path(model_dir).exists() and (
        list(Path(model_dir).rglob("model.bin")) or list(Path(model_dir).rglob("*.bin"))
    )
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type=compute_type,
        download_root=model_dir,
        local_files_only=bool(has_local),
    )
    _MODEL_CACHE[cache_key] = model
    return model


def extract_audio_wav(video_path: Path, sample_rate: int = 16000) -> Path:
    """Extract mono 16 kHz WAV via ffmpeg into a tempfile. Caller deletes."""
    out = Path(tempfile.mkstemp(suffix=".wav", prefix="syntonia_asr_")[1])
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-vn",
            str(out),
        ],
        check=True,
    )
    return out


def transcribe_audio(
    video_path: Path,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_size: str = DEFAULT_MODEL_SIZE,
    compute_type: str = "int8",
    beam_size: int = 5,
    language: Optional[str] = None,
    vad_filter: bool = True,
    word_timestamps: bool = True,
) -> TranscriptResult:
    """Transcribe a video's audio entirely on-device. Returns time-aligned
    segments with word-level timestamps.

    Raises ``RuntimeError`` with a clear message when the model isn't
    available — caller should check ``is_available()`` first and surface a
    graceful UI fallback (e.g. "Transcript unavailable — run scripts/
    fetch_models.sh on a machine with network access").
    """
    if not is_available(model_dir):
        raise RuntimeError(
            f"Whisper model not present at {model_dir}. Run "
            f"scripts/fetch_models.sh on a machine with network access, "
            f"or copy a pre-fetched model into that directory."
        )
    model = _load_model(model_dir, model_size, compute_type)
    wav = extract_audio_wav(video_path)
    try:
        segments_iter, info = model.transcribe(
            str(wav),
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )
        segs: List[TranscriptSegment] = []
        for s in segments_iter:
            words = []
            if word_timestamps and getattr(s, "words", None):
                for w in s.words:
                    words.append(WordTiming(
                        word=w.word, start=float(w.start), end=float(w.end),
                    ))
            segs.append(TranscriptSegment(
                start=float(s.start), end=float(s.end),
                text=s.text.strip(), words=words,
            ))
        return TranscriptResult(
            language=info.language,
            language_probability=float(info.language_probability),
            duration_s=float(info.duration),
            segments=segs,
        )
    finally:
        try:
            wav.unlink()
        except OSError:
            pass
