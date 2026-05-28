"""Voice analytics — V(t) component of the Syntonia score.

Extracts audio from a video file with ffmpeg, computes a per-second
syllables-per-second (SPS) proxy via onset-strength peak picking, and (when
turn-segmentation is feasible) per-utterance response latency. Each metric
is z-scored against a baseline window:

    V(t) = sqrt(z_SPS(t)² + z_RL(t)²)

In the single-speaker case the RL term degenerates and we report
``V(t) = |z_SPS(t)|``.

Implementation notes:

* SPS proxy uses ``librosa.onset.onset_strength`` followed by peak picking,
  not a true syllabification model. Empirically tracks speech rate within
  ~10–20% on conversational speech; good enough as a relative signal.
* Voice activity = onset-strength envelope above a noise floor; gaps longer
  than 0.4 s are treated as utterance boundaries.
* Audio is downmixed to mono 16 kHz for cheap processing.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import librosa
    _AUDIO_AVAILABLE = True
except Exception:  # pragma: no cover
    librosa = None  # type: ignore[assignment]
    _AUDIO_AVAILABLE = False


@dataclass
class VoiceResult:
    times: np.ndarray            # one timestamp per per-second bin (in seconds)
    sps: np.ndarray              # syllables per second
    sps_z: np.ndarray            # z-scored SPS vs. baseline
    rl: np.ndarray               # response latency per bin (sec); zeros if monologue
    rl_z: np.ndarray             # z-scored response latency
    v: np.ndarray                # V(t) = sqrt(z_SPS² + z_RL²)
    baseline_seconds: float
    sps_baseline_mean: float
    sps_baseline_std: float
    rl_baseline_mean: float
    rl_baseline_std: float
    n_audio_frames: int
    sample_rate: int

    def has_response_latency(self) -> bool:
        return bool(np.any(self.rl != 0.0))


def extract_audio_to_wav(video_path: Path, sample_rate: int = 16000) -> Path:
    """Decode the audio track of ``video_path`` to a temporary mono WAV file.
    Returns the WAV path. Caller is responsible for unlinking it."""
    out = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def detect_syllable_nuclei_times(
    y: np.ndarray, sr: int,
    *,
    pre_max: float = 0.07, post_max: float = 0.07,
    pre_avg: float = 0.10, post_avg: float = 0.10,
    delta: float = 0.07, wait: float = 0.05,
) -> np.ndarray:
    """Approximate syllable-nucleus times (in seconds) via peak-picking on
    the librosa onset-strength envelope."""
    hop = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.util.peak_pick(
        onset_env,
        pre_max=max(1, int(pre_max * sr / hop)),
        post_max=max(1, int(post_max * sr / hop)),
        pre_avg=max(1, int(pre_avg * sr / hop)),
        post_avg=max(1, int(post_avg * sr / hop)),
        delta=float(delta),
        wait=max(1, int(wait * sr / hop)),
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    return np.asarray(times)


def detect_speech_segments(
    y: np.ndarray, sr: int,
    *,
    energy_threshold_db: float = -40.0,
    min_silence_s: float = 0.4,
) -> List[Tuple[float, float]]:
    """Return [(start, end), ...] of voiced segments using an energy gate."""
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop).flatten()
    # Convert to dB relative to full scale.
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    voiced = rms_db > energy_threshold_db
    times = librosa.frames_to_time(np.arange(len(voiced)), sr=sr, hop_length=hop)

    segments: List[Tuple[float, float]] = []
    if not voiced.any():
        return segments
    in_seg = False
    seg_start = 0.0
    for i, v in enumerate(voiced):
        if v and not in_seg:
            in_seg = True
            seg_start = times[i]
        elif not v and in_seg:
            # Look ahead `min_silence_s` to confirm it's a real boundary.
            look_ahead_frames = int(min_silence_s * sr / hop)
            if not voiced[i:i + look_ahead_frames].any():
                in_seg = False
                segments.append((seg_start, times[i]))
    if in_seg:
        segments.append((seg_start, times[-1]))
    return segments


def compute_voice(
    video_path: Path,
    bin_s: float = 1.0,
    baseline_seconds: float = 15.0,
    sample_rate: int = 16000,
    estimate_response_latency: bool = False,
) -> VoiceResult:
    """Run the full V(t) extraction on a video file.

    ``estimate_response_latency=True`` is only sensible when the recording
    contains two speakers and we have diarization; for single-speaker
    monologues set this False (default) so the RL component is zero.
    """
    if not _AUDIO_AVAILABLE:
        raise RuntimeError("librosa is required for voice analytics")

    wav_path = extract_audio_to_wav(video_path, sample_rate=sample_rate)
    try:
        y, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass

    duration = librosa.get_duration(y=y, sr=sr)
    n_bins = int(np.ceil(duration / bin_s))
    bin_times = np.arange(n_bins) * bin_s + (bin_s / 2.0)

    # SPS per bin via syllable-nucleus times.
    nucleus_t = detect_syllable_nuclei_times(y, sr)
    sps = np.zeros(n_bins)
    bin_edges = np.arange(n_bins + 1) * bin_s
    counts, _ = np.histogram(nucleus_t, bins=bin_edges)
    sps = counts.astype(float) / bin_s

    # Response latency per bin (sec): if requested, derive from speech
    # segments. Single-speaker mode emits zeros.
    rl = np.zeros(n_bins)
    if estimate_response_latency:
        segments = detect_speech_segments(y, sr)
        # Latency = gap between end of one segment and start of the next.
        # Without speaker labels this isn't true RL; it's just inter-utterance
        # silence, which still captures hesitation patterns.
        for prev, nxt in zip(segments[:-1], segments[1:]):
            gap = nxt[0] - prev[1]
            mid_t = 0.5 * (prev[1] + nxt[0])
            bin_idx = int(mid_t / bin_s)
            if 0 <= bin_idx < n_bins:
                rl[bin_idx] = max(rl[bin_idx], gap)

    # Baseline window over the first ``baseline_seconds``.
    base_mask = bin_times <= baseline_seconds
    sps_baseline = sps[base_mask]
    rl_baseline = rl[base_mask]
    sps_mu = float(np.median(sps_baseline)) if sps_baseline.size else 0.0
    sps_sd = float(1.4826 * np.median(np.abs(sps_baseline - sps_mu))) if sps_baseline.size else 1e-3
    sps_sd = max(sps_sd, 1e-3)
    rl_mu = float(np.median(rl_baseline)) if rl_baseline.size else 0.0
    rl_sd = float(1.4826 * np.median(np.abs(rl_baseline - rl_mu))) if rl_baseline.size else 1e-3
    rl_sd = max(rl_sd, 1e-3)

    sps_z = (sps - sps_mu) / sps_sd
    rl_z = (rl - rl_mu) / rl_sd
    if not estimate_response_latency:
        rl_z = np.zeros_like(rl_z)
    v = np.sqrt(sps_z ** 2 + rl_z ** 2)

    return VoiceResult(
        times=bin_times,
        sps=sps,
        sps_z=sps_z,
        rl=rl,
        rl_z=rl_z,
        v=v,
        baseline_seconds=baseline_seconds,
        sps_baseline_mean=sps_mu,
        sps_baseline_std=sps_sd,
        rl_baseline_mean=rl_mu,
        rl_baseline_std=rl_sd,
        n_audio_frames=int(y.size),
        sample_rate=sr,
    )
