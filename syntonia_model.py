"""Syntonia Model — combine V(t), F(t), M(t) into Syntonia(t).

    Syntonia(t) = α·V(t) + β·F(t) + γ·M(t),    α + β + γ = 1

The Syntonia score is reported in sigma-comparable units; the operational
threshold for a behavioural-friction event is 3.0 within a 5-second sliding
window. This is the technical-engine layer that powers the SyntoniaPro
consumer product.

IMPORTANT INTERPRETATION CAVEAT (preserved as a constant in code so it
cannot be silently stripped from downstream reports):

    The Syntonia score is a behavioural-friction indicator, not a deception
    verdict. It flags moments where the subject's combined voice / fidget /
    face signal departs significantly from their own baseline. Anxiety,
    distraction, physical discomfort, native-language fluency, cultural
    baselines and neurodivergence all produce Syntonia excursions in the
    absence of deception. The 3.0 threshold itself must be empirically
    calibrated on ground-truth-labelled data before any operational use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


SYNTONIA_DISCLAIMER = (
    "The Syntonia score is a behavioural-friction indicator, not a deception "
    "verdict. It flags moments where the subject's combined voice / fidget / face "
    "signal departs significantly from their own baseline. Anxiety, "
    "distraction, physical discomfort, native-language fluency, cultural "
    "baselines and neurodivergence all produce Syntonia excursions in the "
    "absence of deception. The 3.0 threshold must be empirically "
    "calibrated on ground-truth-labelled data before operational use."
)


@dataclass
class SyntoniaWindow:
    """One contiguous 5-second-or-longer window where Syntonia exceeded threshold."""

    start_t: float
    end_t: float
    duration_s: float
    peak_syntonia: float
    peak_t: float
    # Mean per-component contributions inside the window:
    mean_alpha_v: float
    mean_beta_f: float
    mean_gamma_m: float
    # And which channel dominated at the peak:
    dominant_component: str  # 'V', 'F', or 'M'

    def to_dict(self) -> dict:
        return {
            "start_t": self.start_t,
            "end_t": self.end_t,
            "duration_s": self.duration_s,
            "peak_syntonia": self.peak_syntonia,
            "peak_t": self.peak_t,
            "mean_alpha_v": self.mean_alpha_v,
            "mean_beta_f": self.mean_beta_f,
            "mean_gamma_m": self.mean_gamma_m,
            "dominant_component": self.dominant_component,
        }


@dataclass
class SyntoniaReport:
    times: np.ndarray
    v: np.ndarray
    f: np.ndarray
    m: np.ndarray
    syntonia: np.ndarray
    alpha: float
    beta: float
    gamma: float
    threshold: float
    window_s: float
    windows: List[SyntoniaWindow] = field(default_factory=list)
    disclaimer: str = SYNTONIA_DISCLAIMER

    def summary(self) -> dict:
        return {
            "n_samples": int(self.times.size),
            "duration_s": float(self.times[-1] - self.times[0]) if self.times.size else 0.0,
            "weights": {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma},
            "threshold": self.threshold,
            "max_syntonia": float(self.syntonia.max()) if self.syntonia.size else 0.0,
            "mean_syntonia": float(self.syntonia.mean()) if self.syntonia.size else 0.0,
            "windows": [w.to_dict() for w in self.windows],
            "disclaimer": self.disclaimer,
        }


def _resample_to(target_t: np.ndarray, source_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear-interpolate ``values`` (defined on ``source_t``) onto ``target_t``.
    Extrapolated edges hold the nearest in-range value."""
    if source_t.size == 0:
        return np.zeros_like(target_t)
    return np.interp(target_t, source_t, values, left=values[0], right=values[-1])


def compute_syntonia(
    target_times: np.ndarray,
    v_times: Optional[np.ndarray], v_values: Optional[np.ndarray],
    f_times: Optional[np.ndarray], f_values: Optional[np.ndarray],
    m_times: Optional[np.ndarray], m_values: Optional[np.ndarray],
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
    gamma: float = 1.0 / 3.0,
    threshold: float = 3.0,
    window_s: float = 5.0,
) -> SyntoniaReport:
    """Resample each subindex to ``target_times`` (the face-frame timestamps),
    compose Syntonia(t), then scan for contiguous windows where the rolling
    ``window_s`` mean exceeds ``threshold``.

    Missing channels (e.g. V=None for a silent video, F=None when hands are
    out of frame) are treated as zero; the weight on that channel still
    counts toward the α + β + γ normalisation so Syntonia scales sensibly.
    """
    target_times = np.asarray(target_times, dtype=float)
    n = target_times.size

    def _resample(t, v):
        if t is None or v is None or len(t) == 0:
            return np.zeros(n)
        return _resample_to(target_times, np.asarray(t, dtype=float), np.asarray(v, dtype=float))

    v_aligned = _resample(v_times, v_values)
    f_aligned = _resample(f_times, f_values)
    m_aligned = _resample(m_times, m_values)

    syntonia = alpha * v_aligned + beta * f_aligned + gamma * m_aligned

    # Smooth-over a window_s sliding mean before thresholding. Equivalent to a
    # boxcar-filtered Syntonia; the 3.0 threshold then translates to "the mean Syntonia
    # over the last window_s seconds exceeded 3.0".
    windowed_syntonia = _rolling_mean(target_times, syntonia, window_s)

    windows: List[SyntoniaWindow] = []
    above = windowed_syntonia >= threshold
    if above.any():
        starts = []
        ends = []
        in_run = False
        s = 0
        for i, a in enumerate(above):
            if a and not in_run:
                in_run = True
                s = i
            elif not a and in_run:
                in_run = False
                starts.append(s)
                ends.append(i)
        if in_run:
            starts.append(s)
            ends.append(len(above))
        for si, ei in zip(starts, ends):
            window_slice = slice(si, ei)
            peak_local = int(np.argmax(windowed_syntonia[window_slice]))
            peak_idx = si + peak_local
            # Channel-wise mean contributions inside the window.
            mean_av = float(alpha * v_aligned[window_slice].mean())
            mean_bf = float(beta * f_aligned[window_slice].mean())
            mean_gm = float(gamma * m_aligned[window_slice].mean())
            # Dominant component at peak.
            cur = {
                "V": alpha * v_aligned[peak_idx],
                "F": beta * f_aligned[peak_idx],
                "M": gamma * m_aligned[peak_idx],
            }
            dominant = max(cur, key=cur.get)
            windows.append(
                SyntoniaWindow(
                    start_t=float(target_times[si]),
                    end_t=float(target_times[ei - 1]),
                    duration_s=float(target_times[ei - 1] - target_times[si]),
                    peak_syntonia=float(windowed_syntonia[peak_idx]),
                    peak_t=float(target_times[peak_idx]),
                    mean_alpha_v=mean_av,
                    mean_beta_f=mean_bf,
                    mean_gamma_m=mean_gm,
                    dominant_component=dominant,
                )
            )

    return SyntoniaReport(
        times=target_times,
        v=v_aligned,
        f=f_aligned,
        m=m_aligned,
        syntonia=windowed_syntonia,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        threshold=threshold,
        window_s=window_s,
        windows=windows,
    )


def _rolling_mean(times: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    """Backward-looking rolling mean over a ``window_s``-second window.
    O(n) two-pointer implementation."""
    n = times.size
    out = np.zeros(n)
    j = 0
    cum = 0.0
    count = 0
    for i in range(n):
        cum += float(values[i])
        count += 1
        while j < i and (times[i] - times[j]) > window_s:
            cum -= float(values[j])
            count -= 1
            j += 1
        out[i] = cum / max(count, 1)
    return out
