"""Unified Deceptive Friction Index — combine V(t), F(t), M(t) into DFI(t).

    DFI(t) = α·V(t) + β·F(t) + γ·M(t),    α + β + γ = 1

DFI is reported in sigma-comparable units; the operational threshold for a
"behavioral friction point" is 3.0 within a 5-second sliding window.

IMPORTANT INTERPRETATION CAVEAT (preserved as a constant in code so it
cannot be silently stripped from downstream reports):

    DFI is a behavioral-friction indicator, not a deception verdict. It
    flags moments where the subject's combined voice / fidget / face signal
    departs significantly from their own baseline. Anxiety, distraction,
    physical discomfort, native-language fluency, cultural baselines and
    neurodivergence all produce DFI excursions in the absence of
    deception. The 3.0 threshold itself must be empirically calibrated on
    ground-truth-labelled interview data before any operational use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


DFI_DISCLAIMER = (
    "DFI is a behavioral-friction indicator, not a deception verdict. "
    "It flags moments where the subject's combined voice / fidget / face "
    "signal departs significantly from their own baseline. Anxiety, "
    "distraction, physical discomfort, native-language fluency, cultural "
    "baselines and neurodivergence all produce DFI excursions in the "
    "absence of deception. The 3.0 threshold must be empirically "
    "calibrated on ground-truth-labelled data before operational use."
)


@dataclass
class DFIWindow:
    """One contiguous 5-second-or-longer window where DFI exceeded threshold."""

    start_t: float
    end_t: float
    duration_s: float
    peak_dfi: float
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
            "peak_dfi": self.peak_dfi,
            "peak_t": self.peak_t,
            "mean_alpha_v": self.mean_alpha_v,
            "mean_beta_f": self.mean_beta_f,
            "mean_gamma_m": self.mean_gamma_m,
            "dominant_component": self.dominant_component,
        }


@dataclass
class DFIReport:
    times: np.ndarray
    v: np.ndarray
    f: np.ndarray
    m: np.ndarray
    dfi: np.ndarray
    alpha: float
    beta: float
    gamma: float
    threshold: float
    window_s: float
    windows: List[DFIWindow] = field(default_factory=list)
    disclaimer: str = DFI_DISCLAIMER

    def summary(self) -> dict:
        return {
            "n_samples": int(self.times.size),
            "duration_s": float(self.times[-1] - self.times[0]) if self.times.size else 0.0,
            "weights": {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma},
            "threshold": self.threshold,
            "max_dfi": float(self.dfi.max()) if self.dfi.size else 0.0,
            "mean_dfi": float(self.dfi.mean()) if self.dfi.size else 0.0,
            "windows": [w.to_dict() for w in self.windows],
            "disclaimer": self.disclaimer,
        }


def _resample_to(target_t: np.ndarray, source_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear-interpolate ``values`` (defined on ``source_t``) onto ``target_t``.
    Extrapolated edges hold the nearest in-range value."""
    if source_t.size == 0:
        return np.zeros_like(target_t)
    return np.interp(target_t, source_t, values, left=values[0], right=values[-1])


def compute_dfi(
    target_times: np.ndarray,
    v_times: Optional[np.ndarray], v_values: Optional[np.ndarray],
    f_times: Optional[np.ndarray], f_values: Optional[np.ndarray],
    m_times: Optional[np.ndarray], m_values: Optional[np.ndarray],
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
    gamma: float = 1.0 / 3.0,
    threshold: float = 3.0,
    window_s: float = 5.0,
) -> DFIReport:
    """Resample each subindex to ``target_times`` (the face-frame timestamps),
    compose DFI(t), then scan for contiguous windows where the rolling
    ``window_s`` mean exceeds ``threshold``.

    Missing channels (e.g. V=None for a silent video, F=None when hands are
    out of frame) are treated as zero; the weight on that channel still
    counts toward the α + β + γ normalisation so DFI scales sensibly.
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

    dfi = alpha * v_aligned + beta * f_aligned + gamma * m_aligned

    # Smooth-over a window_s sliding mean before thresholding. Equivalent to a
    # boxcar-filtered DFI; the 3.0 threshold then translates to "the mean DFI
    # over the last window_s seconds exceeded 3.0".
    windowed_dfi = _rolling_mean(target_times, dfi, window_s)

    windows: List[DFIWindow] = []
    above = windowed_dfi >= threshold
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
            peak_local = int(np.argmax(windowed_dfi[window_slice]))
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
                DFIWindow(
                    start_t=float(target_times[si]),
                    end_t=float(target_times[ei - 1]),
                    duration_s=float(target_times[ei - 1] - target_times[si]),
                    peak_dfi=float(windowed_dfi[peak_idx]),
                    peak_t=float(target_times[peak_idx]),
                    mean_alpha_v=mean_av,
                    mean_beta_f=mean_bf,
                    mean_gamma_m=mean_gm,
                    dominant_component=dominant,
                )
            )

    return DFIReport(
        times=target_times,
        v=v_aligned,
        f=f_aligned,
        m=m_aligned,
        dfi=windowed_dfi,
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
