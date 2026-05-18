"""Face trapezium (4-corner quadrilateral) geometry, signature, and SPC change detection.

Geometry
--------
For every frame we derive two reference lines:

* Line 1 — the segment between the **outer corners of the two eyes**, where the
  upper and lower eyelids intersect (MediaPipe FaceMesh landmarks 33 and 263).
* Line 2 — the segment between the **two corners of the mouth**, where the upper
  and lower lips intersect (MediaPipe FaceMesh landmarks 61 and 291).

The four endpoints form a quadrilateral. On a real face the eye-line and the
mouth-line are never exactly parallel (asymmetry + head pose), so the shape is
treated as a general **trapezium (no parallel sides)**.

Two additional landmarks — the **central eyebrow points** (105 and 334) — are
sampled to give explicit sensitivity to brow elevation (FACS AU 1 / AU 2). The
trapezium is still defined by 4 vertices; the brow points enter only as two
extra features (`left_brow_height_norm`, `right_brow_height_norm`).

Signature
---------
A scale-invariant feature vector is computed per frame (normalized sides,
interior angles, diagonals, eye/mouth-line ratios, parallelism residual, and
brow elevations). A **baseline** signature is the (mean, std) of these
features over an enrollment recording — that *is* the unique mathematical
signature of the subject's trapezium.

Change detection
----------------
Given a baseline and a fresh recording, each frame's feature vector is
converted to z-scores against the baseline. Three complementary detectors run
in parallel:

  * **RMS aggregate** — `sqrt(mean(z²))` per frame, thresholded at 3σ / 6σ.
    Conservative; misses signals that perturb only a few features.
  * **Hotelling's T²** — `sum(z²)` per frame, chi-square distributed under the
    null. Thresholded at T² quantiles equivalent to 3σ / 6σ via the
    Wilson-Hilferty approximation. More sensitive than RMS when several
    features move modestly together.
  * **Per-feature tabular CUSUM** — accumulates small persistent shifts on
    every feature individually (the textbook SPC tool for sustained drift).
    Alarms when max(S⁺, S⁻) exceeds a decision interval `h`.

All three detectors emit time-windowed events with start/end times, peak
deviation and dominant feature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]
    _MP_AVAILABLE = False


# MediaPipe FaceMesh landmark indices.
LEFT_EYE_OUTER_IDX = 33     # image-left eye, outer canthus
RIGHT_EYE_OUTER_IDX = 263   # image-right eye, outer canthus
LEFT_MOUTH_IDX = 61         # image-left mouth corner
RIGHT_MOUTH_IDX = 291       # image-right mouth corner
LEFT_BROW_CENTER_IDX = 105  # image-left eyebrow, central crown
RIGHT_BROW_CENTER_IDX = 334 # image-right eyebrow, central crown


# Trapezium vertex traversal order (clockwise on a forward-facing image).
_VERT_ORDER = ("left_eye", "right_eye", "right_mouth", "left_mouth")


@dataclass
class TrapeziumSample:
    """One observation of the eye-eye-mouth-mouth trapezium at time ``t`` (seconds).

    The trapezium itself is defined by the four corner vertices. The two
    eyebrow points are auxiliary landmarks used for brow-elevation features.
    """

    t: float
    left_eye: np.ndarray
    right_eye: np.ndarray
    left_mouth: np.ndarray
    right_mouth: np.ndarray
    left_brow: np.ndarray = field(default_factory=lambda: np.zeros(3))
    right_brow: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Dimensional characteristics, populated by compute_derived().
    sides: np.ndarray = field(default_factory=lambda: np.zeros(4))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(4))
    diagonals: np.ndarray = field(default_factory=lambda: np.zeros(2))
    perimeter: float = 0.0
    area: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eye_line: float = 0.0
    mouth_line: float = 0.0
    eye_mouth_ratio: float = 0.0
    parallelism_residual: float = 0.0
    diag_ratio: float = 0.0
    left_brow_height: float = 0.0   # px above the corresponding outer eye corner (image-y)
    right_brow_height: float = 0.0

    def vertices(self) -> List[np.ndarray]:
        return [self.left_eye, self.right_eye, self.right_mouth, self.left_mouth]

    def compute_derived(self) -> "TrapeziumSample":
        v = self.vertices()
        self.sides = np.array([np.linalg.norm(v[(i + 1) % 4] - v[i]) for i in range(4)])
        self.perimeter = float(self.sides.sum())

        self.diagonals = np.array([
            float(np.linalg.norm(v[2] - v[0])),
            float(np.linalg.norm(v[3] - v[1])),
        ])
        d_min, d_max = float(self.diagonals.min()), float(self.diagonals.max())
        self.diag_ratio = d_min / d_max if d_max > 0 else 0.0

        eps = 1e-9
        angles = []
        for i in range(4):
            u = v[(i - 1) % 4] - v[i]
            w = v[(i + 1) % 4] - v[i]
            cos_a = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + eps)
            angles.append(float(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        self.angles = np.array(angles)

        x = np.array([p[0] for p in v])
        y = np.array([p[1] for p in v])
        self.area = float(
            0.5 * abs(
                x[0] * (y[1] - y[3])
                + x[1] * (y[2] - y[0])
                + x[2] * (y[3] - y[1])
                + x[3] * (y[0] - y[2])
            )
        )

        self.centroid = np.mean(np.stack(v), axis=0)

        self.eye_line = float(self.sides[0])
        self.mouth_line = float(self.sides[2])
        self.eye_mouth_ratio = self.eye_line / (self.mouth_line + eps)

        eye_dir = self.right_eye - self.left_eye
        mouth_dir = self.right_mouth - self.left_mouth
        cross = np.linalg.norm(np.cross(eye_dir, mouth_dir))
        denom = np.linalg.norm(eye_dir) * np.linalg.norm(mouth_dir) + eps
        self.parallelism_residual = float(cross / denom)

        # Brow elevation: signed vertical (image-y) distance from each eye corner
        # to the corresponding brow point. Positive = brow above eye (anatomical
        # default). Roll-sensitive; switch to face-plane projection once rotation
        # modelling is added.
        self.left_brow_height = float(self.left_eye[1] - self.left_brow[1])
        self.right_brow_height = float(self.right_eye[1] - self.right_brow[1])
        return self


class FaceTrapeziumDetector:
    """Wraps MediaPipe FaceMesh and returns one TrapeziumSample per frame."""

    def __init__(self) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe / opencv are required for live detection. "
                "Install them via `pip install -r requirements.txt`."
            )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, image_bgr: np.ndarray, t: float) -> Optional[TrapeziumSample]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark

        def pt(i: int) -> np.ndarray:
            return np.array([lm[i].x * w, lm[i].y * h, lm[i].z * w])

        return TrapeziumSample(
            t=t,
            left_eye=pt(LEFT_EYE_OUTER_IDX),
            right_eye=pt(RIGHT_EYE_OUTER_IDX),
            left_mouth=pt(LEFT_MOUTH_IDX),
            right_mouth=pt(RIGHT_MOUTH_IDX),
            left_brow=pt(LEFT_BROW_CENTER_IDX),
            right_brow=pt(RIGHT_BROW_CENTER_IDX),
        ).compute_derived()

    def close(self) -> None:
        self.mesh.close()

    def __enter__(self) -> "FaceTrapeziumDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Feature vector + signature
# ---------------------------------------------------------------------------

FEATURE_NAMES: tuple = (
    "side_eye_norm", "side_right_norm", "side_mouth_norm", "side_left_norm",
    "angle_LE", "angle_RE", "angle_RM", "angle_LM",
    "diag_LE_RM_norm", "diag_RE_LM_norm", "diag_ratio",
    "eye_line_norm", "mouth_line_norm", "eye_mouth_ratio",
    "parallelism_residual",
    "left_brow_height_norm", "right_brow_height_norm",
)


def feature_vector(sample: TrapeziumSample) -> np.ndarray:
    """Scale-invariant dimensional feature vector for one trapezium frame."""
    p = sample.perimeter if sample.perimeter > 0 else 1.0
    eye_line = sample.eye_line if sample.eye_line > 0 else 1.0
    return np.array([
        sample.sides[0] / p,
        sample.sides[1] / p,
        sample.sides[2] / p,
        sample.sides[3] / p,
        sample.angles[0] / np.pi,
        sample.angles[1] / np.pi,
        sample.angles[2] / np.pi,
        sample.angles[3] / np.pi,
        sample.diagonals[0] / p,
        sample.diagonals[1] / p,
        sample.diag_ratio,
        sample.eye_line / p,
        sample.mouth_line / p,
        sample.eye_mouth_ratio,
        sample.parallelism_residual,
        sample.left_brow_height / eye_line,
        sample.right_brow_height / eye_line,
    ])


@dataclass
class Baseline:
    """Unique mathematical signature of the subject's trapezium.

    ``means`` is the per-feature average across enrollment frames and ``stds`` is
    the per-feature standard deviation.
    """

    feature_names: tuple
    means: np.ndarray
    stds: np.ndarray
    n_samples: int
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "n_samples": int(self.n_samples),
            "duration_s": float(self.duration_s),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        return cls(
            feature_names=tuple(d["feature_names"]),
            means=np.asarray(d["means"], dtype=float),
            stds=np.asarray(d["stds"], dtype=float),
            n_samples=int(d["n_samples"]),
            duration_s=float(d["duration_s"]),
        )

    def hash(self, precision: int = 4) -> str:
        q = np.round(self.means, precision).astype(np.float64)
        return hashlib.sha256(q.tobytes()).hexdigest()[:16]


_STD_FLOOR = 1e-4


def fit_baseline(samples: Sequence[TrapeziumSample]) -> Optional[Baseline]:
    if len(samples) < 2:
        return None
    features = np.stack([feature_vector(s) for s in samples])
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds = np.where(stds < _STD_FLOOR, _STD_FLOOR, stds)
    duration = float(samples[-1].t - samples[0].t)
    return Baseline(
        feature_names=FEATURE_NAMES,
        means=means,
        stds=stds,
        n_samples=len(samples),
        duration_s=duration,
    )


def signature_distance(a: Baseline, b: Baseline) -> float:
    if a.means.shape != b.means.shape:
        return float("inf")
    return float(np.linalg.norm(a.means - b.means) / np.sqrt(a.means.size))


# ---------------------------------------------------------------------------
# Aggregation statistics: Hotelling's T² and Wilson-Hilferty conversions
# ---------------------------------------------------------------------------

def hotelling_t2(per_feature_z: np.ndarray) -> np.ndarray:
    """Per-sample T² = Σ z². Under H0 (no shift in mean), T² ~ χ²(k)."""
    z = np.atleast_2d(per_feature_z)
    return np.sum(z * z, axis=1)


def t2_threshold(n_features: int, sigma: float) -> float:
    """Wilson-Hilferty approximation: χ² quantile whose tail probability matches
    a one-sided standard-normal ``sigma`` (e.g. ``sigma=3`` ↔ p≈0.99865)."""
    a = 2.0 / (9.0 * n_features)
    return float(n_features * (1.0 - a + sigma * np.sqrt(a)) ** 3)


def t2_to_sigma_equivalent(t2: np.ndarray, n_features: int) -> np.ndarray:
    """Inverse Wilson-Hilferty — express T² values back as standard-normal sigma."""
    a = 2.0 / (9.0 * n_features)
    t2_safe = np.maximum(np.asarray(t2, dtype=float), 1e-12)
    return (np.power(t2_safe / n_features, 1.0 / 3.0) - (1.0 - a)) / np.sqrt(a)


# ---------------------------------------------------------------------------
# Per-feature tabular CUSUM
# ---------------------------------------------------------------------------

def cusum(
    per_feature_z: np.ndarray,
    k: float = 0.5,
    h: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tabular CUSUM with reset-after-alarm, applied independently per feature.

    For each feature column ``j`` and time index ``i``::

        S⁺[i,j] = max(0, S⁺[i-1,j] + (z[i,j] − k))
        S⁻[i,j] = max(0, S⁻[i-1,j] − (z[i,j] + k))

    ``k`` is the slack (in σ; default 0.5 catches shifts ≥ ~1σ) and ``h`` is
    the decision interval (default 5σ → ARL₀ ≈ 465 in-control). When either
    accumulator exceeds ``h`` an alarm is recorded **and that accumulator is
    reset to 0**, the textbook practice that prevents a single persistent
    shift from generating one runaway event spanning the rest of the
    recording. A sustained shift will instead generate periodic alarms whose
    spacing is inversely proportional to the shift magnitude.

    Returns ``(s_plus, s_minus, alarm_high, alarm_low)`` each shaped
    ``(n_samples, n_features)``. ``alarm_high[i,j]`` is True iff the
    positive-side S⁺ crossed ``h`` at frame ``i``; ``alarm_low[i,j]`` is the
    symmetric flag for negative-side shifts.
    """
    z = np.atleast_2d(per_feature_z)
    n, p = z.shape
    s_plus = np.zeros((n, p))
    s_minus = np.zeros((n, p))
    alarm_high = np.zeros((n, p), dtype=bool)
    alarm_low = np.zeros((n, p), dtype=bool)
    for i in range(1, n):
        cur_p = np.maximum(0.0, s_plus[i - 1] + z[i] - k)
        cur_m = np.maximum(0.0, s_minus[i - 1] - z[i] - k)
        fire_p = cur_p > h
        fire_m = cur_m > h
        alarm_high[i] = fire_p
        alarm_low[i] = fire_m
        s_plus[i] = np.where(fire_p, 0.0, cur_p)
        s_minus[i] = np.where(fire_m, 0.0, cur_m)
    return s_plus, s_minus, alarm_high, alarm_low


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

@dataclass
class ChangeEvent:
    """One contiguous time-window flagged by a multivariate detector."""

    detector: str          # 'rms' or 't2'
    sigma_level: int       # 3 or 6
    start_t: float
    end_t: float
    duration_s: float
    peak_z: float          # for 'rms': aggregate |z|; for 't2': equivalent sigma
    peak_t: float
    dominant_feature: str

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "sigma_level": self.sigma_level,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "duration_s": self.duration_s,
            "peak_z": self.peak_z,
            "peak_t": self.peak_t,
            "dominant_feature": self.dominant_feature,
        }


@dataclass
class CusumEvent:
    """One contiguous time-window where a single feature's CUSUM crosses ``h``."""

    feature_name: str
    direction: str         # 'high' (mean shifted up) or 'low' (mean shifted down)
    start_t: float
    end_t: float
    duration_s: float
    peak_cusum: float
    peak_t: float

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "direction": self.direction,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "duration_s": self.duration_s,
            "peak_cusum": self.peak_cusum,
            "peak_t": self.peak_t,
        }


@dataclass
class ChangeReport:
    times: np.ndarray
    per_feature_z: np.ndarray
    # RMS aggregate detector
    overall_z: np.ndarray
    # Hotelling's T² detector
    t2: np.ndarray
    t2_equivalent_sigma: np.ndarray
    t2_threshold_low: float
    t2_threshold_high: float
    # CUSUM detector
    cusum_plus: np.ndarray
    cusum_minus: np.ndarray
    cusum_h: float
    cusum_k: float
    # Events
    events: List[ChangeEvent]
    cusum_events: List[CusumEvent]
    # Thresholds in use
    sigma_low: float
    sigma_high: float

    def summary(self) -> dict:
        return {
            "n_samples": int(self.times.size),
            "duration_s": float(self.times[-1] - self.times[0]) if self.times.size else 0.0,
            "max_overall_z": float(self.overall_z.max()) if self.overall_z.size else 0.0,
            "max_t2_sigma": float(self.t2_equivalent_sigma.max()) if self.t2.size else 0.0,
            "n_rms_events_3sigma": sum(1 for e in self.events if e.detector == "rms" and e.sigma_level == 3),
            "n_rms_events_6sigma": sum(1 for e in self.events if e.detector == "rms" and e.sigma_level == 6),
            "n_t2_events_3sigma": sum(1 for e in self.events if e.detector == "t2" and e.sigma_level == 3),
            "n_t2_events_6sigma": sum(1 for e in self.events if e.detector == "t2" and e.sigma_level == 6),
            "n_cusum_events": len(self.cusum_events),
            "events": [e.to_dict() for e in self.events],
            "cusum_events": [e.to_dict() for e in self.cusum_events],
        }


def _runs_above(values: np.ndarray, threshold: float, min_run: int = 1) -> List[tuple]:
    runs: List[tuple] = []
    in_run = False
    s = 0
    for i, v in enumerate(values):
        if v >= threshold and not in_run:
            in_run = True
            s = i
        elif v < threshold and in_run:
            in_run = False
            if i - s >= min_run:
                runs.append((s, i))
    if in_run and len(values) - s >= min_run:
        runs.append((s, len(values)))
    return runs


def detect_sigma_changes(
    samples: Sequence[TrapeziumSample],
    baseline: Baseline,
    sigma_low: float = 3.0,
    sigma_high: float = 6.0,
    min_run_samples: int = 2,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0,
) -> ChangeReport:
    """Run all three change detectors against ``baseline`` and return a report.

    * RMS aggregate ``|z|`` → 3σ / 6σ events.
    * Hotelling's T² ``= Σ z²`` → equivalent-sigma events via Wilson-Hilferty.
    * Per-feature tabular CUSUM with slack ``cusum_k`` and decision ``cusum_h``.
    """
    n_features = baseline.means.size

    if not samples:
        return ChangeReport(
            times=np.zeros(0),
            per_feature_z=np.zeros((0, n_features)),
            overall_z=np.zeros(0),
            t2=np.zeros(0),
            t2_equivalent_sigma=np.zeros(0),
            t2_threshold_low=t2_threshold(n_features, sigma_low),
            t2_threshold_high=t2_threshold(n_features, sigma_high),
            cusum_plus=np.zeros((0, n_features)),
            cusum_minus=np.zeros((0, n_features)),
            cusum_h=cusum_h,
            cusum_k=cusum_k,
            events=[],
            cusum_events=[],
            sigma_low=sigma_low,
            sigma_high=sigma_high,
        )

    features = np.stack([feature_vector(s) for s in samples])
    z = (features - baseline.means) / baseline.stds
    overall_z = np.sqrt(np.mean(z * z, axis=1))

    t2 = hotelling_t2(z)
    t2_eq_sigma = t2_to_sigma_equivalent(t2, n_features)
    t2_low_thresh = t2_threshold(n_features, sigma_low)
    t2_high_thresh = t2_threshold(n_features, sigma_high)

    s_plus, s_minus, alarm_high, alarm_low = cusum(z, k=cusum_k, h=cusum_h)

    times = np.array([s.t for s in samples])

    events: List[ChangeEvent] = []

    # ---- RMS aggregate events ------------------------------------------------
    for level, thresh in ((3, sigma_low), (6, sigma_high)):
        for s_idx, e_idx in _runs_above(overall_z, thresh, min_run=min_run_samples):
            window = overall_z[s_idx:e_idx]
            peak_local = int(np.argmax(window))
            peak_idx = s_idx + peak_local
            dom = int(np.argmax(np.abs(z[peak_idx])))
            events.append(
                ChangeEvent(
                    detector="rms",
                    sigma_level=level,
                    start_t=float(times[s_idx]),
                    end_t=float(times[e_idx - 1]),
                    duration_s=float(times[e_idx - 1] - times[s_idx]),
                    peak_z=float(window[peak_local]),
                    peak_t=float(times[peak_idx]),
                    dominant_feature=baseline.feature_names[dom],
                )
            )

    # ---- Hotelling's T² events ----------------------------------------------
    for level, thresh in ((3, t2_low_thresh), (6, t2_high_thresh)):
        for s_idx, e_idx in _runs_above(t2, thresh, min_run=min_run_samples):
            window = t2_eq_sigma[s_idx:e_idx]
            peak_local = int(np.argmax(window))
            peak_idx = s_idx + peak_local
            dom = int(np.argmax(np.abs(z[peak_idx])))
            events.append(
                ChangeEvent(
                    detector="t2",
                    sigma_level=level,
                    start_t=float(times[s_idx]),
                    end_t=float(times[e_idx - 1]),
                    duration_s=float(times[e_idx - 1] - times[s_idx]),
                    peak_z=float(window[peak_local]),
                    peak_t=float(times[peak_idx]),
                    dominant_feature=baseline.feature_names[dom],
                )
            )

    # ---- Per-feature CUSUM events -------------------------------------------
    # With reset-after-alarm, a sustained shift produces a *series* of close-
    # together alarm frames; we collapse each contiguous run of alarms in the
    # same direction (with no more than `cusum_merge_gap` quiet frames between
    # them) into one CusumEvent.
    cusum_merge_gap = max(int(round(cusum_h / max(cusum_k, 1e-6))) + 1, 5)
    cusum_events: List[CusumEvent] = []
    for j, name in enumerate(baseline.feature_names):
        for direction, alarms_col in (
            ("high", alarm_high[:, j]),
            ("low", alarm_low[:, j]),
        ):
            alarm_idxs = np.flatnonzero(alarms_col)
            if alarm_idxs.size == 0:
                continue
            # Group close-together alarms into one event.
            groups: List[List[int]] = [[int(alarm_idxs[0])]]
            for idx in alarm_idxs[1:]:
                if int(idx) - groups[-1][-1] <= cusum_merge_gap:
                    groups[-1].append(int(idx))
                else:
                    groups.append([int(idx)])
            for grp in groups:
                cusum_events.append(
                    CusumEvent(
                        feature_name=name,
                        direction=direction,
                        start_t=float(times[grp[0]]),
                        end_t=float(times[grp[-1]]),
                        duration_s=float(times[grp[-1]] - times[grp[0]]),
                        peak_cusum=float(cusum_h),
                        peak_t=float(times[grp[0]]),
                    )
                )

    events.sort(key=lambda e: (e.start_t, e.detector, -e.sigma_level))
    cusum_events.sort(key=lambda e: e.start_t)

    return ChangeReport(
        times=times,
        per_feature_z=z,
        overall_z=overall_z,
        t2=t2,
        t2_equivalent_sigma=t2_eq_sigma,
        t2_threshold_low=t2_low_thresh,
        t2_threshold_high=t2_high_thresh,
        cusum_plus=s_plus,
        cusum_minus=s_minus,
        cusum_h=cusum_h,
        cusum_k=cusum_k,
        events=events,
        cusum_events=cusum_events,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
    )
