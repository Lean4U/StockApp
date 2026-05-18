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

Signature
---------
A scale-invariant feature vector is computed per frame (normalized sides,
interior angles, diagonals, eye/mouth-line ratios, parallelism residual).
A **baseline** signature is the (mean, std) of these features over an
enrollment recording — that *is* the unique mathematical signature of the
subject's trapezium.

Change detection
----------------
Given a baseline and a fresh recording, each frame's feature vector is
converted to z-scores against the baseline. A per-sample aggregate
deviation (RMS of z-scores) is thresholded at 3-sigma (significant change)
and 6-sigma (extreme change); contiguous runs above each threshold are
reported as events with start time, end time, duration and peak z.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import cv2
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]
    _MP_AVAILABLE = False


# MediaPipe FaceMesh landmark indices for the four trapezium vertices.
LEFT_EYE_OUTER_IDX = 33    # image-left eye, outer canthus
RIGHT_EYE_OUTER_IDX = 263  # image-right eye, outer canthus
LEFT_MOUTH_IDX = 61        # image-left mouth corner
RIGHT_MOUTH_IDX = 291      # image-right mouth corner


# Order of vertices traversed around the quadrilateral:
#   0: left eye outer  -> 1: right eye outer  -> 2: right mouth  -> 3: left mouth
# (clockwise on a forward-facing image)
_VERT_ORDER = ("left_eye", "right_eye", "right_mouth", "left_mouth")


@dataclass
class TrapeziumSample:
    """One observation of the eye-eye-mouth-mouth trapezium at time ``t`` (seconds)."""

    t: float
    left_eye: np.ndarray    # (3,) outer corner of image-left eye
    right_eye: np.ndarray   # (3,) outer corner of image-right eye
    left_mouth: np.ndarray  # (3,) image-left mouth corner
    right_mouth: np.ndarray # (3,) image-right mouth corner

    # Dimensional characteristics, populated by compute_derived().
    sides: np.ndarray = field(default_factory=lambda: np.zeros(4))      # eye, right, mouth, left
    angles: np.ndarray = field(default_factory=lambda: np.zeros(4))     # at LE, RE, RM, LM
    diagonals: np.ndarray = field(default_factory=lambda: np.zeros(2))  # LE-RM, RE-LM
    perimeter: float = 0.0
    area: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eye_line: float = 0.0
    mouth_line: float = 0.0
    eye_mouth_ratio: float = 0.0
    parallelism_residual: float = 0.0  # |sin(theta)| of eye-line vs mouth-line
    diag_ratio: float = 0.0            # shorter diagonal / longer diagonal

    def vertices(self) -> List[np.ndarray]:
        return [self.left_eye, self.right_eye, self.right_mouth, self.left_mouth]

    def compute_derived(self) -> "TrapeziumSample":
        v = self.vertices()
        # Side lengths (eye-line, right-side, mouth-line, left-side)
        self.sides = np.array([np.linalg.norm(v[(i + 1) % 4] - v[i]) for i in range(4)])
        self.perimeter = float(self.sides.sum())

        # Diagonals
        self.diagonals = np.array([
            float(np.linalg.norm(v[2] - v[0])),  # left_eye -> right_mouth
            float(np.linalg.norm(v[3] - v[1])),  # right_eye -> left_mouth
        ])
        d_min, d_max = float(self.diagonals.min()), float(self.diagonals.max())
        self.diag_ratio = d_min / d_max if d_max > 0 else 0.0

        # Interior angles via dot products of edge vectors (in 3D, so robust to head tilt).
        eps = 1e-9
        angles = []
        for i in range(4):
            u = v[(i - 1) % 4] - v[i]
            w = v[(i + 1) % 4] - v[i]
            cos_a = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + eps)
            angles.append(float(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        self.angles = np.array(angles)

        # 2D shoelace area (using image-plane projection).
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

        # Parallelism residual: |sin(angle between eye-line and mouth-line)|.
        eye_dir = self.right_eye - self.left_eye
        mouth_dir = self.right_mouth - self.left_mouth
        cross = np.linalg.norm(np.cross(eye_dir, mouth_dir))
        denom = np.linalg.norm(eye_dir) * np.linalg.norm(mouth_dir) + eps
        self.parallelism_residual = float(cross / denom)
        return self


def _mean_point(landmarks, idxs, w: int, h: int) -> np.ndarray:
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w) for i in idxs],
        dtype=float,
    )
    return pts.mean(axis=0)


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
)


def feature_vector(sample: TrapeziumSample) -> np.ndarray:
    """Scale-invariant dimensional feature vector for one trapezium frame."""
    p = sample.perimeter if sample.perimeter > 0 else 1.0
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
    ])


@dataclass
class Baseline:
    """Unique mathematical signature of the subject's trapezium.

    ``means`` is the per-feature average across enrollment frames and ``stds`` is
    the per-feature standard deviation. Together they parametrize the subject's
    expected operating point; deviations from this point in σ units are how the
    change detector decides whether a new clip is "the same person doing the
    same thing".
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
        """Short stable ID derived from the quantized mean vector."""
        q = np.round(self.means, precision).astype(np.float64)
        return hashlib.sha256(q.tobytes()).hexdigest()[:16]


# Minimum noise floor for sigma so that perfectly-constant features (std=0) can
# still register a sensible z-score when the live frame drifts.
_STD_FLOOR = 1e-4


def fit_baseline(samples: Sequence[TrapeziumSample]) -> Optional[Baseline]:
    """Compute the baseline signature from an enrollment recording.

    Returns ``None`` if fewer than 2 valid samples are supplied.
    """
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
    """RMS distance between two baseline mean vectors. Used for identification."""
    if a.means.shape != b.means.shape:
        return float("inf")
    return float(np.linalg.norm(a.means - b.means) / np.sqrt(a.means.size))


# ---------------------------------------------------------------------------
# 3σ / 6σ statistical-change detection
# ---------------------------------------------------------------------------

@dataclass
class ChangeEvent:
    """One contiguous time-window where the aggregate deviation crossed a sigma threshold."""

    sigma_level: int        # 3 or 6
    start_t: float
    end_t: float
    duration_s: float
    peak_z: float
    peak_t: float
    dominant_feature: str   # feature with the largest absolute z at peak

    def to_dict(self) -> dict:
        return {
            "sigma_level": self.sigma_level,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "duration_s": self.duration_s,
            "peak_z": self.peak_z,
            "peak_t": self.peak_t,
            "dominant_feature": self.dominant_feature,
        }


@dataclass
class ChangeReport:
    times: np.ndarray
    per_feature_z: np.ndarray  # (n_samples, n_features)
    overall_z: np.ndarray      # (n_samples,) per-sample aggregate deviation
    events: List[ChangeEvent]
    sigma_low: float
    sigma_high: float

    def summary(self) -> dict:
        return {
            "n_samples": int(self.times.size),
            "duration_s": float(self.times[-1] - self.times[0]) if self.times.size else 0.0,
            "max_overall_z": float(self.overall_z.max()) if self.overall_z.size else 0.0,
            "n_events_3sigma": sum(1 for e in self.events if e.sigma_level == 3),
            "n_events_6sigma": sum(1 for e in self.events if e.sigma_level == 6),
            "events": [e.to_dict() for e in self.events],
        }


def _runs_above(values: np.ndarray, threshold: float, min_run: int = 1) -> List[tuple]:
    """Return list of (start_idx, end_idx_exclusive) where values >= threshold."""
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
) -> ChangeReport:
    """Compare a new recording against the enrolled baseline and emit a change
    report with all 3σ / 6σ time windows.

    The per-sample aggregate deviation is the RMS of the per-feature z-scores,
    which is a one-dimensional surrogate for a Mahalanobis distance under
    independent features. A 6σ window is also emitted as a 3σ window (the
    6σ event is a strict subset of the broader excursion).
    """
    if not samples:
        return ChangeReport(
            times=np.zeros(0),
            per_feature_z=np.zeros((0, baseline.means.size)),
            overall_z=np.zeros(0),
            events=[],
            sigma_low=sigma_low,
            sigma_high=sigma_high,
        )

    features = np.stack([feature_vector(s) for s in samples])
    z = (features - baseline.means) / baseline.stds  # (n, k)
    overall_z = np.sqrt(np.mean(z * z, axis=1))      # (n,)
    times = np.array([s.t for s in samples])

    events: List[ChangeEvent] = []
    for level, thresh in ((3, sigma_low), (6, sigma_high)):
        for s_idx, e_idx in _runs_above(overall_z, thresh, min_run=min_run_samples):
            window_z = overall_z[s_idx:e_idx]
            peak_local = int(np.argmax(window_z))
            peak_idx = s_idx + peak_local
            dom = int(np.argmax(np.abs(z[peak_idx])))
            events.append(
                ChangeEvent(
                    sigma_level=level,
                    start_t=float(times[s_idx]),
                    end_t=float(times[e_idx - 1]),
                    duration_s=float(times[e_idx - 1] - times[s_idx]),
                    peak_z=float(window_z[peak_local]),
                    peak_t=float(times[peak_idx]),
                    dominant_feature=baseline.feature_names[dom],
                )
            )

    events.sort(key=lambda e: (e.start_t, -e.sigma_level))
    return ChangeReport(
        times=times,
        per_feature_z=z,
        overall_z=overall_z,
        events=events,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
    )
