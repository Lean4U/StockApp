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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _MP_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]
    FaceLandmarker = None  # type: ignore[assignment]
    _MP_AVAILABLE = False


_DEFAULT_MODEL_PATH = "models/face_landmarker.task"
_FACE_LANDMARKER_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


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

    # Dimensional characteristics, populated by compute_derived(). All values
    # below are expressed in the **face-plane 2D frame** (best-fit plane through
    # the 4 corner vertices, with u along the eye-line and v from eye-line
    # toward mouth) so they are invariant to head rotation in 3D.
    sides: np.ndarray = field(default_factory=lambda: np.zeros(4))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(4))
    diagonals: np.ndarray = field(default_factory=lambda: np.zeros(2))
    perimeter: float = 0.0
    area: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 3D image-frame centroid (for overlay only)
    eye_line: float = 0.0
    mouth_line: float = 0.0
    eye_mouth_ratio: float = 0.0
    parallelism_residual: float = 0.0
    diag_ratio: float = 0.0
    left_brow_height: float = 0.0   # face-frame v-distance from eye corner to brow point
    right_brow_height: float = 0.0
    mouth_offset_norm: float = 0.0  # signed lateral offset of mouth midpoint vs. eye midpoint
                                    # in face-frame x (asymmetric-occlusion indicator)
    # Head-pose proxies, derived from the 4-vertex best-fit plane.
    yaw_proxy: float = 0.0    # atan2(n_x, n_z) — rotation about face vertical axis
    pitch_proxy: float = 0.0  # atan2(-n_y, ||n_xz||) — rotation about face horizontal axis
    roll_proxy: float = 0.0   # atan2(u_y, u_x) — in-image-plane rotation of the eye-line
    planarity_residual_3d: float = 0.0  # smallest-/largest-SV ratio from the SVD plane fit

    def vertices(self) -> List[np.ndarray]:
        return [self.left_eye, self.right_eye, self.right_mouth, self.left_mouth]

    def compute_derived(self) -> "TrapeziumSample":
        # --- 1. Best-fit plane through the 4 corner vertices (SVD) -----------
        pts = np.stack(self.vertices())
        centroid_3d = pts.mean(axis=0)
        centered = pts - centroid_3d
        _, S_svd, Vt = np.linalg.svd(centered, full_matrices=False)
        n = Vt[-1]
        n_norm = np.linalg.norm(n)
        n = n / n_norm if n_norm > 1e-12 else np.array([0.0, 0.0, 1.0])

        # --- 2. In-plane orthonormal basis (u along eye-line, v toward mouth) -
        eye_dir = self.right_eye - self.left_eye
        u = eye_dir - np.dot(eye_dir, n) * n
        u_norm = np.linalg.norm(u)
        u = u / u_norm if u_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        v = np.cross(n, u)
        # Orient n (and v with it) so v points from eye-line toward mouth.
        mouth_center = 0.5 * (self.left_mouth + self.right_mouth)
        eye_center = 0.5 * (self.left_eye + self.right_eye)
        if np.dot(v, mouth_center - eye_center) < 0:
            n = -n
            v = -v

        # --- 3. Project all 6 landmarks onto (u, v) ---------------------------
        def proj(p: np.ndarray) -> np.ndarray:
            d = p - centroid_3d
            return np.array([float(np.dot(d, u)), float(np.dot(d, v))])

        le2 = proj(self.left_eye)
        re2 = proj(self.right_eye)
        rm2 = proj(self.right_mouth)
        lm2 = proj(self.left_mouth)
        lb2 = proj(self.left_brow)
        rb2 = proj(self.right_brow)
        v2 = [le2, re2, rm2, lm2]

        # --- 4. Trapezium dimensional characteristics in the 2D face frame ----
        self.sides = np.array(
            [float(np.linalg.norm(v2[(i + 1) % 4] - v2[i])) for i in range(4)]
        )
        self.perimeter = float(self.sides.sum())

        self.diagonals = np.array([
            float(np.linalg.norm(v2[2] - v2[0])),
            float(np.linalg.norm(v2[3] - v2[1])),
        ])
        d_min, d_max = float(self.diagonals.min()), float(self.diagonals.max())
        self.diag_ratio = d_min / d_max if d_max > 0 else 0.0

        eps = 1e-9
        angles = []
        for i in range(4):
            a = v2[(i - 1) % 4] - v2[i]
            b = v2[(i + 1) % 4] - v2[i]
            cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
            angles.append(float(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        self.angles = np.array(angles)

        x = np.array([p[0] for p in v2])
        y = np.array([p[1] for p in v2])
        self.area = float(
            0.5 * abs(
                x[0] * (y[1] - y[3])
                + x[1] * (y[2] - y[0])
                + x[2] * (y[3] - y[1])
                + x[3] * (y[0] - y[2])
            )
        )

        self.centroid = centroid_3d
        self.eye_line = float(self.sides[0])
        self.mouth_line = float(self.sides[2])
        self.eye_mouth_ratio = self.eye_line / (self.mouth_line + eps)

        eye_d2 = re2 - le2
        mouth_d2 = rm2 - lm2
        cross_2d = abs(eye_d2[0] * mouth_d2[1] - eye_d2[1] * mouth_d2[0])
        denom = np.linalg.norm(eye_d2) * np.linalg.norm(mouth_d2) + eps
        self.parallelism_residual = float(cross_2d / denom)

        # Brow heights now measured along the face's vertical axis (v).
        self.left_brow_height = float(le2[1] - lb2[1])
        self.right_brow_height = float(re2[1] - rb2[1])

        # Asymmetric-occlusion indicator: lateral offset of the mouth midpoint
        # vs. the eye midpoint in face-frame x. For a roughly symmetric face
        # the mouth sits directly under the eye-line midpoint, so this stays
        # near zero. A cigarette / pen / similar device pulling one mouth
        # corner laterally shifts the mouth midpoint sideways relative to the
        # eyes, driving this feature away from its baseline value.
        eye_mid_x = 0.5 * (le2[0] + re2[0])
        mouth_mid_x = 0.5 * (lm2[0] + rm2[0])
        scale = self.eye_line if self.eye_line > 0 else 1.0
        self.mouth_offset_norm = float((mouth_mid_x - eye_mid_x) / scale)

        # --- 5. Head-pose proxies (smooth functions of head orientation) -----
        self.yaw_proxy = float(np.arctan2(n[0], n[2]))
        self.pitch_proxy = float(np.arctan2(-n[1], np.sqrt(n[0] ** 2 + n[2] ** 2)))
        self.roll_proxy = float(np.arctan2(u[1], u[0]))

        # --- 6. Planarity residual (4-vertex non-coplanarity) ----------------
        s_max = float(S_svd[0]) if S_svd.size else 1.0
        s_min = float(S_svd[-1]) if S_svd.size else 0.0
        self.planarity_residual_3d = s_min / (s_max + eps)
        return self


class FaceTrapeziumDetector:
    """Wraps MediaPipe's FaceLandmarker (Tasks API) and returns one
    TrapeziumSample per video frame.

    The default model path is ``models/face_landmarker.task``. If it isn't
    present, fetch it once with::

        curl -sSL -o models/face_landmarker.task \\
          https://storage.googleapis.com/mediapipe-models/face_landmarker/\\
          face_landmarker/float16/latest/face_landmarker.task

    The Tasks API requires monotonically increasing per-call timestamps in
    milliseconds; this class enforces that internally so callers can pass
    arbitrary float seconds as ``t``.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe / opencv are required for live detection. "
                "Install them via `pip install -r requirements.txt`."
            )
        if not Path(model_path).exists():
            raise RuntimeError(
                f"FaceLandmarker model not found at {model_path}. "
                f"Download with:\n  curl -sSL -o {model_path} {_FACE_LANDMARKER_DOWNLOAD_URL}"
            )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, image_bgr: np.ndarray, t: float) -> Optional[TrapeziumSample]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(t * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]

        def pt(i: int) -> np.ndarray:
            p = lm[i]
            return np.array([p.x * w, p.y * h, p.z * w])

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
        self._landmarker.close()

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
    "mouth_offset_norm",
    "yaw_proxy", "pitch_proxy", "roll_proxy",
    "planarity_residual_3d",
)


def feature_vector(sample: TrapeziumSample) -> np.ndarray:
    """Scale-invariant dimensional feature vector for one trapezium frame.

    The first 17 entries are rotation-invariant geometric features computed in
    the face-plane 2D frame. The next 3 entries are head-pose proxies; their
    baseline mean encodes the subject's typical pose and z-scores against it
    detect head turns / nods / tilts as separate events from facial
    deformations. The last entry is the 4-vertex non-coplanarity which jumps
    during sneers, jaw drops and other non-rigid face deformations.
    """
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
        sample.mouth_offset_norm,
        sample.yaw_proxy,
        sample.pitch_proxy,
        sample.roll_proxy,
        sample.planarity_residual_3d,
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
_MAD_TO_STD = 1.4826  # consistent scaling so 1.4826 · MAD ≈ σ under Gaussian noise


def fit_baseline(
    samples: Sequence[TrapeziumSample],
    robust: bool = True,
) -> Optional[Baseline]:
    """Compute the per-feature baseline statistics from an enrollment recording.

    With ``robust=True`` (default), the centre is the median and the scale is
    1.4826 · MAD. These break-down at 50% contamination, so occasional
    enrolment outliers (lens glare, single-frame mis-detections, a startled
    blink) do not poison the baseline. Set ``robust=False`` to use the
    classical mean / std instead (matches the previous behaviour exactly).
    """
    if len(samples) < 2:
        return None
    features = np.stack([feature_vector(s) for s in samples])
    if robust:
        means = np.median(features, axis=0)
        mad = np.median(np.abs(features - means), axis=0)
        stds = _MAD_TO_STD * mad
    else:
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
# Robustness layer: Huber clipping + rolling-variance reliability weighting
# ---------------------------------------------------------------------------

def huber_clip(z: np.ndarray, cap: float = 8.0) -> np.ndarray:
    """Clip per-feature z-scores to ±cap. Prevents a single bad frame (e.g.
    lens glare) from blowing up the T² aggregate while still preserving
    the alarm signal for genuinely shifted features."""
    return np.clip(np.asarray(z, dtype=float), -cap, cap)


def rolling_z_variance(per_feature_z: np.ndarray, window: int = 30) -> np.ndarray:
    """Centered rolling variance of the per-feature z-scores, computed via
    cumulative sums so cost is O(n·k) rather than O(n·k·window).

    Under H0 (subject behaving like baseline), the rolling variance of each
    z-score channel is ≈ 1. A sustained mean shift (e.g. smile, brow raise)
    keeps the variance near 1 — only the centre moves. Tracking *noise*
    (lens glare flicker, smoke shimmer, mid-frame landmark dropout) blows
    the variance up to many times unity, and that is what
    `reliability_weights` then down-weights.
    """
    z = np.atleast_2d(np.asarray(per_feature_z, dtype=float))
    n, k = z.shape
    if n < 2 or window < 2:
        return np.ones((n, k))

    half = max(window // 2, 1)
    z_pad = np.pad(z, ((half, half), (0, 0)), mode="edge")
    cs = np.cumsum(z_pad, axis=0)
    cs2 = np.cumsum(z_pad * z_pad, axis=0)
    win = 2 * half  # actual window length after symmetric padding
    sum_x = cs[win:win + n] - cs[:n]
    sum_x2 = cs2[win:win + n] - cs2[:n]
    mean = sum_x / win
    var = sum_x2 / win - mean * mean
    return np.maximum(var, 0.0)


def reliability_weights(rolling_var: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Per-frame, per-feature reliability weight in (0, 1].

    Weight = ``threshold / max(rolling_var, threshold)``. A feature whose
    rolling z-variance stays at the expected unit level keeps weight = 1; a
    feature whose rolling z-variance climbs to 5× the threshold drops to
    weight = 0.2. Hands down-weighting is monotonic and smooth — there is no
    binary "on/off" mask that would create jumps in the aggregate test.
    """
    return threshold / np.maximum(np.asarray(rolling_var, dtype=float), threshold)


# ---------------------------------------------------------------------------
# Per-feature tabular CUSUM
# ---------------------------------------------------------------------------

def cusum(
    per_feature_z: np.ndarray,
    k: float = 0.5,
    h: float = 5.0,
    weights: Optional[np.ndarray] = None,
    update_threshold: float = 0.5,
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

    Optional per-frame reliability ``weights`` (shape ``(n_samples,
    n_features)`` or broadcastable) freeze the accumulator on any feature /
    frame where the weight is below ``update_threshold`` — the previous S
    value carries forward unchanged. This prevents unreliable tracking
    intervals (e.g. heavy glare bursts) from contributing CUSUM mass.
    """
    z = np.atleast_2d(per_feature_z)
    n, p = z.shape
    if weights is None:
        update_mask = np.ones((n, p), dtype=bool)
    else:
        update_mask = np.asarray(weights, dtype=float) >= update_threshold
        if update_mask.ndim == 1:
            update_mask = np.broadcast_to(update_mask, (n, p))
    s_plus = np.zeros((n, p))
    s_minus = np.zeros((n, p))
    alarm_high = np.zeros((n, p), dtype=bool)
    alarm_low = np.zeros((n, p), dtype=bool)
    for i in range(1, n):
        upd = update_mask[i]
        cur_p = np.where(upd, np.maximum(0.0, s_plus[i - 1] + z[i] - k), s_plus[i - 1])
        cur_m = np.where(upd, np.maximum(0.0, s_minus[i - 1] - z[i] - k), s_minus[i - 1])
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
    per_feature_z: np.ndarray              # raw z-scores (un-clipped, un-weighted)
    per_feature_z_clipped: np.ndarray      # after Huber clipping
    per_feature_weights: np.ndarray        # reliability weights in (0, 1]
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
    # Robustness knobs in effect
    huber_cap: float
    reliability_window: int
    reliability_threshold: float

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
    huber_cap: float = 8.0,
    reliability_window: int = 30,
    reliability_threshold: float = 5.0,
) -> ChangeReport:
    """Run all three change detectors against ``baseline`` and return a report.

    The robustness layer applies in order:

    1. **Huber clipping** — per-feature z-scores are clipped to ±``huber_cap``
       before any aggregation. A single bad frame (e.g. lens glare giving a
       50-σ outlier on one feature) can no longer monopolize T² or RMS.
    2. **Reliability weights** — each feature gets a per-frame weight in
       (0, 1] derived from its rolling z-variance over the last
       ``reliability_window`` frames. A feature with stable z (a real shift)
       keeps full weight; a feature with erratic z (occlusion noise) is
       smoothly down-weighted. Weights apply to the RMS aggregate, to T², and
       freeze the CUSUM accumulator for that feature on frames where its
       weight drops below 0.5.

    Detectors:
      * RMS aggregate ``sqrt(Σ w·z̃² / Σ w)`` → 3σ / 6σ events.
      * Hotelling's T² ``Σ w·z̃²`` → equivalent-σ events (threshold uses the
        full feature count for conservatism).
      * Per-feature tabular CUSUM with reset-after-alarm.
    """
    n_features = baseline.means.size

    if not samples:
        return ChangeReport(
            times=np.zeros(0),
            per_feature_z=np.zeros((0, n_features)),
            per_feature_z_clipped=np.zeros((0, n_features)),
            per_feature_weights=np.zeros((0, n_features)),
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
            huber_cap=huber_cap,
            reliability_window=reliability_window,
            reliability_threshold=reliability_threshold,
        )

    features = np.stack([feature_vector(s) for s in samples])
    z = (features - baseline.means) / baseline.stds
    z_clipped = huber_clip(z, cap=huber_cap)

    # Reliability weights from rolling variance of the clipped z-scores.
    rvar = rolling_z_variance(z_clipped, window=reliability_window)
    weights = reliability_weights(rvar, threshold=reliability_threshold)

    # Weighted aggregates.
    wz2 = weights * z_clipped * z_clipped
    sum_w = weights.sum(axis=1)
    sum_w_safe = np.maximum(sum_w, 1e-6)
    overall_z = np.sqrt(wz2.sum(axis=1) / sum_w_safe)
    t2 = wz2.sum(axis=1)
    t2_eq_sigma = t2_to_sigma_equivalent(t2, n_features)
    t2_low_thresh = t2_threshold(n_features, sigma_low)
    t2_high_thresh = t2_threshold(n_features, sigma_high)

    s_plus, s_minus, alarm_high, alarm_low = cusum(
        z_clipped, k=cusum_k, h=cusum_h, weights=weights,
    )

    times = np.array([s.t for s in samples])

    events: List[ChangeEvent] = []

    # ---- RMS aggregate events ------------------------------------------------
    for level, thresh in ((3, sigma_low), (6, sigma_high)):
        for s_idx, e_idx in _runs_above(overall_z, thresh, min_run=min_run_samples):
            window = overall_z[s_idx:e_idx]
            peak_local = int(np.argmax(window))
            peak_idx = s_idx + peak_local
            dom = int(np.argmax(np.abs(z_clipped[peak_idx])))
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
            dom = int(np.argmax(np.abs(z_clipped[peak_idx])))
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
        per_feature_z_clipped=z_clipped,
        per_feature_weights=weights,
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
        huber_cap=huber_cap,
        reliability_window=reliability_window,
        reliability_threshold=reliability_threshold,
    )
