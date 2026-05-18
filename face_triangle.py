"""Face triangle (eyes + nose) geometry and motion-based signature.

The "unique mathematical function" derived for a subject is

    f(t) = (
        centroid_xyz(t),         # 3D triangle centroid trajectory (Fourier series)
        normal_theta_phi(t),     # triangle-plane normal as spherical angles (Fourier series)
        invariant_stats          # mean/std of scale-invariant side ratios + interior angles
    )

The Fourier series are fit on time normalized to [0, 1] with K harmonics, so
the coefficients are themselves the parameters of f. Two signatures can then
be compared by L2 distance of the concatenated coefficient vector, or hashed
to a short ID by quantizing then SHA-256ing the vector.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:  # optional at import time so unit tests don't require mediapipe/cv2
    import cv2
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when deps missing
    cv2 = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]
    _MP_AVAILABLE = False


# MediaPipe FaceMesh landmark indices for the three vertices of our triangle.
# The two eyes are averaged from corner + lid landmarks for stability.
LEFT_EYE_IDXS = (33, 133, 159, 145)
RIGHT_EYE_IDXS = (263, 362, 386, 374)
NOSE_TIP_IDX = 1


@dataclass
class TriangleSample:
    """One observation of the eye-eye-nose triangle at time ``t`` (seconds)."""

    t: float
    left_eye: np.ndarray   # (3,) in pixel-x, pixel-y, mediapipe-z (~depth)
    right_eye: np.ndarray
    nose: np.ndarray

    sides: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(3))
    area: float = 0.0
    perimeter: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def compute_derived(self) -> "TriangleSample":
        a = float(np.linalg.norm(self.right_eye - self.nose))      # opposite left eye
        b = float(np.linalg.norm(self.left_eye - self.nose))       # opposite right eye
        c = float(np.linalg.norm(self.left_eye - self.right_eye))  # opposite nose
        self.sides = np.array([a, b, c])
        self.perimeter = a + b + c

        s = self.perimeter / 2.0
        self.area = float(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0)))

        # Interior angles via the law of cosines (alpha at left eye, beta at right eye,
        # gamma at nose). Clipped to handle floating-point drift.
        eps = 1e-9
        alpha = np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c + eps), -1.0, 1.0))
        beta = np.arccos(np.clip((a * a + c * c - b * b) / (2 * a * c + eps), -1.0, 1.0))
        gamma = np.pi - alpha - beta
        self.angles = np.array([alpha, beta, gamma])

        self.centroid = (self.left_eye + self.right_eye + self.nose) / 3.0

        n = np.cross(self.right_eye - self.left_eye, self.nose - self.left_eye)
        nn = np.linalg.norm(n)
        self.normal = n / nn if nn > 0 else n
        return self


def _mean_point(landmarks, idxs, w: int, h: int) -> np.ndarray:
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w) for i in idxs],
        dtype=float,
    )
    return pts.mean(axis=0)


class FaceTriangleDetector:
    """Wraps MediaPipe FaceMesh and returns one TriangleSample per frame."""

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

    def detect(self, image_bgr: np.ndarray, t: float) -> Optional[TriangleSample]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        left_eye = _mean_point(lm, LEFT_EYE_IDXS, w, h)
        right_eye = _mean_point(lm, RIGHT_EYE_IDXS, w, h)
        nose = np.array([lm[NOSE_TIP_IDX].x * w, lm[NOSE_TIP_IDX].y * h, lm[NOSE_TIP_IDX].z * w])
        return TriangleSample(
            t=t, left_eye=left_eye, right_eye=right_eye, nose=nose
        ).compute_derived()

    def close(self) -> None:
        self.mesh.close()

    def __enter__(self) -> "FaceTriangleDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def normalized_invariants(sample: TriangleSample) -> np.ndarray:
    """Scale- and orientation-invariant feature vector for one triangle:
    sorted side ratios (3) + sorted angles / pi (3). Independent of camera distance."""
    perim = sample.perimeter if sample.perimeter > 0 else 1.0
    side_ratios = np.sort(sample.sides) / perim
    angles_norm = np.sort(sample.angles) / np.pi
    return np.concatenate([side_ratios, angles_norm])


def _normal_to_spherical(n: np.ndarray) -> tuple[float, float]:
    nn = np.linalg.norm(n)
    if nn == 0:
        return 0.0, 0.0
    nx, ny, nz = n / nn
    theta = float(np.arccos(np.clip(nz, -1.0, 1.0)))  # polar angle
    phi = float(np.arctan2(ny, nx))                   # azimuth
    return theta, phi


def _fourier_design(t: np.ndarray, K: int) -> np.ndarray:
    cols = [np.ones_like(t)]
    for k in range(1, K + 1):
        cols.append(np.cos(2.0 * np.pi * k * t))
        cols.append(np.sin(2.0 * np.pi * k * t))
    return np.stack(cols, axis=1)


def _fourier_fit(t: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """Return flattened coefficient vector [a0, a1, b1, ..., aK, bK]."""
    A = _fourier_design(t, K)
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coefs


def evaluate_signature_function(coefs: dict, t: np.ndarray) -> dict:
    """Reconstruct each fitted channel of f(t) on the times ``t in [0, 1]``.
    Useful for visualizing the function vs. the raw samples."""
    K = int(coefs.get("harmonics", 0))
    if K == 0:
        return {}
    A = _fourier_design(np.asarray(t, dtype=float), K)
    out: dict = {}
    for name in ("centroid_x", "centroid_y", "centroid_z", "normal_theta", "normal_phi"):
        c = coefs.get(name)
        if c is None:
            continue
        out[name] = A @ np.asarray(c)
    return out


def fit_signature(samples: Sequence[TriangleSample], harmonics: int = 4) -> dict:
    """Fit f(t) from a recording. Returns {} if too few samples to fit."""
    n_required = 2 * harmonics + 2
    if len(samples) < n_required:
        return {}

    t_raw = np.array([s.t for s in samples], dtype=float)
    span = t_raw[-1] - t_raw[0]
    t = (t_raw - t_raw[0]) / span if span > 0 else np.linspace(0.0, 1.0, len(samples))

    # Scale-normalize the centroid trajectory by the mean inter-ocular distance,
    # so the signature is invariant to subject-camera distance.
    scale = float(np.mean([np.linalg.norm(s.left_eye - s.right_eye) for s in samples]))
    scale = scale if scale > 1e-6 else 1.0

    centroids = np.stack([s.centroid for s in samples])
    centroids = (centroids - centroids.mean(axis=0)) / scale

    # Normal direction expressed as spherical angles so head rotation enters f(t).
    normals = np.stack([s.normal for s in samples])
    thetas, phis = zip(*[_normal_to_spherical(n) for n in normals])
    thetas = np.array(thetas) - np.mean(thetas)
    phis = np.unwrap(np.array(phis))
    phis = phis - np.mean(phis)

    coefs: dict = {"harmonics": int(harmonics), "n_samples": int(len(samples)), "duration_s": float(span)}
    coefs["centroid_x"] = _fourier_fit(t, centroids[:, 0], harmonics)
    coefs["centroid_y"] = _fourier_fit(t, centroids[:, 1], harmonics)
    coefs["centroid_z"] = _fourier_fit(t, centroids[:, 2], harmonics)
    coefs["normal_theta"] = _fourier_fit(t, thetas, harmonics)
    coefs["normal_phi"] = _fourier_fit(t, phis, harmonics)

    invariants = np.stack([normalized_invariants(s) for s in samples])
    coefs["invariants_mean"] = invariants.mean(axis=0)
    coefs["invariants_std"] = invariants.std(axis=0)
    return coefs


_FEATURE_KEYS = (
    "centroid_x", "centroid_y", "centroid_z",
    "normal_theta", "normal_phi",
    "invariants_mean", "invariants_std",
)


def signature_vector(coefs: dict) -> np.ndarray:
    if not coefs:
        return np.zeros(0)
    parts = [np.asarray(coefs[k], dtype=float).ravel() for k in _FEATURE_KEYS if k in coefs]
    return np.concatenate(parts) if parts else np.zeros(0)


def signature_distance(a: dict, b: dict) -> float:
    """Per-coordinate RMS distance between two signature coefficient vectors."""
    va = signature_vector(a)
    vb = signature_vector(b)
    if va.size == 0 or va.size != vb.size:
        return float("inf")
    return float(np.linalg.norm(va - vb) / np.sqrt(va.size))


def signature_hash(coefs: dict, precision: int = 3) -> str:
    """Stable short ID for a signature: quantize -> SHA-256 -> first 16 hex chars."""
    vec = signature_vector(coefs)
    if vec.size == 0:
        return ""
    quantized = np.round(vec, precision).astype(np.float64)
    return hashlib.sha256(quantized.tobytes()).hexdigest()[:16]
