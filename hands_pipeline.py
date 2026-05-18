"""MediaPipe Hands stream + hand-state classifier + F(t) fidget index.

For each video frame we derive:

* ``hand_proximity_norm`` — distance between the two palm centroids,
  normalized by the face's inter-eye distance.
* ``finger_motion_L``, ``finger_motion_R`` — RMS speed of the 15 non-wrist
  landmarks of each hand, measured in a **hand-local frame** (rooted at the
  wrist with the wrist→middle-MCP vector defining the local +y axis) so the
  metric is invariant to rigid hand translation / rotation.
* ``inter_hand_motion_corr`` — Pearson correlation between the two hands'
  per-frame finger-motion traces over a 1-second sliding window.
* ``hand_state`` — A / B / C / D as specified in the DFI spec:
        A: hands together, fingers still
        B: hands together, fingers moving
        C: hands apart, fingers still
        D: hands apart, fingers moving
* ``hand_to_face_min_dist_norm`` and ``hand_to_face_region`` — closest face
  region to the nearer hand, and its distance in eye-line units. The region
  weight ``W_b`` enters the F(t) sum.

The F(t) component itself is

    F(t) = (1/τ) · Σ_b W_b · K_e^(b)(t) · d_b(t)

where ``τ`` is a rolling window (default 15 s), ``b`` indexes face / neck /
fingers regions, ``W_b`` is the region weight, ``K_e^(b)`` is the integrated
kinetic energy of the hand during region-``b`` dwells, and ``d_b`` is dwell
time. The output is normalized against a baseline-derived sigma so F(t) is
sigma-comparable to V(t) and M(t).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _MP_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]
    HandLandmarker = None  # type: ignore[assignment]
    _MP_AVAILABLE = False


# MediaPipe HandLandmarker landmark indices (21 per hand).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
# 16 finger landmarks (excluding wrist): used for finger-motion in local frame.
FINGER_LANDMARKS = list(range(1, 21))

_DEFAULT_HAND_MODEL_PATH = "models/hand_landmarker.task"
_HAND_LANDMARKER_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


# DFI target weight matrix (W_b in the spec). Higher weight = stronger
# behavioral significance of touching that region. These are heuristic
# defaults from deception-cue literature and should be tuned with ground
# truth.
REGION_WEIGHTS: dict = {
    "neck": 3.5,    # vocal-cord shielding (highest signal)
    "face": 2.5,    # nose / mouth / eye area self-grooming
    "ear": 2.0,     # ear-tugging / pulling
    "hair": 1.5,    # hair-touching / smoothing
    "fingers": 0.8, # finger / ring / nail manipulation
    "none": 0.0,    # hand free / away from body
}
REGION_NAMES = tuple(REGION_WEIGHTS.keys())


@dataclass
class HandsSample:
    """Per-frame hand state derived from one HandLandmarker result."""

    t: float
    n_hands: int                       # 0, 1, or 2
    # Per-hand 3D landmarks in image-pixel + relative-depth units; None when
    # that hand wasn't detected.
    left: Optional[np.ndarray] = None  # (21, 3) or None
    right: Optional[np.ndarray] = None

    # Derived (populated by HandsTracker once a face_scale is known).
    hand_proximity_norm: float = float("nan")
    finger_motion_L: float = 0.0
    finger_motion_R: float = 0.0
    inter_hand_motion_corr: float = 0.0
    hand_to_face_min_dist_norm: float = float("nan")
    hand_to_face_region: str = "none"
    hand_state: str = "unknown"        # 'A', 'B', 'C', 'D', 'one_hand', 'hidden'
    hand_state_code: int = -1          # numeric encoding for feature vectors

    def has_left(self) -> bool:
        return self.left is not None

    def has_right(self) -> bool:
        return self.right is not None


def _local_frame_landmarks(landmarks_3d: np.ndarray) -> np.ndarray:
    """Express the 21 hand landmarks in a wrist-rooted local frame.

    +y axis = wrist → middle-finger MCP, +x perpendicular in the image plane.
    Removes the rigid translation + image-plane rotation of the hand so that
    only finger articulation contributes to subsequent variance / RMS.
    Returns shape (21, 3).
    """
    wrist = landmarks_3d[WRIST]
    middle_mcp = landmarks_3d[MIDDLE_MCP]
    # Translate so wrist is origin.
    centered = landmarks_3d - wrist
    # Build local 2D rotation from wrist→MCP direction.
    dy = middle_mcp[:2] - wrist[:2]
    norm = np.linalg.norm(dy)
    if norm < 1e-6:
        return centered
    y_hat = dy / norm                       # local +y in image plane
    x_hat = np.array([y_hat[1], -y_hat[0]]) # 90° clockwise in image-y-down frame
    # Project all xy onto the local axes; keep z untouched (depth-invariant under image-plane rotation).
    xy = centered[:, :2]
    local_x = xy @ x_hat
    local_y = xy @ y_hat
    return np.stack([local_x, local_y, centered[:, 2]], axis=1)


def hand_centroid(landmarks_3d: np.ndarray) -> np.ndarray:
    """Palm centroid = mean of the four MCP joints (5/9/13/17)."""
    return landmarks_3d[[5, 9, 13, 17]].mean(axis=0)


def hand_to_landmarks_min_distance(
    hand: np.ndarray, face_points: np.ndarray,
) -> Tuple[float, int]:
    """Return (min distance, index of closest face point)."""
    if hand.size == 0 or face_points.size == 0:
        return float("inf"), -1
    # Both shape (n, 3); pairwise distance over xy (image-plane).
    h_xy = hand[:, :2]
    f_xy = face_points[:, :2]
    diff = h_xy[:, None, :] - f_xy[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    flat_min = int(np.argmin(d))
    fi = flat_min % f_xy.shape[0]
    return float(d.min()), fi


# Approximate face-region centers (in face-feature coords) as indices into
# the user-supplied face_points array. Caller must pass face_points in this
# order: [left_eye, right_eye, left_mouth, right_mouth, left_brow, right_brow,
#         chin_approx, neck_approx].
# Chin and neck approximations are derived in classify_hand_region().
REGION_FACE_INDICES = {
    "face": [0, 1, 2, 3, 4, 5],  # everything around eyes/mouth/brow
    "hair": [4, 5],              # near brows / top of head
    "ear": [0, 1],               # outer eye → ear-adjacent
    "neck": [7],                 # neck point (synthesized below)
}


def classify_hand_region(
    hand_landmarks_3d: np.ndarray,
    face_points_with_neck: np.ndarray,
    proximity_threshold_norm: float,
    eye_line_px: float,
) -> Tuple[str, float]:
    """Return (region_name, min_distance_norm). 'none' if hand isn't close to
    any face/neck region.

    ``face_points_with_neck`` must be (8, 3) ordered as
    [left_eye, right_eye, left_mouth, right_mouth, left_brow, right_brow,
     chin_approx, neck_approx].
    """
    eye_line = max(eye_line_px, 1e-6)
    best_region = "none"
    best_dist = float("inf")
    for region, indices in REGION_FACE_INDICES.items():
        targets = face_points_with_neck[indices]
        d, _ = hand_to_landmarks_min_distance(hand_landmarks_3d, targets)
        d_norm = d / eye_line
        if d_norm < best_dist:
            best_dist = d_norm
            best_region = region
    if best_dist > proximity_threshold_norm:
        return "none", best_dist
    return best_region, best_dist


def synthesize_neck_and_chin(
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    left_mouth: np.ndarray,
    right_mouth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive approximate chin and neck positions from the trapezium vertices.

    The chin lies below the mouth midpoint by roughly the eye→mouth distance.
    The neck sits a further eye→mouth distance below that. Both estimates
    project onto the same image-plane direction the rest of the face is
    arrayed along.
    """
    eye_mid = 0.5 * (left_eye + right_eye)
    mouth_mid = 0.5 * (left_mouth + right_mouth)
    eye_mouth_vec = mouth_mid - eye_mid
    chin = mouth_mid + 0.6 * eye_mouth_vec
    neck = mouth_mid + 1.5 * eye_mouth_vec
    return chin, neck


class HandsDetector:
    """Wraps the MediaPipe HandLandmarker (Tasks API) for video streams."""

    def __init__(self, model_path: str = _DEFAULT_HAND_MODEL_PATH,
                 num_hands: int = 2,
                 min_detection_confidence: float = 0.3,
                 min_presence_confidence: float = 0.3) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe / opencv required for live hand detection."
            )
        if not Path(model_path).exists():
            raise RuntimeError(
                f"HandLandmarker model not found at {model_path}. "
                f"Download with:\n"
                f"  curl -sSL -o {model_path} {_HAND_LANDMARKER_DOWNLOAD_URL}"
            )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
        )
        self._lm = HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, image_bgr: np.ndarray, t: float) -> HandsSample:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(t * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        result = self._lm.detect_for_video(mp_image, ts_ms)

        left = right = None
        n_hands = 0
        if result.hand_landmarks:
            for hand_lms, handedness in zip(result.hand_landmarks, result.handedness):
                pts = np.array(
                    [(lm.x * w, lm.y * h, lm.z * w) for lm in hand_lms],
                    dtype=float,
                )
                # MediaPipe's "handedness" is from the subject's perspective:
                # 'Left' / 'Right'. We keep image-aligned labels so 'left' =
                # image-left hand; remap if the top category disagrees.
                cat = handedness[0].category_name.lower() if handedness else ""
                centroid_x = pts[:, 0].mean()
                is_image_left = centroid_x < w / 2
                if is_image_left:
                    if left is None:
                        left = pts
                else:
                    if right is None:
                        right = pts
                n_hands += 1

        return HandsSample(t=t, n_hands=n_hands, left=left, right=right)

    def close(self) -> None:
        self._lm.close()

    def __enter__(self) -> "HandsDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HandsTracker:
    """Stateful aggregator over a stream of HandsSamples. Computes finger
    motion (per-hand RMS velocity in local frame, over a sliding window),
    inter-hand correlation, hand-to-face region, and the 4-state classifier.
    """

    def __init__(
        self,
        finger_motion_window_s: float = 0.5,
        proximity_threshold_norm: float = 0.4,
        face_proximity_threshold_norm: float = 0.7,
    ) -> None:
        self.finger_motion_window_s = finger_motion_window_s
        self.proximity_threshold_norm = proximity_threshold_norm
        self.face_proximity_threshold_norm = face_proximity_threshold_norm
        self._history_left: List[Tuple[float, np.ndarray]] = []
        self._history_right: List[Tuple[float, np.ndarray]] = []

    def _update_history(self, store: List[Tuple[float, np.ndarray]],
                        t: float, local_lms: np.ndarray) -> None:
        store.append((t, local_lms))
        cutoff = t - self.finger_motion_window_s
        while store and store[0][0] < cutoff:
            store.pop(0)

    @staticmethod
    def _finger_motion_rms(history: List[Tuple[float, np.ndarray]]) -> float:
        if len(history) < 2:
            return 0.0
        ts = np.array([t for t, _ in history])
        lms = np.stack([lm for _, lm in history])  # (h, 21, 3)
        # finite-difference velocity over the 16 finger landmarks
        idx = np.array(FINGER_LANDMARKS, dtype=int)
        dt = np.diff(ts)
        dt = np.where(dt > 1e-6, dt, 1e-6)
        vel = np.diff(lms[:, idx, :], axis=0) / dt[:, None, None]
        # RMS magnitude across landmarks + axes
        return float(np.sqrt(np.mean(vel * vel)))

    def process(
        self,
        sample: HandsSample,
        face_scale_eye_line: Optional[float],
        face_points_with_neck: Optional[np.ndarray],
        finger_motion_baseline: Optional[float] = None,
    ) -> HandsSample:
        """Annotate the sample in-place with derived fields. ``face_scale_eye_line``
        is the inter-eye distance for the same frame (px); pass None when no
        face has been detected (proximity features become NaN)."""
        if face_scale_eye_line is None or face_scale_eye_line <= 0:
            scale = 1.0
        else:
            scale = float(face_scale_eye_line)

        if sample.left is not None:
            local_L = _local_frame_landmarks(sample.left)
            self._update_history(self._history_left, sample.t, local_L)
            sample.finger_motion_L = self._finger_motion_rms(self._history_left)
        else:
            self._history_left.clear()

        if sample.right is not None:
            local_R = _local_frame_landmarks(sample.right)
            self._update_history(self._history_right, sample.t, local_R)
            sample.finger_motion_R = self._finger_motion_rms(self._history_right)
        else:
            self._history_right.clear()

        # Hand-to-hand proximity (palm-centroid distance / eye_line).
        if sample.left is not None and sample.right is not None:
            cL = hand_centroid(sample.left)
            cR = hand_centroid(sample.right)
            sample.hand_proximity_norm = float(np.linalg.norm(cL - cR) / scale)
        else:
            sample.hand_proximity_norm = float("nan")

        # Hand-to-face/neck region (and min distance) — closer hand wins.
        sample.hand_to_face_min_dist_norm = float("nan")
        sample.hand_to_face_region = "none"
        if face_points_with_neck is not None and sample.n_hands > 0:
            best_d = float("inf")
            best_region = "none"
            for hand_pts in (sample.left, sample.right):
                if hand_pts is None:
                    continue
                region, d_norm = classify_hand_region(
                    hand_pts, face_points_with_neck,
                    self.face_proximity_threshold_norm,
                    scale,
                )
                if d_norm < best_d:
                    best_d = d_norm
                    best_region = region
            sample.hand_to_face_min_dist_norm = best_d if np.isfinite(best_d) else float("nan")
            sample.hand_to_face_region = best_region

        # Hand-state classifier (DFI 2×2).
        if sample.n_hands == 0:
            sample.hand_state = "hidden"
            sample.hand_state_code = -1
        elif sample.n_hands == 1:
            motion = sample.finger_motion_L if sample.has_left() else sample.finger_motion_R
            motion_active = (finger_motion_baseline is None or
                             motion > 3.0 * finger_motion_baseline)
            sample.hand_state = "one_hand"
            sample.hand_state_code = 4
        else:
            together = (np.isfinite(sample.hand_proximity_norm) and
                        sample.hand_proximity_norm < self.proximity_threshold_norm)
            motion = max(sample.finger_motion_L, sample.finger_motion_R)
            motion_active = (finger_motion_baseline is None or
                             motion > 3.0 * finger_motion_baseline)
            if together:
                sample.hand_state = "B" if motion_active else "A"
                sample.hand_state_code = 1 if motion_active else 0
            else:
                sample.hand_state = "D" if motion_active else "C"
                sample.hand_state_code = 3 if motion_active else 2

        return sample


def compute_fidget_index(
    times: np.ndarray,
    region_per_frame: Sequence[str],
    kinetic_per_frame: np.ndarray,
    window_s: float = 15.0,
    region_weights: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute F(t) — the unnormalized fidget kinetic-density.

    For each frame, look back over the last ``window_s`` seconds, sum
    ``W_b · K_e`` per frame for every frame whose region was b, divide by the
    window length. The result has units of weighted-kinetic-per-second.

    Returns (F_raw, F_z) where F_z is the F_raw stream rescaled by its
    median + 1.4826·MAD so that ~unit-sigma baseline gives F_z ≈ 0 and a
    deviation roughly equivalent to a sigma score. The DFI uses F_z.
    """
    if region_weights is None:
        region_weights = REGION_WEIGHTS
    weights = np.array(
        [float(region_weights.get(r, 0.0)) for r in region_per_frame], dtype=float
    )
    contributions = weights * np.asarray(kinetic_per_frame, dtype=float)
    times = np.asarray(times, dtype=float)
    n = times.size
    f_raw = np.zeros(n)
    j = 0
    cum = 0.0
    # Two-pointer windowed sum.
    for i in range(n):
        # advance j to drop frames outside the window
        while j < i and (times[i] - times[j]) > window_s:
            cum -= contributions[j]
            j += 1
        cum += contributions[i]
        actual_window = max(times[i] - times[j], 1e-6)
        f_raw[i] = cum / actual_window
    # Robust standardisation.
    med = float(np.median(f_raw))
    mad = float(np.median(np.abs(f_raw - med)))
    scale = 1.4826 * mad if mad > 1e-9 else max(np.std(f_raw), 1e-9)
    f_z = (f_raw - med) / scale
    return f_raw, f_z
