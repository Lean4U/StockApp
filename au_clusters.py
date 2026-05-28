"""AU-cluster weighting for the M(t) micro-expression component of the Syntonia score.

Groups the 22 face-trapezium features into FACS-style Action-Unit clusters and
applies per-cluster weights, producing a sigma-comparable micro-expression
intensity time series:

    M(t) = (1 + Δ_verbal) · sqrt( Σ_c S_c · z_c² / Σ_c S_c )

where ``z_c`` is the RMS of per-feature z-scores inside cluster ``c``, ``S_c``
is the cluster weight, and ``Δ_verbal ∈ {0, 1}`` is a binary multiplier set to
1 inside time windows where the verbal channel contradicts the facial
expression (when transcript alignment becomes available; defaults to 0).

Cluster weights default to literature-motivated values (see ``DEFAULT_AU_WEIGHTS``)
but the public API accepts overrides, so they're easy to tune once we have
ground-truth-labelled data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from face_trapezium import FEATURE_NAMES


# Cluster → list of face-feature names. Each face feature belongs to exactly
# one cluster (so the weighted sum is well-defined). The semantic names follow
# the Syntonia score spec: heavier weight on threat-cluster AUs.
AU_CLUSTERS: Dict[str, Tuple[str, ...]] = {
    # AU12 asymmetric lip-corner pull / contempt cue.
    "asymmetric_lip": (
        "mouth_offset_norm",
        "side_left_norm",
        "side_right_norm",
        "angle_LM",
        "angle_RM",
    ),
    # AU23 / AU24 lip compression / press / purse — captured by the
    # mouth_line / eye_mouth_ratio changing without offset asymmetry.
    "lip_compression": (
        "side_mouth_norm",
        "mouth_line_norm",
        "eye_mouth_ratio",
    ),
    # AU4 + AU7 brow-knit / squint — the anger/fear cluster.
    "brow_knit": (
        "left_brow_height_norm",
        "right_brow_height_norm",
    ),
    # Structural eye geometry — passive shape, lower weight than threat AUs.
    "eye_shape": (
        "side_eye_norm",
        "eye_line_norm",
        "angle_LE",
        "angle_RE",
    ),
    # Trapezium-level structure (diagonals, overall shape). Catch-all for
    # deformations that don't map cleanly to an AU.
    "structure": (
        "diag_LE_RM_norm",
        "diag_RE_LM_norm",
        "diag_ratio",
        "parallelism_residual",
    ),
    # Rigid head pose — low weight since head motion ≠ micro-expression.
    "pose": (
        "yaw_proxy",
        "pitch_proxy",
        "roll_proxy",
    ),
    # 4-vertex non-planarity — spikes during sneers / jaw-drops / non-rigid
    # facial deformation that the 2D features miss.
    "planarity": (
        "planarity_residual_3d",
    ),
}


# Per-cluster weights (S_AU in the Syntonia score spec). Threat-cluster AUs carry the
# heaviest weight per the spec. These values are tunable.
DEFAULT_AU_WEIGHTS: Dict[str, float] = {
    "asymmetric_lip": 2.0,   # AU12 asymmetric (contempt)
    "lip_compression": 1.5,  # AU23/24
    "brow_knit": 2.5,        # AU4+AU7 (anger/fear)
    "eye_shape": 1.0,
    "structure": 1.0,
    "pose": 0.5,             # rigid motion has reduced clinical value
    "planarity": 1.0,
}


def _validate_partition() -> None:
    """Confirm every face feature belongs to exactly one AU cluster."""
    covered = [f for fs in AU_CLUSTERS.values() for f in fs]
    if sorted(covered) != sorted(FEATURE_NAMES):
        only_in_clusters = set(covered) - set(FEATURE_NAMES)
        only_in_features = set(FEATURE_NAMES) - set(covered)
        raise RuntimeError(
            f"AU_CLUSTERS doesn't partition FEATURE_NAMES. "
            f"In clusters but not in features: {only_in_clusters}. "
            f"In features but not in clusters: {only_in_features}."
        )


_validate_partition()


# Index lookup: precomputed once, mapping cluster → list of feature column
# indices in the face feature_vector.
def _cluster_indices() -> Dict[str, np.ndarray]:
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    return {
        cluster: np.array([name_to_idx[f] for f in feats], dtype=int)
        for cluster, feats in AU_CLUSTERS.items()
    }


CLUSTER_INDICES: Dict[str, np.ndarray] = _cluster_indices()


@dataclass
class AUResult:
    """Per-frame AU cluster aggregation."""

    times: np.ndarray
    # Per-frame, per-cluster RMS z-score (n_samples × n_clusters).
    cluster_z: np.ndarray
    # Per-frame M(t) — sigma-comparable weighted aggregate.
    m: np.ndarray
    # Optional Δ_verbal multiplier applied at each frame.
    delta_verbal: np.ndarray
    cluster_names: Tuple[str, ...]
    cluster_weights: np.ndarray

    def cluster_index(self, name: str) -> int:
        return self.cluster_names.index(name)


def compute_au_clusters(
    times: np.ndarray,
    per_feature_z: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    delta_verbal: Optional[np.ndarray] = None,
) -> AUResult:
    """Aggregate per-feature z-scores into AU clusters and produce M(t).

    Parameters
    ----------
    times : shape (n,)
        Frame timestamps in seconds.
    per_feature_z : shape (n, 22)
        Per-feature z-scores from `detect_sigma_changes(...).per_feature_z`.
    weights : dict, optional
        Override the default cluster weights. Missing keys fall back to
        ``DEFAULT_AU_WEIGHTS``.
    delta_verbal : shape (n,), optional
        Binary multiplier per frame (0 or 1). Default zeros.
    """
    if weights is None:
        weights = {}
    z = np.atleast_2d(np.asarray(per_feature_z, dtype=float))
    n, k = z.shape
    if k != len(FEATURE_NAMES):
        raise ValueError(
            f"per_feature_z has {k} cols; expected {len(FEATURE_NAMES)}"
        )
    cluster_names: List[str] = list(AU_CLUSTERS.keys())
    cluster_weights = np.array(
        [float(weights.get(c, DEFAULT_AU_WEIGHTS[c])) for c in cluster_names]
    )

    cluster_z = np.zeros((n, len(cluster_names)))
    for j, cname in enumerate(cluster_names):
        idxs = CLUSTER_INDICES[cname]
        # Per-cluster RMS over its features; preserves sigma units.
        cluster_z[:, j] = np.sqrt(np.mean(z[:, idxs] ** 2, axis=1))

    sum_w = float(cluster_weights.sum())
    if sum_w <= 0:
        sum_w = 1.0
    weighted_sq = (cluster_z ** 2) @ cluster_weights
    m = np.sqrt(weighted_sq / sum_w)

    if delta_verbal is None:
        dv = np.zeros(n, dtype=float)
    else:
        dv = np.asarray(delta_verbal, dtype=float)
        if dv.shape[0] != n:
            raise ValueError("delta_verbal length must equal n_samples")
    m = m * (1.0 + dv)

    return AUResult(
        times=np.asarray(times, dtype=float),
        cluster_z=cluster_z,
        m=m,
        delta_verbal=dv,
        cluster_names=tuple(cluster_names),
        cluster_weights=cluster_weights,
    )
