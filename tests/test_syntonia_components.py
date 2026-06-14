"""Unit tests for the AU clusters, hand-pipeline math, and Syntonia assembly."""

from __future__ import annotations

import numpy as np
import pytest

from au_clusters import (
    AU_CLUSTERS,
    DEFAULT_AU_WEIGHTS,
    compute_au_clusters,
)
from face_trapezium import FEATURE_NAMES
from syntonia_model import compute_syntonia


# ---------------------------------------------------------------------------
# AU clusters
# ---------------------------------------------------------------------------

def test_au_clusters_partition_features():
    """Every face feature belongs to exactly one cluster."""
    covered = [f for fs in AU_CLUSTERS.values() for f in fs]
    assert sorted(covered) == sorted(FEATURE_NAMES)


def test_au_clusters_zero_input_gives_zero_m():
    n_features = len(FEATURE_NAMES)
    z = np.zeros((100, n_features))
    t = np.arange(100) / 30.0
    r = compute_au_clusters(t, z)
    assert r.m.shape == (100,)
    np.testing.assert_allclose(r.m, 0.0)


def test_au_clusters_unit_z_gives_unit_m():
    """If every feature sits at z=1, M(t) should equal 1 (sigma-comparable)."""
    n_features = len(FEATURE_NAMES)
    z = np.ones((50, n_features))
    t = np.arange(50) / 30.0
    r = compute_au_clusters(t, z)
    np.testing.assert_allclose(r.m, 1.0, atol=1e-9)


def test_au_clusters_brow_knit_weight_amplifies_m():
    """A brow-only shift produces a larger M(t) with the default brow-knit
    weight (2.5) than with brow-knit weight 1.0."""
    n_features = len(FEATURE_NAMES)
    z = np.zeros((10, n_features))
    brow_indices = [FEATURE_NAMES.index(f) for f in AU_CLUSTERS["brow_knit"]]
    for i in brow_indices:
        z[:, i] = 3.0
    t = np.arange(10) / 30.0
    r_default = compute_au_clusters(t, z)
    r_flat = compute_au_clusters(t, z, weights={"brow_knit": 1.0})
    assert r_default.m.mean() > r_flat.m.mean()


def test_au_clusters_delta_verbal_doubles_m():
    n_features = len(FEATURE_NAMES)
    z = np.ones((20, n_features))
    t = np.arange(20) / 30.0
    dv = np.ones(20)
    r = compute_au_clusters(t, z, delta_verbal=dv)
    # With delta_verbal = 1 at every frame, M(t) = (1+1) · 1 = 2.
    np.testing.assert_allclose(r.m, 2.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Hand-pipeline math (the helper functions; the full HandsTracker needs a
# real video stream and is exercised in scripts/process_video.py)
# ---------------------------------------------------------------------------

def test_local_frame_landmarks_invariance_under_translation():
    from hands_pipeline import _local_frame_landmarks
    rng = np.random.default_rng(0)
    base = rng.standard_normal((21, 3)) * 50.0
    # Force wrist→MCP direction to be along +y in image plane for simplicity.
    base[0] = [100, 100, 0]
    base[9] = [100, 200, 0]
    shifted = base + np.array([300.0, -50.0, 5.0])  # rigid translation
    local_a = _local_frame_landmarks(base)
    local_b = _local_frame_landmarks(shifted)
    np.testing.assert_allclose(local_a, local_b, atol=1e-9)


def test_local_frame_landmarks_invariance_under_rotation():
    from hands_pipeline import _local_frame_landmarks
    rng = np.random.default_rng(1)
    base = rng.standard_normal((21, 3)) * 50.0
    base[0] = [100, 100, 0]
    base[9] = [100, 200, 0]
    theta = np.deg2rad(35.0)
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
    rotated = base.copy()
    rotated[:, :2] = base[:, :2] @ R2.T
    local_a = _local_frame_landmarks(base)
    local_b = _local_frame_landmarks(rotated)
    np.testing.assert_allclose(local_a, local_b, atol=1e-6)


def test_hand_centroid():
    from hands_pipeline import hand_centroid
    pts = np.zeros((21, 3))
    pts[5] = [10, 0, 0]
    pts[9] = [10, 10, 0]
    pts[13] = [0, 10, 0]
    pts[17] = [0, 0, 0]
    c = hand_centroid(pts)
    np.testing.assert_allclose(c, [5, 5, 0])


def test_effective_region_falls_back_to_hand_state():
    """When no hand-to-face contact is happening, the effective_region used
    by F(t) must reflect the hand-to-hand state — otherwise hand fidgeting
    that never reaches the face is silently ignored."""
    from hands_pipeline import HandsTracker, HandsSample
    import numpy as np

    tracker = HandsTracker(proximity_threshold_norm=0.4,
                           face_proximity_threshold_norm=0.7)
    # Two hands close to each other, neither close to the face — placed well
    # below the synthesised neck so face/neck thresholds (0.7 * eye_line = 70 px)
    # cannot reach them.
    hand_L = np.zeros((21, 3))
    hand_L[0] = [100, 700, 0]  # wrist, well below neck
    hand_L[9] = [100, 710, 0]
    hand_R = hand_L + np.array([20, 0, 0])  # 20 px apart in lap

    face_points = np.array([
        [50, 200, 0],   # left eye
        [150, 200, 0],  # right eye
        [70, 300, 0],   # left mouth
        [130, 300, 0],  # right mouth
        [50, 180, 0],   # left brow
        [150, 180, 0],  # right brow
        [100, 360, 0],  # chin
        [100, 450, 0],  # neck
    ])

    sample = HandsSample(t=0.0, n_hands=2, left=hand_L, right=hand_R)
    tracker.process(sample, face_scale_eye_line=100.0, face_points_with_neck=face_points,
                    finger_motion_baseline=None)
    # Hands are together (20/100 = 0.2 < 0.4 threshold) — state A or B —
    # AND no face contact. So effective_region should be "clasp_still" or
    # "fingers", never "none".
    assert sample.hand_to_face_region == "none"
    assert sample.effective_region in {"clasp_still", "fingers"}


def test_synthesize_neck_and_chin():
    from hands_pipeline import synthesize_neck_and_chin
    le, re = np.array([0, 0, 0]), np.array([100, 0, 0])
    lm, rm = np.array([20, 100, 0]), np.array([80, 100, 0])
    chin, neck = synthesize_neck_and_chin(le, re, lm, rm)
    # Both should sit *below* the mouth line (larger y in image coords).
    assert chin[1] > 100.0
    assert neck[1] > chin[1]


# ---------------------------------------------------------------------------
# Fidget-index time-windowed math
# ---------------------------------------------------------------------------

def test_compute_fidget_index_zero_when_no_contact():
    from hands_pipeline import compute_fidget_index
    n = 100
    times = np.arange(n) / 30.0
    regions = ["none"] * n
    ke = np.zeros(n)
    f_raw, f_z = compute_fidget_index(times, regions, ke, window_s=2.0)
    np.testing.assert_allclose(f_raw, 0.0)


def test_compute_fidget_index_burst_changes_z():
    from hands_pipeline import compute_fidget_index
    n = 200
    times = np.arange(n) / 30.0
    regions = ["none"] * n
    ke = np.zeros(n)
    # A 1-second burst of fingers-touching activity in the middle.
    for i in range(100, 130):
        regions[i] = "fingers"
        ke[i] = 1.0
    f_raw, f_z = compute_fidget_index(times, regions, ke, window_s=2.0)
    # Burst frames should sit well above the rest of the trace.
    burst_mean = f_z[100:130].mean()
    pre_mean = f_z[:90].mean()
    assert burst_mean > pre_mean + 0.5


# ---------------------------------------------------------------------------
# Syntonia assembly
# ---------------------------------------------------------------------------

def test_dfi_zero_inputs_zero_output():
    target = np.arange(60) / 10.0
    r = compute_syntonia(target,
                    None, None,
                    None, None,
                    None, None)
    np.testing.assert_allclose(r.syntonia, 0.0)
    assert r.windows == []


def test_dfi_threshold_crossings():
    """All three channels constant at 3.0 → Syntonia = 3.0 throughout, threshold met."""
    target = np.linspace(0, 30, 301)
    v = np.full_like(target, 3.0)
    f = np.full_like(target, 3.0)
    m = np.full_like(target, 3.0)
    r = compute_syntonia(target,
                    target, v,
                    target, f,
                    target, m,
                    alpha=1.0 / 3.0, beta=1.0 / 3.0, gamma=1.0 / 3.0,
                    threshold=3.0, window_s=5.0)
    # The rolling mean of the constant signal should equal 3.0 across the
    # entire span, so the whole thing is one window.
    np.testing.assert_allclose(r.syntonia, 3.0, atol=1e-9)
    assert len(r.windows) == 1
    w = r.windows[0]
    assert w.peak_syntonia == pytest.approx(3.0, abs=1e-9)


def test_dfi_isolated_spike_below_window():
    """A single 0.1-second 10σ spike with quiet surroundings will not bring the
    5-second rolling mean above 3.0."""
    target = np.linspace(0, 30, 301)
    m = np.zeros_like(target)
    m[150] = 10.0
    r = compute_syntonia(target,
                    None, None,
                    None, None,
                    target, m,
                    alpha=0.0, beta=0.0, gamma=1.0,
                    threshold=3.0, window_s=5.0)
    # Rolling mean over 5s with a single spike of 10 contributing ~10/50
    # samples = 0.2, never breaches threshold.
    assert r.windows == []


def test_dfi_weights_must_be_respected():
    """If γ=1 and the M channel is at 6.0 sustained, Syntonia = 6.0."""
    target = np.linspace(0, 20, 201)
    m = np.full_like(target, 6.0)
    r = compute_syntonia(target,
                    None, None,
                    None, None,
                    target, m,
                    alpha=0.0, beta=0.0, gamma=1.0)
    np.testing.assert_allclose(r.syntonia, 6.0, atol=1e-9)


def test_dfi_dominant_component_reported_correctly():
    """A pure M(t) excursion should be flagged as dominant_component='M'."""
    target = np.linspace(0, 30, 301)
    m = np.zeros_like(target)
    m[100:200] = 9.0  # sustained
    r = compute_syntonia(target,
                    None, None,
                    None, None,
                    target, m,
                    alpha=1.0 / 3.0, beta=1.0 / 3.0, gamma=1.0 / 3.0,
                    threshold=3.0, window_s=5.0)
    assert r.windows
    assert r.windows[0].dominant_component == "M"
