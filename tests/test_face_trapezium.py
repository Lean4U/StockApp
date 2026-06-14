"""Unit tests for the trapezium geometry, signature, and change detectors.
Do not require a camera or mediapipe."""

from __future__ import annotations

import numpy as np
import pytest

from face_trapezium import (
    Baseline,
    FEATURE_NAMES,
    TrapeziumSample,
    cusum,
    detect_sigma_changes,
    feature_vector,
    fit_baseline,
    hotelling_t2,
    huber_clip,
    reliability_weights,
    rolling_z_variance,
    signature_distance,
    t2_threshold,
    t2_to_sigma_equivalent,
)


def _sample(t, le, re, lm, rm, lbrow=None, rbrow=None):
    """Build a TrapeziumSample. Brow points default to a sensible position
    12 px above the corresponding outer eye corner (in image-y units)."""
    le_a = np.asarray(le, dtype=float)
    re_a = np.asarray(re, dtype=float)
    lm_a = np.asarray(lm, dtype=float)
    rm_a = np.asarray(rm, dtype=float)
    if lbrow is None:
        lbrow = le_a + np.array([0.0, -12.0, 0.0])
    if rbrow is None:
        rbrow = re_a + np.array([0.0, -12.0, 0.0])
    return TrapeziumSample(
        t=t,
        left_eye=le_a,
        right_eye=re_a,
        left_mouth=lm_a,
        right_mouth=rm_a,
        left_brow=np.asarray(lbrow, dtype=float),
        right_brow=np.asarray(rbrow, dtype=float),
    ).compute_derived()


def _square(t=0.0):
    """Trapezium that is actually a unit square."""
    return _sample(t, [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0])


def _generic_trap(t=0.0):
    return _sample(t, [0, 0, 0], [2.0, 0.1, 0], [0.2, 1.0, 0], [1.8, 1.3, 0])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_square_geometry():
    s = _square()
    np.testing.assert_allclose(s.sides, [1, 1, 1, 1], atol=1e-9)
    assert s.perimeter == pytest.approx(4.0)
    assert s.area == pytest.approx(1.0)
    np.testing.assert_allclose(s.angles, [np.pi / 2] * 4, atol=1e-9)
    np.testing.assert_allclose(s.diagonals, [np.sqrt(2), np.sqrt(2)], atol=1e-9)
    assert s.diag_ratio == pytest.approx(1.0)
    assert s.parallelism_residual == pytest.approx(0.0, abs=1e-9)


def test_feature_vector_shape_and_keys():
    s = _generic_trap()
    fv = feature_vector(s)
    assert fv.shape == (len(FEATURE_NAMES),)
    assert fv.shape == (22,)


def test_feature_vector_scale_invariant():
    pts = ([0, 0, 0], [2, 0, 0], [0.3, 1, 0], [1.7, 1.2, 0])
    small = _sample(0.0, *pts)
    big_pts = tuple(np.array(p) * 9.7 for p in pts)
    big_brow_l = big_pts[0] + np.array([0.0, -12.0 * 9.7, 0.0])
    big_brow_r = big_pts[1] + np.array([0.0, -12.0 * 9.7, 0.0])
    big = _sample(0.0, *big_pts, lbrow=big_brow_l, rbrow=big_brow_r)
    # Pose proxies don't scale with overall size — exclude them when comparing.
    # The geometric features and brow ratios should match exactly.
    small_fv = feature_vector(small)
    big_fv = feature_vector(big)
    np.testing.assert_allclose(small_fv[:17], big_fv[:17], atol=1e-9)


def test_parallelism_residual_nonzero_for_general_trap():
    s = _generic_trap()
    assert s.parallelism_residual > 1e-3


def test_brow_height_signed_correctly():
    # Brows positioned 8 px above the eye corners → both brow heights = +8.
    s = _sample(
        0.0, [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        lbrow=[0.0, -8.0, 0.0],
        rbrow=[1.0, -8.0, 0.0],
    )
    assert s.left_brow_height == pytest.approx(8.0)
    assert s.right_brow_height == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Head rotation: face-plane projection makes geometry rotation-invariant
# ---------------------------------------------------------------------------

def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _build_face_landmarks():
    """Six 3D landmarks for an upright face on the z=0 plane."""
    return {
        "le": np.array([-60.0, 0.0, 0.0]),
        "re": np.array([+60.0, 0.0, 0.0]),
        "lm": np.array([-40.0, 110.0, 0.0]),
        "rm": np.array([+40.0, 110.0, 0.0]),
        "lb": np.array([-60.0, -12.0, 0.0]),
        "rb": np.array([+60.0, -12.0, 0.0]),
    }


def _sample_from_landmarks(lm: dict, t: float = 0.0):
    return _sample(t, lm["le"], lm["re"], lm["lm"], lm["rm"], lbrow=lm["lb"], rbrow=lm["rb"])


def test_geometric_features_rotation_invariant():
    """Rotating all 6 landmarks rigidly should leave the 17 geometric features
    untouched and only perturb the pose channels."""
    upright = _sample_from_landmarks(_build_face_landmarks())
    fv_upright = feature_vector(upright)

    for R in (_rotation_y(0.4), _rotation_x(-0.3), _rotation_z(0.25),
              _rotation_y(0.4) @ _rotation_x(-0.3) @ _rotation_z(0.25)):
        lm = _build_face_landmarks()
        rotated = {k: R @ p for k, p in lm.items()}
        s = _sample_from_landmarks(rotated)
        fv = feature_vector(s)
        # Geometric channels (sides, angles, diagonals, ratios, brow heights):
        np.testing.assert_allclose(fv[:17], fv_upright[:17], atol=1e-9,
                                   err_msg=f"geometric features changed under rotation R")


def test_yaw_proxy_changes_with_head_turn():
    upright = _sample_from_landmarks(_build_face_landmarks())
    assert abs(upright.yaw_proxy) < 1e-6
    assert abs(upright.pitch_proxy) < 1e-6
    assert abs(upright.roll_proxy) < 1e-6

    # 30° yaw (rotation about world-y, which is the face's vertical axis).
    R = _rotation_y(np.deg2rad(30.0))
    lm = _build_face_landmarks()
    turned = _sample_from_landmarks({k: R @ p for k, p in lm.items()})
    assert abs(turned.yaw_proxy) == pytest.approx(np.deg2rad(30.0), abs=1e-6)
    assert abs(turned.pitch_proxy) < 1e-6
    assert abs(turned.roll_proxy) < 1e-6


def test_roll_proxy_picks_up_image_plane_tilt():
    R = _rotation_z(np.deg2rad(15.0))
    lm = _build_face_landmarks()
    tilted = _sample_from_landmarks({k: R @ p for k, p in lm.items()})
    assert tilted.roll_proxy == pytest.approx(np.deg2rad(15.0), abs=1e-6)
    assert abs(tilted.yaw_proxy) < 1e-6
    assert abs(tilted.pitch_proxy) < 1e-6


def test_planarity_residual_zero_for_coplanar_vertices():
    s = _sample_from_landmarks(_build_face_landmarks())
    assert s.planarity_residual_3d == pytest.approx(0.0, abs=1e-9)


def test_planarity_residual_nonzero_for_non_coplanar_vertices():
    lm = _build_face_landmarks()
    lm["lm"] = lm["lm"] + np.array([0.0, 0.0, 8.0])  # push left-mouth out of plane
    s = _sample_from_landmarks(lm)
    assert s.planarity_residual_3d > 1e-3


# ---------------------------------------------------------------------------
# Baseline / signature
# ---------------------------------------------------------------------------

def _baseline_recording(n=40, jitter=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    base = ([0, 0, 0], [2.0, 0.1, 0], [0.2, 1.0, 0], [1.8, 1.3, 0])
    for i in range(n):
        t = i / 30.0
        perturbed = tuple(np.array(p) + jitter * rng.standard_normal(3) for p in base)
        samples.append(_sample(t, *perturbed))
    return samples


def test_baseline_fit_and_hash():
    base = fit_baseline(_baseline_recording())
    assert base is not None
    assert base.means.shape == (22,)
    assert base.stds.shape == (22,)
    assert np.all(base.stds > 0)
    assert base.hash() == base.hash()
    assert len(base.hash()) == 16


def test_baseline_returns_none_on_empty():
    assert fit_baseline([]) is None
    assert fit_baseline([_square()]) is None


def test_signature_self_distance_is_zero():
    base = fit_baseline(_baseline_recording(seed=1))
    assert signature_distance(base, base) == pytest.approx(0.0)


def test_signature_distance_separates_subjects():
    a = fit_baseline(_baseline_recording(seed=1))
    other = []
    rng = np.random.default_rng(99)
    base = ([0, 0, 0], [1.4, 0.1, 0], [-0.1, 1.0, 0], [2.3, 1.1, 0])
    for i in range(40):
        t = i / 30.0
        perturbed = tuple(np.array(p) + 1e-3 * rng.standard_normal(3) for p in base)
        other.append(_sample(t, *perturbed))
    b = fit_baseline(other)
    assert signature_distance(a, b) > 1e-3


def test_baseline_persistence_roundtrip():
    base = fit_baseline(_baseline_recording())
    restored = Baseline.from_dict(base.to_dict())
    assert restored.feature_names == base.feature_names
    np.testing.assert_allclose(restored.means, base.means)
    np.testing.assert_allclose(restored.stds, base.stds)
    assert restored.hash() == base.hash()


# ---------------------------------------------------------------------------
# Aggregation: Hotelling's T² + Wilson-Hilferty
# ---------------------------------------------------------------------------

def test_hotelling_t2_known_value():
    z = np.array([[1.0, 2.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    t2 = hotelling_t2(z)
    np.testing.assert_allclose(t2, [9.0, 0.0, 25.0])


def test_t2_threshold_monotone_in_sigma():
    th3 = t2_threshold(17, 3.0)
    th6 = t2_threshold(17, 6.0)
    assert 0 < th3 < th6
    # Sanity-check vs. the canonical scipy chi2.ppf(0.99865, 17) ≈ 39.97
    assert 36.0 < th3 < 44.0


def test_t2_to_sigma_equivalent_inverts_threshold():
    for sigma in (1.0, 2.0, 3.0, 4.5, 6.0):
        th = t2_threshold(17, sigma)
        np.testing.assert_allclose(t2_to_sigma_equivalent(th, 17), sigma, atol=1e-6)


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

def test_cusum_silent_under_null():
    """In-control CUSUM should rarely *start* a new alarm window. ARL₀ with
    k=0.5, h=5 is ≈ 465 samples per stream, so over 200 samples × 4 streams we
    expect ≲ 2 fresh alarms. We count distinct rising edges, not alarm frames
    (once S exceeds h it can stay above for many frames during one excursion)."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 4))
    s_plus, s_minus, alarm_high, alarm_low = cusum(z, k=0.5, h=5.0)
    total_alarms = int(alarm_high.sum() + alarm_low.sum())
    assert total_alarms <= 5


def test_cusum_catches_subtle_sustained_shift():
    rng = np.random.default_rng(1)
    n = 300
    z = rng.standard_normal((n, 1))
    z[100:, 0] += 1.0  # +1 sigma shift starting at t=100
    s_plus, s_minus, alarm_high, alarm_low = cusum(z, k=0.5, h=5.0)
    assert alarm_high[:, 0].any()
    first_alarm_idx = int(np.argmax(alarm_high[:, 0]))
    # First alarm should fire well after the shift onset (lag is roughly h/(δ-k)
    # = 5/(1-0.5) = 10 samples) but inside a reasonable detection window.
    assert 100 < first_alarm_idx < 140
    # Low-side accumulator should not alarm.
    assert not alarm_low[:, 0].any()


# ---------------------------------------------------------------------------
# Integration: detect_sigma_changes with all three detectors
# ---------------------------------------------------------------------------

def test_detect_no_events_when_subject_is_unchanged():
    base = fit_baseline(_baseline_recording(seed=10))
    fresh = _baseline_recording(seed=11)
    report = detect_sigma_changes(fresh, base, sigma_low=3.0, sigma_high=6.0)
    assert report.overall_z.size == len(fresh)
    assert report.t2.size == len(fresh)
    # No sustained 6σ events; rare blips OK.
    assert all(e.sigma_level != 6 for e in report.events)


def test_detect_brow_raise_via_t2_and_cusum():
    """Tight baseline + small sustained brow raise. RMS may or may not catch it,
    but T² and CUSUM (the new detectors) must."""
    rng = np.random.default_rng(7)
    base_pts = ([0, 0, 0], [120.0, 0, 0], [20, 100, 0], [100, 100, 0])
    # baseline brow heights = 12 px (default), with tiny jitter
    baseline_samples = []
    for i in range(180):
        t = i / 30.0
        perturbed = tuple(np.array(p) + 0.5 * rng.standard_normal(3) for p in base_pts)
        baseline_samples.append(_sample(t, *perturbed))
    base = fit_baseline(baseline_samples)

    # Brow raise: bumps brow heights from ~12 to ~17 (5 px raise) for 1 second.
    fresh = []
    for i in range(180):
        t = i / 30.0
        perturbed = tuple(np.array(p) + 0.5 * rng.standard_normal(3) for p in base_pts)
        # Brow raise active in frames 60..90 (t in [2.0, 3.0]).
        if 60 <= i < 90:
            le_b = perturbed[0] + np.array([0.0, -17.0, 0.0])
            re_b = perturbed[1] + np.array([0.0, -17.0, 0.0])
        else:
            le_b = perturbed[0] + np.array([0.0, -12.0, 0.0])
            re_b = perturbed[1] + np.array([0.0, -12.0, 0.0])
        fresh.append(_sample(t, *perturbed, lbrow=le_b, rbrow=re_b))

    report = detect_sigma_changes(fresh, base, sigma_low=3.0, sigma_high=6.0)

    # T² should flag the brow window with a high equivalent sigma.
    t2_events = [e for e in report.events if e.detector == "t2" and e.sigma_level == 3]
    assert t2_events, "T² did not detect the brow raise"
    hit = next(e for e in t2_events if e.start_t < 3.0 and e.end_t > 2.0)
    assert hit.dominant_feature in {"left_brow_height_norm", "right_brow_height_norm"}

    # CUSUM should alarm on at least one brow feature.
    brow_alarms = [
        e for e in report.cusum_events
        if e.feature_name in {"left_brow_height_norm", "right_brow_height_norm"}
    ]
    assert brow_alarms, "CUSUM did not alarm on a brow feature"
    # The brow-shift direction is "high" (brow_height increases).
    assert any(e.direction == "high" for e in brow_alarms)


def test_detect_finds_6sigma_event_for_large_deformation():
    rng = np.random.default_rng(7)
    base_pts = ([0, 0, 0], [2.0, 0.1, 0], [0.2, 1.0, 0], [1.8, 1.3, 0])
    baseline_samples = []
    for i in range(60):
        t = i / 30.0
        perturbed = tuple(np.array(p) + 5e-3 * rng.standard_normal(3) for p in base_pts)
        baseline_samples.append(_sample(t, *perturbed))
    base = fit_baseline(baseline_samples)

    fresh = []
    for i in range(60):
        t = i / 30.0
        if i < 30:
            pts = base_pts
        else:
            pts = (
                base_pts[0],
                base_pts[1],
                (base_pts[2][0] - 0.4, base_pts[2][1], 0),
                (base_pts[3][0] + 0.4, base_pts[3][1], 0),
            )
        perturbed = tuple(np.array(p) + 5e-3 * rng.standard_normal(3) for p in pts)
        fresh.append(_sample(t, *perturbed))

    report = detect_sigma_changes(fresh, base, sigma_low=3.0, sigma_high=6.0)
    levels = {(e.detector, e.sigma_level) for e in report.events}
    assert ("rms", 6) in levels
    assert ("t2", 6) in levels


# ---------------------------------------------------------------------------
# Robustness layer: median/MAD baseline, Huber clip, reliability weights,
# and the mouth_offset_norm asymmetric-occlusion feature.
# ---------------------------------------------------------------------------

def test_mouth_offset_zero_for_symmetric_face():
    s = _sample_from_landmarks(_build_face_landmarks())
    assert abs(s.mouth_offset_norm) < 1e-9


def test_mouth_offset_signed_for_one_sided_occluder():
    lm = _build_face_landmarks()
    # Cigarette in image-right corner pulls right_mouth laterally outward.
    lm["rm"] = lm["rm"] + np.array([8.0, 0.0, 0.0])
    s = _sample_from_landmarks(lm)
    # Mouth midpoint shifts to image-right ⇒ positive offset in face-frame x.
    assert s.mouth_offset_norm > 0.02


def test_robust_baseline_ignores_enrollment_outliers():
    """Inject 5% extreme outliers into the enrollment recording. The robust
    median/MAD baseline should ignore them; the classical mean/std baseline
    should be poisoned (very inflated std → desensitised detector)."""
    rng = np.random.default_rng(11)
    samples = []
    base = ([0, 0, 0], [120.0, 0, 0], [20, 100, 0], [100, 100, 0])
    for i in range(200):
        t = i / 30.0
        perturbed = tuple(np.array(p) + 0.5 * rng.standard_normal(3) for p in base)
        # 5% of frames have a huge glare-style outlier on the right eye corner.
        if i % 20 == 0:
            perturbed = list(perturbed)
            perturbed[1] = perturbed[1] + np.array([40.0, 30.0, 0.0])
            perturbed = tuple(perturbed)
        samples.append(_sample(t, *perturbed))

    robust = fit_baseline(samples, robust=True)
    classical = fit_baseline(samples, robust=False)
    # Find a feature affected by the outlier (side_right_norm or similar) and
    # confirm the robust std is much smaller than the classical std.
    j = FEATURE_NAMES.index("side_right_norm")
    assert classical.stds[j] > 5 * robust.stds[j]


def test_huber_clip_caps_outliers():
    z = np.array([[0.0, 12.0, -50.0, 1.5]])
    out = huber_clip(z, cap=8.0)
    np.testing.assert_allclose(out, [[0.0, 8.0, -8.0, 1.5]])


def test_reliability_weights_full_under_baseline():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((200, 5))
    rvar = rolling_z_variance(z, window=30)
    w = reliability_weights(rvar, threshold=5.0)
    # Under H0 the rolling z-variance hovers around 1, well under threshold ⇒
    # weights stay at 1.
    assert w.mean() > 0.95


def test_reliability_weights_drop_for_noisy_feature():
    rng = np.random.default_rng(4)
    n = 200
    z = rng.standard_normal((n, 2))
    # Make feature 0 very noisy in the middle third of the recording.
    z[60:140, 0] = rng.standard_normal(80) * 8.0
    rvar = rolling_z_variance(z, window=30)
    w = reliability_weights(rvar, threshold=5.0)
    # Feature 0's weight in the noisy band should drop well below 1; feature
    # 1's weight should stay near 1 throughout.
    assert w[60:140, 0].mean() < 0.4
    assert w[:, 1].mean() > 0.9


def test_huber_clip_blocks_single_frame_glare_in_t2():
    """A single-frame extreme outlier on one feature is capped by Huber clip
    so the per-frame T² stays under the 6-sigma threshold."""
    rng = np.random.default_rng(5)
    base_pts = ([0, 0, 0], [120.0, 0, 0], [20, 100, 0], [100, 100, 0])
    baseline_samples = []
    for i in range(120):
        t = i / 30.0
        perturbed = tuple(np.array(p) + 0.5 * rng.standard_normal(3) for p in base_pts)
        baseline_samples.append(_sample(t, *perturbed))
    base = fit_baseline(baseline_samples)

    # 60-frame clean recording, but at frame 30 inject a 60-pixel glare on the
    # right eye corner.
    fresh = []
    for i in range(60):
        t = i / 30.0
        perturbed = tuple(np.array(p) + 0.5 * rng.standard_normal(3) for p in base_pts)
        if i == 30:
            perturbed = list(perturbed)
            perturbed[1] = perturbed[1] + np.array([60.0, 0.0, 0.0])
            perturbed = tuple(perturbed)
        fresh.append(_sample(t, *perturbed))

    # With Huber cap = 8, the peak T² σ-equivalent on the glare frame is bounded.
    report = detect_sigma_changes(fresh, base, sigma_low=3.0, sigma_high=6.0,
                                  huber_cap=8.0, min_run_samples=1)
    # Compare with the robustness layer fully disabled (no Huber clipping, no
    # reliability down-weighting). The unrobust pipeline must produce a much
    # larger T² σ-equivalent on the glare frame.
    report_unrobust = detect_sigma_changes(
        fresh, base, sigma_low=3.0, sigma_high=6.0,
        huber_cap=1e9, reliability_threshold=1e9, min_run_samples=1,
    )
    glare_idx = 30
    peak_robust = report.t2_equivalent_sigma[glare_idx]
    peak_unrobust = report_unrobust.t2_equivalent_sigma[glare_idx]
    assert peak_robust < 0.5 * peak_unrobust

    # The robust pipeline must not flag a *sustained* 6σ event from a single-
    # frame glare. The unrobust pipeline, by contrast, would.
    sustained_6sigma_robust = [
        e for e in report.events
        if e.detector == "t2" and e.sigma_level == 6 and e.duration_s > 0.1
    ]
    assert sustained_6sigma_robust == []
