"""Unit tests for the trapezium geometry, signature, and sigma change detector.
Do not require a camera or mediapipe."""

from __future__ import annotations

import numpy as np
import pytest

from face_trapezium import (
    Baseline,
    FEATURE_NAMES,
    TrapeziumSample,
    detect_sigma_changes,
    feature_vector,
    fit_baseline,
    signature_distance,
)


def _sample(t, le, re, lm, rm):
    return TrapeziumSample(
        t=t,
        left_eye=np.asarray(le, dtype=float),
        right_eye=np.asarray(re, dtype=float),
        left_mouth=np.asarray(lm, dtype=float),
        right_mouth=np.asarray(rm, dtype=float),
    ).compute_derived()


def _square(t=0.0):
    """Trapezium that is actually a unit square (parallelism residual = 0)."""
    return _sample(t, [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0])


def _generic_trap(t=0.0):
    """Trapezium with no two sides parallel."""
    return _sample(t, [0, 0, 0], [2.0, 0.1, 0], [0.2, 1.0, 0], [1.8, 1.3, 0])


def test_square_geometry():
    s = _square()
    np.testing.assert_allclose(s.sides, [1, 1, 1, 1], atol=1e-9)
    assert s.perimeter == pytest.approx(4.0)
    assert s.area == pytest.approx(1.0)
    np.testing.assert_allclose(s.angles, [np.pi / 2] * 4, atol=1e-9)
    np.testing.assert_allclose(s.diagonals, [np.sqrt(2), np.sqrt(2)], atol=1e-9)
    assert s.diag_ratio == pytest.approx(1.0)
    # The eye-line is the segment along y=0, the mouth-line along y=1 — they are parallel.
    assert s.parallelism_residual == pytest.approx(0.0, abs=1e-9)


def test_feature_vector_shape_and_keys():
    s = _generic_trap()
    fv = feature_vector(s)
    assert fv.shape == (len(FEATURE_NAMES),)
    assert fv.shape == (15,)


def test_feature_vector_scale_invariant():
    pts = ([0, 0, 0], [2, 0, 0], [0.3, 1, 0], [1.7, 1.2, 0])
    small = _sample(0.0, *pts)
    big = _sample(0.0, *(np.array(p) * 9.7 for p in pts))
    np.testing.assert_allclose(feature_vector(small), feature_vector(big), atol=1e-9)


def test_parallelism_residual_nonzero_for_general_trap():
    s = _generic_trap()
    assert s.parallelism_residual > 1e-3


def _baseline_recording(n=40, jitter=1e-3, seed=0):
    """A subject sitting still — tiny jitter so the std isn't pathologically zero."""
    rng = np.random.default_rng(seed)
    samples = []
    base = ([0, 0, 0], [2.0, 0.1, 0], [0.2, 1.0, 0], [1.8, 1.3, 0])
    for i in range(n):
        t = i / 30.0
        perturbed = tuple(
            np.array(p) + jitter * rng.standard_normal(3) for p in base
        )
        samples.append(_sample(t, *perturbed))
    return samples


def test_baseline_fit_and_hash():
    base = fit_baseline(_baseline_recording())
    assert base is not None
    assert base.means.shape == (15,)
    assert base.stds.shape == (15,)
    assert np.all(base.stds > 0)
    h1 = base.hash()
    h2 = base.hash()
    assert h1 == h2
    assert len(h1) == 16
    assert all(c in "0123456789abcdef" for c in h1)


def test_baseline_returns_none_on_empty():
    assert fit_baseline([]) is None
    assert fit_baseline([_square()]) is None  # only 1 sample


def test_signature_self_distance_is_zero():
    base = fit_baseline(_baseline_recording(seed=1))
    assert signature_distance(base, base) == pytest.approx(0.0)


def test_signature_distance_separates_subjects():
    a = fit_baseline(_baseline_recording(seed=1))
    # Different geometry: wider mouth, narrower eye-line.
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
    assert restored.n_samples == base.n_samples
    assert restored.hash() == base.hash()


def test_detect_no_events_when_subject_is_unchanged():
    baseline_samples = _baseline_recording(seed=10)
    base = fit_baseline(baseline_samples)
    # Re-record under the same conditions — expect no significant deviations.
    fresh = _baseline_recording(seed=11)
    report = detect_sigma_changes(fresh, base, sigma_low=3.0, sigma_high=6.0)
    assert report.overall_z.size == len(fresh)
    # Almost all aggregate z's should be small; allow the very rare blip.
    assert (report.overall_z >= 3.0).sum() <= 2
    assert all(e.sigma_level != 6 for e in report.events)


def test_detect_finds_3sigma_and_6sigma_events():
    rng = np.random.default_rng(7)
    base_pts = ([0, 0, 0], [2.0, 0.1, 0], [0.2, 1.0, 0], [1.8, 1.3, 0])
    baseline_samples = []
    for i in range(60):
        t = i / 30.0
        perturbed = tuple(np.array(p) + 5e-3 * rng.standard_normal(3) for p in base_pts)
        baseline_samples.append(_sample(t, *perturbed))
    base = fit_baseline(baseline_samples)

    # First 30 frames: baseline-like (no event). Frames 30..60: mouth corners
    # pulled apart by a large amount — should produce both 3σ and 6σ events
    # within that window.
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
    levels = {e.sigma_level for e in report.events}
    assert 3 in levels
    assert 6 in levels

    # The 6σ event must peak inside the heavy-deformation window (t >= 1.0s).
    deform_start_t = fresh[30].t
    six = next(e for e in report.events if e.sigma_level == 6)
    assert six.peak_t >= deform_start_t - 1e-6
    # No 6σ excursion should occur during the baseline-like prefix.
    for e in report.events:
        if e.sigma_level == 6:
            assert e.end_t >= deform_start_t - 1e-6

    # The 6σ event's dominant deviating feature should be one of the
    # mouth/side dimensions since that is what we perturbed.
    assert six.dominant_feature in (
        "side_mouth_norm",
        "side_left_norm",
        "side_right_norm",
        "mouth_line_norm",
        "eye_mouth_ratio",
        "angle_RM",
        "angle_LM",
        "diag_LE_RM_norm",
        "diag_RE_LM_norm",
    )
