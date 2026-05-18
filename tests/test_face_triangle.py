"""Unit tests for the face-triangle math. Do not require a camera or mediapipe."""

from __future__ import annotations

import numpy as np
import pytest

from face_triangle import (
    TriangleSample,
    evaluate_signature_function,
    fit_signature,
    normalized_invariants,
    signature_distance,
    signature_hash,
    signature_vector,
)


def _make(t: float, le, re, no) -> TriangleSample:
    return TriangleSample(
        t=t,
        left_eye=np.asarray(le, dtype=float),
        right_eye=np.asarray(re, dtype=float),
        nose=np.asarray(no, dtype=float),
    ).compute_derived()


def test_equilateral_triangle_geometry():
    s = _make(0.0, [0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0])
    assert s.area == pytest.approx(np.sqrt(3) / 4, abs=1e-9)
    assert s.perimeter == pytest.approx(3.0, abs=1e-9)
    for a in s.angles:
        assert a == pytest.approx(np.pi / 3, abs=1e-9)
    np.testing.assert_allclose(s.centroid, [0.5, np.sqrt(3) / 6, 0.0], atol=1e-9)


def test_right_triangle_geometry():
    s = _make(0.0, [0, 0, 0], [3, 0, 0], [0, 4, 0])
    assert s.area == pytest.approx(6.0, abs=1e-9)
    # Sides 3, 4, 5 in some order.
    np.testing.assert_allclose(sorted(s.sides.tolist()), [3.0, 4.0, 5.0], atol=1e-9)
    assert max(s.angles) == pytest.approx(np.pi / 2, abs=1e-9)


def test_invariants_are_scale_independent():
    base = ([0, 0, 0], [1, 0, 0], [0.4, 0.8, 0.0])
    small = _make(0.0, *base)
    big = _make(0.0, *(np.array(p) * 7.5 for p in base))
    np.testing.assert_allclose(
        normalized_invariants(small), normalized_invariants(big), atol=1e-9
    )


def test_invariants_vector_shape():
    s = _make(0.0, [0, 0, 0], [1, 0, 0], [0.4, 0.7, 0.1])
    inv = normalized_invariants(s)
    assert inv.shape == (6,)
    assert np.all(inv >= 0)


def _synthetic_recording(n: int = 64, seed: int = 0) -> list[TriangleSample]:
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        t = i / (n - 1)
        # Eyes wiggle laterally, nose sweeps in a small Lissajous pattern in 3D.
        left = [0.0 + 0.01 * np.sin(2 * np.pi * t), 0.0, 0.0]
        right = [1.0 + 0.01 * np.sin(2 * np.pi * t + 0.3), 0.0, 0.0]
        nose = [
            0.5 + 0.05 * np.sin(2 * np.pi * 2 * t),
            0.8 + 0.05 * np.cos(2 * np.pi * t),
            0.1 * np.sin(2 * np.pi * 3 * t),
        ]
        # Tiny noise so two recordings of the "same" subject aren't bit-identical.
        nose = [v + 1e-4 * rng.standard_normal() for v in nose]
        samples.append(_make(t, left, right, nose))
    return samples


def test_signature_self_distance_is_zero():
    sig = fit_signature(_synthetic_recording())
    assert sig
    assert signature_distance(sig, sig) == pytest.approx(0.0, abs=1e-12)


def test_signature_distance_different_subjects():
    a = fit_signature(_synthetic_recording(seed=1))
    # A "different subject" with different temporal frequencies.
    other = []
    for i in range(64):
        t = i / 63
        other.append(
            _make(
                t,
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.55 + 0.08 * np.sin(2 * np.pi * 1.0 * t),
                 0.7 + 0.03 * np.cos(2 * np.pi * 2.0 * t),
                 0.15 * np.cos(2 * np.pi * 1.5 * t)],
            )
        )
    b = fit_signature(other)
    assert signature_distance(a, b) > 1e-3


def test_signature_hash_stable_and_short():
    sig = fit_signature(_synthetic_recording())
    h1 = signature_hash(sig)
    h2 = signature_hash(sig)
    assert h1 == h2
    assert len(h1) == 16
    assert all(c in "0123456789abcdef" for c in h1)


def test_signature_vector_dimension():
    sig = fit_signature(_synthetic_recording(), harmonics=4)
    vec = signature_vector(sig)
    # 5 Fourier channels * (2K+1) coefs + 2 invariant stat vectors * 6 dims
    assert vec.size == 5 * (2 * 4 + 1) + 2 * 6


def test_fit_returns_empty_when_too_few_samples():
    samples = _synthetic_recording(n=4)
    assert fit_signature(samples, harmonics=4) == {}


def test_evaluate_signature_reconstructs_channels():
    samples = _synthetic_recording(n=128)
    sig = fit_signature(samples, harmonics=4)
    t = np.linspace(0.0, 1.0, 50)
    recon = evaluate_signature_function(sig, t)
    assert set(recon.keys()) >= {"centroid_x", "centroid_y", "centroid_z",
                                  "normal_theta", "normal_phi"}
    for v in recon.values():
        assert v.shape == (50,)
