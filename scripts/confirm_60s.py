"""60-second end-to-end confirmation of the trapezium signature + change detectors.

The sandbox has no webcam. Instead, this script generates a physically-grounded
60-second / 30-fps recording of the four trapezium vertices **and** the two
eyebrow landmarks with:

  * static neutral geometry placed in pixel coordinates that match a typical
    640×480 webcam frame at ~50 cm from the camera (inter-canthal width ≈ 120 px,
    eye-to-mouth ≈ 110 px, brow-to-eye ≈ 12 px),
  * per-frame tracking jitter ~ N(0, 0.6 px) (the empirical FaceMesh noise floor),
  * slow head translation (low-frequency drift + a small respiratory rhythm),
  * four labelled expression events with Gaussian envelopes:
        smile        @ t = 21 s  (σ = 2.0 s, mouth corners ±8 px out, 4 px up)
        brow raise   @ t = 35 s  (σ = 1.0 s, both brows raised 5 px)
        squint       @ t = 44 s  (σ = 0.8 s, brows lowered 3 px - small signal)
        yawn         @ t = 52 s  (σ = 1.5 s, mouth corners ±15 px out, 12 px down)
  * a labelled **head turn** (rigid rotation about the world-y axis) at t = 28 s
    (σ = 1.0 s, peak yaw = 20°). The rotation is applied to all 6 landmarks
    rigidly, so it should affect ONLY the head-pose features
    (`yaw_proxy`, `pitch_proxy`, `roll_proxy`) and leave the 17 geometric
    features untouched — verifying the face-plane rotation invariance.

The first 10 s are used as the enrollment baseline. The remaining 50 s are
overlaid against that baseline by `detect_sigma_changes`, which now runs
three statistical detectors in parallel: RMS aggregate |z|, Hotelling's T²
(chi-square multivariate test, more sensitive than RMS for diffuse shifts),
and per-feature tabular CUSUM (designed for small sustained shifts).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from face_trapezium import (  # noqa: E402
    FEATURE_NAMES,
    TrapeziumSample,
    detect_sigma_changes,
    fit_baseline,
)


FPS = 30
DURATION_S = 60.0
N = int(FPS * DURATION_S)

LE0 = np.array([260.0, 220.0, 0.0])
RE0 = np.array([380.0, 220.0, 0.0])
LM0 = np.array([282.0, 330.0, 0.0])
RM0 = np.array([358.0, 330.0, 0.0])
LBROW0 = np.array([260.0, 208.0, 0.0])  # 12 px above the corresponding eye corner
RBROW0 = np.array([380.0, 208.0, 0.0])

TRACKING_JITTER_PX = 0.6
BASELINE_END_S = 10.0


def head_motion(t: np.ndarray) -> np.ndarray:
    dx = 8.0 * np.sin(2 * np.pi * 0.07 * t) + 2.0 * np.sin(2 * np.pi * 0.5 * t)
    dy = 3.0 * np.cos(2 * np.pi * 0.05 * t) + 1.5 * np.sin(2 * np.pi * 0.25 * t)
    dz = 2.0 * np.sin(2 * np.pi * 0.12 * t)
    return np.column_stack([dx, dy, dz])


def gaussian_envelope(t: np.ndarray, t0: float, sigma_s: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - t0) / sigma_s) ** 2)


def _rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def build_recording(seed: int = 42):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FPS
    motion = head_motion(t)

    def jitter():
        return TRACKING_JITTER_PX * rng.standard_normal((N, 3))

    le = LE0 + motion + jitter()
    re = RE0 + motion + jitter()
    lm = LM0 + motion + jitter()
    rm = RM0 + motion + jitter()
    lbrow = LBROW0 + motion + jitter()
    rbrow = RBROW0 + motion + jitter()

    env_smile = gaussian_envelope(t, 21.0, 2.0)
    env_brow = gaussian_envelope(t, 35.0, 1.0)
    env_squint = gaussian_envelope(t, 44.0, 0.8)
    env_yawn = gaussian_envelope(t, 52.0, 1.5)
    env_turn = gaussian_envelope(t, 28.0, 1.0)

    lm[:, 0] += -8.0 * env_smile
    lm[:, 1] += -4.0 * env_smile
    rm[:, 0] += +8.0 * env_smile
    rm[:, 1] += -4.0 * env_smile

    lbrow[:, 1] += -5.0 * env_brow
    rbrow[:, 1] += -5.0 * env_brow

    lbrow[:, 1] += +3.0 * env_squint
    rbrow[:, 1] += +3.0 * env_squint

    lm[:, 0] += -15.0 * env_yawn
    lm[:, 1] += +12.0 * env_yawn
    rm[:, 0] += +15.0 * env_yawn
    rm[:, 1] += +12.0 * env_yawn

    # Apply a rigid head-yaw rotation to all 6 landmarks centered on the face
    # centroid. This should perturb yaw_proxy but NOT any geometric feature.
    samples: List = []
    yaw_peak = np.deg2rad(20.0)
    for i in range(N):
        face_center = 0.25 * (le[i] + re[i] + lm[i] + rm[i])
        R = _rotation_y(yaw_peak * env_turn[i])

        def rot(p):
            return R @ (p - face_center) + face_center

        samples.append(
            TrapeziumSample(
                t=float(t[i]),
                left_eye=rot(le[i]),
                right_eye=rot(re[i]),
                left_mouth=rot(lm[i]),
                right_mouth=rot(rm[i]),
                left_brow=rot(lbrow[i]),
                right_brow=rot(rbrow[i]),
            ).compute_derived()
        )
    return t, samples


GROUND_TRUTH = [
    ("smile",      21.0, 2.0),
    ("head turn",  28.0, 1.0),
    ("brow raise", 35.0, 1.0),
    ("squint",     44.0, 0.8),
    ("yawn",       52.0, 1.5),
]


def overlaps(a, b) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def main() -> int:
    t, samples = build_recording()

    baseline_idx = int(BASELINE_END_S * FPS)
    baseline = fit_baseline(samples[:baseline_idx])
    if baseline is None:
        print("could not fit baseline")
        return 1

    print(f"Baseline window:   [0, {BASELINE_END_S:.1f}] s "
          f"({baseline.n_samples} frames @ {FPS} fps)")
    print(f"Baseline hash:     {baseline.hash()}")
    print(f"Baseline feature std ({len(FEATURE_NAMES)}):")
    for name, mu, sd in zip(FEATURE_NAMES, baseline.means, baseline.stds):
        print(f"    {name:24s}  mean={mu:+.6f}  std={sd:.6f}")

    report = detect_sigma_changes(
        samples, baseline,
        sigma_low=3.0, sigma_high=6.0,
        min_run_samples=3,
        cusum_k=0.5, cusum_h=5.0,
    )

    print(f"\nMonitoring window: [0, {DURATION_S:.1f}] s "
          f"({report.times.size} frames)")
    print(f"Max RMS |z|:           {report.overall_z.max():.2f}")
    print(f"Max T² σ-equivalent:   {report.t2_equivalent_sigma.max():.2f}")
    print(f"T² threshold 3σ / 6σ:  "
          f"{report.t2_threshold_low:.1f}  /  {report.t2_threshold_high:.1f}")
    print(f"CUSUM k / h:           {report.cusum_k} / {report.cusum_h}")
    print(f"Events: RMS "
          f"{sum(1 for e in report.events if e.detector == 'rms' and e.sigma_level == 3)}x3σ "
          f"{sum(1 for e in report.events if e.detector == 'rms' and e.sigma_level == 6)}x6σ  |  "
          f"T² "
          f"{sum(1 for e in report.events if e.detector == 't2' and e.sigma_level == 3)}x3σ "
          f"{sum(1 for e in report.events if e.detector == 't2' and e.sigma_level == 6)}x6σ  |  "
          f"CUSUM {len(report.cusum_events)}")
    print()
    print(f"{'det':>4} {'lvl':>4}  {'start':>8} {'end':>8} "
          f"{'dur':>7} {'peak σ':>8}  dominant feature")
    print("-" * 78)
    for e in report.events:
        print(f"{e.detector.upper():>4}  {e.sigma_level}σ   "
              f"{e.start_t:8.2f} {e.end_t:8.2f} "
              f"{e.duration_s:7.2f} {e.peak_z:8.2f}  {e.dominant_feature}")
    print()
    print(f"{'dir':>4}  {'start':>8} {'end':>8} {'dur':>7} {'peak S':>8}  feature")
    print("-" * 78)
    for e in report.cusum_events:
        print(f"{e.direction:>4}  {e.start_t:8.2f} {e.end_t:8.2f} "
              f"{e.duration_s:7.2f} {e.peak_cusum:8.2f}  {e.feature_name}")

    print("\nGround-truth alignment (any detector ⇒ hit):")
    rms_3 = [(e.start_t, e.end_t) for e in report.events if e.detector == "rms" and e.sigma_level == 3]
    rms_6 = [(e.start_t, e.end_t) for e in report.events if e.detector == "rms" and e.sigma_level == 6]
    t2_3 = [(e.start_t, e.end_t) for e in report.events if e.detector == "t2" and e.sigma_level == 3]
    t2_6 = [(e.start_t, e.end_t) for e in report.events if e.detector == "t2" and e.sigma_level == 6]
    cus = [(e.start_t, e.end_t) for e in report.cusum_events]
    for name, t0, sigma_s in GROUND_TRUTH:
        gt = (t0 - 2 * sigma_s, t0 + 2 * sigma_s)
        print(f"  {name:12s} truth=[{gt[0]:5.2f}, {gt[1]:5.2f}]s  "
              f"RMS3σ={any(overlaps(w, gt) for w in rms_3)}  "
              f"RMS6σ={any(overlaps(w, gt) for w in rms_6)}  "
              f"T²3σ={any(overlaps(w, gt) for w in t2_3)}  "
              f"T²6σ={any(overlaps(w, gt) for w in t2_6)}  "
              f"CUSUM={any(overlaps(w, gt) for w in cus)}")

    # ---- plot ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.2, 2.0]})

    # Panel A: RMS aggregate |z|
    axes[0].plot(report.times, report.overall_z, color="steelblue", lw=1.0)
    axes[0].axhline(3.0, color="orange", ls="--", lw=1.0, label="3σ")
    axes[0].axhline(6.0, color="red", ls="--", lw=1.0, label="6σ")
    axes[0].axvspan(0, BASELINE_END_S, alpha=0.10, color="gray", label="baseline window")
    for e in report.events:
        if e.detector != "rms":
            continue
        c = "red" if e.sigma_level == 6 else "orange"
        axes[0].axvspan(e.start_t, e.end_t, alpha=0.18, color=c)
    for name, t0, _ in GROUND_TRUTH:
        axes[0].axvline(t0, color="black", lw=0.8, ls=":")
        axes[0].text(t0, axes[0].get_ylim()[1] * 0.95, f" {name}",
                     fontsize=9, ha="left", va="top")
    axes[0].set_ylabel("RMS aggregate |z|")
    axes[0].set_title("Detector A — RMS aggregate (original)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(7.0, report.overall_z.max() * 1.1))

    # Panel B: T² (σ-equivalent)
    axes[1].plot(report.times, report.t2_equivalent_sigma, color="darkgreen", lw=1.0)
    axes[1].axhline(3.0, color="orange", ls="--", lw=1.0, label="3σ-equiv.")
    axes[1].axhline(6.0, color="red", ls="--", lw=1.0, label="6σ-equiv.")
    axes[1].axvspan(0, BASELINE_END_S, alpha=0.10, color="gray")
    for e in report.events:
        if e.detector != "t2":
            continue
        c = "red" if e.sigma_level == 6 else "orange"
        axes[1].axvspan(e.start_t, e.end_t, alpha=0.18, color=c)
    for _, t0, _ in GROUND_TRUTH:
        axes[1].axvline(t0, color="black", lw=0.8, ls=":")
    axes[1].set_ylabel("T² (σ-equivalent)")
    axes[1].set_title("Detector B — Hotelling's T²  (chi-square aggregate)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, max(7.0, report.t2_equivalent_sigma.max() * 1.1))

    # Panel C: per-feature CUSUM heatmap (max of S+, S-)
    cs = np.maximum(report.cusum_plus, report.cusum_minus)
    im = axes[2].imshow(
        cs.T,
        aspect="auto",
        extent=[0.0, DURATION_S, len(FEATURE_NAMES), 0],
        cmap="magma",
        vmin=0, vmax=max(report.cusum_h * 2, 10),
        interpolation="nearest",
    )
    axes[2].set_yticks(np.arange(len(FEATURE_NAMES)) + 0.5)
    axes[2].set_yticklabels(FEATURE_NAMES, fontsize=8)
    for e in report.cusum_events:
        axes[2].axvline(e.start_t, color="cyan", lw=0.5, alpha=0.6)
    for _, t0, _ in GROUND_TRUTH:
        axes[2].axvline(t0, color="white", lw=0.8, ls=":")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title(
        f"Detector C — per-feature CUSUM  max(S⁺, S⁻)   (alarm when ≥ h={report.cusum_h})"
    )
    cbar = plt.colorbar(im, ax=axes[2], pad=0.01)
    cbar.set_label("CUSUM")

    plt.tight_layout()
    out_png = REPO / "confirm_60s.png"
    plt.savefig(out_png, dpi=140)
    print(f"\nSaved plot: {out_png}")

    out_npz = REPO / "confirm_60s_data.npz"
    np.savez(
        out_npz,
        t=t,
        overall_z=report.overall_z,
        t2=report.t2,
        t2_equivalent_sigma=report.t2_equivalent_sigma,
        cusum_plus=report.cusum_plus,
        cusum_minus=report.cusum_minus,
        per_feature_z=report.per_feature_z,
        feature_names=np.array(FEATURE_NAMES),
        baseline_means=baseline.means,
        baseline_stds=baseline.stds,
    )
    print(f"Saved data: {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
