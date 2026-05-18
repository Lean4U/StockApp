"""60-second end-to-end confirmation of the trapezium signature + 3σ / 6σ change detector.

The sandbox has no webcam. Instead, this script generates a physically-grounded
60-second / 30-fps recording of the four trapezium vertices with:

  * static neutral geometry placed in pixel coordinates that match a typical
    640×480 webcam frame at ~50 cm from the camera,
  * per-frame tracking jitter ~ N(0, 0.6 px) (the empirical FaceMesh noise floor),
  * slow head translation (low-frequency drift + a small respiratory rhythm),
  * three labelled expression events with Gaussian envelopes:
        smile        @ t = 21 s  (σ = 2.0 s, ±8 px outward, 4 px up)
        brow raise   @ t = 35 s  (σ = 1.0 s, eye corners raised 5 px)
        yawn         @ t = 52 s  (σ = 1.5 s, ±15 px outward, 12 px down)

The first 10 s are used as the enrollment baseline. The remaining 50 s are
overlaid against that baseline by `detect_sigma_changes`, and the report is
saved as both a text summary and a 2-panel matplotlib figure.

Run:  python scripts/confirm_60s.py
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

# Neutral resting vertex positions (pixels). Values reflect a face roughly
# centered in a 640×480 frame with an inter-canthal width of ~120 px.
LE0 = np.array([260.0, 220.0, 0.0])
RE0 = np.array([380.0, 220.0, 0.0])
LM0 = np.array([282.0, 330.0, 0.0])
RM0 = np.array([358.0, 330.0, 0.0])

TRACKING_JITTER_PX = 0.6  # per-axis std of FaceMesh tracking noise

BASELINE_END_S = 10.0


def head_motion(t: np.ndarray) -> np.ndarray:
    """Slow head translation + small respiratory bob (no rotation)."""
    dx = 8.0 * np.sin(2 * np.pi * 0.07 * t) + 2.0 * np.sin(2 * np.pi * 0.5 * t)
    dy = 3.0 * np.cos(2 * np.pi * 0.05 * t) + 1.5 * np.sin(2 * np.pi * 0.25 * t)
    dz = 2.0 * np.sin(2 * np.pi * 0.12 * t)
    return np.column_stack([dx, dy, dz])


def gaussian_envelope(t: np.ndarray, t0: float, sigma_s: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - t0) / sigma_s) ** 2)


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

    env_smile = gaussian_envelope(t, 21.0, 2.0)
    env_brow = gaussian_envelope(t, 35.0, 1.0)
    env_yawn = gaussian_envelope(t, 52.0, 1.5)

    # Smile: mouth corners pull apart (±8 px in x) and up (-4 px in y).
    lm[:, 0] += -8.0 * env_smile
    lm[:, 1] += -4.0 * env_smile
    rm[:, 0] += +8.0 * env_smile
    rm[:, 1] += -4.0 * env_smile

    # Brow raise: outer eye corners go up (-5 px) and slightly outward (±2 px).
    le[:, 0] += -2.0 * env_brow
    le[:, 1] += -5.0 * env_brow
    re[:, 0] += +2.0 * env_brow
    re[:, 1] += -5.0 * env_brow

    # Yawn: mouth corners spread far (±15 px) and drop (12 px).
    lm[:, 0] += -15.0 * env_yawn
    lm[:, 1] += +12.0 * env_yawn
    rm[:, 0] += +15.0 * env_yawn
    rm[:, 1] += +12.0 * env_yawn

    samples = []
    for i in range(N):
        samples.append(
            TrapeziumSample(
                t=float(t[i]),
                left_eye=le[i],
                right_eye=re[i],
                left_mouth=lm[i],
                right_mouth=rm[i],
            ).compute_derived()
        )
    return t, samples


GROUND_TRUTH = [
    ("smile", 21.0, 2.0),
    ("brow raise", 35.0, 1.0),
    ("yawn", 52.0, 1.5),
]


def overlaps(a: tuple, b: tuple) -> bool:
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
    print(f"Baseline feature std (15):")
    for name, mu, sd in zip(FEATURE_NAMES, baseline.means, baseline.stds):
        print(f"    {name:24s}  mean={mu:+.6f}  std={sd:.6f}")

    report = detect_sigma_changes(
        samples, baseline, sigma_low=3.0, sigma_high=6.0, min_run_samples=3,
    )

    print(f"\nMonitoring window: [0, {DURATION_S:.1f}] s "
          f"({report.times.size} frames)")
    print(f"Max aggregate |z|: {report.overall_z.max():.2f}")
    print(f"Events detected:   "
          f"{sum(1 for e in report.events if e.sigma_level == 3)}x 3σ, "
          f"{sum(1 for e in report.events if e.sigma_level == 6)}x 6σ")
    print()
    print(f"{'lvl':>4} {'start (s)':>10} {'end (s)':>10} "
          f"{'dur (s)':>9} {'peak |z|':>10}  dominant feature")
    print("-" * 78)
    for e in report.events:
        print(f"{e.sigma_level}σ   {e.start_t:10.2f} {e.end_t:10.2f} "
              f"{e.duration_s:9.2f} {e.peak_z:10.2f}  {e.dominant_feature}")

    # Ground-truth alignment: an event "matches" a labelled expression if the
    # detection window overlaps the labelled ±2σ_s span of its envelope.
    print("\nGround-truth alignment (3σ windows vs. labelled events):")
    sig3 = [(e.start_t, e.end_t) for e in report.events if e.sigma_level == 3]
    sig6 = [(e.start_t, e.end_t) for e in report.events if e.sigma_level == 6]
    for name, t0, sigma_s in GROUND_TRUTH:
        gt_span = (t0 - 2 * sigma_s, t0 + 2 * sigma_s)
        hit3 = any(overlaps(d, gt_span) for d in sig3)
        hit6 = any(overlaps(d, gt_span) for d in sig6)
        print(f"  {name:12s}  truth=[{gt_span[0]:5.2f}, {gt_span[1]:5.2f}]s   "
              f"3σ-detected={hit3}   6σ-detected={hit6}")

    # ---- plot ------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.8]})

    axes[0].plot(report.times, report.overall_z, color="steelblue", lw=1.0,
                 label="aggregate |z|")
    axes[0].axhline(3.0, color="orange", ls="--", lw=1.0, label="3σ")
    axes[0].axhline(6.0, color="red", ls="--", lw=1.0, label="6σ")
    axes[0].axvspan(0, BASELINE_END_S, alpha=0.10, color="gray",
                    label="baseline window")
    for e in report.events:
        c = "red" if e.sigma_level == 6 else "orange"
        axes[0].axvspan(e.start_t, e.end_t, alpha=0.18, color=c)
    for name, t0, sigma_s in GROUND_TRUTH:
        axes[0].axvline(t0, color="black", lw=0.8, ls=":")
        axes[0].text(t0, axes[0].get_ylim()[1] * 0.92, f" {name}",
                     fontsize=9, ha="left", va="top")
    axes[0].set_ylabel("aggregate |z|")
    axes[0].set_title("Trapezium signature: 60-second recording overlaid on baseline")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(7.0, report.overall_z.max() * 1.1))

    im = axes[1].imshow(
        report.per_feature_z.T,
        aspect="auto",
        extent=[0.0, DURATION_S, len(FEATURE_NAMES), 0],
        cmap="RdBu_r",
        vmin=-8, vmax=8,
        interpolation="nearest",
    )
    axes[1].set_yticks(np.arange(len(FEATURE_NAMES)) + 0.5)
    axes[1].set_yticklabels(FEATURE_NAMES, fontsize=8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("per-feature z-scores  (red = above baseline mean)")
    cbar = plt.colorbar(im, ax=axes[1], pad=0.01)
    cbar.set_label("z-score")

    plt.tight_layout()
    out_png = REPO / "confirm_60s.png"
    plt.savefig(out_png, dpi=140)
    print(f"\nSaved plot: {out_png}")

    out_npz = REPO / "confirm_60s_data.npz"
    np.savez(
        out_npz,
        t=t,
        overall_z=report.overall_z,
        per_feature_z=report.per_feature_z,
        feature_names=np.array(FEATURE_NAMES),
        baseline_means=baseline.means,
        baseline_stds=baseline.stds,
    )
    print(f"Saved data: {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
