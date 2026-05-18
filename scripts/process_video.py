"""Run the trapezium signature pipeline on a real face video.

Reads a video file frame-by-frame with OpenCV, runs MediaPipe FaceMesh to
extract the six landmarks (4 trapezium corners + 2 eyebrow points), builds
``TrapeziumSample`` objects with timestamps from the frame index, fits a
robust baseline from the first ``--baseline-seconds`` (default 10 s), and
runs ``detect_sigma_changes`` over the full recording.

Outputs (next to the video, by default):
    <stem>_report.png   — 3-panel detector view (RMS / T² / CUSUM heatmap)
    <stem>_events.json  — per-event report + run config
    <stem>_overlay.mp4  — annotated video with trapezium + alarm indicator

Usage:
    python scripts/process_video.py videos/IMG_5034.MOV
    python scripts/process_video.py videos/IMG_5034.MOV --baseline-seconds 15
    python scripts/process_video.py videos/IMG_5034.MOV --frame-skip 2 --no-overlay
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from face_trapezium import (  # noqa: E402
    FEATURE_NAMES,
    FaceTrapeziumDetector,
    TrapeziumSample,
    detect_sigma_changes,
    fit_baseline,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("video", type=Path, help="Path to the input video file")
    p.add_argument("--baseline-seconds", type=float, default=10.0,
                   help="Length of the enrollment window in seconds (default 10)")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="Process every Nth frame (default 1 = every frame)")
    p.add_argument("--sigma-low", type=float, default=3.0)
    p.add_argument("--sigma-high", type=float, default=6.0)
    p.add_argument("--cusum-h", type=float, default=5.0)
    p.add_argument("--cusum-k", type=float, default=0.5)
    p.add_argument("--huber-cap", type=float, default=8.0)
    p.add_argument("--reliability-window", type=int, default=30)
    p.add_argument("--reliability-threshold", type=float, default=5.0)
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip writing the annotated overlay video")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory for output files (default: alongside the video)")
    return p.parse_args()


def extract_samples(
    video_path: Path,
    frame_skip: int,
    detector: FaceTrapeziumDetector,
) -> tuple[List[TrapeziumSample], List[Optional[int]], float, tuple[int, int]]:
    """Read the whole video, return one TrapeziumSample per processed frame
    where FaceMesh detected a face. Also returns the per-sample frame index
    (so we can re-render an overlay later) and the original fps + (w, h)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Opened {video_path.name}: {w}×{h} @ {fps:.2f} fps, "
          f"{total} frames ({total / fps:.1f} s). Processing every {frame_skip}th frame.")

    samples: List[TrapeziumSample] = []
    sample_frame_idx: List[int] = []
    n_processed = 0
    n_detected = 0
    last_log = time.time()
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue
        n_processed += 1
        t = frame_idx / fps
        sample = detector.detect(frame, t)
        if sample is not None:
            samples.append(sample)
            sample_frame_idx.append(frame_idx)
            n_detected += 1

        now = time.time()
        if now - last_log > 2.0:
            pct = 100.0 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total} frames ({pct:5.1f}%)  "
                  f"detected {n_detected}/{n_processed}")
            last_log = now

    cap.release()
    print(f"Done: detected face in {n_detected}/{n_processed} processed frames "
          f"({100.0 * n_detected / max(n_processed, 1):.1f}%).")
    return samples, sample_frame_idx, fps, (w, h)


def render_3panel_plot(
    out_path: Path,
    report,
    baseline_seconds: float,
    ground_truth: Optional[list] = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.2, 2.0]})

    axes[0].plot(report.times, report.overall_z, color="steelblue", lw=1.0)
    axes[0].axhline(report.sigma_low, color="orange", ls="--", lw=1.0,
                    label=f"{report.sigma_low}σ")
    axes[0].axhline(report.sigma_high, color="red", ls="--", lw=1.0,
                    label=f"{report.sigma_high}σ")
    axes[0].axvspan(0, baseline_seconds, alpha=0.10, color="gray",
                    label="baseline window")
    for e in report.events:
        if e.detector != "rms":
            continue
        c = "red" if e.sigma_level == 6 else "orange"
        axes[0].axvspan(e.start_t, e.end_t, alpha=0.18, color=c)
    axes[0].set_ylabel("RMS aggregate |z|")
    axes[0].set_title("Detector A — RMS aggregate (Huber-clipped, reliability-weighted)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(7.0, float(report.overall_z.max()) * 1.1 if report.overall_z.size else 7.0))

    axes[1].plot(report.times, report.t2_equivalent_sigma, color="darkgreen", lw=1.0)
    axes[1].axhline(report.sigma_low, color="orange", ls="--", lw=1.0)
    axes[1].axhline(report.sigma_high, color="red", ls="--", lw=1.0)
    axes[1].axvspan(0, baseline_seconds, alpha=0.10, color="gray")
    for e in report.events:
        if e.detector != "t2":
            continue
        c = "red" if e.sigma_level == 6 else "orange"
        axes[1].axvspan(e.start_t, e.end_t, alpha=0.18, color=c)
    axes[1].set_ylabel("T² (σ-equivalent)")
    axes[1].set_title("Detector B — Hotelling's T² (chi-square aggregate)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, max(7.0, float(report.t2_equivalent_sigma.max()) * 1.1
                            if report.t2.size else 7.0))

    cs = np.maximum(report.cusum_plus, report.cusum_minus)
    if cs.size:
        im = axes[2].imshow(
            cs.T,
            aspect="auto",
            extent=[float(report.times[0]), float(report.times[-1]),
                    len(FEATURE_NAMES), 0],
            cmap="magma",
            vmin=0, vmax=max(report.cusum_h * 2, 10),
            interpolation="nearest",
        )
        cbar = plt.colorbar(im, ax=axes[2], pad=0.01)
        cbar.set_label("CUSUM")
    axes[2].set_yticks(np.arange(len(FEATURE_NAMES)) + 0.5)
    axes[2].set_yticklabels(FEATURE_NAMES, fontsize=8)
    for e in report.cusum_events:
        axes[2].axvline(e.start_t, color="cyan", lw=0.5, alpha=0.5)
    axes[2].set_xlabel("time (s)")
    axes[2].set_title(
        f"Detector C — per-feature CUSUM  max(S⁺, S⁻)   (alarm when ≥ h={report.cusum_h})"
    )

    if ground_truth:
        for t0, name in ground_truth:
            for ax in axes:
                ax.axvline(t0, color="black", lw=0.8, ls=":", alpha=0.6)
            axes[0].text(t0, axes[0].get_ylim()[1] * 0.95, f" {name}",
                         fontsize=9, ha="left", va="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def write_overlay_video(
    src_video: Path,
    out_path: Path,
    samples: List[TrapeziumSample],
    sample_frame_idx: List[int],
    report,
    fps: float,
    size: tuple[int, int],
) -> None:
    """Re-read the source video and write an annotated copy with the trapezium
    + alarm indicator drawn on each processed frame."""
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not re-open {src_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")

    sample_by_frame = dict(zip(sample_frame_idx, samples))
    times = report.times
    overall_z = report.overall_z
    t2_sigma = report.t2_equivalent_sigma

    # Build a lookup: which sample index corresponds to which frame
    frame_to_sample_idx = {f: i for i, f in enumerate(sample_frame_idx)}

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        sample = sample_by_frame.get(frame_idx)
        if sample is not None:
            pts = np.array([
                sample.left_eye[:2],
                sample.right_eye[:2],
                sample.right_mouth[:2],
                sample.left_mouth[:2],
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 200, 255), 2)
            cv2.line(frame, tuple(pts[3]), tuple(pts[2]), (0, 200, 255), 2)
            for p in pts:
                cv2.circle(frame, tuple(p), 4, (0, 0, 255), -1)
            for brow, eye in ((sample.left_brow, sample.left_eye),
                              (sample.right_brow, sample.right_eye)):
                bx, by = int(brow[0]), int(brow[1])
                ex, ey = int(eye[0]), int(eye[1])
                cv2.line(frame, (ex, ey), (ex, by), (180, 255, 0), 1)
                cv2.circle(frame, (bx, by), 4, (180, 255, 0), -1)

            si = frame_to_sample_idx.get(frame_idx)
            if si is not None and si < len(overall_z):
                rms = float(overall_z[si])
                t2s = float(t2_sigma[si])
                color = (0, 0, 255) if max(rms, t2s) >= report.sigma_high else (
                    (0, 165, 255) if max(rms, t2s) >= report.sigma_low else (200, 200, 200)
                )
                cv2.putText(
                    frame,
                    f"t={sample.t:6.2f}s  RMS={rms:5.2f}σ  T²={t2s:5.2f}σ",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )

        writer.write(frame)

    cap.release()
    writer.release()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}")
        return 1

    out_dir = args.out_dir or args.video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    report_png = out_dir / f"{stem}_report.png"
    events_json = out_dir / f"{stem}_events.json"
    overlay_mp4 = out_dir / f"{stem}_overlay.mp4"

    with FaceTrapeziumDetector() as detector:
        samples, sample_frame_idx, fps, size = extract_samples(
            args.video, args.frame_skip, detector,
        )

    if len(samples) < 30:
        print(f"Too few face detections ({len(samples)}) to fit a baseline.")
        return 2

    # Fit baseline from the first `baseline_seconds` of detected samples.
    baseline_cutoff_t = samples[0].t + args.baseline_seconds
    baseline_samples = [s for s in samples if s.t < baseline_cutoff_t]
    if len(baseline_samples) < 30:
        print(f"Baseline window contains only {len(baseline_samples)} samples "
              f"(< 30). Try a longer --baseline-seconds.")
        return 3

    baseline = fit_baseline(baseline_samples, robust=True)
    if baseline is None:
        print("Could not fit baseline.")
        return 4

    print(f"\nBaseline: {baseline.n_samples} frames, "
          f"{baseline.duration_s:.2f} s, hash={baseline.hash()}")

    report = detect_sigma_changes(
        samples, baseline,
        sigma_low=args.sigma_low, sigma_high=args.sigma_high,
        min_run_samples=3,
        cusum_k=args.cusum_k, cusum_h=args.cusum_h,
        huber_cap=args.huber_cap,
        reliability_window=args.reliability_window,
        reliability_threshold=args.reliability_threshold,
    )

    n_rms3 = sum(1 for e in report.events if e.detector == "rms" and e.sigma_level == 3)
    n_rms6 = sum(1 for e in report.events if e.detector == "rms" and e.sigma_level == 6)
    n_t23 = sum(1 for e in report.events if e.detector == "t2" and e.sigma_level == 3)
    n_t26 = sum(1 for e in report.events if e.detector == "t2" and e.sigma_level == 6)

    print(f"\nMonitoring: {report.times.size} frames over "
          f"[{report.times[0]:.2f}, {report.times[-1]:.2f}] s")
    print(f"Max RMS |z|:           {report.overall_z.max():.2f}")
    print(f"Max T² σ-equivalent:   {report.t2_equivalent_sigma.max():.2f}")
    print(f"T² threshold 3σ / 6σ:  {report.t2_threshold_low:.1f}  /  "
          f"{report.t2_threshold_high:.1f}")
    print(f"Events: RMS {n_rms3}x3σ {n_rms6}x6σ  |  "
          f"T² {n_t23}x3σ {n_t26}x6σ  |  CUSUM {len(report.cusum_events)}")
    print()
    print(f"{'det':>4} {'lvl':>4}  {'start':>8} {'end':>8} {'dur':>7} {'peak σ':>8}  feature")
    print("-" * 78)
    for e in report.events:
        print(f"{e.detector.upper():>4}  {e.sigma_level}σ   "
              f"{e.start_t:8.2f} {e.end_t:8.2f} {e.duration_s:7.2f} "
              f"{e.peak_z:8.2f}  {e.dominant_feature}")

    payload = {
        "video": str(args.video),
        "fps": fps,
        "frame_size": list(size),
        "baseline": {
            "hash": baseline.hash(),
            "n_samples": baseline.n_samples,
            "duration_s": baseline.duration_s,
            "feature_names": list(baseline.feature_names),
            "means": baseline.means.tolist(),
            "stds": baseline.stds.tolist(),
        },
        "config": {
            "frame_skip": args.frame_skip,
            "baseline_seconds": args.baseline_seconds,
            "sigma_low": args.sigma_low,
            "sigma_high": args.sigma_high,
            "cusum_k": args.cusum_k,
            "cusum_h": args.cusum_h,
            "huber_cap": args.huber_cap,
            "reliability_window": args.reliability_window,
            "reliability_threshold": args.reliability_threshold,
        },
        "summary": report.summary(),
    }
    events_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved events: {events_json}")

    render_3panel_plot(report_png, report, args.baseline_seconds)
    print(f"Saved plot:   {report_png}")

    if not args.no_overlay:
        write_overlay_video(args.video, overlay_mp4, samples, sample_frame_idx,
                            report, fps, size)
        print(f"Saved video:  {overlay_mp4}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
