"""End-to-end Syntonia pipeline on a single video.

Runs four independent channels:
  1. FaceLandmarker → TrapeziumSamples → per-feature z-scores → AU-clustered M(t)
  2. HandLandmarker → HandsSamples → finger motion + hand state + region tag
                                  → kinetic-density F(t)
  3. Audio (ffmpeg + librosa) → syllable-rate proxy → V(t)
  4. Syntonia(t) = α·V + β·F + γ·M with 5-second rolling threshold scan.

Output artifacts (saved next to the input video, by default):
  <stem>_syntonia_report.png    7-panel detector view (RMS, T², M, F, V, Syntonia, hand state)
  <stem>_syntonia_events.json   per-channel + Syntonia windows + full run config
  <stem>_syntonia_overlay.mp4   annotated video (face trapezium + hands + per-frame Syntonia readout)

Usage:
  python scripts/process_video_syntonia.py videos/IMG_5034.MOV
  python scripts/process_video_syntonia.py videos/IMG_5034.MOV --no-voice --no-overlay
  python scripts/process_video_syntonia.py videos/IMG_5034.MOV --alpha 0.25 --beta 0.4 --gamma 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path
from typing import List, Optional, Tuple

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
from au_clusters import compute_au_clusters, DEFAULT_AU_WEIGHTS  # noqa: E402
from hands_pipeline import (  # noqa: E402
    HandsDetector,
    HandsSample,
    HandsTracker,
    compute_fidget_index,
    synthesize_neck_and_chin,
    REGION_WEIGHTS,
)
from voice_analytics import compute_voice  # noqa: E402
from syntonia_model import compute_syntonia, SYNTONIA_DISCLAIMER  # noqa: E402


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video", type=Path)
    p.add_argument("--baseline-seconds", type=float, default=15.0)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1.0 / 3.0, help="Voice weight")
    p.add_argument("--beta", type=float, default=1.0 / 3.0, help="Fidget weight")
    p.add_argument("--gamma", type=float, default=1.0 / 3.0, help="Micro-expr weight")
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--window-s", type=float, default=5.0,
                   help="Syntonia rolling window (s)")
    p.add_argument("--fidget-window-s", type=float, default=15.0,
                   help="F(t) kinetic-density window (s)")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--no-hands", action="store_true")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def extract_face_and_hands(
    video_path: Path,
    frame_skip: int,
    use_hands: bool,
) -> Tuple[List[TrapeziumSample], List[Optional[HandsSample]], List[int], float, Tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Opened {video_path.name}: {w}×{h} @ {fps:.2f} fps, "
          f"{total} frames ({total / fps:.1f} s). Frame-skip={frame_skip}.")

    face_samples: List[TrapeziumSample] = []
    hands_samples: List[Optional[HandsSample]] = []
    sample_frame_idx: List[int] = []

    face_det = FaceTrapeziumDetector()
    hands_det: Optional[HandsDetector] = HandsDetector() if use_hands else None
    tracker = HandsTracker()

    n_detected_face = 0
    n_detected_hands = 0
    frame_idx = -1
    last_log = _time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue
            t = frame_idx / fps
            fs = face_det.detect(frame, t)
            if fs is None:
                continue
            face_samples.append(fs)
            sample_frame_idx.append(frame_idx)
            n_detected_face += 1

            if hands_det is None:
                hands_samples.append(None)
            else:
                hs = hands_det.detect(frame, t)
                # Build a 6-vertex face anchor + chin/neck for region tagging.
                chin, neck = synthesize_neck_and_chin(
                    fs.left_eye, fs.right_eye, fs.left_mouth, fs.right_mouth
                )
                face_anchor = np.stack([
                    fs.left_eye, fs.right_eye,
                    fs.left_mouth, fs.right_mouth,
                    fs.left_brow, fs.right_brow,
                    chin, neck,
                ])
                eye_line_px = float(np.linalg.norm(fs.right_eye - fs.left_eye))
                hs = tracker.process(hs, eye_line_px, face_anchor)
                if hs.n_hands > 0:
                    n_detected_hands += 1
                hands_samples.append(hs)

            now = _time.time()
            if now - last_log > 3.0:
                pct = 100.0 * frame_idx / max(total, 1)
                print(f"  {frame_idx}/{total} ({pct:5.1f}%) "
                      f"face={n_detected_face} hands={n_detected_hands}")
                last_log = now
    finally:
        face_det.close()
        if hands_det is not None:
            hands_det.close()
        cap.release()

    print(f"Done. face detected: {n_detected_face}/{len(sample_frame_idx)} "
          f"({100 * n_detected_face / max(len(sample_frame_idx), 1):.1f}%); "
          f"hands detected: {n_detected_hands}/{len(sample_frame_idx)} "
          f"({100 * n_detected_hands / max(len(sample_frame_idx), 1):.1f}%).")
    return face_samples, hands_samples, sample_frame_idx, fps, (w, h)


def render_overlay_video(
    src: Path, out_path: Path,
    face_samples: List[TrapeziumSample],
    hands_samples: List[Optional[HandsSample]],
    sample_frame_idx: List[int],
    syntonia_report,
    fps: float, size: Tuple[int, int],
) -> None:
    cap = cv2.VideoCapture(str(src))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")

    face_by_frame = dict(zip(sample_frame_idx, face_samples))
    hands_by_frame = dict(zip(sample_frame_idx, hands_samples))
    frame_to_sample = {f: i for i, f in enumerate(sample_frame_idx)}

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        fs = face_by_frame.get(frame_idx)
        if fs is not None:
            pts = np.array([
                fs.left_eye[:2], fs.right_eye[:2],
                fs.right_mouth[:2], fs.left_mouth[:2],
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 200, 255), 2)
            cv2.line(frame, tuple(pts[3]), tuple(pts[2]), (0, 200, 255), 2)
            for p in pts:
                cv2.circle(frame, tuple(p), 5, (0, 0, 255), -1)
            for brow in (fs.left_brow, fs.right_brow):
                cv2.circle(frame, (int(brow[0]), int(brow[1])), 5, (180, 255, 0), -1)

        hs = hands_by_frame.get(frame_idx)
        if hs is not None:
            for hand_pts in (hs.left, hs.right):
                if hand_pts is None:
                    continue
                pts = np.asarray(hand_pts[:, :2], dtype=np.int32)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, tuple(p), 4, (0, 255, 255), -1)

        si = frame_to_sample.get(frame_idx)
        if si is not None and si < syntonia_report.times.size:
            dfi_v = float(syntonia_report.syntonia[si])
            v_v = float(syntonia_report.v[si])
            f_v = float(syntonia_report.f[si])
            m_v = float(syntonia_report.m[si])
            color = (0, 0, 255) if dfi_v >= syntonia_report.threshold else (
                (0, 165, 255) if dfi_v >= syntonia_report.threshold * 0.66 else (200, 200, 200))
            cv2.putText(
                frame,
                f"t={fs.t:6.2f}s  Syntonia={dfi_v:5.2f}  V={v_v:4.2f}  F={f_v:4.2f}  M={m_v:4.2f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
            )
            if hs is not None and hs.hand_state != "unknown":
                cv2.putText(
                    frame, f"hand_state={hs.hand_state}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2,
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
    out_png = out_dir / f"{stem}_syntonia_report.png"
    out_json = out_dir / f"{stem}_syntonia_events.json"
    out_mp4 = out_dir / f"{stem}_syntonia_overlay.mp4"

    # ---- 1. Face + Hands extraction ----------------------------------------
    face_samples, hands_samples, sample_frame_idx, fps, size = extract_face_and_hands(
        args.video, args.frame_skip, use_hands=not args.no_hands,
    )
    if len(face_samples) < 30:
        print(f"Too few face detections ({len(face_samples)}).")
        return 2

    times = np.array([s.t for s in face_samples])

    # ---- 2. Face baseline + change report ----------------------------------
    base_cut = face_samples[0].t + args.baseline_seconds
    baseline_face_samples = [s for s in face_samples if s.t < base_cut]
    baseline = fit_baseline(baseline_face_samples, robust=True)
    if baseline is None:
        print("Could not fit face baseline.")
        return 3
    print(f"\nFace baseline: {baseline.n_samples} frames, "
          f"{baseline.duration_s:.2f}s, hash={baseline.hash()}")
    face_report = detect_sigma_changes(
        face_samples, baseline,
        sigma_low=3.0, sigma_high=6.0, min_run_samples=3,
    )

    # ---- 3. AU-weighted M(t) -----------------------------------------------
    au_result = compute_au_clusters(times, face_report.per_feature_z)
    print(f"M(t): max={au_result.m.max():.2f}σ  mean={au_result.m.mean():.2f}σ")

    # ---- 4. Hands → F(t) ----------------------------------------------------
    if args.no_hands or not any(hs and hs.n_hands > 0 for hs in hands_samples):
        f_z = np.zeros_like(times)
        hand_state_codes = np.full_like(times, -1, dtype=float)
        print("F(t): hands disabled or undetected → F = 0.")
    else:
        regions_per_frame = []
        kinetic_per_frame = np.zeros_like(times)
        hand_state_codes = np.zeros_like(times)
        for i, hs in enumerate(hands_samples):
            if hs is None:
                regions_per_frame.append("none")
                hand_state_codes[i] = -1
                continue
            regions_per_frame.append(hs.effective_region or "none")
            # Use combined finger motion (max of two hands) as K_e proxy.
            kinetic_per_frame[i] = max(hs.finger_motion_L, hs.finger_motion_R)
            hand_state_codes[i] = float(hs.hand_state_code)
        f_raw, f_z = compute_fidget_index(
            times, regions_per_frame, kinetic_per_frame,
            window_s=args.fidget_window_s,
        )
        print(f"F(t): max={f_z.max():.2f}σ  mean={f_z.mean():.2f}σ "
              f"(window={args.fidget_window_s}s)")

    # ---- 5. Voice → V(t) ----------------------------------------------------
    voice_result = None
    if not args.no_voice:
        try:
            voice_result = compute_voice(
                args.video,
                bin_s=1.0,
                baseline_seconds=args.baseline_seconds,
                estimate_response_latency=False,  # single-speaker default
            )
            print(f"V(t): max={voice_result.v.max():.2f}σ  "
                  f"mean={voice_result.v.mean():.2f}σ  "
                  f"baseline SPS μ={voice_result.sps_baseline_mean:.2f}±"
                  f"{voice_result.sps_baseline_std:.2f}")
        except Exception as e:
            print(f"V(t): voice analysis failed: {e}")
            voice_result = None

    # ---- 6. Syntonia(t) assembly -------------------------------------------------
    if voice_result is not None:
        v_times = voice_result.times
        v_values = voice_result.v
    else:
        v_times, v_values = None, None
    syntonia_report = compute_syntonia(
        times,
        v_times, v_values,
        times, f_z,
        times, au_result.m,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        threshold=args.threshold, window_s=args.window_s,
    )
    print(f"\nSyntonia(t): max={syntonia_report.syntonia.max():.2f}  "
          f"mean={syntonia_report.syntonia.mean():.2f}  "
          f"threshold={args.threshold}  "
          f"windows={len(syntonia_report.windows)}")
    for w in syntonia_report.windows:
        print(f"  [{w.start_t:6.2f}, {w.end_t:6.2f}]s "
              f"({w.duration_s:5.2f}s)  peak={w.peak_syntonia:.2f}@{w.peak_t:.2f}s  "
              f"dominant={w.dominant_component}")

    # ---- 7. Plot ------------------------------------------------------------
    render_report_plot(
        out_png,
        face_report, au_result, hand_state_codes,
        voice_result, syntonia_report, args.baseline_seconds,
    )
    print(f"\nSaved plot:   {out_png}")

    # ---- 8. JSON ------------------------------------------------------------
    payload = {
        "video": str(args.video),
        "fps": fps,
        "frame_size": list(size),
        "config": {
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "threshold": args.threshold,
            "window_s": args.window_s,
            "fidget_window_s": args.fidget_window_s,
            "baseline_seconds": args.baseline_seconds,
            "frame_skip": args.frame_skip,
            "au_weights": dict(DEFAULT_AU_WEIGHTS),
            "region_weights": dict(REGION_WEIGHTS),
        },
        "face_baseline": {
            "hash": baseline.hash(),
            "n_samples": baseline.n_samples,
            "duration_s": baseline.duration_s,
            "feature_names": list(baseline.feature_names),
            "means": baseline.means.tolist(),
            "stds": baseline.stds.tolist(),
        },
        "voice": (
            None if voice_result is None else {
                "sps_baseline_mean": voice_result.sps_baseline_mean,
                "sps_baseline_std": voice_result.sps_baseline_std,
                "max_v": float(voice_result.v.max()),
            }
        ),
        "dfi_summary": syntonia_report.summary(),
        "face_events": [e.to_dict() for e in face_report.events],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved events: {out_json}")

    # ---- 9. Overlay video ---------------------------------------------------
    if not args.no_overlay:
        render_overlay_video(args.video, out_mp4, face_samples, hands_samples,
                             sample_frame_idx, syntonia_report, fps, size)
        print(f"Saved video:  {out_mp4}")

    return 0


def render_report_plot(out_path, face_report, au_result, hand_state_codes,
                       voice_result, syntonia_report, baseline_seconds):
    n_rows = 6
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1.0] * n_rows})

    # Panel 0: face RMS aggregate |z|
    axes[0].plot(face_report.times, face_report.overall_z, color="steelblue", lw=1.0)
    axes[0].axhline(3.0, color="orange", ls="--", lw=0.8)
    axes[0].axhline(6.0, color="red", ls="--", lw=0.8)
    axes[0].set_ylabel("Face RMS |z|")
    axes[0].axvspan(0, baseline_seconds, alpha=0.08, color="gray")
    axes[0].grid(alpha=0.3)

    # Panel 1: AU-weighted M(t)
    axes[1].plot(au_result.times, au_result.m, color="darkgreen", lw=1.0)
    axes[1].axhline(3.0, color="orange", ls="--", lw=0.8)
    axes[1].set_ylabel("M(t) σ-eq")
    axes[1].axvspan(0, baseline_seconds, alpha=0.08, color="gray")
    axes[1].grid(alpha=0.3)

    # Panel 2: F(t) fidget
    axes[2].plot(syntonia_report.times, syntonia_report.f, color="purple", lw=1.0)
    axes[2].axhline(3.0, color="orange", ls="--", lw=0.8)
    axes[2].set_ylabel("F(t) σ-eq")
    axes[2].axvspan(0, baseline_seconds, alpha=0.08, color="gray")
    axes[2].grid(alpha=0.3)

    # Panel 3: V(t) voice
    axes[3].plot(syntonia_report.times, syntonia_report.v, color="brown", lw=1.0)
    if voice_result is not None:
        axes[3].axhline(3.0, color="orange", ls="--", lw=0.8)
    axes[3].set_ylabel("V(t) σ-eq")
    axes[3].axvspan(0, baseline_seconds, alpha=0.08, color="gray")
    axes[3].grid(alpha=0.3)

    # Panel 4: Syntonia(t) with threshold + flagged windows
    axes[4].plot(syntonia_report.times, syntonia_report.syntonia, color="black", lw=1.2)
    axes[4].axhline(syntonia_report.threshold, color="red", ls="--", lw=1.0,
                    label=f"threshold={syntonia_report.threshold}")
    for w in syntonia_report.windows:
        axes[4].axvspan(w.start_t, w.end_t, alpha=0.18, color="red")
    axes[4].set_ylabel("Syntonia(t)")
    axes[4].axvspan(0, baseline_seconds, alpha=0.08, color="gray")
    axes[4].grid(alpha=0.3)
    axes[4].legend(loc="upper right", fontsize=8)

    # Panel 5: hand state code over time
    axes[5].plot(syntonia_report.times, hand_state_codes, color="teal",
                 lw=0.8, drawstyle="steps-post")
    axes[5].set_yticks([-1, 0, 1, 2, 3, 4])
    axes[5].set_yticklabels(["hidden", "A: together,still",
                             "B: together,moving", "C: apart,still",
                             "D: apart,moving", "one_hand"], fontsize=7)
    axes[5].set_ylabel("Hand state")
    axes[5].set_xlabel("time (s)")
    axes[5].axvspan(0, baseline_seconds, alpha=0.08, color="gray")
    axes[5].grid(alpha=0.3)

    fig.suptitle(
        f"Syntonia pipeline   α={syntonia_report.alpha:.2f}  β={syntonia_report.beta:.2f}  "
        f"γ={syntonia_report.gamma:.2f}  threshold={syntonia_report.threshold}  "
        f"window={syntonia_report.window_s}s",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
