"""Pre-compute the full Syntonia pipeline for a video and cache to .npz.

The real-time dashboard reads this cache so playback is instant.
Re-run with --force to regenerate (e.g. after tuning weights).

Outputs (next to the video, by default):
  <stem>_syntonia_cache.npz   — all per-frame timelines + Syntonia windows + config

Usage:
  python scripts/precompute_syntonia.py videos/test-video-1.MOV
  python scripts/precompute_syntonia.py videos/test-video-1.MOV --force --alpha 0.25
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
from voice_transcript import is_available as transcript_available, transcribe_audio  # noqa: E402
from syntonia_model import compute_syntonia  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--baseline-seconds", type=float, default=15.0)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=1.0 / 3.0)
    ap.add_argument("--beta", type=float, default=1.0 / 3.0)
    ap.add_argument("--gamma", type=float, default=1.0 / 3.0)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--fidget-window-s", type=float, default=15.0)
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-transcript", action="store_true",
                    help="Skip ASR even if the Whisper model is present.")
    ap.add_argument("--ephemeral", action="store_true",
                    help="Run the pipeline in-memory; print summary but do not "
                         "write the .npz cache or any artifact to disk.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}")
        return 1
    cache = args.video.parent / f"{args.video.stem}_syntonia_cache.npz"
    if args.ephemeral:
        print("--ephemeral: results will be computed in memory only; no cache written.")
    elif cache.exists() and not args.force:
        print(f"Cache exists: {cache}. Use --force to regenerate.")
        return 0

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Could not open {args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Opened {args.video.name}: {w}×{h} @ {fps:.2f} fps, "
          f"{total} frames ({total / fps:.1f} s). frame-skip={args.frame_skip}.")

    face_samples: List[TrapeziumSample] = []
    hands_samples: List[Optional[HandsSample]] = []
    sample_frame_idx: List[int] = []

    fd = FaceTrapeziumDetector()
    hd: Optional[HandsDetector] = None if args.no_hands else HandsDetector()
    tr = HandsTracker()

    last_log = _time.time()
    frame_idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % args.frame_skip != 0:
                continue
            t = frame_idx / fps
            fs = fd.detect(frame, t)
            if fs is None:
                continue
            face_samples.append(fs)
            sample_frame_idx.append(frame_idx)
            if hd is None:
                hands_samples.append(None)
            else:
                hs = hd.detect(frame, t)
                chin, neck = synthesize_neck_and_chin(
                    fs.left_eye, fs.right_eye, fs.left_mouth, fs.right_mouth
                )
                anchor = np.stack([
                    fs.left_eye, fs.right_eye,
                    fs.left_mouth, fs.right_mouth,
                    fs.left_brow, fs.right_brow,
                    chin, neck,
                ])
                eye_line_px = float(np.linalg.norm(fs.right_eye - fs.left_eye))
                hands_samples.append(tr.process(hs, eye_line_px, anchor))
            now = _time.time()
            if now - last_log > 3.0:
                pct = 100.0 * frame_idx / max(total, 1)
                print(f"  {frame_idx}/{total} ({pct:5.1f}%) n={len(face_samples)}")
                last_log = now
    finally:
        fd.close()
        if hd is not None:
            hd.close()
        cap.release()

    if len(face_samples) < 30:
        print(f"Too few face detections ({len(face_samples)})")
        return 3

    times = np.array([s.t for s in face_samples])

    # Baseline + face change report
    base_cut = face_samples[0].t + args.baseline_seconds
    base_samples = [s for s in face_samples if s.t < base_cut]
    baseline = fit_baseline(base_samples, robust=True)
    face_rep = detect_sigma_changes(face_samples, baseline)

    # M(t)
    au = compute_au_clusters(times, face_rep.per_feature_z)

    # F(t) — uses effective_region (the fix) which falls back to hand-state
    if args.no_hands or not any(hs and hs.n_hands > 0 for hs in hands_samples):
        f_raw = np.zeros_like(times)
        f_z = np.zeros_like(times)
        regions_per_frame = ["none"] * len(times)
        hand_state_codes = np.full_like(times, -1, dtype=float)
        kinetic_per_frame = np.zeros_like(times)
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
            kinetic_per_frame[i] = max(hs.finger_motion_L, hs.finger_motion_R)
            hand_state_codes[i] = float(hs.hand_state_code)
        f_raw, f_z = compute_fidget_index(
            times, regions_per_frame, kinetic_per_frame,
            window_s=args.fidget_window_s,
        )

    # V(t)
    if args.no_voice:
        v_times, v_values = times, np.zeros_like(times)
        voice_meta = None
    else:
        try:
            vr = compute_voice(args.video, bin_s=1.0,
                               baseline_seconds=args.baseline_seconds,
                               estimate_response_latency=False)
            v_times, v_values = vr.times, vr.v
            voice_meta = {
                "sps_baseline_mean": vr.sps_baseline_mean,
                "sps_baseline_std": vr.sps_baseline_std,
                "sps_per_bin": vr.sps.tolist(),
            }
        except Exception as e:
            print(f"voice failed: {e}")
            v_times, v_values = times, np.zeros_like(times)
            voice_meta = {"error": str(e)}

    # Syntonia(t)
    syntonia_rep = compute_syntonia(
        times,
        v_times, v_values,
        times, f_z,
        times, au.m,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        threshold=args.threshold, window_s=args.window_s,
    )

    # Collect event-time peak-feature info for the event log text.
    face_events = []
    for e in face_rep.events:
        face_events.append({
            "detector": e.detector, "sigma_level": e.sigma_level,
            "start_t": e.start_t, "end_t": e.end_t,
            "duration_s": e.duration_s, "peak_z": e.peak_z, "peak_t": e.peak_t,
            "dominant_feature": e.dominant_feature,
        })

    # ---- Speech transcript (local-only via faster-whisper) ----------------
    transcript_meta = None
    if not args.no_voice and not args.no_transcript:
        if transcript_available():
            try:
                print("Transcribing audio locally (faster-whisper small, no network)…")
                tr = transcribe_audio(args.video)
                transcript_meta = tr.to_dict()
                print(f"  language={tr.language} ({tr.language_probability:.2f}), "
                      f"{len(tr.segments)} segments")
            except Exception as e:
                print(f"  transcript failed: {e}")
                transcript_meta = {"error": str(e)}
        else:
            transcript_meta = {
                "error": "Whisper model not present. Run scripts/fetch_models.sh on a "
                         "machine with network access (~250 MB one-time download).",
            }
            print(f"  {transcript_meta['error']}")

    if args.ephemeral:
        # In-memory mode: emit the summary and exit without writing artifacts.
        print(f"\n[ephemeral] No cache written. "
              f"Summary: max M={au.m.max():.2f}σ  max F={f_z.max():.2f}σ  "
              f"max V={v_values.max():.2f}σ  max Syntonia={syntonia_rep.syntonia.max():.2f}  "
              f"breach windows={len(syntonia_rep.windows)}")
        return 0

    np.savez(
        cache,
        # core arrays
        times=times,
        face_overall_z=face_rep.overall_z,
        face_t2_eq_sigma=face_rep.t2_equivalent_sigma,
        m=au.m,
        f_z=f_z,
        f_raw=f_raw,
        v_times=v_times,
        v_values=v_values,
        syntonia=syntonia_rep.syntonia,
        regions=np.array(regions_per_frame),
        hand_state_codes=hand_state_codes,
        kinetic_per_frame=kinetic_per_frame,
        sample_frame_idx=np.array(sample_frame_idx, dtype=int),
        # cluster z timeline (n_frames × n_clusters)
        cluster_z=au.cluster_z,
        cluster_names=np.array(au.cluster_names),
        # config + metadata as a json blob
        meta_json=json.dumps({
            "video": str(args.video),
            "fps": fps, "frame_size": [w, h],
            "frame_skip": args.frame_skip,
            "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
            "threshold": args.threshold,
            "window_s": args.window_s,
            "fidget_window_s": args.fidget_window_s,
            "baseline_seconds": args.baseline_seconds,
            "baseline_hash": baseline.hash(),
            "feature_names": list(FEATURE_NAMES),
            "au_weights": dict(DEFAULT_AU_WEIGHTS),
            "region_weights": dict(REGION_WEIGHTS),
            "voice": voice_meta,
            "face_events": face_events,
            "syntonia_windows": [w.to_dict() for w in syntonia_rep.windows],
            "summary": {
                "n_face": int(len(face_samples)),
                "n_hands": int(sum(1 for hs in hands_samples if hs and hs.n_hands > 0)),
                "max_m": float(au.m.max()),
                "max_f": float(f_z.max()),
                "max_v": float(v_values.max()),
                "max_syntonia": float(syntonia_rep.syntonia.max()),
            },
            "transcript": transcript_meta,
        }),
    )
    print(f"\nSaved cache: {cache}")
    print(f"Summary: max M={au.m.max():.2f}σ  max F={f_z.max():.2f}σ  "
          f"max V={v_values.max():.2f}σ  max Syntonia={syntonia_rep.syntonia.max():.2f}  "
          f"breach windows={len(syntonia_rep.windows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
