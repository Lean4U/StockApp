"""Standalone OpenCV demo for the face-trapezium signature + 3σ/6σ detection.

Controls (with the OpenCV window focused):
    r  - toggle recording
    c  - clear current samples
    b  - fit baseline from current samples and enroll under a name (prompt)
    m  - match current samples to enrolled baselines (nearest neighbor)
    d  - detect 3σ / 6σ change events of current samples vs. a chosen baseline
    s  - show signature hash + feature vector preview
    q  - quit

Baselines persist to ``signatures.json``, shared with face_recognition_app.py.

Run:  python webcam_demo.py [--camera 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from face_trapezium import (
    Baseline,
    FaceTrapeziumDetector,
    TrapeziumSample,
    detect_sigma_changes,
    feature_vector,
    fit_baseline,
    signature_distance,
)

SIGNATURE_FILE = Path("signatures.json")


def load_signatures() -> Dict[str, Baseline]:
    if not SIGNATURE_FILE.exists():
        return {}
    raw = json.loads(SIGNATURE_FILE.read_text())
    return {name: Baseline.from_dict(payload) for name, payload in raw.items()}


def save_signatures(sigs: Dict[str, Baseline]) -> None:
    SIGNATURE_FILE.write_text(
        json.dumps({n: b.to_dict() for n, b in sigs.items()}, indent=2)
    )


def draw_overlay(img: np.ndarray, sample: TrapeziumSample, recording: bool, n: int) -> None:
    pts = np.array(
        [
            sample.left_eye[:2],
            sample.right_eye[:2],
            sample.right_mouth[:2],
            sample.left_mouth[:2],
        ],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 200, 255), 2)
    cv2.line(img, tuple(pts[3]), tuple(pts[2]), (0, 200, 255), 2)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
    # Brow points + vertical brow-to-eye-corner guides
    for brow, eye in (
        (sample.left_brow, sample.left_eye),
        (sample.right_brow, sample.right_eye),
    ):
        bx, by = int(brow[0]), int(brow[1])
        ex, ey = int(eye[0]), int(eye[1])
        cv2.line(img, (ex, ey), (ex, by), (180, 255, 0), 1)
        cv2.circle(img, (bx, by), 4, (180, 255, 0), -1)
    cx, cy = int(sample.centroid[0]), int(sample.centroid[1])
    cv2.circle(img, (cx, cy), 5, (255, 255, 0), -1)
    cv2.putText(
        img,
        f"eye_line={sample.eye_line:5.1f}  mouth_line={sample.mouth_line:5.1f}  "
        f"ratio={sample.eye_mouth_ratio:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    status = f"REC  n={n}" if recording else "idle  (press r to record)"
    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(img, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(
        img,
        "r=rec  c=clear  b=baseline  m=match  d=detect  s=show  q=quit",
        (10, img.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--sigma-low", type=float, default=3.0)
    parser.add_argument("--sigma-high", type=float, default=6.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return 1

    signatures = load_signatures()
    samples: List[TrapeziumSample] = []
    recording = False
    start_t: float | None = None

    with FaceTrapeziumDetector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            now = time.time()
            if start_t is None:
                start_t = now
            sample = detector.detect(frame, now - start_t)
            if sample is not None:
                draw_overlay(frame, sample, recording, len(samples))
                if recording:
                    samples.append(sample)
            else:
                cv2.putText(frame, "no face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("face trapezium", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                if recording and not samples:
                    start_t = time.time()
                print("recording" if recording else "stopped")
            elif key == ord("c"):
                samples.clear()
                start_t = None
                print("cleared samples")
            elif key == ord("b"):
                base = fit_baseline(samples)
                if base is None:
                    print("need more samples")
                    continue
                name = input("subject name: ").strip() or base.hash()
                signatures[name] = base
                save_signatures(signatures)
                print(f"enrolled '{name}'  hash={base.hash()}  "
                      f"n={base.n_samples}  duration={base.duration_s:.2f}s")
            elif key == ord("m"):
                base = fit_baseline(samples)
                if base is None or not signatures:
                    print("need samples and at least one enrolled baseline")
                    continue
                ranked = sorted(
                    ((n, signature_distance(base, b)) for n, b in signatures.items()),
                    key=lambda r: r[1],
                )
                print("matches (nearest first):")
                for n, d in ranked:
                    print(f"  {n:20s}  distance={d:.4f}")
            elif key == ord("d"):
                if not signatures:
                    print("no enrolled baselines")
                    continue
                if len(samples) < 2:
                    print("need at least 2 samples")
                    continue
                name = input("baseline name to overlay: ").strip()
                if name not in signatures:
                    print(f"unknown baseline '{name}'")
                    continue
                report = detect_sigma_changes(
                    samples, signatures[name],
                    sigma_low=args.sigma_low, sigma_high=args.sigma_high,
                )
                summary = report.summary()
                print(f"frames={summary['n_samples']}  "
                      f"duration={summary['duration_s']:.2f}s  "
                      f"max RMS|z|={summary['max_overall_z']:.2f}  "
                      f"max T²σ={summary['max_t2_sigma']:.2f}")
                print(f"events: RMS {summary['n_rms_events_3sigma']}x3σ "
                      f"{summary['n_rms_events_6sigma']}x6σ  | "
                      f"T² {summary['n_t2_events_3sigma']}x3σ "
                      f"{summary['n_t2_events_6sigma']}x6σ  | "
                      f"CUSUM {summary['n_cusum_events']}")
                for e in summary["events"]:
                    print(f"  {e['detector'].upper():3s} {e['sigma_level']}σ  "
                          f"t=[{e['start_t']:.2f}, {e['end_t']:.2f}]s  "
                          f"({e['duration_s']:.2f}s)  "
                          f"peak={e['peak_z']:.2f}σ  "
                          f"feature={e['dominant_feature']}")
                for e in summary["cusum_events"][:20]:
                    print(f"  CUSUM {e['direction']:>4s}  "
                          f"t=[{e['start_t']:.2f}, {e['end_t']:.2f}]s  "
                          f"S={e['peak_cusum']:.2f}  "
                          f"feature={e['feature_name']}")
            elif key == ord("s"):
                base = fit_baseline(samples)
                if base is None:
                    print("need more samples")
                    continue
                print(f"hash={base.hash()}  n={base.n_samples}  "
                      f"duration={base.duration_s:.2f}s")
                print(f"  means preview: {np.round(base.means[:8], 4)}")
                print(f"  stds  preview: {np.round(base.stds[:8], 4)}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
