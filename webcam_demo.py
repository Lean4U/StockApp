"""Standalone OpenCV demo for the face-triangle signature.

Controls (with the OpenCV window focused):
    r  - toggle recording
    c  - clear current samples
    e  - enroll current samples as a named signature (prompts on stdout)
    i  - identify current samples against the enrolled set
    s  - print the current signature hash + vector preview
    q  - quit

Enrolled signatures are persisted to ``signatures.json`` so they are shared
with face_recognition_app.py.

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

from face_triangle import (
    FaceTriangleDetector,
    TriangleSample,
    fit_signature,
    signature_distance,
    signature_hash,
    signature_vector,
)

SIGNATURE_FILE = Path("signatures.json")


def load_signatures() -> Dict[str, dict]:
    if not SIGNATURE_FILE.exists():
        return {}
    raw = json.loads(SIGNATURE_FILE.read_text())
    out: Dict[str, dict] = {}
    for name, sig in raw.items():
        out[name] = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in sig.items()}
    return out


def save_signatures(sigs: Dict[str, dict]) -> None:
    serializable = {
        name: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sig.items()}
        for name, sig in sigs.items()
    }
    SIGNATURE_FILE.write_text(json.dumps(serializable, indent=2))


def draw_overlay(img: np.ndarray, sample: TriangleSample, recording: bool, n: int) -> None:
    pts = np.array([sample.left_eye[:2], sample.right_eye[:2], sample.nose[:2]], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
    cx, cy = int(sample.centroid[0]), int(sample.centroid[1])
    cv2.circle(img, (cx, cy), 5, (255, 255, 0), -1)
    cv2.putText(
        img,
        f"area={sample.area:6.0f}  perim={sample.perimeter:6.0f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    status = f"REC  n={n}" if recording else "idle  (press r to record)"
    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(
        img,
        "r=record  c=clear  e=enroll  i=identify  s=show  q=quit",
        (10, img.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--harmonics", type=int, default=4)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return 1

    signatures = load_signatures()
    samples: List[TriangleSample] = []
    recording = False
    start_t: float | None = None

    with FaceTriangleDetector() as detector:
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
                cv2.putText(
                    frame, "no face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            cv2.imshow("face triangle", frame)
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
            elif key == ord("e"):
                sig = fit_signature(samples, harmonics=args.harmonics)
                if not sig:
                    print("need more samples first")
                    continue
                name = input("subject name: ").strip() or signature_hash(sig)
                signatures[name] = sig
                save_signatures(signatures)
                print(f"enrolled '{name}'  hash={signature_hash(sig)}")
            elif key == ord("i"):
                sig = fit_signature(samples, harmonics=args.harmonics)
                if not sig:
                    print("need more samples first")
                    continue
                if not signatures:
                    print("no enrolled signatures")
                    continue
                ranked = sorted(
                    ((n, signature_distance(sig, s)) for n, s in signatures.items()),
                    key=lambda r: r[1],
                )
                print("matches (nearest first):")
                for n, d in ranked:
                    print(f"  {n:20s}  distance={d:.4f}")
            elif key == ord("s"):
                sig = fit_signature(samples, harmonics=args.harmonics)
                if not sig:
                    print("need more samples first")
                    continue
                vec = signature_vector(sig)
                print(f"hash={signature_hash(sig)}  dim={vec.size}  "
                      f"preview={np.round(vec[:8], 4)}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
