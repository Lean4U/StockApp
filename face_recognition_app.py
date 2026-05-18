"""Streamlit app: live webcam, draws the eye-eye-nose triangle, records its motion,
fits a unique mathematical function f(t), and supports enrollment + identification.

Run:  streamlit run face_recognition_app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from face_triangle import (
    FaceTriangleDetector,
    TriangleSample,
    fit_signature,
    signature_distance,
    signature_hash,
    signature_vector,
)

try:
    import av
    import cv2
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    _WEBRTC_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _WEBRTC_AVAILABLE = False
    _WEBRTC_ERROR = exc


SIGNATURE_FILE = Path("signatures.json")


def load_signatures() -> Dict[str, dict]:
    if not SIGNATURE_FILE.exists():
        return {}
    raw = json.loads(SIGNATURE_FILE.read_text())
    out: Dict[str, dict] = {}
    for name, sig in raw.items():
        restored = {}
        for k, v in sig.items():
            restored[k] = np.asarray(v) if isinstance(v, list) else v
        out[name] = restored
    return out


def save_signatures(sigs: Dict[str, dict]) -> None:
    serializable = {
        name: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sig.items()}
        for name, sig in sigs.items()
    }
    SIGNATURE_FILE.write_text(json.dumps(serializable, indent=2))


if _WEBRTC_AVAILABLE:

    class TriangleProcessor(VideoProcessorBase):
        """Per-frame: detect landmarks, draw the triangle overlay, optionally record."""

        def __init__(self) -> None:
            self.detector = FaceTriangleDetector()
            self.samples: List[TriangleSample] = []
            self.recording = False
            self.start_time: float | None = None
            self.max_samples = 600

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            now = time.time()
            if self.start_time is None:
                self.start_time = now
            t = now - self.start_time

            sample = self.detector.detect(img, t)
            if sample is not None:
                pts = np.array(
                    [sample.left_eye[:2], sample.right_eye[:2], sample.nose[:2]],
                    dtype=np.int32,
                )
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
                status = f"REC  n={len(self.samples)}" if self.recording else "idle"
                color = (0, 0, 255) if self.recording else (200, 200, 200)
                cv2.putText(img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if self.recording:
                    self.samples.append(sample)
                    if len(self.samples) > self.max_samples:
                        self.samples.pop(0)

            return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Face Triangle Signature", layout="wide")
st.title("Facial Triangle Signature")
st.markdown(
    "Detects the **triangle between your two eyes and your nose tip**, tracks "
    "how it moves in 3D over time, and fits a unique function "
    "**f(t) = Fourier(centroid_xyz, normal_theta_phi) + invariant stats** that "
    "serves as a recognition signature."
)

if not _WEBRTC_AVAILABLE:
    st.error(
        "streamlit-webrtc / opencv / av are not installed. Install with "
        "`pip install -r requirements.txt`."
    )
    st.stop()

if "signatures" not in st.session_state:
    st.session_state.signatures = load_signatures()

left, right = st.columns([3, 2])

with left:
    ctx = webrtc_streamer(
        key="face-triangle",
        video_processor_factory=TriangleProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    btn_cols = st.columns(3)
    if btn_cols[0].button("Start / Stop recording") and ctx.video_processor:
        ctx.video_processor.recording = not ctx.video_processor.recording
    if btn_cols[1].button("Clear samples") and ctx.video_processor:
        ctx.video_processor.samples = []
        ctx.video_processor.start_time = None
    refresh = btn_cols[2].button("Refresh charts")
    _ = refresh  # button click forces a Streamlit rerun

with right:
    st.subheader("Enrollment")
    name = st.text_input("Subject name", placeholder="e.g. alice")
    if st.button("Save signature for this subject"):
        if not ctx.video_processor:
            st.warning("Start the camera first.")
        else:
            sig = fit_signature(ctx.video_processor.samples)
            if not sig:
                st.warning("Need more samples first — record ~5 seconds of motion.")
            else:
                key = name.strip() or signature_hash(sig)
                st.session_state.signatures[key] = sig
                save_signatures(st.session_state.signatures)
                st.success(f"Saved '{key}'  (hash {signature_hash(sig)})")

    st.subheader("Identify")
    if st.button("Match current recording against enrolled"):
        if not ctx.video_processor:
            st.warning("Start the camera first.")
        else:
            sig = fit_signature(ctx.video_processor.samples)
            if not sig:
                st.warning("Not enough samples to fit a signature.")
            elif not st.session_state.signatures:
                st.info("No enrolled signatures yet.")
            else:
                rows = []
                for k, v in st.session_state.signatures.items():
                    rows.append({"name": k, "distance": signature_distance(sig, v)})
                rows.sort(key=lambda r: r["distance"])
                st.table(
                    [{"name": r["name"], "distance": f"{r['distance']:.4f}"} for r in rows]
                )

    if st.session_state.signatures:
        st.caption("Enrolled signatures")
        st.write({k: signature_hash(v) for k, v in st.session_state.signatures.items()})
        if st.button("Delete all enrolled signatures"):
            st.session_state.signatures = {}
            save_signatures({})

if ctx.video_processor and ctx.video_processor.samples:
    samples = ctx.video_processor.samples
    df = pd.DataFrame(
        {
            "time": [s.t for s in samples],
            "area": [s.area for s in samples],
            "perimeter": [s.perimeter for s in samples],
            "centroid_x": [s.centroid[0] for s in samples],
            "centroid_y": [s.centroid[1] for s in samples],
            "centroid_z": [s.centroid[2] for s in samples],
            "alpha": [s.angles[0] for s in samples],
            "beta": [s.angles[1] for s in samples],
            "gamma": [s.angles[2] for s in samples],
        }
    ).set_index("time")

    st.subheader("Triangle invariants over time")
    c1, c2, c3 = st.columns(3)
    c1.line_chart(df[["area"]])
    c2.line_chart(df[["alpha", "beta", "gamma"]])
    c3.line_chart(df[["centroid_x", "centroid_y", "centroid_z"]])

    st.subheader("Current signature f(t)")
    current_sig = fit_signature(samples)
    if current_sig:
        st.write(f"Signature hash: `{signature_hash(current_sig)}`  "
                 f"({len(samples)} samples over {current_sig['duration_s']:.2f}s)")
        st.caption("Concatenated coefficient vector (first 32 entries)")
        st.line_chart(pd.DataFrame({"coef": signature_vector(current_sig)[:32]}))
