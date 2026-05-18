"""Streamlit app: live webcam, draws the eye-eye-mouth-mouth trapezium, records
samples, fits a baseline signature for the subject, and detects 3σ / 6σ
deviation events when a fresh recording is overlaid on the baseline.

Run:  streamlit run face_recognition_app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from face_trapezium import (
    Baseline,
    FEATURE_NAMES,
    FaceTrapeziumDetector,
    TrapeziumSample,
    detect_sigma_changes,
    feature_vector,
    fit_baseline,
    signature_distance,
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


def load_signatures() -> Dict[str, Baseline]:
    if not SIGNATURE_FILE.exists():
        return {}
    raw = json.loads(SIGNATURE_FILE.read_text())
    return {name: Baseline.from_dict(payload) for name, payload in raw.items()}


def save_signatures(sigs: Dict[str, Baseline]) -> None:
    SIGNATURE_FILE.write_text(
        json.dumps({n: b.to_dict() for n, b in sigs.items()}, indent=2)
    )


if _WEBRTC_AVAILABLE:

    class TrapeziumProcessor(VideoProcessorBase):
        """Per-frame: detect 4 landmarks, draw the trapezium, optionally record."""

        def __init__(self) -> None:
            self.detector = FaceTrapeziumDetector()
            self.samples: List[TrapeziumSample] = []
            self.recording = False
            self.start_time: Optional[float] = None
            self.max_samples = 900

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            now = time.time()
            if self.start_time is None:
                self.start_time = now
            t = now - self.start_time

            sample = self.detector.detect(img, t)
            if sample is not None:
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
                # Highlight the two reference lines (eye-line, mouth-line)
                cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 200, 255), 2)
                cv2.line(img, tuple(pts[3]), tuple(pts[2]), (0, 200, 255), 2)
                for p in pts:
                    cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
                cx, cy = int(sample.centroid[0]), int(sample.centroid[1])
                cv2.circle(img, (cx, cy), 5, (255, 255, 0), -1)

                cv2.putText(
                    img,
                    f"area={sample.area:6.0f}  perim={sample.perimeter:6.0f}  "
                    f"ratio={sample.eye_mouth_ratio:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                status = f"REC  n={len(self.samples)}" if self.recording else "idle"
                color = (0, 0, 255) if self.recording else (200, 200, 200)
                cv2.putText(img, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if self.recording:
                    self.samples.append(sample)
                    if len(self.samples) > self.max_samples:
                        self.samples.pop(0)

            return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Face Trapezium Signature", layout="wide")
st.title("Facial Trapezium Signature  (3σ / 6σ change detection)")
st.markdown(
    "Tracks the **trapezium formed by the outer eye corners and the mouth "
    "corners** over time. A subject's *baseline* is the mean ± std of "
    "15 scale-invariant dimensional features (sides, interior angles, "
    "diagonals, eye/mouth-line ratio, parallelism residual). New recordings "
    "are overlaid on the baseline and any contiguous window where the "
    "aggregate z-score exceeds **3σ** (or **6σ**) is reported as a "
    "statistically significant change event."
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
        key="face-trapezium",
        video_processor_factory=TrapeziumProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    cols = st.columns(3)
    if cols[0].button("Start / Stop recording") and ctx.video_processor:
        ctx.video_processor.recording = not ctx.video_processor.recording
    if cols[1].button("Clear samples") and ctx.video_processor:
        ctx.video_processor.samples = []
        ctx.video_processor.start_time = None
    cols[2].button("Refresh charts")  # forces a Streamlit rerun

with right:
    st.subheader("Enroll baseline")
    name = st.text_input("Subject name", placeholder="e.g. alice")
    if st.button("Save baseline as signature"):
        if not ctx.video_processor:
            st.warning("Start the camera first.")
        else:
            base = fit_baseline(ctx.video_processor.samples)
            if base is None:
                st.warning("Need at least 2 frames — record a few seconds first.")
            else:
                key = name.strip() or base.hash()
                st.session_state.signatures[key] = base
                save_signatures(st.session_state.signatures)
                st.success(
                    f"Saved baseline '{key}'  hash={base.hash()}  "
                    f"({base.n_samples} frames over {base.duration_s:.2f}s)"
                )

    st.subheader("Identify")
    if st.button("Match current recording to enrolled"):
        if not ctx.video_processor:
            st.warning("Start the camera first.")
        else:
            base = fit_baseline(ctx.video_processor.samples)
            if base is None:
                st.warning("Not enough samples.")
            elif not st.session_state.signatures:
                st.info("No enrolled signatures yet.")
            else:
                rows = [
                    {"name": k, "distance": signature_distance(base, v)}
                    for k, v in st.session_state.signatures.items()
                ]
                rows.sort(key=lambda r: r["distance"])
                st.table([{"name": r["name"], "distance": f"{r['distance']:.4f}"} for r in rows])

    if st.session_state.signatures:
        st.caption("Enrolled signatures")
        st.write({k: v.hash() for k, v in st.session_state.signatures.items()})
        if st.button("Delete all enrolled signatures"):
            st.session_state.signatures = {}
            save_signatures({})

st.divider()

st.subheader("Statistical change detection (overlay vs. baseline)")
sig_options = list(st.session_state.signatures.keys())
selected = st.selectbox("Baseline to overlay", sig_options, index=0 if sig_options else None)
c_low = st.slider("Lower sigma threshold (significant change)", 1.0, 6.0, 3.0, 0.5)
c_high = st.slider("Upper sigma threshold (extreme change)", 3.0, 10.0, 6.0, 0.5)
min_run = st.slider("Minimum window length (frames)", 1, 30, 2)

if (
    selected
    and ctx.video_processor
    and ctx.video_processor.samples
):
    samples = ctx.video_processor.samples
    baseline = st.session_state.signatures[selected]
    report = detect_sigma_changes(
        samples, baseline,
        sigma_low=c_low, sigma_high=c_high,
        min_run_samples=min_run,
    )

    s1, s2, s3 = st.columns(3)
    s1.metric("Frames analyzed", report.times.size)
    s2.metric("Max |z|", f"{report.overall_z.max():.2f}" if report.overall_z.size else "—")
    s3.metric(
        "Events  (3σ / 6σ)",
        f"{sum(1 for e in report.events if e.sigma_level == 3)} / "
        f"{sum(1 for e in report.events if e.sigma_level == 6)}",
    )

    z_df = pd.DataFrame(
        {"time": report.times, "overall_z": report.overall_z}
    ).set_index("time")
    st.line_chart(z_df, height=240)

    if report.events:
        st.write("Detected change windows:")
        st.table(
            [
                {
                    "level": f"{e.sigma_level}σ",
                    "start (s)": f"{e.start_t:.2f}",
                    "end (s)":   f"{e.end_t:.2f}",
                    "duration (s)": f"{e.duration_s:.2f}",
                    "peak |z|": f"{e.peak_z:.2f}",
                    "dominant feature": e.dominant_feature,
                }
                for e in report.events
            ]
        )
    else:
        st.success("No samples exceeded the configured sigma thresholds.")

    feat_df = pd.DataFrame(report.per_feature_z, columns=list(FEATURE_NAMES))
    feat_df["time"] = report.times
    feat_df = feat_df.set_index("time")
    st.caption("Per-feature z-scores")
    st.line_chart(feat_df, height=240)

if ctx.video_processor and ctx.video_processor.samples:
    samples = ctx.video_processor.samples
    df = pd.DataFrame(
        {
            "time": [s.t for s in samples],
            "area": [s.area for s in samples],
            "perimeter": [s.perimeter for s in samples],
            "eye_line": [s.eye_line for s in samples],
            "mouth_line": [s.mouth_line for s in samples],
            "eye_mouth_ratio": [s.eye_mouth_ratio for s in samples],
            "parallelism": [s.parallelism_residual for s in samples],
        }
    ).set_index("time")
    st.subheader("Raw trapezium dimensions over time")
    c1, c2, c3 = st.columns(3)
    c1.line_chart(df[["eye_line", "mouth_line"]])
    c2.line_chart(df[["area"]])
    c3.line_chart(df[["eye_mouth_ratio", "parallelism"]])
