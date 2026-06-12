"""SyntoniaPro — live webcam dashboard (M-channel only).

Streamlit page that mirrors webcam_demo.py inside the browser:
  * live webcam feed with trapezium overlay
  * live T² / RMS / CUSUM metrics against an enrolled baseline
  * record samples → enroll → detect, all from sidebar buttons
  * baselines persisted to signatures.json (shared with webcam_demo.py)

Privacy posture is the same as the batch dashboard: frames stay on this
machine, Streamlit telemetry is disabled by .streamlit/config.toml, the
server binds to 127.0.0.1.

Run:
    streamlit run dashboard_live.py

Limitations of v1:
  * M-channel only — no V (voice) or F (hands) yet
  * Camera is held for the lifetime of the Streamlit process; press Ctrl-C
    in the launching terminal to release it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from face_trapezium import (
    Baseline,
    FaceTrapeziumDetector,
    TrapeziumSample,
    _LIVE_STD_FLOOR,
    detect_sigma_changes,
    fit_baseline,
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


@st.cache_resource
def get_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    return cap


@st.cache_resource
def get_detector() -> FaceTrapeziumDetector:
    return FaceTrapeziumDetector()


def draw_trapezium(img: np.ndarray, sample: TrapeziumSample, recording: bool) -> None:
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
    status = "REC" if recording else "idle"
    color = (0, 0, 255) if recording else (180, 180, 180)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("samples", [])
    ss.setdefault("recording", False)
    ss.setdefault("start_t", None)
    ss.setdefault("signatures", load_signatures())
    ss.setdefault("event_log", [])


def main() -> None:
    st.set_page_config(page_title="SyntoniaPro — Live", layout="wide")
    st.title("SyntoniaPro — Live (M-channel)")
    st.caption(
        "Local webcam → trapezium → live Hotelling T² / RMS / CUSUM against an "
        "enrolled baseline. No cloud, no data leaves this device."
    )

    init_state()
    ss = st.session_state

    with st.sidebar:
        st.header("Camera")
        camera_idx = st.number_input("Camera index", 0, 4, 0, step=1)

        st.header("Baseline")
        names = list(ss.signatures.keys())
        baseline_choice = st.selectbox(
            "Compare against",
            options=["— none —"] + names,
            index=0,
        )

        st.header("Detector")
        sigma_low = st.slider("σ low threshold", 1.0, 6.0, 3.0, 0.1)
        sigma_high = st.slider("σ high threshold", 3.0, 10.0, 6.0, 0.1)

        st.header("Recording")
        c1, c2 = st.columns(2)
        if c1.button("Record ▶" if not ss.recording else "Stop ■", use_container_width=True):
            ss.recording = not ss.recording
            if ss.recording and not ss.samples:
                ss.start_t = time.time()
        if c2.button("Clear", use_container_width=True):
            ss.samples = []
            ss.start_t = None
            ss.event_log = []

        st.caption(f"Samples collected: **{len(ss.samples)}**")
        if ss.samples:
            dur = ss.samples[-1].t - ss.samples[0].t
            st.caption(f"Duration: **{dur:.2f} s**")

        st.header("Enroll baseline")
        new_name = st.text_input("Name for current samples", value="")
        if st.button("Enroll", use_container_width=True):
            base = fit_baseline(ss.samples, std_floor=_LIVE_STD_FLOOR)
            if base is None:
                st.error("Need more samples first.")
            elif not new_name.strip():
                st.error("Type a name.")
            else:
                ss.signatures[new_name.strip()] = base
                save_signatures(ss.signatures)
                st.success(
                    f"Enrolled '{new_name.strip()}' — n={base.n_samples}, "
                    f"duration={base.duration_s:.2f}s"
                )

        st.header("Detect")
        if st.button("Detect events vs selected baseline", use_container_width=True):
            if baseline_choice == "— none —":
                st.error("Pick a baseline first.")
            elif len(ss.samples) < 5:
                st.error("Need at least 5 samples.")
            else:
                report = detect_sigma_changes(
                    ss.samples,
                    ss.signatures[baseline_choice],
                    sigma_low=sigma_low,
                    sigma_high=sigma_high,
                )
                ss.event_log = report.summary()

        run_live = st.toggle("Run live", value=True)

    # Main pane.
    col_video, col_metrics = st.columns([3, 2])
    video_slot = col_video.empty()
    metric_slot = col_metrics.empty()
    event_slot = st.empty()

    if not run_live:
        video_slot.info("Live mode paused. Toggle 'Run live' in the sidebar to resume.")
    else:
        cap = get_camera(int(camera_idx))
        if not cap.isOpened():
            st.error(f"Could not open camera index {camera_idx}.")
            st.stop()
        detector = get_detector()

        @st.fragment(run_every="100ms")
        def live_tick():
            ok, frame = cap.read()
            if not ok:
                video_slot.warning("Camera read failed.")
                return

            now = time.time()
            if ss.start_t is None:
                ss.start_t = now
            t = now - ss.start_t

            sample = detector.detect(frame, t)
            if sample is not None:
                draw_trapezium(frame, sample, ss.recording)
                if ss.recording:
                    ss.samples.append(sample)
            else:
                cv2.putText(
                    frame,
                    "no face",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if w > 720:
                rgb = cv2.resize(rgb, (720, int(h * 720 / w)))
            video_slot.image(rgb, channels="RGB")

            # Live rolling metric against the chosen baseline (last ~1 s of samples).
            with metric_slot.container():
                if baseline_choice != "— none —" and len(ss.samples) >= 5:
                    base = ss.signatures[baseline_choice]
                    tail = ss.samples[-60:]  # ~2 s at 30 fps
                    rep = detect_sigma_changes(
                        tail, base, sigma_low=sigma_low, sigma_high=sigma_high
                    )
                    s = rep.summary()
                    st.metric("Hotelling T² (peak σ)", f"{s['max_t2_sigma']:.2f}")
                    st.metric("RMS |z| (peak σ)", f"{s['max_overall_z']:.2f}")
                    st.metric("CUSUM events (window)", s["n_cusum_events"])
                    if s["max_t2_sigma"] >= sigma_high:
                        st.error("⚠ BREACH (T² > σ high)")
                    elif s["max_t2_sigma"] >= sigma_low:
                        st.warning("Excursion (T² > σ low)")
                    else:
                        st.success("Stable")
                else:
                    st.metric("Live samples", len(ss.samples))
                    st.caption(
                        "Enroll a baseline and pick it under 'Compare against' "
                        "to see live σ metrics."
                    )

        live_tick()

    # Persistent event log from the last "Detect" run.
    if ss.event_log:
        s = ss.event_log
        with event_slot.container():
            st.subheader("Last detect-run event list")
            st.write(
                f"frames={s['n_samples']}  "
                f"duration={s['duration_s']:.2f}s  "
                f"max RMS|z|={s['max_overall_z']:.2f}  "
                f"max T²σ={s['max_t2_sigma']:.2f}"
            )
            st.write(
                f"RMS: {s['n_rms_events_3sigma']}×3σ  "
                f"{s['n_rms_events_6sigma']}×6σ  · "
                f"T²: {s['n_t2_events_3sigma']}×3σ  "
                f"{s['n_t2_events_6sigma']}×6σ  · "
                f"CUSUM: {s['n_cusum_events']}"
            )
            for e in s["events"][:40]:
                st.text(
                    f"{e['detector'].upper():3s} {e['sigma_level']}σ  "
                    f"t=[{e['start_t']:.2f}, {e['end_t']:.2f}]s  "
                    f"({e['duration_s']:.2f}s)  "
                    f"peak={e['peak_z']:.2f}σ  "
                    f"feature={e['dominant_feature']}"
                )


if __name__ == "__main__":
    main()
