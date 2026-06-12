"""SyntoniaPro — live webcam dashboard (M-channel only).

Streamlit page mirroring webcam_demo.py inside the browser, with:

  * live webcam feed + trapezium overlay
  * live T² / RMS / CUSUM metrics vs an enrolled baseline
  * record → enroll → detect controls from the sidebar
  * adaptive baseline: re-fit hardware-sensitive features from the first
    few seconds of every session (mouth + brow features stay locked)
  * post-Detect "Session story" panel with plain-language, Socratic
    explanations per principle 4 of marketing/brand_architecture.md
  * Altair timeline of all events, color-coded by feature category
  * thumbnail of the frame captured at the peak-T² moment of each event

Run:
    streamlit run dashboard_live.py
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from face_trapezium import (
    Baseline,
    FaceTrapeziumDetector,
    TrapeziumSample,
    _LIVE_STD_FLOOR,
    adapt_baseline,
    detect_sigma_changes,
    fit_baseline,
)
from feature_glossary import category, explain

SIGNATURE_FILE = Path("signatures.json")

CATEGORY_PALETTE = {
    "behavioural": "#d9534f",  # red — the signal we care about
    "pose": "#f0ad4e",          # amber — informative but background
    "hardware": "#6c757d",      # grey — artifact, should be absorbed
    "other": "#5bc0de",
}


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
    return cv2.VideoCapture(index)


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
    ss.setdefault("frame_thumbnails", OrderedDict())
    ss.setdefault("active_baseline_key", None)


def session_intro() -> None:
    """Plain-English orientation banner — Socratic, no instructions."""
    st.markdown(
        """
        ### What this page does

        Your webcam feeds into the **M-channel** of the Syntonia model — the
        geometry of your face moment by moment. Against a baseline you
        enrolled while at rest, it watches for departures: a sudden brow
        raise, a sustained head turn, a smile that pulls one side more than
        the other.

        It does not tell you what your face *meant*. It reports what your
        face *did*. The interpretation stays with you.

        > **Three sources of departure to keep in mind:**
        > 1. **Behavioural** — your brows, mouth corners, mouth offset. These
        >    are the signals worth noticing.
        > 2. **Pose** — head turns, nods, tilts. Often natural; sometimes
        >    informative.
        > 3. **Hardware** — glasses position, camera distance. The same
        >    geometry can shift without anything about you changing.
        >    Enable *Adaptive baseline* in the sidebar to absorb these.
        """
    )


def thumbnail_capture(
    frame: np.ndarray, sample_idx: int, every: int = 10
) -> None:
    """Stash a small BGR frame keyed by sample index, roughly every
    ``every`` samples (~1 s at the 10 fps fragment cadence).
    """
    if sample_idx % every != 0:
        return
    h, w = frame.shape[:2]
    target_w = 240
    if w > target_w:
        thumb = cv2.resize(frame, (target_w, int(h * target_w / w)))
    else:
        thumb = frame.copy()
    st.session_state.frame_thumbnails[sample_idx] = thumb
    if len(st.session_state.frame_thumbnails) > 200:
        st.session_state.frame_thumbnails.popitem(last=False)


def find_thumbnail_for_time(
    t: float, samples: List[TrapeziumSample]
) -> Optional[np.ndarray]:
    """Return the stored thumbnail captured closest in time to ``t``."""
    if not samples:
        return None
    sample_times = np.array([s.t for s in samples])
    closest_sample_idx = int(np.argmin(np.abs(sample_times - t)))
    thumbs = st.session_state.frame_thumbnails
    if not thumbs:
        return None
    keys = np.array(list(thumbs.keys()))
    nearest_key = int(keys[np.argmin(np.abs(keys - closest_sample_idx))])
    return thumbs[nearest_key]


def status_banner(t2_peak: float, sigma_low: float, sigma_high: float) -> None:
    if t2_peak >= sigma_high:
        st.error(
            f"**Breach** — T² {t2_peak:.1f} σ exceeds the high threshold "
            f"({sigma_high:.1f} σ). One or more features are well outside "
            f"your baseline. Notice this moment."
        )
    elif t2_peak >= sigma_low:
        st.warning(
            f"**Excursion** — T² {t2_peak:.1f} σ between the low "
            f"({sigma_low:.1f}) and high ({sigma_high:.1f}) σ thresholds. "
            f"Something is moving."
        )
    else:
        st.success(
            f"**Stable** — T² {t2_peak:.1f} σ, well within your baseline."
        )


def build_session_story(summary: dict, samples: List[TrapeziumSample]) -> str:
    """Translate the detect-run summary into a natural-language paragraph."""
    n = summary["n_samples"]
    dur = summary["duration_s"]
    t2_peak = summary["max_t2_sigma"]
    rms_peak = summary["max_overall_z"]
    events = summary.get("events", [])

    by_feature: Dict[str, dict] = {}
    for e in events:
        f = e["dominant_feature"]
        by_feature.setdefault(f, {"total_dur": 0.0, "peak_z": 0.0, "n": 0})
        by_feature[f]["total_dur"] += e["duration_s"]
        by_feature[f]["peak_z"] = max(by_feature[f]["peak_z"], e["peak_z"])
        by_feature[f]["n"] += 1

    sorted_features = sorted(
        by_feature.items(), key=lambda kv: -kv[1]["total_dur"]
    )
    bucket_counts = {"behavioural": 0, "pose": 0, "hardware": 0, "other": 0}
    for f in by_feature:
        bucket_counts[category(f)] += 1

    parts: List[str] = []
    parts.append(
        f"You recorded **{n} frames** over **{dur:.1f} seconds**. "
        f"Hotelling T² peaked at **{t2_peak:.1f} σ**, RMS at {rms_peak:.1f} σ."
    )

    if not events:
        parts.append(
            "Nothing crossed the σ-low threshold. Your face stayed within "
            "the geometric envelope of your baseline. Stable take."
        )
        return "\n\n".join(parts)

    top_feature, top_stats = sorted_features[0]
    fx = explain(top_feature)
    parts.append(
        f"The longest-sustained departure was on **{fx.short}** "
        f"(`{top_feature}`) — total {top_stats['total_dur']:.1f} s above "
        f"σ-low, peaking at {top_stats['peak_z']:.1f} σ."
    )
    parts.append(f"_What this feature measures._ {fx.physical}")
    cand_lines = "\n".join(f"- {c}" for c in fx.candidates)
    parts.append(f"_Compatible with any of:_\n{cand_lines}")
    if fx.citation:
        parts.append(f"_Cited science._ {fx.citation}")
    parts.append(f"**{fx.socratic}**")

    bucket_lines = []
    if bucket_counts["behavioural"]:
        bucket_lines.append(
            f"- **{bucket_counts['behavioural']} behavioural** feature(s) "
            f"fired — brow, mouth corner, mouth offset. These are the signals "
            f"the methodology is built to surface."
        )
    if bucket_counts["pose"]:
        bucket_lines.append(
            f"- **{bucket_counts['pose']} pose** feature(s) fired — head "
            f"turns, nods, tilts. Often natural; sometimes informative."
        )
    if bucket_counts["hardware"]:
        bucket_lines.append(
            f"- **{bucket_counts['hardware']} hardware** feature(s) fired — "
            f"eye-corner geometry sensitive to glasses position and distance. "
            f"If these dominate, keep *Adaptive baseline* enabled."
        )
    if bucket_lines:
        parts.append("**Breakdown by category:**\n" + "\n".join(bucket_lines))

    return "\n\n".join(parts)


def render_timeline(summary: dict) -> None:
    events = summary.get("events", [])
    if not events:
        return
    rows = []
    for e in events:
        f = e["dominant_feature"]
        rows.append(
            {
                "feature": explain(f).short + f"  ({f})",
                "start_t": e["start_t"],
                "end_t": e["end_t"],
                "peak_z": e["peak_z"],
                "duration_s": e["duration_s"],
                "detector": e["detector"].upper(),
                "sigma_level": e["sigma_level"],
                "category": category(f),
            }
        )
    df = pd.DataFrame(rows)
    order = (
        df.groupby("feature")["duration_s"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    color_scale = alt.Scale(
        domain=list(CATEGORY_PALETTE.keys()),
        range=list(CATEGORY_PALETTE.values()),
    )
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("start_t:Q", title="time (s)"),
            x2="end_t:Q",
            y=alt.Y("feature:N", sort=order, title=None),
            color=alt.Color(
                "category:N",
                scale=color_scale,
                legend=alt.Legend(title="signal category"),
            ),
            tooltip=[
                "feature",
                "category",
                "detector",
                alt.Tooltip("sigma_level:Q", title="σ level"),
                alt.Tooltip("start_t:Q", format=".2f"),
                alt.Tooltip("end_t:Q", format=".2f"),
                alt.Tooltip("duration_s:Q", title="duration s", format=".2f"),
                alt.Tooltip("peak_z:Q", title="peak σ", format=".2f"),
            ],
        )
        .properties(height=22 * max(len(order), 1) + 60)
    )
    st.altair_chart(chart, use_container_width=True)


def render_event_cards(summary: dict, samples: List[TrapeziumSample]) -> None:
    events = sorted(summary.get("events", []), key=lambda e: -e["peak_z"])
    if not events:
        return
    st.markdown("### Inflection points")
    st.caption(
        "Each card is a moment your face geometry departed from baseline. "
        "Thumbnails are pulled from the recording at the peak of each event."
    )
    for i, e in enumerate(events[:8]):
        f = e["dominant_feature"]
        fx = explain(f)
        cat = category(f)
        cat_color = CATEGORY_PALETTE[cat]
        peak_t = 0.5 * (e["start_t"] + e["end_t"])
        thumb = find_thumbnail_for_time(peak_t, samples)
        with st.expander(
            f"#{i + 1}  ·  {fx.short}  "
            f"·  {e['detector'].upper()} {e['sigma_level']}σ  "
            f"·  t=[{e['start_t']:.2f}, {e['end_t']:.2f}]s  "
            f"·  peak {e['peak_z']:.2f}σ",
            expanded=(i == 0),
        ):
            cols = st.columns([1, 2])
            with cols[0]:
                if thumb is not None:
                    st.image(
                        cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
                        caption=f"~t = {peak_t:.2f}s",
                    )
                else:
                    st.caption(
                        "(no frame thumbnail at this moment — try a longer "
                        "recording)"
                    )
            with cols[1]:
                st.markdown(
                    f"<span style='background:{cat_color};color:white;"
                    f"padding:2px 8px;border-radius:4px;font-size:0.8rem'>"
                    f"{cat.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**What the feature measures.** {fx.physical}")
                st.markdown("**Compatible with any of:**")
                for c in fx.candidates:
                    st.markdown(f"- {c}")
                if fx.citation:
                    st.markdown(f"**Cited science.** {fx.citation}")
                st.markdown(f"**A question to sit with:** {fx.socratic}")


def main() -> None:
    st.set_page_config(page_title="SyntoniaPro — Live", layout="wide")
    st.title("SyntoniaPro — Live (M-channel)")
    session_intro()
    st.markdown("---")

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
            help=(
                "The enrolled signature your live feed is being matched "
                "against. Different baseline per language / per session is "
                "encouraged."
            ),
        )
        adapt_on = st.checkbox(
            "Adaptive baseline",
            value=True,
            help=(
                "Recalibrate pose- and eyewear-sensitive features from the "
                "first few seconds of every session. Mouth-corner and brow "
                "features stay locked to the original enrollment, so real "
                "behavioural change still registers."
            ),
        )
        adapt_window_s = st.slider(
            "Adapt-from window (s)",
            2.0,
            15.0,
            5.0,
            0.5,
            help=(
                "How many seconds of fresh sit-still footage to use for "
                "adapting the geometric baseline."
            ),
            disabled=not adapt_on,
        )

        st.header("Detector")
        sigma_low = st.slider("σ low threshold", 1.0, 6.0, 3.0, 0.1)
        sigma_high = st.slider("σ high threshold", 3.0, 10.0, 6.0, 0.1)

        st.header("Recording")
        c1, c2 = st.columns(2)
        if c1.button(
            "Record ▶" if not ss.recording else "Stop ■",
            use_container_width=True,
        ):
            ss.recording = not ss.recording
            if ss.recording and not ss.samples:
                ss.start_t = time.time()
        if c2.button("Clear", use_container_width=True):
            ss.samples = []
            ss.start_t = None
            ss.event_log = []
            ss.frame_thumbnails = OrderedDict()

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
        if st.button(
            "Detect events vs selected baseline",
            use_container_width=True,
        ):
            if baseline_choice == "— none —":
                st.error("Pick a baseline first.")
            elif len(ss.samples) < 5:
                st.error("Need at least 5 samples.")
            else:
                base = ss.signatures[baseline_choice]
                if adapt_on:
                    cut_t = ss.samples[0].t + adapt_window_s
                    adapt_samples = [s for s in ss.samples if s.t <= cut_t]
                    base = adapt_baseline(base, adapt_samples)
                report = detect_sigma_changes(
                    ss.samples,
                    base,
                    sigma_low=sigma_low,
                    sigma_high=sigma_high,
                )
                ss.event_log = report.summary()
                ss.active_baseline_key = baseline_choice + (
                    " (adapted)" if adapt_on else ""
                )

        run_live = st.toggle("Run live", value=True)

    # Main pane.
    col_video, col_metrics = st.columns([3, 2])
    video_slot = col_video.empty()
    metric_slot = col_metrics.empty()
    event_slot = st.container()

    if not run_live:
        video_slot.info(
            "Live mode paused. Toggle 'Run live' in the sidebar to resume."
        )
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
                    sample_idx = len(ss.samples)
                    ss.samples.append(sample)
                    thumbnail_capture(frame, sample_idx)
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

            with metric_slot.container():
                if baseline_choice != "— none —" and len(ss.samples) >= 5:
                    base = ss.signatures[baseline_choice]
                    if adapt_on:
                        cut_t = ss.samples[0].t + adapt_window_s
                        adapt_samples = [s for s in ss.samples if s.t <= cut_t]
                        base = adapt_baseline(base, adapt_samples)
                    tail = ss.samples[-60:]
                    rep = detect_sigma_changes(
                        tail, base, sigma_low=sigma_low, sigma_high=sigma_high
                    )
                    s = rep.summary()
                    st.metric("Hotelling T² (peak σ)", f"{s['max_t2_sigma']:.2f}")
                    st.metric("RMS |z| (peak σ)", f"{s['max_overall_z']:.2f}")
                    st.metric("CUSUM events (window)", s["n_cusum_events"])
                    status_banner(s["max_t2_sigma"], sigma_low, sigma_high)
                else:
                    st.metric("Live samples", len(ss.samples))
                    st.caption(
                        "Enroll a baseline and pick it under 'Compare "
                        "against' to see live σ metrics."
                    )

        live_tick()

    # Persistent post-Detect panel.
    if ss.event_log:
        s = ss.event_log
        with event_slot:
            st.markdown("---")
            st.markdown("## Session story")
            if ss.active_baseline_key:
                st.caption(f"Baseline: **{ss.active_baseline_key}**")
            story_md = build_session_story(s, ss.samples)
            st.markdown(story_md)

            st.markdown("### Timeline of inflection points")
            st.caption(
                "Each bar is a window where one feature departed from your "
                "baseline. Colored by signal category — red = behavioural, "
                "amber = pose, grey = hardware artifact."
            )
            render_timeline(s)

            render_event_cards(s, ss.samples)

            with st.expander("Raw event list (debugging)"):
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
                for e in s["events"][:60]:
                    st.text(
                        f"{e['detector'].upper():3s} "
                        f"{e['sigma_level']}σ  "
                        f"t=[{e['start_t']:.2f}, {e['end_t']:.2f}]s  "
                        f"({e['duration_s']:.2f}s)  "
                        f"peak={e['peak_z']:.2f}σ  "
                        f"feature={e['dominant_feature']}"
                    )


if __name__ == "__main__":
    main()
