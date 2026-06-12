"""SyntoniaPro — live webcam dashboard (M-channel).

Tabbed single-page layout:

  * Live           — webcam, live metrics, status, one-line story
  * Insights       — narrative + interactive timeline + ranked unique
                     inflections (clip + transcript per inflection)
  * Technical      — raw detector counts and per-event list
  * How to read    — plain-language guide to signal categories +
                     detectors + feature glossary preview

Optional live audio capture + faster-whisper transcript alignment so
each behavioural inflection is shown with the words the subject was
actually speaking at that moment.
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

from audio_capture import (
    AudioRecorder,
    TranscriptSegment,
    ephemeral_wav_path,
    text_within,
    transcribe,
)
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

# Plain-language category labels and one-line definitions used everywhere
# in the dashboard. Single source of truth.
CATEGORY_LABEL = {
    "behavioural": "Real face change",
    "pose": "Head movement",
    "hardware": "Setup drift",
    "other": "Other",
}
CATEGORY_DESCRIPTION = {
    "behavioural": (
        "Your actual face moved — a brow raised, a mouth corner pulled, "
        "an asymmetric expression. The signal worth noticing."
    ),
    "pose": (
        "Your head moved — turned, nodded, tilted. Often natural reading "
        "or screen-scanning behaviour; sometimes meaningful."
    ),
    "hardware": (
        "Your face didn't change; the geometry did. Glasses settled "
        "differently, you moved a few centimetres from the camera, "
        "lighting shifted. Background noise, not behaviour."
    ),
    "other": "Uncategorised feature.",
}
CATEGORY_PALETTE = {
    "behavioural": "#d9534f",
    "pose": "#f0ad4e",
    "hardware": "#6c757d",
    "other": "#5bc0de",
}


# ─────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────

def load_signatures() -> Dict[str, Baseline]:
    if not SIGNATURE_FILE.exists():
        return {}
    raw = json.loads(SIGNATURE_FILE.read_text())
    return {name: Baseline.from_dict(payload) for name, payload in raw.items()}


def save_signatures(sigs: Dict[str, Baseline]) -> None:
    SIGNATURE_FILE.write_text(
        json.dumps({n: b.to_dict() for n, b in sigs.items()}, indent=2)
    )


# ─────────────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_camera(index: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(index)


@st.cache_resource
def get_detector() -> FaceTrapeziumDetector:
    return FaceTrapeziumDetector()


@st.cache_resource
def get_audio_recorder() -> AudioRecorder:
    return AudioRecorder()


# ─────────────────────────────────────────────────────────────────────────
# Frame helpers
# ─────────────────────────────────────────────────────────────────────────

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
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
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


def thumbnail_capture(frame: np.ndarray, sample_idx: int, every: int = 5) -> None:
    """Stash a small BGR frame every ``every`` samples (~0.5 s at 10 fps)."""
    if sample_idx % every != 0:
        return
    h, w = frame.shape[:2]
    target_w = 220
    if w > target_w:
        thumb = cv2.resize(frame, (target_w, int(h * target_w / w)))
    else:
        thumb = frame.copy()
    st.session_state.frame_thumbnails[sample_idx] = thumb
    if len(st.session_state.frame_thumbnails) > 400:
        st.session_state.frame_thumbnails.popitem(last=False)


def thumbnail_at_time(
    t: float, samples: List[TrapeziumSample]
) -> Optional[np.ndarray]:
    if not samples or not st.session_state.frame_thumbnails:
        return None
    sample_times = np.array([s.t for s in samples])
    idx = int(np.argmin(np.abs(sample_times - t)))
    keys = np.array(list(st.session_state.frame_thumbnails.keys()))
    nearest = int(keys[np.argmin(np.abs(keys - idx))])
    return st.session_state.frame_thumbnails[nearest]


# ─────────────────────────────────────────────────────────────────────────
# Session-state init
# ─────────────────────────────────────────────────────────────────────────

def init_state() -> None:
    ss = st.session_state
    ss.setdefault("samples", [])
    ss.setdefault("recording", False)
    ss.setdefault("start_t", None)
    ss.setdefault("signatures", load_signatures())
    ss.setdefault("event_log", None)
    ss.setdefault("transcript", [])
    ss.setdefault("frame_thumbnails", OrderedDict())
    ss.setdefault("active_baseline_key", None)
    ss.setdefault("selected_event_idx", 0)


# ─────────────────────────────────────────────────────────────────────────
# Event aggregation
# ─────────────────────────────────────────────────────────────────────────

def aggregate_unique_inflections(events: List[dict]) -> List[dict]:
    """Collapse all detector events down to one row per dominant feature
    with a weight = total breach time × peak σ. Used by the Insights tab.
    """
    by_feature: Dict[str, dict] = {}
    for e in events:
        f = e["dominant_feature"]
        row = by_feature.setdefault(
            f,
            {
                "feature": f,
                "category": category(f),
                "total_dur": 0.0,
                "peak_z": 0.0,
                "n_events": 0,
                "earliest_start": e["start_t"],
                "latest_end": e["end_t"],
                "best_event": e,
            },
        )
        row["total_dur"] += e["duration_s"]
        if e["peak_z"] > row["peak_z"]:
            row["peak_z"] = e["peak_z"]
            row["best_event"] = e
        row["n_events"] += 1
        row["earliest_start"] = min(row["earliest_start"], e["start_t"])
        row["latest_end"] = max(row["latest_end"], e["end_t"])

    rows = list(by_feature.values())
    total_dur_all = sum(r["total_dur"] for r in rows) or 1.0
    for r in rows:
        r["weight"] = r["total_dur"] * r["peak_z"]
    total_weight = sum(r["weight"] for r in rows) or 1.0
    for r in rows:
        r["weight_pct"] = 100.0 * r["weight"] / total_weight
        r["dur_pct"] = 100.0 * r["total_dur"] / total_dur_all
    rows.sort(key=lambda r: -r["weight"])
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────

def category_donut(rows: List[dict]) -> alt.Chart:
    if not rows:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()
    by_cat: Dict[str, float] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0.0) + r["weight"]
    df = pd.DataFrame(
        [
            {
                "category": CATEGORY_LABEL[c],
                "raw_cat": c,
                "weight": w,
            }
            for c, w in by_cat.items()
        ]
    )
    color_scale = alt.Scale(
        domain=[CATEGORY_LABEL[k] for k in CATEGORY_PALETTE],
        range=list(CATEGORY_PALETTE.values()),
    )
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=55, outerRadius=95)
        .encode(
            theta=alt.Theta("weight:Q", stack=True),
            color=alt.Color(
                "category:N", scale=color_scale, legend=alt.Legend(title=None)
            ),
            tooltip=["category", alt.Tooltip("weight:Q", format=".1f")],
        )
        .properties(height=220)
    )


def interactive_timeline(rows: List[dict], events: List[dict]) -> alt.Chart:
    if not events:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()
    df_events = pd.DataFrame(
        [
            {
                "feature_short": explain(e["dominant_feature"]).short,
                "feature": e["dominant_feature"],
                "category": CATEGORY_LABEL[category(e["dominant_feature"])],
                "start_t": e["start_t"],
                "end_t": e["end_t"],
                "duration_s": e["duration_s"],
                "peak_z": e["peak_z"],
                "detector": e["detector"].upper(),
                "sigma_level": e["sigma_level"],
            }
            for e in events
        ]
    )
    order = [explain(r["feature"]).short for r in rows]
    color_scale = alt.Scale(
        domain=[CATEGORY_LABEL[k] for k in CATEGORY_PALETTE],
        range=list(CATEGORY_PALETTE.values()),
    )
    chart = (
        alt.Chart(df_events)
        .mark_bar(opacity=0.85, cornerRadius=2)
        .encode(
            x=alt.X("start_t:Q", title="time (s)"),
            x2="end_t:Q",
            y=alt.Y("feature_short:N", sort=order, title=None),
            color=alt.Color(
                "category:N",
                scale=color_scale,
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("feature_short:N", title="What moved"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("detector:N", title="Detector"),
                alt.Tooltip("sigma_level:Q", title="σ level"),
                alt.Tooltip("start_t:Q", title="from (s)", format=".2f"),
                alt.Tooltip("end_t:Q", title="to (s)", format=".2f"),
                alt.Tooltip("duration_s:Q", title="lasted (s)", format=".2f"),
                alt.Tooltip("peak_z:Q", title="peak σ", format=".2f"),
            ],
        )
        .properties(height=22 * max(len(order), 1) + 60)
    )
    return chart


# ─────────────────────────────────────────────────────────────────────────
# Live tab
# ─────────────────────────────────────────────────────────────────────────

def status_caption(
    t2_peak: float, sigma_low: float, sigma_high: float
) -> str:
    if t2_peak >= sigma_high:
        return "🔴 **Breach** — your face is significantly off baseline."
    if t2_peak >= sigma_low:
        return "🟠 **Excursion** — something is shifting."
    return "🟢 **Stable** — you're within your baseline envelope."


def quick_story_line(summary: dict) -> str:
    if not summary:
        return ""
    events = summary.get("events", [])
    if not events:
        return "Stable take — nothing crossed σ-low."
    rows = aggregate_unique_inflections(events)
    if not rows:
        return ""
    top = rows[0]
    fx = explain(top["feature"])
    return (
        f"Top inflection: **{fx.short}** "
        f"({CATEGORY_LABEL[top['category']]}) — "
        f"{top['total_dur']:.1f} s above baseline, peak {top['peak_z']:.1f} σ "
        f"({top['weight_pct']:.0f}% of the session's weight)."
    )


def render_live_tab(
    baseline_choice: str,
    sigma_low: float,
    sigma_high: float,
    adapt_on: bool,
    adapt_window_s: float,
    camera_idx: int,
    run_live: bool,
) -> None:
    ss = st.session_state
    col_video, col_meta = st.columns([3, 2])
    video_slot = col_video.empty()

    if not run_live:
        video_slot.info("Live mode paused. Toggle ‘Run live’ in the sidebar.")
        return

    cap = get_camera(int(camera_idx))
    if not cap.isOpened():
        st.error(f"Could not open camera index {camera_idx}.")
        st.stop()
    detector = get_detector()

    with col_meta:
        metric_slot = st.empty()
        status_slot = st.empty()
        story_slot = st.empty()

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
        if w > 640:
            rgb = cv2.resize(rgb, (640, int(h * 640 / w)))
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
                c1, c2, c3 = st.columns(3)
                c1.metric("Hotelling T²", f"{s['max_t2_sigma']:.1f} σ")
                c2.metric("RMS |z|", f"{s['max_overall_z']:.1f} σ")
                c3.metric("CUSUM", s["n_cusum_events"])
                status_slot.markdown(
                    status_caption(s["max_t2_sigma"], sigma_low, sigma_high)
                )
            else:
                st.metric("Live samples", len(ss.samples))
                status_slot.info(
                    "Enroll a baseline and pick it under ‘Compare against’ "
                    "to see live metrics."
                )

        with story_slot.container():
            if ss.event_log:
                st.caption(quick_story_line(ss.event_log))

    live_tick()


# ─────────────────────────────────────────────────────────────────────────
# Insights tab — narrative + interactive timeline + ranked unique cards
# ─────────────────────────────────────────────────────────────────────────

def render_insights_tab() -> None:
    ss = st.session_state
    if not ss.event_log:
        st.info(
            "Run **Record → Stop → Detect** in the sidebar to populate "
            "the session insights."
        )
        return

    summary = ss.event_log
    events = summary.get("events", [])
    rows = aggregate_unique_inflections(events)

    # Headline numbers.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Take duration", f"{summary['duration_s']:.1f} s")
    c2.metric("Peak T²", f"{summary['max_t2_sigma']:.1f} σ")
    c3.metric("Unique features fired", len(rows))
    c4.metric("Behavioural rows", sum(1 for r in rows if r["category"] == "behavioural"))

    if ss.active_baseline_key:
        st.caption(f"Baseline: **{ss.active_baseline_key}**")

    # Narrative paragraph.
    if not events:
        st.success(
            "Stable take. No feature crossed σ-low. Your face stayed within "
            "the geometric envelope of your baseline."
        )
        return

    top = rows[0]
    fx_top = explain(top["feature"])
    st.markdown(
        f"#### The story of this take\n"
        f"Across **{summary['duration_s']:.1f} seconds**, the biggest "
        f"single signal came from **{fx_top.short}** — {fx_top.physical} "
        f"It carried **{top['weight_pct']:.0f}%** of the session's total "
        f"weight ({top['total_dur']:.1f} s above baseline, peak "
        f"{top['peak_z']:.1f} σ)."
    )

    # Two-column: donut + summary breakdown
    col_donut, col_breakdown = st.columns([1, 1])
    with col_donut:
        st.markdown("**Where the weight went**")
        st.altair_chart(category_donut(rows), use_container_width=True)
    with col_breakdown:
        st.markdown("**By category**")
        bucket = {"behavioural": 0.0, "pose": 0.0, "hardware": 0.0, "other": 0.0}
        for r in rows:
            bucket[r["category"]] += r["weight_pct"]
        for cat in ("behavioural", "pose", "hardware"):
            w = bucket[cat]
            st.markdown(
                f"<div style='border-left:5px solid {CATEGORY_PALETTE[cat]};"
                f"padding:6px 10px;margin-bottom:8px;background:#0e1117'>"
                f"<b>{CATEGORY_LABEL[cat]}</b> &nbsp; "
                f"<span style='color:{CATEGORY_PALETTE[cat]};font-weight:bold'>"
                f"{w:.0f}%</span>"
                f"<div style='font-size:0.85rem;color:#9aa0a6'>"
                f"{CATEGORY_DESCRIPTION[cat]}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Interactive timeline.
    st.markdown("#### Timeline — click a bar to see the moment")
    chart_event = st.altair_chart(
        interactive_timeline(rows, events),
        use_container_width=True,
        on_select="rerun",
        selection_mode="point",
        key="timeline_chart",
    )

    # Selected moment detail (clip + transcript + plain language).
    selected_t: Optional[float] = None
    selected_feature: Optional[str] = None
    if chart_event and chart_event.selection.get("param_1"):
        # Streamlit returns selected points under a generated key; iterate to be safe.
        for k, v in chart_event.selection.items():
            if v:
                sel = v[0]
                selected_t = 0.5 * (sel["start_t"] + sel["end_t"])
                # The Altair tooltip exposes the feature short name; we need
                # the technical name. Map by short name.
                short = sel.get("feature_short")
                for r in rows:
                    if explain(r["feature"]).short == short:
                        selected_feature = r["feature"]
                        break
                break

    if selected_feature is None and rows:
        selected_feature = rows[0]["feature"]
        selected_t = 0.5 * (rows[0]["best_event"]["start_t"] + rows[0]["best_event"]["end_t"])

    if selected_feature is not None and selected_t is not None:
        _render_moment(selected_feature, selected_t)

    # Ranked unique inflections.
    st.markdown("#### Ranked unique inflection points")
    st.caption(
        "One row per distinct feature, sorted by impact (duration × peak σ). "
        "No duplicates: each feature appears once with its peak frame."
    )
    for i, r in enumerate(rows[:8]):
        _render_inflection_row(i + 1, r)


def _render_moment(feature: str, t: float) -> None:
    ss = st.session_state
    fx = explain(feature)
    cat = category(feature)
    color = CATEGORY_PALETTE[cat]
    label = CATEGORY_LABEL[cat]

    st.markdown(
        f"<div style='border:1px solid {color};border-radius:6px;"
        f"padding:14px;margin-top:6px;background:#0e1117'>"
        f"<div style='display:inline-block;background:{color};color:white;"
        f"padding:2px 10px;border-radius:4px;font-size:0.8rem'>{label}</div>"
        f"&nbsp;&nbsp;<b>{fx.short}</b> at t ≈ {t:.2f} s",
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 2])
    with cols[0]:
        thumb = thumbnail_at_time(t, ss.samples)
        if thumb is not None:
            st.image(
                cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
                caption=f"~t = {t:.2f}s",
            )
        else:
            st.caption("(no frame thumbnail at this time)")
    with cols[1]:
        # Transcript quote
        spoken = text_within(ss.transcript, t - 1.5, t + 1.5)
        if spoken:
            st.markdown(
                f"> **What you said:** “{spoken}”"
            )
        elif not ss.transcript:
            recorder = get_audio_recorder()
            if not recorder.available:
                st.caption(
                    "_Install_ `sounddevice` _to capture audio and align "
                    "transcripts to inflections._"
                )
            else:
                st.caption(
                    "_No transcript available — was the mic on, and did you "
                    "give Terminal microphone permission?_"
                )
        else:
            st.caption("_(silence around this moment)_")
        st.markdown(f"**Geometric meaning.** {fx.physical}")
        st.markdown("**Compatible with any of:**")
        for c in fx.candidates:
            st.markdown(f"- {c}")
        if fx.citation:
            st.markdown(f"**Cited science.** {fx.citation}")
        st.markdown(f"_{fx.socratic}_")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_inflection_row(rank: int, r: dict) -> None:
    fx = explain(r["feature"])
    cat = r["category"]
    color = CATEGORY_PALETTE[cat]
    label = CATEGORY_LABEL[cat]
    cols = st.columns([1, 6, 2])
    with cols[0]:
        st.markdown(
            f"<div style='font-size:1.8rem;font-weight:bold;color:{color};"
            f"text-align:center'>#{rank}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"**{fx.short}** &nbsp;"
            f"<span style='background:{color};color:white;padding:1px 8px;"
            f"border-radius:4px;font-size:0.75rem'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"first at {r['earliest_start']:.2f}s · "
            f"latest at {r['latest_end']:.2f}s · "
            f"{r['n_events']} window(s) · peak {r['peak_z']:.1f} σ"
        )
    with cols[2]:
        st.markdown(
            f"<div style='text-align:right'>"
            f"<div style='font-size:1.4rem;font-weight:bold;color:{color}'>"
            f"{r['weight_pct']:.0f}%</div>"
            f"<div style='font-size:0.75rem;color:#9aa0a6'>"
            f"of session weight</div></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# Technical tab
# ─────────────────────────────────────────────────────────────────────────

def render_technical_tab() -> None:
    ss = st.session_state
    if not ss.event_log:
        st.info("No detect-run results yet. Use the sidebar to run a detection.")
        return
    s = ss.event_log
    st.markdown("### Detector counts")
    st.write(
        f"frames={s['n_samples']}  ·  duration={s['duration_s']:.2f}s  ·  "
        f"max RMS|z|={s['max_overall_z']:.2f}σ  ·  "
        f"max T²σ={s['max_t2_sigma']:.2f}"
    )
    st.write(
        f"RMS events: {s['n_rms_events_3sigma']}×3σ, "
        f"{s['n_rms_events_6sigma']}×6σ  ·  "
        f"T² events: {s['n_t2_events_3sigma']}×3σ, "
        f"{s['n_t2_events_6sigma']}×6σ  ·  "
        f"CUSUM events: {s['n_cusum_events']}"
    )
    st.markdown("### Raw event list")
    df = pd.DataFrame(s["events"])
    if not df.empty:
        df = df[
            [
                "detector",
                "sigma_level",
                "start_t",
                "end_t",
                "duration_s",
                "peak_z",
                "dominant_feature",
            ]
        ]
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("### Baseline metadata")
    if ss.active_baseline_key:
        st.write(f"Active baseline: **{ss.active_baseline_key}**")
    name = (ss.active_baseline_key or "").replace(" (adapted)", "")
    base = ss.signatures.get(name)
    if base is not None:
        st.write(
            f"n_samples={base.n_samples}  ·  "
            f"duration_s={base.duration_s:.2f}  ·  "
            f"feature_count={len(base.feature_names)}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Guide tab — plain-language read-this-once
# ─────────────────────────────────────────────────────────────────────────

def render_guide_tab() -> None:
    st.markdown(
        """
        ### How to read this dashboard

        Your webcam feeds the **M-channel** of the Syntonia model — the
        moment-by-moment geometry of your face. Three statistical detectors
        watch for departures from a baseline you enrolled while at rest.
        Nothing leaves this device.
        """
    )
    st.markdown("#### The three categories of signal")
    for cat in ("behavioural", "pose", "hardware"):
        st.markdown(
            f"<div style='border-left:6px solid {CATEGORY_PALETTE[cat]};"
            f"padding:10px 14px;margin:10px 0;background:#0e1117'>"
            f"<b style='color:{CATEGORY_PALETTE[cat]}'>{CATEGORY_LABEL[cat]}</b>"
            f"<div style='font-size:0.95rem;margin-top:4px'>"
            f"{CATEGORY_DESCRIPTION[cat]}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        #### The three detectors

        - **Hotelling T²** — measures the whole face at once. Catches
          coordinated multi-feature moves (a smile recruits cheek + mouth +
          eye crinkle).
        - **RMS |z|** — catches single-feature blow-ups (one feature wildly
          off baseline).
        - **CUSUM** — catches *sustained* drift even when each frame's
          deviation is small.

        A "breach" is when T² goes past your σ-high threshold (default 6 σ).
        An "excursion" is between σ-low and σ-high.

        #### Why per-subject + per-language baseline matters

        Western-population baselines smooth across millions of faces and
        miss what makes yours yours. A baseline enrolled on you, in the
        language you're about to speak, is sharp enough to surface a
        sub-second brow raise — and to **not** mis-flag a Spanish open
        vowel as deception.
        """
    )

    st.markdown("#### Feature glossary (preview)")
    rows = []
    from feature_glossary import _GLOSSARY
    for name, fx in _GLOSSARY.items():
        rows.append(
            {
                "feature": name,
                "what it means": fx.short,
                "category": CATEGORY_LABEL[category(name)],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    ss = st.session_state
    recorder = get_audio_recorder()
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
        adapt_on = st.checkbox("Adaptive baseline", value=True)
        adapt_window_s = st.slider(
            "Adapt-from window (s)",
            2.0,
            15.0,
            5.0,
            0.5,
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
            now = time.time()
            if not ss.recording:
                # Starting
                if not ss.samples:
                    ss.start_t = now
                ss.recording = True
                if recorder.available:
                    recorder.start(now)
            else:
                # Stopping
                ss.recording = False
                if recorder.available and recorder.recording:
                    recorder.stop()
        if c2.button("Clear", use_container_width=True):
            ss.samples = []
            ss.start_t = None
            ss.event_log = None
            ss.transcript = []
            ss.frame_thumbnails = OrderedDict()
            ss.active_baseline_key = None
            if recorder.available:
                recorder.clear()

        st.caption(f"Samples collected: **{len(ss.samples)}**")
        if ss.samples:
            dur = ss.samples[-1].t - ss.samples[0].t
            st.caption(f"Duration: **{dur:.2f} s**")
        if recorder.available:
            st.caption(
                f"Audio: {'🔴 recording' if recorder.recording else '⚪ idle'}  "
                f"· {recorder.duration_s:.1f} s buffered"
            )
        else:
            st.caption("Audio: _sounddevice not installed_")

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
                    f"Enrolled '{new_name.strip()}' — "
                    f"n={base.n_samples}, "
                    f"duration={base.duration_s:.2f}s"
                )

        st.header("Detect")
        do_transcribe = st.checkbox(
            "Transcribe audio after detect",
            value=recorder.available,
            disabled=not recorder.available,
            help=(
                "Runs faster-whisper locally on the buffered audio so each "
                "inflection card can show what you actually said. ~10 s for "
                "a 30 s take on M4."
            ),
        )
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
                ss.selected_event_idx = 0

                if do_transcribe and recorder.available:
                    with st.spinner("Transcribing audio locally…"):
                        wav_path = ephemeral_wav_path()
                        try:
                            saved = recorder.save_wav(wav_path)
                            if saved:
                                ss.transcript = transcribe(wav_path)
                                st.success(
                                    f"Transcript ready ({len(ss.transcript)} segments)."
                                )
                            else:
                                ss.transcript = []
                                st.caption(
                                    "No audio buffered — recording may not "
                                    "have captured mic input."
                                )
                        finally:
                            try:
                                wav_path.unlink()
                            except Exception:
                                pass

        run_live = st.toggle("Run live", value=True)

    return {
        "camera_idx": camera_idx,
        "baseline_choice": baseline_choice,
        "adapt_on": adapt_on,
        "adapt_window_s": adapt_window_s,
        "sigma_low": sigma_low,
        "sigma_high": sigma_high,
        "run_live": run_live,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="SyntoniaPro — Live", layout="wide")
    init_state()
    sidebar = render_sidebar()

    st.title("SyntoniaPro — Live")
    st.caption(
        "Local webcam → face geometry → behavioural insights. "
        "No cloud, no data leaves this device."
    )

    tab_live, tab_insights, tab_tech, tab_guide = st.tabs(
        ["📹 Live", "🎯 Insights", "🔬 Technical", "❓ Guide"]
    )

    with tab_live:
        render_live_tab(
            baseline_choice=sidebar["baseline_choice"],
            sigma_low=sidebar["sigma_low"],
            sigma_high=sidebar["sigma_high"],
            adapt_on=sidebar["adapt_on"],
            adapt_window_s=sidebar["adapt_window_s"],
            camera_idx=sidebar["camera_idx"],
            run_live=sidebar["run_live"],
        )
    with tab_insights:
        render_insights_tab()
    with tab_tech:
        render_technical_tab()
    with tab_guide:
        render_guide_tab()


if __name__ == "__main__":
    main()
