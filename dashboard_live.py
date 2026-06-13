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

# Visual descriptors for each feature: (pictograph, side-asymmetry label).
# All face-region features use the neutral bust-silhouette (👤) — no race,
# no gender, no expression baked in. Pose and geometric features use
# directional / mathematical glyphs. The side label below the glyph carries
# the directional asymmetry; the four-pip bar carries severity.
FEATURE_GLYPH: Dict[str, tuple] = {
    # Face-behavioural
    "left_brow_height_norm":  ("👤", "◀ LEFT brow"),
    "right_brow_height_norm": ("👤", "RIGHT brow ▶"),
    "angle_LM":               ("👤", "◀ LEFT mouth corner"),
    "angle_RM":               ("👤", "RIGHT mouth corner ▶"),
    "mouth_offset_norm":      ("👤", "mouth off-centre"),
    "mouth_line_norm":        ("👤", "mouth width"),
    "side_mouth_norm":        ("👤", "mouth width"),
    # Trapezium-side and eye geometry
    "side_left_norm":         ("👤", "◀ LEFT side"),
    "side_right_norm":        ("👤", "RIGHT side ▶"),
    "side_eye_norm":          ("👓", "eye line / glasses"),
    "eye_line_norm":          ("👓", "eye line / glasses"),
    "angle_LE":               ("👤", "◀ LEFT eye corner"),
    "angle_RE":               ("👤", "RIGHT eye corner ▶"),
    # Pose (head pose proxies)
    "yaw_proxy":              ("🔄", "◀ head turn ▶"),
    "pitch_proxy":            ("↕", "▲ chin ▼"),
    "roll_proxy":             ("⤵", "head tilt"),
    # Geometric (no face)
    "diag_LE_RM_norm":        ("⟍", "LE → RM diagonal"),
    "diag_RE_LM_norm":        ("⟋", "RE → LM diagonal"),
    "diag_ratio":             ("⚖", "diagonal balance"),
    "eye_mouth_ratio":        ("📏", "vertical proportion"),
    "parallelism_residual":   ("⊥", "non-parallel"),
}


def severity_word(peak_z: float) -> str:
    """Plain-language severity label — no numbers."""
    if peak_z >= 12:
        return "intense"
    if peak_z >= 6:
        return "marked"
    if peak_z >= 3:
        return "noticeable"
    return "subtle"


def severity_bar_html(peak_z: float, color: str) -> str:
    """4-pip horizontal severity bar, filled in proportion to peak σ."""
    levels = 4
    filled = max(1, min(levels, int(peak_z // 3)))
    pips = []
    for i in range(levels):
        c = color if i < filled else "#3a3f4a"
        pips.append(
            f"<span style='display:inline-block;width:18px;height:6px;"
            f"background:{c};margin:0 1px;border-radius:2px'></span>"
        )
    return "".join(pips)


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


def thumbnail_capture(frame: np.ndarray, sample_idx: int, every: int = 3) -> None:
    """Stash a BGR frame every ``every`` samples (~0.3 s at 10 fps).
    Stored as a (frame, scale_to_source) tuple so face-crop helpers can
    translate landmark coordinates from the source frame into the
    downsampled thumbnail. Target width chosen to keep the cropped
    face-region readable at full mobile-tab width.
    """
    if sample_idx % every != 0:
        return
    h, w = frame.shape[:2]
    target_w = 960
    if w > target_w:
        scale = target_w / w
        thumb = cv2.resize(
            frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        thumb = frame.copy()
        scale = 1.0
    st.session_state.frame_thumbnails[sample_idx] = (thumb, scale)
    if len(st.session_state.frame_thumbnails) > 400:
        st.session_state.frame_thumbnails.popitem(last=False)


def _nearest_thumb_record(t: float, samples: List[TrapeziumSample]):
    """Return (sample_idx, (thumb, scale)) closest to time ``t``."""
    if not samples or not st.session_state.frame_thumbnails:
        return None
    sample_times = np.array([s.t for s in samples])
    idx = int(np.argmin(np.abs(sample_times - t)))
    keys = np.array(list(st.session_state.frame_thumbnails.keys()))
    nearest = int(keys[np.argmin(np.abs(keys - idx))])
    return nearest, st.session_state.frame_thumbnails[nearest]


def thumbnail_at_time(
    t: float, samples: List[TrapeziumSample]
) -> Optional[np.ndarray]:
    rec = _nearest_thumb_record(t, samples)
    if rec is None:
        return None
    payload = rec[1]
    # Backward compatibility: older entries may be bare ndarrays.
    if isinstance(payload, tuple):
        return payload[0]
    return payload


def _face_crop_box(
    sample: TrapeziumSample, scale: float, thumb_shape: tuple
) -> Optional[tuple]:
    """Compute a generous face-region crop box in thumbnail coordinates.

    Uses the six trapezium / brow landmarks as the geometric anchor and
    pads up for hair, down for chin / neck, sides for ears so the
    resulting headshot reads as a portrait crop with no room background.
    """
    th, tw = thumb_shape[:2]
    pts = [
        sample.left_eye[:2],
        sample.right_eye[:2],
        sample.left_mouth[:2],
        sample.right_mouth[:2],
        sample.left_brow[:2],
        sample.right_brow[:2],
    ]
    xs = [p[0] * scale for p in pts]
    ys = [p[1] * scale for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    # Crop to head + shoulders rather than face only. The trapezium box
    # is small (eye corners + mouth corners + brows); to capture the head
    # top and the chest line we pad heavily — laterally for shoulders,
    # vertically for hair on top and the chest below.
    pad_x = w * 1.8
    pad_y_top = h * 1.8
    pad_y_bot = h * 3.5
    left = int(max(0, x_min - pad_x))
    top = int(max(0, y_min - pad_y_top))
    right = int(min(tw, x_max + pad_x))
    bottom = int(min(th, y_max + pad_y_bot))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_to_face_at_time(
    t: float, samples: List[TrapeziumSample]
) -> Optional[np.ndarray]:
    """Return a tight face-crop of the thumbnail captured closest to ``t``."""
    rec = _nearest_thumb_record(t, samples)
    if rec is None:
        return None
    sample_idx, payload = rec
    if isinstance(payload, tuple):
        thumb, scale = payload
    else:
        thumb, scale = payload, 1.0
    if sample_idx >= len(samples):
        return thumb
    sample = samples[sample_idx]
    box = _face_crop_box(sample, scale, thumb.shape)
    if box is None:
        return thumb
    left, top, right, bottom = box
    return thumb[top:bottom, left:right]


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
    ss.setdefault("avatar_bgr", None)


def crop_to_square(img: np.ndarray, size: int = 128) -> np.ndarray:
    """Center-crop to square, then resize to ``size`` × ``size``."""
    h, w = img.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    sq = img[top:top + s, left:left + s]
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def avatar_html(avatar_bgr: Optional[np.ndarray], size: int = 84) -> str:
    """Encode a BGR ndarray as an inline base64 PNG <img> tag, rounded."""
    if avatar_bgr is None:
        return ""
    ok, buf = cv2.imencode(".png", avatar_bgr)
    if not ok:
        return ""
    import base64
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return (
        f"<img src='data:image/png;base64,{b64}' "
        f"style='width:{size}px;height:{size}px;border-radius:50%;"
        f"object-fit:cover;display:block;margin:0 auto;"
        f"border:2px solid #2a2f3a'/>"
    )


def _avatar_src(avatar_bgr: Optional[np.ndarray]) -> Optional[str]:
    if avatar_bgr is None:
        return None
    ok, buf = cv2.imencode(".png", avatar_bgr)
    if not ok:
        return None
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def baseline_thumbnail() -> Optional[np.ndarray]:
    """Return the earliest thumbnail in the current take as a proxy for
    'baseline body posture'. The very first frames of a recording are when
    the subject is still settling — closest available stand-in for at-rest.
    """
    thumbs = st.session_state.frame_thumbnails
    if not thumbs:
        return None
    first_key = next(iter(thumbs))
    payload = thumbs[first_key]
    if isinstance(payload, tuple):
        return payload[0]
    return payload


def baseline_face_crop() -> Optional[np.ndarray]:
    """Same idea as baseline_thumbnail(), but cropped tight to the face."""
    thumbs = st.session_state.frame_thumbnails
    samples = st.session_state.samples
    if not thumbs or not samples:
        return baseline_thumbnail()
    first_key = next(iter(thumbs))
    payload = thumbs[first_key]
    if isinstance(payload, tuple):
        thumb, scale = payload
    else:
        thumb, scale = payload, 1.0
    if first_key >= len(samples):
        return thumb
    box = _face_crop_box(samples[first_key], scale, thumb.shape)
    if box is None:
        return thumb
    left, top, right, bottom = box
    return thumb[top:bottom, left:right]


def twin_avatar_block_html(
    inflection_peak_t: float,
    samples: List[TrapeziumSample],
    peak_z: float,
    color: str,
    fallback_avatar_bgr: Optional[np.ndarray],
    size: int = 80,
) -> str:
    """Twin display for face-region inflections, comparing real captured
    body language from the take itself:

      Left  — BASELINE: the earliest frame in the recording (your settled
              starting posture), de-saturated, green ring.
      Right — NOW:      the frame closest to the inflection peak — your
              actual body language at that moment, with a severity-coloured
              ring + glow whose thickness scales with σ.

    If the take has no captured thumbnails yet, fall back to the uploaded
    avatar for both panels (or 👤 if no avatar is set either).
    """
    baseline_ring = "#5cb85c"
    border_w = min(6, max(2, int(peak_z // 3)))

    baseline_frame = baseline_thumbnail()
    now_frame = thumbnail_at_time(inflection_peak_t, samples)

    if baseline_frame is None:
        baseline_frame = fallback_avatar_bgr
    if now_frame is None:
        now_frame = fallback_avatar_bgr

    baseline_src = _avatar_src(baseline_frame)
    now_src = _avatar_src(now_frame)

    if baseline_src is not None:
        baseline_face = (
            f"<img src='{baseline_src}' style='width:{size}px;height:{size}px;"
            f"border-radius:14px;object-fit:cover;"
            f"border:3px solid {baseline_ring};opacity:0.65;"
            f"filter:saturate(0.5)'/>"
        )
    else:
        baseline_face = (
            f"<div style='width:{size}px;height:{size}px;border-radius:14px;"
            f"border:3px solid {baseline_ring};display:flex;align-items:center;"
            f"justify-content:center;font-size:{int(size * 0.6)}px;"
            f"background:#0e1117;opacity:0.65;margin:0 auto'>👤</div>"
        )

    if now_src is not None:
        actual_face = (
            f"<img src='{now_src}' style='width:{size}px;height:{size}px;"
            f"border-radius:14px;object-fit:cover;"
            f"border:{border_w}px solid {color};"
            f"box-shadow:0 0 10px {color}'/>"
        )
    else:
        actual_face = (
            f"<div style='width:{size}px;height:{size}px;border-radius:14px;"
            f"border:{border_w}px solid {color};display:flex;align-items:center;"
            f"justify-content:center;font-size:{int(size * 0.6)}px;"
            f"background:#0e1117;box-shadow:0 0 10px {color};margin:0 auto'>"
            f"👤</div>"
        )

    return (
        f"<div style='display:flex;justify-content:center;gap:6px;"
        f"align-items:center'>"
        f"<div style='text-align:center'>"
        f"{baseline_face}"
        f"<div style='font-size:0.6rem;color:#9aa0a6;margin-top:4px;"
        f"letter-spacing:1.5px'>BASELINE</div>"
        f"</div>"
        f"<div style='color:{color};font-size:1.2rem;font-weight:bold;"
        f"padding:0 2px'>→</div>"
        f"<div style='text-align:center'>"
        f"{actual_face}"
        f"<div style='font-size:0.6rem;color:{color};margin-top:4px;"
        f"letter-spacing:1.5px;font-weight:bold'>NOW</div>"
        f"</div>"
        f"</div>"
    )


# Feature names whose icon should swap for the twin-avatar block on a row.
_FACE_FEATURES = {
    "left_brow_height_norm",
    "right_brow_height_norm",
    "angle_LM",
    "angle_RM",
    "mouth_offset_norm",
    "mouth_line_norm",
    "side_mouth_norm",
    "side_left_norm",
    "side_right_norm",
    "angle_LE",
    "angle_RE",
}


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
    # Explicit selection so Streamlit can wire on_select="rerun" to it.
    pt = alt.selection_point(name="picked", fields=["feature_short"], on="click", empty=False)
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
            opacity=alt.condition(pt, alt.value(1.0), alt.value(0.55)),
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
        .add_params(pt)
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
            # Capture the RAW frame for thumbnails BEFORE drawing the
            # trapezium overlay, so BASELINE / NOW close-ups in the
            # Mobile and Insights tabs show clean photographs.
            if ss.recording:
                sample_idx = len(ss.samples)
                ss.samples.append(sample)
                thumbnail_capture(frame.copy(), sample_idx)
            draw_trapezium(frame, sample, ss.recording)
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

def _img_as_inline_html(
    bgr: Optional[np.ndarray],
    border_color: Optional[str] = None,
    border_w: int = 0,
    radius: int = 14,
) -> str:
    """Encode a BGR array as an inline base64 PNG <img> at full container
    width. Used by the Mobile tab so the cropped headshot fills the screen.
    """
    if bgr is None:
        return ""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    import base64
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    border_style = (
        f"border:{border_w}px solid {border_color};" if border_color else ""
    )
    return (
        f"<img src='data:image/png;base64,{b64}' "
        f"style='width:100%;height:auto;border-radius:{radius}px;"
        f"{border_style}display:block'/>"
    )


def render_mobile_tab() -> None:
    """iPhone 14 Pro–oriented vertical inflection feed with pagination.

    One inflection visible at a time, navigated by Prev / Next chevron
    buttons (the "swipe between inflections" trial). Each card stacks
    vertically:
       1) Tight face-crop close-up at full screen width.
       2) Title + plain-language meaning + cited science + Socratic question
          + (transcript) + timestamps.
       3) Larger twin baseline-vs-now panels.
    """
    ss = st.session_state
    ss.setdefault("mobile_idx", 0)
    if not ss.event_log:
        st.info(
            "Run **Record → Stop → Detect** in the sidebar to populate "
            "the mobile feed."
        )
        return
    summary = ss.event_log
    events = summary.get("events", [])
    rows = aggregate_unique_inflections(events)[:8]
    if not events or not rows:
        st.success(
            "Stable take. No feature crossed σ-low. Your face stayed within "
            "the geometric envelope of your baseline."
        )
        return
    if ss.active_baseline_key:
        st.caption(f"Baseline: **{ss.active_baseline_key}**")

    total = len(rows)
    ss.mobile_idx = max(0, min(ss.mobile_idx, total - 1))

    pad_l, content, pad_r = st.columns([1, 6, 1])
    with content:
        # Swipe-style pagination: ◀ chevron · "N of T" · chevron ▶
        nav_prev, nav_label, nav_next = st.columns([1, 4, 1])
        with nav_prev:
            if st.button(
                "◀",
                use_container_width=True,
                disabled=ss.mobile_idx <= 0,
                key="mobile_prev",
            ):
                ss.mobile_idx -= 1
                st.rerun()
        with nav_label:
            st.markdown(
                f"<div style='text-align:center;font-weight:bold;"
                f"font-size:1rem;padding-top:6px;color:#0a0a0a;"
                f"letter-spacing:1px'>"
                f"INFLECTION {ss.mobile_idx + 1} OF {total}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            if st.button(
                "▶",
                use_container_width=True,
                disabled=ss.mobile_idx >= total - 1,
                key="mobile_next",
            ):
                ss.mobile_idx += 1
                st.rerun()
        # Compact dot-pagination row beneath the chevrons.
        dot_html = " ".join(
            (
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"border-radius:50%;margin:0 3px;background:"
                f"{CATEGORY_PALETTE[rows[i]['category']] if i == ss.mobile_idx else '#3a3f4a'}'></span>"
            )
            for i in range(total)
        )
        st.markdown(
            f"<div style='text-align:center;margin:6px 0 14px 0'>{dot_html}</div>",
            unsafe_allow_html=True,
        )
        _render_mobile_card(ss.mobile_idx + 1, rows[ss.mobile_idx])


def _render_mobile_card(rank: int, r: dict) -> None:
    ss = st.session_state
    fx = explain(r["feature"])
    cat = r["category"]
    color = CATEGORY_PALETTE[cat]
    label = CATEGORY_LABEL[cat]
    best = r["best_event"]
    peak_t = 0.5 * (best["start_t"] + best["end_t"])

    # Title row.
    st.markdown(
        f"<div style='display:flex;align-items:center;"
        f"justify-content:space-between;color:#0a0a0a'>"
        f"<div style='font-size:1.1rem;font-weight:bold;color:#0a0a0a'>"
        f"#{rank} · {fx.short}</div>"
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:4px;font-size:0.78rem;font-weight:bold'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    # Body copy — dark text for light-theme readability.
    st.markdown(
        f"<div style='color:#1a1a1a;font-size:1.0rem;margin-top:10px;"
        f"line-height:1.45'>"
        f"<b style='color:#0a0a0a'>What it measures.</b> {fx.physical}</div>",
        unsafe_allow_html=True,
    )
    if fx.citation:
        st.markdown(
            f"<div style='color:#2c2c2c;font-size:0.92rem;margin-top:6px;"
            f"line-height:1.45'>"
            f"<b style='color:#0a0a0a'>Cited science.</b> {fx.citation}</div>",
            unsafe_allow_html=True,
        )
    if fx.socratic:
        st.markdown(
            f"<div style='color:{color};font-size:1.0rem;font-style:italic;"
            f"margin-top:10px;font-weight:600'>{fx.socratic}</div>",
            unsafe_allow_html=True,
        )
    if ss.transcript:
        spoken = text_within(ss.transcript, best["start_t"], best["end_t"])
        if spoken:
            st.markdown(
                f"<div style='color:#0a1a2e;font-size:0.95rem;margin-top:10px;"
                f"padding:10px 14px;background:#eaf3ff;"
                f"border-left:4px solid #1559b8;border-radius:4px;"
                f"font-style:italic'>"
                f"<span style='color:#1559b8;font-weight:bold;font-style:normal'>"
                f"What you said.</span> “{spoken}”</div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        f"<div style='color:#404040;font-size:0.82rem;margin-top:10px'>"
        f"first at {r['earliest_start']:.2f} s · "
        f"latest at {r['latest_end']:.2f} s · "
        f"{r['n_events']} window(s)</div>",
        unsafe_allow_html=True,
    )

    # Section 3 (was Section 3 of three; first close-up was dropped).
    st.markdown(
        "<div style='margin-top:24px;font-size:0.95rem;color:#0a0a0a;"
        "letter-spacing:2px;text-transform:uppercase;text-align:center;"
        "font-weight:bold'>"
        "Baseline vs. now</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin-top:8px'></div>", unsafe_allow_html=True
    )
    twin_html = _mobile_twin_html(r, peak_t, color)
    st.markdown(twin_html, unsafe_allow_html=True)
    st.markdown(
        "<hr style='margin:24px 0;border:none;border-top:1px solid #2a2f3a'>",
        unsafe_allow_html=True,
    )


def _mobile_twin_html(r: dict, peak_t: float, color: str) -> str:
    """Twin baseline-vs-now panels rendered larger and side-by-side, sized
    to fit a phone column."""
    ss = st.session_state
    is_face = r["feature"] in _FACE_FEATURES
    if is_face:
        baseline_face = baseline_face_crop()
        now_face = crop_to_face_at_time(peak_t, ss.samples)
    else:
        baseline_face = baseline_thumbnail()
        now_face = thumbnail_at_time(peak_t, ss.samples)

    if baseline_face is None:
        baseline_face = ss.avatar_bgr
    if now_face is None:
        now_face = ss.avatar_bgr

    baseline_ring = "#5cb85c"
    border_w = min(6, max(2, int(r["peak_z"] // 3)))

    def panel(bgr, ring, ring_w, label, label_color, dim):
        # `dim` only changes the RING colour + the glow — the image itself
        # always renders at full crispness so BASELINE and NOW are visually
        # the same quality. Distinguishing the two is the ring's job.
        if bgr is not None:
            ok, buf = cv2.imencode(".png", bgr)
            if ok:
                import base64
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                src = f"data:image/png;base64,{b64}"
                glow = "" if dim else f"box-shadow:0 0 14px {ring};"
                img_html = (
                    f"<img src='{src}' style='width:100%;height:auto;"
                    f"border-radius:14px;object-fit:cover;"
                    f"border:{ring_w}px solid {ring};{glow}'/>"
                )
            else:
                img_html = "<div>👤</div>"
        else:
            img_html = (
                f"<div style='aspect-ratio:1/1;border-radius:14px;"
                f"border:{ring_w}px solid {ring};display:flex;"
                f"align-items:center;justify-content:center;font-size:56px;"
                f"background:#0e1117'>👤</div>"
            )
        return (
            f"<div style='flex:1;text-align:center'>"
            f"{img_html}"
            f"<div style='font-size:0.82rem;color:{label_color};"
            f"margin-top:8px;letter-spacing:1.5px;font-weight:bold'>"
            f"{label}</div></div>"
        )

    return (
        f"<div style='display:flex;gap:16px;align-items:center;"
        f"max-width:100%;margin:0 auto'>"
        f"{panel(baseline_face, baseline_ring, 4, 'BASELINE', baseline_ring, dim=True)}"
        f"<div style='color:{color};font-size:2.2rem;font-weight:bold;"
        f"line-height:1'>→</div>"
        f"{panel(now_face, color, border_w + 2, 'NOW', color, dim=False)}"
        f"</div>"
    )


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

    if ss.active_baseline_key:
        st.caption(f"Baseline: **{ss.active_baseline_key}**")

    if not events:
        st.success(
            "Stable take. No feature crossed σ-low. Your face stayed within "
            "the geometric envelope of your baseline."
        )
        return

    st.markdown("### Ranked unique inflection points")
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
    ss = st.session_state
    fx = explain(r["feature"])
    cat = r["category"]
    color = CATEGORY_PALETTE[cat]
    label = CATEGORY_LABEL[cat]
    best = r["best_event"]
    peak_t = 0.5 * (best["start_t"] + best["end_t"])
    thumb = thumbnail_at_time(peak_t, ss.samples)

    cols = st.columns([1, 3, 7, 2])
    with cols[0]:
        st.markdown(
            f"<div style='font-size:1.8rem;font-weight:bold;color:{color};"
            f"text-align:center;padding-top:14px'>#{rank}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        if thumb is not None:
            st.image(
                cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
                caption=f"close-up · t ≈ {peak_t:.2f} s",
                use_container_width=True,
            )
        else:
            st.caption("(no close-up frame captured for this moment)")
    with cols[2]:
        st.markdown(
            f"<span style='color:#0a0a0a;font-weight:bold'>{fx.short}</span>"
            f" &nbsp;"
            f"<span style='background:{color};color:white;padding:1px 8px;"
            f"border-radius:4px;font-size:0.75rem'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#1a1a1a;font-size:0.95rem;margin-top:4px;"
            f"line-height:1.4'>"
            f"<b style='color:#0a0a0a'>What it measures.</b> {fx.physical}</div>",
            unsafe_allow_html=True,
        )
        if fx.citation:
            st.markdown(
                f"<div style='color:#2c2c2c;font-size:0.9rem;margin-top:3px;"
                f"line-height:1.4'>"
                f"<b style='color:#0a0a0a'>Cited science.</b> {fx.citation}</div>",
                unsafe_allow_html=True,
            )
        if fx.socratic:
            st.markdown(
                f"<div style='color:{color};font-size:0.95rem;font-style:italic;"
                f"margin-top:6px;font-weight:600'>"
                f"{fx.socratic}</div>",
                unsafe_allow_html=True,
            )
        if ss.transcript:
            spoken = text_within(ss.transcript, best["start_t"], best["end_t"])
            if spoken:
                st.markdown(
                    f"<div style='color:#0a1a2e;font-size:0.9rem;margin-top:6px;"
                    f"padding:6px 10px;background:#eaf3ff;"
                    f"border-left:3px solid #1559b8;border-radius:4px;"
                    f"font-style:italic'>"
                    f"<span style='color:#1559b8;font-weight:bold;font-style:normal'>"
                    f"What you said.</span> “{spoken}”</div>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            f"<div style='color:#404040;font-size:0.78rem;margin-top:6px'>"
            f"first at {r['earliest_start']:.2f}s · "
            f"latest at {r['latest_end']:.2f}s · "
            f"{r['n_events']} window(s)</div>",
            unsafe_allow_html=True,
        )
    with cols[3]:
        glyph, side_label = FEATURE_GLYPH.get(r["feature"], ("◉", ""))
        severity = severity_word(r["peak_z"])
        bars_html = severity_bar_html(r["peak_z"], color)
        is_face = r["feature"] in _FACE_FEATURES
        if is_face:
            head_html = twin_avatar_block_html(
                inflection_peak_t=peak_t,
                samples=ss.samples,
                peak_z=r["peak_z"],
                color=color,
                fallback_avatar_bgr=ss.avatar_bgr,
                size=80,
            )
        else:
            head_html = (
                f"<div style='font-size:2.6rem;line-height:1'>{glyph}</div>"
            )
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"{head_html}"
            f"<div style='font-size:0.72rem;color:{color};font-weight:bold;"
            f"letter-spacing:1px;margin-top:6px'>{side_label}</div>"
            f"<div style='margin-top:8px'>{bars_html}</div>"
            f"<div style='font-size:0.7rem;color:#9aa0a6;margin-top:6px;"
            f"text-transform:uppercase;letter-spacing:1.5px'>"
            f"{severity}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<hr style='margin:10px 0;border:none;border-top:1px solid #2a2f3a'>",
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

    # Defaults that used to be exposed in the sidebar. The cross-language
    # trial confirmed these are the right values across cases; surfacing
    # them as knobs only added cognitive load.
    camera_idx = 0           # default webcam
    adapt_on = True          # always-on adaptive baseline
    adapt_window_s = 5.0     # 5 s of fresh sit-still re-calibrates
    sigma_low = 3.0
    sigma_high = 6.0

    with st.sidebar:
        st.header("👤  Avatar")
        if st.button(
            "📷  Snap from webcam",
            use_container_width=True,
            key="snap_avatar",
            help="Grab the current webcam frame and use it as your avatar.",
        ):
            snap_cap = get_camera(int(camera_idx))
            ok, frame = snap_cap.read()
            if ok:
                ss.avatar_bgr = crop_to_square(frame)
                st.success("Avatar set from webcam.")
            else:
                st.error("Could not read from the webcam.")
        if ss.avatar_bgr is not None:
            cols_av = st.columns([1, 1])
            with cols_av[0]:
                st.image(
                    cv2.cvtColor(ss.avatar_bgr, cv2.COLOR_BGR2RGB),
                    width=88,
                )
            with cols_av[1]:
                if st.button("Clear", use_container_width=True, key="clear_avatar"):
                    ss.avatar_bgr = None
                    st.rerun()

        st.header("🎯  Baseline")
        names = list(ss.signatures.keys())
        baseline_choice = st.selectbox(
            "Compare against",
            options=["— none —"] + names,
            index=0,
        )
        st.header("🎬  Recording")
        c1, c2 = st.columns(2)
        if c1.button(
            "Record ▶" if not ss.recording else "Stop ■",
            use_container_width=True,
            key="rec_toggle",
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
        if c2.button("Clear", use_container_width=True, key="clear_take"):
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

        st.header("💾  Enroll new baseline")
        new_name = st.text_input(
            "Name for current samples",
            value="",
            placeholder="e.g. julio_seated_english",
            help="The label you'll pick from 'Compare against' next time.",
        )
        if st.button("Enroll", use_container_width=True, key="enroll_baseline"):
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

        st.header("🔍  Detect")
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
            key="detect_btn",
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
                ss.mobile_idx = 0

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

    tab_live, tab_insights, tab_mobile, tab_tech, tab_guide = st.tabs(
        ["📹 Live", "🎯 Insights", "📱 Mobile", "🔬 Technical", "❓ Guide"]
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
    with tab_mobile:
        render_mobile_tab()
    with tab_tech:
        render_technical_tab()
    with tab_guide:
        render_guide_tab()


if __name__ == "__main__":
    main()
