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
from voice_command import VoiceCommandListener
from face_trapezium import (
    Baseline,
    FaceTrapeziumDetector,
    FEATURE_NAMES,
    TrapeziumSample,
    _LIVE_STD_FLOOR,
    adapt_baseline,
    detect_sigma_changes,
    feature_vector,
    fit_baseline,
)
from feature_glossary import category, explain
from i18n import OUTPUT_LANGUAGES, has_full_translation, t

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

# Foreground colour to use for text drawn ON a category chip. Per WCAG
# luminance: black wins on the red and amber chips, white on grey/blue.
CATEGORY_TEXT = {
    "behavioural": "#0a0a0a",
    "pose": "#0a0a0a",
    "hardware": "#ffffff",
    "other": "#0a0a0a",
}

# Whisper language codes for the spoken-language picker in the sidebar.
# None = auto-detect (Whisper picks its best guess from the first 30 s
# of audio). Per the cross-language trial, forcing the correct language
# meaningfully improves transcript accuracy on shorter takes.
_LANGUAGE_OPTIONS: "Dict[str, Optional[str]]" = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Tamil": "ta",
    "Japanese": "ja",
    "Mandarin Chinese": "zh",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Yoruba": "yo",
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
    """Plain-language severity label, translated to the active output
    language — no numbers."""
    lang = st.session_state.get("out_lang", "en")
    if peak_z >= 12:
        return t("severity.intense", lang)
    if peak_z >= 6:
        return t("severity.marked", lang)
    if peak_z >= 3:
        return t("severity.noticeable", lang)
    return t("severity.subtle", lang)


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


@st.cache_resource
def get_voice_listener() -> VoiceCommandListener:
    return VoiceCommandListener()


def _toggle_recording_via_command(cmd: str) -> None:
    """Start or stop a take in response to a voice command. Mirrors the
    Record / Stop button handlers so the same state changes happen
    regardless of whether the user clicked or spoke.

    The listener is intentionally NOT suspended during recording so the
    user can also say 'stop' to end the take. macOS Core Audio allows
    sounddevice to keep its short capture chunks running alongside the
    AudioRecorder's continuous InputStream; if a particular platform
    won't share the device, the listener simply logs an error in its
    diagnostic strip and the user falls back to clicking Stop.
    """
    ss = st.session_state
    recorder = get_audio_recorder()
    now = time.time()
    if cmd == "record" and not ss.recording:
        if not ss.samples:
            ss.start_t = now
        ss.recording = True
        # Mark the voice trigger so Detect can trim the head seconds
        # contaminated by the spoken command + user hesitation.
        ss.voice_record_at = now
        ss.voice_stop_at = None
        if recorder.available:
            recorder.start(now)
    elif cmd == "stop" and ss.recording:
        ss.recording = False
        ss.voice_stop_at = now
        if recorder.available and recorder.recording:
            recorder.stop()


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
    _draw_recording_overlay(img, recording)


def _draw_recording_overlay(img: np.ndarray, recording: bool) -> None:
    """Big, unmissable REC / IDLE overlay drawn directly onto the live
    webcam frame. When recording: pulsing red border, large 'REC' badge
    with a blinking dot, and the elapsed take duration. When idle: small
    grey 'IDLE' tag in the corner.
    """
    h, w = img.shape[:2]
    if recording:
        # Pulsing red border: thickness oscillates between 4 and 10 px
        # over a ~1-second cycle so the user sees motion at the edges.
        pulse = int(7 + 3 * np.sin(time.time() * 4))
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 40, 230), pulse)
        # Translucent dark strip for the badge.
        cv2.rectangle(img, (0, 0), (340, 70), (0, 0, 0), -1)
        # Blinking red dot — full bright for 0.6 s, dim for 0.4 s.
        on = (time.time() % 1.0) < 0.6
        dot_color = (40, 40, 235) if on else (60, 60, 110)
        cv2.circle(img, (32, 35), 12, dot_color, -1)
        cv2.putText(
            img, "REC", (58, 47),
            cv2.FONT_HERSHEY_DUPLEX, 1.1, (60, 60, 235), 2, cv2.LINE_AA,
        )
        start_t = st.session_state.get("start_t")
        if start_t is not None:
            elapsed = max(0.0, time.time() - start_t)
            cv2.putText(
                img,
                f"{elapsed:5.1f} s",
                (140, 47),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
    else:
        cv2.rectangle(img, (0, 0), (110, 36), (0, 0, 0), -1)
        cv2.circle(img, (20, 18), 7, (170, 170, 170), -1)
        cv2.putText(
            img, "IDLE", (38, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA,
        )


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
    """Return a head + shoulders crop of the thumbnail captured closest to t."""
    payload = _crop_payload_at_time(t, samples)
    if payload is None:
        return None
    cropped, _, _, _ = payload
    return cropped


def _crop_payload_at_time(
    t: float, samples: List[TrapeziumSample]
):
    """Return (cropped_image, sample, scale, crop_box) for time t, or None.
    crop_box is (left, top, right, bottom) in thumbnail coordinates so callers
    can translate landmark coordinates onto the cropped image.
    """
    rec = _nearest_thumb_record(t, samples)
    if rec is None:
        return None
    sample_idx, payload = rec
    if isinstance(payload, tuple):
        thumb, scale = payload
    else:
        thumb, scale = payload, 1.0
    if sample_idx >= len(samples):
        return thumb, None, scale, (0, 0, thumb.shape[1], thumb.shape[0])
    sample = samples[sample_idx]
    box = _face_crop_box(sample, scale, thumb.shape)
    if box is None:
        h, w = thumb.shape[:2]
        return thumb, sample, scale, (0, 0, w, h)
    left, top, right, bottom = box
    cropped = thumb[top:bottom, left:right].copy()
    return cropped, sample, scale, box


def _draw_trap_on_crop(
    img: np.ndarray,
    sample: Optional[TrapeziumSample],
    scale: float,
    crop_box: tuple,
    color_bgr: tuple,
    thickness: int = 2,
) -> None:
    """Draw the trapezium polyline onto a cropped image in-place."""
    if sample is None:
        return
    left, top, _, _ = crop_box
    pts = []
    for p in (
        sample.left_eye[:2],
        sample.right_eye[:2],
        sample.right_mouth[:2],
        sample.left_mouth[:2],
    ):
        tx = int(p[0] * scale - left)
        ty = int(p[1] * scale - top)
        pts.append([tx, ty])
    cv2.polylines(
        img,
        [np.array(pts, dtype=np.int32)],
        isClosed=True,
        color=color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _active_baseline() -> Optional[Baseline]:
    """Return the Baseline currently used for detection (adapted form when
    the take has settled enough samples for the adapt window). Used by the
    spider + line charts to compute z-scores on the same scale the
    detector uses.
    """
    ss = st.session_state
    if not ss.active_baseline_key:
        return None
    key = ss.active_baseline_key.replace(" (adapted)", "")
    base = ss.signatures.get(key)
    if base is None:
        return None
    if ss.active_baseline_key.endswith("(adapted)") and ss.samples:
        cut_t = ss.samples[0].t + 5.0
        adapt_samples = [s for s in ss.samples if s.t <= cut_t]
        if len(adapt_samples) >= 5:
            base = adapt_baseline(base, adapt_samples)
    return base


def baseline_payload():
    """Return (cropped_baseline_img, baseline_sample, scale, crop_box) for
    the earliest captured thumbnail, or None."""
    thumbs = st.session_state.frame_thumbnails
    samples = st.session_state.samples
    if not thumbs or not samples:
        return None
    first_key = next(iter(thumbs))
    if first_key >= len(samples):
        return None
    payload = thumbs[first_key]
    if isinstance(payload, tuple):
        thumb, scale = payload
    else:
        thumb, scale = payload, 1.0
    sample = samples[first_key]
    box = _face_crop_box(sample, scale, thumb.shape)
    if box is None:
        h, w = thumb.shape[:2]
        box = (0, 0, w, h)
    left, top, right, bottom = box
    cropped = thumb[top:bottom, left:right].copy()
    return cropped, sample, scale, box


# Colors used for trapezium overlays (BGR for cv2).
TRAP_GREEN_BGR = (60, 200, 60)   # baseline
TRAP_RED_BGR = (60, 60, 220)     # NOW

# Curated subset of features to show on the spider chart per inflection.
# 22 axes would be unreadable; these 12 cover brow, mouth, head pose,
# trapezium sides and the parallelism / ratio summaries.
_SPIDER_FEATURES = (
    "left_brow_height_norm",
    "right_brow_height_norm",
    "angle_LM",
    "angle_RM",
    "mouth_offset_norm",
    "side_left_norm",
    "side_right_norm",
    "yaw_proxy",
    "pitch_proxy",
    "roll_proxy",
    "eye_mouth_ratio",
    "diag_ratio",
)


def _zscores_at_time(
    t: float,
    samples: List[TrapeziumSample],
    baseline: Optional[Baseline],
) -> Optional[Dict[str, float]]:
    """Return {feature_name: z_score} at the sample closest to ``t``."""
    if not samples or baseline is None:
        return None
    sample_times = np.array([s.t for s in samples])
    idx = int(np.argmin(np.abs(sample_times - t)))
    sample = samples[idx]
    feats = feature_vector(sample)
    z = (feats - baseline.means) / baseline.stds
    return {name: float(zi) for name, zi in zip(baseline.feature_names, z)}


def render_spider_chart(
    peak_t: float,
    samples: List[TrapeziumSample],
    baseline: Optional[Baseline],
    feature_being_highlighted: str,
    accent_color: str,
) -> None:
    """Polar (radar) chart comparing BASELINE (acceptable range, shaded
    green) vs NOW (percent of breach threshold per dimension).

    Axis labels are the technical feature names (the text that lives in
    parentheses on the row title). Values are mapped from σ to percent
    of the OUT-OF-NORM line via z/6 × 100, so watch-line = 50 %,
    out-of-norm = 100 %.
    """
    z_map = _zscores_at_time(peak_t, samples, baseline)
    if z_map is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    features = [f for f in _SPIDER_FEATURES if f in z_map]
    if not features:
        return

    # Map σ → percent of breach threshold (6 σ = 100 %).
    # Cap to 200 % so a single extreme outlier doesn't eat the chart.
    pct_vals = [min(abs(z_map[f]) / 6.0 * 100.0, 200.0) for f in features]
    # Spider axis labels: plain-English category names like 'LEFT SMILE',
    # 'HEAD TURN', 'LIP SHIFT' — translated to the active output language
    # via the spider.<feature> i18n keys, falling back to a generated
    # uppercase version of the technical name when no translation exists.
    spider_lang = st.session_state.get("out_lang", "en")
    labels = [
        t(f"spider.{f}", spider_lang)
        if t(f"spider.{f}", spider_lang) != f"spider.{f}"
        else f.replace("_", " ").upper()
        for f in features
    ]

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    now_vals = pct_vals + [pct_vals[0]]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfd")

    lang = st.session_state.get("out_lang", "en")
    # Green-shaded BASELINE acceptable range: full circle from 0 to 50 %.
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    r_outer = np.full_like(theta_ring, 50.0)
    ax.fill(theta_ring, r_outer, color="#2a8a3a", alpha=0.22,
            label=t("chart.spider.baseline", lang))
    ax.plot(theta_ring, r_outer, color="#2a8a3a", linewidth=1.5)

    # Dashed OUT-OF-NORM circle at 100% — the σ-high statistical threshold.
    r_breach = np.full_like(theta_ring, 100.0)
    ax.plot(theta_ring, r_breach, color="#d9534f",
            linewidth=1.5, linestyle=(0, (5, 4)),
            label=t("label.breach_line", lang))

    # NOW polygon.
    ax.plot(angles, now_vals, color=accent_color, linewidth=2.2,
            label=t("chart.spider.now", lang),
            marker="o", markersize=5)
    ax.fill(angles, now_vals, color=accent_color, alpha=0.22)

    theta_labels = ax.set_thetagrids(
        np.degrees(angles[:-1]), labels, fontsize=9, color="#1a1a1a"
    )
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    rmax = max(max(pct_vals) * 1.15, 120.0)
    ax.set_ylim(0, rmax)
    ax.set_rticks([50, 100])
    ax.set_yticklabels(["50%", "100%"], fontsize=8, color="#555555")
    ax.tick_params(axis="y", colors="#555555")
    ax.grid(color="#cccccc", linewidth=0.6)
    ax.spines["polar"].set_visible(False)
    # Place the legend WELL BELOW the chart so it never sits on top of
    # an axis category label. bbox_to_anchor uses axes-fraction
    # coordinates: y < 0 is under the plotting area.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        fontsize=10,
        frameon=False,
    )

    buf = BytesIO()
    # subplots_adjust gives the legend a dedicated band; bbox_inches='tight'
    # then crops without clipping the legend itself.
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.25,
                dpi=110, facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    st.markdown(
        f"<div style='text-align:center;margin-top:6px'>"
        f"<img src='data:image/png;base64,{b64}' "
        f"style='width:100%;height:auto'/>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_line_chart(
    samples: List[TrapeziumSample],
    baseline: Optional[Baseline],
    feature: str,
    window: tuple,
    accent_color: str,
) -> None:
    """Time-series of the dominant feature over the take, scaled to the
    same percent-of-out-of-norm axis the spider chart uses (so 0 = at
    baseline, ±50 % = watch line, ±100 % = out of norm). BASELINE drawn
    as the green zero rule; NOW drawn as the deviation trace; the
    inflection window highlighted in the accent colour; the watch
    (±50 %) and out-of-norm (±100 %) thresholds dashed.
    """
    if not samples or baseline is None:
        return
    if feature not in baseline.feature_names:
        return
    j = list(baseline.feature_names).index(feature)
    rows = []
    for s in samples:
        feats = feature_vector(s)
        z = (feats - baseline.means) / baseline.stds
        # Map σ to percent of out-of-norm: σ-high (6 σ) = 100 %.
        # Preserve sign so the trace can dip below the baseline rule too.
        pct = float(z[j]) / 6.0 * 100.0
        rows.append({
            "t": s.t,
            "pct": pct,
            "abs_pct": abs(pct),
            "t_int": int(round(s.t)),
            "abs_pct_int": int(round(abs(pct))),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    lang = st.session_state.get("out_lang", "en")

    # Render with matplotlib for parity with the spider chart: same
    # bold-uppercase axis titles, same bottom-centered legend block,
    # same approximate square-ish proportions.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfd")

    # Inflection-window shading (labelled this time).
    if window:
        ax.axvspan(
            window[0], window[1],
            color=accent_color, alpha=0.18,
            label=t("chart.legend.episode", lang),
        )

    # Threshold rules.
    ax.axhline(0, color="#2a8a3a", linewidth=2,
               label=t("chart.legend.baseline", lang))
    ax.axhline(50, color="#9aa0a6", linewidth=1.4,
               linestyle=(0, (5, 4)),
               label=t("chart.legend.watch", lang))
    ax.axhline(-50, color="#9aa0a6", linewidth=1.4,
               linestyle=(0, (5, 4)))
    ax.axhline(100, color="#d9534f", linewidth=1.4,
               linestyle=(0, (5, 4)),
               label=t("chart.legend.oon", lang))
    ax.axhline(-100, color="#d9534f", linewidth=1.4,
               linestyle=(0, (5, 4)))

    # NOW trace.
    ax.plot(
        df["t"].values, df["pct"].values,
        color=accent_color, linewidth=2.2,
        label=t("chart.legend.now", lang),
    )

    # Bold-uppercase axis titles to mirror the spider chart's category
    # font.
    ax.set_xlabel(
        t("chart.time", lang).upper(),
        fontsize=11, fontweight="bold", color="#1a1a1a",
    )
    ax.set_ylabel(
        t("chart.y_pct", lang).upper(),
        fontsize=11, fontweight="bold", color="#1a1a1a",
    )

    # Y axis ticks at the same magnitudes the spider uses.
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_yticklabels(["-100%", "-50%", "0%", "50%", "100%"],
                       fontsize=9, color="#555555")
    ax.tick_params(axis="x", labelsize=9, colors="#555555")
    rmax = max(abs(df["pct"].max()), abs(df["pct"].min()), 110.0)
    ax.set_ylim(-rmax * 1.05, rmax * 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(color="#e8e8e8", linewidth=0.6)

    # Bottom-centred single-column legend — matches the spider's format.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        fontsize=10,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.30)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.25,
                dpi=110, facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    st.markdown(
        f"<div style='text-align:center;margin-top:6px'>"
        f"<img src='data:image/png;base64,{b64}' "
        f"style='width:100%;height:auto'/>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# Session-state init
# ─────────────────────────────────────────────────────────────────────────

def crop_to_face_avatar(
    frame: np.ndarray, size: int = 256
) -> Optional[np.ndarray]:
    """Run the face detector on a single frame, tightly crop to the head
    region, then apply a circular alpha mask so the avatar reads as a
    cleanly-isolated headshot — no room background, no chair, no
    surrounding shoulders. Returns the masked BGR image (with the
    background filled white). None if no face detected."""
    detector = get_detector()
    sample = detector.detect(frame, 0.0)
    if sample is None:
        return None
    h, w = frame.shape[:2]
    xs = [
        sample.left_eye[0], sample.right_eye[0],
        sample.left_mouth[0], sample.right_mouth[0],
        sample.left_brow[0], sample.right_brow[0],
    ]
    ys = [
        sample.left_eye[1], sample.right_eye[1],
        sample.left_mouth[1], sample.right_mouth[1],
        sample.left_brow[1], sample.right_brow[1],
    ]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    fw = x_max - x_min
    fh = y_max - y_min
    if fw <= 0 or fh <= 0:
        return None
    # Generous head crop: hair top, sliver of neck, full ears.
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    span = max(fw, fh) * 2.2
    left = int(max(0, cx - span / 2))
    right = int(min(w, cx + span / 2))
    top = int(max(0, cy - span * 0.55))
    bottom = int(min(h, cy + span * 0.55))
    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right].copy()
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    # Soft-edged circular mask → white background. Looks like a stock
    # avatar headshot rather than a webcam capture.
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.46), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=size * 0.025)
    mask3 = cv2.merge([mask, mask, mask])
    white = np.full_like(crop, 255, dtype=np.uint8)
    blended = (crop.astype(np.float32) * mask3 +
               white.astype(np.float32) * (1.0 - mask3))
    return blended.astype(np.uint8)


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
    ss.setdefault("out_lang", "en")
    ss.setdefault("last_enrolled", None)
    # Voice-trigger time stamps used to trim the first/last second(s)
    # of the analyzed take so the spoken command word + any user
    # hesitation right after detection doesn't contaminate Detect.
    ss.setdefault("voice_record_at", None)
    ss.setdefault("voice_stop_at", None)


# How much voice-triggered head / tail to trim out of the analysed take.
VOICE_HEAD_TRIM_S = 1.0
VOICE_TAIL_TRIM_S = 0.5


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
                "feature_short": explain(e["dominant_feature"], st.session_state.get("out_lang", "en")).short,
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
    order = [explain(r["feature"], st.session_state.get("out_lang", "en")).short for r in rows]
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
    lang = st.session_state.get("out_lang", "en")
    if t2_peak >= sigma_high:
        return t("status.out_of_norm", lang)
    if t2_peak >= sigma_low:
        return t("status.excursion", lang)
    return t("status.stable", lang)


def quick_story_line(summary: dict) -> str:
    if not summary:
        return ""
    events = summary.get("events", [])
    if not events:
        return "Stable take — nothing crossed the watch line."
    rows = aggregate_unique_inflections(events)
    if not rows:
        return ""
    top = rows[0]
    fx = explain(top["feature"], st.session_state.get("out_lang", "en"))
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
        # Voice-command polling moved to a dedicated top-level fragment
        # in main() so commands fire regardless of active tab.
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
            "Stable take. No feature crossed the watch line. Your face stayed within "
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
                f"{t('section.nuance', ss.get('out_lang', 'en'))} "
                f"{ss.mobile_idx + 1} "
                f"{t('section.of', ss.get('out_lang', 'en'))} {total}"
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
    fx = explain(r["feature"], st.session_state.get("out_lang", "en"))
    cat = r["category"]
    color = CATEGORY_PALETTE[cat]
    label = t(f"cat.{cat}", st.session_state.get("out_lang", "en"))
    best = r["best_event"]
    peak_t = 0.5 * (best["start_t"] + best["end_t"])

    # Title row.
    st.markdown(
        f"<div style='display:flex;align-items:center;"
        f"justify-content:space-between;color:#0a0a0a'>"
        f"<div style='font-size:1.1rem;font-weight:bold;color:#0a0a0a'>"
        f"#{rank} · {fx.short}</div>"
        f"<span style='background:{color};color:{CATEGORY_TEXT[cat]};"
        f"padding:3px 10px;border-radius:4px;font-size:0.78rem;"
        f"font-weight:bold'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    # Body copy — dark text for light-theme readability.
    st.markdown(
        f"<div style='color:#1a1a1a;font-size:1.0rem;margin-top:10px;"
        f"line-height:1.45'>"
        f"<b style='color:#0a0a0a'>{t('field.what_measures', st.session_state.get('out_lang','en'))}</b> {fx.physical}</div>",
        unsafe_allow_html=True,
    )
    if fx.citation:
        st.markdown(
            f"<div style='color:#2c2c2c;font-size:0.92rem;margin-top:6px;"
            f"line-height:1.45'>"
            f"<b style='color:#0a0a0a'>{t('field.cited_science', st.session_state.get('out_lang','en'))}</b> {fx.citation}</div>",
            unsafe_allow_html=True,
        )
    if fx.science_summary:
        st.markdown(
            f"<div style='color:#1a1a1a;font-size:0.95rem;margin-top:4px;"
            f"line-height:1.45;padding:8px 12px;background:#f6f8fc;"
            f"border-left:3px solid #1a1a1a;border-radius:4px'>"
            f"<b style='color:#0a0a0a'>{t('field.plain_lang', st.session_state.get('out_lang','en'))}</b> "
            f"{fx.science_summary}</div>",
            unsafe_allow_html=True,
        )
    if fx.socratic:
        st.markdown(
            f"<div style='color:{color};font-size:1.0rem;font-style:italic;"
            f"margin-top:10px;font-weight:600'>{fx.socratic}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#404040;font-size:0.85rem;margin-top:4px;"
            f"font-style:italic'>"
            f"{t('implication.html', st.session_state.get('out_lang','en'))}</div>",
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
                f"{t('field.what_said', st.session_state.get('out_lang','en'))}</span> “{spoken}”</div>",
                unsafe_allow_html=True,
            )
    _lang_ts = st.session_state.get("out_lang", "en")
    st.markdown(
        f"<div style='color:#404040;font-size:0.82rem;margin-top:10px'>"
        f"{t('ts.first', _lang_ts)} {r['earliest_start']:.2f} s · "
        f"{t('ts.latest', _lang_ts)} {r['latest_end']:.2f} s · "
        f"{r['n_events']} {t('ts.windows', _lang_ts)}</div>",
        unsafe_allow_html=True,
    )

    # Section 3 (was Section 3 of three; first close-up was dropped).
    _lang_bvn = st.session_state.get("out_lang", "en")
    st.markdown(
        "<div style='margin-top:24px;font-size:0.95rem;color:#0a0a0a;"
        "letter-spacing:2px;text-transform:uppercase;text-align:center;"
        "font-weight:bold'>"
        f"{t('section.baseline_vs', _lang_bvn)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin-top:8px'></div>", unsafe_allow_html=True
    )
    twin_html = _mobile_twin_html(r, peak_t, color)
    st.markdown(twin_html, unsafe_allow_html=True)

    # Side label + severity ladder + severity word (per request:
    # matches the Insights tab's "RIGHT brow ▶ / 4-pip bar / NOTICEABLE"
    # descriptor block).
    _, _default_side = FEATURE_GLYPH.get(r["feature"], ("◉", ""))
    _side_lang = st.session_state.get("out_lang", "en")
    side_label = t(f"side_label.{r['feature']}", _side_lang)
    if side_label == f"side_label.{r['feature']}":
        side_label = _default_side
    severity = severity_word(r["peak_z"])
    bars_html = severity_bar_html(r["peak_z"], color)
    st.markdown(
        f"<div style='text-align:center;margin-top:18px'>"
        f"<div style='font-size:1.05rem;color:{color};font-weight:bold;"
        f"letter-spacing:1.2px'>{side_label}</div>"
        f"<div style='margin-top:12px'>{bars_html}</div>"
        f"<div style='font-size:0.88rem;color:{color};margin-top:10px;"
        f"letter-spacing:2px;text-transform:uppercase;font-weight:bold'>"
        f"{severity}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Spider + line charts.
    baseline = _active_baseline()
    lang = ss.get("out_lang", "en")
    if baseline is not None:
        st.markdown(
            "<div style='margin-top:28px;font-size:1.35rem;color:#0a0a0a;"
            "letter-spacing:2px;text-transform:uppercase;text-align:center;"
            "font-weight:800'>"
            f"{t('section.where_moved', lang)}</div>"
            "<div style='text-align:center;color:#404040;font-size:0.9rem;"
            "font-style:italic;margin-top:4px'>"
            f"{t('spider.subtitle', lang)}"
            "</div>",
            unsafe_allow_html=True,
        )
        render_spider_chart(peak_t, ss.samples, baseline, r["feature"], color)

        # Strip <b> tags from the implication HTML for use inside an HTML
        # title= attribute (which can't contain markup).
        implication_plain = (
            t("implication.html", lang)
            .replace("<b>", "")
            .replace("</b>", "")
            .replace("&nbsp;", " ")
        )
        # Data callout above the line chart: nuance number + feature title.
        # Size + weight scaled up to be proportional / promotional to the
        # 'HOW IT CHANGED OVER TIME' section header that sits directly
        # below it (1.35 rem, weight 800).
        st.markdown(
            f"<div style='text-align:center;margin-top:24px;margin-bottom:6px'>"
            f"<span style='display:inline-block;padding:10px 20px;"
            f"background:#f6f8fc;border:1px solid #c8d0db;border-radius:24px;"
            f"font-size:1.30rem;font-weight:800;color:#0a0a0a;"
            f"letter-spacing:0.5px;"
            f"cursor:help;border-bottom:3px dotted {color};"
            f"box-shadow:0 1px 3px rgba(0,0,0,0.05)' "
            f"title='{implication_plain}'>"
            f"#{rank} · {fx.short}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin-top:8px;font-size:1.35rem;color:#0a0a0a;"
            "letter-spacing:2px;text-transform:uppercase;text-align:center;"
            "font-weight:800'>"
            f"{t('section.over_time', lang)} · "
            + explain(r["feature"], lang).short
            + "</div>",
            unsafe_allow_html=True,
        )
        render_line_chart(
            ss.samples,
            baseline,
            r["feature"],
            window=(best["start_t"], best["end_t"]),
            accent_color=color,
        )

    st.markdown(
        "<hr style='margin:24px 0;border:none;border-top:1px solid #cccccc'>",
        unsafe_allow_html=True,
    )


def _mobile_twin_html(r: dict, peak_t: float, color: str) -> str:
    """Twin baseline-vs-now panels rendered larger and side-by-side.

    Both panels are head + shoulders crops regardless of feature type so
    every inflection looks the same scale. Baseline gets a GREEN trapezium
    drawn onto it; NOW gets a RED trapezium. A third image stacked below
    overlays both trapezia onto the NOW frame so the geometric shift is
    visible at a glance.
    """
    ss = st.session_state
    baseline = baseline_payload()
    now = _crop_payload_at_time(peak_t, ss.samples)

    baseline_img = baseline_sample = baseline_scale = baseline_box = None
    if baseline is not None:
        baseline_img, baseline_sample, baseline_scale, baseline_box = baseline
    now_img = now_sample = now_scale = now_box = None
    if now is not None:
        now_img, now_sample, now_scale, now_box = now

    # Draw colour-coded trapezia in-place.
    if baseline_img is not None:
        _draw_trap_on_crop(
            baseline_img,
            baseline_sample,
            baseline_scale or 1.0,
            baseline_box,
            TRAP_GREEN_BGR,
            thickness=2,
        )
    if now_img is not None:
        _draw_trap_on_crop(
            now_img,
            now_sample,
            now_scale or 1.0,
            now_box,
            TRAP_RED_BGR,
            thickness=2,
        )

    # Fall back to the user's avatar if no take frames yet.
    if baseline_img is None:
        baseline_img = ss.avatar_bgr
    if now_img is None:
        now_img = ss.avatar_bgr

    baseline_ring = "#5cb85c"
    border_w = min(6, max(2, int(r["peak_z"] // 3)))
    skew_label = f"NOW ({r['peak_z']:.1f}σ)"

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

    # Build the fusion overlay: same NOW frame, with the BASELINE
    # trapezium drawn on top in green so the geometric shift is visible.
    fusion_img = None
    if now_img is not None and baseline_sample is not None and now_box is not None:
        fusion_img = now_img.copy()  # already has the RED now-trap drawn
        _draw_trap_on_crop(
            fusion_img,
            baseline_sample,
            now_scale or 1.0,
            now_box,
            TRAP_GREEN_BGR,
            thickness=2,
        )

    if fusion_img is None:
        return ""
    ok, buf = cv2.imencode(".png", fusion_img)
    if not ok:
        return ""
    import base64
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    src = f"data:image/png;base64,{b64}"
    lang = st.session_state.get("out_lang", "en")
    baseline_tooltip = (
        "Your reference profile — what NOW is being compared against. "
        "Two patterns are common: "
        "(a) at-rest baseline — how your face sits when calm, captured before "
        "any performance starts; "
        "(b) prior-take baseline — a previous attempt at the same content "
        "(an earlier elevator-pitch run, a yesterday's-keynote run) that you "
        "now want to refine against. "
        "Stored as the median + spread of dozens of enrollment frames."
    )
    now_tooltip = (
        "Your current take at the moment of this nuance — the frame closest "
        "to the peak of the detected deviation window in the most recent "
        "recording. Compared against whichever baseline you selected."
    )
    return (
        f"<div style='margin-top:8px;text-align:center'>"
        f"<div style='font-size:0.85rem;color:#0a0a0a;"
        f"letter-spacing:1.5px;font-weight:bold;margin-bottom:8px'>"
        f"{t('label.overlay', lang)} "
        f"<span style='color:{baseline_ring};cursor:help;"
        f"border-bottom:1px dotted {baseline_ring}' "
        f"title='{baseline_tooltip}'>{t('label.baseline_vs', lang)}</span> "
        f"<span style='color:#0a0a0a'>{t('label.vs', lang)}</span> "
        f"<span style='color:#d9534f;cursor:help;"
        f"border-bottom:1px dotted #d9534f' "
        f"title='{now_tooltip}'>{t('label.now_vs', lang)}</span>"
        f"</div>"
        # Cap the displayed width so the border traces the image edge
        # rather than the column edge; display:block + width:100% inside
        # the cap means the border always sits flush against the photo
        # as the user expands or shrinks the browser window.
        f"<div style='max-width:520px;margin:0 auto'>"
        f"<img src='{src}' style='display:block;width:100%;height:auto;"
        f"border-radius:14px;border:2px solid #1a1a1a;"
        f"box-shadow:0 0 14px {color}'/>"
        f"</div>"
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
            "Stable take. No feature crossed the watch line. Your face stayed within "
            "the geometric envelope of your baseline."
        )
        return

    st.markdown("### Ranked unique nuances")
    st.caption(
        "One row per distinct feature, sorted by impact (duration × peak σ). "
        "No duplicates: each feature appears once with its peak frame."
    )

    for i, r in enumerate(rows[:8]):
        _render_inflection_row(i + 1, r)


def _render_moment(feature: str, t: float) -> None:
    ss = st.session_state
    fx = explain(feature, st.session_state.get("out_lang", "en"))
    cat = category(feature)
    color = CATEGORY_PALETTE[cat]
    label = t(f"cat.{cat}", st.session_state.get("out_lang", "en"))

    st.markdown(
        f"<div style='border:1px solid {color};border-radius:6px;"
        f"padding:14px;margin-top:6px;background:#0e1117'>"
        f"<div style='display:inline-block;background:{color};"
        f"color:{CATEGORY_TEXT[cat]};padding:2px 10px;border-radius:4px;"
        f"font-size:0.8rem;font-weight:bold'>{label}</div>"
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
    fx = explain(r["feature"], st.session_state.get("out_lang", "en"))
    cat = r["category"]
    color = CATEGORY_PALETTE[cat]
    label = t(f"cat.{cat}", st.session_state.get("out_lang", "en"))
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
            f"<span style='background:{color};color:{CATEGORY_TEXT[cat]};"
            f"padding:1px 8px;border-radius:4px;font-size:0.75rem;"
            f"font-weight:bold'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#1a1a1a;font-size:0.95rem;margin-top:4px;"
            f"line-height:1.4'>"
            f"<b style='color:#0a0a0a'>{t('field.what_measures', st.session_state.get('out_lang','en'))}</b> {fx.physical}</div>",
            unsafe_allow_html=True,
        )
        if fx.citation:
            st.markdown(
                f"<div style='color:#2c2c2c;font-size:0.9rem;margin-top:3px;"
                f"line-height:1.4'>"
                f"<b style='color:#0a0a0a'>{t('field.cited_science', st.session_state.get('out_lang','en'))}</b> {fx.citation}</div>",
                unsafe_allow_html=True,
            )
        if fx.science_summary:
            st.markdown(
                f"<div style='color:#1a1a1a;font-size:0.9rem;margin-top:3px;"
                f"line-height:1.4;padding:6px 10px;background:#f6f8fc;"
                f"border-left:3px solid #1a1a1a;border-radius:4px'>"
                f"<b style='color:#0a0a0a'>{t('field.plain_lang', st.session_state.get('out_lang','en'))}</b> "
                f"{fx.science_summary}</div>",
                unsafe_allow_html=True,
            )
        if fx.socratic:
            st.markdown(
                f"<div style='color:{color};font-size:0.95rem;font-style:italic;"
                f"margin-top:6px;font-weight:600'>"
                f"{fx.socratic}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:#404040;font-size:0.8rem;margin-top:2px;"
                f"font-style:italic'>"
                f"{t('implication.html', st.session_state.get('out_lang','en'))}</div>",
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
                    f"{t('field.what_said', st.session_state.get('out_lang','en'))}</span> “{spoken}”</div>",
                    unsafe_allow_html=True,
                )
        _lang_it = st.session_state.get("out_lang", "en")
        st.markdown(
            f"<div style='color:#404040;font-size:0.78rem;margin-top:6px'>"
            f"{t('ts.first', _lang_it)} {r['earliest_start']:.2f}s · "
            f"{t('ts.latest', _lang_it)} {r['latest_end']:.2f}s · "
            f"{r['n_events']} {t('ts.windows', _lang_it)}</div>",
            unsafe_allow_html=True,
        )
    with cols[3]:
        glyph, _default_side = FEATURE_GLYPH.get(r["feature"], ("◉", ""))
        _side_lang2 = st.session_state.get("out_lang", "en")
        side_label = t(f"side_label.{r['feature']}", _side_lang2)
        if side_label == f"side_label.{r['feature']}":
            side_label = _default_side
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

        We call it **"out of norm"** when T² goes past the σ-high
        threshold (default 6 σ) — the dashboard's red dashed ring.
        An **"excursion"** is between the watch line (3 σ) and out-of-norm.

        #### Two ways to use this

        The math is the same in both flows. What changes is **what you
        enroll as the baseline**.

        **A — Coaching against your at-rest self.**
        Enroll a 30-second neutral-and-calm take as your baseline. Then
        deliver the actual speech / interview answer. The system surfaces
        nuances where you departed from rest — moments of stress, surprise,
        concentration, fatigue.

        ```
        Record 30 s calm  →  Enroll as  julio_at_rest_en
        Compare against   →  julio_at_rest_en
        Record the pitch  →  Detect
        ```

        **B — Rehearsing against your prior take.**
        Record attempt #1 of a 30-second elevator pitch, enroll it. Then
        record attempt #2 against that enrollment. The system surfaces
        where attempt #2 differs from attempt #1 — improvement, drift, a
        more lopsided smile, a held brow that didn't show up before. This
        is the iterative-rehearsal loop.

        ```
        Record attempt 1  →  Enroll as  pitch_attempt_1
        Compare against   →  pitch_attempt_1
        Record attempt 2  →  Detect  →  see what changed
        Re-enroll best take  →  Compare against it next session
        ```

        Either way, the **NOW** in every nuance card is your most recent
        recording. The **BASELINE** is whichever profile you pointed
        "Compare against" at.

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

def _render_next_step_guidance(ss) -> None:
    """State-aware 'do this next' hint at the top of the sidebar.
    Walks the user through use case A (at-rest coaching) and use
    case B (rehearsal against prior take) without making them read
    the Guide tab first.
    """
    n_baselines = len(ss.signatures)
    n_samples = len(ss.samples)
    recording = ss.recording
    baseline_selected = ss.active_baseline_key

    if recording:
        msg = "🔴  **Recording…** click **Stop ■** when done."
        st.info(msg)
        return

    if n_baselines == 0 and n_samples == 0:
        st.info(
            "👋  **First time here?**\n\n"
            "**Use case A · live coaching.** Record 30 s sitting calm and "
            "neutral → **Enroll** as `at_rest_en`.\n\n"
            "**Use case B · rehearsal.** Record your 30-s elevator pitch → "
            "**Enroll** as `pitch_attempt_1`.\n\n"
            "Either way: record your first take, give it a name, hit Enroll."
        )
        return

    if n_baselines > 0 and n_samples == 0:
        st.info(
            "✓  You have enrolled baselines.\n\n"
            "**Next:** pick one under **🎯 Compare against**, then "
            "**Record ▶** a fresh take, then **🔍 Detect** to see the "
            "ranked nuances vs. that baseline."
        )
        return

    if n_samples > 0 and not baseline_selected:
        # Take captured but no Detect yet.
        if n_baselines == 0:
            st.info(
                "📥  Take captured ({n} samples). "
                "**Next:** name it under **💾 Enroll new baseline** and "
                "click **Enroll** so you can compare future takes against "
                "it.".format(n=n_samples)
            )
        else:
            st.info(
                "📊  Take captured ({n} samples).\n\n"
                "**Two options:**\n"
                "1. **Detect** vs an existing baseline (pick one above + "
                "🔍 Detect).\n"
                "2. **Enroll** this take itself as a new baseline (give it "
                "a name + 💾 Enroll).".format(n=n_samples)
            )
        return

    if n_samples > 0 and baseline_selected:
        st.success(
            f"🎯  Active baseline: **{baseline_selected}**.\n\n"
            f"Recorded take in buffer. **Click 🔍 Detect** to see the "
            f"ranked nuances on the Mobile / Insights tabs."
        )
        return


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
        _render_next_step_guidance(ss)

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
                isolated = crop_to_face_avatar(frame, size=256)
                if isolated is not None:
                    ss.avatar_bgr = isolated
                    st.success("Avatar set — face isolated, background hidden.")
                else:
                    # Fallback: no face detected → save the centre square.
                    ss.avatar_bgr = crop_to_square(frame)
                    st.warning(
                        "No face detected — saved a centre square crop. "
                        "Move closer to the camera and try again for "
                        "background isolation."
                    )
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

        st.header("🗣  Insights language")
        out_label = st.selectbox(
            "Output language",
            options=list(OUTPUT_LANGUAGES.keys()),
            index=0,
            help=(
                "The language the dashboard renders insights in — labels, "
                "category chips, section headers, plain-language science "
                "summaries. Independent of what you SPEAK; spoken language "
                "is always auto-detected by Whisper in the background.\n\n"
                "Translations are rolling out progressively. English and "
                "Español are most complete; other choices show with a "
                "(English fallback) note while their translations land."
            ),
        )
        out_lang = OUTPUT_LANGUAGES[out_label]
        if not has_full_translation(out_lang):
            st.caption(
                f"_{out_label}: translations in progress — falling back to "
                f"English where keys aren't translated yet._"
            )
        # Spoken language for Whisper transcription stays on auto-detect.
        language_code = None

        st.header("🎯  Baseline")
        names = list(ss.signatures.keys())
        baseline_choice = st.selectbox(
            "Compare against",
            options=["— none —"] + names,
            index=0,
            help=(
                "Your reference profile — what NOW will be compared against. "
                "Common patterns: at-rest baseline (your calm neutral state), "
                "prior-take baseline (a previous run of the same elevator "
                "pitch / keynote you want to refine against), per-language "
                "baseline, per-glasses-configuration baseline. One enrolled "
                "name per pattern is encouraged."
            ),
        )
        st.header("🎬  Recording")
        listener = get_voice_listener()
        if listener.available:
            voice_on = st.checkbox(
                "🎤  Hands-free voice control",
                value=listener.enabled,
                key="voice_control_toggle",
                help=(
                    "Say 'record' (or 'start' / 'begin' / 'grabar' / "
                    "'comenzar') to start a take. Say 'stop' (or 'halt' / "
                    "'parar' / 'detener') to stop.\n\n"
                    "Eliminates the hand and head movement that comes from "
                    "clicking the Record button — the take you save is "
                    "less contaminated by the act of starting it.\n\n"
                    "Detection uses the same local Whisper-small model — "
                    "no cloud."
                ),
            )
            if voice_on and not listener.enabled:
                listener.enable()
            elif not voice_on and listener.enabled:
                listener.disable()

            # Diagnostic strip — surfaces what the listener is actually doing.
            if listener.enabled:
                model_status = (
                    "✅ model loaded" if listener.model_loaded
                    else "⏳ model loading…"
                )
                heard = listener.last_transcript
                err = listener.last_error
                st.caption(
                    f"{model_status}  ·  "
                    f"chunks: {listener.chunks_processed}  ·  "
                    f"cmds: {listener.commands_detected}  ·  "
                    f"mic peak: {listener.last_chunk_peak:.3f}"
                )
                if heard:
                    st.caption(f"🎤 heard: _{heard}_")
                else:
                    st.caption(
                        "🎤 listening… (say something to see what it hears)"
                    )
                if err:
                    st.error(f"⚠ {err}")

            # One-shot test button — captures one 1.5-s chunk synchronously
            # and shows what was heard + what command (if any) matched.
            if st.button(
                "🧪 Test mic (1.5 s)",
                use_container_width=True,
                key="test_voice",
                help=(
                    "Records 1.5 s right now, transcribes locally, and "
                    "tells you what the listener thinks you said. Use this "
                    "to verify the mic + model + keyword match all work "
                    "before relying on the background listener."
                ),
            ):
                with st.spinner("Listening…"):
                    cmd = listener.test_one_shot()
                heard = listener.last_transcript
                err = listener.last_error
                if err:
                    st.error(f"⚠ {err}")
                elif heard:
                    if cmd:
                        st.success(
                            f"✅ heard '{heard}' → matched **{cmd}**"
                        )
                    else:
                        st.info(
                            f"heard '{heard}' — no keyword match. "
                            f"Try 'record' or 'stop'."
                        )
                else:
                    st.warning(
                        "Nothing heard. Check macOS Privacy & Security → "
                        "Microphone → make sure Terminal (or your Python) "
                        "is allowed."
                    )
        else:
            st.caption(
                "🎤 voice control unavailable — install `sounddevice` to "
                "enable hands-free record/stop."
            )
        c1, c2 = st.columns(2)
        rec_help = (
            "Start or stop a take.\n\n"
            "• **Use case A — coaching.** Record 30 s sitting calm and "
            "neutral, then Enroll as your at-rest baseline.\n"
            "• **Use case B — rehearsal.** Record your 30-s elevator pitch "
            "or keynote opener. Enroll it the first time, then Detect "
            "against it on every subsequent attempt.\n\n"
            "Recording captures face landmarks + audio; nothing leaves "
            "this device."
        )
        if c1.button(
            "Record ▶" if not ss.recording else "Stop ■",
            use_container_width=True,
            key="rec_toggle",
            help=rec_help,
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
            # Force an immediate re-render so the button label flips to
            # "Stop ■" / "Record ▶" on this same click instead of waiting
            # for the next user interaction.
            st.rerun()
        if c2.button(
            "Clear",
            use_container_width=True,
            key="clear_take",
            help=(
                "Discard the current take's samples, transcript and frames. "
                "Does NOT touch your enrolled baselines — those are saved "
                "separately to signatures.json. Use before each new take "
                "in either use case."
            ),
        ):
            ss.samples = []
            ss.start_t = None
            ss.event_log = None
            ss.transcript = []
            ss.frame_thumbnails = OrderedDict()
            ss.active_baseline_key = None
            ss.voice_record_at = None
            ss.voice_stop_at = None
            if recorder.available:
                recorder.clear()
            st.rerun()

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
            placeholder="e.g. julio_at_rest_en  or  pitch_attempt_1",
            help=(
                "The label you'll pick from 'Compare against' next time. "
                "Naming patterns: \n"
                "• julio_at_rest_en — your calm neutral, English\n"
                "• julio_at_rest_es — your calm neutral, Spanish\n"
                "• pitch_attempt_1 — first run of the 30-s elevator pitch\n"
                "• pitch_attempt_5 — fifth run, ready to compare against #4\n"
                "• keynote_opener_v1 — first take of the keynote opener\n"
                "Re-enroll any time conditions change (glasses on/off, new "
                "chair, new lighting)."
            ),
        )
        if st.button(
            "Enroll",
            use_container_width=True,
            key="enroll_baseline",
            help=(
                "Save the current take as a reference profile that future "
                "takes can compare against.\n\n"
                "• **Use case A — coaching.** Enroll your calm 30-s take "
                "as `at_rest_en` (or `_es`, `_hi`, …). Pick it as Compare "
                "Against before every coaching session.\n"
                "• **Use case B — rehearsal.** Enroll attempt #1 as "
                "`pitch_attempt_1`. Tomorrow record attempt #2, Compare "
                "Against `pitch_attempt_1`, Detect — see what changed.\n\n"
                "Each baseline is the median + spread of the take's "
                "frames, stored locally."
            ),
        ):
            base = fit_baseline(ss.samples, std_floor=_LIVE_STD_FLOOR)
            if base is None:
                st.error("Need more samples first.")
            elif not new_name.strip():
                st.error("Type a name.")
            else:
                ss.signatures[new_name.strip()] = base
                save_signatures(ss.signatures)
                ss.last_enrolled = {
                    "name": new_name.strip(),
                    "n_samples": base.n_samples,
                    "duration_s": float(base.duration_s),
                    "at": time.time(),
                }
                # Toast persists across the upcoming st.rerun(), unlike
                # st.success() which gets wiped when the script restarts.
                st.toast(
                    f"✅ Enrolled '{new_name.strip()}' "
                    f"(n={base.n_samples}, "
                    f"duration={base.duration_s:.1f}s)",
                    icon="💾",
                )
                # Re-run immediately so the new baseline shows up in the
                # Compare-against selectbox and the Next-step guidance
                # panel reflects the new state on this same click.
                st.rerun()

        # Sticky 'last enrolled' indicator so the user can see at a glance
        # which baseline they most recently saved (toasts disappear after
        # a few seconds; the sticky badge stays).
        if ss.last_enrolled:
            le = ss.last_enrolled
            st.markdown(
                f"<div style='margin-top:8px;padding:10px 12px;"
                f"background:#eafbf0;border:1px solid #5cb85c;"
                f"border-radius:6px;color:#0a3a18;font-size:0.85rem'>"
                f"<b>💾 Last enrolled:</b> "
                f"<code style='background:#d0f0d8;padding:1px 5px;"
                f"border-radius:3px;color:#0a3a18'>{le['name']}</code>"
                f"<br/><span style='color:#2a6a3a;font-size:0.78rem'>"
                f"{le['n_samples']} samples · "
                f"{le['duration_s']:.1f} s · saved to signatures.json"
                f"</span></div>",
                unsafe_allow_html=True,
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
            help=(
                "Compare the current take against the baseline selected "
                "in Compare Against above. After it runs, the Mobile and "
                "Insights tabs show the ranked NUANCES where the two "
                "differ — coloured by category (real face change / head "
                "movement / setup drift), with side-by-side trapezium "
                "overlays, spider chart of all dimensions, and a timeline "
                "of the dominant feature. Use case A: see where you "
                "departed from rest. Use case B: see where today's "
                "attempt differs from yesterday's enrolled version."
            ),
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
                # Voice-trigger trim — drop the head seconds polluted by
                # the spoken 'record' word + the user repeating it /
                # hesitating; drop the tail seconds polluted by 'stop'.
                # Only applied when the take was actually voice-triggered.
                samples_for_detect = ss.samples
                if ss.voice_record_at and ss.start_t is not None:
                    cut_in = (ss.voice_record_at - ss.start_t) + VOICE_HEAD_TRIM_S
                    samples_for_detect = [
                        s for s in samples_for_detect if s.t >= cut_in
                    ]
                if ss.voice_stop_at and ss.start_t is not None:
                    cut_out = (ss.voice_stop_at - ss.start_t) - VOICE_TAIL_TRIM_S
                    samples_for_detect = [
                        s for s in samples_for_detect if s.t <= cut_out
                    ]
                # Fallback: if the trim left us with too few samples (e.g.
                # the user spoke 'stop' almost immediately after 'record'),
                # ignore the trim and run on the full buffer.
                if len(samples_for_detect) < 10:
                    samples_for_detect = ss.samples
                report = detect_sigma_changes(
                    samples_for_detect,
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
                                ss.transcript = transcribe(
                                    wav_path, language=language_code
                                )
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
                # Re-run immediately so the Mobile / Insights tabs and the
                # sidebar's Next-step guidance reflect the new event_log
                # on this same click.
                st.rerun()

        run_live = st.toggle("Run live", value=True)

    return {
        "camera_idx": camera_idx,
        "baseline_choice": baseline_choice,
        "adapt_on": adapt_on,
        "adapt_window_s": adapt_window_s,
        "sigma_low": sigma_low,
        "sigma_high": sigma_high,
        "run_live": run_live,
        "language_code": language_code,
        "out_lang": out_lang,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

_RECORDING_CSS = """
<style>
@keyframes syntonia_pulse {
  0%   { transform: scale(1);   opacity: 1; }
  50%  { transform: scale(1.4); opacity: 0.55; }
  100% { transform: scale(1);   opacity: 1; }
}
.syntonia-rec-dot {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #d9534f;
  margin-right: 10px;
  animation: syntonia_pulse 1s ease-in-out infinite;
  box-shadow: 0 0 10px #d9534f;
}
.syntonia-banner {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px 16px;
  border-radius: 8px;
  font-weight: 800;
  font-size: 1.05rem;
  letter-spacing: 2px;
  margin: 6px 0 12px 0;
}
.syntonia-banner.rec {
  background: #fdecea;
  color: #a94442;
  border: 2px solid #d9534f;
}
.syntonia-banner.idle {
  background: #f1f5f9;
  color: #404040;
  border: 1px solid #c8d0db;
}
</style>
"""


def _render_recording_banner() -> None:
    """Top-of-page banner that pulses red while recording, grey while idle.
    Wrapped in a fragment so the elapsed-time and frame counter update
    continuously without requiring a full page rerun on every tick.
    """
    st.markdown(_RECORDING_CSS, unsafe_allow_html=True)

    @st.fragment(run_every="250ms")
    def _banner_tick():
        ss = st.session_state
        # Voice-command flash: when the listener heard 'record' or 'stop'
        # within the last ~1.8 s, override the normal banner with an
        # amber-bordered HEARD-overlay. This is what guarantees the user
        # sees confirmation in their main focus area (the page banner),
        # rather than needing to track a toast in the top-right corner.
        try:
            listener = get_voice_listener()
            voice_age = time.time() - listener.last_event_time
            if (
                listener.last_event_command
                and listener.last_event_time > 0
                and voice_age <= 1.8
            ):
                word = listener.last_event_command
                key = "voice.heard_start" if word == "record" else "voice.heard_stop"
                text = t(key, ss.get("out_lang", "en")).format(w=word)
                st.markdown(
                    f"<div class='syntonia-banner' style='"
                    f"background:#fff7e6;color:#7a4a00;"
                    f"border:2px solid #f0ad4e'>"
                    f"<span style='font-size:1.15rem;margin-right:6px'>🎤</span>"
                    f"{text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                return
        except Exception:
            pass
        if ss.recording:
            elapsed = 0.0
            if ss.start_t is not None:
                elapsed = max(0.0, time.time() - ss.start_t)
            # During the voice-trigger head-trim window, tell the user
            # the first second is being dropped from analysis — settles
            # them and explains away the head-spike.
            settling = (
                ss.voice_record_at is not None
                and (time.time() - ss.voice_record_at) < VOICE_HEAD_TRIM_S
            )
            if settling:
                grace_remaining = max(
                    0.0, VOICE_HEAD_TRIM_S - (time.time() - ss.voice_record_at)
                )
                st.markdown(
                    f"<div class='syntonia-banner rec'>"
                    f"<span class='syntonia-rec-dot'></span>"
                    f"RECORDING · SETTLING ({grace_remaining:0.1f} s) — "
                    f"analysis starts after this — sit still"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='syntonia-banner rec'>"
                    f"<span class='syntonia-rec-dot'></span>"
                    f"RECORDING · {elapsed:0.1f} s · "
                    f"{len(ss.samples)} frames"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            n = len(ss.samples)
            if n > 0:
                dur = (ss.samples[-1].t - ss.samples[0].t) if ss.samples else 0.0
                text = (
                    f"IDLE · take captured ({n} frames · {dur:0.1f} s) — "
                    f"Detect to see nuances"
                )
            else:
                text = "IDLE · ready to record"
            st.markdown(
                f"<div class='syntonia-banner idle'>{text}</div>",
                unsafe_allow_html=True,
            )

    _banner_tick()


def _render_voice_command_poller() -> None:
    """Dedicated 200 ms fragment that polls the voice listener regardless of
    which tab is currently active. Without this, voice commands only got
    consumed when the user was on the Live tab (because the live_tick
    fragment was the only thing checking listener.consume_pending()).
    """
    @st.fragment(run_every="200ms")
    def _voice_tick():
        listener = get_voice_listener()
        # Show a toast for every command actually detected, even if it
        # didn't change the recording state (e.g. 'stop' said while not
        # recording). Gives the user clear feedback that the listener
        # IS hearing them.
        announce = listener.take_announcement()
        if announce:
            label = "🎙 Recording started" if announce == "record" else "⏹ Recording stopped"
            st.toast(f"{label}  (heard: '{announce}')", icon="🎤")
        cmd = listener.consume_pending()
        if not cmd:
            return
        _toggle_recording_via_command(cmd)
        # Full app rerun so the recording banner, sidebar button label,
        # and Live-tab webcam overlay all reflect the new state on the
        # same voice command.
        st.rerun()

    _voice_tick()


def main() -> None:
    st.set_page_config(page_title="SyntoniaPro — Live", layout="wide")
    init_state()
    sidebar = render_sidebar()
    # Stash the chosen output language so helper renderers can call t(...)
    # without threading the value through every signature.
    st.session_state.out_lang = sidebar["out_lang"]

    st.title("SyntoniaPro — Live")
    st.caption(
        "Local webcam → face geometry → behavioural insights. "
        "No cloud, no data leaves this device."
    )

    # Top-of-page recording state banner, visible from every tab so the
    # user always knows whether their voice-triggered take is rolling.
    _render_recording_banner()

    # Voice-command poller — runs regardless of active tab.
    _render_voice_command_poller()

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
