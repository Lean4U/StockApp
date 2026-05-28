"""SyntoniaPro — Real-Time Analytics Dashboard.

Synchronized playback of voice (V), fidget (F), and facial-mesh (M) channels
plus the unified Syntonia Model (Syntonia). When the Syntonia score crosses the
configured threshold inside a 5-second rolling window, the dashboard surfaces
the *exact moment a stimulus shatters the subject's baseline equilibrium*:

  * a banner at the breach time,
  * a text response with the threshold value (e.g.
    `"Syntonia = 3.42 > 3.00 at t = 47.23s — dominant: M (face), feature
    yaw_proxy peak 12.4σ"`),
  * a synchronized current-frame video preview with the trapezium and hand
    landmarks overlaid,
  * all three channel timelines on a shared axis with a moving time cursor.

Two source modes:

  * **Replay** — load a video file. Run ``precompute_syntonia.py`` first to build
    the per-frame cache; the dashboard reads that cache instantly. Works
    everywhere, deterministic.
  * **Live** — same dashboard fed by a webcam + microphone stream. Designed
    for; activated only when you set ``source = "live"`` on a host with a
    camera, mic and audio drivers.

Run (replay):

    python scripts/precompute_syntonia.py videos/IMG_5034.MOV
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import time as _time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from syntonia_model import SYNTONIA_DISCLAIMER
from voice_transcript import TranscriptResult


REPO = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_cache(cache_path: str) -> dict:
    npz = np.load(cache_path, allow_pickle=True)
    meta = json.loads(str(npz["meta_json"]))
    return {
        "times": npz["times"],
        "face_overall_z": npz["face_overall_z"],
        "face_t2_eq_sigma": npz["face_t2_eq_sigma"],
        "m": npz["m"],
        "f_z": npz["f_z"],
        "v_times": npz["v_times"],
        "v_values": npz["v_values"],
        "syntonia": npz["syntonia"],
        "regions": npz["regions"],
        "hand_state_codes": npz["hand_state_codes"],
        "kinetic_per_frame": npz["kinetic_per_frame"],
        "sample_frame_idx": npz["sample_frame_idx"],
        "cluster_z": npz["cluster_z"],
        "cluster_names": npz["cluster_names"],
        "meta": meta,
    }


@st.cache_resource
def open_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    return cap


def _transcript_from_meta(meta: dict) -> Optional[TranscriptResult]:
    """Reconstruct a TranscriptResult from a cache's meta blob, if present
    and well-formed (i.e. not an error dict from a missing-model run)."""
    tr = meta.get("transcript")
    if not isinstance(tr, dict) or "segments" not in tr:
        return None
    try:
        return TranscriptResult.from_dict(tr)
    except Exception:
        return None


def read_frame_at(cap, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SyntoniaPro — Real-Time Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.markdown(
        "<h1 style='margin-bottom:0'>SyntoniaPro — Real-Time Analytics</h1>"
        "<div style='color:gray;font-size:0.9rem'>"
        "Synchronized voice + fidget + facial-mesh friction monitoring</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Source")
        source = st.radio("Mode", ["Replay (video file)", "Live (webcam + mic)"],
                          index=0)
        if source.startswith("Live"):
            st.warning(
                "Live mode requires a webcam, microphone and the host-level "
                "permissions to access them — not available in this sandbox. "
                "Run `streamlit run dashboard.py` on a local machine."
            )
            st.stop()

        # Find available caches
        video_dir = REPO / "videos"
        caches = sorted(video_dir.glob("*_syntonia_cache.npz"))
        if not caches:
            st.error(
                "No Syntonia caches found in `videos/`. Run:\n\n"
                "    python scripts/precompute_syntonia.py videos/<your.mp4>"
            )
            st.stop()

        cache_choice = st.selectbox(
            "Cache",
            options=caches,
            format_func=lambda p: p.name,
        )
        st.divider()
        st.header("Thresholds")
        live_threshold = st.slider("Syntonia threshold", 1.0, 6.0, 3.0, 0.1)
        st.caption("Threshold used for the alarm banner. The cache stores the value of α, β, γ and the rolling-window length the Syntonia(t) curve was built with.")
        st.divider()
        st.header("Playback")
        play_speed = st.select_slider(
            "Speed", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0,
        )
        auto_play = st.toggle("Auto-play")

        st.divider()
        st.header("Privacy")
        st.caption(
            "All processing runs locally on this host. Models (MediaPipe "
            "face/hand, Whisper) are loaded from `models/` — once fetched, "
            "no audio, video, or derived data leaves the machine. Streamlit "
            "telemetry is disabled via `.streamlit/config.toml`."
        )
        delete_target = st.selectbox(
            "Delete which artifacts?",
            options=["(select)", "current cache only", "all caches + plots + overlays"],
        )
        if st.button("Delete now", type="secondary"):
            removed = []
            video_dir = REPO / "videos"
            if delete_target == "current cache only":
                p = Path(cache_choice)
                if p.exists():
                    p.unlink()
                    removed.append(p.name)
            elif delete_target == "all caches + plots + overlays":
                patterns = [
                    "*_syntonia_cache.npz", "*_syntonia_report.png", "*_syntonia_events.json",
                    "*_syntonia_overlay.mp4", "dashboard_*.png", "*_report.png",
                    "frame_*.jpg", "hands_*.jpg",
                ]
                for pat in patterns:
                    for p in video_dir.glob(pat):
                        p.unlink()
                        removed.append(p.name)
            if removed:
                st.success(f"Deleted {len(removed)} file(s): {', '.join(removed[:6])}"
                           + (" …" if len(removed) > 6 else ""))
                load_cache.clear()
                st.rerun()
            else:
                st.info("Nothing matched.")

    data = load_cache(str(cache_choice))
    meta = data["meta"]
    transcript: Optional[TranscriptResult] = _transcript_from_meta(meta)
    video_path = meta["video"]
    cap = open_video(video_path)
    if cap is None:
        st.error(f"Could not open source video at {video_path}")
        st.stop()

    times = data["times"]
    duration = float(times[-1] - times[0]) if times.size else 0.0
    fps = float(meta["fps"])
    frame_idxs = data["sample_frame_idx"]
    syntonia = data["syntonia"]

    # ---- Time control --------------------------------------------------------
    if "playhead_t" not in st.session_state:
        st.session_state["playhead_t"] = float(times[0])

    if auto_play:
        # Advance playhead by one face-sample step on each rerun.
        # Speed multiplier scales how many seconds to advance per rerun tick.
        next_t = st.session_state["playhead_t"] + (1.0 / fps) * play_speed
        if next_t >= float(times[-1]):
            next_t = float(times[0])
        st.session_state["playhead_t"] = next_t

    playhead_t = st.slider(
        "Time (s)",
        min_value=float(times[0]),
        max_value=float(times[-1]),
        value=float(st.session_state["playhead_t"]),
        step=float(1.0 / fps),
        key="playhead_slider",
    )
    st.session_state["playhead_t"] = playhead_t

    # Resolve playhead to nearest face-sample index.
    cursor_idx = int(np.argmin(np.abs(times - playhead_t)))
    cursor_t = float(times[cursor_idx])

    # ---- Top row: video frame + current readout -----------------------------
    col_video, col_state = st.columns([3, 2])

    with col_video:
        frame = read_frame_at(cap, int(frame_idxs[cursor_idx]))
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Shrink portrait frames for display.
            h, w = rgb.shape[:2]
            max_w = 480
            if w > max_w:
                scale = max_w / w
                rgb = cv2.resize(rgb, (max_w, int(h * scale)))
            st.image(rgb, caption=f"t = {cursor_t:.2f} s")
        else:
            st.warning("Could not read frame")

        # Live caption strip — what the subject is saying *right now*.
        if transcript is not None:
            cur = transcript.segment_at(cursor_t)
            if cur is not None and cur.text.strip():
                st.markdown(
                    "<div style='padding:10px 14px;background:#101820;"
                    "border-left:4px solid #4dabf7;border-radius:4px;"
                    "color:#dde6f1;font-size:1.05rem;font-style:italic'>"
                    f"“{cur.text.strip()}”"
                    f"<div style='font-size:0.75rem;color:#7891a8;margin-top:4px'>"
                    f"{cur.start:.2f}s — {cur.end:.2f}s "
                    f"·   {transcript.language} "
                    f"({transcript.language_probability:.0%} conf.)"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("(silence)")
        else:
            tr_meta = meta.get("transcript")
            if isinstance(tr_meta, dict) and "error" in tr_meta:
                st.caption(f"⚠ Transcript unavailable: {tr_meta['error']}")
            else:
                st.caption("Transcript: not enabled (run with ASR to populate).")

    with col_state:
        v_at = float(np.interp(cursor_t, data["v_times"], data["v_values"]))
        f_at = float(data["f_z"][cursor_idx])
        m_at = float(data["m"][cursor_idx])
        syntonia_at = float(syntonia_arr[cursor_idx])

        status_color = (
            "#d9534f" if syntonia_at >= live_threshold else
            "#f0ad4e" if syntonia_at >= live_threshold * 0.66 else "#5cb85c"
        )
        st.markdown(
            f"<div style='padding:18px;background:#0f1116;border-left:6px solid {status_color};border-radius:6px'>"
            f"<div style='color:{status_color};font-size:0.85rem;font-weight:bold'>"
            f"{'⚠ BREACH' if syntonia_at >= live_threshold else '— stable —'}</div>"
            f"<div style='font-size:2.5rem;font-weight:bold;color:white'>"
            f"Syntonia = {syntonia_at:.2f}</div>"
            f"<div style='color:#9aa0a6'>threshold = {live_threshold:.2f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("V (voice)", f"{v_at:.2f} σ")
        c2.metric("F (fidget)", f"{f_at:.2f} σ")
        c3.metric("M (face)", f"{m_at:.2f} σ")

        # Hand state code mapping
        hand_state_names = {
            -1: "hidden", 0: "A (clasp/still)", 1: "B (clasp/moving)",
            2: "C (apart/still)", 3: "D (apart/moving)", 4: "single hand",
        }
        hsc = int(data["hand_state_codes"][cursor_idx])
        st.caption(
            f"Hand state: **{hand_state_names.get(hsc, '?')}** | "
            f"region: **{data['regions'][cursor_idx]}**"
        )

    # ---- Synchronized channel timeline --------------------------------------
    st.markdown("### Synchronized channel view")
    # Resample V onto face timeline so all curves share the same x-axis.
    v_resampled = np.interp(times, data["v_times"], data["v_values"])
    chart_df = pd.DataFrame(
        {
            "t": times,
            "V (voice)": v_resampled,
            "F (fidget)": data["f_z"],
            "M (face)": data["m"],
            "Syntonia": syntonia,
        }
    ).set_index("t")

    # Use Altair so we can overlay the threshold and the playhead.
    import altair as alt
    long = chart_df.reset_index().melt("t", var_name="channel", value_name="value")
    cursor_df = pd.DataFrame({"t": [cursor_t]})
    threshold_df = pd.DataFrame({"v": [live_threshold]})

    base = alt.Chart(long).encode(x=alt.X("t", title="time (s)"))
    line = base.mark_line(strokeWidth=1.2).encode(
        y=alt.Y("value", title="σ"),
        color=alt.Color(
            "channel",
            scale=alt.Scale(
                domain=["V (voice)", "F (fidget)", "M (face)", "Syntonia"],
                range=["#a0522d", "#8a2be2", "#2e8b57", "#000000"],
            ),
        ),
        tooltip=["t:Q", "channel:N", "value:Q"],
    )
    cursor = alt.Chart(cursor_df).mark_rule(color="red", strokeWidth=1).encode(x="t:Q")
    thr_rule = alt.Chart(threshold_df).mark_rule(
        color="red", strokeDash=[4, 4], strokeWidth=1.5
    ).encode(y="v:Q")
    chart = (line + thr_rule + cursor).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    # ---- AU cluster heatmap -------------------------------------------------
    with st.expander("AU cluster breakdown (M channel)"):
        cluster_z = data["cluster_z"]
        cluster_names = list(data["cluster_names"])
        df_long = []
        for j, cn in enumerate(cluster_names):
            for i, t in enumerate(times):
                df_long.append({"t": float(t), "cluster": cn,
                                "z": float(cluster_z[i, j])})
        df_clu = pd.DataFrame(df_long)
        heat = alt.Chart(df_clu).mark_rect().encode(
            x=alt.X("t:Q", title="time (s)"),
            y=alt.Y("cluster:N", sort=cluster_names),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="reds")),
            tooltip=["t:Q", "cluster:N", "z:Q"],
        ).properties(height=160)
        playhead = alt.Chart(cursor_df).mark_rule(color="blue", strokeWidth=1).encode(x="t:Q")
        st.altair_chart(heat + playhead, use_container_width=True)

    # ---- Event log: equilibrium breaches as text ---------------------------
    st.markdown("### ⚡ Equilibrium-shatter events")
    syntonia_windows = meta.get("syntonia_windows", [])
    # Re-threshold against the live-tuned threshold (the cache stored a fixed one)
    live_breaches = [
        i for i in range(len(times)) if syntonia_arr[i] >= live_threshold
    ]
    if not live_breaches and not syntonia_windows:
        st.success(
            f"No Syntonia threshold breaches over {duration:.1f} s of recording. "
            f"Max Syntonia = **{float(syntonia.max()):.2f}** (threshold = {live_threshold:.2f})."
        )
    else:
        # Group consecutive breach indices into windows.
        groups: list[tuple[int, int]] = []
        if live_breaches:
            start = live_breaches[0]
            prev = start
            for i in live_breaches[1:]:
                if i == prev + 1:
                    prev = i
                else:
                    groups.append((start, prev))
                    start = i
                    prev = i
            groups.append((start, prev))

        for si, ei in groups:
            peak_local = si + int(np.argmax(syntonia_arr[si:ei + 1]))
            peak_t = float(times[peak_local])
            peak_syntonia = float(syntonia_arr[peak_local])
            v_p = float(v_resampled[peak_local])
            f_p = float(data["f_z"][peak_local])
            m_p = float(data["m"][peak_local])
            contribs = {"V": meta["alpha"] * v_p,
                        "F": meta["beta"] * f_p,
                        "M": meta["gamma"] * m_p}
            dom = max(contribs, key=contribs.get)
            # Find dominant face feature near peak from face_events
            dom_feature_text = ""
            for fe in meta.get("face_events", []):
                if fe["start_t"] <= peak_t <= fe["end_t"] and fe["sigma_level"] >= 3:
                    dom_feature_text = (
                        f" — face peak feature **{fe['dominant_feature']}** "
                        f"({fe['peak_z']:.1f}σ at t={fe['peak_t']:.2f}s)"
                    )
                    break

            # Verbatim transcript spanning the breach window (if available).
            spoken_text = ""
            if transcript is not None:
                spoken_text = transcript.text_within(
                    float(times[si]), float(times[ei]),
                ).strip()
            verbatim_html = ""
            if spoken_text:
                verbatim_html = (
                    "<div style='margin-top:10px;padding:10px;background:#1a0a0a;"
                    "border-left:3px solid #ffb86b;border-radius:3px;"
                    "color:#ffd9a8;font-style:italic;font-size:0.95rem'>"
                    "<span style='color:#ffb86b;font-style:normal;"
                    "font-weight:bold;font-size:0.8rem'>VERBATIM RESPONSE — </span>"
                    f"“{spoken_text}”"
                    "</div>"
                )
            elif transcript is not None:
                verbatim_html = (
                    "<div style='margin-top:8px;color:#888;font-size:0.85rem'>"
                    "(no speech detected in the breach window)</div>"
                )

            st.markdown(
                f"<div style='padding:12px;background:#2a1212;border-left:5px solid #d9534f;border-radius:4px;margin-bottom:8px'>"
                f"<b style='color:#ff6b6b'>⚠ Syntonia = {peak_syntonia:.2f} > {live_threshold:.2f}</b>"
                f" at <b>t = {peak_t:.2f}s</b> "
                f"(window {float(times[si]):.2f}–{float(times[ei]):.2f}s, "
                f"{float(times[ei] - times[si]):.2f}s)<br>"
                f"Dominant channel: <b>{dom}</b> "
                f"(V={contribs['V']:+.2f}, F={contribs['F']:+.2f}, "
                f"M={contribs['M']:+.2f}){dom_feature_text}"
                f"{verbatim_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ---- Single-channel detector events (face) -----------------------------
    with st.expander(
        f"Face channel events ({len([e for e in meta.get('face_events', []) if e['sigma_level'] >= 3])}"
        " ≥3σ events)"
    ):
        face_events = meta.get("face_events", [])
        if face_events:
            df = pd.DataFrame(face_events)
            df = df[["detector", "sigma_level", "start_t", "end_t",
                     "duration_s", "peak_z", "peak_t", "dominant_feature"]]
            df.columns = ["detector", "level (σ)", "start (s)", "end (s)",
                          "dur (s)", "peak (σ)", "peak t (s)", "dominant feature"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No face-channel events.")

    # ---- Disclaimer ---------------------------------------------------------
    st.markdown("---")
    st.caption(f"**Interpretation note.** {SYNTONIA_DISCLAIMER}")

    # ---- Auto-play rerun ---------------------------------------------------
    if auto_play:
        # Sleep one frame interval (in real time) before rerunning.
        _time.sleep(max(1.0 / (fps * play_speed), 0.01))
        st.rerun()


if __name__ == "__main__":
    main()
