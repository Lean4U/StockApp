"""Render dashboard panels as static images (same data, same look as the
Streamlit dashboard; bypasses the browser since the sandbox cannot run one).

Produces:
    videos/dashboard_01_baseline.png  — playhead at t=15s, channels all near 0
    videos/dashboard_02_event_70s.png — playhead at the kinetic-burst at t=70s
    videos/dashboard_03_breach.png    — playhead at the equilibrium-shatter
                                        at t=122s where DFI > 3 and the event
                                        log card surfaces the breach
    videos/dashboard_04_timeline.png  — full V/F/M/DFI timeline with breach
                                        window + AU heatmap
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from voice_transcript import TranscriptResult  # noqa: E402

CACHE_PATH = REPO / "videos" / "test-video-1-demo_dfi_cache.npz"
VIDEO_PATH = REPO / "videos" / "test-video-1.MOV"


def _bg_dark(ax):
    ax.set_facecolor("#0f1116")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.tick_params(colors="#aaa")
    ax.yaxis.label.set_color("#ddd")
    ax.xaxis.label.set_color("#ddd")
    ax.title.set_color("#ddd")


def load() -> dict:
    npz = np.load(CACHE_PATH, allow_pickle=True)
    meta = json.loads(str(npz["meta_json"]))
    transcript = None
    tr_meta = meta.get("transcript")
    if isinstance(tr_meta, dict) and "segments" in tr_meta:
        try:
            transcript = TranscriptResult.from_dict(tr_meta)
        except Exception:
            transcript = None
    return {
        "times": npz["times"],
        "m": npz["m"],
        "f_z": npz["f_z"],
        "v_times": npz["v_times"],
        "v_values": npz["v_values"],
        "dfi": npz["dfi"],
        "regions": npz["regions"],
        "hand_state_codes": npz["hand_state_codes"],
        "sample_frame_idx": npz["sample_frame_idx"],
        "cluster_z": npz["cluster_z"],
        "cluster_names": [str(s) for s in npz["cluster_names"]],
        "meta": meta,
        "transcript": transcript,
    }


def read_video_frame(idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((400, 300, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = 400.0 / w
    frame = cv2.resize(frame, (400, int(h * scale)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_panel(out_path: Path, data: dict, playhead_t: float, title: str):
    """One dashboard-style page snapshot showing:
       - video preview (left)
       - state panel (right)
       - synchronized V/F/M/DFI chart (middle row)
       - AU cluster heatmap (bottom)
    """
    times = data["times"]
    cursor_idx = int(np.argmin(np.abs(times - playhead_t)))
    cursor_t = float(times[cursor_idx])
    v_resampled = np.interp(times, data["v_times"], data["v_values"])
    threshold = data["meta"]["threshold"]

    v_at = float(v_resampled[cursor_idx])
    f_at = float(data["f_z"][cursor_idx])
    m_at = float(data["m"][cursor_idx])
    dfi_at = float(data["dfi"][cursor_idx])
    in_breach = dfi_at >= threshold

    fig = plt.figure(figsize=(16, 11), facecolor="#0a0c10")
    gs = GridSpec(3, 4, height_ratios=[1.2, 1.5, 1.0], hspace=0.45, wspace=0.30,
                  left=0.05, right=0.97, top=0.93, bottom=0.07)

    fig.suptitle(
        "DFI Real-Time Analytics — " + title,
        color="#e0e0e0", fontsize=18, fontweight="bold", y=0.97,
    )

    # ---- Video frame ---------------------------------------------------------
    ax_video = fig.add_subplot(gs[0, :2])
    ax_video.set_facecolor("#0f1116")
    frame_idx = int(data["sample_frame_idx"][cursor_idx])
    img = read_video_frame(frame_idx)
    ax_video.imshow(img)
    ax_video.set_xticks([]); ax_video.set_yticks([])
    title = f"Subject — t = {cursor_t:.2f}s"
    # Live caption (current spoken segment, if any).
    transcript: Optional[TranscriptResult] = data.get("transcript")
    caption_text = ""
    if transcript is not None:
        seg = transcript.segment_at(cursor_t)
        if seg is not None:
            caption_text = f'  "{seg.text.strip()}"'
    ax_video.set_title(title + caption_text, color="#9cd0f0",
                       fontsize=10, loc="left", style="italic")
    for sp in ax_video.spines.values():
        sp.set_color("#333")

    # ---- State panel ---------------------------------------------------------
    ax_state = fig.add_subplot(gs[0, 2:])
    ax_state.set_facecolor("#0f1116")
    ax_state.set_xticks([]); ax_state.set_yticks([])
    for sp in ax_state.spines.values():
        sp.set_color("#333")

    breach_color = "#d9534f" if in_breach else (
        "#f0ad4e" if dfi_at >= threshold * 0.66 else "#5cb85c")
    label = "⚠ BREACH" if in_breach else (
        "— elevated —" if dfi_at >= threshold * 0.66 else "— stable —")
    ax_state.barh([0.85], [1.0], color=breach_color, height=0.02,
                  transform=ax_state.transAxes)
    ax_state.text(0.05, 0.78, label, transform=ax_state.transAxes,
                  color=breach_color, fontsize=14, fontweight="bold")
    ax_state.text(0.05, 0.50, f"DFI = {dfi_at:.2f}", transform=ax_state.transAxes,
                  color="white", fontsize=44, fontweight="bold")
    ax_state.text(0.05, 0.38, f"threshold = {threshold:.2f}",
                  transform=ax_state.transAxes, color="#9aa0a6", fontsize=12)
    # V/F/M sub-readouts.
    for i, (name, val, color) in enumerate([
        ("V (voice)", v_at, "#cd853f"),
        ("F (fidget)", f_at, "#9966cc"),
        ("M (face)", m_at, "#3cb371"),
    ]):
        x = 0.05 + i * 0.32
        ax_state.text(x, 0.20, name, transform=ax_state.transAxes,
                      color="#aaa", fontsize=10)
        ax_state.text(x, 0.10, f"{val:.2f}σ", transform=ax_state.transAxes,
                      color=color, fontsize=20, fontweight="bold")
    # Hand state badge.
    hsc = int(data["hand_state_codes"][cursor_idx])
    hand_names = {-1: "hidden", 0: "A clasp/still", 1: "B clasp/moving",
                  2: "C apart/still", 3: "D apart/moving", 4: "one hand"}
    region = str(data["regions"][cursor_idx])
    ax_state.text(0.55, 0.78,
                  f"hand: {hand_names.get(hsc, '?')}", transform=ax_state.transAxes,
                  color="#c0e0c0", fontsize=11)
    ax_state.text(0.55, 0.74,
                  f"region: {region}", transform=ax_state.transAxes,
                  color="#c0e0c0", fontsize=10)

    # ---- Synchronized timeline ----------------------------------------------
    ax_t = fig.add_subplot(gs[1, :])
    _bg_dark(ax_t)
    ax_t.plot(times, v_resampled, color="#cd853f", lw=1.2, label="V (voice)")
    ax_t.plot(times, data["f_z"], color="#9966cc", lw=1.2, label="F (fidget)")
    ax_t.plot(times, data["m"], color="#3cb371", lw=1.2, label="M (face)")
    ax_t.plot(times, data["dfi"], color="white", lw=1.8, label="DFI(t)")
    ax_t.axhline(threshold, color="#d9534f", ls="--", lw=1.5,
                 label=f"threshold = {threshold:.1f}")
    # Mark each breach window from meta.
    for w in data["meta"].get("dfi_windows", []):
        ax_t.axvspan(w["start_t"], w["end_t"], alpha=0.20, color="#d9534f")
    ax_t.axvline(cursor_t, color="#fff200", lw=1.2, alpha=0.9)
    ax_t.set_xlim(times[0], times[-1])
    ax_t.set_ylabel("σ-equivalent")
    ax_t.set_xlabel("time (s)")
    ax_t.set_title("Synchronized channels — V (acoustic) + F (kinetic) + M (head/face/hands)",
                   fontsize=11)
    ax_t.legend(loc="upper left", fontsize=9, framealpha=0.5)
    ax_t.grid(alpha=0.15)

    # ---- AU cluster heatmap -------------------------------------------------
    ax_au = fig.add_subplot(gs[2, :])
    _bg_dark(ax_au)
    cz = data["cluster_z"].T  # (n_clusters, n_frames)
    im = ax_au.imshow(
        cz, aspect="auto",
        extent=[float(times[0]), float(times[-1]), len(data["cluster_names"]), 0],
        cmap="Reds", vmin=0, vmax=max(2.0, cz.max()), interpolation="nearest",
    )
    ax_au.set_yticks(np.arange(len(data["cluster_names"])) + 0.5)
    ax_au.set_yticklabels(data["cluster_names"], fontsize=8)
    ax_au.set_xlabel("time (s)")
    ax_au.set_title("M(t) cluster decomposition — FACS-AU groupings", fontsize=11)
    ax_au.axvline(cursor_t, color="#fff200", lw=1.0, alpha=0.8)

    fig.savefig(out_path, dpi=130, facecolor="#0a0c10")
    plt.close(fig)


def render_event_log(out_path: Path, data: dict):
    """Plain text-card rendering of the breach event log."""
    windows = data["meta"].get("dfi_windows", [])
    threshold = data["meta"]["threshold"]
    transcript: Optional[TranscriptResult] = data.get("transcript")
    fig = plt.figure(figsize=(16, max(3, 2.5 + 2.4 * max(len(windows), 1))),
                     facecolor="#0a0c10")
    fig.suptitle("⚡ Equilibrium-Shatter Event Log",
                 color="white", fontsize=18, fontweight="bold", x=0.05,
                 horizontalalignment="left", y=0.95)
    if not windows:
        ax = fig.add_axes([0.05, 0.1, 0.9, 0.7])
        ax.set_facecolor("#1a3a1a")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.5,
                f"No DFI threshold breaches detected.\n"
                f"Max DFI = {data['dfi'].max():.2f} (threshold = {threshold:.2f}).",
                color="#90ee90", fontsize=14, fontweight="bold",
                verticalalignment="center")
    else:
        y0 = 0.90
        h_each = min(0.26, 0.90 / len(windows))
        v_resampled = np.interp(data["times"], data["v_times"], data["v_values"])
        alpha = data["meta"]["alpha"]
        beta = data["meta"]["beta"]
        gamma = data["meta"]["gamma"]
        for i, w in enumerate(windows):
            y = y0 - (i + 1) * (h_each + 0.03)
            ax = fig.add_axes([0.05, y, 0.9, h_each])
            ax.set_facecolor("#2a1212")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#d9534f")
                sp.set_linewidth(3)
            peak_idx = int(np.argmin(np.abs(data["times"] - w["peak_t"])))
            v_p = float(v_resampled[peak_idx])
            f_p = float(data["f_z"][peak_idx])
            m_p = float(data["m"][peak_idx])
            line1 = (
                f"⚠ DFI = {w['peak_dfi']:.2f}  >  threshold {threshold:.2f}     "
                f"at t = {w['peak_t']:.2f}s"
            )
            ax.text(0.02, 0.84, line1, transform=ax.transAxes,
                    color="#ff6b6b", fontsize=16, fontweight="bold")
            line2 = (
                f"window: {w['start_t']:.2f}s — {w['end_t']:.2f}s  "
                f"({w['duration_s']:.2f}s)   "
                f"dominant channel: {w['dominant_component']}"
            )
            ax.text(0.02, 0.66, line2, transform=ax.transAxes,
                    color="#ffd0d0", fontsize=11)
            line3 = (
                f"channel contributions:  α·V = {alpha*v_p:+.2f}     "
                f"β·F = {beta*f_p:+.2f}     γ·M = {gamma*m_p:+.2f}"
            )
            ax.text(0.02, 0.50, line3, transform=ax.transAxes,
                    color="#ffe0e0", fontsize=11, family="monospace")
            # Verbatim transcript in the breach window — the actual response.
            verbatim = ""
            if transcript is not None:
                verbatim = transcript.text_within(
                    float(w["start_t"]), float(w["end_t"]),
                ).strip()
            if verbatim:
                ax.text(0.02, 0.30,
                        "VERBATIM RESPONSE", transform=ax.transAxes,
                        color="#ffb86b", fontsize=9, fontweight="bold")
                ax.text(0.02, 0.12,
                        f'"{verbatim}"', transform=ax.transAxes,
                        color="#ffd9a8", fontsize=13, style="italic",
                        wrap=True)
            elif transcript is not None:
                ax.text(0.02, 0.20,
                        "(no speech in this window)", transform=ax.transAxes,
                        color="#888", fontsize=10)

    fig.savefig(out_path, dpi=130, facecolor="#0a0c10")
    plt.close(fig)


def main():
    if not CACHE_PATH.exists():
        print(f"Cache not found at {CACHE_PATH}")
        return 1
    data = load()
    out = REPO / "videos"

    # Baseline frame.
    render_panel(out / "dashboard_01_baseline.png", data, playhead_t=7.0,
                 title="baseline window (t = 7.0 s)")
    # Event 2 — kinetic-dominant burst.
    render_panel(out / "dashboard_02_kinetic_70s.png", data, playhead_t=70.0,
                 title="kinetic burst (t = 70.0 s)  —  fidget dominates")
    # Event 3 — co-firing breach.
    render_panel(out / "dashboard_03_breach_122s.png", data, playhead_t=122.31,
                 title="EQUILIBRIUM-SHATTER (t = 122.31 s)  —  DFI breach")
    # Event log card.
    render_event_log(out / "dashboard_04_event_log.png", data)
    print("Rendered:")
    for f in sorted((REPO / "videos").glob("dashboard_*.png")):
        print(f"  {f.relative_to(REPO)}  ({f.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
