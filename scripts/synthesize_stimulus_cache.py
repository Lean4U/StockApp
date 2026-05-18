"""Build a demo DFI cache with synthetic stimulus events on top of an existing
real-video cache, so we can showcase the dashboard's breach detection in action.

Starts from the real test-video-1 cache (preserves real face / hand frames so
the dashboard can still show frame previews) and overlays three deliberate
stimulus events:

    t = 30 s   acoustic-dominant   (V spike — sudden speech-rate drop)
    t = 70 s   kinetic-dominant    (F spike — strong fidget burst)
    t = 120 s  co-firing event     (V + F + M all spike together — the
                                    classic "equilibrium-shattering" moment)

The DFI is recomputed from the modified V/F/M traces using the cache's
existing α/β/γ weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dfi import compute_dfi  # noqa: E402


def gaussian(t, t0, sigma_s, amp):
    return amp * np.exp(-0.5 * ((t - t0) / sigma_s) ** 2)


def main():
    src = REPO / "videos" / "test-video-1_dfi_cache.npz"
    dst = REPO / "videos" / "test-video-1-demo_dfi_cache.npz"
    if not src.exists():
        print(f"Source cache not found: {src}")
        return 1

    npz = np.load(src, allow_pickle=True)
    meta = json.loads(str(npz["meta_json"]))
    times = npz["times"]

    # Start from the real V/F/M.
    v_times = npz["v_times"]
    v_orig = npz["v_values"]
    f_orig = npz["f_z"]
    m_orig = npz["m"]

    # Overlay three synthetic stimulus events.
    # Event 1 (acoustic stress at t=30): big V bump, small M correlation.
    v_overlay_face = gaussian(times, 30.0, 1.5, amp=6.0)
    v_overlay = np.interp(v_times, times, v_overlay_face, left=0, right=0)
    m_overlay = gaussian(times, 30.0, 1.5, amp=1.5)
    f_overlay = gaussian(times, 30.0, 2.0, amp=0.5)

    # Event 2 (kinetic burst at t=70): big F spike, modest V, small M.
    v_overlay += np.interp(v_times, times, gaussian(times, 70.0, 1.2, amp=1.5), left=0, right=0)
    f_overlay += gaussian(times, 70.0, 1.0, amp=7.0)
    m_overlay += gaussian(times, 70.0, 1.5, amp=1.0)

    # Event 3 (co-firing stimulus at t=120 — the equilibrium-shatter): all
    # three fire together for ~3 seconds.
    v_overlay += np.interp(v_times, times, gaussian(times, 120.0, 1.5, amp=4.5), left=0, right=0)
    f_overlay += gaussian(times, 120.0, 1.5, amp=5.0)
    m_overlay += gaussian(times, 120.0, 1.5, amp=4.0)

    v_new = v_orig + v_overlay
    f_new = f_orig + f_overlay
    m_new = m_orig + m_overlay

    alpha = meta["alpha"]; beta = meta["beta"]; gamma = meta["gamma"]
    threshold = meta["threshold"]; window_s = meta["window_s"]
    rep = compute_dfi(
        times,
        v_times, v_new,
        times, f_new,
        times, m_new,
        alpha=alpha, beta=beta, gamma=gamma,
        threshold=threshold, window_s=window_s,
    )

    new_meta = dict(meta)
    new_meta["dfi_windows"] = [w.to_dict() for w in rep.windows]
    new_meta["summary"] = dict(meta["summary"])
    new_meta["summary"]["max_m"] = float(m_new.max())
    new_meta["summary"]["max_f"] = float(f_new.max())
    new_meta["summary"]["max_v"] = float(v_new.max())
    new_meta["summary"]["max_dfi"] = float(rep.dfi.max())
    new_meta["video"] = str(REPO / "videos" / "test-video-1.MOV")
    new_meta["synthetic"] = {
        "note": "Demo cache: real face/hand frames + synthetic stimulus events.",
        "events": [
            {"t": 30.0, "kind": "acoustic-dominant", "channels": "V↑ M↑ F↑(small)"},
            {"t": 70.0, "kind": "kinetic-dominant", "channels": "F↑↑ V↑ M↑"},
            {"t": 120.0, "kind": "co-firing breach", "channels": "V↑ F↑ M↑ — all together"},
        ],
    }
    # Synthetic transcript so the dashboard's caption + breach verbatim
    # behavior can be demonstrated even though the real Whisper model is
    # not downloadable in the sandbox. Real runs replace this with the
    # output of faster-whisper.
    synthetic_segments = [
        (5.0, 9.5, "Sure, I'd be happy to walk you through it."),
        (12.0, 17.0, "We met on a Tuesday afternoon, around three I think."),
        (28.0, 33.5, "Honestly, I have no idea what you're talking about."),
        (50.0, 55.0, "Yes, we spoke briefly, but only about the project."),
        (68.5, 74.0, "I might have mentioned the meeting in passing, that's all."),
        (95.0, 101.0, "I was at home that evening, just like every other Tuesday."),
        (118.5, 124.5, "I never saw that document. I have no recollection of signing it."),
        (140.0, 146.0, "Look, I've answered this question multiple times already."),
    ]
    seg_dicts = []
    for s, e, text in synthetic_segments:
        seg_dicts.append({
            "start": s, "end": e, "text": text, "words": [],
        })
    new_meta["transcript"] = {
        "language": "en", "language_probability": 0.99,
        "duration_s": float(times[-1] - times[0]) if times.size else 0.0,
        "segments": seg_dicts,
        "synthetic_demo": True,
    }

    # Save (keep all other arrays unchanged so the dashboard still has the real frames + AU clusters).
    np.savez(
        dst,
        times=times,
        face_overall_z=npz["face_overall_z"],
        face_t2_eq_sigma=npz["face_t2_eq_sigma"],
        m=m_new,
        f_z=f_new,
        f_raw=npz["f_raw"],
        v_times=v_times,
        v_values=v_new,
        dfi=rep.dfi,
        regions=npz["regions"],
        hand_state_codes=npz["hand_state_codes"],
        kinetic_per_frame=npz["kinetic_per_frame"],
        sample_frame_idx=npz["sample_frame_idx"],
        cluster_z=npz["cluster_z"],
        cluster_names=npz["cluster_names"],
        meta_json=json.dumps(new_meta),
    )

    print(f"Saved demo cache: {dst}")
    print(f"Max V: {v_new.max():.2f}σ")
    print(f"Max F: {f_new.max():.2f}σ")
    print(f"Max M: {m_new.max():.2f}σ")
    print(f"Max DFI (rolling-mean): {rep.dfi.max():.2f}")
    print(f"Threshold = {threshold:.2f}")
    print(f"Breach windows: {len(rep.windows)}")
    for w in rep.windows:
        print(f"  [{w.start_t:.2f}, {w.end_t:.2f}] s  "
              f"peak {w.peak_dfi:.2f} @ {w.peak_t:.2f}s  dom={w.dominant_component}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
