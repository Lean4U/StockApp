# Privacy & Data Flow

This document inventories **what data the Syntonia score pipeline touches, where it
lives, and which network calls it makes** so the threat model is explicit
and re-checkable. It exists because the system processes biometric
information (video, audio, facial / hand landmarks, derived behavioral
indicators) that the user has explicitly required must not leave the
host.

## Summary

All inference — face landmarks, hand landmarks, speech-to-text, syllable
rate, the Syntonia score itself — runs **locally on the host CPU**. No SaaS / cloud
ML / proxy ASR is used. Model files are downloaded *once* from public
CDNs and cached under `models/`. After that the entire pipeline can run
with the host firewalled.

## Stack inventory

| Component                | Where it runs        | Network at runtime | Notes |
|---|---|---|---|
| Video read (OpenCV)      | local                | none               | reads the file from disk |
| Audio extraction (ffmpeg)| local                | none               | mono-16k WAV to a tempfile, deleted after use |
| Face landmarks (MediaPipe Tasks `FaceLandmarker`) | local TFLite | **only** the one-time model download | model: `models/face_landmarker.task`, ~3.6 MB |
| Hand landmarks (MediaPipe Tasks `HandLandmarker`) | local TFLite | **only** the one-time model download | model: `models/hand_landmarker.task`, ~7.5 MB |
| Syllable-rate / VAD (librosa) | local           | none               | pure DSP, no model fetch |
| Speech transcript (faster-whisper, CTranslate2)   | local int8 | **only** the one-time model download | model: `models/whisper-small/`, ~250 MB |
| Syntonia assembly + change detectors | local         | none               | numpy only |
| Streamlit dashboard      | local (127.0.0.1)    | none*              | * `[browser] gatherUsageStats = false` in `.streamlit/config.toml` disables Streamlit's opt-in analytics |
| Plots (matplotlib / altair) | local             | none               | rendered to PNG / in-browser SVG |

\* On first launch Streamlit also opens a browser to `http://localhost:PORT/`
on the same machine. No external network calls.

## What we do **not** use

- No cloud transcription API (OpenAI Whisper API, Google Speech, AWS
  Transcribe, AssemblyAI, Deepgram, Otter, Rev). All ASR is the on-device
  `faster-whisper` model.
- No cloud face-API or commercial deception-analytics service.
- No analytics / telemetry beacon from this codebase.
- No third-party CDN at runtime for inference data — the only CDNs are
  contacted *once*, by the model-fetch script, to download the model
  weight files listed above.

## One-time network actions (model fetch)

The script `scripts/fetch_models.sh` makes exactly three outbound
requests, on the machine where you first install the project:

1. `storage.googleapis.com` — MediaPipe `face_landmarker.task`
2. `storage.googleapis.com` — MediaPipe `hand_landmarker.task`
3. `huggingface.co` — `Systran/faster-whisper-small` snapshot

After this completes, `models/` contains everything needed. The host
network may be torn down or firewalled, and `precompute_syntonia.py` +
`dashboard.py` keep working. Re-staging is only necessary if the model
files are deleted.

If you cannot reach those URLs at all (e.g. an air-gapped deployment),
run the fetch script on any internet-capable machine, then copy the
populated `models/` directory across.

## Local artifacts written to disk

- `videos/<stem>_dfi_cache.npz` — per-frame V/F/M/Syntonia traces, cluster
  decomposition, hand state codes, and the embedded transcript (if
  ASR enabled). **Contains derived biometric data.**
- `videos/<stem>_dfi_report.png` — a 6-panel static summary plot.
- `videos/<stem>_dfi_events.json` — per-event JSON record.
- `videos/<stem>_dfi_overlay.mp4` (optional) — the source video with
  face / hand landmarks burned in. **Contains the original biometric
  video.**
- `models/...` — model weights (no user data).
- Tempfile WAVs during voice extraction — deleted immediately after the
  transcribe call returns; failure paths also delete them.

All of these live entirely under the repository root. To wipe them:

```bash
rm -rf videos/*_dfi_cache.npz \
       videos/*_dfi_report.png \
       videos/*_dfi_events.json \
       videos/*_dfi_overlay.mp4 \
       videos/dashboard_*.png
```

The dashboard sidebar exposes a "Delete artifacts" button that performs
the same wipe interactively.

## Ephemeral / no-disk mode

`scripts/precompute_syntonia.py` accepts `--ephemeral`, which runs the full
pipeline in memory and prints a summary without writing the `.npz`
cache, report PNG, or events JSON. The dashboard cannot use an ephemeral
run directly (it reads from cache files), but the flag is useful for
one-shot CLI processing where no biometric data should ever hit disk.

## Host-level concerns this app cannot control

The pipeline keeps everything local, but the *host* still needs care:

- The `videos/` directory should sit on an encrypted volume.
- Exclude `videos/` from any cloud-sync (iCloud Drive, OneDrive,
  Dropbox, Google Drive, Backblaze, etc.).
- If running in a hosted/managed VM, audit the VM provider's snapshot
  and disk-image policies — the local-file artifacts are visible to
  whoever administers the host.
- Audit operating-system or IDE telemetry / cloud-clipboard features
  separately.

## Syntonia interpretation caveat (reiterated)

Independent of privacy: the Syntonia score is a **behavioral-friction indicator**,
not a deception verdict. It flags moments where the subject's combined
voice / fidget / face signal departs from their own baseline. Anxiety,
distraction, physical discomfort, native-language fluency, cultural
baselines, and neurodivergence all produce Syntonia excursions in the absence
of deception. The 3.0 threshold itself must be empirically calibrated on
ground-truth-labelled data before any operational use.
