#!/usr/bin/env bash
# Pre-stage all local model files for the Syntonia score pipeline. After this script
# runs once, the entire pipeline (face / hand / voice rate / transcript) can
# operate offline — the host network can be firewalled or torn down and
# everything still works.
#
# Run from the repository root:  bash scripts/fetch_models.sh

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

FACE_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

echo "→ MediaPipe Face Landmarker"
if [ ! -s models/face_landmarker.task ]; then
  curl -fsSL --retry 3 -o models/face_landmarker.task "$FACE_URL"
fi
ls -l models/face_landmarker.task

echo "→ MediaPipe Hand Landmarker"
if [ ! -s models/hand_landmarker.task ]; then
  curl -fsSL --retry 3 -o models/hand_landmarker.task "$HAND_URL"
fi
ls -l models/hand_landmarker.task

echo "→ faster-whisper small (CTranslate2 int8, ~250 MB)"
if [ ! -d models/whisper-small ] || [ -z "$(find models/whisper-small -name model.bin -print -quit)" ]; then
  python - <<'PYEOF'
from faster_whisper import WhisperModel
# Force the download into models/whisper-small so the entire app's
# data plane lives under one local directory.
WhisperModel("small", device="cpu", compute_type="int8",
             download_root="models/whisper-small")
print("Whisper small model staged.")
PYEOF
fi
du -sh models/whisper-small || true

echo
echo "All models present. The host network can now be firewalled."
echo "Inventory:"
ls -lh models/
