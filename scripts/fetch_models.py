"""Cross-platform model staging for the Syntonia score pipeline.

Works on Windows, macOS, and Linux. Replaces scripts/fetch_models.sh.

Run from the repo root:
    python scripts/fetch_models.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"

ASSETS = [
    {
        "name": "MediaPipe Face Landmarker",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        ),
        "dest": MODELS / "face_landmarker.task",
        "size_mb": 3.6,
    },
    {
        "name": "MediaPipe Hand Landmarker",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        ),
        "dest": MODELS / "hand_landmarker.task",
        "size_mb": 7.5,
    },
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    tmp.replace(dest)
    mb = dest.stat().st_size / (1024 * 1024)
    print(f"  saved {dest}  ({mb:.1f} MB)")


def fetch_mediapipe_assets() -> int:
    failures = 0
    for asset in ASSETS:
        print(f"→ {asset['name']}")
        if asset["dest"].exists() and asset["dest"].stat().st_size > 0:
            print(f"  already present ({asset['dest'].stat().st_size / (1024*1024):.1f} MB)")
            continue
        try:
            _download(asset["url"], asset["dest"])
        except Exception as e:
            print(f"  FAILED: {e}")
            failures += 1
    return failures


def fetch_whisper() -> int:
    """Stage faster-whisper 'small' (int8 CTranslate2) under models/whisper-small.
    This requires huggingface.co reachable; it falls back to a clear error
    that points the user to a manual / offline path."""
    target = MODELS / "whisper-small"
    if target.exists() and list(target.rglob("model.bin")):
        print("→ faster-whisper small: already present")
        return 0
    print("→ faster-whisper small  (~250 MB, one-time download from huggingface.co)")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  faster-whisper is not installed. Skipping. Install with:")
        print("    pip install faster-whisper")
        return 1
    try:
        target.mkdir(parents=True, exist_ok=True)
        WhisperModel("small", device="cpu", compute_type="int8",
                     download_root=str(target))
        # Spot-check the snapshot
        bins = list(target.rglob("model.bin"))
        if bins:
            print(f"  staged {bins[0].parent}")
            return 0
        print("  download completed but no model.bin found — try re-running.")
        return 1
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  If huggingface.co is unreachable on this host, run this script "
              "on any internet-capable machine and copy the populated "
              "models/whisper-small/ directory across.")
        return 1


def check_ffmpeg() -> int:
    exe = shutil.which("ffmpeg")
    if exe:
        print(f"→ ffmpeg: found at {exe}")
        return 0
    print("→ ffmpeg: NOT FOUND on PATH.")
    print("  Install instructions:")
    if sys.platform.startswith("win"):
        print("    winget install --id Gyan.FFmpeg")
        print("    (or download a build from https://www.gyan.dev/ffmpeg/builds/ "
              "and add the bin/ folder to PATH)")
    elif sys.platform == "darwin":
        print("    brew install ffmpeg")
    else:
        print("    apt install ffmpeg     (or your distro's equivalent)")
    return 1


def main() -> int:
    print(f"Repo root: {REPO}")
    MODELS.mkdir(exist_ok=True)
    failures = 0
    failures += fetch_mediapipe_assets()
    failures += fetch_whisper()
    failures += check_ffmpeg()
    print()
    print("Final model inventory:")
    for p in sorted(MODELS.iterdir()):
        if p.is_dir():
            size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
            print(f"  {p.relative_to(REPO)}/   ({size_mb:.1f} MB)")
        else:
            print(f"  {p.relative_to(REPO)}   ({p.stat().st_size / (1024*1024):.1f} MB)")
    if failures == 0:
        print("\nAll set. The host can be firewalled — every subsequent run is offline.")
        return 0
    print(f"\n{failures} step(s) need attention. See messages above.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
