"""Verify the project directory is NOT inside a known cloud-sync root.

Use this as a pre-recording sanity check on Windows / macOS / Linux.

Exit codes:
  0  isolated — safe to proceed
  1  caution — inside Documents/Desktop/Pictures, which some sync apps
              (Google Drive Backup & Sync, OneDrive's "Backup important
              folders") are typically configured to upload. Check the
              sync client's settings.
  2  unsafe — inside an explicit cloud-sync root. Move the project out
              before recording any biometric data.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
HOME = Path.home()

SYNC_ROOTS = {
    "windows": [
        HOME / "OneDrive",
        HOME / "iCloudDrive",
        HOME / "iCloud Drive",
        HOME / "Google Drive",
        HOME / "GoogleDrive",
        HOME / "My Drive",
        HOME / "Dropbox",
        HOME / "Dropbox (Personal)",
        HOME / "Dropbox (Business)",
        HOME / "Box",
        HOME / "Box Sync",
        HOME / "pCloud Drive",
        HOME / "Sync",
    ],
    "darwin": [
        HOME / "Library" / "Mobile Documents" / "com~apple~CloudDocs",
        HOME / "Library" / "CloudStorage",   # any cloud provider on macOS Ventura+
        HOME / "Google Drive",
        HOME / "Dropbox",
        HOME / "OneDrive",
        HOME / "Box",
    ],
    "linux": [
        HOME / "Dropbox",
        HOME / "Insync",
        HOME / "OneDrive",
        HOME / "Nextcloud",
        HOME / "ownCloud",
    ],
}

CAUTION_FOLDERS = [
    HOME / "Documents",
    HOME / "Desktop",
    HOME / "Pictures",
    HOME / "Movies",       # macOS
    HOME / "Videos",       # Windows / Linux
]


def _platform_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
    except OSError:
        return False


def main() -> int:
    print(f"Project location:  {PROJECT}")
    print(f"User home:         {HOME}")
    print(f"Platform:          {sys.platform}")
    print()

    roots = SYNC_ROOTS.get(_platform_key(), [])
    unsafe = [r for r in roots if r.exists() and _is_inside(PROJECT, r)]
    caution = [c for c in CAUTION_FOLDERS if c.exists() and _is_inside(PROJECT, c)]

    if unsafe:
        print("[UNSAFE] Project sits inside an explicit cloud-sync root:")
        for r in unsafe:
            print(f"   - {r}")
        print()
        print("  Biometric artifacts written under videos/ ARE syncing to")
        print("  a third-party server. Move the project to a folder outside")
        print("  these paths before recording any sessions. Suggested target:")
        if _platform_key() == "windows":
            print("    C:\\Projects\\stockapp")
        else:
            print("    ~/dev/stockapp")
        return 2

    if caution:
        print("[CAUTION] Project sits inside a folder that cloud-sync apps")
        print("commonly back up:")
        for c in caution:
            print(f"   - {c}")
        print()
        print("  Google Drive (Backup & Sync), OneDrive ('Backup important")
        print("  folders'), iCloud Desktop & Documents, etc. can each be")
        print("  configured to upload these folders. Confirm in each sync")
        print("  client's Preferences that the project path is excluded, or")
        print("  move the project to a folder outside the user profile.")
        return 1

    print("[OK] Project is NOT inside any known cloud-sync root.")
    print("    Biometric artifacts written under videos/ stay on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
