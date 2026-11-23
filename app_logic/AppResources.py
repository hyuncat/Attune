"""Read-only bundled assets and the user's editable copy of the demos."""
from pathlib import Path
import json
import lzma
import os
import shutil
import sys
import tempfile

from PyQt6.QtCore import QStandardPaths


ROOT = Path(__file__).resolve().parents[1]


def demo_directory() -> Path:
    source = ROOT / "resources" / "demo" / "scales"
    if not getattr(sys, "frozen", False):
        return source

    # Seed once, rather than restoring recordings the user has renamed/deleted.
    parent = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericDataLocation
    )) / "Attune"
    destination = parent / "scales"
    if not destination.exists():
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".demos-", dir=parent) as temporary:
            staged = Path(temporary) / "scales"
            shutil.copytree(source, staged)
            # PyInstaller and archive extraction change file timestamps. Restore
            # the fingerprint on these fresh seed copies before the normal cache
            # loader sees them. Later user edits still invalidate the cache.
            for sidecar in staged.rglob(".*.json.xz"):
                with lzma.open(sidecar, "rt", encoding="utf-8") as stream:
                    meta = json.load(stream)["recording"]
                audio = sidecar.with_name(meta["audio_file"])
                stat = audio.stat()
                if stat.st_size != meta["audio_size"]:
                    raise ValueError(f"Demo cache does not match {audio.name}")
                os.utime(audio, ns=(stat.st_atime_ns, meta["audio_mtime_ns"]))
            try:
                staged.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    return destination
