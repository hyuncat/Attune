"""PyInstaller runtime hook: configure native dependencies before app imports."""
import ctypes.util
import os
from pathlib import Path
import sys

root = Path(sys._MEIPASS)
os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("NUMBA_CACHE_DIR", str(
    Path.home() / "Library" / "Caches" / "Attune" / "numba"
))

# ctypes' macOS search otherwise finds Homebrew's library on the build machine.
# Bind pyfluidsynth to the shipped library, then restore normal lookup behavior.
original_find_library = ctypes.util.find_library


def bundled_find_library(name):
    if name in {"fluidsynth", "libfluidsynth"}:
        return str(root / "libfluidsynth.dylib")
    return original_find_library(name)


ctypes.util.find_library = bundled_find_library
try:
    import fluidsynth
finally:
    ctypes.util.find_library = original_find_library
