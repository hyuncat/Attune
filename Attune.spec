# macOS build: python -m PyInstaller --noconfirm Attune.spec
import os
from pathlib import Path
import shutil
import sys
from ctypes.util import find_library

ROOT = Path(SPECPATH)
sys.path.insert(0, str(ROOT / "packaging"))
from bundle_config import data_entries, EXCLUDES

if sys.platform != "darwin":
    raise SystemExit("This spec targets macOS (the app currently uses CoreAudio).")


def native_file(variable, fallback):
    value = os.environ.get(variable) or fallback
    if not value or not Path(value).is_file():
        raise SystemExit(f"Set {variable} to an existing native binary/library path.")
    return str(Path(value).absolute())


fluid_default = next((str(p) for p in (
    Path("/opt/homebrew/lib/libfluidsynth.dylib"),
    Path("/usr/local/lib/libfluidsynth.dylib"),
) if p.is_file()), None)
fluid = native_file("ATTUNE_FLUIDSYNTH", fluid_default or find_library("fluidsynth"))
ffmpeg = native_file("ATTUNE_FFMPEG", shutil.which("ffmpeg"))
ffprobe = native_file("ATTUNE_FFPROBE", shutil.which("ffprobe"))
# Preserve a stable filename for runtime ctypes lookup; PyInstaller follows the
# native dependencies of all three binaries, including non-system dylibs.
if Path(fluid).name != "libfluidsynth.dylib":
    raise SystemExit("ATTUNE_FLUIDSYNTH must point to the libfluidsynth.dylib symlink/file.")

a = Analysis(
    [str(ROOT / "app.py")], pathex=[str(ROOT)],
    binaries=[(fluid, "."), (ffmpeg, "bin"), (ffprobe, "bin")],
    datas=data_entries(), hiddenimports=[],
    hookspath=[], runtime_hooks=[str(ROOT / "packaging/runtime.py")],
    excludes=EXCLUDES,
    # Numba's cache needs a real source file for the decorated Viterbi function.
    module_collection_mode={"algorithms.PitchSmoother": "pyz+py"},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="Attune",
    debug=False, strip=False, upx=False, console=False,
    codesign_identity=os.environ.get("ATTUNE_CODESIGN_IDENTITY"),
)
# Build the .app directly, avoiding a second full copy in dist/Attune.
app = BUNDLE(
    exe, a.binaries, a.datas,
    name="Attune.app", bundle_identifier="app.attune.desktop",
    icon=str(ROOT / "packaging" / "Attune.icns"),
    info_plist={
        "CFBundleName": "Attune",
        "NSMicrophoneUsageDescription": "Attune records your instrument to give practice feedback.",
        "NSHighResolutionCapable": True,
    },
)
