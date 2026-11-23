"""Explicit repository data manifest; never copy the whole resources tree."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICONS = (
    "play.png", "pause.png", "record.png", "plus.svg", "minus.svg",
    "flatsign.svg", "sharpsign.svg", "trash-2.svg", "undo-2.svg",
    "pencil.svg", "check.svg", "circle-help.svg", "volume.svg", "volume-2.svg",
)
VIEWER = (
    "viewer.html", "viewer.css", "viewer.js",
    "imports/verovio-toolkit-wasm.js", "imports/verovio-toolkit-wasm.wasm",
)
SCALE_TAKES = {
    "scales/major": {"flute_scale_good", "flute_scale_bad"},
    "scales/minor": {"major_singing"},
}
SCORE_SUFFIXES = {".mid", ".midi", ".xml", ".musicxml", ".mxl", ".mei"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aif", ".aiff"}


def include_demo(relative: Path) -> bool:
    if relative.parts[0].lower().startswith("fugue"):
        return False
    if relative.name == ".DS_Store" or "__pycache__" in relative.parts:
        return False
    allowed = SCALE_TAKES.get(relative.parent.as_posix())
    if allowed is None:
        return True
    if relative.suffix.lower() in SCORE_SUFFIXES:
        return True
    if relative.suffix.lower() in AUDIO_SUFFIXES:
        return relative.stem in allowed
    # Hidden sidecars are essential: don't drop the precomputed demo analysis.
    return any(relative.name == f".{take}{extension}"
               for take in allowed for extension in (".json.xz", ".json.gz", ".json"))


def resource_files(root: Path = ROOT) -> list[Path]:
    files = [root / "resources/MuseScore_General.sf3"]
    files += [root / "resources/icons" / name for name in ICONS]
    files += [root / "ui/score/verovio" / name for name in VIEWER]
    demo = root / "resources/demo"
    files += [p for p in demo.rglob("*") if p.is_file() and include_demo(p.relative_to(demo))]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing app resource: {path}")
    return sorted(files)


def data_entries() -> list[tuple[str, str]]:
    return [(str(p), p.relative_to(ROOT).parent.as_posix()) for p in resource_files()]


# Optional visualization/notebook backends imported by third-party packages are
# not used by Attune. Required transitive dependencies remain hook-managed.
EXCLUDES = [
    "benchmarks", "experiments", "data", "notebooks", "web", "basic_pitch", "tensorflow", "torch",
    "librosa", "parselmouth", "hmmlearn", "IPython", "ipykernel", "jupyter",
    "notebook", "pytest", "matplotlib", "pandas", "sklearn",
    "PyQt5", "PySide2", "PySide6", "tkinter",
]
