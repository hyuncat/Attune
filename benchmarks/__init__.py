import os as _os
import warnings as _warnings
from importlib import import_module as _import_module

_warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_EXPORTS = {
    "NoteDetectorBase": (
        "benchmarks.modules.note.NoteDetectorBase",
        "NoteDetectorBase",
    ),
    "PitchBenchmarker": (
        "benchmarks.modules.pitch.PitchBenchmarker",
        "PitchBenchmarker",
    ),
    "PitchDetectorBase": (
        "benchmarks.modules.pitch.PitchDetectorBase",
        "PitchDetectorBase",
    ),
    "CocoChorales": ("benchmarks.modules.pitch.datasets.CocoChorales", "CocoChorales"),
    "NoteBenchmarker": ("benchmarks.modules.note.NoteBenchmarker", "NoteBenchmarker"),
    "BenchmarkNoteDetector": (
        "benchmarks.modules.note.NoteDetectorBase",
        "NoteDetectorBase",
    ),
    "MistakeBenchmarker": (
        "benchmarks.modules.mistake.MistakeBenchmarker",
        "MistakeBenchmarker",
    ),
    "MistakeInjector": (
        "benchmarks.modules.mistake.datasets.MistakeInjector",
        "MistakeInjector",
    ),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _EXPORTS[name]
    value = getattr(_import_module(module_name), attr)
    globals()[name] = value
    return value


__all__ = [
    "NoteDetectorBase",
    "PitchBenchmarker",
    "PitchDetectorBase",
    "CocoChorales",
    "NoteBenchmarker",
    "BenchmarkNoteDetector",
    "MistakeBenchmarker",
    "MistakeInjector",
]
