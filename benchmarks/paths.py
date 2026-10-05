from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
            return candidate

    raise RuntimeError(f"could not locate Attune repo root from {current}")


REPO_ROOT = find_repo_root()
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
DATASETS_ROOT = BENCHMARKS_ROOT / "datasets"
RESULTS_ROOT = BENCHMARKS_ROOT / "results"
SWEEPS_ROOT = BENCHMARKS_ROOT / "notebooks" / "sweeps"
SWEEP_RESULTS_ROOT = RESULTS_ROOT / "sweeps"
SWEEP_SCRIPTS_ROOT = SWEEPS_ROOT / "scripts"


def ensure_repo_on_path(root: Path = REPO_ROOT) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
