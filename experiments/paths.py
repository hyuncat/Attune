from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "app.py").is_file() and (candidate / "experiments").is_dir():
            return candidate

    raise RuntimeError(f"could not locate Attune repo root from {current}")


REPO_ROOT = find_repo_root()
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
BENCHMARKS_ROOT = EXPERIMENTS_ROOT  # Compatibility for existing module callers.
DATASETS_ROOT = BENCHMARKS_ROOT / "datasets"
RESULTS_ROOT = EXPERIMENTS_ROOT / "results" / "caches"
SWEEPS_ROOT = EXPERIMENTS_ROOT / "sweeps"
SWEEP_RESULTS_ROOT = RESULTS_ROOT / "sweeps"
SWEEP_SCRIPTS_ROOT = EXPERIMENTS_ROOT / "modules" / "sweeps" / "scripts"


def ensure_repo_on_path(root: Path = REPO_ROOT) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# Physical locations; dataset identifiers in published tables remain unchanged.
DATASET_LOCATIONS = {'bach10-original': 'datasets/bach10', 'cocochorales_tiny': 'datasets/coco/cocochorales_tiny', 'native_errors': 'datasets/coco/native_errors', 'mistake-db': 'datasets/coco-e2/mistake-db', 'spice': 'modules/pitch/competitors/data/spice', 'rmvpe': 'modules/pitch/competitors/data/rmvpe', 'vibrato': 'modules/vibrato/competitors/data/vibrato', 'vibrato_driedger': 'modules/vibrato/competitors/data/vibrato_driedger', 'cocochorales_vibrato': 'modules/vibrato/competitors/data/cocochorales_vibrato', 'pretrained/polytune': 'modules/mistake/competitors/data/polytune', 'pretrained/laddersym': 'modules/mistake/competitors/data/laddersym', 'pretrained/nakamura': 'modules/mistake/competitors/data/nakamura', 'pretrained/mt3-port': 'modules/note/competitors/data/mt3-port', 'pretrained/yourmt3': 'modules/note/competitors/data/yourmt3', 'bach10-mf0-synth': 'archive/datasets/bach10-mf0-synth', 'mdb-mf0-synth': 'archive/datasets/mdb-mf0-synth', 'mdb-stem-synth': 'archive/datasets/mdb-stem-synth', 'mdb-melody-synth': 'archive/datasets/mdb-melody-synth', 'violin-etudes': 'archive/datasets/violin-etudes', 'urmp': 'datasets/urmp'}


def dataset_root(name: str) -> Path:
    return EXPERIMENTS_ROOT / DATASET_LOCATIONS.get(name, "datasets/" + name)


def relocated_path(value: str | Path) -> Path:
    """Resolve original manifest paths without modifying historical evidence."""
    text = str(value)
    marker = "benchmarks/"
    if marker in text and (text.startswith(marker) or "/" + marker in text):
        tail = text.split(marker, 1)[1]
        if tail.startswith("results/"):
            return RESULTS_ROOT / tail[len("results/"):]
        for name, location in sorted(DATASET_LOCATIONS.items(), key=lambda item: -len(item[0])):
            prefix = "datasets/" + name
            if tail == prefix or tail.startswith(prefix + "/"):
                return EXPERIMENTS_ROOT / location / tail[len(prefix):].lstrip("/")
        return EXPERIMENTS_ROOT / tail
    return Path(value)


def relocate_paths(value):
    """Relocate paths in an in-memory manifest, retaining original files/hashes."""
    if isinstance(value, dict):
        return {str(relocated_path(k)) if isinstance(k, str) and "benchmarks/" in k else k:
                relocate_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_paths(v) for v in value]
    if isinstance(value, str) and (value.startswith("benchmarks/") or "/benchmarks/" in value):
        return str(relocated_path(value))
    return value
