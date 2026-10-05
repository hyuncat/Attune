#!/usr/bin/env python3
from __future__ import annotations

"""Run the parallel CocoChorales pYIN-HMM parameter sweep.

The default sample contains two stems from every available
(ensemble, instrument) stratum. All selected stems contribute to one combined,
stratum-balanced recommendation. Stage-1 candidates are read from the existing
raw pitch caches; this runner never reruns pitch detection.
"""

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
            return candidate
    raise RuntimeError("could not locate Attune repo root")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.modules.pitch.sweeps.PitchHMMParamSweep import (  # noqa: E402
    OBSERVATION_MODES,
    ParameterAxes,
    SweepOptions,
    run_sweep,
)


def _csv_values(text: str, cast):
    return tuple(cast(value.strip()) for value in text.split(",") if value.strip())


def parse_args() -> argparse.Namespace:
    defaults = ParameterAxes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="worker processes; defaults to every logical CPU except one",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-current-cache",
        action="store_true",
        help="fail if any raw cache predates the current Stage-1 detector",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--tune-per-stratum", type=int, default=1)
    parser.add_argument("--max-strata", type=int, default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument(
        "--variant-limit",
        type=int,
        default=None,
        help="truncate the grid for a smoke run",
    )
    parser.add_argument(
        "--observation-mode",
        default=",".join(defaults.observation_mode),
        help=f"comma-separated values from {OBSERVATION_MODES}",
    )
    parser.add_argument(
        "--max-jump-bins",
        default=",".join(map(str, defaults.max_jump_bins)),
    )
    parser.add_argument(
        "--switch-prob",
        default=",".join(map(str, defaults.switch_prob)),
    )
    parser.add_argument(
        "--yin-trust",
        default=",".join(map(str, defaults.yin_trust)),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    options = SweepOptions(
        split=args.split,
        seed=args.seed,
        per_stratum=args.per_stratum,
        tune_per_stratum=args.tune_per_stratum,
        max_strata=args.max_strata,
        max_tracks=args.max_tracks,
    )
    axes = ParameterAxes(
        observation_mode=_csv_values(args.observation_mode, str),
        max_jump_bins=_csv_values(args.max_jump_bins, int),
        switch_prob=_csv_values(args.switch_prob, float),
        yin_trust=_csv_values(args.yin_trust, float),
    )
    result = run_sweep(
        output_dir=args.output_dir,
        workers=args.workers,
        options=options,
        axes=axes,
        force=args.force,
        variant_limit=args.variant_limit,
        require_current_cache=args.require_current_cache,
    )
    print("recommended parameters:", result["recommendation"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
