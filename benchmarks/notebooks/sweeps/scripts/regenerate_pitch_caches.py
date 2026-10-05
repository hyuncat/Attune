#!/usr/bin/env python3
from __future__ import annotations

"""Regenerate selected CocoChorales pitch caches with current Stage-1 YIN."""

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app.py").is_file() and (
            candidate / "benchmarks"
        ).is_dir():
            return candidate
    raise RuntimeError("could not locate Attune repo root")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.modules.pitch.sweeps.PitchHMMParamSweep import (  # noqa: E402
    SweepOptions,
    regenerate_sample_caches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="worker processes; defaults to every logical CPU except one",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--tune-per-stratum", type=int, default=1)
    parser.add_argument("--max-strata", type=int, default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
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
    rows = regenerate_sample_caches(
        output_dir=args.output_dir,
        workers=args.workers,
        options=options,
        force=args.force,
    )
    errors = rows.loc[rows["error"].notna()]
    print(
        f"pitch caches: {rows.status.eq('regenerated').sum()} regenerated, "
        f"{rows.status.eq('current').sum()} already current, "
        f"{len(errors)} errors",
        flush=True,
    )
    if not errors.empty:
        print(
            errors[["track_id", "error"]].to_string(index=False),
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
