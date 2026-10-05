from __future__ import annotations
'Run the parallel CocoChorales time-aware alignment parameter sweep.\n\nThe runner writes resumable raw stages and compact CSV summaries under\n``benchmarks/results/sweeps/mistake``. The companion notebook only launches\nthis script and visualizes those files.\n'
import argparse
import os
import sys
from pathlib import Path

def _bootstrap_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / 'app.py').is_file() and (candidate / 'benchmarks').is_dir():
            return candidate
    raise RuntimeError('could not locate Attune repo root')
ROOT = _bootstrap_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.modules.note.sweeps.StringEditParamSweep import SweepOptions
from benchmarks.modules.note.sweeps.StringEditParamSweep import run_sweep

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1), help='worker processes; defaults to all logical CPUs except one')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--force', action='store_true', help='ignore matching stage checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-stratum', type=int, default=10)
    parser.add_argument('--tune-per-stratum', type=int, default=8)
    parser.add_argument('--max-strata', type=int, default=None)
    parser.add_argument('--stage1-limit', type=int, default=None, help='limit the first-stage grid for a smoke run; production is always included')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    options = SweepOptions(seed=args.seed, per_stratum=args.per_stratum, tune_per_stratum=args.tune_per_stratum, max_strata=args.max_strata, stage1_limit=args.stage1_limit)
    run_sweep(output_dir=args.output_dir, workers=args.workers, resume=not args.force, options=options)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
