from __future__ import annotations
'Run the two-stage CocoChorales note-detection parameter sweep.\n\nThe parameter stage is fixed to linear KernelCPD, uses no transition handling\nor adjacent merging, and evaluates the full three-parameter Cartesian grid.\n'
import argparse
import json
import os
import sys
from pathlib import Path

def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / 'app.py').is_file() and (candidate / 'benchmarks').is_dir():
            return candidate
    raise RuntimeError('could not locate Attune repo root')
ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.modules.note.sweeps.NoteDetectionParamSweep import FIXED_PARAMETER_METHOD
from benchmarks.modules.note.sweeps.NoteDetectionParamSweep import METHOD_CONFIGS
from benchmarks.modules.note.sweeps.NoteDetectionParamSweep import ParameterAxes
from benchmarks.modules.note.sweeps.NoteDetectionParamSweep import SweepOptions
from benchmarks.modules.note.sweeps.NoteDetectionParamSweep import run_method_sweep
from benchmarks.modules.note.sweeps.NoteDetectionParamSweep import run_parameter_sweep

def _csv_values(text: str, cast):
    return tuple((cast(value.strip()) for value in text.split(',') if value.strip()))

def parse_args() -> argparse.Namespace:
    defaults = ParameterAxes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('methods', 'parameters', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--split', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-stratum', type=int, default=10)
    parser.add_argument('--tune-per-stratum', type=int, default=8)
    parser.add_argument('--max-strata', type=int, default=None)
    parser.add_argument('--max-tracks', type=int, default=None)
    parser.add_argument('--onset-tolerance', type=float, default=0.05)
    parser.add_argument('--method', action='append', choices=tuple(METHOD_CONFIGS), dest='methods')
    parser.add_argument('--fixed-method', choices=(FIXED_PARAMETER_METHOD,), default=FIXED_PARAMETER_METHOD, help='parameter stage is intentionally fixed to linear KernelCPD')
    parser.add_argument('--pitch-step-semitones', default=','.join(map(str, defaults.pitch_step_semitones)))
    parser.add_argument('--min-note-length-factor', default=','.join(map(str, defaults.min_note_length_factor)))
    parser.add_argument('--min-silence-duration-ms', default=','.join(map(str, defaults.min_silence_duration_ms)))
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    options = SweepOptions(split=args.split, seed=args.seed, per_stratum=args.per_stratum, tune_per_stratum=args.tune_per_stratum, onset_tolerance_sec=args.onset_tolerance, max_strata=args.max_strata, max_tracks=args.max_tracks)
    axes = ParameterAxes(pitch_step_semitones=_csv_values(args.pitch_step_semitones, float), min_note_length_factor=_csv_values(args.min_note_length_factor, float), min_silence_duration_ms=_csv_values(args.min_silence_duration_ms, float))
    selected = args.fixed_method
    if args.stage in {'methods', 'all'}:
        _, _, selected = run_method_sweep(output_dir=args.output_dir, workers=args.workers, options=options, methods=args.methods, force=args.force)
        print(f'selected method: {selected}', flush=True)
    if args.stage in {'parameters', 'all'}:
        _, _, recommendation = run_parameter_sweep(output_dir=args.output_dir, workers=args.workers, options=options, axes=axes, fixed_method=FIXED_PARAMETER_METHOD, force=args.force)
        print('recommended parameters:', json.dumps(recommendation, indent=2), flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
