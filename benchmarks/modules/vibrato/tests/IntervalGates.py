"""Paired minimum-cycle/duration ablation using completed score-assisted fits."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import numpy as np

from benchmarks.modules.vibrato.tests.FrameGates import (
    DEFAULT_GATES, FrozenScoreAttune, load_score_examples,
)
from benchmarks.modules.vibrato.tests.YangScore import digest


@dataclass(frozen=True)
class IntervalGate:
    suffix: str
    min_cycles: float = 0.
    min_seconds: float = 0.

    def __post_init__(self):
        if not np.isfinite([self.min_cycles, self.min_seconds]).all() or min(self.min_cycles, self.min_seconds) < 0:
            raise ValueError('Interval minima must be finite and nonnegative')


INTERVAL_GATES = (IntervalGate('one_cycle', min_cycles=1.),
                  IntervalGate('250ms', min_seconds=.25))


def apply_interval_gate(example, estimate, gate):
    """Prune contiguous positive runs; never bridge gaps or reset at note cuts.

    Integrate rate over frame support: sum(rate * dt). Duration uses the same
    half-open support, n*dt. The 250 ms arm is a duration-only analogue of Yang,
    not its 125 ms-window decisions, frame cleanup, or six-frame criterion.
    """
    estimate.validate_for(example)
    times = np.asarray(example.times, float)
    if len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError('A regular time grid with at least two frames is required')
    dt = float(np.median(np.diff(times)))
    if dt <= 0 or not np.allclose(np.diff(times), dt, rtol=.01, atol=1e-9):
        raise ValueError('A regular increasing time grid is required')
    rates = np.asarray(estimate.rate_hz)
    mask = np.asarray(estimate.detected, bool) & np.isfinite(rates) & (rates > 0)
    mask &= np.isfinite(example.pitch_midi)
    indices = np.flatnonzero(mask)
    output = np.zeros(len(times), bool)
    for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1):
        if len(run) and len(run)*dt+1e-9 >= gate.min_seconds and rates[run].sum()*dt+1e-9 >= gate.min_cycles:
            output[run] = True
    return replace(estimate, detected=output,
                   metadata={**estimate.metadata, 'interval_gate': asdict(gate)})


class IntervalScoreAttune(FrozenScoreAttune):
    def __init__(self, fit_root, frame_gate, interval_gate):
        super().__init__(fit_root, frame_gate)
        self.interval_gate = interval_gate
        self.name += '_' + interval_gate.suffix
        self.description += f'; interval gate {asdict(interval_gate)}'

    def estimate(self, example):
        return apply_interval_gate(example, super().estimate(example), self.interval_gate)


def run_interval_gates(*, prepared, score_run, source_run, output, workers=2):
    """Rescore frozen estimates only; preserve all earlier experiment outputs."""
    import pandas as pd
    from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker

    prepared, score_run, source_run, output = map(Path, (prepared, score_run, source_run, output))
    if workers < 1:
        raise ValueError('workers must be positive')
    examples, provenance = load_score_examples(prepared, score_run)
    source_config = json.loads((source_run/'run_config.json').read_text())
    if not (source_run/'COMPLETE').exists() or any(source_config.get(k) != v for k,v in provenance.items()):
        raise ValueError('Completed source gate run must use these exact score-assisted inputs')
    root = Path(__file__).resolve().parents[4]
    fits = source_run/'fits'
    frame_gates = DEFAULT_GATES[1:3]  # rate/width, then rate/width + quality .30
    for gate in frame_gates:
        if asdict(gate) not in source_config['gates']:
            raise ValueError('Source frame-gate definitions differ')
    config = dict(version='score_interval_gates_v1', **provenance,
        source_run_config_sha256=digest(source_run/'run_config.json'),
        source_comparison_sha256=digest(source_run/'comparison.csv'),
        source_fit_sha256={e.case_id: digest(fits/f'{e.case_id}.pkl') for e in examples},
        source_sha256={str(p.relative_to(root)): digest(p) for p in (
            Path(__file__), Path(__file__).with_name('FrameGates.py'),
            root/'benchmarks/modules/vibrato/VibratoBenchmarker.py',
            root/'benchmarks/modules/vibrato/VibratoDetectorBase.py')},
        frame_gates=[asdict(g) for g in frame_gates], interval_gates=[asdict(g) for g in INTERVAL_GATES],
        protocol='Frozen score-assisted fits; contiguous positive runs; sum(rate*dt) cycles and n*dt seconds; no note resets or gap filling',
        timing='Attune timings cover postprocessing only; reference Yang rows retain original timing')
    if output.exists():
        if not (output/'COMPLETE').exists() or json.loads((output/'run_config.json').read_text()) != config:
            raise ValueError('Changed or incomplete interval run; choose a fresh output directory')
        return pd.read_csv(output/'comparison.csv')
    output.mkdir(parents=True)
    (output/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    benchmark = VibratoBenchmarker(
        rate_accuracy_tolerance_hz=source_config['rate_tolerance_hz'],
        amplitude_accuracy_tolerance_semitones=source_config['extent_tolerance_semitones'],
        center_accuracy_tolerance_cents=source_config['center_tolerance_cents'])
    estimators = [FrozenScoreAttune(fits)]
    for frame in frame_gates:
        estimators.append(FrozenScoreAttune(fits, frame))
        estimators.extend(IntervalScoreAttune(fits, frame, interval) for interval in INTERVAL_GATES)
    raw = benchmark.run(examples, estimators, workers=workers, strict=True, cache_dir=output/'checkpoints')
    benchmark.write_reports(raw, output)
    summary = benchmark.summarize(raw)
    # Verify unfiltered rows reproduce the completed experiment before comparing
    # interval filters. Metrics have moved modules but their definitions must not change.
    old = pd.read_csv(source_run/'comparison.csv').set_index('method')
    metrics = ['frame_precision','frame_recall','frame_f1','yang_note_f1','aggregate_soft_f1']
    for method in ['attune_score_ungated', *[g.name for g in frame_gates]]:
        np.testing.assert_allclose(summary.set_index('method').loc[method, metrics].astype(float),
                                   old.loc[method, metrics].astype(float), atol=1e-10, rtol=1e-10, equal_nan=True)
    summary['result_source'] = 'frozen_fit_replay'
    references = old.loc[['yang_dt','yang_br']].reset_index()
    references['result_source'] = 'saved_same_cohort_reference'
    combined = pd.concat([summary, references], ignore_index=True)
    combined.to_csv(output/'comparison.csv', index=False)
    benchmark.summarize(raw, group_by=('method','case_id')).to_csv(output/'per_recording.csv', index=False)
    (output/'COMPLETE').write_text('complete\n')
    return combined


def show_interval_results(summary):
    from IPython.display import display
    columns = ['method','frame_precision','frame_recall','frame_f1','yang_note_f1','aggregate_soft_f1']
    display(summary[columns])
    indexed = summary.set_index('method')
    metrics = columns[1:]
    rows = []
    for frame in DEFAULT_GATES[1:3]:
        for interval in INTERVAL_GATES:
            name = frame.name+'_'+interval.suffix
            delta = 100*(indexed.loc[name,metrics].astype(float)-indexed.loc[frame.name,metrics].astype(float))
            rows.append({'method':name, **delta.to_dict()})
    import pandas as pd
    print('Changes versus each matching frame-only gate (percentage points):')
    display(pd.DataFrame(rows))
