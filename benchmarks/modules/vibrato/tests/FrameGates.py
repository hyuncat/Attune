"""Benchmark-only frame gates on frozen production score-assisted notes.

The notebook calls this module; synthetic checks live alongside it.
No production detector, fitted curve, or note boundary is changed here.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np

from benchmarks.modules.vibrato.competitors.Attune import Attune
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoEstimate
from benchmarks.modules.vibrato.tests.YangScore import digest


@dataclass(frozen=True)
class FrameGate:
    name: str
    min_rate_hz: float
    max_rate_hz: float
    min_width_cents: float
    min_quality: float | None = None

    def __post_init__(self):
        values = [self.min_rate_hz, self.max_rate_hz, self.min_width_cents]
        if not np.isfinite(values).all() or not 0 < self.min_rate_hz <= self.max_rate_hz or self.min_width_cents < 0:
            raise ValueError('Invalid rate/width gate')
        if self.min_quality is not None and not 0 <= self.min_quality <= 1:
            raise ValueError('Quality must be in [0, 1]')


DEFAULT_GATES = (
    FrameGate('attune_score_local', 3., 10., 10.),
    FrameGate('attune_score_local_yang', 4., 9., 20.),
    FrameGate('attune_score_local_yang_q030', 4., 9., 20., .3),
    FrameGate('attune_score_local_yang_q050', 4., 9., 20., .5),
)


def apply_gate(example, estimate: VibratoEstimate, gate: FrameGate) -> VibratoEstimate:
    """Intersect existing detections with local evidence; never consult truth.

    Width is peak-to-peak cents (20 cents = 0.1-semitone one-sided extent).
    Keep fitted curves for diagnostics; the detected mask controls detection and
    soft parameter scoring. No smoothing, gap bridging, or minimum-run filter.
    """
    estimate.validate_for(example)
    rate, width = np.asarray(estimate.rate_hz), np.asarray(estimate.width_cents)
    detected = (np.asarray(estimate.detected, dtype=bool)
                & np.isfinite(example.pitch_midi)
                & np.isfinite(rate) & np.isfinite(width)
                & (rate >= gate.min_rate_hz) & (rate <= gate.max_rate_hz)
                & (width >= gate.min_width_cents))
    if gate.min_quality is not None:
        if estimate.quality is None:
            # Empty segmentation has no quality and cannot generate detections.
            if detected.any():
                raise ValueError('Quality gate requires saved detector quality')
        else:
            quality = np.asarray(estimate.quality)
            detected &= np.isfinite(quality) & (quality >= gate.min_quality)
    return replace(estimate, detected=detected,
                   metadata={**estimate.metadata, 'frame_gate': asdict(gate)})


class FrozenScoreAttune(Attune):
    """Read the shared fitted curves, then optionally apply one frame gate."""

    def __init__(self, fit_root: Path, gate: FrameGate | None = None):
        super().__init__()
        self.fit_root = Path(fit_root)
        self.gate = gate
        self.name = gate.name if gate else 'attune_score_ungated'
        self.description = ('Attune score-assisted; ' +
                            (f'frame gate {asdict(gate)}' if gate else 'original whole-note decision'))

    def prepare(self, example):
        with (self.fit_root/f'{example.case_id}.pkl').open('rb') as handle:
            self.frozen_estimate = pickle.load(handle)['estimate']

    def estimate(self, example):
        estimate = self.frozen_estimate.validate_for(example)
        return apply_gate(example, estimate, self.gate) if self.gate else estimate


def _fit_case(job):
    """One fit per recording, reused identically by every gate."""
    from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker
    example, output = job
    with VibratoBenchmarker._single_threaded_numerics():
        start = time.process_time()
        estimate = Attune().estimate(example).validate_for(example)
        elapsed = time.process_time()-start
    with (output/f'{example.case_id}.pkl').open('wb') as handle:
        pickle.dump({'estimate': estimate, 'fit_cpu_seconds': elapsed}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return {'case_id': example.case_id, 'fit_cpu_seconds': elapsed}


def load_score_examples(prepared: Path, score_run: Path):
    """Reuse the completed production detect/align/repeat arm, never DTW cuts."""
    manifest_path = prepared/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    provenance = json.loads((score_run/'run_config.json').read_text())
    if provenance.get('mode') != 'score' or provenance.get('prepared_manifest_sha256') != digest(manifest_path):
        raise ValueError('Expected a score-assisted run using exactly these prepared inputs')
    if digest(prepared/'inputs.pkl') != manifest['inputs_sha256']:
        raise ValueError('Prepared pitch/truth inputs changed')
    with (prepared/'inputs.pkl').open('rb') as handle:
        examples, _ = pickle.load(handle)  # trusted local benchmark artifact
    bounds_path = score_run/'attune_score/note_bounds.json'
    bounds = json.loads(bounds_path.read_text())
    if set(bounds) != {e.case_id for e in examples}:
        raise ValueError('Score boundaries and prepared cohort differ')
    for example in examples:
        previous_end = -np.inf
        for start, end, pitch in bounds[example.case_id]:
            if not np.isfinite([start, end, pitch]).all() or end <= start or start < previous_end-1e-8:
                raise ValueError(f'Invalid score-assisted boundaries: {example.case_id}')
            previous_end = end
    variants = [replace(e, metadata={**e.metadata,
        'analysis_note_bounds': bounds[e.case_id],
        'note_boundary_source': 'frozen_production_detect_align_repeat_with_candidate_score'})
        for e in examples]
    return variants, {'prepared_manifest_sha256': digest(manifest_path),
                      'inputs_sha256': manifest['inputs_sha256'],
                      'score_run_config_sha256': digest(score_run/'run_config.json'),
                      'score_bounds_sha256': digest(bounds_path)}


def run_frame_gates(*, prepared, score_run, output, workers=1, gates=DEFAULT_GATES,
                    rate_tolerance_hz=.5, extent_tolerance_semitones=.05,
                    center_tolerance_cents=25.):
    """Notebook runner. Writes a fresh run or reads a matching completed run.

    Refit only the unchanged vibrato model on frozen score-assisted notes; no
    pitch inference, score matching, segmentation, or DTW is performed.
    """
    import pandas as pd
    from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker
    from benchmarks.modules.vibrato.competitors.Yang import YangDT, YangBR

    prepared, score_run, output = map(Path, (prepared, score_run, output))
    if workers < 1:
        raise ValueError('workers must be positive')
    if len({g.name for g in gates}) != len(gates) or any(g.name in {'attune_score_ungated', 'yang_dt', 'yang_br'} for g in gates):
        raise ValueError('Gate method names must be unique')
    examples, provenance = load_score_examples(prepared, score_run)
    root = Path(__file__).resolve().parents[4]
    sources = ['algorithms/VibratoDetector.py', 'algorithms/Config.py',
               'app_logic/user/ds/VibratoData.py', 'app_logic/user/ds/PitchData.py',
               'benchmarks/modules/vibrato/competitors/Attune.py',
               'benchmarks/modules/vibrato/competitors/Yang.py',
               'benchmarks/modules/vibrato/VibratoBenchmarker.py',
               'benchmarks/modules/vibrato/VibratoDetectorBase.py']
    config = dict(version='score_frame_gates_v1', **provenance,
        module_sha256=digest(__file__), source_sha256={p: digest(root/p) for p in sources},
        gates=[asdict(g) for g in gates], cases=[e.case_id for e in examples],
        rate_tolerance_hz=rate_tolerance_hz, extent_tolerance_semitones=extent_tolerance_semitones,
        center_tolerance_cents=center_tolerance_cents,
        boundary_source='score-assisted production detect/align/repeat (frozen)',
        protocol='Paired exploratory six-candidate comparison; full-recording masks; no automatic winner selection',
        timing='Attune report timings are gate-only; shared fitting cost is in fit_timing.csv. Do not compare throughput with Yang.')
    if output.exists():
        saved = output/'run_config.json'
        if not saved.exists() or json.loads(saved.read_text()) != config or not (output/'COMPLETE').exists():
            raise ValueError('Output exists with changed inputs/code or an incomplete run; choose a fresh output tag')
        return pd.read_csv(output/'comparison.csv')
    output.mkdir(parents=True)
    (output/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    fits = output/'fits'
    fits.mkdir()
    jobs = [(example, fits) for example in examples]
    if workers == 1:
        timing = [_fit_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            timing = list(pool.map(_fit_case, jobs))
    pd.DataFrame(timing).to_csv(output/'fit_timing.csv', index=False)
    benchmark = VibratoBenchmarker(rate_accuracy_tolerance_hz=rate_tolerance_hz,
        amplitude_accuracy_tolerance_semitones=extent_tolerance_semitones,
        center_accuracy_tolerance_cents=center_tolerance_cents)
    estimators = [FrozenScoreAttune(fits), *[FrozenScoreAttune(fits, g) for g in gates], YangDT(), YangBR()]
    raw = benchmark.run(examples, estimators, workers=workers, strict=True,
                        cache_dir=output/'checkpoints')
    benchmark.write_reports(raw, output)
    summary = benchmark.summarize(raw)
    summary.to_csv(output/'comparison.csv', index=False)
    benchmark.summarize(raw, group_by=('method', 'case_id')).to_csv(output/'per_recording.csv', index=False)
    (output/'note_bounds.json').write_text(json.dumps({e.case_id: e.metadata['analysis_note_bounds'] for e in examples}, indent=2)+'\n')
    (output/'COMPLETE').write_text('complete\n')
    return summary
