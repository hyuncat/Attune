from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
from dataclasses import replace
import lzma
import json
import math
import multiprocessing
import os
import pickle
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Every detector's inner linear algebra is small -- Attune alone issues
# thousands of tiny QR, solve, and SVD calls per analysis group -- so a BLAS
# thread pool inside each worker spends its time spinning rather than
# computing. Unpinned, one Yang group costs 0.7 s of wall time but 5.7 s of
# process CPU on its own, and nine spawned workers each opening eight threads
# on ten cores never finish it at all. A spawned worker imports this module to
# resolve its job function, so setting these before numpy loads pins the
# worker too. (Same guard as the pitch/note sweeps.)
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from benchmarks.modules.vibrato.VibratoDetectorBase import (
    VibratoDetectorBase,
    VibratoEstimate,
    VibratoExample,
)


ProgressCallback = Callable[[str, int, int, VibratoExample], None]


class YangMetrics:
    """Yang interval and soft-credit metrics, owned by the benchmark scorer.

    Boundary matching and overlap diagnostics follow the existing adapted
    protocol; the latter are not a port of Molina's evaluator.
    """

    @staticmethod
    def soft_counts(truth, predicted, reference, estimate, eligible=None):
        truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
        reference, estimate = np.asarray(reference), np.asarray(estimate)
        eligible = np.ones(len(truth), bool) if eligible is None else np.asarray(eligible, bool)
        valid = eligible & (~truth | (np.isfinite(reference) & (reference > 0)))
        target = valid & truth
        prediction = valid & predicted & np.isfinite(estimate) & (estimate > 0)
        matched = target & prediction
        credit = float(np.maximum(0, 1 - np.abs(estimate[matched] - reference[matched]) / reference[matched]).sum())
        return credit, float(prediction.sum()), float(target.sum())

    @staticmethod
    def soft_prf(credit, predictions, references):
        return (credit / predictions if predictions else 0.,
                credit / references if references else 0.,
                2 * credit / (predictions + references) if predictions + references else 0.)

    @staticmethod
    def intervals(times, mask, minimum_duration=0.28):
        dt = float(np.median(np.diff(times)))
        indices = np.flatnonzero(mask)
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        return [(float(times[g[0]]), float(times[g[-1]] + dt), g)
                for g in groups if len(g) and len(g)*dt > minimum_duration]

    @staticmethod
    def evaluate(times, detected, rates, extents, references):
        """references: start/end plus optional rate_hz/extent_semitones.

        Notes: maximum-cardinality one-to-one boundary matches; 100 ms onset,
        max(100 ms, 20% reference duration) offset. Parameters: >=50% of the
        detected interval inside reference, averaging split detections equally.
        """
        runs = YangMetrics.intervals(times, detected)
        nr, nd = len(references), len(runs)
        overlap = np.zeros((nr, nd)); onset = np.zeros((nr, nd), bool); offset = onset.copy()
        for i, ref in enumerate(references):
            for j, (start, end, _) in enumerate(runs):
                overlap[i,j] = max(0., min(ref['end'], end) - max(ref['start'], start))
                onset[i,j] = abs(start-ref['start']) <= .1 + 1e-9
                offset[i,j] = abs(end-ref['end']) <= max(.1, .2*(ref['end']-ref['start'])) + 1e-9
        valid = onset & offset & (overlap > 0)
        ri, pi = linear_sum_assignment(-valid.astype(float))
        tp = int(valid[ri, pi].sum())
        edges = overlap > 0
        rd, pd = edges.sum(axis=1), edges.sum(axis=0)
        isolated = edges & (rd[:,None] == 1) & (pd[None,:] == 1)
        result = dict(yang_note_tp=tp, yang_note_fp=nd-tp, yang_note_fn=nr-tp,
            yang_reference_notes=nr, yang_predicted_notes=nd,
            yang_only_bad_onset=int((isolated & ~onset & offset).sum()),
            yang_only_bad_offset=int((isolated & onset & ~offset).sum()),
            yang_split=int((rd > 1).sum()), yang_merge=int((pd > 1).sum()),
            yang_spurious=int((pd == 0).sum()), yang_non_detected=int((rd == 0).sum()))
        scores = []
        parameter_truth = 0
        for i, ref in enumerate(references):
            if not (ref.get('rate_hz', 0) > 0 and ref.get('extent_semitones', 0) > 0):
                continue
            parameter_truth += 1
            selected = [run for j, run in enumerate(runs) if overlap[i,j] >= .5*(run[1]-run[0])]
            if selected:
                estimated = [np.mean([np.mean(values[run[2]]) for run in selected]) for values in (rates, extents)]
                scores.append([max(0., 1-abs(e-r)/r) for e,r in zip(estimated, (ref['rate_hz'],ref['extent_semitones']))])
        result.update(yang_parameter_truth=parameter_truth, yang_parameter_matched=len(scores),
            yang_rate_sum=float(sum(s[0] for s in scores)), yang_extent_sum=float(sum(s[1] for s in scores)))
        return result

    @staticmethod
    def summary(counts):
        tp, fp, fn = (counts.get('yang_note_'+k, 0) for k in ('tp','fp','fn'))
        nr, nd = counts.get('yang_reference_notes', 0), counts.get('yang_predicted_notes', 0)
        p, r, f = YangMetrics.soft_prf(tp, tp+fp, tp+fn)
        out = dict(yang_note_precision=p, yang_note_recall=r, yang_note_f1=f,
                   yang_reference_notes=nr, yang_predicted_notes=nd,
                   yang_matched_notes=counts.get('yang_parameter_matched', 0),
                   yang_parameter_notes=counts.get('yang_parameter_truth', 0))
        for key in ('only_bad_onset','only_bad_offset','split','merge','spurious','non_detected'):
            denominator = nd if key == 'spurious' else nr
            out['yang_'+key+'_rate'] = counts.get('yang_'+key, 0) / denominator if denominator else np.nan
        return out


class VibratoBenchmarker:
    """Run, summarize, and report every vibrato detector on shared corpora."""

    PROFILES = ('none', 'constant', 'accelerating', 'decelerating', 'widening', 'narrowing')
    METRICS = ('frame_f1', 'rate_f1', 'extent_f1', 'aggregate_f1',
               'rate_soft_f1', 'extent_soft_f1', 'aggregate_soft_f1')
    COMPETITORS = {'yang': ('yang_br', 'yang_dt'),
                   'coco': ('mcleod', 'yang_br', 'yang_dt', 'driedger_benchmark_range')}
    COUNT_NAMES = ('frame', 'rate', 'extent', 'rate_soft', 'extent_soft')

    @staticmethod
    def _digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def yang_examples(cls, root=REPO_ROOT):
        """Run current production note analysis on the six prepared candidate scores."""
        from app_logic.midi.ScoreData import ScoreData
        from app_logic.user.ds.Recording import Recording
        from benchmarks.modules.vibrato.datasets.YangFullDataset import YangParameterDataset

        root = Path(root)
        prepared = root / 'benchmarks/results/yang_score_boundary_v1_inputs'
        manifest = json.loads((prepared / 'manifest.json').read_text())
        if cls._digest(prepared / 'inputs.pkl') != manifest['inputs_sha256']:
            raise ValueError('Prepared inputs changed; rebuild the Yang score inputs.')
        recordings = YangParameterDataset.discover(root / 'benchmarks/datasets/vibrato')
        expected = {r.recording_id for r in recordings}
        # Trusted local benchmark artifact, with the checksum verified above.
        with (prepared / 'inputs.pkl').open('rb') as handle:
            examples, inputs = pickle.load(handle)
        rows = {r['recording']: r for r in manifest['recordings']}
        if len(examples) != 6 or len(expected) != 6 or {e.case_id for e in examples} != expected:
            raise ValueError('Expected exactly all six parameter-annotated recordings.')
        variants = []
        for example in examples:
            row = rows[example.case_id]
            score_path = prepared / example.case_id / 'score.mid'
            if cls._digest(score_path) != row['score_sha256']:
                raise ValueError(f'Prepared score changed: {example.case_id}')
            recording = next(r for r in recordings if r.recording_id == example.case_id)
            if cls._digest(recording.audio_path) != row['audio_sha256']:
                raise ValueError(f'Prepared audio changed: {example.case_id}')
            # Reconstruct event-averaged references from the current annotations
            # every run, while retaining the paired pitch inputs.
            import numpy as np
            from benchmarks.modules.vibrato.datasets.YangDataset import YangDataset
            from benchmarks.modules.vibrato.datasets.YangFullDataset import YangFullDataset
            references = [dict(start=a, end=b) for a, b in YangFullDataset.areas(recording)]
            truth = np.zeros(len(example.times), dtype=bool)
            rate = np.zeros(len(example.times))
            width = np.zeros(len(example.times))
            for ref in references:
                mask = (example.times >= ref['start']) & (example.times < ref['end'])
                truth[mask] = True
                rate[mask] = width[mask] = np.nan
            for region in YangDataset._examples_from_annotations(
                    recording, example.times, example.pitch_midi, pitch_stage='common_pyin'):
                mask = region.score_mask
                rate[mask], width[mask] = region.rate_hz[mask], region.width_cents[mask]
                for ref in references:
                    if (abs(ref['start']-region.metadata['target_start_time']) < 1e-6
                            and abs(ref['end']-region.metadata['target_end_time']) < 1e-6):
                        ref.update(rate_hz=float(region.rate_hz[mask][0]),
                                   extent_semitones=float(region.width_cents[mask][0])/200.)
            example = replace(example, rate_hz=rate, width_cents=width, is_vibrato=truth,
                metadata={**example.metadata, 'yang_references': references,
                          'parameter_target_protocol': 'event_half_cycle_mean_v1'})
            print(f'Production score-aware notes: {example.case_id}', flush=True)
            config, pitch_data = cls._thaw_pitch(inputs[example.case_id])
            score = ScoreData(score_path)
            if len(score.note_datas[score.active_instrument].times) != row['score_note_count']:
                raise ValueError(f'Score import changed note count: {example.case_id}')
            take = Recording(score_data=score, config=config)
            take.pitch_data = pitch_data
            take.analyze_notes()
            bounds = [(float(n.start_time), float(n.end_time), float(n.midi_num[0]))
                      for n in take.note_data.data.values()]
            variants.append(replace(example, metadata={**example.metadata,
                'analysis_note_bounds': bounds,
                'note_boundary_source': 'production_detect_align_repeat_with_candidate_score',
                'score_match': row}))
        return variants

    @classmethod
    def coco_examples(cls, root=REPO_ROOT, workers=2):
        from benchmarks.modules.vibrato.datasets.CocoDataset import CocoDataset
        return CocoDataset.build(
            coco_root=Path(root) / 'benchmarks/datasets/cocochorales_tiny',
            output_root=Path(root) / 'benchmarks/datasets/cocochorales_vibrato/paired',
            max_stems=20, notes_per_stem=len(cls.PROFILES), profiles=cls.PROFILES,
            snrs_db=(float('inf'), 20., 10.), min_note_seconds=.75,
            injection_range='native', seed=0, adaptive_yin_window=True, workers=workers)

    @classmethod
    def run_suite(cls, suite, *, root=REPO_ROOT, workers=2):
        """Rebuild examples with current production notes; reuse matching job checkpoints."""
        from benchmarks.modules.vibrato.competitors.Attune import GatedAttune

        if suite not in {'yang', 'coco'} or workers < 1:
            raise ValueError('Choose yang/coco and a positive worker count.')
        root = Path(root)
        examples = cls.yang_examples(root) if suite == 'yang' else cls.coco_examples(root, workers)
        detectors = [GatedAttune() if d.name == 'attune' else d
                     for d in VibratoBenchmarker.available_detectors()
                     if d.name != 'herrera_bonada_yang_window']
        assert sum(d.name == 'attune' for d in detectors) == 1
        # A source-keyed directory protects earlier runs; per-job checkpoints also
        # fingerprint examples/settings and handle interrupted executions.
        source_paths = sorted(set(
            list((root / 'algorithms').glob('*.py'))
            + list((root / 'app_logic').rglob('*.py'))
            + list((root / 'benchmarks/modules/vibrato').rglob('*.py'))))
        sources = {str(p.relative_to(root)): cls._digest(p) for p in source_paths}
        import hashlib
        key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:12]
        output = root / 'benchmarks/results/vibrato_runs' / f'paired_{suite}_{key}'
        output.mkdir(parents=True, exist_ok=True)
        benchmark = VibratoBenchmarker()
        raw = benchmark.run(examples, detectors, workers=workers, strict=True,
                            cache_dir=output / 'checkpoints')
        benchmark.write_reports(raw, output)
        summary = benchmark.summarize(raw)
        summary.to_csv(output / 'comparison.csv', index=False)
        (output / 'note_bounds.json').write_text(json.dumps(
            {e.case_id: e.metadata['analysis_note_bounds'] for e in examples}, indent=2) + '\n')
        (output / 'run_config.json').write_text(json.dumps(dict(
            suite=suite, gate=GatedAttune.frame_gate(), methods=[d.name for d in detectors],
            source_sha256=sources, workers=workers, cases=[e.case_id for e in examples],
            profiles=cls.PROFILES if suite == 'coco' else None,
            boundary_sources=sorted({e.metadata['note_boundary_source'] for e in examples}),
            rate_tolerance_hz=.5, one_sided_extent_tolerance_semitones=.05,
        ), indent=2) + '\n')
        return summary, output

    @staticmethod
    def _thaw_pitch(frozen):
        from app_logic.user.ds.PitchData import PitchData
        config, origin, frames = frozen
        config = replace(config)
        pitch_data = PitchData(config)
        pitch_data.t_origin = origin
        pitch_data.load(frames)
        return config, pitch_data

    @classmethod
    def score(cls, counts, metric):
        """counts shape (..., five metrics, TP/FP/FN); soft TP is fractional credit."""
        def f1(index):
            tp, fp, fn = np.moveaxis(counts[..., index, :], -1, 0)
            denominator = 2*tp+fp+fn
            return np.divide(2*tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
        if metric == 'aggregate_f1':
            return (f1(1)+f1(2))/2
        if metric == 'aggregate_soft_f1':
            return (f1(3)+f1(4))/2
        return f1(cls.COUNT_NAMES.index(metric.removesuffix('_f1')))

    @classmethod
    def load_counts(cls, run, methods):
        """Keep sufficient statistics; validate common references across methods."""
        run = Path(run)
        records, provenance = [], {}
        for path in sorted((run/'checkpoints').glob('*.pkl.xz')):
            # Only load checkpoint files generated by this local benchmark.
            with lzma.open(path, 'rb') as handle:
                rows = pickle.load(handle)
            relevant = [r for r in rows if r['method'] in methods]
            if not relevant:
                continue
            provenance[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for r in relevant:
                if r.get('error') or r.get('skipped'):
                    continue
                h = hashlib.sha256()
                for key in ('_curve_times', '_curve_evaluation_mask', '_curve_truth_rate_hz',
                            '_curve_truth_amplitude_semitones', '_curve_truth_vibrato'):
                    arr = np.asarray(r[key])
                    h.update(key.encode()); h.update(str(arr.shape).encode()); h.update(arr.tobytes())
                record = dict(method=r['method'], case_id=r['case_id'], reference_hash=h.hexdigest(),
                              recording=r.get('meta_recording'), source=r.get('meta_track'),
                              performer=r.get('meta_performer'), ensemble=r.get('meta_ensemble'))
                for name in cls.COUNT_NAMES:
                    if name.endswith('_soft'):
                        prefix = '_soft_'+name.removesuffix('_soft')
                        tp = float(r[prefix+'_credit'])
                        fp = float(r[prefix+'_predictions'])-tp
                        fn = float(r[prefix+'_references'])-tp
                    else:
                        tp, fp, fn = (float(r[f'_{name}_{k}']) for k in ('tp','fp','fn'))
                    for key, value in zip(('tp','fp','fn'), (tp,fp,fn)):
                        record[f'{name}_{key}'] = value
                records.append(record)
        rows = pd.DataFrame(records)
        if rows.empty:
            raise ValueError('No successful selected checkpoint rows')
        if rows.duplicated(['method','case_id']).any():
            raise ValueError('Multiple checkpoint versions for one method/case; use a single frozen run')
        expected = set(json.loads((run/'run_config.json').read_text())['cases'])
        # Fail rather than silently testing a partial/stale run.
        for method in methods:
            if set(rows.loc[rows.method.eq(method), 'case_id']) != expected:
                raise ValueError(f'{method}: checkpoint coverage differs from run manifest')
        for key in ('reference_hash','recording','source','performer','ensemble'):
            if rows.groupby('case_id')[key].nunique(dropna=False).gt(1).any():
                raise ValueError(f'Paired methods disagree on {key}')
        return rows, provenance

    @classmethod
    def paired_tests(cls, rows, competitors, *, metrics=METRICS, draws=9999, seed=0):
        """Two-sided paired swaps, percentile cluster CIs, Holm across all tests."""
        if draws < 99:
            raise ValueError('At least 99 draws required')
        methods = ('attune', *competitors)
        if len(set(methods)) != len(methods) or not metrics or any(m not in cls.METRICS for m in metrics):
            raise ValueError('Select distinct methods and supported metrics')
        if rows.duplicated(['method', 'case_id']).any():
            raise ValueError('Duplicate method/case rows')
        expected = set(rows.loc[rows.method.eq('attune'), 'case_id'])
        for method in methods:
            if set(rows.loc[rows.method.eq(method), 'case_id']) != expected:
                raise ValueError('All methods must have identical case coverage')
        if rows[['cluster','stratum']].isna().any().any():
            raise ValueError('Missing cluster or stratum')
        groups = sorted(rows.cluster.unique())
        if len(groups) < 2:
            raise ValueError('At least two resampling groups required')
        if rows.groupby('cluster').stratum.nunique().gt(1).any():
            raise ValueError('A cluster crosses bootstrap strata')
        columns = [f'{name}_{key}' for name in cls.COUNT_NAMES for key in ('tp','fp','fn')]
        matrices = {}
        for method in methods:
            subset = rows.loc[rows.method.eq(method)]
            if subset.empty:
                raise ValueError(f'Missing {method}')
            matrices[method] = subset.groupby('cluster')[columns].sum().reindex(groups).to_numpy().reshape(len(groups),5,3)
        if not all(np.isfinite(v).all() and (v >= -1e-8).all() for v in matrices.values()):
            raise ValueError('Invalid counts')
        strata = rows.drop_duplicates('cluster').set_index('cluster').loc[groups,'stratum'].to_numpy()
        rng = np.random.default_rng(seed)
        boot = np.concatenate([rng.choice(np.flatnonzero(strata == s), size=(draws, sum(strata == s)))
                               for s in sorted(set(strata))], axis=1)
        exact = 2**len(groups) <= draws
        swaps = (np.array(list(itertools.product((False,True),repeat=len(groups)))) if exact
                 else rng.integers(0,2,size=(draws,len(groups))).astype(bool))
        a = matrices['attune']; results = []
        for competitor in competitors:
            b = matrices[competitor]
            swap_a = np.where(swaps[:,:,None,None], b, a).sum(axis=1)
            swap_b = np.where(swaps[:,:,None,None], a, b).sum(axis=1)
            boot_a, boot_b = a[boot].sum(axis=1), b[boot].sum(axis=1)
            for metric in metrics:
                observed = float(cls.score(a.sum(axis=0),metric)-cls.score(b.sum(axis=0),metric))
                null = cls.score(swap_a,metric)-cls.score(swap_b,metric)
                extreme = int(np.sum(np.abs(null) >= abs(observed)-1e-12))
                p = extreme/len(null) if exact else (extreme+1)/(len(null)+1)
                low,high = np.quantile(cls.score(boot_a,metric)-cls.score(boot_b,metric),[.025,.975])
                results.append(dict(competitor=competitor,metric=metric,clusters=len(groups),
                    cases=rows.case_id.nunique(),attune_score=float(cls.score(a.sum(axis=0),metric)),
                    competitor_score=float(cls.score(b.sum(axis=0),metric)),difference_pp=100*observed,
                    ci_low_pp=100*low,ci_high_pp=100*high,p_value=p,exact=exact,
                    permutations=len(null),bootstrap_draws=draws,seed=seed))
        result = pd.DataFrame(results)
        ordered = result.p_value.sort_values()
        adjusted = np.minimum(1,np.maximum.accumulate(ordered.to_numpy()*np.arange(len(ordered),0,-1)))
        result['p_holm'] = pd.Series(adjusted,index=ordered.index)
        result['significant'] = result.p_holm < .05
        return result

    @classmethod
    def run_significance(cls, run, suite, *, draws=9999, seed=0):
        """One separate family per corpus; additional performer sensitivity for Yang."""
        methods = ('attune', *cls.COMPETITORS[suite])
        rows, inputs = cls.load_counts(run,methods)
        rows['cluster'] = rows.recording if suite == 'yang' else rows.source
        rows['stratum'] = 'all' if suite == 'yang' else rows.ensemble
        if suite == 'yang' and rows.cluster.nunique() != 6:
            raise ValueError('Yang analysis expects all six recording files')
        result = cls.paired_tests(rows,cls.COMPETITORS[suite],draws=draws,seed=seed)
        output = Path(run)/'paired_significance'
        output.mkdir(exist_ok=True)
        # Confirm pooled estimates reproduce the saved notebook scores exactly.
        summary = pd.read_csv(Path(run)/'comparison.csv').set_index('method')
        for r in result.itertuples():
            for name,value in [('attune',r.attune_score),(r.competitor,r.competitor_score)]:
                if not np.isclose(value,summary.loc[name,r.metric],rtol=1e-9,atol=1e-10):
                    raise ValueError(f'Checkpoint scores differ from report: {name}/{r.metric}')
        result.to_csv(output/'paired_tests.csv',index=False)
        rows.to_csv(output/'paired_counts.csv',index=False)
        if suite == 'yang':
            sensitivity = rows.copy()
            sensitivity['cluster'] = sensitivity.performer
            cls.paired_tests(sensitivity,cls.COMPETITORS[suite],draws=draws,seed=seed).to_csv(
                output/'performer_sensitivity.csv',index=False)
        config = dict(suite=suite,methods=methods,metrics=cls.METRICS,draws=draws,seed=seed,
            alternatives='two-sided',holm_family='all competitors × all seven metrics within this corpus',
            ci='95% unadjusted percentile cluster bootstrap; Coco stratified by ensemble',
            interpretation=('Exploratory recording-file inference; excerpts share performers and repertoire. '
                            'See performer sensitivity; not independent-piece validation.' if suite=='yang' else
                            'Source-track clusters retain every stem/note/noise version; ensemble-stratified bootstrap. '
                            'Inference is conditional on sampled synthetic source tracks, not new real performances.'),
            checkpoints_sha256=inputs,module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (output/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
        return result, output

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    @classmethod
    def _f1(cls, tp: float, fp: float, fn: float) -> float:
        return cls._safe_div(2.0 * tp, 2.0 * tp + fp + fn)

    @staticmethod
    def _parameter_prf(
        tp: float,
        fp: float,
        fn: float,
    ) -> tuple[float, float, float]:
        """Precision/recall/F1 for one tolerance-matched parameter.

        Precision is undefined when a method makes no prediction, recall is
        undefined when a scoring group contains no positive truth, and F1 is
        undefined only when neither predictions nor positive truths exist.
        """
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        f1_denominator = 2.0 * tp + fp + fn
        return (
            float(tp / precision_denominator) if precision_denominator else np.nan,
            float(tp / recall_denominator) if recall_denominator else np.nan,
            float(2.0 * tp / f1_denominator) if f1_denominator else np.nan,
        )

    @staticmethod
    def _mean_if_all_finite(*values: float) -> float:
        array = np.asarray(values, dtype=np.float64)
        return (
            float(np.mean(array))
            if len(array) and np.all(np.isfinite(array))
            else np.nan
        )

    @staticmethod
    def _true_runs(mask: np.ndarray) -> list[np.ndarray]:
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if len(indices) == 0:
            return []
        return [
            group
            for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            if len(group)
        ]

    @staticmethod
    def _as_bool(values: pd.Series) -> pd.Series:
        """Coerce in-memory booleans and CSV-loaded boolean strings uniformly."""
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False)
        return values.fillna(False).astype(str).str.lower().isin({"1", "true", "yes"})

    @staticmethod
    def _finite_median(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        return float(finite.median()) if len(finite) else np.nan

    @staticmethod
    def _finite_fraction(mask: pd.Series) -> float:
        return float(mask.mean()) if len(mask) else np.nan

    @staticmethod
    def _yang_relative_accuracy(estimate: float, truth: float) -> float:
        """Equation 19 of Yang, Rajab, and Chew (2017)."""
        if not np.isfinite(estimate) or not np.isfinite(truth) or truth <= 0.0:
            return np.nan
        if estimate > 2.0 * truth:
            return 0.0
        return float(max(0.0, 1.0 - abs(estimate - truth) / truth))

    @staticmethod
    def _empty_error_row(
        estimator: VibratoDetectorBase,
        example: VibratoExample,
        elapsed: float,
        error: Exception,
        *,
        wall_elapsed: float | None = None,
        compute_clock: str = "process_cpu",
        runtime_warnings: str = "",
        audio_seconds: float | None = None,
        analysis_reuse_count: int = 1,
    ) -> dict[str, Any]:
        attributed_audio = (
            example.duration if audio_seconds is None else float(audio_seconds)
        )
        return {
            "method": estimator.name,
            "method_description": estimator.description,
            "case_id": example.case_id,
            "scenario": example.scenario,
            "family": str(example.metadata.get("family", "external")),
            "split": example.split,
            "has_vibrato": example.has_vibrato,
            "input_seconds": example.duration,
            "audio_seconds": attributed_audio,
            "scored_seconds": example.scored_duration,
            "compute_seconds": elapsed,
            "audio_per_compute": VibratoBenchmarker._safe_div(
                attributed_audio, elapsed
            ),
            "wall_compute_seconds": (
                elapsed if wall_elapsed is None else float(wall_elapsed)
            ),
            "compute_clock": compute_clock,
            "runtime_warnings": runtime_warnings,
            "analysis_reuse_count": analysis_reuse_count,
            "skipped": False,
            "skip_reason": "",
            "error": f"{type(error).__name__}: {error}",
        }

    @staticmethod
    def _empty_skip_row(
        estimator: VibratoDetectorBase,
        example: VibratoExample,
        missing: set[str],
        *,
        audio_seconds: float | None = None,
        analysis_reuse_count: int = 1,
    ) -> dict[str, Any]:
        """Represent an inapplicable estimator without turning it into a crash."""

        attributed_audio = (
            example.duration if audio_seconds is None else float(audio_seconds)
        )
        return {
            "method": estimator.name,
            "method_description": estimator.description,
            "case_id": example.case_id,
            "scenario": example.scenario,
            "family": str(example.metadata.get("family", "external")),
            "split": example.split,
            "has_vibrato": example.has_vibrato,
            "input_seconds": example.duration,
            "audio_seconds": attributed_audio,
            "scored_seconds": example.scored_duration,
            "compute_seconds": 0.0,
            "audio_per_compute": np.nan,
            "wall_compute_seconds": 0.0,
            "compute_clock": "not_timed",
            "runtime_warnings": "",
            "analysis_reuse_count": analysis_reuse_count,
            "skipped": True,
            "skip_reason": f"missing required input: {', '.join(sorted(missing))}",
            "error": "",
        }

    @staticmethod
    @contextmanager
    def _single_threaded_numerics():
        """Hold already-loaded numerical libraries to one thread.

        The module-level environment guard only reaches a library that has yet
        to load, which covers a spawned worker but not an interpreter that
        imported numpy first -- a notebook or test calling :meth:`run`
        directly, and its serial and thread-pool jobs. threadpoolctl switches
        those at runtime; without it the environment guard still stands.
        """
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            yield
        else:
            with threadpool_limits(limits=1):
                yield

    @staticmethod
    def _checkpoint_paths(benchmarker, estimators, groups, cache_dir):
        """Fingerprint inputs, settings, and implementation before jobs mutate state."""
        if cache_dir is None:
            return {}
        source = hashlib.sha256()
        for root in (REPO_ROOT / "algorithms", REPO_ROOT / "app_logic",
                     REPO_ROOT / "benchmarks/modules/vibrato"):
            for path in sorted(root.rglob("*.py")):
                if "tests" not in path.parts:
                    source.update(str(path.relative_to(REPO_ROOT)).encode())
                    source.update(path.read_bytes())
        source.update(pickle.dumps(vars(benchmarker)))
        group_keys = []
        for group in groups:
            digest = source.copy()
            digest.update(pickle.dumps(group, protocol=pickle.HIGHEST_PROTOCOL))
            # Audio methods can consume the file, not just its pitch contour.
            for name in sorted({item.audio_path for item in group if item.audio_path}):
                path = Path(name)
                stat = path.stat() if path.is_file() else None
                digest.update(repr((name, stat.st_size, stat.st_mtime_ns) if stat
                                   else (name, None)).encode())
            group_keys.append(digest)
        paths = {}
        for i, estimator in enumerate(estimators):
            try:
                state = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
            except (AttributeError, pickle.PickleError, TypeError):
                # Arbitrary notebook callables cannot be fingerprinted reliably.
                continue
            for j, digest in enumerate(group_keys):
                key = digest.copy()
                key.update(state)
                paths[i, j] = Path(cache_dir) / (key.hexdigest() + ".pkl.xz")
        return paths

    @staticmethod
    def _run_detector_group_job(
        benchmarker, estimator, group, strict, *, compute_clock, cache_path=None,
    ):
        """Persist each completed track/method even if a later job is interrupted."""
        if cache_path is not None and cache_path.is_file():
            try:
                with lzma.open(cache_path, "rb") as stream:
                    rows = pickle.load(stream)
                if (isinstance(rows, list) and len(rows) == len(group)
                        and all(isinstance(row, dict) and not row.get("error")
                                and not row.get("skipped") for row in rows)):
                    return rows
            except (EOFError, OSError, ValueError, pickle.UnpicklingError, lzma.LZMAError):
                pass  # A damaged checkpoint is a cache miss.
        rows = VibratoBenchmarker._compute_detector_group_job(
            benchmarker, estimator, group, strict, compute_clock=compute_clock,
        )
        if cache_path is not None and all(
            not row.get("error") and not row.get("skipped") for row in rows
        ):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
            os.close(fd)
            try:
                with lzma.open(temporary, "wb") as stream:
                    pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, cache_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return rows

    @staticmethod
    def _compute_detector_group_job(
        benchmarker: VibratoBenchmarker,
        estimator: VibratoDetectorBase,
        group: list[VibratoExample],
        strict: bool,
        *,
        compute_clock: str,
    ) -> list[dict[str, Any]]:
        """Score one detector/input group with a worker-local CPU clock."""
        if compute_clock not in {"process_cpu", "thread_cpu"}:
            raise ValueError(f"unknown compute clock: {compute_clock}")
        cpu_now = (
            time.process_time if compute_clock == "process_cpu" else time.thread_time
        )
        group_rows: list[dict[str, Any]] = []
        reference = group[0]
        reuse_count = len(group)
        requires = set(getattr(estimator, "requires", {"pitch"}))
        missing = requires - benchmarker._available_inputs(reference)
        if missing:
            audio_share = reference.duration / reuse_count
            for example in group:
                group_rows.append(
                    VibratoBenchmarker._empty_skip_row(
                        estimator,
                        example,
                        missing,
                        audio_seconds=audio_share,
                        analysis_reuse_count=reuse_count,
                    )
                )
            return group_rows

        cpu_started: float | None = None
        wall_started: float | None = None
        caught: list[warnings.WarningMessage] = []

        def warning_text() -> str:
            messages = dict.fromkeys(
                f"{warning.category.__name__}: {warning.message} "
                f"[{Path(warning.filename).name}:{warning.lineno}]"
                for warning in caught
            )
            return " | ".join(messages)

        try:
            with (
                VibratoBenchmarker._single_threaded_numerics(),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                # Optional input preparation remains outside both timers. Audio
                # methods decode a shared file only once for the analysis group.
                prepare = getattr(estimator, "prepare", None)
                if prepare is not None:
                    prepare(reference)
                cpu_started = cpu_now()
                wall_started = time.perf_counter()
                estimate = estimator.estimate(reference)
            cpu_elapsed = max(cpu_now() - cpu_started, np.finfo(float).eps)
            wall_elapsed = max(
                time.perf_counter() - wall_started,
                np.finfo(float).eps,
            )
            cpu_share = cpu_elapsed / reuse_count
            wall_share = wall_elapsed / reuse_count
            audio_share = reference.duration / reuse_count
            for example in group:
                group_rows.append(
                    benchmarker._score(
                        estimator,
                        example,
                        estimate,
                        cpu_share,
                        wall_elapsed=wall_share,
                        compute_clock=compute_clock,
                        runtime_warnings=warning_text(),
                        audio_seconds=audio_share,
                        analysis_reuse_count=reuse_count,
                    )
                )
        except Exception as error:
            cpu_elapsed = (
                max(cpu_now() - cpu_started, np.finfo(float).eps)
                if cpu_started is not None
                else np.finfo(float).eps
            )
            wall_elapsed = (
                max(time.perf_counter() - wall_started, np.finfo(float).eps)
                if wall_started is not None
                else np.finfo(float).eps
            )
            if strict:
                raise
            cpu_share = cpu_elapsed / reuse_count
            wall_share = wall_elapsed / reuse_count
            audio_share = reference.duration / reuse_count
            for example in group:
                group_rows.append(
                    VibratoBenchmarker._empty_error_row(
                        estimator,
                        example,
                        cpu_share,
                        error,
                        wall_elapsed=wall_share,
                        compute_clock=compute_clock,
                        runtime_warnings=warning_text(),
                        audio_seconds=audio_share,
                        analysis_reuse_count=reuse_count,
                    )
                )
        return group_rows

    @staticmethod
    def available_detectors() -> list[VibratoDetectorBase]:
        from benchmarks.modules.vibrato.competitors.Attune import Attune
        from benchmarks.modules.vibrato.competitors.Driedger import Driedger
        from benchmarks.modules.vibrato.competitors.HerreraBonada import HerreraBonada
        from benchmarks.modules.vibrato.competitors.McLeod import McLeod
        from benchmarks.modules.vibrato.competitors.Rossignol import Rossignol
        from benchmarks.modules.vibrato.competitors.VenturaSousaFerreira import (
            VenturaSousaFerreira,
        )
        from benchmarks.modules.vibrato.competitors.Yang import YangBR, YangDT

        return [
            Rossignol(),
            HerreraBonada(),
            HerreraBonada(window_mode="yang_comparison"),
            VenturaSousaFerreira(),
            Driedger(),
            Driedger(template_mode="benchmark_range"),
            YangDT(),
            YangBR(),
            McLeod(),
            Attune(),
        ]

    @classmethod
    def default_detectors(cls) -> list[VibratoDetectorBase]:
        ablations = {
            "driedger_benchmark_range",
            "herrera_bonada_yang_window",
        }
        return [
            detector
            for detector in cls.available_detectors()
            if detector.name not in ablations
        ]

    def __init__(
        self,
        *,
        rate_accuracy_tolerance_hz: float = 0.5,
        amplitude_accuracy_tolerance_semitones: float = 0.05,
        center_accuracy_tolerance_cents: float = 25.0,
    ) -> None:
        if rate_accuracy_tolerance_hz < 0.0:
            raise ValueError("rate_accuracy_tolerance_hz cannot be negative")
        if amplitude_accuracy_tolerance_semitones < 0.0:
            raise ValueError(
                "amplitude_accuracy_tolerance_semitones cannot be negative"
            )
        if center_accuracy_tolerance_cents < 0.0:
            raise ValueError("center_accuracy_tolerance_cents cannot be negative")
        self.rate_accuracy_tolerance_hz = float(rate_accuracy_tolerance_hz)
        self.amplitude_accuracy_tolerance_semitones = float(
            amplitude_accuracy_tolerance_semitones
        )
        self.center_accuracy_tolerance_cents = float(center_accuracy_tolerance_cents)

    def _score(
        self,
        estimator: VibratoDetectorBase,
        example: VibratoExample,
        estimate: VibratoEstimate,
        elapsed: float,
        *,
        wall_elapsed: float | None = None,
        compute_clock: str = "process_cpu",
        runtime_warnings: str = "",
        audio_seconds: float | None = None,
        analysis_reuse_count: int = 1,
    ) -> dict[str, Any]:
        estimate.validate_for(example)
        truth = np.asarray(example.is_vibrato, dtype=bool)
        detected = np.asarray(estimate.detected, dtype=bool)
        evaluation = example.score_mask
        raw_fit_confidence = estimate.metadata.get("fit_confidence")
        if raw_fit_confidence is None:
            fit_confidence = np.full(len(truth), np.nan, dtype=np.float64)
        else:
            fit_confidence = np.asarray(raw_fit_confidence, dtype=np.float64)
            if fit_confidence.shape != (len(truth),):
                raise ValueError("fit_confidence must match the example grid")
            if (
                not np.isfinite(fit_confidence).all()
                or np.any(fit_confidence <= 0.0)
                or np.any(fit_confidence > 1.0)
            ):
                raise ValueError("fit_confidence must be finite and in (0, 1]")
        evaluated_fit_confidence = fit_confidence[evaluation]
        has_fit_confidence = bool(np.isfinite(evaluated_fit_confidence).all())
        tapered_fit_frames = (
            int(np.sum(evaluated_fit_confidence < 1.0)) if has_fit_confidence else 0
        )
        tp = int(np.sum(evaluation & truth & detected))
        fp = int(np.sum(evaluation & ~truth & detected))
        fn = int(np.sum(evaluation & truth & ~detected))
        tn = int(np.sum(evaluation & ~truth & ~detected))

        # VibratoData stores full peak-to-peak width in cents. The benchmark
        # presents the musically conventional one-sided amplitude in semitones:
        # a width of 100 cents is therefore an amplitude of 0.5 semitones.
        true_amplitude = (
            np.abs(np.asarray(example.width_cents, dtype=np.float64)) / 200.0
        )
        estimated_amplitude = np.where(
            np.isfinite(estimate.width_cents),
            np.abs(np.asarray(estimate.width_cents, dtype=np.float64)) / 200.0,
            0.0,
        )
        estimated_rate = np.where(
            np.isfinite(estimate.rate_hz),
            np.maximum(np.asarray(estimate.rate_hz, dtype=np.float64), 0.0),
            0.0,
        )
        amplitude_mask = evaluation & np.isfinite(true_amplitude)
        rate_mask = (
            evaluation
            & truth
            & amplitude_mask
            & np.isfinite(example.rate_hz)
            & (true_amplitude > 0.0)
            & (np.asarray(example.rate_hz, dtype=np.float64) > 0.0)
        )

        # Missing estimates are reconstructed as zero rather than omitted in
        # the retained curve-error diagnostics. The public parameter PRF below
        # uses the method's final decision and treats the same frame as a miss.
        rate_errors = np.abs(estimated_rate[rate_mask] - example.rate_hz[rate_mask])
        amplitude_errors = np.abs(
            estimated_amplitude[amplitude_mask] - true_amplitude[amplitude_mask]
        )
        active_amplitude_errors = np.abs(
            estimated_amplitude[rate_mask] - true_amplitude[rate_mask]
        )
        # Retain a per-frame version of Yang et al.'s Equation 19 as a detailed
        # diagnostic. The original paper first reduced each vibrato to one
        # rate/extent value; applying the same relative score per frame keeps
        # time-varying injected curves auditable. It is no longer a public
        # leaderboard metric because it has no negative-case penalty.
        relative_extent_scores = np.maximum(
            0.0,
            1.0 - active_amplitude_errors / true_amplitude[rate_mask],
        )
        relative_rate_scores = np.maximum(
            0.0,
            1.0 - rate_errors / example.rate_hz[rate_mask],
        )
        relative_extent_accuracy = (
            float(np.mean(relative_extent_scores))
            if len(relative_extent_scores)
            else np.nan
        )
        relative_rate_accuracy = (
            float(np.mean(relative_rate_scores))
            if len(relative_rate_scores)
            else np.nan
        )
        amplitude_within = (
            np.abs(estimated_amplitude - true_amplitude)
            <= self.amplitude_accuracy_tolerance_semitones
        )
        rate_within = np.zeros(len(truth), dtype=bool)
        rate_within[rate_mask] = rate_errors <= self.rate_accuracy_tolerance_hz
        # Parameter precision/recall uses the method's final binary decision,
        # not an ungated internal estimate. A prediction matches only when the
        # method declares vibrato and the parameter is within its configured
        # tolerance. A wrong-valued prediction on a positive frame is both a
        # false positive and a false negative, as in ordinary matched-event
        # scoring. Predictions on negative controls are false positives.
        parameter_truth = rate_mask
        parameter_evaluation = evaluation & (~truth | parameter_truth)
        if example.metadata.get("parameter_annotations") is False:
            parameter_evaluation[:] = False
        extent_predicted = parameter_evaluation & detected & (estimated_amplitude > 0.0)
        rate_predicted = parameter_evaluation & detected & (estimated_rate > 0.0)
        extent_matched = parameter_truth & extent_predicted & amplitude_within
        rate_matched = parameter_truth & rate_predicted & rate_within
        extent_tp = int(np.sum(extent_matched))
        extent_fp = int(np.sum(extent_predicted & ~extent_matched))
        extent_fn = int(np.sum(parameter_truth & ~extent_matched))
        rate_tp = int(np.sum(rate_matched))
        rate_fp = int(np.sum(rate_predicted & ~rate_matched))
        rate_fn = int(np.sum(parameter_truth & ~rate_matched))
        extent_precision, extent_recall, extent_f1 = VibratoBenchmarker._parameter_prf(
            extent_tp,
            extent_fp,
            extent_fn,
        )
        rate_precision, rate_recall, rate_f1 = VibratoBenchmarker._parameter_prf(
            rate_tp,
            rate_fp,
            rate_fn,
        )
        aggregate_precision = VibratoBenchmarker._mean_if_all_finite(
            extent_precision,
            rate_precision,
        )
        aggregate_recall = VibratoBenchmarker._mean_if_all_finite(
            extent_recall,
            rate_recall,
        )
        aggregate_f1 = VibratoBenchmarker._mean_if_all_finite(extent_f1, rate_f1)
        curve_within = amplitude_within.copy()
        curve_within[rate_mask] &= rate_within[rate_mask]
        curve_mask = amplitude_mask

        # Center is a property of the reconstructed vibrato curve, so it is
        # scored on the same positive ground-truth frames as rate and extent.
        # The public center-inclusive statistic is deliberately Attune-only.
        # Missing Attune center frames are failures; every comparison method
        # remains N/A even if its adapter happens to expose a center track.
        score_center = bool(
            getattr(
                estimator,
                "scores_center",
                False,
            )
        )
        center_truth_mask = rate_mask & np.isfinite(example.center_midi)
        center_reported_mask = np.zeros(len(truth), dtype=bool)
        center_errors = np.empty(0, dtype=np.float64)
        if score_center and estimate.center_midi is not None:
            center_reported_mask = center_truth_mask & np.isfinite(estimate.center_midi)
            center_errors = 100.0 * np.abs(
                estimate.center_midi[center_reported_mask]
                - example.center_midi[center_reported_mask]
            )
        center_n = int(np.sum(center_truth_mask)) if score_center else 0
        center_reported_n = len(center_errors)
        center_within = int(
            np.sum(center_errors <= self.center_accuracy_tolerance_cents)
        )
        reported_mask = (
            rate_mask
            & np.isfinite(estimate.rate_hz)
            & np.isfinite(estimate.width_cents)
            & (estimated_rate > 0.0)
            & (estimated_amplitude > 0.0)
        )

        # Yang et al. did not use an absolute rate/extent tolerance.  They
        # aggregated each corresponding detected vibrato to one parameter and
        # scored relative error with Equation 19.  A detected run corresponds
        # to this ground-truth interval when at least half of the detected run
        # lies inside it; multiple corresponding runs are averaged.  Unmatched
        # ground-truth vibratos are excluded from their parameter comparison,
        # so coverage is reported alongside the matched-only accuracy.
        corresponding_runs = (
            [
                run
                for run in VibratoBenchmarker._true_runs(detected)
                if float(np.mean(evaluation[run])) >= 0.5
            ]
            if np.any(rate_mask)
            else []
        )
        yang_parameter_matched = bool(corresponding_runs)
        yang_rate_accuracy = np.nan
        yang_extent_accuracy = np.nan
        if yang_parameter_matched:
            run_rates = [
                float(np.mean(estimated_rate[run])) for run in corresponding_runs
            ]
            run_extents = [
                float(np.mean(estimated_amplitude[run])) for run in corresponding_runs
            ]
            yang_rate_accuracy = VibratoBenchmarker._yang_relative_accuracy(
                float(np.mean(run_rates)),
                float(np.mean(example.rate_hz[rate_mask])),
            )
            yang_extent_accuracy = VibratoBenchmarker._yang_relative_accuracy(
                float(np.mean(run_extents)),
                float(np.mean(true_amplitude[rate_mask])),
            )
        yang_overall_accuracy = VibratoBenchmarker._mean_if_all_finite(
            yang_extent_accuracy,
            yang_rate_accuracy,
        )

        attributed_audio = (
            example.duration if audio_seconds is None else float(audio_seconds)
        )
        extent_within_tolerance = (
            float(np.mean(amplitude_within[rate_mask])) if np.any(rate_mask) else np.nan
        )
        rate_within_tolerance_accuracy = (
            float(np.mean(rate_errors <= self.rate_accuracy_tolerance_hz))
            if len(rate_errors)
            else np.nan
        )
        center_accuracy = (
            VibratoBenchmarker._safe_div(center_within, center_n)
            if center_n
            else np.nan
        )
        row: dict[str, Any] = {
            "method": estimator.name,
            "method_description": estimator.description,
            "case_id": example.case_id,
            "scenario": example.scenario,
            "family": str(example.metadata.get("family", "external")),
            "split": example.split,
            "has_vibrato": example.has_vibrato,
            "frame_accuracy": VibratoBenchmarker._safe_div(tp + tn, tp + fp + fn + tn),
            "frame_precision": VibratoBenchmarker._safe_div(tp, tp + fp),
            "frame_recall": VibratoBenchmarker._safe_div(tp, tp + fn),
            "frame_f1": VibratoBenchmarker._f1(tp, fp, fn),
            "aggregate_precision": aggregate_precision,
            "aggregate_recall": aggregate_recall,
            "aggregate_f1": aggregate_f1,
            "extent_precision": extent_precision,
            "extent_recall": extent_recall,
            "extent_f1": extent_f1,
            "rate_precision": rate_precision,
            "rate_recall": rate_recall,
            "rate_f1": rate_f1,
            "overall_curve_accuracy": VibratoBenchmarker._mean_if_all_finite(
                relative_extent_accuracy,
                relative_rate_accuracy,
            ),
            "relative_extent_accuracy": relative_extent_accuracy,
            "relative_rate_accuracy": relative_rate_accuracy,
            "curve_within_tolerance": (
                float(np.mean(curve_within[curve_mask]))
                if np.any(curve_mask)
                else np.nan
            ),
            "active_curve_within_tolerance": (
                float(np.mean(curve_within[rate_mask])) if np.any(rate_mask) else np.nan
            ),
            "estimate_coverage": VibratoBenchmarker._safe_div(
                int(np.sum(reported_mask)), int(np.sum(rate_mask))
            ),
            "yang_overall_accuracy": yang_overall_accuracy,
            "yang_extent_accuracy": yang_extent_accuracy,
            "yang_rate_accuracy": yang_rate_accuracy,
            "yang_parameter_matched": yang_parameter_matched,
            "rate_mae_hz": float(np.mean(rate_errors)) if len(rate_errors) else np.nan,
            "rate_rmse_hz": (
                float(np.sqrt(np.mean(rate_errors**2))) if len(rate_errors) else np.nan
            ),
            "rate_within_tolerance": rate_within_tolerance_accuracy,
            "amplitude_mae_semitones": (
                float(np.mean(amplitude_errors)) if len(amplitude_errors) else np.nan
            ),
            "amplitude_rmse_semitones": (
                float(np.sqrt(np.mean(amplitude_errors**2)))
                if len(amplitude_errors)
                else np.nan
            ),
            "amplitude_within_tolerance": (
                float(np.mean(amplitude_within[amplitude_mask]))
                if np.any(amplitude_mask)
                else np.nan
            ),
            "active_amplitude_mae_semitones": (
                float(np.mean(active_amplitude_errors))
                if len(active_amplitude_errors)
                else np.nan
            ),
            "active_amplitude_rmse_semitones": (
                float(np.sqrt(np.mean(active_amplitude_errors**2)))
                if len(active_amplitude_errors)
                else np.nan
            ),
            "active_amplitude_within_tolerance": extent_within_tolerance,
            "center_within_tolerance": center_accuracy,
            "center_coverage": (
                VibratoBenchmarker._safe_div(center_reported_n, center_n)
                if center_n
                else np.nan
            ),
            "center_mae_cents": (
                float(np.mean(center_errors)) if len(center_errors) else np.nan
            ),
            "fit_confidence_min": (
                float(np.min(evaluated_fit_confidence))
                if has_fit_confidence
                else np.nan
            ),
            "fit_confidence_mean": (
                float(np.mean(evaluated_fit_confidence))
                if has_fit_confidence
                else np.nan
            ),
            "fit_confidence_tapered_frames": (
                tapered_fit_frames if has_fit_confidence else np.nan
            ),
            "fit_confidence_tapered_fraction": (
                VibratoBenchmarker._safe_div(
                    tapered_fit_frames, len(evaluated_fit_confidence)
                )
                if has_fit_confidence
                else np.nan
            ),
            "input_seconds": example.duration,
            "audio_seconds": attributed_audio,
            "scored_seconds": example.scored_duration,
            "compute_seconds": elapsed,
            "audio_per_compute": VibratoBenchmarker._safe_div(
                attributed_audio, elapsed
            ),
            "wall_compute_seconds": (
                elapsed if wall_elapsed is None else float(wall_elapsed)
            ),
            "compute_clock": compute_clock,
            "runtime_warnings": runtime_warnings,
            "analysis_reuse_count": analysis_reuse_count,
            "skipped": False,
            "skip_reason": "",
            "error": "",
            "_frame_tp": tp,
            "_frame_fp": fp,
            "_frame_fn": fn,
            "_frame_tn": tn,
            "_extent_tp": extent_tp,
            "_extent_fp": extent_fp,
            "_extent_fn": extent_fn,
            "_rate_tp": rate_tp,
            "_rate_fp": rate_fp,
            "_rate_fn": rate_fn,
            "_rate_abs_sum": float(np.sum(rate_errors)),
            "_rate_sq_sum": float(np.sum(rate_errors**2)),
            "_rate_within": int(np.sum(rate_errors <= self.rate_accuracy_tolerance_hz)),
            "_relative_rate_sum": float(np.sum(relative_rate_scores)),
            "_relative_extent_sum": float(np.sum(relative_extent_scores)),
            "_amplitude_abs_sum": float(np.sum(amplitude_errors)),
            "_amplitude_sq_sum": float(np.sum(amplitude_errors**2)),
            "_amplitude_within": int(np.sum(amplitude_within[amplitude_mask])),
            "_amplitude_n": len(amplitude_errors),
            "_active_amplitude_abs_sum": float(np.sum(active_amplitude_errors)),
            "_active_amplitude_sq_sum": float(np.sum(active_amplitude_errors**2)),
            "_active_amplitude_within": int(np.sum(amplitude_within[rate_mask])),
            "_active_amplitude_n": len(active_amplitude_errors),
            "_rate_n": len(rate_errors),
            "_reported_n": int(np.sum(reported_mask)),
            "_yang_extent_accuracy": yang_extent_accuracy,
            "_yang_rate_accuracy": yang_rate_accuracy,
            "_yang_parameter_matched": int(yang_parameter_matched),
            "_yang_parameter_truth": int(np.any(rate_mask)),
            "_curve_within": int(np.sum(curve_within[curve_mask])),
            "_curve_n": int(np.sum(curve_mask)),
            "_active_curve_within": int(np.sum(curve_within[rate_mask])),
            "_active_curve_n": int(np.sum(rate_mask)),
            "_center_abs_sum": float(np.sum(center_errors)),
            "_center_within": center_within,
            "_center_n": center_n,
            "_center_reported_n": center_reported_n,
            "_curve_times": np.asarray(example.times, dtype=np.float64),
            "_curve_evaluation_mask": evaluation,
            "_curve_truth_rate_hz": np.asarray(example.rate_hz, dtype=np.float64),
            "_curve_estimated_rate_hz": estimated_rate,
            "_curve_truth_amplitude_semitones": true_amplitude,
            "_curve_estimated_amplitude_semitones": estimated_amplitude,
            "_curve_truth_vibrato": truth,
            "_curve_detected": detected,
            "_curve_extent_predicted": extent_predicted,
            "_curve_extent_matched": extent_matched,
            "_curve_rate_predicted": rate_predicted,
            "_curve_rate_matched": rate_matched,
            "_curve_frame_within_tolerance": curve_within,
            "_curve_truth_center_midi": np.asarray(
                example.center_midi,
                dtype=np.float64,
            ),
            "_curve_commanded_pitch_midi": (
                np.asarray(example.commanded_pitch_midi, dtype=np.float64)
                if example.commanded_pitch_midi is not None
                else np.full(len(truth), np.nan, dtype=np.float64)
            ),
            "_curve_raw_pitch_midi": (
                np.asarray(example.raw_pitch_midi, dtype=np.float64)
                if example.raw_pitch_midi is not None
                else np.full(len(truth), np.nan, dtype=np.float64)
            ),
            "_curve_smoothed_pitch_midi": np.asarray(
                example.pitch_midi,
                dtype=np.float64,
            ),
            "_curve_pitch_transition": (
                np.asarray(example.transition_mask, dtype=bool)
                if example.transition_mask is not None
                else np.zeros(len(truth), dtype=bool)
            ),
            "_curve_fit_confidence": fit_confidence,
            "_curve_estimated_center_midi": (
                np.asarray(estimate.center_midi, dtype=np.float64)
                if estimate.center_midi is not None
                else np.full(len(truth), np.nan, dtype=np.float64)
            ),
        }
        # Soft parameter scores exclude unknown positive annotations, never
        # treating an unannotated vibrato as a negative. Area-only recordings
        # have no parameter score at all.
        eligible = evaluation.copy()
        if example.metadata.get("parameter_annotations") is False:
            eligible[:] = False
        for name, reference, estimated in (
            ("extent", true_amplitude, estimated_amplitude),
            ("rate", np.asarray(example.rate_hz), estimated_rate),
        ):
            credit, predictions, references = YangMetrics.soft_counts(
                truth, detected, reference, estimated, eligible)
            n = int(np.sum(eligible & (~truth | (np.isfinite(reference) & (reference > 0)))))
            metrics = YangMetrics.soft_prf(credit, predictions, references) if n else (np.nan,)*3
            for metric, value in zip(("precision", "recall", "f1"), metrics):
                row[f"{name}_soft_{metric}"] = value
            for key, value in zip(("credit", "predictions", "references", "frames"), (credit, predictions, references, n)):
                row[f"_soft_{name}_{key}"] = value
        for metric in ("precision", "recall", "f1"):
            row[f"aggregate_soft_{metric}"] = self._mean_if_all_finite(
                row[f"extent_soft_{metric}"], row[f"rate_soft_{metric}"])
        if example.metadata.get("parameter_annotations") is False:
            for name in ("extent", "rate", "aggregate"):
                for metric in ("precision", "recall", "f1"):
                    row[f"{name}_{metric}"] = np.nan
        if "yang_references" in example.metadata:
            event = YangMetrics.evaluate(example.times, detected, estimated_rate,
                estimated_amplitude, example.metadata["yang_references"])
            row.update(event)
            row.update(YangMetrics.summary(event))
            row["yang_parameter_coverage"] = event["yang_parameter_matched"] / event["yang_parameter_truth"] if event["yang_parameter_truth"] else np.nan
            row.update({f"_event_{key}": value for key, value in event.items()})
            count = event["yang_parameter_matched"]
            for parameter in ("extent", "rate"):
                value = event[f"yang_{parameter}_sum"] / count if count else np.nan
                row[f"yang_{parameter}_accuracy"] = value
                row[f"_yang_{parameter}_accuracy"] = value
            row["_yang_parameter_truth"] = event["yang_parameter_truth"]
            row["_yang_parameter_matched"] = count
            row["yang_parameter_matched"] = count
            row["yang_overall_accuracy"] = self._mean_if_all_finite(
                row["yang_extent_accuracy"], row["yang_rate_accuracy"])
        row.update(
            {
                f"meta_{key}": value
                for key, value in example.metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
        )
        return row

    @staticmethod
    def _analysis_groups(
        examples: Sequence[VibratoExample],
    ) -> list[list[VibratoExample]]:
        """Group scoring crops that share one estimator invocation.

        Generated corpora mark complete, continuous contours with an analysis
        group and every note boundary. External/legacy cases remain one call
        per example unless they opt into that same explicit contract.
        """
        grouped: dict[tuple[str, str], list[VibratoExample]] = {}
        for index, example in enumerate(examples):
            shared = (
                bool(example.metadata.get("continuous_context"))
                and bool(example.metadata.get("analysis_group"))
                and bool(example.metadata.get("analysis_note_bounds"))
            )
            key = (
                ("shared", str(example.metadata["analysis_group"]))
                if shared
                else ("case", str(index))
            )
            grouped.setdefault(key, []).append(example)

        for key, group in grouped.items():
            reference = group[0]
            for example in group[1:]:
                same_times = np.array_equal(reference.times, example.times)
                same_pitch = np.array_equal(
                    reference.pitch_midi,
                    example.pitch_midi,
                    equal_nan=True,
                )
                same_transitions = np.array_equal(
                    (
                        np.asarray(reference.transition_mask, dtype=bool)
                        if reference.transition_mask is not None
                        else np.zeros(len(reference.times), dtype=bool)
                    ),
                    (
                        np.asarray(example.transition_mask, dtype=bool)
                        if example.transition_mask is not None
                        else np.zeros(len(example.times), dtype=bool)
                    ),
                )
                same_notes = reference.metadata.get(
                    "analysis_note_bounds"
                ) == example.metadata.get("analysis_note_bounds")
                same_audio = reference.audio_path == example.audio_path
                if not (
                    same_times
                    and same_pitch
                    and same_transitions
                    and same_notes
                    and same_audio
                ):
                    raise ValueError(
                        f"analysis group {key[1]!r} does not share one contour "
                        "transition mask, audio input, and note-boundary set"
                    )
        return list(grouped.values())

    @staticmethod
    def _available_inputs(example: VibratoExample) -> set[str]:
        available: set[str] = set()
        if np.any(np.isfinite(np.asarray(example.pitch_midi, dtype=np.float64))):
            available.add("pitch")
        if example.audio_path and Path(example.audio_path).is_file():
            available.add("audio")
        return available

    def run(
        self,
        examples: Sequence[VibratoExample],
        estimators: Sequence[VibratoDetectorBase],
        *,
        progress: ProgressCallback | None = None,
        strict: bool = False,
        workers: int = 1,
        cache_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Run independent detector/analysis-group jobs in parallel."""
        if workers < 1:
            raise ValueError("workers must be positive")
        scoring_started = time.perf_counter()
        total = len(examples) * len(estimators)
        analysis_groups = self._analysis_groups(examples)
        checkpoints = self._checkpoint_paths(self, estimators, analysis_groups, cache_dir)
        for estimator in estimators:
            requires = set(getattr(estimator, "requires", {"pitch"}))
            unknown = requires - {"pitch", "audio"}
            if unknown:
                raise ValueError(
                    f"{estimator.name} declares unknown inputs: {sorted(unknown)}"
                )

        def finish(
            rows: list[dict[str, Any]],
            backends: list[str],
        ) -> pd.DataFrame:
            raw = pd.DataFrame(rows)
            raw.attrs["scoring_wall_seconds"] = max(
                time.perf_counter() - scoring_started,
                np.finfo(float).eps,
            )
            raw.attrs["scoring_backends"] = tuple(backends)
            raw.attrs["compute_timing_basis"] = (
                "worker_cpu_seconds_per_estimator_analysis_group"
            )
            return raw

        if workers == 1 or len(analysis_groups) * len(estimators) <= 1:
            rows: list[dict[str, Any]] = []
            index = 0
            for estimator_index, estimator in enumerate(estimators):
                for group_index, group in enumerate(analysis_groups):
                    for example in group:
                        index += 1
                        if progress is not None:
                            progress(estimator.name, index, total, example)
                    rows.extend(
                        VibratoBenchmarker._run_detector_group_job(
                            self,
                            estimator,
                            group,
                            strict,
                            compute_clock="process_cpu",
                            cache_path=checkpoints.get((estimator_index, group_index)),
                        )
                    )
            return finish(rows, ["serial"])

        ordered_results: dict[tuple[int, int], list[dict[str, Any]]] = {}
        completed_cases = 0
        job_count = len(estimators) * len(analysis_groups)
        process_estimators: set[int] = set()
        thread_estimators: set[int] = set()
        for estimator_index, estimator in enumerate(estimators):
            try:
                pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
            except (AttributeError, pickle.PickleError, TypeError):
                thread_estimators.add(estimator_index)
            else:
                process_estimators.add(estimator_index)

        def retain(future, metadata) -> None:
            nonlocal completed_cases
            estimator_index, group_index, method, group = metadata
            try:
                ordered_results[(estimator_index, group_index)] = future.result()
            except Exception:
                raise
            for example in group:
                completed_cases += 1
                if progress is not None:
                    progress(method, completed_cases, total, example)

        backends: list[str] = []
        process_jobs = [
            (estimator_index, group_index, estimator, group)
            for estimator_index, estimator in enumerate(estimators)
            if estimator_index in process_estimators
            for group_index, group in enumerate(analysis_groups)
        ]
        if process_jobs:
            backends.append("spawn_process_pool")
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(workers, len(process_jobs)),
                mp_context=context,
            ) as pool:
                futures = {
                    pool.submit(
                        VibratoBenchmarker._run_detector_group_job,
                        self,
                        estimator,
                        group,
                        strict,
                        compute_clock="process_cpu",
                        cache_path=checkpoints.get((estimator_index, group_index)),
                    ): (
                        estimator_index,
                        group_index,
                        estimator.name,
                        group,
                    )
                    for estimator_index, group_index, estimator, group in process_jobs
                }
                for future in as_completed(futures):
                    try:
                        retain(future, futures[future])
                    except Exception:
                        for pending in futures:
                            pending.cancel()
                        raise

        thread_jobs = [
            (estimator_index, group_index, estimator, group)
            for estimator_index, estimator in enumerate(estimators)
            if estimator_index in thread_estimators
            for group_index, group in enumerate(analysis_groups)
        ]
        if thread_jobs:
            backends.append("thread_fallback_for_unpickleable_estimators")
        with ThreadPoolExecutor(
            max_workers=min(workers, len(thread_jobs)) if thread_jobs else 1
        ) as pool:
            futures = {
                pool.submit(
                    VibratoBenchmarker._run_detector_group_job,
                    self,
                    copy.copy(estimator),
                    group,
                    strict,
                    compute_clock="thread_cpu",
                    cache_path=checkpoints.get((estimator_index, group_index)),
                ): (
                    estimator_index,
                    group_index,
                    estimator.name,
                    group,
                )
                for estimator_index, group_index, estimator, group in thread_jobs
            }
            for future in as_completed(futures):
                try:
                    retain(future, futures[future])
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise

        rows = [
            row
            for estimator_index in range(len(estimators))
            for group_index in range(len(analysis_groups))
            for row in ordered_results[(estimator_index, group_index)]
        ]
        return finish(rows, backends)

    @staticmethod
    def summarize(
        raw: pd.DataFrame, group_by: Iterable[str] = ("method",)
    ) -> pd.DataFrame:
        group_columns = list(group_by)
        summaries: list[dict[str, Any]] = []
        for key, group in raw.groupby(group_columns, sort=False, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            skipped_mask = (
                group.get(
                    "skipped",
                    pd.Series(False, index=group.index),
                )
                .fillna(False)
                .astype(bool)
            )
            error_mask = group["error"].fillna("") != ""
            good = group[~skipped_mask & ~error_mask]
            summary = dict(zip(group_columns, key))
            summary["cases"] = len(good)
            summary["skipped"] = int(skipped_mask.sum())
            summary["errors"] = int(error_mask.sum())
            if good.empty:
                summaries.append(summary)
                continue

            frame_tp = float(good["_frame_tp"].sum())
            frame_fp = float(good["_frame_fp"].sum())
            frame_fn = float(good["_frame_fn"].sum())
            frame_tn = float(good["_frame_tn"].sum())
            extent_tp = float(good["_extent_tp"].sum())
            extent_fp = float(good["_extent_fp"].sum())
            extent_fn = float(good["_extent_fn"].sum())
            rate_tp = float(good["_rate_tp"].sum())
            rate_fp = float(good["_rate_fp"].sum())
            rate_fn = float(good["_rate_fn"].sum())
            extent_precision, extent_recall, extent_f1 = (
                VibratoBenchmarker._parameter_prf(
                    extent_tp,
                    extent_fp,
                    extent_fn,
                )
            )
            rate_precision, rate_recall, rate_f1 = VibratoBenchmarker._parameter_prf(
                rate_tp,
                rate_fp,
                rate_fn,
            )
            if not float(good["_soft_extent_frames"].sum()):
                extent_precision = extent_recall = extent_f1 = np.nan
            if not float(good["_soft_rate_frames"].sum()):
                rate_precision = rate_recall = rate_f1 = np.nan
            rate_n = float(good["_rate_n"].sum())
            amplitude_n = float(good["_amplitude_n"].sum())
            active_amplitude_n = float(good["_active_amplitude_n"].sum())
            curve_n = float(good["_curve_n"].sum())
            active_curve_n = float(good["_active_curve_n"].sum())
            center_n = float(good["_center_n"].sum())
            center_reported_n = float(good["_center_reported_n"].sum())
            yang_truth_n = float(good["_yang_parameter_truth"].sum())
            yang_matched_n = float(good["_yang_parameter_matched"].sum())
            yang_extent_values = good["_yang_extent_accuracy"].dropna()
            yang_rate_values = good["_yang_rate_accuracy"].dropna()
            yang_extent_accuracy = (
                float(yang_extent_values.mean()) if len(yang_extent_values) else np.nan
            )
            yang_rate_accuracy = (
                float(yang_rate_values.mean()) if len(yang_rate_values) else np.nan
            )

            event_columns = [c for c in good if c.startswith("_event_")]
            if event_columns:
                event = {c.removeprefix("_event_"): float(good[c].sum()) for c in event_columns}
                yang_extent_accuracy = event["yang_extent_sum"] / yang_matched_n if yang_matched_n else np.nan
                yang_rate_accuracy = event["yang_rate_sum"] / yang_matched_n if yang_matched_n else np.nan
                summary.update(YangMetrics.summary(event))
                for metric in ("precision", "recall", "f1"):
                    summary[f"yang_frame_macro_{metric}"] = float(good[f"frame_{metric}"].mean())
                    summary[f"yang_note_macro_{metric}"] = float(good[f"yang_note_{metric}"].mean())
                summary["yang_recordings"] = len(good)
                summary["yang_parameter_recordings"] = int((good["_yang_parameter_truth"] > 0).sum())
            summary["yang_matched_notes"] = yang_matched_n
            summary["yang_parameter_notes"] = yang_truth_n
            for name in ("extent", "rate"):
                values = [float(good[f"_soft_{name}_{k}"].sum()) for k in ("credit", "predictions", "references", "frames")]
                metrics = YangMetrics.soft_prf(*values[:3]) if values[3] else (np.nan,)*3
                for metric, value in zip(("precision", "recall", "f1"), metrics):
                    summary[f"{name}_soft_{metric}"] = value
            for metric in ("precision", "recall", "f1"):
                summary[f"aggregate_soft_{metric}"] = VibratoBenchmarker._mean_if_all_finite(
                    summary[f"extent_soft_{metric}"], summary[f"rate_soft_{metric}"])
            extent_within_tolerance = (
                VibratoBenchmarker._safe_div(
                    float(good["_active_amplitude_within"].sum()),
                    active_amplitude_n,
                )
                if active_amplitude_n
                else np.nan
            )
            rate_within_tolerance_accuracy = (
                VibratoBenchmarker._safe_div(float(good["_rate_within"].sum()), rate_n)
                if rate_n
                else np.nan
            )
            relative_extent_accuracy = (
                VibratoBenchmarker._safe_div(
                    float(good["_relative_extent_sum"].sum()), rate_n
                )
                if rate_n
                else np.nan
            )
            relative_rate_accuracy = (
                VibratoBenchmarker._safe_div(
                    float(good["_relative_rate_sum"].sum()), rate_n
                )
                if rate_n
                else np.nan
            )
            center_accuracy = (
                VibratoBenchmarker._safe_div(
                    float(good["_center_within"].sum()), center_n
                )
                if center_n
                else np.nan
            )

            summary.update(
                {
                    "frame_accuracy": VibratoBenchmarker._safe_div(
                        frame_tp + frame_tn,
                        frame_tp + frame_fp + frame_fn + frame_tn,
                    ),
                    "frame_precision": VibratoBenchmarker._safe_div(
                        frame_tp, frame_tp + frame_fp
                    ),
                    "frame_recall": (
                        VibratoBenchmarker._safe_div(frame_tp, frame_tp + frame_fn)
                        if frame_tp + frame_fn
                        else np.nan
                    ),
                    "frame_false_alarm": (
                        VibratoBenchmarker._safe_div(frame_fp, frame_fp + frame_tn)
                        if frame_fp + frame_tn
                        else np.nan
                    ),
                    "frame_f1": VibratoBenchmarker._f1(frame_tp, frame_fp, frame_fn),
                    "aggregate_precision": VibratoBenchmarker._mean_if_all_finite(
                        extent_precision,
                        rate_precision,
                    ),
                    "aggregate_recall": VibratoBenchmarker._mean_if_all_finite(
                        extent_recall,
                        rate_recall,
                    ),
                    "aggregate_f1": VibratoBenchmarker._mean_if_all_finite(
                        extent_f1, rate_f1
                    ),
                    "extent_precision": extent_precision,
                    "extent_recall": extent_recall,
                    "extent_f1": extent_f1,
                    "rate_precision": rate_precision,
                    "rate_recall": rate_recall,
                    "rate_f1": rate_f1,
                    "overall_curve_accuracy": VibratoBenchmarker._mean_if_all_finite(
                        relative_extent_accuracy,
                        relative_rate_accuracy,
                    ),
                    "relative_extent_accuracy": relative_extent_accuracy,
                    "relative_rate_accuracy": relative_rate_accuracy,
                    "curve_within_tolerance": (
                        VibratoBenchmarker._safe_div(
                            float(good["_curve_within"].sum()), curve_n
                        )
                        if curve_n
                        else np.nan
                    ),
                    "active_curve_within_tolerance": (
                        VibratoBenchmarker._safe_div(
                            float(good["_active_curve_within"].sum()), active_curve_n
                        )
                        if active_curve_n
                        else np.nan
                    ),
                    "estimate_coverage": (
                        VibratoBenchmarker._safe_div(
                            float(good["_reported_n"].sum()), rate_n
                        )
                        if rate_n
                        else np.nan
                    ),
                    "yang_overall_accuracy": VibratoBenchmarker._mean_if_all_finite(
                        yang_extent_accuracy,
                        yang_rate_accuracy,
                    ),
                    "yang_extent_accuracy": yang_extent_accuracy,
                    "yang_rate_accuracy": yang_rate_accuracy,
                    "yang_parameter_coverage": (
                        VibratoBenchmarker._safe_div(yang_matched_n, yang_truth_n)
                        if yang_truth_n
                        else np.nan
                    ),
                    "rate_mae_hz": (
                        VibratoBenchmarker._safe_div(
                            float(good["_rate_abs_sum"].sum()), rate_n
                        )
                        if rate_n
                        else np.nan
                    ),
                    "rate_rmse_hz": (
                        np.sqrt(
                            VibratoBenchmarker._safe_div(
                                float(good["_rate_sq_sum"].sum()), rate_n
                            )
                        )
                        if rate_n
                        else np.nan
                    ),
                    "rate_within_tolerance": rate_within_tolerance_accuracy,
                    "amplitude_mae_semitones": (
                        VibratoBenchmarker._safe_div(
                            float(good["_amplitude_abs_sum"].sum()), amplitude_n
                        )
                        if amplitude_n
                        else np.nan
                    ),
                    "amplitude_rmse_semitones": (
                        np.sqrt(
                            VibratoBenchmarker._safe_div(
                                float(good["_amplitude_sq_sum"].sum()), amplitude_n
                            )
                        )
                        if amplitude_n
                        else np.nan
                    ),
                    "amplitude_within_tolerance": (
                        VibratoBenchmarker._safe_div(
                            float(good["_amplitude_within"].sum()), amplitude_n
                        )
                        if amplitude_n
                        else np.nan
                    ),
                    "active_amplitude_mae_semitones": (
                        VibratoBenchmarker._safe_div(
                            float(good["_active_amplitude_abs_sum"].sum()),
                            active_amplitude_n,
                        )
                        if active_amplitude_n
                        else np.nan
                    ),
                    "active_amplitude_rmse_semitones": (
                        np.sqrt(
                            VibratoBenchmarker._safe_div(
                                float(good["_active_amplitude_sq_sum"].sum()),
                                active_amplitude_n,
                            )
                        )
                        if active_amplitude_n
                        else np.nan
                    ),
                    "active_amplitude_within_tolerance": extent_within_tolerance,
                    "center_within_tolerance": center_accuracy,
                    "center_coverage": (
                        VibratoBenchmarker._safe_div(center_reported_n, center_n)
                        if center_n
                        else np.nan
                    ),
                    "center_mae_cents": (
                        VibratoBenchmarker._safe_div(
                            float(good["_center_abs_sum"].sum()),
                            center_reported_n,
                        )
                        if center_reported_n
                        else np.nan
                    ),
                    "audio_seconds": float(good["audio_seconds"].sum()),
                    "scored_seconds": float(good["scored_seconds"].sum()),
                    "compute_seconds": float(good["compute_seconds"].sum()),
                    "audio_per_compute": VibratoBenchmarker._safe_div(
                        float(good["audio_seconds"].sum()),
                        float(good["compute_seconds"].sum()),
                    ),
                    "wall_compute_seconds": float(good["wall_compute_seconds"].sum()),
                    "audio_per_wall_compute": VibratoBenchmarker._safe_div(
                        float(good["audio_seconds"].sum()),
                        float(good["wall_compute_seconds"].sum()),
                    ),
                }
            )
            summaries.append(summary)
        return pd.DataFrame(summaries)

    @staticmethod
    def display_summary(summary: pd.DataFrame) -> pd.DataFrame:
        """Return the compact public comparison with F1 first per metric."""
        columns = [
            "method",
            "aggregate_soft_f1", "extent_soft_f1", "rate_soft_f1",
            "extent_soft_precision", "extent_soft_recall", "rate_soft_precision", "rate_soft_recall",
            "yang_matched_notes", "yang_parameter_notes", "yang_parameter_coverage",
            "yang_note_precision", "yang_note_recall", "yang_note_f1",
            "yang_frame_macro_precision", "yang_frame_macro_recall", "yang_frame_macro_f1",
            "yang_note_macro_precision", "yang_note_macro_recall", "yang_note_macro_f1",
            "yang_recordings", "yang_parameter_recordings",
            "yang_reference_notes", "yang_predicted_notes",
            "yang_only_bad_onset_rate", "yang_only_bad_offset_rate", "yang_split_rate",
            "yang_merge_rate", "yang_spurious_rate", "yang_non_detected_rate",
            "aggregate_f1",
            "aggregate_precision",
            "aggregate_recall",
            "extent_f1",
            "extent_precision",
            "extent_recall",
            "yang_extent_accuracy",
            "rate_f1",
            "rate_precision",
            "rate_recall",
            "yang_rate_accuracy",
            "center_within_tolerance",
            "frame_f1",
            "frame_precision",
            "frame_recall",
            "frame_accuracy",
            "frame_false_alarm",
            "skipped",
            "errors",
            "audio_per_compute",
        ]
        columns = [column for column in columns if column in summary]
        displayed = summary[columns].copy()
        displayed = displayed.rename(
            columns={
                "method": "Method",
                "aggregate_soft_f1": "Overall Soft F1",
                "extent_soft_f1": "Extent Soft F1", "rate_soft_f1": "Rate Soft F1",
                "extent_soft_precision": "Extent Soft Precision", "extent_soft_recall": "Extent Soft Recall",
                "rate_soft_precision": "Rate Soft Precision", "rate_soft_recall": "Rate Soft Recall",
                "yang_matched_notes": "Matched Parameter Notes (Yang)",
                "yang_parameter_notes": "Annotated Parameter Notes (Yang)",
                "yang_parameter_coverage": "Parameter Coverage (Yang)",
                "yang_recordings": "Evaluated Recordings (Yang)",
                "yang_parameter_recordings": "Parameter Recordings (Yang)",
                "yang_frame_macro_precision": "Frame Precision (recording mean)",
                "yang_frame_macro_recall": "Frame Recall (recording mean)",
                "yang_frame_macro_f1": "Frame F1 (recording mean)",
                "yang_note_macro_precision": "Note Precision (recording mean)",
                "yang_note_macro_recall": "Note Recall (recording mean)",
                "yang_note_macro_f1": "Note F1 (recording mean)",
                "yang_note_precision": "Note Precision (Yang)", "yang_note_recall": "Note Recall (Yang)",
                "yang_note_f1": "Note F1 (Yang)", "yang_reference_notes": "Reference Vibratos (Yang)",
                "yang_predicted_notes": "Detected Vibratos (Yang)",
                "yang_only_bad_onset_rate": "Only Bad Onset Rate (overlap)",
                "yang_only_bad_offset_rate": "Only Bad Offset Rate (overlap)",
                "yang_split_rate": "Split Rate (overlap)", "yang_merge_rate": "Merge Rate (overlap)",
                "yang_spurious_rate": "Spurious Rate (overlap)", "yang_non_detected_rate": "Non-Detected Rate (overlap)",
                "aggregate_precision": "Overall Precision",
                "aggregate_recall": "Overall Recall",
                "aggregate_f1": "Overall F1 (hard)",
                "extent_precision": "Extent Precision",
                "extent_recall": "Extent Recall",
                "extent_f1": "Extent F1 (hard)",
                "yang_extent_accuracy": "Extent Accuracy (Yang)",
                "rate_precision": "Rate Precision",
                "rate_recall": "Rate Recall",
                "rate_f1": "Rate F1 (hard)",
                "yang_rate_accuracy": "Rate Accuracy (Yang)",
                "center_within_tolerance": "Center Accuracy (Attune)",
                "frame_accuracy": "Detection Accuracy",
                "frame_precision": "Detection Precision",
                "frame_recall": "Detection Recall",
                "frame_f1": "Detection F1",
                "frame_false_alarm": "False Alarms",
                "skipped": "Skipped",
                "errors": "Errors",
                "audio_per_compute": "Audio(s)/Compute(s)",
            }
        )
        numeric = displayed.select_dtypes(include=[np.number]).columns
        displayed[numeric] = displayed[numeric].round(4)
        return displayed

    @staticmethod
    def no_vibrato_diagnostics(
        cases: pd.DataFrame,
        frames: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Summarize every false positive on explicitly non-vibrato cases.

        The reports retain case, instrument, temporal-location, pitch-tracking,
        transition, rate, and extent evidence. The diagnostic labels are
        descriptive flags, not additional benchmark gates.
        """
        required_case_columns = {"case_id", "has_vibrato"}
        required_frame_columns = {
            "case_id",
            "time",
            "truth_vibrato",
            "method_detected",
        }
        missing = required_case_columns - set(cases) | required_frame_columns - set(
            frames
        )
        if missing:
            raise ValueError(
                "no-vibrato diagnostics require columns: " + ", ".join(sorted(missing))
            )

        negative_cases = cases.loc[
            ~VibratoBenchmarker._as_bool(cases["has_vibrato"])
        ].copy()
        metadata_columns = [
            column for column in negative_cases.columns if column.startswith("meta_")
        ]
        identity_columns = [
            column
            for column in ("case_id", "scenario", "family", "split")
            if column in negative_cases
        ]
        case_info = negative_cases[
            [*identity_columns, *metadata_columns]
        ].drop_duplicates("case_id")
        frame_columns_to_drop = [
            column
            for column in ("scenario", "family", "split")
            if column in frames and column in case_info
        ]
        negative_frames = frames.drop(
            columns=frame_columns_to_drop,
            errors="ignore",
        ).merge(case_info, on="case_id", how="inner", validate="many_to_one")
        negative_frames["truth_vibrato"] = VibratoBenchmarker._as_bool(
            negative_frames["truth_vibrato"]
        )
        negative_frames["method_detected"] = VibratoBenchmarker._as_bool(
            negative_frames["method_detected"]
        )
        group_sizes = (
            negative_frames.groupby("case_id")["case_id"]
            .transform("size")
            .clip(lower=1)
        )
        negative_frames["note_position"] = (
            negative_frames.groupby("case_id").cumcount() + 0.5
        ) / group_sizes
        false_positive_frames = negative_frames.loc[
            ~negative_frames["truth_vibrato"] & negative_frames["method_detected"]
        ].copy()

        for stage in ("raw", "smoothed"):
            pitch_column = f"{stage}_pitch_midi"
            if pitch_column in false_positive_frames:
                false_positive_frames[f"{stage}_pitch_error_cents"] = (
                    100.0
                    * (
                        pd.to_numeric(
                            false_positive_frames[pitch_column],
                            errors="coerce",
                        )
                        - pd.to_numeric(
                            false_positive_frames.get("commanded_pitch_midi"),
                            errors="coerce",
                        )
                    ).abs()
                )

        case_rows: list[dict[str, Any]] = []
        false_positive_frame_groups = {
            case_id: group.sort_values("time").copy()
            for case_id, group in false_positive_frames.groupby("case_id")
        }
        for _, case in negative_cases.iterrows():
            case_id = case["case_id"]
            evaluated = negative_frames.loc[
                negative_frames["case_id"] == case_id
            ].sort_values("time")
            false_positives = false_positive_frame_groups.get(
                case_id,
                evaluated.iloc[0:0].copy(),
            )
            frame_count = len(evaluated)
            false_positive_count = len(false_positives)
            frame_step = (
                float(np.median(np.diff(evaluated["time"]))) if frame_count > 1 else 0.0
            )
            detected_mask = VibratoBenchmarker._as_bool(
                evaluated["method_detected"]
            ).to_numpy()
            runs = VibratoBenchmarker._true_runs(detected_mask)
            run_seconds = [len(run) * frame_step for run in runs]

            if frame_count:
                position = (np.arange(frame_count, dtype=float) + 0.5) / frame_count
                evaluated = evaluated.assign(note_position=position)
                false_positions = position[detected_mask]
            else:
                false_positions = np.empty(0, dtype=float)
            onset_count = int(np.sum(false_positions < 0.2))
            release_count = int(np.sum(false_positions >= 0.8))
            edge_fraction = (
                (onset_count + release_count) / false_positive_count
                if false_positive_count
                else np.nan
            )

            smooth_error = pd.to_numeric(
                false_positives.get(
                    "smoothed_pitch_error_cents",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            )
            transition_fraction = (
                VibratoBenchmarker._finite_fraction(
                    VibratoBenchmarker._as_bool(
                        false_positives["smoothed_pitch_transition"]
                    )
                )
                if "smoothed_pitch_transition" in false_positives
                else np.nan
            )
            octave_fraction = VibratoBenchmarker._finite_fraction(smooth_error >= 600.0)
            median_smooth_error = VibratoBenchmarker._finite_median(smooth_error)

            flags: list[str] = []
            if false_positive_count:
                if np.isfinite(octave_fraction) and octave_fraction >= 0.25:
                    flags.append("octave tracking error")
                elif np.isfinite(median_smooth_error) and median_smooth_error >= 50.0:
                    flags.append("pitch tracking error")
                if np.isfinite(transition_fraction) and transition_fraction >= 0.25:
                    flags.append("HMM transition overlap")
                if np.isfinite(edge_fraction) and edge_fraction >= 0.75:
                    flags.append("note-edge localized")
                if not flags:
                    flags.append("stable-contour periodicity")

            row = {
                "case_id": case_id,
                "scenario": case.get("scenario", ""),
                "evaluated_frames": frame_count,
                "false_positive_frames": false_positive_count,
                "false_positive_rate": (
                    false_positive_count / frame_count if frame_count else np.nan
                ),
                "false_positive_runs": len(runs),
                "false_positive_seconds": false_positive_count * frame_step,
                "longest_false_positive_run_seconds": (max(run_seconds, default=0.0)),
                "onset_false_positive_frames": onset_count,
                "middle_false_positive_frames": (
                    false_positive_count - onset_count - release_count
                ),
                "release_false_positive_frames": release_count,
                "edge_false_positive_fraction": edge_fraction,
                "median_raw_pitch_error_cents": VibratoBenchmarker._finite_median(
                    false_positives.get(
                        "raw_pitch_error_cents",
                        pd.Series(dtype=float),
                    )
                ),
                "median_smoothed_pitch_error_cents": median_smooth_error,
                "smoothed_octave_error_fraction": octave_fraction,
                "transition_overlap_fraction": transition_fraction,
                "median_estimated_rate_hz": VibratoBenchmarker._finite_median(
                    false_positives.get(
                        "estimated_rate_hz",
                        pd.Series(dtype=float),
                    )
                ),
                "median_estimated_amplitude_semitones": VibratoBenchmarker._finite_median(
                    false_positives.get(
                        "estimated_amplitude_semitones",
                        pd.Series(dtype=float),
                    )
                ),
                "diagnostic_pattern": "; ".join(flags) if flags else "none",
            }
            row.update({column: case.get(column) for column in metadata_columns})
            case_rows.append(row)

        case_table = pd.DataFrame(case_rows)
        case_false_positive_frames = case_table.get(
            "false_positive_frames",
            pd.Series(dtype=float),
        )
        case_false_positive_rates = case_table.get(
            "false_positive_rate",
            pd.Series(dtype=float),
        )
        case_evaluated_frames = case_table.get(
            "evaluated_frames",
            pd.Series(dtype=float),
        )
        overview = pd.DataFrame(
            [
                {
                    "negative_cases": len(case_table),
                    "cases_with_false_positives": int(
                        (case_false_positive_frames > 0).sum()
                    ),
                    "whole_note_false_positive_cases": int(
                        (case_false_positive_rates >= 0.95).sum()
                    ),
                    "evaluated_frames": int(case_evaluated_frames.sum()),
                    "false_positive_frames": int(case_false_positive_frames.sum()),
                    "false_positive_rate": VibratoBenchmarker._safe_div(
                        float(case_false_positive_frames.sum()),
                        float(case_evaluated_frames.sum()),
                    ),
                }
            ]
        )

        group_columns = [
            column
            for column in (
                "meta_ensemble",
                "meta_instrument",
                "meta_yin_integration_size",
            )
            if column in case_table
        ]
        if group_columns and len(case_table):
            by_instrument = (
                case_table.groupby(
                    group_columns,
                    dropna=False,
                    sort=True,
                )
                .agg(
                    negative_cases=("case_id", "size"),
                    cases_with_false_positives=(
                        "false_positive_frames",
                        lambda values: int((values > 0).sum()),
                    ),
                    evaluated_frames=("evaluated_frames", "sum"),
                    false_positive_frames=("false_positive_frames", "sum"),
                    median_case_false_positive_rate=("false_positive_rate", "median"),
                    worst_case_false_positive_rate=("false_positive_rate", "max"),
                )
                .reset_index()
            )
            by_instrument["false_positive_rate"] = (
                by_instrument["false_positive_frames"]
                / by_instrument["evaluated_frames"]
            )
        else:
            by_instrument = pd.DataFrame()

        failures = case_table.loc[
            case_table.get(
                "false_positive_frames",
                pd.Series(dtype=float),
            )
            > 0
        ]
        if len(failures):
            by_pattern = (
                failures.groupby(
                    "diagnostic_pattern",
                    sort=True,
                )
                .agg(
                    cases=("case_id", "size"),
                    false_positive_frames=("false_positive_frames", "sum"),
                    false_positive_seconds=("false_positive_seconds", "sum"),
                    median_case_false_positive_rate=("false_positive_rate", "median"),
                    median_pitch_error_cents=(
                        "median_smoothed_pitch_error_cents",
                        "median",
                    ),
                    median_estimated_rate_hz=("median_estimated_rate_hz", "median"),
                    median_estimated_amplitude_semitones=(
                        "median_estimated_amplitude_semitones",
                        "median",
                    ),
                )
                .reset_index()
            )
        else:
            by_pattern = pd.DataFrame(
                columns=[
                    "diagnostic_pattern",
                    "cases",
                    "false_positive_frames",
                    "false_positive_seconds",
                    "median_case_false_positive_rate",
                    "median_pitch_error_cents",
                    "median_estimated_rate_hz",
                    "median_estimated_amplitude_semitones",
                ]
            )

        if len(false_positive_frames):
            false_positive_frames = false_positive_frames.sort_values(
                ["case_id", "time"]
            ).reset_index(drop=True)

        return {
            "overview": overview,
            "cases": case_table,
            "by_instrument": by_instrument,
            "by_pattern": by_pattern,
            "false_positive_frames": false_positive_frames,
        }

    def write_reports(
        self,
        raw: pd.DataFrame,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write canonical summaries and per-method diagnostics."""
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        raw_root = destination / "raw_outputs"
        if raw_root.is_symlink():
            raise ValueError(
                f"refusing to replace symlinked raw-output directory: {raw_root}"
            )
        if raw_root.exists():
            # ``raw_outputs`` is generated state for one canonical run. Clear it
            # so a method-filtered rerun cannot silently retain stale methods.
            shutil.rmtree(raw_root)
        raw_root.mkdir(parents=True, exist_ok=True)
        detailed_summary = self.summarize(raw)
        detailed_scenario = self.summarize(raw, group_by=("method", "scenario"))
        summary = self.display_summary(detailed_summary)
        paths = {
            "summary": destination / "summary.csv",
            "note_macro": destination / "note_macro.csv",
            "raw_outputs": raw_root,
        }
        public_raw = raw.drop(
            columns=[column for column in raw.columns if column.startswith("_")],
            errors="ignore",
        )
        curve_frames: list[pd.DataFrame] = []
        required = {
            "_curve_times",
            "_curve_evaluation_mask",
            "_curve_truth_rate_hz",
            "_curve_estimated_rate_hz",
            "_curve_truth_amplitude_semitones",
            "_curve_estimated_amplitude_semitones",
            "_curve_truth_vibrato",
            "_curve_detected",
            "_curve_extent_predicted",
            "_curve_extent_matched",
            "_curve_rate_predicted",
            "_curve_rate_matched",
            "_curve_frame_within_tolerance",
            "_curve_truth_center_midi",
            "_curve_commanded_pitch_midi",
            "_curve_raw_pitch_midi",
            "_curve_smoothed_pitch_midi",
            "_curve_pitch_transition",
            "_curve_fit_confidence",
            "_curve_estimated_center_midi",
        }
        if required.issubset(raw.columns):
            for _, values in raw.iterrows():
                if values.get("error", ""):
                    continue
                evaluation = np.asarray(
                    values["_curve_evaluation_mask"],
                    dtype=bool,
                )
                curve_frames.append(
                    pd.DataFrame(
                        {
                            "method": values["method"],
                            "case_id": values["case_id"],
                            "scenario": values["scenario"],
                            "time": values["_curve_times"][evaluation],
                            "truth_rate_hz": values["_curve_truth_rate_hz"][evaluation],
                            "estimated_rate_hz": values["_curve_estimated_rate_hz"][
                                evaluation
                            ],
                            "truth_amplitude_semitones": values[
                                "_curve_truth_amplitude_semitones"
                            ][evaluation],
                            "estimated_amplitude_semitones": values[
                                "_curve_estimated_amplitude_semitones"
                            ][evaluation],
                            "truth_width_cents": 200.0
                            * values["_curve_truth_amplitude_semitones"][evaluation],
                            "estimated_width_cents": 200.0
                            * values["_curve_estimated_amplitude_semitones"][
                                evaluation
                            ],
                            "truth_center_midi": values["_curve_truth_center_midi"][
                                evaluation
                            ],
                            "commanded_pitch_midi": values[
                                "_curve_commanded_pitch_midi"
                            ][evaluation],
                            "raw_pitch_midi": values["_curve_raw_pitch_midi"][
                                evaluation
                            ],
                            "smoothed_pitch_midi": values["_curve_smoothed_pitch_midi"][
                                evaluation
                            ],
                            "smoothed_pitch_transition": values[
                                "_curve_pitch_transition"
                            ][evaluation],
                            "fit_confidence": values["_curve_fit_confidence"][
                                evaluation
                            ],
                            "estimated_center_midi": values[
                                "_curve_estimated_center_midi"
                            ][evaluation],
                            "truth_vibrato": values["_curve_truth_vibrato"][evaluation],
                            "method_detected": values["_curve_detected"][evaluation],
                            "extent_predicted": values["_curve_extent_predicted"][
                                evaluation
                            ],
                            "extent_matched": values["_curve_extent_matched"][
                                evaluation
                            ],
                            "rate_predicted": values["_curve_rate_predicted"][
                                evaluation
                            ],
                            "rate_matched": values["_curve_rate_matched"][evaluation],
                            "curve_within_tolerance": values[
                                "_curve_frame_within_tolerance"
                            ][evaluation],
                        }
                    )
                )
        curve_columns = [
            "method",
            "case_id",
            "scenario",
            "time",
            "truth_rate_hz",
            "estimated_rate_hz",
            "truth_amplitude_semitones",
            "estimated_amplitude_semitones",
            "truth_width_cents",
            "estimated_width_cents",
            "truth_center_midi",
            "commanded_pitch_midi",
            "raw_pitch_midi",
            "smoothed_pitch_midi",
            "smoothed_pitch_transition",
            "fit_confidence",
            "estimated_center_midi",
            "truth_vibrato",
            "method_detected",
            "extent_predicted",
            "extent_matched",
            "rate_predicted",
            "rate_matched",
            "curve_within_tolerance",
        ]
        curve_table = (
            pd.concat(curve_frames, ignore_index=True)
            if curve_frames
            else pd.DataFrame(columns=curve_columns)
        )
        summary.to_csv(paths["summary"], index=False)
        note_macro_metrics = [
            "aggregate_soft_f1", "extent_soft_f1", "rate_soft_f1",
            "extent_soft_precision", "extent_soft_recall", "rate_soft_precision", "rate_soft_recall",
            "yang_note_precision", "yang_note_recall", "yang_note_f1",
            "aggregate_f1",
            "aggregate_precision",
            "aggregate_recall",
            "extent_f1",
            "extent_precision",
            "extent_recall",
            "rate_f1",
            "rate_precision",
            "rate_recall",
            "center_within_tolerance",
            "center_mae_cents",
            "frame_f1",
            "frame_precision",
            "frame_recall",
            "frame_accuracy",
            "rate_mae_hz",
            "active_amplitude_mae_semitones",
            "active_amplitude_within_tolerance",
            "audio_per_compute",
        ]
        good_cases = public_raw.loc[
            ~public_raw["skipped"].fillna(False).astype(bool)
            & public_raw["error"].fillna("").eq("")
        ]

        def note_macro(group_columns: list[str]) -> pd.DataFrame:
            columns = [
                column for column in note_macro_metrics if column in good_cases.columns
            ]
            grouped = good_cases.groupby(
                group_columns,
                sort=False,
                dropna=False,
            )
            result = grouped[columns].mean(numeric_only=True).reset_index()
            counts = grouped.size().rename("cases").reset_index()
            return counts.merge(result, on=group_columns, how="left")

        note_macro_by_method = note_macro(["method"])
        note_macro_by_method.to_csv(paths["note_macro"], index=False)
        note_macro_by_scenario = note_macro(["method", "scenario"])

        methods = [str(method) for method in detailed_summary["method"]]
        for method in methods:
            method_dir = raw_root / method
            method_dir.mkdir(parents=True, exist_ok=True)

            method_cases = public_raw.loc[public_raw["method"] == method]
            method_cases.to_csv(method_dir / "cases.csv", index=False)

            method_frames = curve_table.loc[curve_table["method"] == method]
            method_frames.to_csv(method_dir / "frames.csv", index=False)

            no_vibrato = self.no_vibrato_diagnostics(
                method_cases,
                method_frames,
            )
            no_vibrato["overview"].to_csv(
                method_dir / "no_vibrato_overview.csv", index=False
            )
            no_vibrato["cases"].to_csv(method_dir / "no_vibrato_cases.csv", index=False)
            no_vibrato["by_instrument"].to_csv(
                method_dir / "no_vibrato_by_instrument.csv", index=False
            )
            no_vibrato["by_pattern"].to_csv(
                method_dir / "no_vibrato_by_pattern.csv", index=False
            )
            no_vibrato["false_positive_frames"].to_csv(
                method_dir / "no_vibrato_false_positive_frames.csv", index=False
            )

            method_scenario = detailed_scenario.loc[
                detailed_scenario["method"] == method
            ].reset_index(drop=True)
            scenario_metrics = self.display_summary(method_scenario)
            scenario = pd.concat(
                [
                    method_scenario[["scenario"]],
                    scenario_metrics.drop(columns=["Method"]),
                ],
                axis=1,
            )
            scenario.to_csv(method_dir / "by_scenario.csv", index=False)
            note_macro_by_scenario.loc[note_macro_by_scenario["method"] == method].drop(
                columns=["method"]
            ).to_csv(
                method_dir / "by_scenario_note_macro.csv",
                index=False,
            )
        return paths

    @classmethod
    def parse_args(cls, argv: Sequence[str] | None = None) -> argparse.Namespace:
        from algorithms.Config import Config
        from benchmarks.modules.vibrato.datasets.CocoDataset import (
            DEFAULT_PROFILES,
        )
        from benchmarks.modules.vibrato.datasets.CocoRenderer import (
            DEFAULT_SOUNDFONTS_ROOT,
            CocoRenderer,
        )

        root = Path(__file__).resolve().parents[2]
        parser = argparse.ArgumentParser(
            description="Run Attune's parallel vibrato detector benchmark."
        )
        parser.add_argument("--methods", help="comma-separated detector names")
        parser.add_argument("--list-methods", action="store_true")
        sources = parser.add_mutually_exclusive_group()
        sources.add_argument("--dataset-csv", type=Path)
        sources.add_argument("--coco", action="store_true")
        sources.add_argument("--yang", action="store_true")
        sources.add_argument("--driedger", action="store_true")
        sources.add_argument(
            "--coco-control",
            action="store_true",
            help="validate a rendered straight note for false vibrato",
        )

        parser.add_argument(
            "--yang-root",
            type=Path,
            default=root / "benchmarks" / "datasets" / "vibrato",
        )
        parser.add_argument(
            "--yang-cache-root",
            type=Path,
            default=(
                root / "benchmarks" / "datasets" / "vibrato" / "pitch_data" / "attune"
            ),
        )
        parser.add_argument("--yang-recording-limit", type=int, default=2)
        parser.add_argument("--yang-all-recordings", action="store_true")
        parser.add_argument("--yang-full-audio", action="store_true")
        parser.add_argument("--yang-parameters-only", action="store_true")
        parser.add_argument("--yang-no-smooth", action="store_true")
        parser.add_argument("--yang-force", action="store_true")
        parser.add_argument(
            "--driedger-root",
            type=Path,
            default=root / "benchmarks" / "datasets" / "vibrato_driedger",
        )

        parser.add_argument("--coco-root", type=Path)
        parser.add_argument(
            "--coco-output-root",
            type=Path,
            default=(root / "benchmarks" / "datasets" / "cocochorales_vibrato"),
        )
        parser.add_argument("--coco-sfizz-render", type=Path)
        parser.add_argument(
            "--coco-soundfonts-root",
            type=Path,
            default=DEFAULT_SOUNDFONTS_ROOT,
        )
        parser.add_argument(
            "--coco-split", choices=("test", "valid", "train"), default="test"
        )
        parser.add_argument("--coco-stem-limit", "--coco-stems", type=int, default=1)
        parser.add_argument("--coco-all-stems", action="store_true")
        parser.add_argument(
            "--coco-notes-per-stem", type=int, default=len(DEFAULT_PROFILES)
        )
        parser.add_argument("--coco-profiles", default=",".join(DEFAULT_PROFILES))
        parser.add_argument("--coco-snrs", default="clean")
        parser.add_argument("--coco-noise-file", type=Path)
        parser.add_argument("--coco-min-note-seconds", type=float, default=0.75)
        parser.add_argument("--coco-bend-sample-hz", type=float, default=100.0)
        parser.add_argument(
            "--coco-injection-range", choices=("native", "yang"), default="native"
        )
        parser.add_argument("--coco-no-smooth", action="store_true")
        yin = parser.add_mutually_exclusive_group()
        yin.add_argument(
            "--coco-auto-yin-window",
            "--coco-adaptive-yin-window",
            action="store_true",
        )
        yin.add_argument("--coco-yin-window-size", type=int)
        parser.add_argument("--coco-force", action="store_true")
        parser.add_argument("--force-pitch", action="store_true")
        parser.add_argument(
            "--coco-control-instrument",
            choices=tuple(sorted(CocoRenderer.PATCHES)),
            default="violin",
        )
        parser.add_argument("--coco-control-pitch", type=int)
        parser.add_argument("--coco-control-duration", type=float, default=3.0)
        parser.add_argument("--coco-control-velocity", type=int, default=96)

        parser.add_argument("--replicates", type=int, default=3)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--frame-rate", type=float, default=100.0)
        parser.add_argument("--noise-cents", type=float, default=3.0)
        parser.add_argument("--dropout-probability", type=float, default=0.01)
        parser.add_argument("--outlier-probability", type=float, default=0.004)
        parser.add_argument("--outlier-scale-cents", type=float, default=35.0)
        parser.add_argument("--context-seconds", type=float, default=0.5)
        parser.add_argument("--rate-tolerance-hz", type=float, default=0.5)
        parser.add_argument("--amplitude-tolerance-semitones", type=float, default=0.05)
        parser.add_argument("--center-tolerance-cents", type=float, default=25.0)

        parser.add_argument("--attune-onset-taper-seconds", type=float, default=0.0)
        parser.add_argument(
            "--attune-onset-taper-max-fraction", type=float, default=0.25
        )
        parser.add_argument("--attune-onset-taper-floor", type=float, default=0.25)
        parser.add_argument("--attune-phase-smoothness", type=float, default=20.0)
        parser.add_argument("--attune-width-smoothness", type=float, default=1.0)
        parser.add_argument(
            "--attune-curve-sec", type=float, default=Config.vib2_curve_sec
        )
        parser.add_argument(
            "--attune-edge-policy", choices=("hold", "fitted"), default="hold"
        )

        parser.add_argument(
            "--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1)
        )
        parser.add_argument("--quick", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--strict", action="store_true")
        parser.add_argument("--no-console-summary", action="store_true")
        parser.add_argument("--export-corpus", type=Path)
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=root / "benchmarks" / "results" / "vibrato",
        )
        return parser.parse_args(argv)

    @classmethod
    def _selected_detectors(
        cls,
        args: argparse.Namespace,
    ) -> list[VibratoDetectorBase]:
        available = cls.available_detectors()
        selected = cls.default_detectors()
        if args.list_methods:
            defaults = {detector.name for detector in selected}
            for detector in available:
                suffix = " [default]" if detector.name in defaults else " [ablation]"
                print(f"{detector.name}{suffix}: {detector.description}")
            return []
        if args.methods:
            names = [name.strip() for name in args.methods.split(",") if name.strip()]
            by_name = {detector.name: detector for detector in available}
            unknown = sorted(set(names) - set(by_name))
            if unknown:
                raise SystemExit(f"unknown methods: {', '.join(unknown)}")
            selected = [by_name[name] for name in names]
        elif args.driedger:
            selected = [
                detector for detector in available if detector.name == "driedger"
            ]
        for detector in selected:
            if detector.name == "attune":
                detector.set_config_overrides(
                    vib2_onset_taper_seconds=args.attune_onset_taper_seconds,
                    vib2_onset_taper_max_fraction=args.attune_onset_taper_max_fraction,
                    vib2_onset_taper_floor=args.attune_onset_taper_floor,
                    vib2_phase_smoothness=args.attune_phase_smoothness,
                    vib2_width_smoothness=args.attune_width_smoothness,
                    vib2_curve_sec=args.attune_curve_sec,
                    vib2_hold_edge_values=args.attune_edge_policy == "hold",
                )
        return selected

    @classmethod
    def _load_examples(
        cls,
        args: argparse.Namespace,
    ) -> tuple[list[VibratoExample], str]:
        from benchmarks.modules.vibrato.competitors.DriedgerDataset import DriedgerDataset
        from benchmarks.modules.vibrato.datasets.CocoDataset import CocoDataset
        from benchmarks.modules.vibrato.datasets.SyntheticDataset import SyntheticDataset
        from benchmarks.modules.vibrato.datasets.YangDataset import YangDataset

        if args.dataset_csv:
            return SyntheticDataset.load_csv(args.dataset_csv), str(args.dataset_csv)
        if args.coco:
            profiles = tuple(
                value.strip()
                for value in args.coco_profiles.split(",")
                if value.strip()
            )
            snrs = tuple(
                (
                    math.inf
                    if value.strip().lower() in {"clean", "inf", "none"}
                    else float(value)
                )
                for value in args.coco_snrs.split(",")
                if value.strip()
            )
            stem_limit = (
                1
                if args.quick
                else None if args.coco_all_stems else args.coco_stem_limit
            )
            return (
                CocoDataset.build(
                    coco_root=args.coco_root,
                    output_root=args.coco_output_root,
                    split=args.coco_split,
                    max_stems=stem_limit,
                    notes_per_stem=(
                        min(2, args.coco_notes_per_stem)
                        if args.quick
                        else args.coco_notes_per_stem
                    ),
                    profiles=profiles,
                    snrs_db=snrs,
                    noise_file=args.coco_noise_file,
                    min_note_seconds=args.coco_min_note_seconds,
                    bend_sample_hz=args.coco_bend_sample_hz,
                    injection_range=args.coco_injection_range,
                    seed=args.seed,
                    smooth_pitch=not args.coco_no_smooth,
                    adaptive_yin_window=args.coco_auto_yin_window,
                    fixed_yin_window_size=args.coco_yin_window_size,
                    sfizz_render=args.coco_sfizz_render,
                    soundfonts_root=args.coco_soundfonts_root,
                    force=args.coco_force,
                    force_pitch=args.force_pitch,
                    workers=args.workers,
                ),
                "cocochorales_midi_injected",
            )
        if args.yang:
            limit = (
                1
                if args.quick
                else None if args.yang_all_recordings else args.yang_recording_limit
            )
            from benchmarks.modules.vibrato.datasets.YangFullDataset import YangFullDataset, YangParameterDataset
            dataset_class = (YangParameterDataset if args.yang_parameters_only else
                             YangFullDataset if args.yang_full_audio else YangDataset)
            return (
                dataset_class.build(
                    dataset_root=args.yang_root,
                    cache_root=args.yang_cache_root,
                    max_recordings=limit,
                    smooth_pitch=not args.yang_no_smooth,
                    force=args.yang_force or args.force_pitch,
                    workers=args.workers,
                ),
                "yang_parameter_recordings_full_audio" if args.yang_parameters_only else
                "yang_full_audio" if args.yang_full_audio else "yang_parameter_subset",
            )
        if args.driedger:
            return DriedgerDataset.load(args.driedger_root), "driedger_validation"
        return (
            SyntheticDataset.build(
                replicates=1 if args.quick else args.replicates,
                seed=args.seed,
                frame_rate=args.frame_rate,
                noise_cents=args.noise_cents,
                dropout_probability=args.dropout_probability,
                outlier_probability=args.outlier_probability,
                outlier_scale_cents=args.outlier_scale_cents,
                context_seconds=args.context_seconds,
            ),
            "synthetic_pitch_contours",
        )

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        from benchmarks.modules.vibrato.datasets.CocoDataset import (
            PROFILE_PARAMETER_SAMPLER_VERSION,
        )

        args = cls.parse_args(argv)
        if args.workers < 1:
            raise SystemExit("--workers must be positive")
        if args.coco_control:
            from benchmarks.modules.vibrato.datasets.CocoControl import CocoControl

            report = CocoControl.run(
                args.output_dir,
                instrument=args.coco_control_instrument,
                pitch=args.coco_control_pitch,
                duration=args.coco_control_duration,
                velocity=args.coco_control_velocity,
                sfizz_render=args.coco_sfizz_render,
                soundfonts_root=args.coco_soundfonts_root,
            )
            print(json.dumps(report, indent=2))
            print(f"\nReport: {args.output_dir / 'zero_vibrato_validation.json'}")
            return 0 if report["passed"] else 1
        detectors = cls._selected_detectors(args)
        if args.list_methods:
            return 0
        examples, source = cls._load_examples(args)
        if args.export_corpus:
            from benchmarks.modules.vibrato.datasets.SyntheticDataset import SyntheticDataset

            SyntheticDataset.write_csv(examples, args.export_corpus)

        benchmarker = cls(
            rate_accuracy_tolerance_hz=args.rate_tolerance_hz,
            amplitude_accuracy_tolerance_semitones=(args.amplitude_tolerance_semitones),
            center_accuracy_tolerance_cents=args.center_tolerance_cents,
        )
        progress_width = 0

        def progress(
            method: str,
            index: int,
            total: int,
            example: VibratoExample,
        ) -> None:
            nonlocal progress_width
            message = f"[{index:>4}/{total}] {method}: {example.case_id}"
            progress_width = max(progress_width, len(message))
            print(f"\r{message:<{progress_width}}", end="", flush=True)

        raw = benchmarker.run(
            examples,
            detectors,
            progress=None if args.quiet else progress,
            strict=args.strict,
            workers=args.workers,
            cache_dir=args.output_dir / "checkpoints",
        )
        if progress_width:
            print()
        paths = benchmarker.write_reports(raw, args.output_dir)
        groups = cls._analysis_groups(examples)
        coco_synthesis = {
            str(example.metadata["instrument"]): example.metadata["synthesis_renderer"]
            for example in examples
            if example.metadata.get("instrument")
            and example.metadata.get("synthesis_renderer")
        }
        if args.coco:
            from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
            from benchmarks.modules.vibrato.datasets.CocoDataset import (
                PROFILE_PARAMETER_SAMPLER_VERSION,
            )
        manifest = {
            "dataset": {
                "source": source,
                "examples": len(examples),
                "analysis_groups": len(groups),
                "audio_seconds": float(sum(group[0].duration for group in groups)),
                "scored_seconds": float(sum(item.scored_duration for item in examples)),
                "seed": args.seed,
                "coco_track_selection_policy": (
                    CocoChorales.BALANCED_SELECTION_POLICY if args.coco else None
                ),
                "coco_profile_parameter_sampler": (
                    PROFILE_PARAMETER_SAMPLER_VERSION if args.coco else None
                ),
                "coco_synthesis": coco_synthesis or None,
                "workers": args.workers,
            },
            "metrics": {
                "version": "soft_yang_v1",
                "soft_credit": "max(0, 1-abs(estimate-reference)/reference)",
                "soft_aggregation": "pooled evaluated frames; 2*credit/(predictions+references)",
                "yang_note_min_seconds": 0.28,
                "yang_onset_tolerance_seconds": 0.1,
                "yang_offset_tolerance": "max(0.1 seconds, 0.2*reference duration)",
                "yang_error_diagnostics": "interval overlap topology adaptation, not Molina implementation",
                "rate_tolerance_hz": args.rate_tolerance_hz,
                "amplitude_tolerance_semitones": (args.amplitude_tolerance_semitones),
                "center_tolerance_cents": args.center_tolerance_cents,
            },
            "methods": [
                {"name": detector.name, "description": detector.description}
                for detector in detectors
            ],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run_config.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        summary = benchmarker.display_summary(benchmarker.summarize(raw))
        if not args.no_console_summary:
            print("\n" + summary.to_string(index=False))
        print(f"\nReports: {paths['summary'].parent}")
        return int(bool(raw["error"].fillna("").astype(bool).any()))


if __name__ == "__main__":
    raise SystemExit(VibratoBenchmarker.main())
