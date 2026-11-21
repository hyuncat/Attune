"""Six-method ruptures comparison followed by tuning of the winning method.

Kept beside its notebook so adding this experiment does not invalidate the
already completed pitch runner's code identity. This file has its own fingerprint.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from benchmarks.modules.sweeps import AppendixSweeps as base
from benchmarks.modules.note.competitors.Ruptures import Ruptures
from benchmarks.modules.note.sweeps.NoteDetectionParamSweep import METHOD_CONFIGS

METHOD_LABELS = {
    'pelt-l1': 'PELT (L1)', 'pelt-l2': 'PELT (L2)', 'pelt-rbf': 'PELT (RBF)',
    'kernelcpd-linear': 'KernelCPD (linear)',
    'kernelcpd-gaussian': 'KernelCPD (Gaussian)',
    'kernelcpd-cosine': 'KernelCPD (cosine)',
}
EVALUATION_STAGE = 'production_consolidation_and_repeat_v2'
BASELINE = dict(min_silence_duration_ms=40., min_note_length_factor=.5,
                min_note_length_cap=.2, pitch_thresh=.75)


class CappedRuptures(Ruptures):
    def _predict_bkps(self, signal, algorithm, model, min_size, jump, penalty, width, n_bkps):
        if algorithm != 'pelt' or model != 'rbf':
            return super()._predict_bkps(signal, algorithm, model, min_size, jump,
                                         penalty, width, n_bkps)
        import ruptures as rpt
        from benchmarks.notebooks.sweeps.scripts.fast_rbf import PrefixCostRbf
        if len(signal) < 2 * min_size:
            return [len(signal)]
        try:
            bkps = rpt.Pelt(custom_cost=PrefixCostRbf(), min_size=min_size,
                            jump=jump).fit(signal).predict(pen=penalty)
        except (rpt.exceptions.BadSegmentationParameters, ValueError):
            return [len(signal)]
        return self._sanitize_bkps(bkps, len(signal))

    def _notes_from_run_breakpoints(self, runs, bkps_by_run, merge_adjacent):
        import numpy as np
        from algorithms.NoteDetector import NoteDetector
        # Apply the exact production range-of-original-medians merge per run.
        # The legacy merge_adjacent flag uses a different rule and stays off.
        merged = [NoteDetector._merge_same_pitch_segments(
            np.asarray([p.value for p in run]).reshape(-1, 1), bkps)
            for run, bkps in zip(runs, bkps_by_run)]
        return super()._notes_from_run_breakpoints(runs, merged, False)

    def _pelt_segment_pitch(self, pitches):
        # Label notes from the very same smoothed track that was segmented,
        # matching production, rather than falling back to raw candidate ranks.
        import numpy as np
        return [float(np.median([p.value for p in pitches]))]

    def _pelt_min_size_from_score(self):
        # The legacy adapter predates the duration ceiling. Use the same
        # min(factor * shortest fitted score note, cap) for runs AND segments.
        return self.config.note_detection_min_frames()


def method_selection_items(items):
    """Use the entire frozen cohort, including all 195 CocoChorales tracks."""
    from collections import Counter
    items = list(items)
    counts = Counter(item['instrument'] for item in items
                     if item.get('dataset', 'coco') == 'coco')
    if len(counts) != 13 or set(counts.values()) != {15}:
        raise ValueError('Method selection requires all 195 Coco tracks (15 per instrument)')
    if len({item['track_id'] for item in items}) != len(items):
        raise ValueError('Duplicate tracks in the frozen method-selection cohort')
    return items


def reuse_previous_checkpoints(root, source, items, variants):
    """Migrate only the reviewed predecessor with identical inference and metrics.

    Changing the selection cohort invalidates rankings, but not per-track results.
    In particular, retain accelerated RBF results and parameter-sweep checkpoints.
    Never import the older pre-consolidation caches or their selected winners.
    """
    previous = dict(source, runner_sha256=
        '992183e5c81823432aa1e1e023d1099d4e66b50bcea9ce7a60abd4abadcf8b94')
    old_root = root.parent / base.digest(previous)[:16]
    contract = old_root / 'contract.json'
    if contract.exists() and base.read(contract) == previous:
        _copy_previous_checkpoints(old_root, root, items, variants)


def _copy_previous_checkpoints(old_root, root, items, variants):
    import os
    for item in items:
        for variant in variants:
            name = variant['variant']
            old = old_root / 'checkpoints' / item['track_id']
            new = root / 'checkpoints' / item['track_id']
            checkpoint = old / (name + '.json')
            if (new / checkpoint.name).exists() or not checkpoint.exists():
                continue
            if not (old / (name + '.refined.pkl')).exists():
                continue
            row = base.read(checkpoint)
            new.mkdir(parents=True, exist_ok=True)
            for suffix in ('.initial.pkl', '.refined.pkl', '.timing.json'):
                src, dst = old / (name + suffix), new / (name + suffix)
                if src.exists() and not dst.exists():
                    try:
                        os.link(src, dst)
                    except FileExistsError:
                        pass  # The other background stage may have migrated it too.
            row['artifact'] = str(new / (name + '.refined.pkl'))
            row['initial_artifact'] = str(new / (name + '.initial.pkl'))
            base.atomic(new / checkpoint.name, row)


def variants_for(methods=None, axes=None):
    methods = tuple(METHOD_CONFIGS) if methods is None else tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - METHOD_CONFIGS.keys():
        raise ValueError('Select distinct methods from METHOD_CONFIGS')
    parameters = [dict(BASELINE)] if axes is None else [v['params'] for v in base.grid(axes)]
    if any(set(p) != set(BASELINE) for p in parameters):
        raise ValueError('Supply all four note parameter axes')
    return [dict(method=method, params=params,
                 variant=base.digest(dict(method=method, params=params))[:16])
            for method in methods for params in parameters]


def parameter_table(variants):
    import pandas as pd
    return base.result_table(pd.DataFrame([
        dict(Method=METHOD_LABELS[v['method']], **v['params']) for v in variants
    ]))


def detect_notes(recording, method):
    if method == 'kernelcpd-linear':
        recording.detect_notes()
        return recording.note_data
    recording.transition_detector.clear_transitions(recording.pitch_data.data)
    recording.resize_score(to_span='pitch', include_transitions=False)
    recording.update_min_note_length()
    return CappedRuptures(recording).detect(
        **METHOD_CONFIGS[method], pitch_step_semitones=recording.config.pitch_thresh,
        exclude_transitions=False, merge_adjacent=False,
        features=('pitch',), standardize_features=False,
    )


def _note(task):
    """Importable spawn worker; base.parallel counts each track/combination."""
    from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation
    item, directory, variants, pitch_params, pitch_path = task
    directory = Path(directory) / item['track_id']
    reference = NoteEvaluation.reference_notes(item.get('note_reference', dict(dataset='coco', reference=item['paths']['midi'])))
    pitches = None
    rows = []
    for variant in variants:
        label = dict(item, track_id=f"{item['track_id']} | {METHOD_LABELS[variant['method']]}")
        base.progress(label, variant)
        checkpoint = directory / (variant['variant'] + '.json')
        artifact = directory / (variant['variant'] + '.refined.pkl')
        initial = directory / (variant['variant'] + '.initial.pkl')
        timing_path = directory / (variant['variant'] + '.timing.json')
        if checkpoint.exists() and artifact.exists():
            row = base.read(checkpoint)
            if (row['method'] != variant['method'] or row['track_id'] != item['track_id']
                    or row.get('evaluation_stage') != EVALUATION_STAGE
                    or any(row[k] != value for k, value in variant['params'].items())):
                raise ValueError(f'Incompatible checkpoint: {checkpoint}')
            rows.append(row)
            base.progress(label, variant, done=True, phase='Cached')
            continue
        if pitches is None:
            pitches = base.blob(pitch_path)
        recording = base._recording(item, dict(pitch_params, **variant['params']), deepcopy(pitches))
        timing = base.read(timing_path) if timing_path.exists() else {}
        if artifact.exists():
            notes = base.blob(artifact)
        else:
            if initial.exists():
                recording.note_data = base.blob(initial)
                # Rebuild the same bootstrap score state as fresh segmentation.
                recording.resize_score(to_span='pitch', include_transitions=False)
                recording.update_min_note_length()
            else:
                started = perf_counter()
                recording.note_data = detect_notes(recording, variant['method'])
                timing['segmentation_seconds'] = perf_counter() - started
                base.atomic(initial, recording.note_data, True)
                base.atomic(timing_path, timing)
            # Exactly the score-conditioned refinement used by the note benchmark:
            # original-note alignment, one robust fit, and local repeat recovery.
            started = perf_counter()
            recording.align_score_and_refine()
            timing['refinement_seconds'] = perf_counter() - started
            notes = recording.note_data
            base.atomic(artifact, notes, True)
            base.atomic(timing_path, timing)
        segmentation = timing.get('segmentation_seconds')
        refinement = timing.get('refinement_seconds')
        compute = segmentation + refinement if segmentation is not None and refinement is not None else None
        metrics = base._note_metrics(reference, notes, recording.config)
        row = dict(track_id=item['track_id'], instrument=item['instrument'], dataset=item.get('dataset', 'coco'),
                   method=variant['method'], variant=variant['variant'],
                   artifact=str(artifact), initial_artifact=str(initial),
                   evaluation_stage=EVALUATION_STAGE,
                   segmentation_seconds=segmentation, refinement_seconds=refinement,
                   compute_seconds=compute,
                   score=metrics['f1_offset'], **metrics, **variant['params'])
        base.atomic(checkpoint, row)
        rows.append(row)
        base.progress(label, variant, done=True, phase='Tested')
    return rows


def _finish(directory, batches, variants, items):
    import pandas as pd
    frame = pd.DataFrame([row for batch in batches for row in batch])
    expected_ids = {item['track_id'] for item in items}
    if (frame.empty or frame.duplicated(['track_id', 'variant']).any()
            or set(frame.variant) != {v['variant'] for v in variants}
            or any(set(group.track_id) != expected_ids for _, group in frame.groupby('variant'))):
        raise ValueError('Every method/parameter combination must cover the complete cohort')
    summary = base.summarize_scores(frame, 'note')
    specs = {v['variant']: v for v in variants}
    summary.insert(1, 'method', summary.variant.map(lambda key: specs[key]['method']))
    for name in BASELINE:
        summary[name] = summary.variant.map(lambda key: specs[key]['params'][name])
    summary = summary.sort_values(['score', 'variant'], ascending=[False, True], kind='stable')
    leaders = summary.drop_duplicates('method')

    def display_table(table):
        return (base.result_table(table, 'note').rename(columns={'method': 'Method'})
                .replace({'Method': METHOD_LABELS}))

    directory = Path(directory)
    if 'dataset' in frame:
        datasets = pd.concat([base.summarize_scores(g.drop(columns='dataset'), 'note').assign(dataset=d)
                              for d, g in frame.groupby('dataset')], ignore_index=True)
        datasets['method'] = datasets.variant.map(lambda v: specs[v]['method'])
        for name in BASELINE:
            datasets[name] = datasets.variant.map(lambda v: specs[v]['params'][name])
        display_table(datasets).to_csv(directory / 'by_dataset.csv', index=False)
    display_table(frame).to_csv(directory / 'rows.csv', index=False)
    display_table(summary).to_csv(directory / 'summary.csv', index=False)
    display_table(leaders).to_csv(directory / 'best_by_method.csv', index=False)
    winners = {}
    for row in leaders.itertuples():
        winners[row.method] = dict(
            method=row.method, params=specs[row.variant]['params'],
            transcription_f1=row.score, evaluation_stage=EVALUATION_STAGE,
            artifacts={r.track_id: dict(path=r.artifact, sha256=base.file_hash(r.artifact))
                       for r in frame[frame.variant == row.variant].itertuples()},
        )
    base.atomic(directory / 'best_by_method.json', winners)
    base.atomic(directory / 'best.json', winners[leaders.iloc[0]['method']])
    return display_table(summary), display_table(leaders)


def winning_method_variants(root, methods=None, axes=None):
    comparison = variants_for(methods)
    path = Path(root) / 'methods' / base.digest(comparison)[:16] / 'best.json'
    if not path.is_file():
        raise RuntimeError('Complete the baseline method-comparison cell before parameter tuning')
    winner = base.read(path)
    method = winner['method']
    if method not in {v['method'] for v in comparison}:
        raise ValueError('Saved winner is not in the baseline method comparison')
    return (variants_for((method,), base.NOTE_AXES if axes is None else axes),
            dict(method=method, comparison_best=str(path), sha256=base.file_hash(path)))


def validate_pitch_upstream(upstream, identity, items, winner):
    """Accept frozen pitch artifacts across the explicitly audited promotion only.

    Current note code keeps its own (unmodified) identity, so changed refinement
    defaults never reuse historical note checkpoints under a false identity.
    """
    if upstream['items'] != items:
        raise ValueError('Pitch inputs changed; refresh the pitch run before note tuning')
    if winner.get('contract_sha256') != base.digest(upstream):
        raise ValueError('Pitch winner does not match its saved contract')
    if set(winner['artifacts']) != {item['track_id'] for item in items}:
        raise ValueError('Pitch winner does not cover the frozen cohort')
    if upstream['identity'] == identity:
        return
    normalized = deepcopy(identity)
    promotion = base.read(Path(__file__).with_name('production_promotion_2026_11_17.json'))
    for path, change in promotion.items():
        if (upstream['identity']['code'].get(path) == change['before']
                and normalized['code'].get(path) == change['after']):
            normalized['code'][path] = change['before']
    if normalized != upstream['identity']:
        raise ValueError('Pitch inputs/code changed beyond the audited defaults promotion; '
                         'refresh the pitch run before note tuning')


def prepare(output):
    output = Path(output).resolve()
    cohort = base.read(output / 'cohort.json')
    items = base.inputs(cohort)
    pitch_winner = base.load_winner(output, 'pitch')
    identity = base.code_identity()
    upstream = base.read(output / 'pitch/contract.json')
    validate_pitch_upstream(upstream, identity, items, pitch_winner)
    source = dict(cohort=cohort, items=items, pitch_winner=pitch_winner,
                  identity=identity, runner_sha256=base.file_hash(__file__),
                  rbf_cost_sha256=base.file_hash(Path(__file__).with_name('fast_rbf.py')),
                  method_configs=METHOD_CONFIGS,
                  protocol='44100 Hz; jump=1; capped minimum; smoothed pitch values; no transitions; production consolidation; production alignment/refit/repeat recovery; equal dataset weights',
                  evaluation_stage=EVALUATION_STAGE)
    root = output / 'note_ruptures' / base.digest(source)[:16]
    # Both phases share immutable, method+parameter keyed note caches.
    base.contract(root, source)
    return items, pitch_winner, root, source


def run(output=base.DEFAULT_OUTPUT, *, stage, methods=None, axes=None, workers=8, selected_method=None):
    if selected_method is not None and (stage != 'parameters' or selected_method not in METHOD_CONFIGS):
        raise ValueError('selected_method must name a known method for parameter tuning')
    if stage not in ('methods', 'parameters'):
        raise ValueError('Stage must be methods or parameters')
    if stage == 'methods' and axes is not None:
        raise ValueError('The method comparison uses the common BASELINE')
    variants = variants_for(methods)
    items, pitch_winner, root, source = prepare(output)
    selection_items = method_selection_items(items)
    base.atomic(root / 'method_selection.json', dict(
        rule='entire frozen cohort: all 195 Coco tracks plus all supplemental dataset tracks',
        track_ids=[item['track_id'] for item in selection_items]))
    selection = None
    if stage == 'parameters':
        if selected_method is None:
            variants, selection = winning_method_variants(root, methods, axes)
        else:
            variants = variants_for((selected_method,), base.NOTE_AXES if axes is None else axes)
            selection = dict(method=selected_method, basis='explicit provisional choice; comparison not required')
    if stage == 'methods':
        items = selection_items
    reuse_previous_checkpoints(root, source, items, variants)
    stage_key = 'parameters_provisional' if selected_method is not None else stage
    directory = root / stage_key / base.digest(variants)[:16]
    base.contract(directory, dict(source=source, variants=variants, selection=selection,
                                 track_ids=[item['track_id'] for item in items]))
    batches = base.parallel(_note, [(item, str(root / 'checkpoints'), variants,
        pitch_winner['params'], pitch_winner['artifacts'][item['track_id']]['path'])
        for item in items], workers=workers)
    summary, leaders = _finish(directory, batches, variants, items)
    if stage == 'parameters' and selected_method is None:
        destination = Path(output) / 'note'
        base.atomic(destination / 'best.json', base.read(directory / 'best.json'))
        base.atomic(destination / 'selection.json', dict(
            source_sha256=base.digest(source), best_path=str(directory / 'best.json'),
            best_sha256=base.file_hash(directory / 'best.json'),
            pitch_best_sha256=base.file_hash(Path(output) / 'pitch/best.json')))
    return dict(summary=summary, best_by_method=leaders, directory=directory,
                selected_method=base.read(directory / 'best.json')['method'])
