"""Frozen 195 Coco + one-per-instrument Bach10/URMP development cohort.

Historical results stay immutable. Only the reviewed 44.1 kHz pitch computation
is migrated; note results must be recomputed with production consolidation.
"""
from pathlib import Path
import os
import shutil

from benchmarks.modules.sweeps import AppendixSweeps as base

LEGACY_OUTPUT = base.ROOT / 'benchmarks/results/sweeps/appendix_195_44100_v2'
LEGACY_RUNNER_HASH = '6a205fdabade7cf79518ca330b944bb4f331c13c71ad85891b66024cc4c1b6d1'
DATASETS = {'coco': 195, 'bach10-original': 4, 'urmp': 13}


def validate_items(items):
    for item in items:
        if {k: base.file_hash(p) for k, p in item['paths'].items()} != item['hashes']:
            raise ValueError(f"Source changed: {item['track_id']}; use a new cohort")
    return items


def materialize(output=base.DEFAULT_OUTPUT, *, workers=8, seed=20261116,
                root=None, split='test', legacy_output=LEGACY_OUTPUT):
    from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation, NotebookConfig
    import pretty_midi
    output, legacy_output = Path(output).resolve(), Path(legacy_output).resolve()
    legacy = base.read(legacy_output / 'cohort.json')
    if split != legacy['selection']['split'] or (root is not None and
            Path(root).resolve() != Path(legacy['selection']['root']).resolve()):
        raise ValueError('Keep the frozen Coco cohort and its original dataset root')
    selection = dict(seed=seed, counts=DATASETS, legacy_cohort_sha256=base.file_hash(legacy_output / 'cohort.json'),
                     root=legacy['selection']['root'], split=split)
    target = output / 'cohort.json'
    if target.exists():
        saved = base.read(target)
        if saved['selection'] != selection:
            raise ValueError('Frozen sample changed; choose a new output directory')
        validate_items(saved['items'])
        return saved
    items = [dict(item, dataset='coco', group=item['record']['track'],
                  note_reference=dict(dataset='coco', reference=item['paths']['midi']))
             for item in base.inputs(legacy)]
    evaluation = NoteEvaluation(NotebookConfig(preliminary_tracks_per_instrument=1, seed=seed))
    for dataset in ('bach10-original', 'urmp'):
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        source = PitchBenchmarker.dataset_for(dataset, PitchBenchmarker.Options())
        tracks = {t.track_id: t for t in source.tracks()}
        for task in evaluation.select_tracks(dataset):
            track = tracks[task['track_id']]
            track_id = dataset + '__' + task['track_id'].replace('/', '__')
            midi_path = output / 'scores' / (track_id + '.mid')
            original = pretty_midi.PrettyMIDI(task['score'])
            part = original.instruments[task['score_part']]
            if part.is_drum or not part.notes:
                raise ValueError(f'Invalid score part: {track_id}')
            midi = pretty_midi.PrettyMIDI()
            midi.instruments.append(part)
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            midi.write(str(midi_path))
            paths = dict(audio=task['audio'], midi=str(midi_path), f0=task['pitch_annotation'],
                         reference=task['reference'], original_score=task['score'])
            items.append(dict(track_id=track_id, source_track_id=track.track_id,
                              dataset=dataset, instrument=task['instrument'], group=task['group'],
                              pitch_metadata=track.metadata, note_reference=task, paths=paths,
                              hashes={k: base.file_hash(p) for k, p in paths.items()}))
    from collections import Counter
    if Counter(i['dataset'] for i in items) != DATASETS:
        raise ValueError('Expected exactly 195 Coco, 4 Bach10 and 13 URMP stems')
    for dataset, count in [('bach10-original', 4), ('urmp', 13)]:
        counts = Counter(i['instrument'] for i in items if i['dataset'] == dataset)
        if len(counts) != count or set(counts.values()) != {1}:
            raise ValueError(f'Expected one stem per instrument: {dataset}')
    saved = dict(selection=selection, records=legacy['records'], items=items,
                 reuse_pitch_from=str(legacy_output / 'pitch'),
                 policy='frozen Coco; seeded random one-per-instrument real stems; equal dataset metric weights')
    base.atomic(target, saved)
    return saved


def pitch_reference(item):
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    from benchmarks.modules.pitch.datasets.PitchDataset import PitchTrack
    source = PitchBenchmarker.dataset_for(item['dataset'], PitchBenchmarker.Options())
    track = PitchTrack(item['source_track_id'], item['dataset'], Path(item['paths']['audio']),
                       Path(item['paths']['f0']), item['pitch_metadata'])
    return source.reference(track)


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def reuse_pitch(source, destination, current):
    """Validate source/code/grid identity, then reuse completed Coco evidence.

    The only allowed code differences are this new orchestration module and the
    reviewed AppendixSweeps update (dataset routing/reporting, unchanged Coco DSP).
    No legacy 16 kHz or changed detector/config cache is accepted.
    """
    source, destination = Path(source), Path(destination)
    old = base.read(source / 'contract.json')
    helper = 'benchmarks/modules/sweeps/AppendixSweeps.py'
    added = 'benchmarks/modules/sweeps/MixedCohort.py'
    old_code, new_code = old['identity']['code'], current['identity']['code']
    if (old_code.get(helper) != LEGACY_RUNNER_HASH
            or {k: v for k, v in old_code.items() if k != helper} !=
               {k: v for k, v in new_code.items() if k not in (helper, added)}
            or old['identity']['packages'] != current['identity']['packages']
            or not all(v in current['variants'] for v in old['variants'])):
        raise ValueError('Historical pitch code/packages/grid incompatible; do not reuse')
    originals = {i['track_id']: i for i in old['items']}
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    import pandas as pd
    for item in current['items']:
        if item['dataset'] != 'coco':
            continue
        old_item = originals.get(item['track_id'])
        if old_item is None or any(item.get(k) != v for k, v in old_item.items()):
            raise ValueError(f"Historical pitch source changed: {item['track_id']}")
        src, dst = source / item['track_id'], destination / item['track_id']
        rows = base.read(src / 'rows.json')
        expected = {v['variant']: v['params'] for v in old['variants']}
        if (len(rows) != len(expected) or {r['variant'] for r in rows} != set(expected)
                or any(r['track_id'] != item['track_id'] or
                       any(r[k] != v for k, v in expected[r['variant']].items()) for r in rows)
                or not PitchBenchmarker.current_scores(pd.DataFrame(rows))):
            raise ValueError(f'Invalid historical pitch rows: {src}')
        # HMM evidence for every volume is required if the expanded winner changes.
        required = [src / 'raw.pkl'] + [src / ('tracking_' + base.digest(dict(volume=v))[:16] + '.pkl')
                    for v in {p['posthoc_min_volume'] for p in expected.values()}]
        if any(not p.is_file() for p in required):
            raise FileNotFoundError(f'Incomplete reusable pitch evidence: {src}')
        for path in src.glob('*.pkl'):
            _link(path, dst / path.name)
        if not (dst / 'rows.json').exists():
            base.atomic(dst / 'rows.json', [dict(r, dataset='coco',
                artifact=str(dst / (r['variant'] + '.pkl'))) for r in rows])
    base.atomic(destination / 'reuse.json', dict(source=str(source),
                source_contract_sha256=base.file_hash(source / 'contract.json'), tracks=len(originals)))


def selected_frontend(output):
    """Resolve completed, compatible winners; never guess defaults or directory order."""
    output = Path(output)
    pitch = base.load_winner(output, 'pitch')
    note = base.load_winner(output, 'note')
    pointer = base.read(output / 'note/selection.json')
    from benchmarks.notebooks.sweeps.scripts import note_method_sweeps as runner
    items, current_pitch, _, source = runner.prepare(output)
    if (pointer['source_sha256'] != base.digest(source)
            or pointer['best_sha256'] != base.file_hash(pointer['best_path'])
            or pointer['pitch_best_sha256'] != base.file_hash(output / 'pitch/best.json')
            or note != base.read(pointer['best_path'])):
        raise ValueError('Note winner is stale; complete note tuning after pitch tuning')
    if set(note['artifacts']) != {i['track_id'] for i in items}:
        raise ValueError('Note winner does not cover the complete cohort')
    params = dict(pitch['params'], **note['params'], note_method=note['method'])
    return params, dict(pitch_best_sha256=base.file_hash(output / 'pitch/best.json'),
                        note_best_sha256=base.file_hash(output / 'note/best.json'), selection=pointer)


from functools import lru_cache


@lru_cache(maxsize=1)
def _legacy_mistake_contract():
    source = base.ROOT / 'benchmarks/results/sweeps/appendix_195_v1/mistake'
    path = source / 'contract.json'
    if not path.exists():
        return None
    old = base.read(path)
    current = base.code_identity()
    ignored = {'benchmarks/modules/sweeps/AppendixSweeps.py',
               'benchmarks/modules/sweeps/MixedCohort.py'}
    if (old['identity']['code'].get('benchmarks/modules/sweeps/AppendixSweeps.py') != LEGACY_RUNNER_HASH
            or {k: v for k, v in old['identity']['code'].items() if k not in ignored} !=
               {k: v for k, v in current['code'].items() if k not in ignored}
            or old['identity']['packages'] != current['packages']):
        return None
    from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker
    from algorithms.Config import Config
    if (old['soundfont_sha256'] != base.file_hash(NoteBenchmarker.SOUNDFONT_PATH)
            or old['defaults']['timing_tolerance'] != Config().timing_tolerance):
        return None
    return source, old


def reusable_e2(item, seed, rate):
    """Read-only reuse of identical injected MIDI/audio and raw acoustic evidence."""
    legacy = _legacy_mistake_contract()
    if legacy is None or item.get('dataset', 'coco') != 'coco':
        return None
    source, _ = legacy
    folder = source / 'frontend' / item['track_id'] / f'{seed}_{rate}'
    path = folder / 'provenance.json'
    if not path.exists():
        return None
    saved = base.read(path)
    asset = saved['source']
    spec = asset['spec']
    if (spec['source_hash'] != item['hashes']['midi'] or spec['seed'] != seed
            or spec['rate'] != rate or spec['weights'] != [.2] * 5
            or spec['timeline_protocol'] != 'monophonic_edits_v3'):
        return None
    if any(base.file_hash(asset[k]) != h for k, h in saved['hashes'].items()):
        raise ValueError(f'Historical E2 asset changed: {path}')
    return dict(asset=asset, config=saved['config'], pitch=str(folder / 'pitch.pkl.xz'))


def restore_e2_raw(recording, reusable):
    if reusable is None:
        return False
    # Raw candidates precede the selected post-hoc gates and segmentation.
    # All other configuration, including pitch range and live gate, must match.
    from dataclasses import asdict
    ignored = set(base.NOTE_AXES) | set(base.PITCH_AXES) | {'min_note_length'}
    current = asdict(recording.config)
    if any(current.get(k) != v for k, v in reusable['config'].items() if k not in ignored):
        return False
    from benchmarks.modules.pitch.PitchCache import PitchCache
    hit = PitchCache(reusable['pitch']).read(PitchCache.RAW, recording.config)
    if hit is None:
        return False
    from algorithms.PitchSmoother import PitchSmoother, VoicingSmoother
    from benchmarks.modules.pitch.competitors.Attune import Attune
    raw, _ = hit
    tracked = PitchSmoother(config=recording.config).smooth(raw.data, verbose=False)
    final = VoicingSmoother(config=recording.config).smooth(tracked)
    recording.pitch_data = Attune._pitch_data(final, recording.config, raw.t_origin)
    return True


def extend_pitch_contract(directory, current):
    """Reviewed additive-grid migration; preserve all per-track evidence."""
    from copy import deepcopy
    directory = Path(directory)
    path = directory / 'contract.json'
    if not path.exists():
        return
    old = base.read(path)
    if old == current:
        return
    predecessors = {
        'benchmarks/modules/sweeps/AppendixSweeps.py': 'a1084247de416c0d0e949aff4a4bb2171ee9c79cf3e98fd19c3ae8680015adb3',
        'benchmarks/modules/sweeps/MixedCohort.py': '03e6cbdb78e8b0715c50f13ec857aaad18bab8767aadcc0b12c71f742f3ad3b0',
    }
    normalized = deepcopy(old)
    for name, expected in predecessors.items():
        if normalized['identity']['code'].get(name) == expected:
            normalized['identity']['code'][name] = current['identity']['code'][name]
    if not all(v in current['variants'] for v in old['variants']):
        raise ValueError('Pitch grid migration only accepts additive candidates')
    normalized['variants'] = current['variants']
    if normalized != current:
        raise ValueError('Pitch inputs or inference changed; cannot reuse this run')
    archive = directory / 'report_history' / base.digest(old)[:16]
    for name in ('contract.json', 'rows.csv', 'summary.csv', 'by_dataset.csv', 'best.json'):
        source = directory / name
        if source.exists():
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
    # A winner is valid only after all added candidates have completed.
    (directory / 'best.json').unlink(missing_ok=True)
    base.atomic(path, current)
