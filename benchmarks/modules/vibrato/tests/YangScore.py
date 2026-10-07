"""Opt-in paired Yang score/boundary experiment; never imported by production.

prepare: freeze common input, candidate DTW, symbolic excerpts and boundaries.
run score: compare audio-only, production score/repeat flow, Yang DT and BR.
run dtw: run Attune with the frozen DTW boundaries (an estimate, not an oracle).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAP = Path(__file__).with_name('yang_score_candidates.json')
VERSION = 'yang_score_boundary_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def melody_spans(midi, instrument=None):
    """Exact skyline intervals retaining source-event identity, including repeats."""
    notes = [(n.start, n.end, n.pitch, i, j)
             for i, part in enumerate(midi.instruments)
             if not part.is_drum and (instrument is None or i == instrument)
             for j, n in enumerate(part.notes) if n.end > n.start]
    if not notes:
        raise ValueError('No notes in selected MIDI instrument')
    edges = sorted({t for n in notes for t in n[:2]})
    spans = []
    for start, end in zip(edges, edges[1:]):
        active = [n for n in notes if n[0] <= start < n[1]]
        if not active:
            continue
        chosen = max(active, key=lambda n: (n[2], n[0], n[3], n[4]))
        identity = chosen[3:]
        if spans and spans[-1][1] == start and spans[-1][3] == identity:
            spans[-1] = (spans[-1][0], end, chosen[2], identity)
        else:
            spans.append((start, end, chosen[2], identity))
    return spans


def match(spans, times, pitch, step=.1, shifts=range(-24, 25)):
    """Frozen earlier pitch-cost search; never consults vibrato labels."""
    import librosa
    mt = np.arange(spans[0][0], spans[-1][1], step)
    mp = np.full(len(mt), np.nan)
    for start, end, midi, _ in spans:
        mp[(mt >= start) & (mt < end)] = midi
    valid = np.isfinite(mp)
    mt, mp = mt[valid], mp[valid]
    at = np.arange(times[0], times[-1], step)
    ap = np.array([np.median(pitch[(abs(times-t) <= .1) & np.isfinite(pitch)])
                   if np.any((abs(times-t) <= .1) & np.isfinite(pitch)) else np.nan
                   for t in at])
    valid = np.isfinite(ap)
    at, ap = at[valid], ap[valid]
    if min(len(mt), len(at)) < 3:
        raise ValueError('Insufficient voiced samples for DTW')
    tuning = float(np.median(ap - np.round(ap)))
    best = None
    for shift in shifts:
        cost = np.minimum(abs(ap[:, None] - (mp[None, :] + shift + tuning)), 4.)
        try:
            distance, path = librosa.sequence.dtw(
                C=cost, subseq=True,
                step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1], [1, 3], [3, 1]]),
                weights_mul=np.array([1., 1.5, 1.5, 2., 2.]))
        except librosa.util.exceptions.ParameterError:
            continue
        # librosa flips the returned axes for tall precomputed cost matrices.
        path = path[::-1]
        if cost.shape[0] > cost.shape[1]:
            path = path[:, ::-1]
        ai, mi = path.T
        if ai[0] != 0 or ai[-1] != len(at)-1:
            raise ValueError('DTW did not cover the complete voiced audio query')
        score = float(distance[-1, mi[-1]] / len(ap))
        if best is None or score < best[0]:
            best = score, int(shift), mi, ai
    if best is None:
        raise ValueError('No feasible DTW path; score/query duration mismatch')
    score, shift, mi, ai = best
    path = np.column_stack((mt[mi], at[ai], mp[mi]+shift+tuning, ap[ai]))
    error = abs(path[:, 2]-path[:, 3])
    diagnostics = dict(mean_clipped_cost=score, transpose_semitones=shift,
        tuning_cents=tuning*100, fraction_path_within_1_semitone=float(np.mean(error <= 1)),
        median_abs_error_semitones=float(np.median(error)), step_seconds=step,
        status='unverified_candidate', boundary_source='dtw_estimated_not_oracle')
    return path, diagnostics


def excerpts(spans, path, shift, audio_end, step=.1):
    """Clip symbolic rhythm; independently warp its event edges to audio time."""
    left, right = float(path[0, 0]), float(path[-1, 0]+step)
    # DTW steps are strictly increasing along each axis. Include a final edge
    # so the final sampled note is not silently dropped or zero-duration.
    x = np.r_[path[:, 0], right]
    y = np.r_[path[:, 1], min(audio_end, path[-1, 1]+step)]
    symbolic, warped = [], []
    for start, end, pitch, _ in spans:
        start, end = max(start, left), min(end, right)
        if end <= start:
            continue
        midi = int(pitch + shift)
        if not 0 <= midi <= 127:
            raise ValueError('Transposed score pitch outside MIDI range')
        symbolic.append((start-left, end-left, midi))
        a, b = np.interp([start, end], x, y)
        if b <= a:
            raise ValueError('DTW produced a collapsed note boundary')
        warped.append((float(a), float(b), midi))
    if not symbolic:
        raise ValueError('Empty matched score excerpt')
    return symbolic, warped


def write_midi(bounds, path):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=9600)
    part = pretty_midi.Instrument(program=0)
    part.notes = [pretty_midi.Note(90, int(p), float(a), float(b)) for a, b, p in bounds]
    midi.instruments.append(part)
    midi.write(str(path))


def freeze_pitch(config, pitch_data):
    # PitchData itself contains a lock and lambda; freeze only its data/state.
    return config, pitch_data.t_origin, pitch_data.data[:pitch_data.frames_available()]


def thaw_pitch(frozen):
    from app_logic.user.ds.PitchData import PitchData
    config, origin, frames = frozen
    config = replace(config)
    pitch_data = PitchData(config)
    pitch_data.t_origin = origin
    pitch_data.load(frames)
    return config, pitch_data


def prepare(args):
    import pretty_midi
    from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
    from benchmarks.modules.vibrato.datasets.CocoDataset import CocoDataset
    from benchmarks.modules.vibrato.datasets.YangFullDataset import YangFullDataset

    mapping = json.loads(args.mapping.read_text())['recordings']
    discovered = YangFullDataset.discover(args.dataset_root)
    by_id = {r.recording_id: r for r in discovered}
    if len(by_id) != len(discovered):
        raise ValueError('Duplicate recording IDs: use a corpus with unique stems')
    if not mapping:
        raise ValueError('The recording-to-score mapping is empty')
    missing = set(mapping)-set(by_id)
    if missing:
        raise ValueError(f'Mapped recordings missing from corpus: {sorted(missing)}')
    args.output.mkdir(parents=True, exist_ok=False)
    rows, examples, inputs = [], [], {}
    for index, (recording_id, spec) in enumerate(mapping.items()):
        print(f'Preparing {recording_id}', flush=True)
        recording = by_id[recording_id]
        source = (args.mapping.parent / spec['midi']).resolve()
        job = YangFullDataset._PitchJob(index, recording, str(args.cache_root), True, False)
        _, built = YangFullDataset._build_recording(job)
        example = built[0]
        # Keep the real cached confidence/volume frames for production segmentation.
        bench = AttuneRealtime()
        fmin, fmax = recording.pitch_range_hz
        config = bench.config_for(fmin, fmax, w1=CocoDataset.automatic_yin_window_size(fmin, padding_semitones=0.))
        cache = args.cache_root/'full_audio_pad8_v1'/recording.instrument/recording.performer/f'{recording_id}.pitch.pkl.xz'
        pitch_data, _ = bench.load_pitch_data(cache, config, smooth=True)
        times, values = YangFullDataset._pitch_arrays(pitch_data, config)
        if not (np.array_equal(times, example.times) and np.array_equal(values, example.pitch_midi, equal_nan=True)):
            raise ValueError('Production pitch cache differs from common benchmark contour')
        inputs[recording_id] = freeze_pitch(config, pitch_data)
        spans = melody_spans(pretty_midi.PrettyMIDI(str(source)), spec.get('instrument'))
        path, diagnostics = match(spans, example.times, example.pitch_midi)
        symbolic, warped = excerpts(spans, path, diagnostics['transpose_semitones'],
                                     example.times[-1]+1/example.frame_rate)
        folder = args.output/recording_id
        folder.mkdir()
        write_midi(symbolic, folder/'score.mid')
        np.savetxt(folder/'dtw_path.csv', path, delimiter=',',
            header='score_seconds,audio_seconds,score_pitch_plus_tuning,audio_pitch', comments='')
        (folder/'dtw_bounds.json').write_text(json.dumps(warped, indent=2)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(example.times, example.pitch_midi, lw=.5, alpha=.6, label='Common pitch')
        ax.plot(path[:, 1], path[:, 2], lw=1, label='Candidate DTW score')
        for start, end, midi in warped:
            ax.hlines(midi, start, end, color='black', lw=2)
            ax.axvline(start, color='grey', alpha=.25, lw=.5)
        ax.set(title=f'{recording_id}: unverified DTW boundaries', xlabel='Audio seconds', ylabel='MIDI pitch')
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder/'alignment.png', dpi=140)
        plt.close(fig)
        # Keep the baseline segmentation, including an explicit empty result.
        from algorithms.NoteDetector import NoteDetector
        take = bench.recording_for(config)
        notes = NoteDetector(take, config=config).detect_notes(pitch_data.data[:pitch_data.frames_available()])
        baseline = [(float(n.start_time), float(n.end_time), float(n.midi_num[0])) for n in notes.data.values()]
        examples.append(replace(example, metadata={**example.metadata,
            'analysis_note_bounds': baseline, 'experiment_version': VERSION}))
        rows.append(dict(recording=recording_id, source_midi=str(source),
            source_sha256=digest(source), score_sha256=digest(folder/'score.mid'),
            audio_sha256=digest(recording.audio_path), pitch_cache_sha256=digest(cache),
            dtw_bounds_sha256=digest(folder/'dtw_bounds.json'),
            score_note_count=len(symbolic), **diagnostics))
    with (args.output/'inputs.pkl').open('wb') as f:
        pickle.dump((examples, inputs), f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = dict(version=VERSION, recordings=rows,
        inputs_sha256=digest(args.output/'inputs.pkl'), mapping=mapping,
        excluded_recordings=sorted(set(by_id)-set(mapping)),
        protocol='Same full-recording pitch/truth; no match-quality exclusions; candidate scores, not verified truth')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Prepared {len(rows)} candidates; {len(manifest["excluded_recordings"])} unmapped recordings. No F1 scoring performed.')


def run(args):
    from app_logic.midi.ScoreData import ScoreData
    from app_logic.user.ds.Recording import Recording
    from benchmarks.modules.vibrato.competitors.Attune import Attune
    from benchmarks.modules.vibrato.competitors.Yang import YangDT, YangBR
    from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker

    manifest = json.loads((args.prepared/'manifest.json').read_text())
    if manifest['version'] != VERSION or digest(args.prepared/'inputs.pkl') != manifest['inputs_sha256']:
        raise ValueError('Prepared inputs changed; prepare a new experiment')
    # Only load the locally generated experiment pickle.
    with (args.prepared/'inputs.pkl').open('rb') as f:
        examples, inputs = pickle.load(f)
    args.output.mkdir(parents=True, exist_ok=False)
    variants = []
    for example, row in zip(examples, manifest['recordings'], strict=True):
        folder = args.prepared/example.case_id
        for name, key in [('score.mid', 'score_sha256'), ('dtw_bounds.json', 'dtw_bounds_sha256')]:
            if digest(folder/name) != row[key]:
                raise ValueError(f'{folder/name} changed; prepare a new experiment')
        if args.mode == 'score':
            config, pitch_data = thaw_pitch(inputs[example.case_id])
            score = ScoreData(folder/'score.mid')
            actual = score.note_datas[score.active_instrument]
            if len(actual.times) != row['score_note_count']:
                raise ValueError(f'{example.case_id}: MIDI import changed note count; inspect score before running')
            take = Recording(score_data=score, config=replace(config))
            take.pitch_data = pitch_data
            take.analyze_notes()
            bounds = [(float(n.start_time), float(n.end_time), float(n.midi_num[0]))
                      for n in take.note_data.data.values()]
            source = 'production_detect_align_repeat_with_candidate_score'
        else:
            bounds = json.loads((folder/'dtw_bounds.json').read_text())
            source = 'dtw_estimated_boundaries_not_oracle'
        metadata = {**example.metadata, 'analysis_note_bounds': bounds,
                    'note_boundary_source': source, 'score_match': row}
        variants.append(replace(example, metadata=metadata))
    benchmark = VibratoBenchmarker()
    # Separate calls keep detector identities/configurations identical; same
    # examples and evaluation masks are used for every arm.
    arms = [('attune_score' if args.mode == 'score' else 'attune_dtw', variants, [Attune()])]
    if args.mode == 'score':
        arms.insert(0, ('common_baselines', examples, [Attune(), YangDT(), YangBR()]))
    summaries = []
    for label, cases, detectors in arms:
        output = args.output/label
        raw = benchmark.run(cases, detectors, workers=args.workers, strict=True,
                            cache_dir=output/'checkpoints')
        benchmark.write_reports(raw, output)
        (output/'note_bounds.json').write_text(json.dumps({e.case_id: e.metadata['analysis_note_bounds'] for e in cases}, indent=2)+'\n')
        summary = benchmark.summarize(raw)
        summary.insert(0, 'arm', label)
        summaries.append(summary)
        print(label)
        print(benchmark.display_summary(benchmark.summarize(raw)).to_string(index=False))
    import pandas as pd
    pd.concat(summaries, ignore_index=True).to_csv(args.output/'comparison.csv', index=False)
    (args.output/'run_config.json').write_text(json.dumps(dict(mode=args.mode,
        prepared_manifest=manifest, prepared_manifest_sha256=digest(args.prepared/'manifest.json'),
        module_sha256=digest(__file__), workers=args.workers), indent=2)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare', help='Prepare common pitch, DTW and MIDI assets; does not score F1')
    prep.add_argument('--mapping', type=Path, default=DEFAULT_MAP)
    prep.add_argument('--dataset-root', type=Path, default=ROOT/'benchmarks/datasets/vibrato')
    prep.add_argument('--cache-root', type=Path, default=ROOT/'benchmarks/datasets/vibrato/pitch_data/attune')
    prep.add_argument('--output', type=Path, required=True)
    runner = subs.add_parser('run')
    runner.add_argument('--prepared', type=Path, required=True)
    runner.add_argument('--mode', choices=['score', 'dtw'], required=True)
    runner.add_argument('--output', type=Path, required=True)
    runner.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if getattr(args, 'workers', 1) < 1:
        parser.error('--workers must be positive')
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
