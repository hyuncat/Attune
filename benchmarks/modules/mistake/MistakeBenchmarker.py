"""MistakeBenchmarker implementation and owned benchmark helpers."""
from __future__ import annotations
from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
from benchmarks.modules.mistake.MistakeCache import MistakeCache
import sys
import time
import json
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Literal
from typing import TypedDict
from typing import TypeAlias
import numpy as np
_mistakebenchmarker_BOOTSTRAP_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file()))
if str(_mistakebenchmarker_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_mistakebenchmarker_BOOTSTRAP_ROOT))
from benchmarks.paths import REPO_ROOT
from benchmarks.paths import ensure_repo_on_path
ensure_repo_on_path()
ROOT = REPO_ROOT
from app_logic.Alignment import Mistake
from algorithms.Config import Config
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData
from app_logic.user.ds.AudioData import AudioData
from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector
from benchmarks.modules.mistake.datasets.MistakeInjector import TruthEvent
from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
from benchmarks.modules.note.NoteBenchmarker import PathLike
MistakeMode: TypeAlias = Literal['symbolic', 'audio']
MistakeCounts: TypeAlias = dict[str, tuple[int, int, int]]
_mistakebenchmarker_SOURCE_SCORE_ID_UNSET = object()
from contextlib import redirect_stdout
from contextlib import redirect_stderr
import traceback
import os
from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace
from importlib.metadata import version
import hashlib
from time import perf_counter
import pandas as pd
from app_logic.user.ds.PitchData import PitchData
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.modules.mistake.datasets.MistakeCases import prepare_case
from benchmarks.modules.mistake.competitors.PolyTune import PolyTune
from benchmarks.modules.mistake.competitors.LadderSym import LadderSym
COMPETITOR_METHODS = ('Attune (repeat only)', 'Attune (Checker 3)', 'Attune (no refinement)', 'Parangonar DualDTW', 'Parangonar Automatic', 'Parangonar TheGlueNote', 'Nakamura', 'PolyTune', 'LadderSym')
COMPETITOR_AUDIO_METHODS = ('PolyTune', 'LadderSym')
COMPETITOR_SYMBOLIC_METHODS = tuple((m for m in COMPETITOR_METHODS if m not in ('Attune (repeat only)', 'Attune (Checker 3)', *COMPETITOR_AUDIO_METHODS)))
COMPETITOR_TYPES = ('substitution', 'deletion', 'insertion', 'short', 'long')
from benchmarks.modules.mistake.datasets.NativeDatasets import digest
from benchmarks.modules.mistake.datasets.NativeDatasets import save_json
NATIVE_METHODS = ('Attune', 'PolyTune', 'LadderSym')
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from concurrent.futures import FIRST_COMPLETED
from contextlib import contextmanager
import multiprocessing as mp
import queue
import threading
_parallel_EVENTS = None
_parallel_LIMITER = None
from concurrent.futures import as_completed
import resource
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')
import argparse
from benchmarks.modules.mistake.datasets.NativeDatasets import download
from benchmarks.modules.mistake.datasets.NativeDatasets import tree
SOUNDFONT_ROOT = Path(__file__).resolve().parents[3]
SOUNDFONT_BASE = SOUNDFONT_ROOT / 'benchmarks/results/native_v5_min100_seed0_pieces3/coco'
SOUNDFONT_OUT = SOUNDFONT_ROOT / 'benchmarks/results/native_soundfont_2026-09-29'

class MistakeBenchmarker(NoteBenchmarker):
    """Owns benchmark execution, parallel workers, scoring, and timing."""

    @staticmethod
    def _bootstrap_repo_root() -> Path:
        for candidate in Path(__file__).resolve().parents:
            if (candidate / 'app.py').is_file() and (candidate / 'benchmarks').is_dir():
                return candidate
        raise RuntimeError('could not locate Attune repo root')

    class MistakeScore(TypedDict, total=False):
        counts: MistakeCounts
        pitch_detector_compute_time: float
        pitch_smoother_compute_time: float
        pitch_compute_time: float
        note_compute_time: float
        mistake_detection_compute_time: float
        mistake_check_compute_time: float

    @staticmethod
    def preflight(methods=COMPETITOR_METHODS, polytune=None, laddersym=None):
        """Fail before expensive generation if an explicitly selected method is absent."""
        if not methods or len(set(methods)) != len(methods) or set(methods) - set(COMPETITOR_METHODS):
            raise ValueError(f'Select unique methods from {COMPETITOR_METHODS}')
        packages = {p: version(p) for p in ('numpy', 'scipy', 'mir_eval')}
        if any((m.startswith('Parangonar') for m in methods)):
            try:
                import parangonar
            except ImportError as exc:
                raise ImportError('Install benchmark dependencies in the notebook kernel: %pip install parangonar==3.3.3') from exc
            packages['parangonar'] = parangonar.__version__
            packages['partitura'] = version('partitura')
        if 'Parangonar TheGlueNote' in methods:
            try:
                import torch
                import symusic
                import miditok
            except ImportError as exc:
                raise ImportError('TheGlueNote requires torch, symusic and miditok. See the mistake README.') from exc
            checkpoint = Path(parangonar.THEGLUENOTE_CHECKPOINT)
            if not checkpoint.is_file():
                raise FileNotFoundError(f'TheGlueNote bundled weights missing: {checkpoint}')
            packages['TheGlueNote'] = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), device='cuda' if torch.cuda.is_available() else 'cpu', **{p: version(p) for p in ('torch', 'symusic', 'miditok', 'tokenizers')})
        if 'Nakamura' in methods:
            import benchmarks.modules.mistake.competitors.Nakamura as _api_Nakamura
            packages['Nakamura'] = nakamura_preflight()
        if 'PolyTune' in methods:
            packages['PolyTune'] = (polytune or PolyTune()).preflight()
        if 'LadderSym' in methods:
            packages['LadderSym'] = (laddersym or LadderSym()).preflight()
        return packages

    @staticmethod
    def note_array(notes, *, score=False, bpm=120.0):
        """Adapt the benchmark's monophonic, constant-tempo score representation."""
        fields = [('id', 'U64'), ('pitch', 'i4'), ('onset_sec', 'f8'), ('duration_sec', 'f8'), ('velocity', 'i4')]
        if score:
            fields += [('onset_beat', 'f8'), ('duration_beat', 'f8'), ('onset_quarter', 'f8'), ('duration_quarter', 'f8'), ('is_grace', '?')]
        array = np.zeros(len(notes), dtype=fields)
        for i, note in enumerate(notes):
            if len(note.midi_num) != 1:
                raise ValueError('This comparison requires monophonic note events.')
            array[i]['id'] = str(i)
            array[i]['pitch'] = int(np.rint(note.midi_num[0]))
            array[i]['onset_sec'] = note.start_time
            array[i]['duration_sec'] = note.duration()
            array[i]['velocity'] = note.velocity or 64
            if score:
                array[i]['onset_beat'] = note.start_time * bpm / 60.0
                array[i]['duration_beat'] = note.duration() * bpm / 60.0
                array[i]['onset_quarter'] = array[i]['onset_beat']
                array[i]['duration_quarter'] = array[i]['duration_beat']
        return array

    @staticmethod
    def external_pairs(method, score_notes, user_notes, bpm):
        """Validate one-to-one coverage; never silently discard competitor output."""
        if not score_notes or not user_notes:
            return [(None, s) for s in score_notes] + [(u, None) for u in user_notes]
        if method == 'Nakamura':
            import benchmarks.modules.mistake.competitors.Nakamura as _api_Nakamura
            alignment = align(score_notes, user_notes)
        else:
            import parangonar
            classes = {'Parangonar DualDTW': parangonar.DualDTWNoteMatcher, 'Parangonar Automatic': parangonar.AutomaticNoteMatcher, 'Parangonar TheGlueNote': parangonar.TheGlueNoteMatcher}
            alignment = classes[method]()(MistakeBenchmarker.note_array(score_notes, score=True, bpm=bpm), MistakeBenchmarker.note_array(user_notes))
        return MistakeBenchmarker.validated_pairs(alignment, score_notes, user_notes)

    @staticmethod
    def validated_pairs(alignment, score_notes, user_notes):
        """All competitors must return complete, one-to-one note coverage."""
        pairs, seen_s, seen_u = ([], set(), set())
        for item in alignment:
            label = item['label']
            if label not in ('match', 'insertion', 'deletion'):
                raise ValueError(f'Unsupported external alignment label: {label}')
            si = int(item['score_id']) if label != 'insertion' else None
            ui = int(item['performance_id']) if label != 'deletion' else None
            for idx, seen, size in ((si, seen_s, len(score_notes)), (ui, seen_u, len(user_notes))):
                if idx is not None:
                    if idx in seen or not 0 <= idx < size:
                        raise ValueError(f'Duplicate/invalid alignment index: {item}')
                    seen.add(idx)
            pairs.append((user_notes[ui] if ui is not None else None, score_notes[si] if si is not None else None))
        if len(seen_s) != len(score_notes) or len(seen_u) != len(user_notes):
            raise ValueError('External alignment omitted input notes')
        return pairs

    @staticmethod
    def label_pairs(pairs, config):
        """Common decision thresholds; durations use each method's score timeline."""
        mistakes = []
        for user, score in pairs:
            if user is None:
                mistakes.append(Mistake('deletion', None, score))
            elif score is None:
                mistakes.append(Mistake('insertion', user, None))
            else:
                if min((abs(user.midi_num[0] - p) for p in score.midi_num)) >= config.pitch_tolerance:
                    mistakes.append(Mistake('substitution', user, score))
                delta = user.duration() - score.duration()
                if abs(delta) > config.timing_tolerance:
                    mistakes.append(Mistake('long' if delta > 0 else 'short', user, score))
        return mistakes

    @staticmethod
    def evaluate_case(midi, seed, rate, output, methods, tolerances, polytune=None, source_info=None, cached=None, on_result=None, defer_polytune=False, input_kinds=('detected', 'oracle_notes', 'audio'), polytune_identity=None, force_pitch_detection=False, laddersym=None, laddersym_identity=None):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
        needs_detection = 'detected' in input_kinds and any((('detected', m) not in (cached or {}) for m in methods if m not in COMPETITOR_AUDIO_METHODS))
        needs_audio = needs_detection or ('audio' in input_kinds and any((m in methods for m in COMPETITOR_AUDIO_METHODS)))
        bench, item = prepare_case(midi, seed, rate, output, source_info=source_info, prepare_audio=needs_audio, prepare_pitches=False)
        source = OneInstrumentScoreData(midi)
        performed = OneInstrumentScoreData(item['midi']).note_data
        config = bench.config_for_performance(performed)
        base = bench.attune.recording_for(config, score_data=source)
        pitch_cpu = note_cpu = pitch_execution_cpu = frontend_wall = 0.0
        if needs_detection:
            base.audio_data = AudioData(audio_filepath=item['audio'], config=base.config)
            frontend_wall_start, frontend_start = (perf_counter(), MistakeBenchmarker.cpu_seconds())
            pitch_timing = bench.load_mistake_pitches(base, item['pitch_data'], smooth=True, use_cache=not force_pitch_detection)
            pitch_execution_cpu = MistakeBenchmarker.cpu_seconds() - frontend_start
            base.reset_analysis()
            note_start = MistakeBenchmarker.cpu_seconds()
            base.detect_notes()
            note_cpu = MistakeBenchmarker.cpu_seconds() - note_start
            frontend_wall = perf_counter() - frontend_wall_start
            pitch_cpu = float(pitch_timing['pitch_compute_time'])
        truth_payload = json.loads(Path(item['truth']).read_text())
        injection_truth = truth_payload['truth']
        score_onsets = MistakeDetectorBase.timeline_score_onsets(truth_payload, performed)
        truth_reference = OneInstrumentScoreData(midi).note_data
        truth, truth_pairs = MistakeDetectorBase.net_mistakes(truth_reference, performed, pitch_tolerance=config.pitch_tolerance, duration_tolerance=config.timing_tolerance, score_onsets=score_onsets)
        if rate == 0 and truth:
            raise ValueError('Clean performance has net errors; inspect score/MIDI time conversion.')
        (Path(output) / 'cases' / item['case_id'] / 'net_truth.json').write_text(json.dumps(dict(injection_history=injection_truth, net_truth=truth, pairs=truth_pairs, onset_gate_sec=0.1, pitch_tolerance=config.pitch_tolerance, duration_tolerance=config.timing_tolerance, score_onsets=score_onsets, timeline_protocol=truth_payload.get('injector', {}).get('timeline_protocol', 'legacy_overlap')), indent=2))
        canonical_truth = MistakeDetectorBase.truth_events(truth, truth_reference, performed)
        cached = cached or {}
        rows = [row for batch in cached.values() for row in batch]

        def emit(batch):
            for row in batch:
                row.update({key: value for key, value in (source_info or {}).items() if key != 'source'})
                row['program'] = item['spec']['program']
            if on_result is not None:
                on_result(batch)
        for input_kind, initial in [('detected', base.note_data), ('oracle_notes', performed)]:
            if input_kind not in input_kinds:
                continue
            for method in methods:
                if method in COMPETITOR_AUDIO_METHODS:
                    continue
                if input_kind == 'oracle_notes' and method in ('Attune (repeat only)', 'Attune (Checker 3)'):
                    continue
                if (input_kind, method) in cached:
                    continue
                first_row = len(rows)
                rec = bench.attune.recording_for(replace(base.config), score_data=OneInstrumentScoreData(midi))
                rec.note_data = NoteDetectorBase.clone_note_data(initial)
                rec.pitch_data = PitchData(rec.config)
                rec.pitch_data.data = deepcopy(base.pitch_data.data, {id(base.config): rec.config})
                rec.pitch_data.t_origin = base.pitch_data.t_origin
                rec.pitch_data.end_index = base.pitch_data.end_index
                start, cpu_start = (perf_counter(), MistakeBenchmarker.cpu_seconds())
                if method.startswith('Attune'):
                    if rec.note_data.times:
                        rec.resize_score(to_span='onset')
                    rec.detect_mistakes()
                    if method != 'Attune (Checker 3)':
                        rec.refit_score_alignment_once()
                    if method == 'Attune (Checker 3)':
                        from notebooks.archive.MistakeChecker3 import MistakeChecker as LegacyRepeatSplitter
                        rec.repeat_splitter = LegacyRepeatSplitter(recording=rec)
                        rec.stabilize_score_alignment()
                    elif method == 'Attune (repeat only)':
                        rec.stabilize_score_alignment()
                    rec.reindex_mistakes()
                    pairs = rec.alignment.pairs
                else:
                    pairs = MistakeBenchmarker.external_pairs(method, [rec.score_data.note_data.data[t] for t in rec.score_data.note_data.times], [rec.note_data.data[t] for t in rec.note_data.times], rec.score_data.bpm)
                mistakes = MistakeBenchmarker.label_pairs(pairs, rec.config)
                seconds = perf_counter() - start
                alignment_cpu = MistakeBenchmarker.cpu_seconds() - cpu_start
                detected = input_kind == 'detected'
                mistakes = MistakeDetectorBase.with_reference_score_ids(mistakes, truth_reference, rec.score_data.note_data)
                for tolerance in tolerances:
                    canonical = bench.score_symbolic(mistakes, truth, onset_tolerance=tolerance, canonical=True)
                    historical = bench.score_onset(mistakes, truth, onset_tolerance=tolerance, canonical=False)
                    metrics = {kind: canonical.get(kind, (0, 0, 0)) for kind in COMPETITOR_TYPES}
                    metrics['pitch'] = tuple((sum((metrics[k][i] for k in ('deletion', 'insertion'))) for i in range(3)))
                    metrics['duration'] = tuple((sum((metrics[k][i] for k in ('short', 'long'))) for i in range(3)))
                    metrics['legacy_five_type'] = tuple((sum((historical.get(k, (0, 0, 0))[i] for k in COMPETITOR_TYPES)) for i in range(3)))
                    metrics.update(MistakeDetectorBase.score_events(MistakeDetectorBase.predicted_events(mistakes, truth_reference), canonical_truth, tolerance))
                    for metric, (tp, fp, fn) in metrics.items():
                        precision, recall, f1 = MistakeDetectorBase.prf(tp, fp, fn)
                        rows.append(dict(case_id=item['case_id'], source=str(midi), seed=seed, rate=rate, input=input_kind, method=method, tolerance=tolerance, metric=metric, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1 if tp + fp + fn else np.nan, seconds=seconds, cpu_seconds=alignment_cpu + (pitch_cpu + note_cpu if detected else 0.0), execution_cpu_seconds=alignment_cpu + (pitch_execution_cpu + note_cpu if detected else 0.0), frontend_cpu_seconds=pitch_cpu if detected else 0.0, segmentation_cpu_seconds=note_cpu if detected else 0.0, alignment_cpu_seconds=alignment_cpu, wall_seconds=seconds + (frontend_wall if detected else 0.0), compute_clock='process+reaped_children', timing_version='cpu_v1', worker_pid=os.getpid(), timing_scope='pitch + note + alignment' if detected else 'oracle alignment only', score_notes=len(source.note_data.times), initial_notes=len(initial.times), final_notes=len(rec.note_data.times), truth_events=len(truth), injection_events=len(injection_truth), config=json.dumps(asdict(base.config), sort_keys=True)))
                emit(rows[first_row:])
        requests = []
        pending_audio = [m for m in methods if m in COMPETITOR_AUDIO_METHODS and 'audio' in input_kinds and (('audio', m) not in cached)]
        if pending_audio:
            case_dir = Path(output) / 'cases' / item['case_id']
            clean_midi = case_dir / 'clean_score.mid'
            if not clean_midi.exists():
                bench.notedata_to_pm(truth_reference, program=item['spec']['program']).write(str(clean_midi))
            score_audio = bench.synth_midi(clean_midi, out_dir=case_dir / 'score_audio', force=False)
            for method in pending_audio:
                task = dict(audio=item['audio'], score_audio=str(score_audio), score_midi=str(clean_midi), directory=str(case_dir / method.lower()), truth=canonical_truth, tolerances=list(tolerances), common=dict(case_id=item['case_id'], source=str(midi), seed=seed, rate=rate, input='audio', method=method, score_notes=len(truth_reference.times), initial_notes=np.nan, final_notes=np.nan, truth_events=len(truth), injection_events=len(injection_truth), config=json.dumps(asdict(base.config), sort_keys=True), program=item['spec']['program'], **{k: v for k, v in (source_info or {}).items() if k != 'source'}))
                reused = None
                import benchmarks.modules.mistake.MistakeCache as _local_MistakeCache
                cached_prediction = _local_MistakeCache.MistakeCache.cached_prediction
                reused = cached_prediction(task, polytune_identity if method == 'PolyTune' else laddersym_identity, method=method)
                if reused is not None:
                    payload = reused
                elif defer_polytune:
                    requests.append(task)
                    continue
                elif method == 'LadderSym':
                    payload = (laddersym or LadderSym()).predict(task['audio'], task['score_audio'], task['directory'], score_midi=task['score_midi'])
                else:
                    payload = (polytune or PolyTune()).predict(task['audio'], task['score_audio'], task['directory'])
                batch = MistakeBenchmarker.polytune_rows(task, payload)
                rows.extend(batch)
                emit(batch)
        if defer_polytune:
            return (rows, requests if len(requests) > 1 else requests[0] if requests else None)
        return rows

    @staticmethod
    def polytune_rows(task, payload):
        """Shared audio-event scoring for both models (historical public name)."""
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
        rows = []
        timing_keys = ('cpu_seconds', 'execution_cpu_seconds', 'setup_cpu_seconds', 'model_load_cpu_seconds', 'model_load_wall_seconds', 'model_reused', 'worker_pid', 'compute_clock', 'timing_version', 'inference_cache_hit', 'inference_cache_source')
        for tolerance in task['tolerances']:
            for metric, (tp, fp, fn) in MistakeDetectorBase.score_events(payload['events'], task['truth'], tolerance).items():
                precision, recall, f1 = MistakeDetectorBase.prf(tp, fp, fn)
                rows.append(dict(**task['common'], tolerance=tolerance, metric=metric, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1 if tp + fp + fn else np.nan, seconds=payload['inference_seconds'], setup_seconds=payload['setup_seconds'], wall_seconds=payload['wall_seconds'], timing_scope='audio loading + neural inference + event decoding; model setup separate', **{k: payload[k] for k in timing_keys if k in payload}))
        return rows

    @staticmethod
    def summarize(rows):
        result = []
        keys = ['input', 'rate', 'method', 'tolerance', 'metric']
        for values, group in rows.groupby(keys):
            tp, fp, fn = (int(group[s].sum()) for s in ('tp', 'fp', 'fn'))
            precision, recall, f1 = MistakeDetectorBase.prf(tp, fp, fn)
            result.append(dict(zip(keys, values), cases=len(group), tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1 if tp + fp + fn else np.nan, macro_f1=group.f1.mean(), fp_per_100_score_notes=100 * fp / group.score_notes.sum(), mean_seconds=group.seconds.mean(), **{f'mean_{k}': group[k].mean() for k in ('cpu_seconds', 'execution_cpu_seconds', 'frontend_cpu_seconds', 'segmentation_cpu_seconds', 'alignment_cpu_seconds', 'setup_cpu_seconds', 'wall_seconds') if k in group}, cpu_timed_cases=int(group.cpu_seconds.notna().sum()) if 'cpu_seconds' in group else 0))
        return pd.DataFrame(result)

    @staticmethod
    def run_comparison(midis, output, *, seeds=(0,), rates=(0.0, 0.25), methods=COMPETITOR_METHODS, tolerances=(0.05, 0.1, 0.2), polytune=None, source_metadata=None, workers=None, neural_workers=None, parallel=True, input_kinds=('detected', 'oracle_notes', 'audio'), force_pitch_detection=False, polytune_last=False, laddersym=None):
        from benchmarks.modules.mistake.MistakeCache import MistakeCache
        input_kinds = tuple(input_kinds)
        if not input_kinds or len(set(input_kinds)) != len(input_kinds) or set(input_kinds) - {'detected', 'oracle_notes', 'audio'}:
            raise ValueError('Select distinct input kinds: detected, oracle_notes, audio')
        selected_units = [unit for unit in MistakeCache.units(methods) if unit[0] in input_kinds]
        if set(methods) != {method for _, method in selected_units}:
            raise ValueError('Every selected method must support at least one selected input kind')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / 'run.log'
        with log_path.open('a') as log, redirect_stdout(log), redirect_stderr(log):
            print('Starting/resuming mistake comparison', flush=True)
            try:
                packages = MistakeBenchmarker.preflight(methods, polytune=polytune, laddersym=laddersym)
            except Exception as exc:
                traceback.print_exc()
                exc.add_note(f'Benchmark details: {log_path}')
                raise
        midis = list(dict.fromkeys((str(Path(m).resolve()) for m in midis)))
        source_metadata = {str(Path(p).resolve()): info for p, info in (source_metadata or {}).items()}
        seeds, rates, tolerances = (list(seeds), list(rates), list(tolerances))
        if not midis or not seeds or (not rates) or (not tolerances):
            raise ValueError('Select at least one source, seed, rate and tolerance.')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        code = MistakeCache.pipeline_fingerprint()
        for p in ('benchmarks/modules/mistake/MistakeBenchmarker.py', 'algorithms/RepeatSplitter.py', 'benchmarks/modules/mistake/MistakeBenchmarker.py', 'benchmarks/modules/mistake/MistakeDetectorBase.py', 'benchmarks/modules/mistake/competitors/Nakamura.py', 'benchmarks/modules/mistake/datasets/MistakeCases.py', 'benchmarks/modules/mistake/MistakeDetectorBase.py', 'benchmarks/modules/mistake/competitors/PolyTune.py', 'benchmarks/modules/mistake/competitors/PolyTune.py', 'benchmarks/modules/mistake/competitors/PolyTune.py', 'benchmarks/modules/mistake/competitors/LadderSym.py', 'benchmarks/modules/mistake/competitors/LadderSym.py'):
            code[p] = hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()
        metadata = dict(sources=midis, source_metadata=source_metadata, seeds=seeds, rates=rates, methods=list(methods), tolerances=tolerances, packages=packages, code=code, input_kinds=list(input_kinds), force_pitch_detection=bool(force_pitch_detection), ladder_contiguous_inference=bool(getattr(laddersym, 'contiguous_inference', False)), score_fit_protocol='matched_onsets_once_v1', scoring='primary audio_pitch: canonical missed/extra with one-to-one onset+pitch matching; pitch: legacy ID diagnostic; duration separate', pipeline='aligners: shared pitches/fresh unmodified notes, no truth boundary correction; PolyTune/LadderSym: raw shared performance audio + clean score audio; LadderSym also uses clean score MIDI prompt; oracle input skips refinement', timeline_protocol='monophonic_edits_v3', checkpoint_schema=1, status='running')
        metadata['execution_code'] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ('MistakeBenchmarker.py', 'MistakeCache.py')}
        run_path = output / 'run.json'
        previous = json.loads(run_path.read_text()) if run_path.exists() else None
        checkpoints = MistakeCache(output, metadata, previous)
        jobs = {}
        recovered = {}
        for midi in midis:
            for seed in seeds:
                for rate in rates:
                    key = (midi, seed, rate)
                    jobs[key] = {unit: checkpoints.job(midi, seed, rate, *unit) for unit in selected_units}
                    recovered[key] = {unit: saved for unit, job in jobs[key].items() if not (force_pitch_detection and unit[0] == 'detected') if (saved := checkpoints.load(job)) is not None}
        MistakeCache.atomic_json(run_path, metadata)
        rows = [row for cached in recovered.values() for batch in cached.values() for row in batch]

        def save_reports():
            if not rows:
                return
            frame = pd.DataFrame(rows)
            with log_path.open('a') as log, redirect_stdout(log), redirect_stderr(log):
                for name, table in [('rows.csv', frame), ('summary.csv', MistakeBenchmarker.summarize(frame))]:
                    temporary = output / (name + '.tmp')
                    table.to_csv(temporary, index=False)
                    temporary.replace(output / name)
        save_reports()
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        PitchBenchmarker = PitchBenchmarker
        progress = PitchBenchmarker.Progress(enabled=True, compact=True)
        total = len(midis) * len(seeds) * len(rates)
        try:
            if parallel:
                import benchmarks.modules.mistake.MistakeBenchmarker as _local_MistakeBenchmarker
                worker_limits = _local_MistakeBenchmarker.MistakeBenchmarker.worker_limits
                import benchmarks.modules.mistake.MistakeBenchmarker as _local_MistakeBenchmarker
                run_parallel = _local_MistakeBenchmarker.MistakeBenchmarker.run_parallel
                cpu_workers, model_workers = worker_limits(workers, neural_workers, neural=any((m in methods for m in COMPETITOR_AUDIO_METHODS)))
                tasks = []
                total = sum((len(case_jobs) for case_jobs in jobs.values()))
                if rows:
                    progress.update_compact(total, 'cached', 'completed evaluations', count=sum((len(c) for c in recovered.values())))
                for key, case_jobs in jobs.items():
                    if len(recovered[key]) == len(case_jobs):
                        continue
                    midi, seed, rate = key
                    task_key = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
                    tasks.append(dict(key=key, args=(midi, seed, rate, output, methods, tolerances), cached=recovered[key], source_info=source_metadata.get(midi), input_kinds=input_kinds, polytune_identity=packages.get('PolyTune'), laddersym_identity=None if getattr(laddersym, 'contiguous_inference', False) else packages.get('LadderSym'), force_pitch_detection=force_pitch_detection, jobs={unit: (job, str(checkpoints.path(job))) for unit, job in case_jobs.items()}, log=str(output / 'logs' / f'{task_key}.log')))
                cpu_workers = min(cpu_workers, max(1, len(tasks)))
                pending_models = sum((('audio', m) in t['jobs'] and ('audio', m) not in t['cached'] for t in tasks for m in COMPETITOR_AUDIO_METHODS))
                model_workers = min(model_workers, pending_models)
                if polytune_last and neural_workers is None and pending_models:
                    model_workers = None
                metadata['parallelism'] = dict(cpu_workers=cpu_workers, neural_workers=model_workers, numerical_threads=1, persistent_models=True, schedule='cpu_then_audio_models' if polytune_last else 'overlap')
                MistakeCache.atomic_json(run_path, metadata)

                def receive(key, batch):
                    unit = (batch[0]['input'], batch[0]['method'])
                    if unit in recovered[key]:
                        return
                    checkpoints.save(jobs[key][unit], batch)
                    recovered[key][unit] = batch
                    rows.extend(batch)
                    save_reports()
                    label = f'{unit[1]} (cached)' if batch[0].get('inference_cache_hit') else unit[1]
                    progress.update_compact(total, label, f'{Path(key[0]).stem} seed={key[1]} {unit[0]}')
                if tasks:

                    def neural_started(count):
                        from benchmarks.modules.mistake.MistakeCache import MistakeCache
                        metadata['parallelism']['neural_workers'] = count
                        MistakeCache.atomic_json(run_path, metadata)
                        progress.update_compact(total, 'audio models', f'{count} persistent model workers', count=0)
                    joiner = ' then ' if polytune_last else ' + '
                    model_label = model_workers if model_workers is not None else 'auto-sized'
                    progress.update_compact(total, 'starting', f'{cpu_workers} CPU{joiner}{model_label} model workers', count=0)
                    actual_models = run_parallel(tasks, output=output, polytune=polytune or PolyTune(), workers=cpu_workers, neural_workers=model_workers, on_result=receive, polytune_last=polytune_last, on_neural_start=neural_started, laddersym=laddersym or LadderSym())
                    metadata['parallelism']['neural_workers'] = actual_models
                    MistakeCache.atomic_json(run_path, metadata)
                if any((len(recovered[key]) != len(jobs[key]) for key in jobs)):
                    raise ValueError('Parallel run did not checkpoint every selected evaluation')
            else:
                for midi in midis:
                    for seed in seeds:
                        for rate in rates:
                            track = f'{Path(midi).stem} seed={seed} rate={rate:g}'
                            cached = recovered[midi, seed, rate]
                            if len(cached) == len(jobs[midi, seed, rate]):
                                progress.update_compact(total, 'cached', track)
                                continue
                            progress.update_compact(total, 'mistakes', track, count=0)

                            def on_result(batch):
                                unit = (batch[0]['input'], batch[0]['method'])
                                checkpoints.save(jobs[midi, seed, rate][unit], batch)
                                cached[unit] = batch
                                rows.extend(batch)
                                save_reports()
                            with log_path.open('a') as log, redirect_stdout(log), redirect_stderr(log):
                                print(f'Case: {midi}, seed={seed}, rate={rate}', flush=True)
                                MistakeBenchmarker.evaluate_case(midi, seed, rate, output, methods, tolerances, polytune=polytune, laddersym=laddersym, source_info=source_metadata.get(midi), cached=cached, on_result=on_result, input_kinds=input_kinds, polytune_identity=packages.get('PolyTune'), laddersym_identity=None if getattr(laddersym, 'contiguous_inference', False) else packages.get('LadderSym'), force_pitch_detection=force_pitch_detection)
                            if len(cached) != len(jobs[midi, seed, rate]):
                                raise ValueError('Case ended without checkpointing every selected evaluation')
                            progress.update_compact(total, 'mistakes', track)
            frame = pd.DataFrame(rows)
            save_reports()
            metadata['status'] = 'complete'
        except BaseException as exc:
            metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
            with log_path.open('a') as log:
                traceback.print_exc(file=log)
            exc.add_note(f'Benchmark details: {log_path}')
            raise
        finally:
            progress.finish()
            MistakeCache.atomic_json(run_path, metadata)
        return frame

    @staticmethod
    def run_symbolic_comparison(midis, output, **kwargs):
        """Run weighted Attune string editing and external aligners on exact MIDI notes.

        No audio rendering, extraction or acoustic refinement. Uses the same injector,
        net truth and event metric as the detected-note comparison.
        """
        kwargs.setdefault('methods', COMPETITOR_SYMBOLIC_METHODS)
        return MistakeBenchmarker.run_comparison(midis, output, input_kinds=('oracle_notes',), **kwargs)

    @staticmethod
    def native_midi_events(path, kind):
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(str(path))
        return [dict(kind=kind, onset=float(n.start), end=float(n.end), pitch=int(n.pitch), program=int(i.program)) for i in midi.instruments for n in i.notes if not i.is_drum]

    @staticmethod
    def native_event_metrics(predicted, truth, tolerances=(0.05, 0.1, 0.2)):
        """Common pooled errors plus native-style 50 ms class-macro onset scores.

        mir_eval's onset-only convention (50 cents, no offsets) is used for the
        native-style rows. Empty classes score zero, consistent with mir_eval and
        the upstream Coco evaluator. Semantic labels survive empty MIDI tracks.
        """
        import mir_eval
        import numpy as np
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
        score_events = MistakeDetectorBase.score_events
        rows = []
        for gate in tolerances:
            for metric, (tp, fp, fn) in score_events(predicted, truth, gate).items():
                rows.append(dict(protocol='common_pooled_events', metric=metric, tolerance=gate, tp=tp, fp=fp, fn=fn, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None))
        scores = []
        for kind in ('extra', 'missed', 'correct'):
            ref = [e for e in truth if e['kind'] == kind]
            est = [e for e in predicted if e['kind'] == kind]

            def arrays(events):
                intervals = np.array([[e['onset'], e['end']] for e in events], dtype=float).reshape(-1, 2)
                pitches = np.array([440 * 2 ** ((e['pitch'] - 69) / 12) for e in events])
                return (intervals, pitches)
            if not ref or not est:
                precision = recall = f1 = 0.0
            else:
                precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(*arrays(ref), *arrays(est), onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
            scores.append(f1)
            rows.append(dict(protocol='native_style_macro', metric=kind, tolerance=0.05, precision=precision, recall=recall, f1=f1))
        rows.append(dict(protocol='native_style_macro', metric='three_class_average', tolerance=0.05, f1=sum(scores) / 3))
        return rows

    @staticmethod
    def native_performed_range(files):
        """Oracle-assisted pitch limits only; omitted notes are not performed."""
        from algorithms.Config import Config
        pitches = [e['pitch'] for kind in ('correct', 'extra') for e in MistakeBenchmarker.native_midi_events(files[kind], kind)]
        if not pitches:
            raise ValueError('Cannot derive a performed-note range from empty labels')
        return Config.padded_midi_range(min(pitches), max(pitches))

    @staticmethod
    def native_attune_audio(performance, config):
        """Preserve 4096/44100 s windows and 128/44100 s hops without altering source WAVs."""
        from math import gcd
        from scipy.signal import resample_poly
        from app_logic.user.ds.AudioData import AudioData
        audio = AudioData(audio_filepath=str(performance), config=config)
        source_sr = audio.sr
        if source_sr != config.sr:
            divisor = gcd(source_sr, config.sr)
            audio.data = resample_poly(audio.read_all(), config.sr // divisor, source_sr // divisor)
            audio.sr = config.sr
            audio.capacity = audio.end_index = len(audio.data)
        return (audio, source_sr)

    @staticmethod
    def native_attune_events(performance, score_midi, *, midi_range):
        """Keep production chord behavior and the calibrated analysis time grid."""
        from benchmarks.modules.mistake.MistakeCache import MistakeCache
        from app_logic.midi.ScoreData import ScoreData
        from benchmarks.modules.pitch.competitors.Attune import Attune
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
        with_reference_score_ids = MistakeDetectorBase.with_reference_score_ids
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        label_pairs = MistakeBenchmarker.label_pairs
        adapter = Attune()
        config = adapter.config_for(*adapter.range_from_midi(midi_range), sr=44100, pitch_tolerance=0.5, min_note_length_cap=0.1)
        score = ScoreData(score_midi)
        channels = [c for c, nd in score.note_datas.items() if c != score.metronome_channel and nd.times]
        if len(channels) != 1:
            raise ValueError('Native Attune requires a single instrument MIDI (chords are retained)')
        score.active_instrument = channels[0]
        reference = deepcopy(score.clipped_note_data())
        rec = adapter.recording_for(config, score_data=score)
        rec.audio_data, source_sr = MistakeBenchmarker.native_attune_audio(performance, rec.config)
        rec.config.sr = rec.audio_data.sr
        pitch_cache = MistakeCache.cached_native_pitches(rec, performance)
        note_cache = MistakeCache.cached_native_notes(rec, pitch_cache, score_midi)
        rec.align_score_and_refine()
        mistakes = with_reference_score_ids(label_pairs(rec.alignment.pairs, config), reference, score.clipped_note_data())
        events = MistakeBenchmarker.native_native_attune_events(mistakes, reference, rec.alignment.pairs)
        return dict(events=events, config=asdict(rec.config), pitch_cache=pitch_cache, note_cache=note_cache, frontend=dict(source_sr=source_sr, analysis_sr=rec.config.sr, window_seconds=rec.config.w1 / rec.config.sr, hop_seconds=rec.config.h1 / rec.config.sr, midi_range=list(midi_range), minimum_segment_seconds=rec.config.note_detection_min_seconds()), timestamp_policy='substitution missed+extra use detected replacement time; pure deletions use original score time', alignment_policy='original-note alignment, robust matched-onset refit, local missing-repeat recovery; no truth timing input', note='Production chord acceptance is unchanged: matching one chord pitch does not invent missed-member flags.')

    @staticmethod
    def native_native_attune_events(mistakes, reference, pairs):
        """Export native label semantics without reading ground-truth timestamps."""
        original = reference.notes_by_id()
        events = []
        wrong_users = set()
        for mistake in mistakes:
            if mistake.type in ('deletion', 'substitution'):
                note = original[mistake.midi_note.id]
                timing = mistake.user_note if mistake.type == 'substitution' else note
                for pitch in note.midi_num:
                    events.append(dict(kind='missed', onset=timing.start_time, end=timing.end_time, pitch=pitch))
            if mistake.type in ('insertion', 'substitution'):
                note = mistake.user_note
                wrong_users.add(id(note))
                events.append(dict(kind='extra', onset=note.start_time, end=note.end_time, pitch=note.midi_num[0]))
        for user, target in pairs:
            if user is not None and target is not None and (id(user) not in wrong_users):
                events.append(dict(kind='correct', onset=user.start_time, end=user.end_time, pitch=user.midi_num[0]))
        events = [dict(kind=e['kind'], onset=float(e['onset']), end=float(e['end']), pitch=float(e['pitch'])) for e in events]
        return events

    @staticmethod
    def native_models_for(dataset, device):
        from benchmarks.modules.mistake.competitors.PolyTune import PolyTune
        from benchmarks.modules.mistake.competitors.LadderSym import LadderSym
        if dataset != 'coco':
            raise ValueError('Only native CocoChorales-E instrument stems are supported')
        return {'PolyTune': PolyTune(device=device), 'LadderSym': LadderSym(device=device, contiguous_inference=True)}

    @staticmethod
    def native_code_identity():
        import ast
        import hashlib
        from benchmarks.paths import REPO_ROOT
        files = set()
        for folder in ('algorithms', 'app_logic', 'benchmarks/modules/mistake', 'benchmarks/modules/pitch'):
            files.update((REPO_ROOT / folder).rglob('*.py'))
        identity = {str(p.relative_to(REPO_ROOT)): digest(p) for p in sorted(files) if 'tests' not in p.parts and (not p.name.startswith('test_')) and (p.name != 'MistakeNotebook.py')}
        functions = ('midi_events', 'event_metrics', 'performed_range', 'attune_audio', 'cached_native_pitches', 'cached_native_notes', 'attune_events', 'native_attune_events', 'models_for', 'summarize')
        source = Path(__file__).read_text()
        semantics = '\n'.join((ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name in {'native_' + name for name in functions}))
        identity['NativeComparison.inference'] = hashlib.sha256(semantics.encode()).hexdigest()
        return identity

    @staticmethod
    def native_summarize(rows):
        import pandas as pd
        frame = pd.DataFrame(rows)
        common = frame[frame.protocol == 'common_pooled_events']
        keys = ['dataset', 'method', 'protocol', 'metric', 'tolerance']
        pooled = common.groupby(keys, as_index=False)[['tp', 'fp', 'fn']].sum()
        denominator = 2 * pooled.tp + pooled.fp + pooled.fn
        pooled['f1'] = 2 * pooled.tp / denominator.where(denominator != 0)
        native = frame[frame.protocol == 'native_style_macro'].groupby(keys, as_index=False).f1.mean()
        return pd.concat([pooled, native], ignore_index=True)

    @staticmethod
    def native_run(manifest_path, output, *, methods=NATIVE_METHODS, device='cpu', midi_range=(21, 108), tolerances=(0.05, 0.1, 0.2), workers=None, neural_workers=None, range_policy='fixed', reuse_audio_from=None):
        from benchmarks.modules.mistake.MistakeCache import MistakeCache
        from algorithms.Config import Config
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        run_tasks = MistakeBenchmarker.native_run_tasks
        import warnings
        warnings.filterwarnings('ignore', message='(Reference|Estimated) notes are empty\\.', module='mir_eval.transcription')
        import pandas as pd
        if range_policy not in ('fixed', 'performed_notes'):
            raise ValueError('range_policy must be fixed or performed_notes')
        if not methods or len(set(methods)) != len(methods) or set(methods) - set(NATIVE_METHODS):
            raise ValueError('Select unique native methods from Attune, PolyTune, LadderSym')
        if workers is not None and workers < 1 or (neural_workers is not None and neural_workers < 1):
            raise ValueError('Worker counts must be positive')
        if len(midi_range) != 2 or not 0 <= midi_range[0] < midi_range[1] <= 127:
            raise ValueError('midi_range must be fixed increasing MIDI limits')
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest['split'] != 'test' or not manifest['cases']:
            raise ValueError('A nonempty official test manifest is required')
        dataset = manifest['dataset']
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        from concurrent.futures import ThreadPoolExecutor
        expected_hashes = {}
        for case in manifest['cases']:
            for key, path in case['files'].items():
                expected = case['hashes'][key]
                if path in expected_hashes and expected_hashes[path] != expected:
                    raise ValueError(f'Conflicting native asset hashes: {path}')
                expected_hashes[path] = expected
        models = MistakeBenchmarker.native_models_for(dataset, device)

        def validate_asset(item):
            path, expected = item
            if digest(path) != expected:
                raise ValueError(f'Native asset changed: {path}')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(validate_asset, expected_hashes.items()))
            selected_models = [m for m in methods if m in models]
            identities = dict(zip(selected_models, pool.map(lambda m: models[m].preflight(), selected_models)))
        contract = dict(version=3, attune_analysis_sr=44100, attune_overrides=dict(pitch_tolerance=0.5, min_note_length_cap=0.1), attune_timestamps='substitutions: detected replacement; deletions: original score', range_policy=range_policy, manifest=manifest, methods=list(methods), device=device, midi_range=list(midi_range), tolerances=list(tolerances), models=identities, code=MistakeBenchmarker.native_code_identity(), defaults=asdict(Config()), packages={p: version(p) for p in ('numpy', 'scipy', 'pretty_midi', 'mir_eval', 'music21', 'librosa')}, native_metrics='50ms onset-only mir_eval, semantic classes, per-case macro; not literal upstream MIDI-track evaluator', input_policy='authored audio; Attune resampled to 44100 Hz; ' + ('oracle-assisted correct+extra pitch extrema, padded by 4 semitones' if range_policy == 'performed_notes' else 'fixed pitch range'), attune_scope='CocoChorales-E instrument stems; monophonic frontend', ladder_contiguous_inference=True)
        metadata_path = output / 'run.json'
        if metadata_path.exists():
            prior = json.loads(metadata_path.read_text())
            if not MistakeCache.compatible_contract(prior['contract'], contract):
                raise ValueError('Native run inputs/code/settings changed. Choose a new output directory; historical results are preserved.')
        metadata = dict(status='running', contract=contract, reuse_audio_from=str(Path(reuse_audio_from).resolve()) if reuse_audio_from else None, execution_code={name: digest(Path(__file__).with_name(name)) for name in ('MistakeBenchmarker.py', 'MistakeNotebook.py', 'MistakeCache.py')}, requested_parallelism=dict(workers=workers, neural_workers=neural_workers))
        if metadata_path.exists():
            metadata['original_contract'] = prior.get('original_contract', prior['contract'])
        save_json(metadata_path, metadata)
        rows = []

        def report():
            for name, table in [('rows.csv', pd.DataFrame(rows)), ('summary.csv', MistakeBenchmarker.native_summarize(rows))]:
                temporary = output / (name + '.tmp')
                table.to_csv(temporary, index=False)
                temporary.replace(output / name)
        tasks = []
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        PitchBenchmarker = PitchBenchmarker
        progress = PitchBenchmarker.Progress(enabled=True, compact=True)
        total = len(manifest['cases']) * len(methods)

        def receive(payload, *, cached=False):
            rows.extend(payload['rows'])
            report()
            row = payload['rows'][0]
            method = row['method']
            input_kind = 'detected' if method == 'Attune' else 'audio'
            track = f'{row['case_id']} {input_kind}'
            progress.update_compact(total, 'cached' if cached else method, f'{method} {track}' if cached else track)

        def stage_started(method, count):
            detail = f'{count} CPU workers' if method == 'Attune' else f'{count} persistent model workers'
            progress.update_compact(total, method, detail, count=0)
        progress.update_compact(total, 'starting', f'{workers or 'auto-sized'} CPU then {neural_workers or 'auto-sized'} model workers', count=0)
        try:
            for method in methods:
                for case in manifest['cases']:
                    directory = output / 'cases' / case['case_id'] / method.lower()
                    checkpoint = directory / 'result.json'
                    if checkpoint.exists():
                        receive(json.loads(checkpoint.read_text()), cached=True)
                    else:
                        task = dict(case=case, method=method, dataset=dataset, directory=str(directory), midi_range=midi_range, tolerances=tolerances, range_policy=range_policy)
                        prediction = MistakeCache.reusable_audio_prediction(reuse_audio_from, contract, case, method)
                        if prediction is not None:
                            import benchmarks.modules.mistake.MistakeBenchmarker as _local_MistakeBenchmarker
                            evaluate = MistakeBenchmarker.native_evaluate
                            receive(evaluate(task, reused_prediction=prediction), cached=True)
                        else:
                            tasks.append(task)
            metadata['parallelism'] = run_tasks(tasks, output=output, models=models, workers=workers, neural_workers=neural_workers, on_result=receive, on_stage=stage_started)
            metadata['status'] = 'complete'
            save_json(metadata_path, metadata)
        finally:
            progress.finish()
        return pd.DataFrame(rows)

    @staticmethod
    def worker_limits(workers=None, neural_workers=None, *, neural=True):
        import psutil
        cores = max(1, (os.cpu_count() or 4) - 1)
        budget_gb = max(1.0, psutil.virtual_memory().available / 2 ** 30 - 0.75)
        models = min(3, max(1, int(budget_gb / 2.5)), max(1, cores - 1)) if neural else 0
        models = int(neural_workers) if neural and neural_workers is not None else models
        cpus = min(max(1, cores - models), max(1, int((budget_gb - models * 1.6) / 0.45)))
        cpus = int(workers) if workers is not None else cpus
        if cpus < 1 or (neural and models < 1):
            raise ValueError('Worker counts must be positive')
        return (cpus, models)

    @staticmethod
    def staged_neural_limit(pending, *, device='cpu'):
        """Size the model pool using RAM available after CPU workers have exited.

        CPU checkpoint tensors are read-only memory maps shared by the OS. Budget
        1.6 GiB per interpreter for imports, activations and decoding, keeping .75
        GiB free. This is an estimate; explicit neural_workers overrides it.
        GPU memory needs a separate budget, so accelerators default to one model.
        """
        if device != 'cpu':
            return min(pending, 1)
        import psutil
        budget_gb = max(0.0, psutil.virtual_memory().available / 2 ** 30 - 0.75)
        return min(pending, max(1, (os.cpu_count() or 4) - 1), max(1, int(budget_gb / 1.6)))

    @staticmethod
    @contextmanager
    def single_thread_environment():
        original = {name: os.environ.get(name) for name in THREAD_ENV}
        os.environ.update({name: '1' for name in THREAD_ENV})
        try:
            yield
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    @staticmethod
    def _init_cpu(events):
        global _parallel_EVENTS, _parallel_LIMITER
        from threadpoolctl import threadpool_limits
        _parallel_EVENTS = events
        _parallel_LIMITER = threadpool_limits(limits=1)

    @staticmethod
    def _case_worker(task):
        from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        evaluate_case = MistakeBenchmarker.evaluate_case
        path = Path(task['log'])
        path.parent.mkdir(parents=True, exist_ok=True)

        def emit(batch):
            from benchmarks.modules.mistake.MistakeCache import MistakeCache
            unit = (batch[0]['input'], batch[0]['method'])
            job, checkpoint = task['jobs'][unit]
            MistakeCache.atomic_json(checkpoint, dict(job=job, rows=batch))
            _parallel_EVENTS.put((task['key'], batch))
        with NoteEvaluation._worker_log(path):
            try:
                return evaluate_case(*task['args'], cached=task['cached'], on_result=emit, source_info=task['source_info'], defer_polytune=True, input_kinds=task.get('input_kinds', ('detected', 'oracle_notes', 'audio')), polytune_identity=task.get('polytune_identity'), laddersym_identity=task.get('laddersym_identity'), force_pitch_detection=task.get('force_pitch_detection', False))
            except BaseException:
                traceback.print_exc()
                raise

    class NeuralPool:

        def __init__(self, config, output, workers, laddersym=None):
            self.config, self.output = (config, Path(output))
            self.laddersym = laddersym
            self.local = threading.local()
            self.clients = []
            self.lock = threading.Lock()
            self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='audio_model')

        def _client(self, method='PolyTune'):
            from benchmarks.modules.mistake.competitors.PolyTune import PolyTune
            from benchmarks.modules.mistake.competitors.LadderSym import LadderSym
            if getattr(self.local, 'method', None) != method:
                if hasattr(self.local, 'client'):
                    old = self.local.client
                    old.close()
                    with self.lock:
                        self.clients.remove(old)
                config = self.laddersym or LadderSym() if method == 'LadderSym' else self.config
                factory = LadderSym.Client if method == 'LadderSym' else PolyTune.Client
                client = factory(config, self.output / 'models' / method.lower() / threading.current_thread().name)
                self.local.client, self.local.method = (client, method)
                with self.lock:
                    self.clients.append(client)
            return self.local.client

        def warm(self, workers):
            return [self.pool.submit(self._client) for _ in range(workers)]

        def execute(self, task, job, checkpoint):
            from benchmarks.modules.mistake.MistakeCache import MistakeCache
            from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
            polytune_rows = MistakeBenchmarker.polytune_rows
            method = task['common']['method']
            client = self._client(method)
            prompt = {'score_midi': task['score_midi']} if method == 'LadderSym' else {}
            payload = client.predict(task['audio'], task['score_audio'], task['directory'], **prompt)
            batch = polytune_rows(task, payload)
            MistakeCache.atomic_json(checkpoint, dict(job=job, rows=batch))
            return batch

        def submit(self, *args):
            return self.pool.submit(self.execute, *args)

        def close(self):
            self.pool.shutdown(wait=True, cancel_futures=True)
            for client in self.clients:
                client.close()

    @staticmethod
    def run_parallel(tasks, *, output, polytune, workers, neural_workers, on_result, polytune_last=False, on_neural_start=None, laddersym=None):
        """Optionally release all CPU processes before starting persistent models.

        Only the parent writes aggregate CSVs/progress; workers own unique logs.
        Deferred requests contain file paths/metadata, not loaded audio or models.
        """
        context = mp.get_context('spawn')
        events = context.Queue()
        neural = MistakeBenchmarker.NeuralPool(polytune, output, neural_workers, laddersym=laddersym) if neural_workers and (not polytune_last) else None
        actual_neural_workers = neural_workers if neural is not None else 0
        cpu = None
        futures = {}
        deferred = []

        def drain():
            while True:
                try:
                    key, batch = events.get_nowait()
                except queue.Empty:
                    return
                on_result(key, batch)
        try:
            with MistakeBenchmarker.single_thread_environment():
                cpu = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=MistakeBenchmarker._init_cpu, initargs=(events,))
                for task in tasks:
                    futures[cpu.submit(MistakeBenchmarker._case_worker, task)] = ('cpu', task)
                while futures:
                    ready, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                    drain()
                    for future in ready:
                        kind, task = futures.pop(future)
                        result = future.result()
                        if kind == 'cpu':
                            batches, request = result
                            for unit in task['jobs']:
                                batch = [r for r in batches if (r['input'], r['method']) == unit]
                                if batch:
                                    on_result(task['key'], batch)
                            requests = request if isinstance(request, list) else [request] if request is not None else []
                            for request in requests:
                                if neural_workers == 0 or (neural is None and (not polytune_last)):
                                    raise ValueError('Audio model requests require a neural worker')
                                job, checkpoint = task['jobs']['audio', request['common']['method']]
                                if polytune_last:
                                    deferred.append((task, request, job, checkpoint))
                                else:
                                    futures[neural.submit(request, job, checkpoint)] = ('neural', task)
                        else:
                            on_result(task['key'], result)
                    if not futures and deferred:
                        cpu.shutdown(wait=True)
                        cpu = None
                        drain()
                        if neural_workers is None:
                            actual_neural_workers = MistakeBenchmarker.staged_neural_limit(len(deferred), device='accelerator' if any((getattr(c, 'device', 'cpu') != 'cpu' for c in (polytune, laddersym))) else 'cpu')
                        else:
                            actual_neural_workers = min(len(deferred), neural_workers)
                        if on_neural_start is not None:
                            on_neural_start(actual_neural_workers)
                        neural = MistakeBenchmarker.NeuralPool(polytune, output, actual_neural_workers, laddersym=laddersym)
                        deferred.sort(key=lambda entry: entry[1]['common']['method'])
                        for task, request, job, checkpoint in deferred:
                            futures[neural.submit(request, job, checkpoint)] = ('neural', task)
                        deferred.clear()
                drain()
        finally:
            for future in futures:
                future.cancel()
            if cpu is not None:
                cpu.shutdown(wait=False, cancel_futures=True)
                pending = [f for f, (kind, _) in futures.items() if kind == 'cpu' and (not f.cancelled())]
                while pending:
                    wait(pending, timeout=0.1)
                    drain()
                    pending = [f for f in pending if not f.done()]
                cpu.shutdown(wait=True, cancel_futures=True)
            if neural is not None:
                neural.close()
            drain()
            for future, (kind, task) in futures.items():
                if kind == 'neural' and future.done() and (not future.cancelled()) and (future.exception() is None):
                    on_result(task['key'], future.result())
            events.close()
            events.join_thread()
        return actual_neural_workers

    @staticmethod
    def native_evaluate(task, client=None, *, reused_prediction=None):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        attune_events = MistakeBenchmarker.native_attune_events
        midi_events = MistakeBenchmarker.native_midi_events
        event_metrics = MistakeBenchmarker.native_event_metrics
        performed_range = MistakeBenchmarker.native_performed_range
        from benchmarks.modules.mistake.datasets.NativeDatasets import save_json
        from time import perf_counter
        case, method = (task['case'], task['method'])
        directory = Path(task['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        start = perf_counter()
        if reused_prediction is not None:
            if method == 'Attune':
                raise ValueError('Attune predictions cannot be reused across frontend changes')
            prediction = reused_prediction
        elif client is None:
            prediction = attune_events(case['files']['performance'], case['files']['score_midi'], midi_range=performed_range(case['files']) if task.get('range_policy') == 'performed_notes' else task['midi_range'])
            prediction['frontend'] = dict(prediction.get('frontend', {}), range_policy=task.get('range_policy', 'fixed'))
        else:
            kwargs = {'score_midi': case['files']['score_midi']} if method == 'LadderSym' else {}
            prediction = client.predict(case['files']['performance'], case['files']['score_audio'], directory, **kwargs)
        truth = [e for kind in ('extra', 'missed', 'correct') for e in midi_events(case['files'][kind], kind)]
        metrics = event_metrics(prediction['events'], truth, task['tolerances'])
        for row in metrics:
            row.update(dataset=task['dataset'], method=method, case_id=case['case_id'], piece=case['piece'], seconds=perf_counter() - start)
        payload = dict(prediction=prediction, truth=truth, rows=metrics)
        save_json(directory / 'result.json', payload)
        return payload

    @staticmethod
    def native_cpu_worker(task):
        from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation
        directory = Path(task['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        with NoteEvaluation._worker_log(directory / 'run.log'):
            try:
                return MistakeBenchmarker.native_evaluate(task)
            except BaseException:
                traceback.print_exc()
                raise

    class native_NativeNeuralPool(NeuralPool):

        def execute(self, task):
            try:
                return MistakeBenchmarker.native_evaluate(task, self._client(task['method']))
            except BaseException:
                path = Path(task['directory']) / 'run.log'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as log:
                    traceback.print_exc(file=log)
                raise

    @staticmethod
    def native_run_tasks(tasks, *, output, models, workers=None, neural_workers=None, on_result, on_stage=None):
        if workers is not None and workers < 1 or (neural_workers is not None and neural_workers < 1):
            raise ValueError('Worker counts must be positive')
        cpus = [t for t in tasks if t['method'] == 'Attune']
        counts = dict(cpu_workers=0, neural_workers={})

        def collect(pool, submit, batch):
            futures = {submit(task): task for task in batch}
            delivered = set()
            try:
                for future in as_completed(futures):
                    payload = future.result()
                    delivered.add(future)
                    on_result(payload)
            finally:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                for future in futures:
                    if future not in delivered and (not future.cancelled()) and (future.exception() is None):
                        on_result(future.result())
        with MistakeBenchmarker.single_thread_environment():
            if cpus:
                n = min(len(cpus), MistakeBenchmarker.worker_limits(workers, neural=False)[0])
                counts['cpu_workers'] = n
                if on_stage is not None:
                    on_stage('Attune', n)
                if workers == 1:
                    from threadpoolctl import threadpool_limits
                    with threadpool_limits(limits=1):
                        for task in cpus:
                            on_result(MistakeBenchmarker.native_cpu_worker(task))
                else:
                    pool = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context('spawn'), initializer=MistakeBenchmarker._init_cpu, initargs=(None,))
                    collect(pool, lambda t: pool.submit(MistakeBenchmarker.native_cpu_worker, t), cpus)
            for method in ('PolyTune', 'LadderSym'):
                batch = [t for t in tasks if t['method'] == method]
                if not batch:
                    continue
                n = min(len(batch), neural_workers) if neural_workers is not None else MistakeBenchmarker.staged_neural_limit(len(batch), device=models[method].device)
                counts['neural_workers'][method] = n
                if on_stage is not None:
                    on_stage(method, n)
                pool = MistakeBenchmarker.native_NativeNeuralPool(models['PolyTune'], output, n, laddersym=models['LadderSym'])
                try:
                    collect(pool.pool, pool.submit, batch)
                finally:
                    pool.close()
        return counts

    @staticmethod
    def cpu_seconds():
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time() + children.ru_utime + children.ru_stime

    @staticmethod
    def soundfont_prepare(render_workers=None, *, base=SOUNDFONT_BASE, output=SOUNDFONT_OUT):
        base, output = (Path(base).resolve(), Path(output).resolve())
        import numpy as np
        import pretty_midi
        import soundfile as sf
        from scipy.signal import resample_poly
        from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        PitchBenchmarker = PitchBenchmarker
        if (output / 'manifest.json').exists():
            saved = json.loads((output / 'manifest.json').read_text())
            for case in saved['cases']:
                for name, path in case['files'].items():
                    if digest(path) != case['hashes'][name]:
                        raise ValueError('Prepared input changed: ' + path)
            print('Soundfont inputs already prepared and verified.', flush=True)
            return
        original = json.loads((base / 'run.json').read_text())['contract']['manifest']
        manifest = deepcopy(original)
        renderer = NoteBenchmarker.__new__(NoteBenchmarker)

        def inventory(piece):
            folder = 'mistake/' + piece + '/stems_midi'
            return (piece, tree(original['repo'], original['revision'], folder, output / 'inventory' / (piece + '.json')))
        with ThreadPoolExecutor(max_workers=min(8, len(original['selected_pieces']))) as pool:
            inventories = dict(pool.map(inventory, original['selected_pieces']))

        def render_case(case):
            folder = 'mistake/' + case['piece'] + '/stems_midi'
            entries = inventories[case['piece']]
            source = next((e for e in entries if Path(e['path']).stem.lower() == case['stem'].lower()))
            midi = download(original['repo'], original['revision'], source['path'], output / 'original_midi' / source['path'], sha256=source.get('lfs', {}).get('oid'))

            def notes(path):
                pm = pretty_midi.PrettyMIDI(str(path))
                return sorted(((i.program, n.pitch, n.velocity, round(n.start, 6), round(n.end, 6)) for i in pm.instruments for n in i.notes))
            assert notes(midi) == sorted(notes(case['files']['correct']) + notes(case['files']['extra'])), case['case_id']
            case['rendering'] = dict(performance_midi=str(midi), performance_midi_sha256=digest(midi), original_audio={k: case['files'][k] for k in ('score_audio', 'performance')}, audio={})
            for kind, source_midi in [('score_audio', Path(case['files']['score_midi'])), ('performance', midi)]:
                target = output / 'audio' / case['case_id'] / kind
                wav = renderer.synth_midi(source_midi, out_dir=target / 'render_44100', force=True)
                y, sr = sf.read(wav, always_2d=True)
                y = resample_poly(y.mean(axis=1), 160, 441)
                assert sr == 44100
                converted = target / 'audio_16000.wav'
                sf.write(converted, y, 16000, subtype='PCM_24')
                case['files'][kind] = str(converted)
                case['hashes'][kind] = digest(converted)
                case['rendering']['audio'][kind] = dict(peak=float(np.max(np.abs(y))), seconds=len(y) / 16000, clipped_samples=int(np.count_nonzero(np.abs(y) >= 1)))
            return case['case_id']
        n = min(len(manifest['cases']), render_workers or max(1, (os.cpu_count() or 4) - 1))
        progress = PitchBenchmarker.Progress(enabled=True, compact=True)
        progress.update_compact(len(manifest['cases']), 'render', f'{n} parallel soundfont workers', count=0)
        try:
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(render_case, case) for case in manifest['cases']]
                for future in as_completed(futures):
                    progress.update_compact(len(futures), 'render', future.result())
        finally:
            progress.finish()
        manifest['intervention'] = dict(soundfont=str(renderer.SOUNDFONT_PATH), soundfont_sha256=digest(renderer.SOUNDFONT_PATH), renderer='Existing NoteBenchmarker.synth_midi: FluidSynth gain 1, reverb/chorus off, 44100 Hz', conversion='Stereo mean; scipy resample_poly 160/441; mono PCM24 16000 Hz', labels='Byte-identical native labels; original score and performed MIDI notes unchanged', original_run=str(base), boundary_assistance=False)
        save_json(output / 'manifest.json', manifest)

    @staticmethod
    def soundfont_run(workers=None, neural_workers=None, *, base=SOUNDFONT_BASE, output=SOUNDFONT_OUT):
        base, output = (Path(base).resolve(), Path(output).resolve())
        import pandas as pd
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        models_for = MistakeBenchmarker.native_models_for
        summarize = MistakeBenchmarker.native_summarize
        code_identity = MistakeBenchmarker.native_code_identity
        run_tasks = MistakeBenchmarker.native_run_tasks
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        PitchBenchmarker = PitchBenchmarker
        manifest = json.loads((output / 'manifest.json').read_text())
        baseline = json.loads((base / 'run.json').read_text())['contract']
        models = models_for('coco', baseline['device'])
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            identities = dict(zip(models, pool.map(lambda model: model.preflight(), models.values())))
        for case in manifest['cases']:
            for name, path in case['files'].items():
                assert digest(path) == case['hashes'][name]
        contract = dict(source=baseline, intervention=manifest, models=identities, code=code_identity(), experiment_sha256=digest(__file__))
        prior = output / 'run.json'
        if prior.exists():
            assert json.loads(prior.read_text())['contract'] == contract, 'Run inputs changed; use a new output directory'
        metadata = dict(status='running', contract=contract)
        save_json(prior, metadata)
        rows, tasks = ([], [])
        progress = PitchBenchmarker.Progress(enabled=True, compact=True)
        total = len(manifest['cases']) * 3
        progress.update_compact(total, 'starting', 'auto-sized CPU then persistent model workers', count=0)

        def receive(payload):
            rows.extend(payload['rows'])
            pd.DataFrame(rows).to_csv(output / 'rows.csv', index=False)
            summarize(rows).to_csv(output / 'summary.csv', index=False)
            first = payload['rows'][0]
            kind = 'detected' if first['method'] == 'Attune' else 'audio'
            progress.update_compact(total, first['method'], f'{first['case_id']} {kind}')
        for method in ('Attune', 'PolyTune', 'LadderSym'):
            for case in manifest['cases']:
                directory = output / 'cases' / case['case_id'] / method.lower()
                checkpoint = directory / 'result.json'
                if checkpoint.exists():
                    receive(json.loads(checkpoint.read_text()))
                else:
                    tasks.append(dict(case=case, method=method, dataset='coco', directory=str(directory), midi_range=baseline['midi_range'], tolerances=baseline['tolerances'], range_policy=baseline['range_policy']))

        def stage_started(method, n):
            kind = 'CPU workers' if method == 'Attune' else 'persistent model workers'
            progress.update_compact(total, method, f'{n} {kind}', count=0)
        try:
            metadata['parallelism'] = run_tasks(tasks, output=output, models=models, workers=workers, neural_workers=neural_workers, on_result=receive, on_stage=stage_started)
            metadata['status'] = 'complete'
            save_json(prior, metadata)
        finally:
            progress.finish()
    MISTAKE_DB_OUTPUT_ALIASES = {'wohlfahrt': 'wolfhart', 'wolfhart': 'wolfhart'}

    def __init__(self, onset_tolerance: float=0.05) -> None:
        super().__init__(onset_tolerance=onset_tolerance)

    def mistake_db_dataset_dir(self, dataset: str) -> Path:
        """Output corpus directory under benchmarks/datasets/mistake-db.

        The source corpus is spelled ``wohlfahrt`` in violin-etudes. The user-
        facing benchmark DB folder is kept as ``wolfhart`` to match the requested
        layout.
        """
        output_dataset = self.MISTAKE_DB_OUTPUT_ALIASES.get(dataset, dataset)
        return self.MISTAKE_DIR / output_dataset

    @staticmethod
    def dataset_from_etude_midi(midi_path: PathLike) -> str:
        return self.dataset_name_for_midi(midi_path)

    def mistake_db_paths(self, dataset: str, track_id: str) -> dict[str, Path]:
        from benchmarks.modules.pitch.PitchCache import PitchCache
        dataset_dir = self.mistake_db_dataset_dir(dataset)
        return {'dataset': dataset_dir, 'audio': dataset_dir / 'audio' / f'{track_id}.wav', 'midi': dataset_dir / 'midi' / f'{track_id}.mid', 'pitch_data': PitchCache.path_for(dataset_dir, track_id), 'note_data': self.note_cache_path(dataset_dir, track_id), 'truth': dataset_dir / 'truth' / f'{track_id}.truth.json'}

    @staticmethod
    def _jsonable_truth(truth: Sequence[TruthEvent]) -> list[dict[str, Any]]:
        return [{key: int(value) if isinstance(value, np.integer) else value for key, value in event.items()} for event in truth]

    @staticmethod
    def performance_timeline(performance_notes):
        return [dict(id=int(n.id), onset=float(n.start_time), end=float(n.end_time), pitch=float(n.midi_num[0]), score_onset=float(getattr(n, 'comparison_time', n.start_time)), delay=float(getattr(n, 'timeline_delay', 0.0))) for n in performance_notes.data.values()]

    def write_truth_data(self, truth_path: PathLike, *, dataset: str, source_midi: Path, track_id: str, seed: int, truth: Sequence[TruthEvent], injector: MistakeInjector, pitch_timing: dict[str, float] | None=None, note_timing: dict[str, float] | None=None, performance_notes: NoteData | None=None) -> Path:
        truth_path = Path(truth_path)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': 1, 'dataset': dataset, 'track_id': track_id, 'seed': int(seed), 'source_midi': str(source_midi), 'truth': self._jsonable_truth(truth), 'injector': injector.last_metadata, 'pitch_timing': pitch_timing or {}, 'note_timing': note_timing or {}}
        if performance_notes is not None:
            payload['performance_timeline'] = self.performance_timeline(performance_notes)
        with open(truth_path, 'w') as fh:
            json.dump(payload, fh, indent=2)
        return truth_path

    def config_for_performance(self, performance_notes):
        """Synthetic range coverage: final injected MIDI, with app margin.

        Injection offsets have unbounded Gaussian tails before MIDI clamping;
        a fixed score-only margin cannot guarantee coverage. Only the pitch
        extrema inform this setup, never note boundaries or correspondences.
        """
        pitches = [pitch for note in performance_notes.data.values() for pitch in note.midi_num if pitch >= 0]
        bounds = Config.padded_midi_range(min(pitches), max(pitches)) if pitches else ()
        return self.attune.config_for(*self.attune.range_from_midi(bounds))

    def load_mistake_pitches(self, recording, cache_path, *, smooth=True, use_cache=True, write_cache=True):
        """Invalidate range-incompatible caches only in the mistake benchmark.

        The shared pitch cache and historical diagnostic reads remain unchanged.
        A missing range stamp requires one fresh extraction for this benchmark.
        """
        cache_path = Path(cache_path)
        stamp_path = cache_path.with_suffix(cache_path.suffix + '.mistake-range.json')
        config = recording.config
        signature = dict(version=1, fmin=config.fmin, fmax=config.fmax, sr=config.sr, w1=config.w1, h1=config.h1, tuning=config.tuning, smooth=smooth)
        try:
            compatible = json.loads(stamp_path.read_text()) == signature
        except (OSError, ValueError):
            compatible = False
        timing = self.attune.load_or_detect_pitches(recording, cache_path, smooth=smooth, use_cache=use_cache and compatible, write_cache=write_cache)
        if write_cache and cache_path.is_file():
            import benchmarks.modules.mistake.MistakeCache as _local_MistakeCache
            atomic_json = _local_MistakeCache.MistakeCache.atomic_json
            atomic_json(stamp_path, signature)
        return timing

    def generate_mistake_db_track(self, midi_path: PathLike, dataset: str, seed: int, injector: MistakeInjector | None=None, force: bool=False, program: int=40, prepare_audio: bool=True, prepare_pitches: bool=True) -> dict[str, Any]:
        """Generate one mistake-db item and its analysis caches.

        Writes:
          - <mistake-db>/<dataset>/audio/<track>_seed<seed>.wav
          - <mistake-db>/<dataset>/pitch_data/<track>_seed<seed>.pitch.pkl.xz
          - <mistake-db>/<dataset>/note_data/<track>_seed<seed>.note.json

        A generated MIDI and truth JSON are also written for reproducibility.
        Cached notes are unmodified detector output; no reference durations enter extraction.
        """
        midi_path = Path(midi_path)
        injector = injector or MistakeInjector(out_dir=self.MISTAKE_DIR)
        score_data = OneInstrumentScoreData(midi_path)
        reference_notes = score_data.note_data
        performance_notes, truth = injector.inject(reference_notes, np.random.default_rng(seed))
        track_id = f'{midi_path.stem}_seed{seed}'
        paths = self.mistake_db_paths(dataset, track_id)
        for key in ('audio', 'midi', 'pitch_data', 'note_data', 'truth'):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
        if force or not paths['midi'].exists():
            self.notedata_to_pm(performance_notes, program=program).write(str(paths['midi']))
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
        timeline_score_onsets = MistakeDetectorBase.timeline_score_onsets
        timeline_score_onsets(dict(injector=injector.last_metadata, performance_timeline=self.performance_timeline(performance_notes)), OneInstrumentScoreData(paths['midi']).note_data)
        audio_path = paths['audio']
        pitch_timing, note_timing = ({}, {})
        if prepare_audio:
            audio_path = self.synth_midi(paths['midi'], out_dir=paths['audio'].parent, force=force)
        if prepare_audio and prepare_pitches:
            config = self.config_for_performance(OneInstrumentScoreData(paths['midi']).note_data)
            recording = self.attune.recording_for(config, score_data=OneInstrumentScoreData(midi_path))
            recording.audio_data = AudioData(audio_filepath=str(audio_path), config=recording.config)
            pitch_timing = self.load_mistake_pitches(recording, cache_path=paths['pitch_data'], smooth=True, write_cache=True)
            note_timing: dict[str, float]
            cached_notes = None
            if paths['note_data'].exists() and (not force):
                candidate, metadata = self.load_note_data(paths['note_data'])
                if metadata.get('trimmed_boundaries') is False:
                    cached_notes = (candidate, metadata)
            if cached_notes is not None:
                recording.note_data, note_metadata = cached_notes
                note_timing = {'note_compute_time': float(note_metadata.get('note_compute_time', 0.0))}
            else:
                recording.reset_analysis()
                _, note_compute_time = self.detect_recording_notes_timed(recording)
                note_timing = {'note_compute_time': note_compute_time}
                self.save_note_data(recording.note_data, paths['note_data'], metadata={**note_timing, 'model': 'l2', 'method': 'recording.detect_notes', 'trimmed_boundaries': False})
        self.write_truth_data(paths['truth'], dataset=dataset, source_midi=midi_path, track_id=track_id, seed=seed, truth=truth, injector=injector, pitch_timing=pitch_timing, note_timing=note_timing, performance_notes=performance_notes)
        return {'dataset': dataset, 'track_id': track_id, 'seed': int(seed), 'source_midi': str(midi_path), 'audio': str(audio_path), 'midi': str(paths['midi']), 'pitch_data': str(paths['pitch_data']), 'note_data': str(paths['note_data']), 'truth': str(paths['truth']), 'truth_events': len(truth), **pitch_timing, **note_timing}

    def build_mistake_db(self, datasets: Sequence[str]=('kayser', 'wohlfahrt'), seeds: Iterable[int]=range(6), injector: MistakeInjector | None=None, max_tracks: int | None=None, force: bool=False, verbose: bool=True, write_manifest: bool=True) -> pd.DataFrame:
        """Build the Polytune-style mistake database for the etude corpora."""
        import pandas as pd
        injector = injector or MistakeInjector(out_dir=self.MISTAKE_DIR)
        seed_values = list(seeds)
        rows: list[dict[str, Any]] = []
        for dataset in datasets:
            tracks = self._limit(list(self.iter_etudes(dataset)), max_tracks)
            for track_index, (title, midi_path) in enumerate(tracks, start=1):
                for seed in seed_values:
                    row = self.generate_mistake_db_track(midi_path, dataset=dataset, seed=int(seed), injector=injector, force=force)
                    rows.append(row)
                    if verbose:
                        print(f'[mistake-db/{dataset}] {track_index}/{len(tracks)} seed={seed} {title[:40]}')
        df = pd.DataFrame(rows)
        if write_manifest:
            manifest_path = self.MISTAKE_DIR / 'manifest.csv'
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(manifest_path, index=False)
        return df

    def analyze_recording(self, recording, method: str='pelt', model: str='l2', do_correction: bool=False, update_distances: bool=True, truncate: bool=False, note_cache_path: PathLike | None=None, trim_reference: NoteData | None=None) -> dict[str, float]:
        del trim_reference
        recording.reset_analysis()
        cached_notes = None
        if note_cache_path is not None and Path(note_cache_path).exists():
            candidate, metadata = self.load_note_data(note_cache_path)
            if metadata.get('trimmed_boundaries') is False:
                cached_notes = (candidate, metadata)
        if cached_notes is not None:
            recording.note_data, metadata = cached_notes
            note_compute_time = float(metadata.get('note_compute_time', 0.0))
        else:
            _, note_compute_time = self.detect_recording_notes_timed(recording)
            if note_cache_path is not None:
                self.save_note_data(recording.note_data, note_cache_path, metadata={'note_compute_time': note_compute_time, 'model': 'l2', 'method': 'recording.detect_notes', 'trimmed_boundaries': False})
        if recording.note_data.times:
            recording.resize_score(to_span='note')
        mistake_start = time.perf_counter()
        recording.detect_mistakes()
        mistake_detection_compute_time = time.perf_counter() - mistake_start
        mistake_check_compute_time = 0.0
        if do_correction:
            check_start = time.perf_counter()
            recording.repeat_splitter.check_mistakes()
            mistake_check_compute_time = time.perf_counter() - check_start
        if update_distances:
            recording.update_alignment_distances()
        if truncate:
            recording.trim_end(mark_unsaved=False)
        return {'note_compute_time': note_compute_time, 'mistake_detection_compute_time': mistake_detection_compute_time, 'mistake_check_compute_time': mistake_check_compute_time}

    def bench_mistake_track(self, midi_path: PathLike, injector: MistakeInjector, seeds: Iterable[int]=range(6), mode: MistakeMode='symbolic', max_sec: float | None=None, correct_symbolic: bool=False, canonical: bool=True) -> pd.DataFrame:
        return self.aggregate(self._mistake_scores(midi_path, injector, seeds, mode, max_sec, correct_symbolic, canonical), canonical=canonical)

    def bench_mistake_dataset(self, dataset: str, injector: MistakeInjector, seeds: Iterable[int]=range(6), mode: MistakeMode='symbolic', max_tracks: int | None=None, max_sec: float | None=None, correct_symbolic: bool=False, canonical: bool=True, verbose: bool=True, write: bool=False) -> pd.DataFrame:
        tracks = self._limit(list(self.iter_etudes(dataset)), max_tracks)
        scores: list[MistakeBenchmarker.MistakeScore] = []
        for i, (title, midi_path) in enumerate(tracks):
            scores += self._mistake_scores(midi_path, injector, seeds, mode, max_sec, correct_symbolic, canonical)
            if verbose:
                print(f'[{dataset}/{mode}] {i + 1}/{len(tracks)} {title[:32]}')
        df = self.aggregate(scores, canonical=canonical)
        if write:
            self.write_dataset_result(df, 'mistakes', dataset)
        return df

    def clean_render_mistakes(self, midi_path: PathLike, max_sec: float | None=None) -> dict[str, float | int | str | bool]:
        from benchmarks.modules.pitch.PitchCache import PitchCache
        midi_path = Path(midi_path)
        score_data = OneInstrumentScoreData(midi_path)
        reference_notes = score_data.note_data
        config = self.config_for_performance(reference_notes)
        recording = self.attune.recording_for(config, score_data=score_data)
        recording.audio_data = AudioData(audio_filepath=str(self.synth_midi(midi_path)), config=recording.config)
        if max_sec:
            recording.audio_data.end_index = min(recording.audio_data.end_index, int(max_sec * config.sr))
        pitch_timing = self.load_mistake_pitches(recording, cache_path=PitchCache.path_for(self.etude_corpus_dir_for_midi(midi_path), midi_path.stem), smooth=True, write_cache=True)
        analyze_timing = self.analyze_recording(recording, note_cache_path=self.note_cache_path(self.etude_corpus_dir_for_midi(midi_path), midi_path.stem))
        reference_note_count = sum((1 for note_time in reference_notes.times if max_sec is None or note_time < max_sec))
        return dict(track=midi_path.stem, spurious=len(recording.alignment.pitch_mistakes), timing_spurious=len(recording.alignment.timing_mistakes), **{'Reference Notes': reference_note_count}, **pitch_timing, **analyze_timing)

    def _mistake_scores(self, midi_path: PathLike, injector: MistakeInjector, seeds: Iterable[int], mode: MistakeMode, max_sec: float | None, correct_symbolic: bool=False, canonical: bool=True) -> list[MistakeScore]:
        midi_path = Path(midi_path)
        score_data = OneInstrumentScoreData(midi_path)
        reference_notes = score_data.note_data
        scores: list[MistakeBenchmarker.MistakeScore] = []
        if mode == 'symbolic':
            recording = self.attune.recording_for(self.attune.config_for(196, 3000), score_data=score_data)
            for seed in seeds:
                performance_notes, truth = injector.inject(reference_notes, np.random.default_rng(seed))
                recording.note_data = performance_notes
                mistake_detection_start = time.perf_counter()
                recording.detect_mistakes()
                mistake_detection_compute_time = time.perf_counter() - mistake_detection_start
                mistake_check_compute_time = 0.0
                if correct_symbolic:
                    mistake_start = time.perf_counter()
                    recording.repeat_splitter.check_mistakes()
                    mistake_check_compute_time = time.perf_counter() - mistake_start
                has_duration_truth = any((event['type'] in {'short', 'long'} for event in truth))
                detected_mistakes = [*recording.alignment.pitch_mistakes, *(recording.alignment.timing_mistakes if has_duration_truth else [])]
                scores.append(MistakeBenchmarker.MistakeScore(counts=self.score_symbolic(detected_mistakes, truth, pairs=recording.alignment.pairs, canonical=canonical), mistake_detection_compute_time=mistake_detection_compute_time, mistake_check_compute_time=mistake_check_compute_time))
            return scores
        if mode == 'audio':
            dataset = self.dataset_from_etude_midi(midi_path)
            for seed in seeds:
                performance_notes, truth = injector.inject(reference_notes, np.random.default_rng(seed))
                performance_name = f'{midi_path.stem}_seed{seed}'
                paths = self.mistake_db_paths(dataset, performance_name)
                for key in ('audio', 'midi', 'pitch_data', 'note_data', 'truth'):
                    paths[key].parent.mkdir(parents=True, exist_ok=True)
                if not paths['midi'].exists():
                    self.notedata_to_pm(performance_notes).write(str(paths['midi']))
                audio_path = self.synth_midi(paths['midi'], out_dir=paths['audio'].parent, force=False)
                if max_sec:
                    truth = [event for event in truth if event['time'] < max_sec]
                config = self.config_for_performance(OneInstrumentScoreData(paths['midi']).note_data)
                recording = self.attune.recording_for(config, score_data=OneInstrumentScoreData(midi_path))
                recording.audio_data = AudioData(audio_filepath=str(audio_path), config=recording.config)
                if max_sec:
                    recording.audio_data.end_index = min(recording.audio_data.end_index, int(max_sec * config.sr))
                pitch_timing = self.load_mistake_pitches(recording, cache_path=paths['pitch_data'], smooth=True, write_cache=True)
                has_duration_truth = any((event['type'] in {'short', 'long'} for event in truth))
                analyze_timing = self.analyze_recording(recording, note_cache_path=paths['note_data'])
                self.write_truth_data(paths['truth'], dataset=dataset, source_midi=midi_path, track_id=performance_name, seed=int(seed), truth=truth, injector=injector, pitch_timing=pitch_timing, note_timing={'note_compute_time': analyze_timing.get('note_compute_time', 0.0)})
                detected_mistakes = [*recording.alignment.pitch_mistakes, *(recording.alignment.timing_mistakes if has_duration_truth else [])]
                scores.append(MistakeBenchmarker.MistakeScore(counts=self.score_onset(detected_mistakes, truth, onset_tolerance=0.1, pairs=recording.alignment.pairs, canonical=canonical), **pitch_timing, **analyze_timing))
            return scores
        raise ValueError(f'unknown mistake mode: {mode!r}')

    @staticmethod
    def _match_onsets(detected_times: Sequence[float], truth_times: Sequence[float], onset_tolerance: float) -> tuple[int, int, int]:
        used_truth_events = [False] * len(truth_times)
        true_positives = 0
        for detected_time in sorted(detected_times):
            for truth_index, truth_time in enumerate(truth_times):
                if not used_truth_events[truth_index] and abs(detected_time - truth_time) <= onset_tolerance:
                    used_truth_events[truth_index] = True
                    true_positives += 1
                    break
        return (true_positives, len(detected_times) - true_positives, len(truth_times) - true_positives)

    @classmethod
    def score_correct_alignment(cls, pairs: Sequence[tuple[Note | None, Note | None]] | None, mistakes: Sequence[Mistake], truth: Sequence[TruthEvent]) -> tuple[int, int, int]:
        if pairs is None:
            return (0, 0, 0)
        truth_incorrect_ids = {int(event['score_note_id']) for event in truth if 'score_note_id' in event}
        detected_incorrect_ids = {int(mistake.midi_note.id) for mistake in mistakes if mistake.type != 'insertion' and mistake.midi_note is not None}
        all_score_ids: set[int] = set()
        predicted_correct_ids: set[int] = set()
        for user_note, score_note in pairs:
            if score_note is None:
                continue
            score_id = int(score_note.id)
            all_score_ids.add(score_id)
            if user_note is None or score_id in detected_incorrect_ids:
                continue
            source_score_id = getattr(user_note, 'source_score_id', _mistakebenchmarker_SOURCE_SCORE_ID_UNSET)
            if source_score_id is not _mistakebenchmarker_SOURCE_SCORE_ID_UNSET:
                if source_score_id is None or int(source_score_id) != score_id:
                    continue
            predicted_correct_ids.add(score_id)
        truth_correct_ids = all_score_ids - truth_incorrect_ids
        true_positives = len(predicted_correct_ids & truth_correct_ids)
        false_positives = len(predicted_correct_ids - truth_correct_ids)
        false_negatives = len(truth_correct_ids - predicted_correct_ids)
        return (true_positives, false_positives, false_negatives)

    @classmethod
    def score_symbolic(cls, mistakes: Sequence[Mistake], truth: Sequence[TruthEvent], onset_tolerance: float=0.06, pairs: Sequence[tuple[Note | None, Note | None]] | None=None, canonical: bool=True) -> MistakeCounts:
        """Score pitch edits plus explicit short/long duration mistakes.

        `canonical` maps substitution losslessly onto PolyTune's missed/extra space:
        a wrong note is the intended score note MISSED (a deletion) plus the played
        pitch being EXTRA (an insertion). Folding both truth and detected
        substitutions into the deletion/insertion tallies makes the score invariant
        to whether a wrong note is represented as one substitution or as a
        deletion+insertion (and stops a detector substitution on a screwup-2/3 extra
        note from being a pure false positive). The `substitution` row is still
        reported as the direct sub-vs-sub agreement, but it is informational under
        `canonical` (OVERALL sums deletion+insertion only — see aggregate)."""
        truth_substitution_ids = {event['score_note_id'] for event in truth if event['type'] == 'substitution'}
        truth_substitution_times = [event['time'] for event in truth if event['type'] == 'substitution']
        truth_deletion_ids = {event['score_note_id'] for event in truth if event['type'] == 'deletion'}
        truth_insertion_times = [event['time'] for event in truth if event['type'] == 'insertion']
        detected_substitution_ids = [mistake.midi_note.id for mistake in mistakes if mistake.type == 'substitution']
        detected_substitution_times = [mistake.user_note.start_time for mistake in mistakes if mistake.type == 'substitution']
        detected_deletion_ids = [mistake.midi_note.id for mistake in mistakes if mistake.type == 'deletion']
        detected_insertion_times = [mistake.user_note.start_time for mistake in mistakes if mistake.type == 'insertion']
        deletion_truth = set(truth_deletion_ids)
        deletion_detected = list(detected_deletion_ids)
        insertion_truth_times = list(truth_insertion_times)
        insertion_detected_times = list(detected_insertion_times)
        if canonical:
            deletion_truth |= truth_substitution_ids
            deletion_detected += detected_substitution_ids
            insertion_truth_times += truth_substitution_times
            insertion_detected_times += detected_substitution_times
        counts_by_type: MistakeCounts = {}
        sub_tp = len(set(detected_substitution_ids) & truth_substitution_ids)
        counts_by_type['substitution'] = (sub_tp, len(detected_substitution_ids) - sub_tp, len(truth_substitution_ids) - sub_tp)
        del_tp = len(set(deletion_detected) & deletion_truth)
        counts_by_type['deletion'] = (del_tp, len(deletion_detected) - del_tp, len(deletion_truth) - del_tp)
        counts_by_type['insertion'] = cls._match_onsets(insertion_detected_times, sorted(insertion_truth_times), onset_tolerance)
        for duration_type in ('short', 'long'):
            truth_ids = {event['score_note_id'] for event in truth if event['type'] == duration_type}
            detected_ids = {mistake.midi_note.id for mistake in mistakes if mistake.type == duration_type and mistake.midi_note is not None}
            if truth_ids or detected_ids:
                tp = len(truth_ids & detected_ids)
                counts_by_type[duration_type] = (tp, len(detected_ids) - tp, len(truth_ids) - tp)
        if pairs is not None:
            counts_by_type['correct'] = cls.score_correct_alignment(pairs, mistakes, truth)
        return counts_by_type

    @classmethod
    def score_onset(cls, mistakes: Sequence[Mistake], truth: Sequence[TruthEvent], onset_tolerance: float=0.1, pairs: Sequence[tuple[Note | None, Note | None]] | None=None, canonical: bool=True) -> MistakeCounts:
        """Onset-matched scoring for the audio bench. `canonical` folds substitution
        into missed/extra exactly as score_symbolic does: the substitution's score
        onset joins the deletion (missed) times, its played onset joins the insertion
        (extra) times. The `substitution` row stays as the direct agreement
        (informational under canonical)."""

        def truth_times(mistake_type: str) -> list[float]:
            return [e['time'] for e in truth if e['type'] == mistake_type]

        def detected_times(mistake_type: str, attr: str) -> list[float]:
            return [getattr(mistake, attr).start_time for mistake in mistakes if mistake.type == mistake_type and getattr(mistake, attr) is not None]
        substitution_truth = truth_times('substitution')
        substitution_played = detected_times('substitution', 'user_note')
        counts_by_type: MistakeCounts = {}
        counts_by_type['substitution'] = cls._match_onsets(substitution_played, substitution_truth, onset_tolerance)
        deletion_truth = truth_times('deletion')
        deletion_detected = detected_times('deletion', 'midi_note')
        insertion_truth = truth_times('insertion')
        insertion_detected = detected_times('insertion', 'user_note')
        if canonical:
            deletion_truth += substitution_truth
            deletion_detected += detected_times('substitution', 'midi_note')
            insertion_truth += substitution_truth
            insertion_detected += substitution_played
        counts_by_type['deletion'] = cls._match_onsets(deletion_detected, deletion_truth, onset_tolerance)
        counts_by_type['insertion'] = cls._match_onsets(insertion_detected, insertion_truth, onset_tolerance)
        for duration_type in ('short', 'long'):
            duration_truth = truth_times(duration_type)
            duration_detected = detected_times(duration_type, 'user_note')
            if duration_truth or duration_detected:
                counts_by_type[duration_type] = cls._match_onsets(duration_detected, duration_truth, onset_tolerance)
        if pairs is not None:
            counts_by_type['correct'] = cls.score_correct_alignment(pairs, mistakes, truth)
        return counts_by_type

    @staticmethod
    def aggregate(scores: Sequence[MistakeScore], canonical: bool=True) -> pd.DataFrame:
        import pandas as pd
        pitch_types = ('substitution', 'deletion', 'insertion')
        duration_types = tuple((mistake_type for mistake_type in ('short', 'long') if any((mistake_type in score['counts'] for score in scores))))
        mistake_types = (*pitch_types, *duration_types)
        overall_types = ('deletion', 'insertion', *duration_types) if canonical else mistake_types
        include_correct = any(('correct' in score['counts'] for score in scores))
        row_types = (*mistake_types, 'correct') if include_correct else mistake_types
        aggregate_counts = {mistake_type: [0, 0, 0] for mistake_type in row_types}
        for score in scores:
            counts = score['counts']
            for mistake_type in row_types:
                aggregate_counts[mistake_type] = [aggregate_value + score_value for aggregate_value, score_value in zip(aggregate_counts[mistake_type], counts.get(mistake_type, (0, 0, 0)))]

        def precision_recall_f_measure(true_positive_count: int, false_positive_count: int, false_negative_count: int) -> dict[str, float]:
            precision = true_positive_count / (true_positive_count + false_positive_count) if true_positive_count + false_positive_count else 1.0 if false_negative_count == 0 else 0.0
            recall = true_positive_count / (true_positive_count + false_negative_count) if true_positive_count + false_negative_count else 1.0
            f_measure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            return {'Precision': precision, 'Recall': recall, 'F-measure': f_measure}
        timing_keys = ['pitch_detector_compute_time', 'pitch_smoother_compute_time', 'pitch_compute_time', 'note_compute_time', 'mistake_detection_compute_time', 'mistake_check_compute_time']
        timing = {key: float(np.mean([float(score.get(key, 0.0)) for score in scores])) if scores else 0.0 for key in timing_keys}
        rows: dict[str, dict[str, float]] = {mistake_type: {**precision_recall_f_measure(*aggregate_counts[mistake_type]), **timing} for mistake_type in row_types}
        overall_counts = [sum((aggregate_counts[mistake_type][count_index] for mistake_type in overall_types)) for count_index in range(3)]
        rows['OVERALL'] = {**precision_recall_f_measure(*overall_counts), **timing}
        return pd.DataFrame(rows).T

    @staticmethod
    def run_request(request):
        from benchmarks.modules.mistake.MistakeCache import MistakeCache
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        run_comparison = MistakeBenchmarker.run_comparison
        run_symbolic_comparison = MistakeBenchmarker.run_symbolic_comparison
        from benchmarks.modules.mistake.competitors.PolyTune import PolyTune
        from benchmarks.modules.mistake.competitors.LadderSym import LadderSym
        atomic_json = MistakeCache.atomic_json
        saved = MistakeCache.completed_results(request)
        if saved is not None:
            return saved
        defaults = asdict(Config())
        if defaults != request['expected_defaults']:
            raise ValueError('Production defaults changed after notebook setup; run all cells again.')
        output = Path(request['output'])
        options = dict(seeds=request['seeds'], rates=request['rates'], methods=request['methods'], tolerances=request['tolerances'], workers=request['workers'], source_metadata=request['source_metadata'])
        atomic_json(output / 'production_defaults.json', defaults)
        if request['stage'] == 'symbolic':
            return run_symbolic_comparison(request['sources'], output, **options)
        if request['stage'] == 'audio':
            return run_comparison(request['sources'], output, **options, input_kinds=('detected', 'audio'), polytune=PolyTune(device=request['polytune_device']), laddersym=LadderSym(device=request.get('laddersym_device', 'cpu')), neural_workers=request['neural_workers'], polytune_last=request['polytune_last'], force_pitch_detection=request.get('force_pitch_detection', False))
        raise ValueError(f'Unknown stage: {request['stage']}')

    @classmethod
    def main(cls, argv=None):
        import argparse
        import json
        import sys
        from dataclasses import asdict
        from algorithms.Config import Config
        parser = argparse.ArgumentParser(description='Mistake benchmarks: injected cases, native author labels, or COCO-E soundfont rendering.')
        parser.add_argument('--defaults', action='store_true')
        commands = parser.add_subparsers(dest='command')
        commands.add_parser('injected', help='Read an injected-run JSON request from stdin')
        commands.add_parser('native', help='Read a native-run JSON request from stdin')
        soundfont = commands.add_parser('soundfont', help='Prepare or evaluate FluidSynth COCO-E inputs')
        soundfont.add_argument('action', choices=('prepare', 'run'))
        soundfont.add_argument('--base', type=Path, default=SOUNDFONT_BASE)
        soundfont.add_argument('--output', type=Path, default=SOUNDFONT_OUT)
        soundfont.add_argument('--workers', type=int)
        soundfont.add_argument('--neural-workers', type=int)
        soundfont.add_argument('--render-workers', type=int)
        args = parser.parse_args(argv)
        if args.defaults:
            print(json.dumps(asdict(Config()), sort_keys=True))
        elif args.command == 'injected':
            cls.run_request(json.load(sys.stdin))
        elif args.command == 'native':
            cls.native_run(**json.load(sys.stdin))
        elif args.command == 'soundfont':
            for count in (args.workers, args.neural_workers, args.render_workers):
                if count is not None and count < 1:
                    parser.error('Worker counts must be positive')
            if args.action == 'prepare':
                cls.soundfont_prepare(args.render_workers, base=args.base, output=args.output)
            else:
                cls.soundfont_run(args.workers, args.neural_workers, base=args.base, output=args.output)
        elif not sys.stdin.isatty():
            cls.run_request(json.load(sys.stdin))
        else:
            parser.print_help()
        return 0
if __name__ == '__main__':
    raise SystemExit(MistakeBenchmarker.main())
