"""Exclude exact tuning recordings and refresh production Attune; competitors are cache-only.

Original results remain immutable. Derived manifests identify the source rows,
excluded recordings, and the exact tuning manifest used for filtering.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT/'benchmarks/results'
TUNING = RESULTS/'sweeps/balanced_212_v3/cohort.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


class Holdout:
    def __init__(self, manifest=TUNING):
        self.manifest=Path(manifest)
        self.items=read(manifest)['items']
        self.track_ids = {
            dataset: {item['source_track_id'] for item in self.items
                      if item['dataset'] == dataset}
            for dataset in ('urmp', 'bach10-original')
        }
        self.signature=hashlib.sha256((sha(manifest)+'bach_urmp_track_exclusion_v3').encode()).hexdigest()[:16]
        self.output=RESULTS/'evaluation_holdout'/self.signature

    def excluded(self,dataset,track_id,group=None):
        # Both pitch and note adapters use the manifest's source_track_id.
        # The group is deliberately ignored: other stems of a tuning piece stay.
        if dataset == 'coco':
            return False
        if dataset not in self.track_ids:
            raise ValueError(f'Unknown dataset {dataset}')
        return str(track_id) in self.track_ids[dataset]

    def filter(self,rows):
        mask=[not self.excluded(r.dataset,r.track_id) for r in rows.itertuples()]
        return rows.loc[mask].copy()

    def tracks(self,tracks):
        return [t for t in tracks if not self.excluded(t['dataset'],t['track_id'],t.get('group'))]

    def save_rows(self,source,rows,label,removed):
        directory=self.output/label;directory.mkdir(parents=True,exist_ok=True)
        rows.to_csv(directory/'rows.csv',index=False)
        write(directory/'exclusions.json',dict(
            policy='exclude exact Bach10/URMP tuning recordings; Coco evaluation unchanged',
            tuning_manifest=str(self.manifest),tuning_sha256=sha(self.manifest),
            source_rows=str(source),source_rows_sha256=sha(source),
            removed=removed,report_only=True,
            limitation='Excludes current frozen tuning cohort; does not undo historical dataset consultation'))
        return directory


def _pitch_result(bench,source,label,methods):
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    from benchmarks.modules.pitch.PitchNotebook import PitchNotebook
    holdout=bench.holdout
    source=Path(source)
    if not source.exists():raise FileNotFoundError(f'Missing saved pitch run; no inference launched: {source}')
    original=pd.read_csv(source)
    rows=original.loc[original.model.isin(methods)].copy()
    if not PitchBenchmarker.current_scores(rows):raise ValueError(f'Incompatible pitch metric counts: {source}')
    selected=holdout.filter(rows)
    ids=set(zip(selected.dataset,selected.track_id))
    if not ids or selected.duplicated(['dataset','track_id','model']).any():raise ValueError('Empty/duplicate pitch results')
    for method in methods:
        group=selected[selected.model==method]
        if set(zip(group.dataset,group.track_id))!=ids:
            raise ValueError(f'Incomplete saved pitch results for {method}: {source}')
    removed=rows.loc[~rows.index.isin(selected.index),['dataset','track_id']].drop_duplicates().to_dict('records')
    directory=holdout.save_rows(source,selected,label,removed)
    summary=PitchBenchmarker.pool_scores(selected)
    summary.to_csv(directory/'summary.csv')
    return PitchNotebook.SuiteResult(label,directory/'rows.csv',selected,summary)


def pitch_notebook(config):
    from benchmarks.modules.pitch.PitchNotebook import PitchNotebook
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    class CachedPitchNotebook(PitchNotebook):
        def __init__(self,config):
            super().__init__(config);self.holdout=Holdout()
        def run_suite(self,name,datasets,max_tracks,methods=None,force=False,cache_variant=None,**kwargs):
            if force or methods is not None:raise ValueError('Filtered evaluation is cache-only; forced inference disabled')
            # This folder is frozen competitor evidence, not the destination for
            # current Attune inference. Its name must not follow Config defaults.
            if name == 'preliminary_urmp' and cache_variant:
                selection = cache_variant.split('__attune_', 1)[0]
                cache_variant = (selection + '__attune_pyin_plus_ordinary'
                    '__live_unv_0p985_volume_0p04__posthoc_unv_0p9875_volume_0p04')
            path=self.runs_root/name
            if cache_variant:path/=cache_variant
            result=_pitch_result(self,path/'rows.csv','pitch/'+name,self.methods)
            result=refresh_attune(self,result,self.options(datasets=datasets,max_tracks=max_tracks,**kwargs))
            from IPython.display import display
            display(result.summary)
            return result
        def run_streaming_comparison(self,instruments=(),tracks_per_instrument=2,dataset='urmp',max_seconds=math.inf,
                                     force=False,methods=None,suite_methods=None):
            if force or methods is not None:raise ValueError('Filtered evaluation is cache-only')
            all_methods=self._resolve_methods(PitchBenchmarker.PitchStreaming.METHODS if suite_methods is None else suite_methods,
                                              PitchBenchmarker.PitchStreaming.SUPPORTED_METHODS)
            instrument_tag='all' if not instruments else '-'.join(instruments)
            variant=f'{PitchBenchmarker.PitchStreaming.VERSION}__methods_{"+".join(all_methods)}__instruments_{instrument_tag}__per_instrument_{tracks_per_instrument}__seed_{self.config.seed}__max_{max_seconds:g}s__unv_{0.985:g}__dynamic_floor_{0.03:g}'.replace('.','p')
            result=_pitch_result(self,self.runs_root/f'streaming_{dataset}'/variant/'rows.csv',
                                 f'pitch/streaming_preliminary_{dataset}',all_methods)
            return refresh_attune(self,result,self.options(datasets=(dataset,),instruments=tuple(instruments),
                per_instrument=tracks_per_instrument,max_tracks=None),
                streaming=PitchBenchmarker.StreamingConfig(workers=self.config.streaming_workers,max_seconds=max_seconds))
        def run_native_feedback_evaluation(self,*,dry_run=True):
            from benchmarks.modules.pitch.NativeFeedbackExperiment import METHODS,report
            results=tuple(_pitch_result(self,self.runs_root/f'streaming_{dataset}'/'native_feedback_2026_11_v1/rows.csv',
                          f'pitch/native_feedback_{dataset}',METHODS) for dataset in ('bach10-original','urmp','coco'))
            if dry_run:
                return pd.concat([r.rows[['dataset','track_id','model']].assign(action='competitors cached; Attune checked/refreshed on execution') for r in results])
            results=tuple(refresh_attune(self,r,self.options(datasets=(dataset,),max_tracks=None,
                per_instrument=20 if dataset=='coco' else None),
                streaming=PitchBenchmarker.StreamingConfig(workers=self.config.streaming_workers,native_feedback_policy=True))
                for dataset,r in zip(('bach10-original','urmp','coco'),results))
            directory=self.holdout.output/'pitch/native_feedback';directory.mkdir(parents=True,exist_ok=True)
            return results,report(results,directory)
    return CachedPitchNotebook(config)


def note_notebook(config):
    from benchmarks.modules.note.NoteNotebook import NoteNotebook
    class HoldoutNotes(NoteNotebook):
        def select_tracks(self,dataset):
            return Holdout().tracks(super().select_tracks(dataset))
    return HoldoutNotes(config)


def seed_streaming_checkpoint(path):
    """Copy an identical per-recording fingerprint from a prior selection.

    The streaming runner validates row identity and latency evidence before reuse.
    Keep this report-level lookup outside PitchCache so changing cohort scheduling
    does not invalidate note inference caches that depend on PitchCache's source.
    """
    import shutil
    path = Path(path)
    if path.exists():
        return
    root = next((parent for parent in path.parents
                 if parent.name == 'evaluation_holdout'), None)
    if root is None:
        return
    pattern = f'*/pitch/*/production_defaults/checkpoints/{path.parent.name}/{path.name}'
    for candidate in sorted(root.glob(pattern)):
        if candidate != path:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, path)
            return


def refresh_attune(bench, result, options, *, streaming=None):
    """Refresh Attune only; source competitor rows never enter an inference job."""
    from dataclasses import asdict
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    from benchmarks.modules.pitch.PitchNotebook import PitchNotebook
    from algorithms.Config import Config
    signature = dict(defaults=asdict(Config()), source_rows_sha256=sha(result.path),
                     code={p:sha(ROOT/p) for p in (
                         'algorithms/PitchDetector.py','algorithms/PitchSmoother.py',
                         'benchmarks/modules/pitch/competitors/Attune.py',
                         'benchmarks/modules/pitch/PitchCache.py',
                         'benchmarks/modules/pitch/PitchBenchmarker.py')},
                     streaming=asdict(streaming) if streaming is not None else None)
    directory = result.path.parent/'production_defaults'
    path=directory/'rows.csv'; contract=directory/'contract.json'
    if path.exists() and contract.exists() and read(contract)==signature:
        rows=pd.read_csv(path)
    else:
        reporter=PitchBenchmarker(options)
        keys=set(zip(result.rows.dataset,result.rows.track_id))
        tracks=[t for t in reporter.tracks() if (t.dataset,t.track_id) in keys]
        if {(t.dataset,t.track_id) for t in tracks} != keys:
            raise ValueError('Attune refresh does not match the frozen evaluation selection')
        if streaming is None:
            jobs,_=reporter.plan(['attune'],tracks,workers=bench.config.workers)
            fresh,errors,skipped=reporter.run(jobs,workers=bench.config.workers,verbose=False,
                                             compact_progress=True)
            if errors or skipped:raise RuntimeError('Incomplete Attune pitch refresh; rerun to resume caches')
        else:
            runner=PitchBenchmarker.PitchStreaming(options,streaming)
            examples=[reporter.dataset(t.dataset).example(t) for t in tracks]
            for example in examples:
                seed_streaming_checkpoint(runner._checkpoint_path(directory/'checkpoints', 'attune', example))
            fresh=runner.run(examples,methods=['attune'],checkpoint_dir=directory/'checkpoints')
        if set(zip(fresh.dataset,fresh.track_id)) != keys or len(fresh)!=len(keys):
            raise ValueError('Incomplete Attune pitch results')
        rows=bench._merge_method_rows(result.rows,fresh,['attune'],list(result.rows.model.unique()))
        directory.mkdir(parents=True,exist_ok=True)
        rows.to_csv(path,index=False);write(contract,signature)
    summary=PitchBenchmarker.pool_scores(rows)
    summary.to_csv(directory/'summary.csv')
    return PitchNotebook.SuiteResult(result.name,path,rows,summary)
