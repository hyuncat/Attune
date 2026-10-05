# Mistake benchmarks

This package follows the same ownership model as pitch and vibrato. There are
four implementation files at the package root:

| File | Responsibility |
| --- | --- |
| `MistakeBenchmarker.py` | Injected/native orchestration, parallel workers, CPU timing, and COCO-E soundfont runs |
| `MistakeDetectorBase.py` | Competitor contract and shared mistake-event scoring |
| `MistakeNotebook.py` | Notebook requests, result tables, and COCO-E audit presentation |
| `MistakeCache.py` | Evaluation checkpoints, generated-asset reuse, native pitch/note stages, and neural prediction adapters |

`competitors/` has one file per method: `PolyTune.py`, `LadderSym.py`, and
`Nakamura.py`. Each class owns its pinned setup and prediction implementation.
Neural competitors also contain their persistent client and isolated worker;
there are no separate setup or worker modules. All inherit `MistakeDetectorBase`.

`datasets/` contains corpus preparation and injection, `sweeps/` retains repeat
comparisons and parameter studies, `tests/` contains regression checks, and
`provenance/` preserves historical audits and notes. Retired one-off experiments
remain in [`benchmarks/archive/experiments`](../../archive/experiments/README.md).

## Use

`benchmarks/notebooks/mistake.ipynb` compares Attune, PolyTune, and LadderSym on
injected CocoChorales-E2 and re-rendered author-labeled CocoChorales-E. It reads
saved results unless `RUN_E2` or `RUN_E` is enabled.
`benchmarks/notebooks/coco-e_audit.ipynb` contains the rendering investigation.

```shell
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker --defaults
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker soundfont --help
# Injected and native commands read their request JSON from stdin.
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker injected < request.json
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker native < request.json

# Explicit optional setup; --help does not download or install anything.
at-venv/bin/python -m benchmarks.modules.mistake.competitors.PolyTune setup --help
at-venv/bin/python -m benchmarks.modules.mistake.competitors.LadderSym setup --help
at-venv/bin/python -m benchmarks.modules.mistake.competitors.Nakamura setup --help

QT_QPA_PLATFORM=offscreen at-venv/bin/python -m unittest discover \
  -s benchmarks/modules/mistake/tests -t . -p '*Test.py'
```

The neural worker entry points use the same competitor files with the `worker`
command in their dedicated environments. Their imports do not require the Qt
application or the benchmark orchestrator.

## Cache and provenance

Saved results, input assets, and historical notebooks are preserved. New runs
should use a fresh run tag after this consolidation. Evaluation contract version
2 fingerprints the consolidated orchestrator as inference code; a scheduling-file
exclusion must not accidentally exclude the inference now living in that file.
Raw neural reuse still validates model, input bytes, and implementation identity;
it never relaxes checks simply because a file moved.

Historical implementation notes are in
[`provenance/notes/benchmark-history.md`](provenance/notes/benchmark-history.md).
Those notes describe prior protocols, not current entry points.
