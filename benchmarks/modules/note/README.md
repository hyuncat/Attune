# Note benchmarks

The module follows the pitch benchmark layout:

- `NoteBenchmarker.py`: method registry, historical component benchmarks, dataset orchestration, CPU-timed evaluation, parallel workers, scoring, significance and CLI.
- `NoteDetectorBase.py`: shared detector contract, method dispatch, recording preparation and note conversion helpers.
- `NoteNotebook.py`: notebook configuration exports and presentation facade over the evaluation runner.
- `NoteCache.py`: note serialization, pitch-cache adapters, result compatibility and interrupted-run recovery for all methods.
- `competitors/`: one class-owned implementation per competitor: Attune, BasicPitch, CrepeNotes, Tony, Ruptures and SlopeWindow. Ruptures owns the PELT, KernelCPD, bottom-up, window and dynamic-programming variants.
- `sweeps/`: note-detection, injected-note and string-edit parameter experiments.
- `tests/`: focused validation without requiring neural models or benchmark corpora.

## Entry points

Run commands from the repository root:

```sh
python -m benchmarks.modules.note.NoteBenchmarker --list-methods
python -m benchmarks.modules.note.NoteBenchmarker --help
python -m unittest discover -s benchmarks/modules/note/tests
```

The CLI retains the historical cached-pitch component experiment. The current end-to-end evaluation is launched from `benchmarks/notebooks/note.ipynb`:

```python
from benchmarks.modules.note.NoteNotebook import NoteNotebook, NotebookConfig

benchmark = NoteNotebook(NotebookConfig(force_methods=()))
rows = benchmark.run_preliminary_coco()
```

`NoteNotebook` also exposes `run_preliminary_urmp()` and `run_preliminary_bach10()`.
Use `NoteEvaluation` from `NoteBenchmarker` for the same evaluation without notebook presentation helpers. Sweeps import from `benchmarks.modules.note.sweeps`.

## Evaluation and cache behavior

The current notebook evaluates score-conditioned Attune after alignment, robust timing refitting and repeat recovery. The optional `attune-audio-only` diagnostic stops after segmentation. External competitors use audio-only transcription. CPU totals include frontend, segmentation and, for score-conditioned Attune, refinement. Cached frontends retain their original CPU cost.

Summaries pool note-event TP/FP/FN for micro precision, recall and F1, with onset-only and offset-aware metrics reported separately. Paired significance tests compare the common successful track intersection and swap complete source groups.

`force_methods=("attune",)` refreshes Attune predictions while retaining compatible frontend caches and competitor results. Use `force_methods=()` to resume completed jobs. Result keys track the relevant competitor implementation, shared detector/cache code, scoring, inputs and configuration. The structural refactor gets new result identities; historical artifacts remain untouched and are not automatically certified as equivalent.

Historical results measure older configurations and are not current production or held-out validation. The full [September 23 audit, with October 4 protocol updates](../../../docs/archive/note-benchmark-audit-2026-09-23.md) is archived for provenance.
