# Mistake benchmarks

This package follows the same ownership model as pitch and vibrato. There are
four implementation files at the package root:

| File | Responsibility |
| --- | --- |
| `MistakeBenchmarker.py` | Benchmark design, isolated E/E2 cohorts, paired significance, injected/native orchestration, and rendering |
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
injected CocoChorales-E2 and re-rendered author-labeled CocoChorales-E. Audio
benchmark cells are executable directly. Uncomment the isolated benchmark cell
for exact-note comparisons on 260 source tracks per dataset (20 per instrument).
E2 uses one injection
seed (0) and one clean control per source. These runs never render or analyze audio;
E uses correct + extra label MIDI as its exact performed-note input, auditing
raw-MIDI inconsistencies in `annotation_audit.json`. E downloads only pinned
author MIDI/labels, and E2 extracts only selected MIDI
from local CocoChorales shards. Output tags preserve old runs and allow matching
per-case checkpoints to resume. The annotation-input protocol uses a new v2
output tag to avoid mixing it with earlier raw-MIDI results. Each worker also
atomically checkpoints every completed method under its case's `methods/`
directory before reporting progress. Interrupted cases reuse these results,
including separate E2 seeds and clean controls; only unfinished methods rerun.
Existing whole-case checkpoints remain reusable. Worker count is
automatically sized to available CPU cores and matcher memory (override with
`ISOLATED_WORKERS`). One notebook display shows
`E1 [ XX/2600]: Attune: song_name`; the global counter advances when all of a
track–method combination's conditions finish. Its total is calculated from all
enabled datasets and selected methods (260 × 2 × 5 by default). Worker output is saved to per-case logs.
Parangonar Automatic may omit notes when merging its windowed alignments;
the adapter retains its matches and counts uncovered score/performance notes as
deletions/insertions. Duplicate and invalid IDs still fail validation.

The isolated significance cell calls `MistakeBenchmarker.isolated_significance`:
two-sided paired permutations of pooled event F1 at 100 ms / 50 cents, clustered
by source piece across seeds/stems, with 9,999 resamples and Holm correction
across all eight Attune-versus-competitor comparisons on E and E2. It rejects
incomplete pairings and excludes E2 clean controls from the error-condition
tests. The list-based alignment adapter preserves simultaneous notes and uses
production costs with an endpoint fit and one robust affine refit; it omits
acoustic repeat recovery. All design/selection/statistics live in
`MistakeBenchmarker.py`; notebook cells only configure, launch, and display.
`isolated_average_table` averages the two datasets' pooled metrics with equal
weight. The final notebook cell calls `end_to_end_significance` on saved
re-rendered E and E2 audio results, using the same clustered swaps and one
four-comparison Holm family for PolyTune/LadderSym. Small cohorts use exact
enumeration; clean controls are excluded and incomplete pairings fail.
`benchmarks/notebooks/coco-e_audit.ipynb` contains the rendering investigation.

```shell
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker --defaults
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker soundfont --help
# Injected, native, and isolated commands read JSON requests from stdin.
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker injected < request.json
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker native < request.json
at-venv/bin/python -m benchmarks.modules.mistake.MistakeBenchmarker isolated < request.json

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

The active notebook's Run All reads completed historical isolated results and
evaluates the full frozen 260-track v2 cohorts for end-to-end systems. Audio selection includes all stems, including
those sharing a source piece; significance tests cluster by source piece. E2 uses
seed 0, with 260 altered performances and 260 clean controls. E evaluates 260
performances with native labels. Fixed promotion output folders retain their
historical `distinct_pieces` names for asset/checkpoint reuse. Completed compatible
evaluations are skipped, and `force_pitch_detection=False` retains compatible
pitch caches. Changed inference settings still require fresh results; historical
results are not silently treated as current. The isolated execution cell defaults to `RUN_ISOLATED = False`, preserving the
saved symbolic results and their original provenance. Enable it explicitly for a
new isolated run, with a fresh output tag when code/configuration has changed.

The incremental runner change has an exact audited hash mapping in
`provenance/incremental-runner-upgrades.json`. It permits reuse of the immediately
preceding runner without ignoring future inference-code changes. Actual runner
hashes remain recorded in execution provenance. Source hashes, labels, models,
settings, and all other code fingerprints still have to match.

Audio evaluation runs PolyTune first, then LadderSym. Each algorithm gets the
full RAM-bounded worker pool, processing independent tracks in parallel with
persistent models. All PolyTune clients close before LadderSym starts. CPU threads
are divided across that algorithm's workers, with one core reserved. Attune
workers exit before the neural stages to release memory. Compatible checkpoints
remain reusable across this execution-only scheduling change.
