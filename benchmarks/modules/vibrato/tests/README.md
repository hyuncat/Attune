# Current notebook experiments

Use `benchmarks/notebooks/vibrato.ipynb` for the current frame and interval gates.
The notebook reads completed score-assisted frame results, then offers a short
cell to compare one accumulated cycle and 250 ms against each matching
frame-only gate (with and without quality ≥0.30). It reuses saved fits and
preserves the completed results. No production algorithm is changed.

- `FrameGates.py`: local rate, width and quality gates.
- `IntervalGates.py`: contiguous-run cycle/duration filters and paired replay.
- `FrameGateDiagnostics.py`: read-only tables and plots.
- `test_*.py`: synthetic checks, runnable with
  `at-venv/bin/python -m unittest discover -s benchmarks/modules/vibrato/tests -p 'test_*.py'`.

Shared evaluation lives in `VibratoBenchmarker.py` (`YangMetrics`); its existing
scoring definitions are unchanged. In particular, the >280 ms predicted-event
cutoff is independent of the new 250 ms detection filter. The latter uses
half-open frame support (number of frames × frame spacing); it is a duration-only
analogue, not a reproduction of Yang's frame cleanup and candidate extraction.

# Historical score and boundary preparation

Open `benchmarks/notebooks/yang_score_boundaries.ipynb`, or use the commands below
from the repository root. No benchmark was run while implementing this experiment.

The default cohort is the **six previously attempted candidates**, not the entire
Yang corpus: Huangjiangqin 1–3 (erhu) and Yangjian 1–3 (violin). The earlier score
search found an Erquan piano arrangement, not independently verified scores for
all these performances. The mapping deliberately calls these candidates. The
full release includes many different pieces; do not apply this one MIDI to all
of them. Extend `yang_score_candidates.json` with a recording ID and a matching
MIDI path (relative to that JSON file) for each additional piece. `instrument`
can select a zero-based melody instrument; null uses the highest active note.
All mapped cases are retained regardless of match quality. Unmapped recordings
are listed in the prepared manifest and are not part of any comparison.

## 1. Prepare, then inspect the matches

```sh
at-venv/bin/python -m benchmarks.modules.vibrato.tests.YangScore prepare \
  --output benchmarks/results/yang_score_boundary_v1_inputs
```

This uses the full-recording Yang adapter and its existing common pYIN caches;
missing/incompatible caches trigger pitch inference. It runs DTW and baseline
note segmentation but **does not calculate vibrato F1**. It freezes pitch frames
(including confidence and volume), annotations and configuration in `inputs.pkl`.
Only load this locally generated pickle. Preparation may take time.

Each recording folder contains:

- `score.mid`: transposed, monophonic excerpt with **original symbolic rhythm**,
  starting at zero. DTW chooses the excerpt, not its internal note timings.
- `dtw_bounds.json`: the separate audio-time boundary estimate.
- `dtw_path.csv` and `alignment.png`: inspection aids.

The skyline uses source event identity so adjacent equal-pitch MIDI events remain
separate notes. Rests stay as gaps; accompaniment may still contaminate the
skyline. DTW uses the earlier 100 ms grid, 200 ms pitch median, integer shift
search ±24 semitones, fractional tuning estimate, capped pitch error and strict
positive-axis steps. Tuning is used for matching only; MIDI transposition is
integer. Endpoints are limited to the matched voiced span. DTW interpolates
through unvoiced gaps; repeated equal pitches have little boundary evidence.
These are estimated boundaries, **not perfect knowledge or ground truth**.

Inspect before interpreting results. Do not drop poor matches based on vibrato
F1. A revised score/mapping requires a fresh preparation and output directory.
Existing result directories are never overwritten by this runner.

## 2. Score-assisted production run

```sh
at-venv/bin/python -m benchmarks.modules.vibrato.tests.YangScore run \
  --prepared benchmarks/results/yang_score_boundary_v1_inputs \
  --mode score --output benchmarks/results/yang_score_boundary_v1_score --workers 2
```

This reports audio-only Attune and Yang DT/BR under `common_baselines`, then
score-assisted Attune under `attune_score`. It constructs a fresh `ScoreData`
from each excerpt and calls `Recording.analyze_notes()` on the frozen original
pitch frames: production segmentation, alignment, robust fitting and local repeat
recovery. DTW boundaries are **not** supplied to this arm. Vibrato uses the final
recovered notes. No production parameters change.

`comparison.csv` joins the summaries with an `arm` column. Compare `frame_f1`
(detection F1), `yang_note_f1` (interval F1) and `aggregate_soft_f1` (parameter
F1) separately. Choose the metric of interest before deciding whether Attune
beats Yang; do not switch metrics after seeing the winner. Use both Yang variants
on this identical cohort, not a previously reported score on another subset or
Yang's published number under a different protocol. Per-recording details and
final `note_bounds.json` are also saved.

## 3. Optional boundary diagnostic

If score-assisted performance remains below Yang on your chosen metric:

```sh
at-venv/bin/python -m benchmarks.modules.vibrato.tests.YangScore run \
  --prepared benchmarks/results/yang_score_boundary_v1_inputs \
  --mode dtw --output benchmarks/results/yang_score_boundary_v1_dtw --workers 2
```

This bypasses note detection/alignment/repeat recovery and supplies the frozen
DTW boundaries directly to the unchanged Attune vibrato detector. Compare with
step 2 using the same prepared inputs. Full-recording evaluation masks, pitch
and annotations remain unchanged, including frames outside the matched span.
Inputs and MIDI/boundary assets are hashed to prevent accidental reuse after
edits. Both runs should use the same code checkout.

A gain suggests boundary sensitivity; a loss may reflect bad score matches or
DTW boundaries. Neither result alone measures a true oracle upper bound. For
that, manually verify/correct *musical note* onsets/offsets independently of the
vibrato annotations. Vibrato-area annotations are not musical note boundaries.

If reliable boundaries still leave weak F1, the next isolated experiment should
address the within-note model: separate slow pitch drift/portamento from periodic
modulation, and allow vibrato to begin/end within a note. Do not retune those
components simultaneously with this boundary experiment.
