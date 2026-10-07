# Vibrato benchmark

This package compares vibrato detectors on one frame-aligned API. The same
runner is used from the terminal and from `vibrato.ipynb`; dataset builders and
competitor files do not contain separate entry points.

The notebook uses `VibratoNotebook.run_suite('yang' | 'coco')`, inherited from
`VibratoBenchmarker`. Computation, source fingerprints, reports, and paired
significance tests live on `VibratoBenchmarker`; `VibratoNotebook` only formats
tables. The three core modules have no module-level functions.

Current paired suites rebuild production note boundaries and reuse matching
per-track checkpoints under source-keyed `paired_<suite>_<hash>` result folders.
Yang requires the prepared candidate-score assets in
`benchmarks/results/yang_score_boundary_v1_inputs`. Coco uses 20 stems, six
injection profiles, and clean/20 dB/10 dB conditions. Each comparison uses one
`GatedAttune` adapter with Yang's rate/extent thresholds; app defaults are unchanged.

New CLI runs also checkpoint each successful detector/continuous-track job in
`<output-dir>/checkpoints/`, immediately in the worker, before final reports are
written. Interrupted runs reuse those checkpoints and compute only missing jobs.
Writes are atomic; damaged files are recomputed, and failed or skipped jobs are
retried. Checkpoint keys include example data, audio file size/modification time,
detector settings, scoring tolerances, and analysis source code. Suite reruns rebuild reports but reuse matching track checkpoints; remove the
checkpoint directory to repeat estimator computation. Direct Python
callers opt in with `VibratoBenchmarker.run(..., cache_dir=Path(...))`; arbitrary
unpickleable notebook estimators remain uncached.

## Layout

```text
benchmarks/modules/vibrato/
├── VibratoDetectorBase.py       shared detector API and frame adapters
├── VibratoBenchmarker.py        suites, paired inference, metrics, reports, and CLI
├── VibratoNotebook.py           notebook presentation helpers
├── competitors/
│   ├── Attune.py
│   ├── Driedger.py
│   ├── DriedgerDataset.py       paper-port validation corpus only
│   ├── HerreraBonada.py
│   ├── McLeod.py
│   ├── Rossignol.py
│   ├── VenturaSousaFerreira.py
│   ├── Yang.py                  Yang, YangDT, and YangBR
│   └── data/yang/               runtime Bayes model and its provenance
├── datasets/
│   ├── CocoDataset.py
│   ├── CocoControl.py
│   ├── CocoRenderer.py
│   ├── SyntheticDataset.py
│   └── YangDataset.py
├── provenance/
│   ├── notes/                   archived experiment rationale and results
│   └── yang/                    native AVA validation fixture and harness
├── tests/VibratoBenchmarkerTest.py
└── (notebook: ../../notebooks/vibrato.ipynb)
```

Every detector inherits `VibratoDetectorBase` and implements:

```python
def estimate(self, example: VibratoExample) -> VibratoEstimate:
    ...
```

`VibratoEstimate` is always sampled on the input example's frame grid. Shared
missing-pitch interpolation, sub-frame extrema, and sparse-to-dense frame
mapping live on the base class. Method-specific numerical operations stay on
their detector class.

## Detector names

The CLI uses short, self-contained names:

- `attune`
- `attune_prony` (ablation)
- `driedger`
- `driedger_benchmark_range`
- `herrera_bonada`
- `herrera_bonada_yang_window` (ablation)
- `mcleod`
- `rossignol`
- `ventura_sousa_ferreira`
- `yang_dt`
- `yang_br`

Run `--list-methods` for the authoritative registry.

## Dataset organization

The datasets fall into two different categories.

Shared comparison corpora run every selected detector:

- `SyntheticDataset` creates controlled pitch-contour cases and reads/writes
  the benchmark's annotated CSV format.
- `CocoDataset` injects known curves into CocoChorales MIDI, renders audio,
  detects pitch, and supplies the same cases to every selected detector.
  Note-local methods use audio-detected notes after production same-pitch merging,
  score alignment, and repeat recovery (`Recording.analyze_notes()`), separately
  for each noise condition. MIDI remains the score and evaluation truth, not the
  supplied analysis boundaries. The app uses the same orchestration entry point;
  vibrato is refreshed after final note recovery. Existing pitch caches can
  still be reused.
- `YangDataset` turns Yang et al.'s released real-audio annotations into shared
  cases. Its annotated spans are pseudo-note boundaries, so this is a
  preliminary parameter comparison rather than a complete false-alarm test.

`DriedgerDataset` is different. It exists only to verify that the Driedger port
tracks the paper's released acceptance corpus. It therefore lives beside
`Driedger.py` instead of among the shared datasets. The unified runner exposes
it through `--driedger`.

The Yang Bayes `.mat` file is required at runtime. Its JSON provenance and the
native MATLAB harness are retained because they establish how the portable
model was produced and checked. They are data/provenance assets, not adapter
modules.

## Running

Use the repository environment:

```shell
# Show the registry.
at-venv/bin/python benchmarks/modules/vibrato/VibratoBenchmarker.py --list-methods

# Fast shared-contour smoke run.
at-venv/bin/python benchmarks/modules/vibrato/VibratoBenchmarker.py \
  --quick --methods rossignol,yang_dt,attune

# Parallel synthetic run.
at-venv/bin/python benchmarks/modules/vibrato/VibratoBenchmarker.py \
  --workers 8 --replicates 5

# End-to-end Coco run.
at-venv/bin/python benchmarks/modules/vibrato/VibratoBenchmarker.py \
  --coco --workers 8 --coco-stem-limit 20 \
  --coco-profiles constant,accelerating,decelerating,widening,narrowing,none \
  --coco-snrs clean,20,10 --coco-auto-yin-window

# Yang real-audio parameter subset.
at-venv/bin/python benchmarks/modules/vibrato/VibratoBenchmarker.py \
  --yang --workers 2 --yang-recording-limit 2

# Driedger paper-port validation corpus.
at-venv/bin/python benchmarks/modules/vibrato/VibratoBenchmarker.py \
  --driedger --driedger-root benchmarks/datasets/vibrato_driedger

# Straight-note sfizz/Coco control validation (same entry point).
at-venv/bin/python benchmarks/modules/vibrato/VibratoBenchmarker.py \
  --coco-control --coco-control-instrument violin

# Any tidy annotated contour CSV.
at-venv/bin/python benchmarks/modules/vibrato/VibratoBenchmarker.py \
  --dataset-csv annotations.csv
```

The notebook imports `VibratoNotebook` from
`benchmarks.modules.vibrato.VibratoNotebook`. After running a suite, call
`VibratoBenchmarker.run_significance(output, suite)` to compute paired pooled-F1
label-swap tests and cluster-bootstrap intervals from its checkpoints. Each corpus
forms one Holm family across seven metrics and its selected competitors. Yang
also saves performer-grouped sensitivity results. `show_results()` and
`show_significance()` on `VibratoNotebook` display the tables.

## Reports

The output directory contains:

- `summary.csv`: frame-weighted comparison;
- `note_macro.csv`: case/note-macro comparison;
- `run_config.json`: dataset, tolerances, workers, and detector registry;
- `raw_outputs/<method>/cases.csv`: per-case metrics;
- `raw_outputs/<method>/frames.csv`: scored frame curves;
- scenario and no-vibrato diagnostics beside each method's raw outputs.

Scoring jobs are grouped by detector and shared analysis input. Pickleable
detectors use a spawned process pool; local callback detectors fall back to a
thread pool. CPU and wall time are recorded separately.

Every worker holds BLAS and OpenMP to one thread. The detectors' inner linear
algebra is small and numerous, so a thread pool per worker spins instead of
computing: unpinned, Attune's Yang analysis groups cost eight times more
process CPU than wall time on their own and never finished at all under
`--workers 9`. `--workers` is therefore the only parallelism, and the reported
`Audio(s)/Compute(s)` measures one thread's work per detector.

## Soft F1 and complete Yang release (2026-10-05)

The notebook's Yang suite selects all six recordings with manual half-cycle
marks and evaluates their complete audio, including negatives. The generic CLI
also supports all 76 bundled WAV files via `--yang --yang-full-audio
--yang-all-recordings`. Missing parameter labels are NaN and are excluded rather
than counted as non-vibrato. Negative regions of parameter-annotated recordings
still penalize parameter false alarms.

Primary parameter scores are **Extent Soft F1**, **Rate Soft F1**, and their
arithmetic mean **Overall Soft F1**. For each evaluated reference-positive,
predicted-positive frame with parameter p>0, credit is
`max(0, 1 - abs(estimate-p)/p)`. Sum credit C and positive counts P (predictions)
and R (references); precision=C/P, recall=C/R, F1=2C/(P+R).
Empty denominators score zero when the annotation scope is evaluable;
non-evaluable parameter scopes report NaN. Counts pool frames, so long recordings
have more weight. `note_macro.csv` also exposes unweighted case means (a case
is a recording for the full Yang dataset). Hard-tolerance scores remain labeled
`F1 (hard)` for comparison. This is a soft-PRF adaptation using Yang's credit,
not a metric claimed to have been validated specifically for vibrato.

Yang event reports use detected runs longer than 0.28 seconds, 100 ms onset
and max(100 ms, 20% reference duration) offset tolerance. Boundary matching is
maximum-cardinality one-to-one. Matched parameter accuracy follows Equation 19:
at least half of a detected interval must overlap the reference; multiple
corresponding detections are averaged equally. Parameter averages pool matched
reference intervals, with explicit matched/annotated counts and coverage.
Frame and note PRF are reported pooled and as means over successful recordings;
these recording means are not a reproduction of the paper's train/test iterations.

Error diagnostics use positive temporal overlap: a reference overlapping multiple
detections is a split, a detection overlapping multiple references is a merge;
zero-overlap detections/references are spurious/non-detected. Only-bad-onset and
only-bad-offset apply to isolated one-to-one overlapping pairs with exactly one
failed boundary. Counts may overlap and do not partition all errors (both-bad
boundaries are not onset-only or offset-only). Rates divide by reference count,
except spurious rate divides by detection count. These explicitly labeled overlap
diagnostics are an adaptation, not an exact port of Molina et al.'s taxonomy.

Full Yang runs derive note boundaries using the production note detector on the
common pitch contour, never vibrato annotation boundaries. Extent truth still
samples that same contour at manual extrema; it is not an independent annotated
pitch track. The released Yin tracks are used only to configure an eight-semitone
range margin, with annotated-frequency fallback where absent. Both Yang and Coco
configure padding explicitly because the current pYIN/HMM no longer pad internally.
Coco first includes every injected excursion, then adds eight semitones per side.
Adaptive YIN windows cover four periods at the resulting lower bound.

References: [Yang et al. (2017), §§4.2–4.3](https://luweiyang.com/wp-content/uploads/2017/05/jmm2017_luwei_yang.pdf);
[Fränti and Mariescu-Istodor (2023), Soft precision and recall](https://doi.org/10.1016/j.patrec.2023.02.005).
