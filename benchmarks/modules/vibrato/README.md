# Vibrato benchmark

This package compares vibrato detectors on one frame-aligned API. The same
runner is used from the terminal and from `vibrato.ipynb`; dataset builders and
competitor files do not contain separate entry points.

## Layout

```text
benchmarks/vibrato/
├── VibratoDetectorBase.py       shared detector API and frame adapters
├── VibratoBenchmarker.py        parallel runner, metrics, reports, and CLI
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
└── vibrato.ipynb
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
at-venv/bin/python benchmarks/vibrato/VibratoBenchmarker.py --list-methods

# Fast shared-contour smoke run.
at-venv/bin/python benchmarks/vibrato/VibratoBenchmarker.py \
  --quick --methods rossignol,yang_dt,attune

# Parallel synthetic run.
at-venv/bin/python benchmarks/vibrato/VibratoBenchmarker.py \
  --workers 8 --replicates 5

# End-to-end Coco run.
at-venv/bin/python benchmarks/vibrato/VibratoBenchmarker.py \
  --coco --workers 8 --coco-stem-limit 20 \
  --coco-profiles constant,accelerating,decelerating,widening,narrowing,none \
  --coco-snrs clean,20,10 --coco-auto-yin-window

# Yang real-audio parameter subset.
at-venv/bin/python benchmarks/vibrato/VibratoBenchmarker.py \
  --yang --workers 2 --yang-recording-limit 2

# Driedger paper-port validation corpus.
at-venv/bin/python benchmarks/vibrato/VibratoBenchmarker.py \
  --driedger --driedger-root benchmarks/datasets/vibrato_driedger

# Straight-note sfizz/Coco control validation (same entry point).
at-venv/bin/python benchmarks/vibrato/VibratoBenchmarker.py \
  --coco-control --coco-control-instrument violin

# Any tidy annotated contour CSV.
at-venv/bin/python benchmarks/vibrato/VibratoBenchmarker.py \
  --dataset-csv annotations.csv
```

The notebook imports `NotebookConfig` and `VibratoNotebook` from
`benchmarks.vibrato.VibratoNotebook`; its commands point at the same
`VibratoBenchmarker.py` entry point.

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
