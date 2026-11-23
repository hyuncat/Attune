# Pitch benchmarks

The package root has four implementation files, matching the vibrato ownership
model with a dedicated cache:

| File | Responsibility |
| --- | --- |
| `PitchBenchmarker.py` | Offline/streaming orchestration, workers, scoring, timing, and comparison runs |
| `PitchDetectorBase.py` | Common detector contract, examples, estimates, and audio transformations |
| `PitchNotebook.py` | Notebook configuration, suite presentation, paired reports, and ablation notebook support |
| `PitchCache.py` | Pitch estimates, streaming checkpoints/latency samples, ablation evidence, and completed-row reads |

Metric, resource, timing, smoother-ablation, and streaming helpers are owned by
these classes; there are no separate top-level helper modules. Independent
configured streaming and ablation runners are nested in their owning class.
The pYIN variant detectors live with the pYIN competitor implementation.

`competitors/` has one source file per competitor family. Praat's offline and
streaming adapters both live in `Praat.py`. pYIN's frontend, smoother, Viterbi
helper, and controlled variants live in `PYIN.py`; executable helpers are class
methods. Detector adapters inherit `PitchDetectorBase` directly or through their
family's adapter. Dataset adapters, historical parameter sweeps, and tests remain
in `datasets/`, `sweeps/`, and `tests/`.

```shell
at-venv/bin/python -m experiments.modules.pitch.PitchBenchmarker --help
at-venv/bin/python -m experiments.modules.pitch.PitchBenchmarker --list
QT_QPA_PLATFORM=offscreen at-venv/bin/python -m unittest discover \
  -s experiments/modules/pitch/tests -t . -p '*Test.py'
```

Use `experiments/experiments/pitch.ipynb` for the main suites and
`experiments/experiments/pyin_ablation.ipynb` for the controlled ablation views.
Persisted results and datasets remain at their existing locations. Cache readers
retain their version, configuration, and input checks; no benchmark results were
regenerated during the package consolidation.

## CREPE tiny streaming comparison

The preliminary online cells in `pitch.ipynb` pass `ONLINE_METHODS`, including
`crepe_tiny` alongside McLeod and the existing five streaming methods on two
tracks per instrument. The full online cell uses the same membership through
`PAPER_ONLINE_METHODS`. McLeod
is also included in preliminary streaming and all default offline suites. `run_streaming_comparison(suite_methods=...)` selects
membership without forcing inference; `methods=...` remains the selective-force
option. With `force_rerun=False`, compatible per-track checkpoints and latency
sidecars are reused across suite folders, including the old five-method suite.

`CrepeStreaming.py` uses the supported pip CREPE **tiny** weights with the
[browser demo's input policy](https://github.com/marl/crepe/blob/de4888e6d448357ceafea10fc6010061c6f19a55/crepe.js).
The pip weights differ from the older gh-pages model export; this row is not an
exact reproduction of that exported model.

- 1024 input samples at 16 kHz; per-frame mean/std normalization; nine-bin local
  weighted decoding; strict confidence > 0.5; no Viterbi or temporal smoothing.
- Power-of-two capture buffers: at 44.1 kHz, 4096 samples per callback, with only
  the first 64 ms linearly resampled for inference. The remaining callback tail
  is unused, matching the demo. Updates occur every 92.88 ms; center timestamps
  imply 60.88 ms of acquisition delay before compute/backlog.
- The common score-range gate applies after inference. Digital silence is
  explicitly unvoiced instead of carrying NaNs from division by zero.

This is Python/TensorFlow CPU replay, not browser or microphone-to-screen timing.
Model loading and warm-up are outside steady-state timing. The runner retains
its dedicated-worker process-CPU latency simulation, plus measured wall compute
time. The new method alone includes its adapter version, CREPE package version,
and tiny-weight hash in checkpoint identity; existing baseline hashes are
unchanged. Expanded paired tests include six competitors and three online
metrics (18 Holm-corrected comparisons).

Restart the notebook kernel, run setup, and execute the expanded online cell
when ready. The offline cell adds McLeod while retaining cached peer rows. Interrupted `crepe_demo` results from
the older exported weights are separate and cannot satisfy `crepe_tiny` caches.

## McLeod Pitch Method

`McLeod.py` executes the original C++ engine from the official Tartini 1.2 source
archive. It uses the release's default MPM profile: 2048-sample frames, 1024-sample
hops at 44.1 kHz, persistent high-pass filtering, automatic noise-floor scoring,
warped NSDF aggregation, note tracking, and retrospective octave correction.
Streaming commits causal estimates (23.22 ms acquisition look-ahead); offline
returns the revised stored track on the native grid. Both retain the common
score-range gate after inference.

A C++ compiler and single-precision FFTW3 are required. The pinned archive,
license, settings, explicit compatibility fixes, native build instructions,
validation, and the distinction from the later thesis pipeline are documented in
[`competitors/tartini/README.md`](competitors/tartini/README.md).

McLeod prediction caches, streaming checkpoints, and completed summary rows are
versioned. Old local-MPM results are refreshed on the next suite run while peers
remain reusable. No corpus benchmark was rerun for this replacement.

### November 2026 volume promotion

Production live voicing uses 4% of causal running p95 RMS, including silence
and the current frame; post-hoc keeps 4% of whole-recording p95. Confidence
thresholds are unchanged. Benchmark pitch cache v16 refreshes Attune only;
streaming includes that version only in Attune's checkpoint identity, keeping
competitor hashes and the historical suite locations stable. App cache versions
and demo files are not changed by this promotion.

`pitch.ipynb` enables `NotebookConfig(attune_only_refresh=True)`. This preserves
competitor rows verbatim and refuses missing/incompatible scores or streaming
latency sidecars instead of falling back to detection. Preliminary clean suites
can recover their exact track subset from matching expanded saved suites.
Pairwise analyses recompute from the updated returned rows. Restart the kernel
and Run All; later runs reuse completed Attune results. No full benchmark was
executed as part of this change.

Attune benchmark scheduling explicitly uses eight worker processes, starting
with the first batch in both offline and streaming runs. It skips memory
calibration and RAM-based throttling for Attune only; fewer workers are used
when fewer than eight jobs remain. Offline Attune uses one track per job to
keep all eight workers available even for small preliminary suites. Other
methods retain their adaptive scheduling. This changes no cache identities;
completed results remain reusable. Restart the notebook kernel to load it.


### November 2026: native-geometry 449-track online comparison

In `experiments/experiments/pitch.ipynb`, run setup and the two cells under
**Expanded online native-geometry experiment**. Offline and preliminary cells are
not prerequisites. The first prints and saves the exact dry-run plan; the second
returns `(paper_online, paper_online_table)` and recomputes paired tests.

The API is `BENCH.run_native_feedback_evaluation(dry_run=True/False)`.
This explicit selective policy works with `attune_only_refresh=True` but requires
`force_rerun=False`. It requires compatible Attune running-p95, SwiftF0, SPICE and
Tartini checkpoints across all 449 tracks before launching any inference. Only
new pYIN quarter-window-hop, native single-frame Praat, and rolling 10 ms
CREPE-tiny configurations run. Completed method/track checkpoints resume.
Legacy and offline identities remain unchanged.

Reports live in `notebook_runs/native_feedback_2026_11_v1`, and each corpus gets
its own sibling `native_feedback_2026_11_v1/checkpoints` directory. The comparison
CSV pools frame-count accuracy and exact duration-weighted steady-state P95
feedback latency. Startup/coverage, geometry and source-selection manifests are
exported separately. Timing source sidecars remain lossless; derived interval
caches are keyed by metric policy, clock, sidecar digest and audio duration.
Missing timestamps or invalid timing block unchanged methods and invalidate only
the affected changed method/track. New sidecars also preserve frequencies so
future score-only changes can rescore without inference.
