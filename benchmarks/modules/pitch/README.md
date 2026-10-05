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
at-venv/bin/python -m benchmarks.modules.pitch.PitchBenchmarker --help
at-venv/bin/python -m benchmarks.modules.pitch.PitchBenchmarker --list
QT_QPA_PLATFORM=offscreen at-venv/bin/python -m unittest discover \
  -s benchmarks/modules/pitch/tests -t . -p '*Test.py'
```

Use `benchmarks/notebooks/pitch.ipynb` for the main suites and
`benchmarks/notebooks/pyin_ablation.ipynb` for the controlled ablation views.
Persisted results and datasets remain at their existing locations. Cache readers
retain their version, configuration, and input checks; no benchmark results were
regenerated during the package consolidation.
