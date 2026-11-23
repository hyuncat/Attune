# AVA native MATLAB parity harness

This harness runs the released AVA FDM+DT and FDM+BR implementation at commit
`77e4dfe8affa6014112bdf23769b2a97dde0d2b7`. It also exports the four released
Bayes-rule kernel-density models into formats that Python and SciPy can read.

The GitHub Actions artifact `ava-native-results` contains:

- `ava_native.csv`: native frame values, DT/BR detections, component KDE PDFs,
  BR posteriors, and the joint BR probability for the smoke contour.
- `ava_native.mat`: the same run with detailed MATLAB arrays.
- `br_models/br_model_manifest.csv`: model type, kernel, support, bandwidth,
  and sample counts.
- `br_models/pd*_samples.csv`: plain training samples, frequencies, and
  censoring flags for each released KDE model.
- `br_models/br_kde_models_plain.mat`: all four models as plain MATLAB structs
  saved with `-v7`, readable through `scipy.io.loadmat`.
- `br_models/br_rate_pdf_reference.csv` and
  `br_models/br_extent_pdf_reference.csv`: dense native PDF/posterior oracles.
- `br_models/br_decision_reference.csv`: a compact Cartesian native BR decision
  oracle spanning rate 0--20 Hz and extent 0--3 semitones.
- `br_models/br_export_provenance.json`: source commit and released BR constants.

The action asserts that the exported component-posterior calculation produces
the same raw frame decisions as AVA's released `BayesRule.m`. A disagreement
fails the run.

The two files in `matlab/ava_compat/` repair AVA's released zero/single-passage
postprocessing crash without changing its FDM, DT, KDE, Bayes, or pruning
criteria.
