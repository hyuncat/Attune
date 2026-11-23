# Benchmark and sweep data

- `benchmarks/{pitch,note,mistake}/`: evaluation results. Pitch and note use 432 tracks (36 Bach10, 136 URMP, 260 CocoChorales). Only recordings used in parameter tuning are excluded.
- `benchmarks/mistake/`: latest final-setting results used by the manuscript.
- `sweeps/{pitch,note,mistake}/`: parameter-search results; `sweeps/cohort.json` identifies tuning tracks.
- CSV files contain per-track rows, summaries, or statistical tests. Most metrics are fractions; `paper_table.csv` and mistake pooled summaries use percentages.
- JSON files record selections, settings, and provenance. `manifest.json` maps exports to source files and checksums; `validation.json` records checks.
- End-to-end mistake p-values use separate two-comparison Holm families for E and E2.

Local inference and notebook caches are gitignored under `caches/`; they are not part of the paper exports. Original audio and model weights live under the experiment datasets and competitor directories. Absolute paths in metadata refer to the original local runs.
