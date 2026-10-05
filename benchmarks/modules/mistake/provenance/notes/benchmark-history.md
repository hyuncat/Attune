# Mistake benchmark

Use `benchmarks/notebooks/mistake.ipynb` for two focused audio comparisons:
**CocoChorales-E2** (our injection pipeline) and **CocoChorales-E** (author MIDI
and unchanged labels, re-rendered with FluidSynth / MuseScore). Run All reads
saved results by default. Enable `RUN_E2` or `RUN_E` to run the corresponding
parallel benchmark. `WORKERS`, `RENDER_WORKERS`, and `NEURAL_WORKERS` control
CPU, rendering, and RAM-bounded persistent model pools. Change `RUN_TAG` for a
new configuration/code revision; matching partial runs retain checkpoints.
No symbolic baselines or legacy refinement ablations run in this notebook.

Use `benchmarks/notebooks/coco-e_audit.ipynb` for the rendering investigation:
parallel MIDI-only overlap census, synchronized listening/spectrogram examples,
independent spectral checks, and cached timing diagnostics. `RUN_CENSUS` defaults
to false; the default requested scope covers all available author-mirror splits.
Counts describe MIDI risk, not verified acoustic-error prevalence. The notebook
records coverage, failures, the pinned dataset revision and input hashes.

Historical experiments are preserved in
`benchmarks/notebooks/archive/mistake_experiments_2026-10-04.ipynb`; the former
`mistake_audit.ipynb` is preserved as
`benchmarks/notebooks/archive/mistake_promoted_audit_2026-09-27.ipynb`.
The [historical handoff](../../../../../docs/archive/mistake-benchmark-promoted-2026-09-27.md)
records the promoted run's results. Existing result folders are unchanged.

Attune's current flow bootstraps with the first/last-onset fit, aligns, runs the
existing robust matched-onset fitter once, and re-aligns before repeat-only
refinement. No-refinement and exact-note Attune rows use the same single fit;
legacy Checker 3 retains its iterative fitting protocol. Runs save
`score_fit_protocol="matched_onsets_once_v1"`; audits use that marker to retain
historical endpoint-only behavior for older runs. Recording code fingerprints
invalidate stale Attune evaluation checkpoints.
The extended and symbolic historical runs are preserved, including the source
corpus used by note2/mistake2. See [promotion evidence](../../../../../docs/alignment-promotion-2026-09-27.md).
Missing dependencies fail explicitly. During execution, one updating counter uses
the same renderer as the pitch/note benchmarks. Routine pipeline output is saved
to `run.log`; exceptions still stop the run and identify that log.

**PolyTune reuse across suites:** evaluation checkpoints now scope package keys to
the selected method, so adding another competitor does not invalidate PolyTune.
When an evaluation really is missing, completed sibling result directories are
searched for raw predictions with byte-identical performance/score WAVs and the
same model, device, dependency and inference-code provenance. Predictions are
rescored using current truth/tolerances. `inference_cache_hit` and
`inference_cache_source` identify reuse; original compute cost is retained while
the new invocation's inference execution time is zero. Changed input bytes or
model provenance require fresh inference. Model processes start lazily, so a run
satisfied by cached predictions does not load the neural model.

**Stage-level investigation:** [the audit report](../../../../../docs/mistake-benchmark-audit.md)
explains the completed extended run. To regenerate pitch/note metrics, refinement
traces, alignment-cost probes and oracle-pitch diagnostics on that exact corpus:

```shell
python -m benchmarks.modules.mistake.provenance.PipelineAudit benchmarks/results/mistake_competitors_coco_extended
python -m benchmarks.modules.mistake.provenance.PipelineAuditDetails benchmarks/results/mistake_competitors_coco_extended
```

These commands replay saved pitch frames and write only `pipeline_audit/`.
They do not change production settings or rerun neural competitors. The notebook
can display the saved stage table without repeating the audit. Pitch references
are final performed-MIDI occupancy, including nominal attack/release boundaries;
exploratory cost probes are not held-out parameter recommendations.

## One-time setup

Install the optional benchmark dependency in this notebook's kernel if needed: `%pip install parangonar==3.3.3`. Restart the kernel after installation. Preflight fails if a selected competitor is unavailable; missing methods are never silently excluded or given zero scores. Package versions and code hashes are saved with each run.

**TheGlueNote:** install the optional dependencies in the same environment:

```shell
python -m pip install parangonar==3.3.3 torch symusic==0.5.9 miditok==3.0.6.post1
```

The adapter uses Parangonar's bundled `thegluenote_small_checkpoint.pt` and
`TheGlueNoteMatcher` without changing weights or decoding. It supplies the
quarter-note fields required by the upstream converter. Checkpoint SHA-256 and
Torch/Symusic/MidiTok/tokenizer versions are recorded. Model construction is
included in every invocation's alignment CPU time (no hidden warm-model timing).
The upstream matcher uses CUDA when available, otherwise CPU; the selected device
is recorded. This is a symbolic note aligner in both input modes.

**Nakamura:** compile the pinned official v240109 source (C++ compiler required):

```shell
python -m benchmarks.modules.mistake.competitors.setup_nakamura
```

The setup checks the archive SHA-256 and keeps source, licence and executables in
gitignored `benchmarks/datasets/pretrained/nakamura/AlignmentTool`. For an existing
installation, set `ATTUNE_NAKAMURA_ROOT` to its `AlignmentTool` directory. Binary
hashes are recorded in each run. `--archive /path/to/AlignmentTool_v240109.zip`
supports offline setup; `--cxx clang++` selects a compiler.

The adapter preserves the official MIDI-to-MIDI algorithm: SPR-to-score conversion,
HMM alignment, error detection, merged-output HMM realignment, then correspondence
export, with the upstream script's parameters. It writes SPR directly from the
input notes to retain explicit IDs and avoid an additional MIDI quantization step.
Each invocation has an isolated temporary directory and bounded subprocess time.
Correspondences must cover every input exactly once; missing or duplicated IDs
fail explicitly. All aligners use the same final mistake-label thresholds and
event scorer; Nakamura's error detector still participates in its realignment.

Sources: [Nakamura tool and ISMIR 2017 paper](https://midialignment.github.io/demo.html),
[TheGlueNote, ISMIR 2024](https://github.com/sildater/thegluenote).

Both competitors are enabled in `METHODS` and `SYMBOLIC_METHODS`. To select a
smaller suite, edit `SELECTED_METHODS` in the notebook. Programmatic symbolic-only
runs use `run_symbolic_comparison(..., methods=SYMBOLIC_METHODS)`; detected/audio
runs use `run_comparison(..., input_kinds=('detected', 'audio'))`. Unsupported
method/input combinations are rejected before any case preparation.

Partitura may try to download a soundfont on first import when FluidSynth is installed. If that download fails, reuse Attune's local font once before preflight:

```python
import importlib.util, shutil
assets = Path(importlib.util.find_spec("partitura").origin).parent / "assets"
if not (assets / "MuseScore_General.sf3").exists():
    shutil.copyfile(REPO_ROOT / "resources/MuseScore_General.sf3", assets / "MuseScore_General.sf3")
```

**PolyTune setup:** run the following once from the repository root (Python 3.11 or 3.12):

```shell
python -m benchmarks.modules.mistake.competitors.setup_polytune
```

This downloads the pinned upstream source, creates a dedicated inference environment, and downloads the official 2.3 GB CocoChorales-E checkpoint into gitignored `benchmarks/datasets/pretrained/polytune`. It does not modify Attune's environment. The adapter defaults to CPU; set `POLYTUNE = PolyTune(device="cuda")` on a CUDA host. Preflight checks source revision, checkpoint SHA-256 and imports before generating cases. No automatic fallback to another model occurs.

**LadderSym setup:** `python -m benchmarks.modules.mistake.competitors.setup_laddersym`
installs an isolated inference environment and the author's official prompted
CocoChorales checkpoint (2.07 GB). `LadderSym` is selected by default in the
notebook; `LADDERSYM_DEVICE` controls its device. It receives performance audio,
clean score audio and the clean reference MIDI prompt. Its original inference
handler predicts missed/extra/correct events directly; Attune supplies no
transcribed notes or post-hoc alignment. Only audio-event metrics are applicable.
See the [source audit and integration details](../../../../../docs/mistake-laddersym-2026-09-28.md)
for provenance, pins, original-author code search and validation. It is a general
music error detector evaluated on monophonic stems here, not a monophonic-only model.

**MT3 status: deferred, not an executed row.** [Original MT3](https://github.com/magenta/mt3) uses T5X/JAX and a separate inference stack. [PolyTune's PyTorch MT3 baseline](https://github.com/ben2002chou/Polytune/tree/main/baseline/MT3_baseline) explicitly cannot load the original MT3 weights; its authors retrained it. The official checkpoint listing inspected here provides PolyTune weights, not that retrained MT3 checkpoint. MT3 also needs a specified transcription-to-error alignment stage. It is therefore not an easy import into this runtime. No placeholder or zero-valued MT3 scores are included.

## Net truth: recount after all injections

Scoring uses the **final serialized performed MIDI**, not the sequence of attempted injections. **Monophonic injection:** an extra note follows its host and delays subsequent notes as needed, consuming available silence first. Deletions move later notes earlier by the removed note's duration while preserving existing gaps. Duration overruns also shift later notes. Inserted/substituted pitches are integer MIDI semitones with a nonzero pitch change. Every note occupies at least one MIDI tick (220 PPQ at 120 BPM), preventing zero-tick note-off/on pairs from becoming hanging notes. Serialized performances are checked for overlap, matching onsets/pitches/durations, and missing notes before synthesis. Case fingerprints invalidate older generated MIDI and audio.

**Bounded, salient duration errors (`relative_salient_v1`):** short/long edits
sample uniformly over MIDI-representable durations within 0.5–1.5× that note's
original duration, restricted to the requested direction and an absolute change
of at least `max(0.30, Config().timing_tolerance + 0.05)` seconds. All alignment
competitors use the shared `label_pairs` duration threshold (currently 250 ms),
so the current minimum change is 300 ms. PolyTune has no duration-error output;
its pitch-event results still measure robustness to these temporal changes.
This input salience margin does not guarantee detection after extraction or
Attune's score-tempo fitting.

Notes too short to satisfy both limits are left unchanged; the selection log
records `skipped_reason`. With symmetric 0.5–1.5 bounds and a 300 ms floor,
eligibility starts at approximately 600 ms (subject to MIDI tick resolution).
Consequently the realized error rate/type balance can differ from requested
weights. Inspect attempted/skipped edits and final net truth, not just the
requested injection rate. Cumulative onset drift is intentionally unrestricted:
long edits and insertions delay the suffix; deletions advance it. No clipping or
realignment is introduced to accommodate PolyTune.

Factor bounds, salience floor and the shared threshold are saved in the new case
fingerprints. The original timeline mapping protocol is retained. Historical
absolute-second sampling remains available only by explicitly supplying
`duration_error_range_sec`; current benchmark entry points use relative bounds.
The fresh-run notebook writes new `_robust_fit` directories and checks
case IDs before presenting a paired comparison. Changed injection conditions
must not be described as an algorithm F1 gain.

Validation of this policy on the existing 13-stem, two-seed sample generated all
52 clean/injected MIDI cases without audio/model inference. All 42 emitted
duration edits (21 short, 21 long) passed bounds and salience checks after MIDI
serialization; 64 attempted duration edits were explicitly skipped. Evidence:
`benchmarks/results/duration_generation_validation_2026-09-27/validation.json`.
These are generator checks, not new F1 results.

The generator saves each note's nominal onset and signed timing shift. Net-truth matching removes only this known generated timing shift; scoring still uses actual performed onsets for extras and original score onsets for missed notes. This prevents insertion-induced shifts from manufacturing errors on every later note. Evaluated algorithms do not receive this time map.

**Insertion pitch safeguard (case version 4):** an injected insertion must differ
from both immediate neighbors in the final performed sequence, including after
substitutions, deletions, and adjacent insertions. Colliding inserted pitches are
changed to the nearest allowed integer MIDI pitch (seeded tie-breaking), also
excluding the original host pitch. Neighboring notes and timing are unchanged.
Same-pitch extensions are outside this insertion test category; the existing
short/long generator supplies duration errors. Metadata records inserted indices,
the policy version, and repaired pitches. Serialized MIDI is checked again before
rendering. Existing cases remain historical; new case keys and code fingerprints
prevent reusing them as results under this policy.

Validation: seven dedicated tests cover both insertion forms, changed neighbors,
adjacent insertions, MIDI-range boundaries, seed reproducibility, and MIDI
round-tripping. A symbolic-only sweep of the existing 13 Coco sources over 20
seeds generated 321 insertions in 260 cases, repaired 44 collisions, and left zero
same-pitch neighbor collisions among either injected or final net-truth
insertions. No audio inference was rerun for this check.

A separate one-to-one assignment within a fixed 100 ms score-onset gate first maximizes pitch-correct matches, then same-slot wrong-pitch matches, breaking ties by onset distance. Remaining score/performed notes become missed/extra notes. Duration labels are recomputed from final matched durations using the configured timing tolerance.

Thus a deletion plus a replacement at the same slot becomes a substitution; a replacement with the correct pitch cancels the pitch error. Clamped pitches and shortened/clipped durations are counted by their final values. `cases/<case_id>/net_truth.json` preserves the original injection history, final net truth, matching pairs, and thresholds. No evaluated aligner or source-note identity determines this assignment; the generator time map only removes the introduced timing shift.

This is an explicit operational definition for synthetic performances after undoing generated shifts, not universal ground truth for expressive timing. A replacement outside the 100 ms gate remains missed+extra. The truth gate stays fixed during evaluation-tolerance sweeps.

**Primary `audio_pitch` F1** counts missed+extra note events, mapping every substitution to both, for truth and predictions alike. All competitors use the same maximum one-to-one onset+pitch matching (50-cent pitch gate; 50/100/200 ms onset gates). Aligner missed-note predictions are mapped by their score IDs to the original clean-score timeline; PolyTune predicts timestamps and pitches directly. This gives equal credit for substitution and equivalent deletion+insertion predictions. `audio_missed` and `audio_extra` show the separate counts.

The earlier ID-based `pitch` metric and type-sensitive `legacy_five_type` remain aligner diagnostics, not the cross-model ranking metric. Short/long F1 is separate and only available for aligners. PolyTune does not predict these duration labels, so it has no duration, direct substitution or oracle-note rows; missing capabilities are never represented as zero scores.

The aligners use the same pitch/duration decision thresholds. Parangonar receives rounded MIDI pitches and the original score timeline; Attune retains its production score fitting. Duration differences therefore include score-fitting behavior. Early/late timing is outside this experiment.

## Reading results

**Detection range:** new mistake extractions use the final serialized injected
MIDI's minimum/maximum pitch, with four semitones of margin on each side. The
original score's extrema do not constrain this range. This is synthetic
ground-truth-assisted range setup; no performed boundaries or correspondences
are passed to the detector. Production app defaults instead pad the selected
score's range by four semitones, with manual range settings still explicit.

Mistake pitch caches carry a sibling `.mistake-range.json` stamp. The mistake
benchmark recomputes pitches when this stamp is absent or its range/frame setup
differs, including on a resumed run that needs extraction. The shared pitch-cache
format, cache version, other benchmarks' cache readers and production recording
sidecars are unchanged. Historical audit readers can still inspect old caches.

The primary table reports pooled (micro) precision, recall and F1 at 100 ms.
The exact-note table bypasses extraction for aligners; it is a diagnostic, not
a deployable result. PolyTune takes audio directly, while Parangonar shares
Attune's extracted notes. Clean controls count each source once across seeds.
All tolerance, type, and per-case diagnostics remain available in `rows.csv`
and `summary.csv`; these are multiple evaluations of the same predictions,
not independent samples.

This is a small synthetic experiment using original MIDI instrument programs,
with performed-MIDI-based boundary trimming. It does not establish performance
on real recordings or verify PolyTune training-set independence. Aligner timing
excludes shared extraction; PolyTune timing includes neural inference.

See [the injected-case audit](INJECTED_CASE_AUDIT.md) for historical findings
and the corrected monophonic protocol. Old overlapping results are retained
in their original directory and are never loaded by the notebook.

## PolyTune duration mismatch fix

The pinned upstream frame splitter could compute negative slice lengths once
the shorter recording ended. The local inference adapter clamps these to empty,
zero-padded windows. It keeps the original context sizes, masks and values for
previously successful windows; model weights and decoding are unchanged.
`PolyTuneFrames.py` contains the bounded inference splitter; the pinned checkout
remains unmodified. Regression checks cover both audio-length orderings, parity
with successful upstream windows, and the actual failing trombone recording.

Compatible rows from the pre-checkpoint runner are migrated once, checking the
known code revision, packages, source hashes and complete metric coverage.
Subsequent resumes use checkpoints as the source of truth. Detailed logs remain
in `run.log` and each case's `polytune/polytune.log`.

## Parallel execution and timing

Run All uses spawned CPU workers for independent cases and a separate pool of
isolated PolyTune/LadderSym interpreters. Each neural slot loads a model when
its first uncached inference request arrives, then reuses it across cases of that
method. A slot closes the old model before switching methods. The model checkpoint
is memory-mapped to avoid reading unused optimizer tensors into memory.
The notebook uses eight CPU workers followed by an automatically sized persistent
audio-model pool (`WORKERS = 8`, `NEURAL_WORKERS = None`, `POLYTUNE_LAST = True`).
The CPU pool is shut down before measuring available RAM and loading models.
CPU inference uses up to one model per available core (leaving one core free),
budgeting 1.6 GiB per model with .75 GiB reserved. The OS shares read-only
memory-mapped checkpoint tensors; each interpreter owns its activations and
decoding state. This memory budget is an estimate, not a hard memory limit.
Accelerators default to one model because host RAM does not describe GPU capacity.
Set `NEURAL_WORKERS` to an integer to override sizing. Staged requests are grouped by method to limit model reloads; each slot
processes multiple cases; fully cached predictions require no model loading.
The Python API retains CPU/RAM-aware defaults and overlapping execution;
pass `polytune_last=True` to `run_comparison` to use staged execution.
All numerical libraries and Torch intra/inter-op execution use one thread per
worker to avoid nested oversubscription. CPU workers share extracted notes
across aligners within a case. Set `POLYTUNE_LAST = False` to let model workers
consume prepared cases as they become ready. Staging reduces resource contention
but is not guaranteed to improve elapsed time when model inference dominates.

Only the parent updates the single progress line and aggregate CSVs. Workers
atomically checkpoint each completed evaluation before notifying the parent,
so completed work survives worker/parent failure. Per-case logs are under
`logs/`; persistent model startup logs are under `models/`.

| Field | Scope |
|---|---|
| `cpu_seconds` | Full algorithm CPU: saved pitch CPU + freshly measured note and alignment CPU for detected-note aligners; inference/audio-loading/event-decoding CPU for PolyTune |
| `execution_cpu_seconds` | CPU consumed by the invocation, including pitch-cache loading instead of cached pitch computation; PolyTune includes model setup only on the first successful case per worker |
| `frontend_cpu_seconds`, `segmentation_cpu_seconds`, `alignment_cpu_seconds` | Separable aligner stages; oracle rows omit the audio front end |
| `setup_cpu_seconds` | PolyTune import/model setup CPU, charged once per model worker |
| `model_load_cpu_seconds` | Model worker's setup diagnostic, repeated for provenance; do not sum it across cases |
| `wall_seconds` | Elapsed method work, kept separate from CPU seconds and excluding scheduling queues |
| `seconds` | Legacy elapsed alignment or neural-inference stage; never a CPU measurement |

CPU clocks match the note notebook: process CPU plus reaped child CPU inside
an isolated case worker. PolyTune measures process CPU inside its dedicated
interpreter. No timing comes from the notebook process or a sum of wall times
across concurrent tasks. Audio synthesis, score loading and metric evaluation
are benchmark preparation/evaluation and excluded from algorithm timings.
Cached pitch computation retains its recorded CPU cost. Oracle rows measure
alignment only and must not be compared as end-to-end methods.

Existing compatible F1 checkpoints are preserved. Their old wall-clock numbers
are **not** relabeled as CPU time; CPU fields stay missing and the timing table
reports `cpu_timed_cases`. Use a new output directory for a fully CPU-timed run.
Execution-only checkpoint migrations accept explicitly audited before/after
code hashes, retaining all source, settings and dependency checks.

## Score identity after tempo fitting

MIDI note IDs include metronome events and can change when a score is rebuilt
after tempo fitting. The evaluator maps the fitted score to the original by
verified note order and pitch sequence, then assigns original IDs to copies of
the mistake notes. Fitted timings stay intact for duration diagnostics; missed
events use the original score-audio onset. Missing or aliased parser IDs cannot
change the correspondence. A changed score sequence or stale alignment object
fails explicitly instead of guessing.

The ID-mapping fix invalidates older Attune scoring checkpoints, including ones
that may have silently used an aliased ID. PolyTune and Parangonar do not use
Attune's tempo-fitted score IDs and retain compatible checkpoints. Generated
audio and pitch caches are unaffected.

## Native author-dataset comparisons

The bottom of `benchmarks/notebooks/mistake.ipynb` now has independent setup,
download and CocoChorales-E cells. They call `NativeDatasets` and run
`NativeComparison` in a fresh process. The default downloads three deterministically
selected official Coco test pieces (all instrument stems per selected piece).
`NATIVE_MAX_PIECES = None` selects the full pinned author-mirror test split;
change `NATIVE_RUN_TAG` whenever changing settings or code.

The original WAVs, clean score MIDI, and three authored label MIDIs are used
unchanged on disk. Attune resamples audio to 44.1 kHz to preserve its calibrated
4096-sample window / 128-sample hop (92.9 / 2.9 ms). The notebook defaults to
`range_policy="performed_notes"`: the union of correct+extra label pitch extrema,
padded ±4 semitones like the earlier injected benchmark. Missed notes are excluded.
This is oracle-assisted range selection; label times/identities do not enter inference.
The runner also supports `range_policy="fixed"` (its API default) for ablations.
No local error injection, resynthesis, truth-boundary correction, or truth timing fit is used.
Dataset revisions, checksummed official splits, file hashes, model hashes,
packages, source code and settings are recorded. Only individual Coco instrument stems are evaluated; ensemble mixes are excluded.

Both models use official Coco weights; existing checkpoints are reused. Install the
existing isolated environments with `setup_polytune` / `setup_laddersym` first.
Native LadderSym runs enable contiguous inference, as in the upstream evaluation
commands; existing synthetic runs retain their original handler default.
Native Attune cases run in parallel spawned processes. After they exit, a RAM-bounded
persistent pool processes all PolyTune cases, then all LadderSym cases. Each worker
loads its model once per method and reuses it across cases. The notebook exposes
`NATIVE_WORKERS` and `NATIVE_NEURAL_WORKERS` (both default to automatic sizing).
The earlier cells' compact progress renderer shows completed/total, method, stem/input,
and stage worker counts on one updating line, e.g.
`[  17/36] PolyTune: 239244/0_flute audio`. The total includes every case/method
evaluation and cache hit: 12 stems × three methods = 36. Worker chatter stays in logs.
Changing worker counts does not invalidate results. The new frontend requires a fresh
run folder. `reuse_audio_from` can reuse only PolyTune/LadderSym predictions from a
prior run after verifying asset hashes, model identities, packages and adapter code;
metrics are recomputed and the source checkpoint hash/path recorded. Attune is never
reused across frontend changes. Absent source runs/cases simply run fresh inference.
Each completed case/method is saved immediately; failures stop without zero rows.

Outputs under `benchmarks/results/<NATIVE_RUN_TAG>/coco` include raw
semantic events, authored truth, per-case metrics, logs, resumable checkpoints,
`rows.csv`, `summary.csv`, and `run.json`. Matching runs resume; changes fail
explicitly instead of overwriting historical output. Models are preflighted on
resume, but completed cases do not run inference again.

Reporting distinguishes common pooled missed/extra F1 at 50/100/200 ms from
native-style per-case macro extra/missed/correct F1 (50 ms, 50 cents, no offsets,
mir_eval, empty class F1=0). Semantic JSON avoids MIDI empty-track renumbering.
The latter is not a literal call to the upstream evaluator or its Coco
instrument-macro aggregation. The pinned Coco mirror describes itself as the
LadderSym training subset; equivalence to PolyTune's original Globus release is
not established. Neither default small-subset results nor modified aggregation
should be called a reproduction of published headline scores.

Validation fixtures live in `benchmarks/modules/mistake/tests/NativeComparisonTest.py`. Real downloaded native
inputs and any inference smoke outputs are kept under the existing ignored
`benchmarks/datasets/native_errors/` and `benchmarks/results/` directories.

The cell immediately after the native run now shows `NativeReport.overview()`:
all-case mean error F1 at 100 ms, pooled error F1 at 100 ms, and the separate
native-style three-class mean at 50 ms, all in percent, plus coverage and pooled
TP/FP/FN. For the all-case mean only, an error-free case with no predictions counts
as F1=1; its raw undefined event F1 is unchanged. Incomplete methods have blank
aggregate scores. The reporting helper reads saved rows without model inference.
It is excluded from inference fingerprints, so presentation edits retain cached
results. The notebook explains per-class scores and contains a diagnostic snapshot
of the preserved `native_v1_seed0_pieces3` run. The updated
`native_v2_44100_performed_range_seed0_pieces3` run has Attune pooled error F1 48.9%
(11 TP / 12 FP / 11 FN), up from 18.4% (8 / 57 / 14), and all-case mean F1 45.2%.
Both frontend changes were applied together; this is not a tuning promotion or paper reproduction.


Native v3 (`native_v3_timestamps_pitch05_min50_seed0_pieces3`) uses a 0.5-semitone
pitch threshold and caps the score-relative segment minimum at 50 ms. This is a
native benchmark override; app defaults are unchanged. Wrong-note substitution
missed/extra events share the detected replacement onset/end, matching the author
convention without consulting label times. Pure deletions keep original score time.
Audio-based onset fitting, robust refitting and repeat refinement remain enabled;
truth-based first/last duration correction is not used. Similar-pitch consolidation
still applies, so shorter minimum duration alone need not resolve repeated pitches.
The 12-stem v3 run gives 16 TP / 27 FP / 6 FN, 49.2% pooled F1 and 37.7% mean-case F1.
This combined change recovers five events versus v2 but adds fifteen false positives.
The notebook displays v3 and keeps v1/v2 historical results; neural predictions are
verified and reused with zero model reloads.


September 29 boundary-policy correction: all live injected mistake comparison and
stage-audit paths now preserve detected boundaries. Performed-MIDI first/last
correction was removed; `analyze_recording` ignores its legacy `trim_reference`
argument, and trimmed/unmarked note caches are regenerated rather than reused.
Completed historical audio runs with the old boundary policy cannot silently be
replayed by NotebookRun. The notebook targets a fresh full-comparison output folder.
Ordinary audio-based score fitting remains enabled.

A fresh Attune-only replay at `mistake_competitors_coco_unassisted_2026-09-29`
reran 26 injected cases and 26 clean seed instances (13 unique controls) in eight
workers. All MIDI/audio files match historical inputs byte-for-byte. Injected
100 ms error F1 remains 95.3% (71 TP / 4 FP / 3 FN), with zero clean false alarms.
The corrected policy therefore does not change that headline on this sample.
Native `native_v4_no_boundary_assistance_seed0_pieces3` reproduces v3's 49.2%
(16 / 27 / 6) under the current source provenance, with native 0.5-semitone and
50 ms overrides, native substitution timestamps, and zero neural-model reloads.


Cache reuse on user-triggered Run All:
- `CaseAssetReuse` copies compatible generated MIDI/audio/truth and available pitch
  caches into new result folders; copies cannot mutate historical files. Generation
  identity retains source hash, seed, error policy and injector hash while ignoring
  the audited orchestration/boundary-only changes. Range stamps and PitchCache format
  versions remain validated. Old corrected note caches are never copied.
- `PolyTuneReuse` also serves LadderSym, checking byte-identical performance audio,
  score audio, and (for LadderSym) clean MIDI prompts, plus worker hashes/model identity.
  Missing or incompatible predictions alone cause a model inference request.
- Native Attune caches pre-segmentation pitches under `_native_pitch_cache`, keyed by
  audio, frontend configuration/code, resampler and numerical package versions. A
  separate `_native_note_cache` preserves unrefined notes with segmentation settings.
  Pitch-error threshold changes reuse both stages; minimum-length changes redo only
  notes and downstream alignment. Cached notes replay the same inferred score timing
  setup as `detect_notes`, without consulting labels. Matching completed checkpoints
  still bypass all stages. Exact audited cache-only hash upgrades preserve v4 results.
- Existing native event-only checkpoints cannot reconstruct missing pitch/note stages;
  these are first saved when the user next requests an affected analysis.

No benchmarks were launched after the user requested code-only cache work. Validation
uses small temporary cache fixtures/mocks and read-only provenance comparisons.


The configured native minimum-length cap is now **100 ms** (`min_note_length_cap=0.100`),
with shorter score-derived minima still allowed. The notebook targets
`native_v5_min100_seed0_pieces3`, preserving the historical 50 ms outputs and reusing
compatible pitch caches and neural predictions. Segmentation and downstream alignment
are invalidated by the setting change. No 100 ms benchmark has been run.


LadderSym cache fix: the exact worker revision that only adds the optional
`--contiguous-inference` flag is compatible with the older default-mode worker.
This exception never crosses inference modes. Completed atomic prediction files
can be reused from interrupted/running parent runs; malformed partial files are
skipped. Progress explicitly labels reused audio results `(cached)`. A read-only
check found matching audio, score audio and MIDI prompt for all 52 cases in the
current full injected run. No inference was run to validate this fix.


Report pairing uses source/seed/rate, verifying original score hashes, performed-MIDI
bytes and net error labels when cached symbolic and fresh detected case IDs differ.
Implementation hashes in case IDs no longer prevent reporting on identical inputs.
Changed MIDI/labels or unmatched selections still fail explicitly. The notebook
reloads ResultOverview before generating the table in an existing kernel.
