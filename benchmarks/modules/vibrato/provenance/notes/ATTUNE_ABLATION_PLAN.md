# Attune vibrato implementation checklist

> Historical experiment plan retained as scientific provenance. Current runs
> use `VibratoBenchmarker.py` and `VibratoNotebook.py`.

Work through this list in order and change one mechanism at a time. The goal is
a simple real-time pipeline: one score-aware pitch tracker followed by one
continuous vibrato model whose center, rate, and width may all vary within a
note.

Do not add constant-versus-changing model selection, vibrato templates, or a
bank of separately selected trajectory models. Constant vibrato should emerge
from the same model as the zero-slope case of changing vibrato.

## Experiment status: do not repeat without new evidence

This is the authoritative short status list. A rejected experiment remains
rejected even when its benchmark switch is retained for reproducibility.

- **Pitch-stage comparison — completed, no change:** raw pYIN scored `0.798`
  overall / `0.688` extent F1; HMM-smoothed pYIN scored `0.813` / `0.725`.
  Retain the production HMM.
- **Fine one-period pYIN window — completed preliminary, rejected:** the old
  512/768-sample policy regressed three of four instrument strata and created a
  full-note tuba false alarm. It remains historical evidence, not the current
  automatic policy. The notebook now tests four guarded periods on a
  1024/2048/4096/... power-of-two ladder and audits the resulting distribution.
- **Lower phase-curvature regularization — completed, rejected:** it improves
  exact and audio rate turns, but the settings that materially help turns raise
  negative-control false alarms. Retain `vib2_phase_smoothness=20`.
- **Width regularization and fitted edge values — completed, rejected:** both
  exact-contour gains failed the audio/control safeguards. Retain
  `vib2_width_smoothness=1` and the two-cycle edge-value hold. A later
  four-order-of-magnitude sweep confirmed `vib2_width_smoothness` is already at
  its optimum: the curve is flat over `0.1--1.0` and falls off on both sides,
  and the best row is worth `+0.001` extent F1. Do not revisit it.
- **Center/width knot spacing — completed, adopted:** `vib2_curve_sec` is the
  step-3 control the width penalty was mistaken for. Four values were run on
  the changing tier; extent F1 rose monotonically from `0.4429` at the old
  `0.4` to `0.5014` at `0.1`, rate F1 from `0.8858` to `0.9058`, and straight-
  note false alarms fell from `0.1194` to `0.0937`. `Config.vib2_curve_sec` is
  now `0.1`. Offline throughput roughly halves; live analysis is unchanged,
  since the causal basis has no interior knots. This is not a reversal of the
  rejected center-only entry below: that was a separate center-specific
  mechanism, since removed, whereas `vib2_curve_sec` is the one shared spacing
  the existing center and width bases already use. Still outstanding: confirmed
  constant/control and Yang runs (replay-only so far, where `0.15` is within
  noise of `0.1`), and the frozen held-out validation below.
- **Five-cent HMM and both jump-aware HMM variants — completed, rejected:** none
  improved vibrato without an ordinary-pitch or false-voicing regression.
- **Center-only smoothness/knot changes — completed, rejected:** neither
  mechanism reached the constant target and both were removed.

Still not completed: the onset-confidence taper run, a successful independent
straight-note rejection rule, and frozen balanced final validation. Do not
describe these as tried. Note that every suite summary cached before the
`vib2_curve_sec` change was produced at `0.4`; re-run with
`force_preliminary_rerun=True` before quoting Attune numbers from them.

## Success criteria

Use the preliminary Coco subset for development only. Before changing
production defaults, freeze a held-out Coco subset and retain the independently
annotated Yang recordings as a separate real-audio check.

For each proposed production change, report:

- constant, widening/narrowing, accelerating/decelerating, and rate-turn
  results separately;
- straight-pitch false alarms, including attacks, pitch bends, portamento, and
  center wander;
- rate, one-sided extent, and center bias as well as tolerance-matched F1;
- ordinary pitch and downstream note-detection accuracy, split out for the
  lowest-register instruments;
- offline throughput and live algorithmic latency.

The working target for the clean constant-vibrato suite is at least `0.95`
extent F1 and `0.95` rate F1 without increasing the current straight-pitch
false-alarm rate. A pitch-front-end change should not reduce ordinary overall or
raw pitch accuracy by more than `0.5` percentage points, either overall or on
the lowest-register evaluation stratum.

## Completed diagnostics

- [x] Confirm the exact injected pitch contour gives Attune `1.000` overall,
  extent, rate, and detection F1 with no false alarms.
- [x] Confirm the adaptive-window Coco run actually used `w1=1024` for all 40
  constant/straight cases.
- [x] Measure the fixed-window improvement: changing pYIN from 4096 to 1024
  samples raised Attune overall F1 from `0.438` to `0.813` and extent F1 from
  `0.004` to `0.725`.
- [x] Compare pitch stages. Raw pYIN produced `0.798` overall / `0.688` extent
  F1; HMM-smoothed pYIN produced `0.813` overall / `0.725` extent F1. Keep the
  HMM unless a later controlled test beats it.
- [x] Measure constant-contour attenuation. The injected one-sided amplitude is
  `0.500` semitones, a direct sinusoidal fit to smoothed pYIN averages `0.483`,
  and Attune's median fitted amplitude averages `0.463`.
- [x] Identify center/extent coupling as a likely downstream issue. Across the
  20 clean constant notes, extent MAE and center MAE correlate at approximately
  `0.78`.

## 1. Fix constant-vibrato reconstruction

Do not proceed to changing-width or changing-rate tuning until this section
meets the clean constant target or a documented front-end accuracy limit is
demonstrated.

### 1.1 Add stage-level benchmark output

- [x] Record, per note, the commanded contour, raw pYIN contour, smoothed pYIN
  contour, and final Attune center/rate/width curves in one diagnostic artifact.
- [x] Report pitch-stage amplitude gain, rate error, center error, dropout, and
  octave errors before reporting the final vibrato score.
- [x] Add note-macro results alongside frame-weighted results so long notes do
  not hide short-note failures.

Implemented in the canonical run output: Attune's `frames.csv` now contains the
commanded, raw-pYIN, smoothed-pYIN, and final fitted curves; `cases.csv` contains
the per-note pitch-stage diagnostics; and `note_macro.csv` plus
`by_scenario_note_macro.csv` accompany the existing frame-weighted summaries.

Stop condition: every failed constant note can be assigned to the pitch contour,
the continuous vibrato fit, or the final detection gate without relying on
visual guesswork.

### 1.2 Test finer score-adaptive YIN windows — one-period policy rejected

The existing `1024 * n` rule selected 1024 for every preliminary Coco stem. It
therefore demonstrated that 1024 is much better than 4096, but did not test
finer adaptation between instruments.

The current follow-up is deliberately different: it retains four periods at
the guarded low bound and rounds upward to a power of two, with a 1024 minimum.
On the frozen 20-stem selection this predicts 13 windows at 1024, five at 2048,
and low bassoon/tuba windows at 4096. The notebook verifies those choices and
the shared ±8-semitone autocorrelation/HMM range before results are interpreted.

- [x] Add a benchmark-only policy that chooses the shortest multiple of 256
  samples, with a 512-sample minimum, containing at least one period at the
  guarded lower frequency.
- [x] Use the same lower-frequency guard as the actual pitch detector rather
  than maintaining a separate benchmark-only padding convention.
- [x] Compare the fine-adaptive 512/768-sample choices against the accepted
  fixed-1024 preliminary baseline. A redundant exhaustive fixed-window sweep
  was not continued after the adaptive policy failed the first safeguard.
- [x] Run constant vibrato first and measure its resulting parameter accuracy
  before running the full pitch and note benchmarks.
- [ ] Check ordinary pitch accuracy, voicing, low-register accuracy, note
  detection, throughput, and live latency for the best candidate.
- [ ] If it passes the success criteria, replace the benchmark-only rule with
  one shared production helper and use it in both offline and live pYIN.

Decision:

- [ ] Adopt the simplest passing adaptive-window policy.
- [x] If no shorter valid window improves constant reconstruction without a
  pitch regression, retain fixed-1024 pYIN and move to section 1.3. Do not
  keep adding window variants.

Initial rejection evidence (first song / four instrument strata): the fine
policy selected 512 samples for trumpet, horn, and trombone and 768 for tuba.
It improved the trumpet constant note, but regressed horn, trombone, and tuba
and created a full-note tuba straight-pitch false alarm. Across these four
constant/control pairs, overall F1 fell from the 1024-window baseline to
`0.629`, extent F1 to `0.523`, rate F1 to `0.734`, and straight-pitch false
alarms rose to `0.208`. The policy remains available as an explicit benchmark
experiment, while the preliminary notebook retains fixed 1024 and validates
that it covers the production-guarded lower period for every selected stem.

### 1.3 Check HMM pitch resolution only if the contour remains limiting

Attune currently quantizes the smoothed track to 0.1-semitone bins while the
constant extent metric uses a 0.05-semitone tolerance. HMM smoothing is useful,
so test its resolution rather than removing it.

- [x] Compare 10-cent and 5-cent HMM bins using the selected YIN window, keeping
  the maximum permitted pitch movement in cents unchanged.
- [ ] Measure constant extent, ordinary pitch/voicing accuracy, memory,
  throughput, and live latency.
- [x] Adopt 5-cent bins only if they materially improve contour and vibrato
  accuracy while remaining comfortably real time; otherwise retain 10 cents.

Rejected on the 20-stem preliminary constant/control corpus using the same
cached raw 1024-sample pYIN tracks. Five-cent decoding changed overall F1 from
`0.813` to `0.799`, extent F1 from `0.725` to `0.722`, rate F1 from `0.901` to
`0.877`, and straight-pitch false alarms from `0.087` to `0.173`. Retain the
10-cent HMM; the experimental override was removed.

#### 1.3a Test a rare HMM new-note jump path

The constant-case stage diagnostics show that the compact 90-cent/frame HMM
kernel forces multi-frame glides after larger melodic intervals. Keep the
production HMM unchanged while testing this transition policy separately.

- [x] Implement `PitchSmoother2` v1 with the original local kernel plus a 0.1%
  broad Laplace jump path and an optional unvoiced-to-voiced pitch restart.
- [x] Give every jump-policy pitch cache and vibrato result a distinct name.
- [x] Compare original and jump-aware smoothing on the same raw pYIN tracks
  from a quick head of uninjected CocoChorales stems.
- [x] Run the 20-stem preliminary constant/straight vibrato suite with the jump
  smoother and compare onset error by melodic interval, extent/rate F1, and
  straight-note false alarms.
- [x] Reject v1: base raw-pitch accuracy changed by only `+0.0006` while
  voicing false alarms increased by `0.0052`; constant vibrato F1 fell from
  `0.8364` to `0.8259` and extent F1 from `0.7459` to `0.7252`. Every frame
  voiced by both policies retained the same decoded pitch; the harmful changes
  were unvoiced-restart decisions that exposed preceding-note onset frames.
- [x] Implement v2 with uniform restart disabled, a separate broad-jump attack
  state, a 20 ms attack pre-roll flag, and transition exclusion in
  `VibratoDetector2`'s data residual. Parallelize and version the safety check.
- [x] Run the version-2 base-pitch and preliminary constant/control cells.
- [x] Adopt nothing in production unless ordinary pitch accuracy does not
  decline and the vibrato improvement survives the complete preliminary run.

Version 2 improved constant frame-weighted F1 from `0.8364` to `0.8641`, extent
F1 from `0.7459` to `0.7756`, and rate F1 from `0.9270` to `0.9527`. However,
the ordinary-pitch safety head changed overall accuracy by `-0.0002` and
increased voicing false alarms by `0.0107`. Only 17 of 10,613 scored pitch
frames changed, and transition flags reached just 6 of 20 constant notes. Keep
the production HMM unchanged and do not promote the v2 jump path.

#### 1.3b Taper confidence at corrected note onsets

The residual onset error remains concentrated in the first few frames after a
score-corrected note boundary. Test a downstream note-aware confidence policy
without changing pYIN or either HMM.

- [x] Add an optional raised-cosine onset confidence ramp to every data term in
  Attune's rate seed, center/width solve, nonlinear phase fit, and fit-quality
  calculation.
- [x] Keep every voiced frame in the fit with strictly positive confidence.
  Bound the tapered span by both 20 ms and 25% of the complete note; use a 0.25
  first-frame confidence for the preliminary experiment.
- [x] Record exact per-frame confidence plus per-note minimum, mean, count, and
  fraction diagnostics in the benchmark artifacts.
- [x] Keep the production default disabled and give the preliminary taper run
  a distinct result directory while reusing the original-HMM pitch cache.
- [ ] Run the notebook's taper invariant and preliminary average-case cells.
- [ ] Compare constant extent/rate/center metrics and straight-pitch false
  alarms directly with `preliminary_average_w1024`; adopt only if the gain is
  consistent across notes and controls do not regress.

### 1.4 Prevent the center curve from absorbing vibrato

Run this only after selecting the pitch-front-end settings. Keep one continuous
center/rate/width fit.

- [x] Sweep the existing `vib2_center_smoothness` over a small, declared range.
- [ ] For each value, report constant extent error, center error, and results on
  genuine center drift, pitch bends, and portamento.
- [x] If the existing control cannot separate slow center movement from 3--10
  Hz vibrato, give the center its own coarser spline spacing. Do not change the
  rate or width parameterization in the same experiment.
- [ ] Select the weakest center constraint that reaches the constant target
  without degrading legitimate slow center movement.

No tested center-only change reached the target. On the 20-stem preliminary
constant/control corpus, `vib2_center_smoothness` values
`50, 200, 800, 3200, 12800, 51200` peaked at overall F1 `0.830` (value 50);
stronger constraints slightly worsened extent and raised false alarms. Giving
the center a separate `0.6--2.4 s` knot spacing peaked at overall F1 `0.816`
(`0.6 s`) versus `0.813` at the shared `0.4 s` spacing. Both rejected
mechanisms were removed rather than promoted to production.

### 1.5 Resolve straight-pitch false alarms with existing evidence first

- [x] Inspect the preliminary full-note false positives in the rendered audio
  and raw/smoothed contours. In the expanded changing run, six of 20 straight
  notes repeat as full-note false positives at clean, 20 dB, and 10 dB. The
  stable identities across SNR point to deterministic within-note pYIN contour
  structure rather than additive noise; Driedger does not fire on the same
  unbent rendered audio.
- [ ] Re-evaluate the existing RMS-ratio and sustained-width gates after the
  window, HMM, and center changes.
- [ ] Adjust an existing gate only if it removes false positives without losing
  narrow real Yang vibrato. Do not solve this by simply raising the global width
  floor.

Checkpoint:

- [ ] Constant-vibrato and straight-pitch success criteria pass on development
  data.
- [ ] The chosen settings pass the frozen constant/straight held-out subset.

## 2. Recover changing rate

Keep the accepted pitch, center, width, and edge settings fixed.

### 2.1 Monotonic acceleration and deceleration

- [x] Establish exact-contour and audio/pYIN baselines for accelerating and
  decelerating notes separately.
- [x] Inspect failed notes for wrong cycle count, quality-gate rejection, or
  biased input contour before changing the optimizer.
- [x] Tune existing rate controls only when the diagnostic identifies their
  failure mode. Do not add multiple trajectory-specific seeds unless repeated
  wrong-cycle convergence is demonstrated.
- [x] Run the frozen constant and straight subsets before accepting a rate
  change.

### 2.2 Within-note rate turns

The current `vib2_phase_smoothness` penalty directly discourages curvature in
the three-control rate curve, so rate turns are the last targeted case.

- [x] Sweep `vib2_phase_smoothness` from the accepted value down through a
  small set including zero.
- [x] Report rate-turn accuracy separately from monotonic rate change and
  constant rate.
- [ ] Select the largest penalty that preserves genuine turns; this retains as
  much noise resistance as the data supports.
- [x] Re-run all prior checkpoints before accepting the value.

Checkpoint:

- [ ] Accelerating, decelerating, and rate-turn cases improve without regressing
  constant vibrato or false alarms.

Decision (2026-08-22): retain `vib2_phase_smoothness=20`. The exact-contour
baseline already reconstructs monotonic acceleration and deceleration well
(rate F1 `0.969` and `0.981`), while the new Coco baseline reaches `0.604` and
`0.634`. Across the audio cases, the vibrato-input pitch contour has about
`9--11` cents median MAE, zero median dropout and octave-error fractions, and
small median bias; the remaining failure is not a gross pitch-track or
quality-gate rejection.

The declared phase-penalty sweep was `20, 5, 1, 0.1, 0.01, 0.001, 0` on exact
contours and `20, 5, 1, 0.1, 0.01, 0` on cached Coco audio. Removing the
penalty raises exact rate-turn F1 from `0.503` to `0.821` and the dedicated
Coco rate-turn F1 from `0.451` to `0.564`, confirming that curvature
regularization suppresses real turns. No candidate passes the checkpoint,
however: `0.1` preserves the old control false-alarm rate but leaves turn F1
at `0.529` on audio (`0.608` exact), while `0.01` and zero improve turns but
raise the noisy changing-suite false-alarm rate from `0.112` to `0.123`.
Consequently the mechanism is diagnosed but no production rate change is
accepted.

## 3. Recover changing width and revisit endpoints

Keep the accepted constant and changing-rate settings fixed.

- [x] Establish exact-contour and audio/pYIN baselines for widening and
  narrowing separately.
- [x] Sweep only `vib2_width_smoothness`; select the weakest regularization that
  suppresses contour jitter while preserving the injected width trajectory.
- [x] Re-test `_stabilize_offline_edges()` across changing-rate and
  changing-width cases: compare the current two-cycle value hold with retaining
  fitted edge values while keeping the quality taper.
- [x] Remove the edge value hold only if changing contours improve without
  regressing constant extent or straight-pitch false alarms. The earlier small
  Coco diagnostic found that removing it prematurely made constant performance
  worse.
- [x] Run the frozen constant, straight, and changing-rate subsets before
  accepting any width or edge change.

Checkpoint:

- [ ] Widening and narrowing improve on both exact contours and Coco audio.
- [ ] Constant, changing-rate, and negative-control success criteria still
  pass.

Decision (2026-08-22): retain `vib2_width_smoothness=1` and the two-cycle edge
value hold. The width sweep `0, 0.1, 1, 10` slightly favors `0.1` on exact
contours, but the same value lowers the cached changing-audio overall F1 from
`0.661` to `0.642` and raises false alarms from `0.112` to `0.133`. Zero is
substantially worse and `10` raises false alarms to `0.172`; production `1` is
the only balanced result.

Retaining fitted edge values is the clearest exact-contour improvement:
widening extent F1 rises from `0.944` to `0.988` and narrowing from `0.943` to
`1.000`. It fails both audio safeguards. Changing-audio false alarms rise from
`0.112` to `0.170`, and the independent constant/straight suite falls from
`0.813` to `0.749` overall F1 while false alarms rise from `0.087` to `0.180`.
The fitted-edge policy remains an explicit benchmark option, with the same
quality taper, but is rejected for production.

### Replacement decision after steps 2--3

Initial decision before the sampling audit: keep Attune's whole-note model. The
strongest validated alternative on the
new clean rate-turn suite is McLeod/Tartini Prony (`0.716` overall F1 versus
Attune's `0.602`), and it also leads the older changing-Coco suite (`0.698`
versus `0.661`). That reconstruction advantage comes with control false-alarm
rates of `0.581` on both suites, versus Attune's `0.147` and `0.112`. Attune
also leads on clean constant/control audio (`0.813` versus `0.674`, with
`0.087` versus `0.557` false alarms) and on the independently annotated Yang
real-audio parameter subset (`0.649` versus `0.463`). Driedger's opt-in method
is not a replacement candidate because its independent port still misses the
released-dataset acceptance gate at -10 dB. The evidence therefore supports
Attune for production while identifying within-note rate turns as its main
remaining weakness.

Sampling-audit revision (2026-08-22): the later preliminary changing-profile
run put Driedger at `0.702` overall F1 and Attune at `0.542`, but that run used
the first 20 sorted manifest rows (all brass) and fixed 6 Hz / 0.50-semitone or
0.50-semitone changing-rate anchors that align unusually closely with
Driedger's template grid. That ranking is diagnostic, not a final replacement
decision. The corrected balanced/randomized run below must supersede it.

## Benchmark sampling correction before final validation

- [x] Replace sorted-manifest truncation with seeded selection whose ensemble
  and instrument quotas are fixed from manifest proportions. Randomness only
  chooses the concrete track/stem within each `(ensemble, instrument)` stratum.
  For a 20-stem test split this gives five brass, five random, five string, and
  five woodwind stems for every seed. Selected record identities are persisted
  in `run_config.json`.
- [x] Replace fixed injected anchors with stable per-note/profile draws derived
  from the global seed plus track, note index, and profile. Constant rates draw
  uniformly from 3--10 Hz and constant one-sided amplitudes from 0.15--1.00
  semitones.
- [x] Keep changing profiles human-plausible: rate spans draw from 1--3 Hz and
  one-sided amplitude spans from 0.15--0.50 semitones, with the lower anchor
  chosen so the whole curve remains inside product support.
- [x] Persist the sampler version, per-note seed, scalar curve anchors, exact
  sampled arrays, global selection policy, and selected record list. Namespace
  rendered audio and pitch caches by sampler configuration and seed.
- [x] Update `vibrato.ipynb` to describe and invoke the corrected corpus. Its
  default preliminary cells use sampling-specific result/artifact names, the
  four-period automatic YIN-window audit, and no rejected onset-taper,
  lower-curvature, jump-HMM, or fitted-edge ablation. The setup, selection/
  parameter invariant, and generated average/changing commands pass a notebook
  smoke check.
- [ ] Run the corrected balanced preliminary suite. No result from the earlier
  sorted-head/fixed-anchor corpus should be treated as the final replacement
  comparison.

## 4. Final validation and production promotion

- [ ] Freeze all selected settings before opening the final held-out results.
- [ ] Run the full constant, changing-width, changing-rate, negative-control,
  noisy, dropout, and outlier synthetic/Coco suites.
- [ ] Run the independently annotated Yang real-audio evaluation.
- [ ] Run the ordinary pitch and note benchmarks, including the lowest-register
  strata.
- [ ] Verify offline throughput and live latency on CPU.
- [ ] Promote only the changes that pass all earlier checkpoints.
- [ ] Record rejected alternatives and their measured regression briefly below
  the relevant checklist item; do not leave rejected mechanisms in production
  code.
