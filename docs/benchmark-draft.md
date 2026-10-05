# Experimental Results and Discussion
We evaluate Attune's performance on both component and system-wide levels.

Our component experiments benchmark Attune's performance to research baselines in the following areas:
1. Pitch detection
2. Note detection
3. Vibrato detection
4. Mistake detection

System-wide experiments assess end-to-end performance error detection, and are conducted quantitatively against leading research systems and qualitatively against commercial music performance analysis software.

The quantitative experiments assess performance on the following annotated music datasets:
- **Bach10:** Ten recorded chorales, each with separate violin, clarinet, saxophone, and bassoon.
- **University of Rochester Multi-Modal Performance (URMP) Dataset:** 44 recorded ensemble pieces, covering 13 string string, woodwind, and brass instruments. Annotations and recordings span 4.6 hours of music across 149 isolated instrument tracks.
- **CocoChorales:** Synthesized isolated instrument stems covering the same 13 instruments as the URMP dataset. Covers 5,644 hours of isolated instrument tracks. 
- **Yang et. al:** 76 excerpts of strings, woodwinds, brass, and voice, with annotations on vibrato rate and extent.

The following table summarizes the datasets used in each of the experiments described here. 

| Dataset            | Pitch detection | Note detection | Vibrato detection and parameter estimation | Mistake detection |
| ------------------ | :-------------: | :------------: | :----------------------------------------: | :---------------: |
| Bach10             |        ✓        |       ✓        |                     —                      |         —         |
| URMP               |        ✓        |       ✓        |                     —                      |         —         |
| CocoChorales       |        ✓        |       ✓        |                     ✓*                     |        ✓*         |
| Yang et. al (2017) |        —        |       —        |                     ✓                      |         —         |

Table X. Dataset inclusion across the different benchmark components. The (`*`) denotes where we modify CocoChorales to include locally injected vibrato and mistakes for testing.

## Component Analysis
### Pitch Detection
We evaluate pitch detection using the standard [MIREX melody-extraction metrics](https://www.music-ir.org/mirex/wiki/2005%3AAudio_Melody_Extraction_Results) to measure overall accuracy, raw pitch accuracy, raw chroma accuracy, voicing recall, and voicing false-alarm rate. Metrics are computed using `mir_eval` with a 50-cent pitch tolerance, pooling frame counts across recordings and datasets. 

All methods receive the frequency range of the reference score to reduce octave errors. This is a convenience which is also feasible in practice, as Attune always records audio in reference to a music score.

**Offline Pitch Detection:** We compare Attune with seven state-of-the-art pitch detection methods which analyze complete recordings. Benchmarks are run on 6.5 total hours of music, including all tracks in the Bach10 and URMP datasets, and a representative random subset of the CocoChorales dataset.

| Method | Overall accuracy | Raw pitch accuracy | Raw chroma accuracy | Voicing recall | Voicing false-alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attune | **93.68** | 94.95 | 95.42 | 97.07 | 11.67 |
| pYIN | 87.71 | **96.07** | **96.56** | **98.57** | 47.45 |
| Praat | 93.04 | 93.82 | 94.82 | 96.52 | **10.26** |
| Basic Pitch | 89.37 | 92.07 | 92.31 | 98.07 | 21.97 |
| SwiftF0 | 85.69 | 87.08 | 87.31 | 89.14 | 20.17 |
| CREPE | 92.68 | 94.07 | 94.36 | 95.72 | 13.13 |
| SPICE | 86.56 | 89.90 | 90.13 | 95.04 | 27.51 |
| RMVPE | 89.28 | 92.59 | 93.93 | 96.89 | 24.62 |

Table X. Offline pitch-detection metrics (%) pooled across tracks from Bach10, URMP, and CocoChorales.

Attune achieved the highest overall accuracy of 93.68%, reflecting its balance across all of the statistics. Paired permutation tests across all of the recordings shows Attune has significantly higher overall accuracy than six of the seven baselines (Holm-corrected $p_{\text{adj}} \leq 0.014$), with Attune's 0.65% advantage over Praat having $p_{\text{adj}}=0.115$.

**Online Pitch Detection:** An important factor in Attune's final model choice was its ability for accurate, low-latency pitch feedback. After excluding models like CREPE which were too slow to be run in real time, or Basic Pitch which required 5 seconds of look-ahead, we benchmarked Attune against four streaming implementations on the same datasets as the offline experiments. 

Among the examined baselines, we included a version of the original PYIN algorithm without the post-hoc Hidden Markov Model pitch smoothing as an ablation against the changes Attune made. Audio look-ahead and update intervals were set at 46.44ms / 2.90ms for Attune, streaming PYIN, and Praat; 152.03ms/16ms for SwiftF0; and 32ms/32ms for SPICE.

| Method        | Overall accuracy | Raw pitch accuracy | Raw chroma accuracy | Voicing recall | Voicing false-alarm rate | P95 latency (ms) |
| ------------- | ---------------: | -----------------: | ------------------: | -------------: | -----------------------: | ---------------: |
| Attune        |        **93.13** |              94.44 |               95.08 |          96.91 |                    12.39 |            47.30 |
| pYIN (no HMM) |            77.09 |          **96.03** |           **96.89** |      **99.94** |                    85.25 |            47.98 |
| Praat         |            85.23 |              93.28 |               95.37 |          99.82 |                    48.66 |            48.14 |
| SwiftF0       |            85.62 |              86.75 |               86.97 |          88.77 |                    19.14 |           163.84 |
| SPICE         |            82.06 |              84.61 |               85.80 |          90.92 |                    28.67 |        **37.02** |

Table Y. Online pitch-detection metrics (%) pooled across Bach10, URMP, and CocoChorales. P95 latency measures the duration which 95% of requests are processed faster than.

Attune achieved **93.13% overall accuracy**, exceeding all four baselines by 5.59-16.04 percentage points, all statistically significant with $p_{\text{adj}}\leq_{0}.0015$. In particular, Attune's low voicing false-alarm rate compared to the baselines contributes to much of the overall accuracy difference. This is particularly important for real-time feedback, where spurious pitch estimates during rests or pauses due to background noise can produce misleading visual cues or incorrectly advance pitch-gated practice in Attune's *supervised* mode.

### Note Detection
We evaluate note detection using the standard [MIREX note-transcription criteria](https://mir-eval.readthedocs.io/latest/api/transcription.html) to measure precision, recall, and F1 for onset-and-pitch matching, both with and without offset matching. Metrics are computed using `mir_eval.transcription` with its default tolerances: 50 ms for onsets, 50 cents for pitch, and the larger of 50 ms or 20% of the reference-note duration for offsets. The primary metric ignores offsets; offset-aware F1 additionally requires offset matching. Metrics pool matched, extra, and missed notes across recordings and datasets, with one-to-one matching restricted to each recording.

We compared Attune with three audio-to-note systems: Basic Pitch, CREPE Notes, and Tony/pYIN Vamp. All four methods completed evaluation on 60 isolated tracks—eight from Bach10 and 26 each from URMP and CocoChorales—totaling 0.89 hours (3,187 seconds) of audio. Attune used its production pitch frontend, a score-derived minimum note duration, score alignment, timing refitting, and repeat-note recovery. Its pitch range came from reference F0; competitors retained their native pipelines, including simultaneous Basic Pitch predictions. Performed-note annotations supplied evaluation boundaries, without reference-based timing correction. This evaluates score-informed Attune against audio-only competitors.

| Method | Precision | Recall | Onset/pitch F1 | Offset-aware F1 |
| --- | ---: | ---: | ---: | ---: |
| Attune | **89.82** | **89.08** | **89.45** | **86.24** |
| Basic Pitch | 69.79 | 84.96 | 76.63 | 64.03 |
| CREPE Notes | 78.19 | 88.45 | 83.00 | 75.33 |
| Tony/pYIN Vamp | 84.91 | 57.22 | 68.37 | 62.69 |

Table X. Note-detection metrics (%) pooled across Bach10, URMP, and CocoChorales. Precision and recall use onset/pitch matching.

Attune achieved the highest pooled onset/pitch F1 of **89.45%**, exceeding Basic Pitch, CREPE Notes, and Tony/pYIN Vamp by 12.82, 6.45, and 21.08 percentage points, respectively. Two-sided paired permutation tests using 9,999 label swaps across 48 source groups, with repeated URMP titles grouped together, gave Holm-corrected $p_{\text{adj}}=0.0003$, $0.0010$, and $0.0003$, respectively. All three differences were significant at the 0.05 family-wise threshold. Attune also had the highest offset-aware F1; its 3.21-point reduction reflects the additional difficulty of locating note endings accurately. These aggregate results do not establish superiority on every dataset or instrument.

### Vibrato Detection and Parameter Estimation

The vibrato experiments use:

- **CocoChorales constant/control suite:** 20 tracks, with five from each of the brass, random, string, and woodwind ensemble categories. Eligible notes are at least 0.75 s long. Conditions comprise constant vibrato and straight-tone controls.
- **CocoChorales changing-profile suite:** the same 20 tracks with accelerating, decelerating, widening, narrowing, and straight-tone profiles under clean, 20 dB, and 10 dB conditions.
- **Yang real-audio subset:** one erhu and one violin recording from the released parameter-annotation subset, with manual extrema defining reference rate and extent. The full release contains six recordings with the required half-cycle annotations (5.6 min); the other erhu/violin recordings and the area-only collection do not support this parameter-estimation protocol.

For the controlled suites, we inject known pitch-modulation curves into MIDI and render them using a benchmark-specific sfizz straight-tone policy. Constant-vibrato rates are sampled over 3–10 Hz and one-sided extents over 0.15–1.00 semitones. Changing profiles span 1–3 Hz in rate or 0.15–0.50 semitones in extent while remaining within those bounds. Parameter draws are reproducible from the global seed and stable note identity. Rendered audio passes through the common Attune pitch frontend before each vibrato estimator receives its analysis input; scoring crops are withheld from estimators. Competitors are implementations or ports of Rossignol, Herrera–Bonada, Ventura–Sousa–Ferreira, Driedger, Yang AVA/FDM and AVA/Bayes, and McLeod/Tartini. Driedger's original configuration and its explicitly labeled benchmark-range adaptation are reported separately.

We distinguish binary vibrato detection from tolerance-matched parameter estimation. A parameter true positive requires a positive detection and an estimate within 0.5 Hz for rate or 0.05 semitones for one-sided extent. Wrong-valued positive predictions contribute both a false positive and a false negative; positive predictions on straight-tone frames contribute false positives. Overall parameter F1 is the arithmetic mean of rate F1 and extent F1. The real-audio comparison uses a common estimated pitch contour and annotated vibrato spans as pseudo-note boundaries. Consequently, it measures parameter estimation within known regions and does not provide a full false-alarm or note-boundary evaluation. The AVA Bayes model also has training provenance in this corpus.

| Method | Constant/control parameter F1 | Changing-profile parameter F1 | Yang parameter F1 |
| --- | ---: | ---: | ---: |
| Attune | **69.95** | **70.36** | **62.62** |
| Rossignol | 47.34 | 58.12 | 17.12 |
| Herrera–Bonada | 24.50 | 40.75 | 4.43 |
| Ventura–Sousa–Ferreira | 44.92 | 51.82 | 40.10 |
| Driedger | 53.32 | 49.54 | 18.61 |
| Driedger, benchmark range | 59.74 | 69.80 | 38.82 |
| Yang AVA/FDM | 30.93 | 40.16 | 18.23 |
| Yang AVA/Bayes | 15.37 | 17.16 | 28.30 |
| McLeod/Tartini | 55.45 | 62.62 | 44.78 |

| Attune metric | Constant/control | Changing profiles | Yang real audio |
| --- | ---: | ---: | ---: |
| Rate F1 | 89.96 | 90.58 | 67.15 |
| Extent F1 | 49.93 | 50.14 | 58.10 |
| Binary detection F1 | 93.02 | 98.62 | 100.00 |
| Straight-tone frame false-alarm rate | 13.11 | 9.37 | Not assessed |

Attune's largest advantage is in rate estimation, while extent estimation remains substantially less accurate. Its changing-profile aggregate is close to the benchmark-range Driedger adaptation, which has higher extent F1 but predicts vibrato on all scored straight-tone frames in that suite. Attune's corresponding false-alarm rate is 9.37%. The real-audio results support a preliminary parameter comparison, but the supplied region boundaries and absent explicit negative controls prevent interpreting its 100% binary detection F1 as whole-recording detection performance.

## System-wide Evaluation

### Symbolic Mistake Detection

The symbolic experiment uses:

- **CocoChorales source scores:** 13 test-split tracks, one per instrument, selected with seed 0.
- **Injected performances:** two injection seeds per track under a 25% error-injection condition, producing 26 altered cases.
- **Clean controls:** the same tracks evaluated under a zero-injection condition for both seeds, producing 26 clean evaluations over 13 unique tracks.

We construct monophonic performed-note sequences by injecting substitutions, insertions, deletions, shortened notes, and lengthened notes with equal configured weights. Insertions and deletions shift subsequent onsets, and cumulative timing changes are retained. Duration edits are restricted to 0.5–1.5 times the note's original duration and must exceed the configured duration tolerance by at least 50 ms, with a 300 ms minimum in the saved experiment; ineligible edits are skipped. Ground-truth net mistakes are recounted after the edits. Exact notes from the resulting performed MIDI are supplied to Attune's weighted string-editing aligner, Nakamura, and Parangonar Automatic, DualDTW, and TheGlueNote. This experiment evaluates the alignment component directly, without acoustic extraction or repeat splitting.

The primary metric is pooled mistake-event precision, recall, and F1 at 100 ms onset and 50 cents pitch tolerance, with 50 and 200 ms onset gates as secondary evaluations. Substitutions are represented as one missed event and one extra event. These scores measure detected mistake events rather than the fraction of correctly aligned notes. Short/long duration mistakes are evaluated separately and are not included in the pitch-error headline table.

| Method | Precision | Recall | Mistake-event F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | --- |
| Attune alignment | 97.4 | **100.0** | **98.7** | 74 / 2 / 0 |
| Parangonar DualDTW | 97.3 | 97.3 | 97.3 | 72 / 2 / 2 |
| Parangonar Automatic | **98.5** | 90.5 | 94.4 | 67 / 1 / 7 |
| Nakamura | 77.2 | 95.9 | 85.5 | 71 / 21 / 3 |
| Parangonar TheGlueNote | 69.6 | 95.9 | 80.7 | 71 / 31 / 3 |

The saved preliminary comparison shows high event recall for Attune and a small F1 advantage over DualDTW. The sample contains only 74 reference pitch-error events, so small changes in event counts can change the ranking. These measurements use the saved symbolic run and will be replaced or reverified against the frozen implementation before final reporting.

### Full-pipeline Mistake Detection

The full-pipeline experiments use:

- **Locally injected CocoChorales:** the same 13 source tracks, two seeds, and clean/25% injection conditions as the symbolic experiment, rendered with FluidSynth and the MuseScore soundfont while retaining source instrument programs.
- **Native CocoChorales-E:** three deterministically selected test pieces containing 12 isolated instrument tracks, with original author audio, score MIDI, and correct/extra/missed-note labels. The pinned release is the LadderSym subset; equivalence to PolyTune's original dataset release has not been established. The 300+ h figure in the dataset overview describes the published full corpus, not this downloaded subset or the evaluation split.
- **URMP repeat-recovery check:** 10 saved development recordings with repeated-note regions, used as a focused verification of the current production recovery stage rather than an independent mistake-event benchmark.

For locally injected audio, Attune performs pitch extraction, note segmentation, score alignment, robust timing fitting, and production repeat recovery. The current recovery method locally splits eligible matched notes to recover same-pitch score deletions using duration error and deletion-neutral recovery fees; it preserves existing matched boundaries and uses the configured pitch-mistake tolerance for eligibility. External symbolic aligners receive the same extracted notes, while PolyTune and prompted LadderSym run their own audio-based systems. No performed-MIDI boundary correction is applied. We report the same pooled pitch-error event metrics as in the symbolic experiment and count false alarms separately on clean performances. Because the saved audio comparison predates the current recovery method and later frontend changes, its numbers below are explicitly retained as provisional earlier-run values pending a fresh evaluation.

The native CocoChorales-E experiment uses the authors' supplied audio and labels without local error injection or resynthesis. Attune's saved adapter resamples audio to 44.1 kHz, uses the union of correct and extra label pitches with a four-semitone margin for range selection, and applies a 0.5-semitone pitch-error threshold and 100 ms cap on the score-derived segmentation minimum. The range is therefore oracle-assisted, although label timestamps are withheld from inference. PolyTune and LadderSym use their official dataset checkpoints. In addition to pooled missed/extra F1 at 100 ms, we report mean per-case three-class F1 for extra, missed, and correct notes at 50 ms / 50 cents, with offsets ignored. Empty classes score zero for that secondary metric. This explicit semantic-class evaluator is not a literal reproduction of the authors' evaluator or instrument-macro aggregation.

**Native rendering and annotation mismatch.** Overlapping MIDI events do not necessarily correspond to overlapping notes in the supplied native audio. Our renderer audit found that MIDI-DDSP's per-instrument synthesis path converts notes into a sequential duration representation: negative inter-note gaps are not preserved, so overlapping MIDI notes can be rendered consecutively, delaying the following notes and producing effectively monophonic audio. In an inspected violin track, a MIDI overlap of approximately 550 ms coincides with a following-note audio onset delayed by approximately the same amount; a second violin track shows approximately 216 ms of accumulated delay. Cached boundaries and an independent spectral check support this interpretation. FluidSynth preserves the overlapping MIDI event schedule, so rendering the same file through FluidSynth changes both the acoustic realization and, in affected cases, the realized timeline. This is a rendering/timeline issue rather than simply a soundfont-timbre difference. The exact historical native renderer revision has not been recovered, and the prevalence of the mismatch across the full corpus remains unmeasured. We retain official labels and all selected tracks in the native table; its timing-sensitive scores must therefore be interpreted with this confound and cannot be attributed entirely to detector quality.

| Method | Injected-audio precision | Injected-audio recall | Injected-audio F1 | Clean false alarms / 100 notes |
| --- | ---: | ---: | ---: | ---: |
| Attune, saved production run | **94.7** | **95.9** | **95.3** | 0.00 |
| Parangonar DualDTW | 92.2 | **95.9** | 94.0 | 0.00 |
| Parangonar Automatic | 90.4 | 89.2 | 89.8 | 0.00 |
| Nakamura | 76.9 | 94.6 | 84.8 | 0.00 |
| Parangonar TheGlueNote | 65.0 | 90.5 | 75.7 | 1.92 |
| LadderSym | 23.4 | 20.3 | 21.7 | 3.51 |
| PolyTune | 26.8 | 14.9 | 19.1 | 1.28 |

| Method | Native Coco-E tracks | Pooled error F1, 100 ms | Mean three-class F1, 50 ms | TP / FP / FN at 100 ms |
| --- | --- | ---: | ---: | --- |
| Attune, saved native configuration | 12/12 | 60.4 | 55.4 | 16 / 15 / 6 |
| PolyTune | 12/12 | **85.7** | 66.9 | 21 / 6 / 1 |
| LadderSym | 12/12 | 80.0 | **67.4** | 18 / 5 / 4 |

| Current production repeat-recovery check | Recordings | Note F1, 100 ms / 50 cents | TP / FP / FN |
| --- | ---: | ---: | --- |
| Whole recordings | 10 | 94.03 | 748 / 47 / 48 |
| Repeated-note regions | 10 | 91.09 | 271 / 24 / 29 |

The saved locally injected result gives Attune 71 true positives, four false positives, and three false negatives, with no clean-control false alarms. Its F1 is lower than the corresponding exact-note symbolic result, consistent with additional error introduced by acoustic extraction. The native dataset produces a different ranking, with PolyTune and LadderSym ahead under the official-label evaluation. Differences in error structure, synthesis, model training distribution, and the demonstrated audio–label timing mismatch prevent interpreting this reversal as a controlled comparison of timbre robustness alone. The current-production URMP replay verifies repeated-note recovery on saved acoustic note inputs, but its note F1 is not directly comparable to mistake-event F1 and does not constitute a fresh end-to-end run.

## Final Evaluation and Reporting

The full evaluation will replace the provisional tables after freezing the production configuration and completing all selected method–dataset pairs. For note detection, this includes replacing the Bach10 placeholders and the provisional CocoChorales/URMP values with completed runs of all four methods, including the repaired CREPE Notes adapter. We will report completed coverage, separate development measurements from evaluation measurements, and retain clean-control false alarms alongside positive-error performance. Instrument-level breakdowns and paired uncertainty estimates grouped by source piece will accompany the aggregate comparisons where feasible. A larger run will improve coverage, but repeated injection seeds, multiple tracks from one composition, or previously tuned test samples will not be treated as independent evidence merely because the number of evaluated cases increases.

Parameter sweeps, alternative internal configurations, and before/after recovery experiments will appear in the appendix. The main results will retain the selected production method, external competitors, and the methodological qualifications necessary to interpret their inputs and metrics. Native CocoChorales-E results will continue to be identified as official-label measurements with a documented rendering mismatch; diagnostic timeline reconstructions will not silently replace the reference labels or headline scores.

## Appendix A. Pitch Evaluation Details and Preliminary Results

### Aggregation and statistical analysis

The notebook scores every recording using `mir_eval` on a 10 ms grid, preserving annotation timestamps and using its voicing-aware interpolation. Overall accuracy pools all reference frames; raw pitch accuracy, raw chroma accuracy, and voicing recall pool reference-voiced frames; voicing false-alarm rate pools reference-unvoiced frames. Tracks with no eligible frames contribute zero denominator rather than an equally weighted zero score. The scorer saves numerators, denominators, a reference fingerprint, and a scoring-version identifier. Older score rows are recomputed from cached estimates where available; streaming checkpoints without frame counts require replay.

The offline and online paired analyses use separate test families. Each selects the common completed recording set across its specified methods and reports coverage and exclusions. Two-sided paired permutation tests swap method labels within source-piece groups, keeping all stems together and recomputing pooled score differences. Small designs enumerate all label swaps; larger designs use 9,999 random swaps with a plus-one correction. Confidence intervals use 9,999 paired cluster-bootstrap samples, stratified by dataset. Known shared compositions can be linked across datasets through an explicit grouping map. Frames retain equal score weight, but are not treated as independent replicates. The implementation follows the [paired label-swap scheme](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

Holm correction applies separately to seven offline overall-accuracy comparisons and twelve online comparisons (streaming pYIN, Praat, SwiftF0, and SPICE across overall accuracy and both voicing metrics), at a family-wise significance level of 0.05 per mode. The 95% confidence intervals are unadjusted. A nonsignificant recall difference does not establish equivalence. Dataset-specific scores, paired comparisons, coverage, per-track counts, grouping IDs, and run manifests are exported alongside each analysis.

The proposed 449-track selection is available locally: 40 Bach10 tracks, 149 URMP tracks, and 260 seed-selected CocoChorales tracks. Bach10 has ten recorded pieces and URMP has 44 ensemble performances; repeated URMP title labels are conservatively grouped together, leaving 27 title groups in the full selection. Independent composition coverage, rather than frame count alone, determines precision. Before the final runs, use pilot piece-level counts to assess power under the same weighting and multiple-comparison procedure; expand independent-piece coverage if needed. Freeze sample size and configurations before inspecting final significance results, identify development pieces, and report held-out evidence separately. Adding synthetic tracks does not guarantee precision for claims restricted to real recordings.

### Earlier track-averaged measurements

The following material preserves the previous draft's measured values and protocol provenance. Its percentages are track averages, not the frame-pooled results requested for the revised main tables. Rankings, sample counts, and latency statements below refer only to those preliminary runs; the paired tests above have not been run.


The pitch experiments use:

- **Bach10:** eight original isolated recordings from the v1.1 release, comprising two recordings each of violin, clarinet, saxophone, and bassoon, selected with seed 0 from 40 available tracks. All eight methods completed all eight selected recordings (64 method–recording pairs). These measurements replace the earlier Bach10-mf0-synth results.
- **URMP:** 26 isolated real-recording tracks, comprising two tracks from each of 13 instrument classes. A separate causal replay uses 26 tracks sampled under the same instrument-balanced contract.
- **CocoChorales:** 26 isolated test-split tracks, comprising two tracks from each of 13 instrument classes.

Bach10's supplied frame-level references were estimated using YIN and manually corrected. The annotations store fractional MIDI pitches, which we convert to Hz while preserving unvoiced frames; their 46 ms analysis windows have a 10 ms hop and a first frame center at 23 ms. They therefore describe the performed pitch contours rather than only the nominal score pitches, although their YIN-derived provenance should be considered when comparing pitch estimators.

We compare Attune's production pYIN-based pitch pipeline with a local librosa-equivalent pYIN implementation, Praat, Basic Pitch, SwiftF0, CREPE, SPICE, and RMVPE. Every method receives the same reference-derived admissible pitch range: methods with range-aware inference use it internally, while fixed-output methods receive a final range gate. The Basic Pitch adapter selects the strongest in-range contour bin at each frame and uses its note output for voicing; it is distinct from the complete note-transcription system evaluated below. We compute raw pitch accuracy, raw chroma accuracy, overall accuracy, voicing recall, and voicing false-alarm rate using `mir_eval`, with a 50-cent pitch tolerance. The primary table reports mean overall accuracy, which accounts for both pitched frames and unvoiced regions.

For streaming evaluation, estimates are committed only when the samples required by the adapter are available. Attune and Praat receive identical 4096-sample frames and 128-sample hops at 44.1 kHz, while neural methods use their respective model-native rolling contexts and update intervals. Models are warmed before steady-state timing. Latency and deadline misses use a dedicated-worker process-CPU simulation and should not be interpreted as measured microphone-to-display latency. The final evaluation will expand the complete-track comparison to the configured full corpora, including all 40 original Bach10 tracks and all 149 URMP tracks; the current notebook leaves these full runs disabled by default.

| Method | Bach10 original overall accuracy | URMP overall accuracy | CocoChorales overall accuracy | URMP streaming overall accuracy |
| --- | ---: | ---: | ---: | ---: |
| Attune | **93.50** | **92.86** | **95.43** | **92.27** |
| pYIN | 92.05 | 86.02 | 94.77 | — |
| Praat | 92.97 | 92.31 | 94.20 | 83.12 |
| Basic Pitch | 89.01 | 87.07 | 94.25 | — |
| SwiftF0 | 88.91 | 83.28 | 89.04 | 83.15 |
| CREPE | 92.91 | 91.74 | 94.63 | — |
| SPICE | 85.23 | 82.78 | 92.15 | 76.41 |
| RMVPE | 91.18 | 87.75 | 92.05 | 84.70 |

Attune has the highest mean overall accuracy on all three sampled corpora. On original Bach10, Attune scores 93.50%, followed by Praat at 92.97% and CREPE at 92.91%; the 0.53-percentage-point lead over Praat is descriptive and does not establish ranking stability on this eight-recording sample. CREPE has the highest raw pitch accuracy (97.19%), while Praat has the lowest voicing false-alarm rate (27.76%); Attune's corresponding values are 96.96% and 33.10%. Thus, the overall-accuracy ranking does not imply that Attune is best on each individual metric. These original-recording results supersede the synthesized Bach10 column; differences from the historical results are not a controlled estimate of the effect of synthesis because sampled tracks and detector versions may differ.

The URMP result reflects a pitch–voicing tradeoff: Attune's raw pitch accuracy is 93.88%, compared with pYIN's 95.30%, but its voicing false-alarm rate is 11.65%, compared with 45.69%. In causal replay, Attune's median and p95 output latencies are 46.81 and 47.30 ms, respectively, with no simulated deadline misses. These preliminary findings support evaluating accurate rest handling alongside voiced-frame pitch accuracy.


<!-- Author notes: These links are for drafting provenance and should be replaced with bibliographic citations and final artifact references in the paper. -->

## Drafting Sources

- [MIREX melody-extraction metric definitions](https://www.music-ir.org/mirex/wiki/2005%3AAudio_Melody_Extraction_Results)
- [`mir_eval` melody metric documentation](https://mir-eval.readthedocs.io/latest/api/melody.html)

- [Pitch notebook](../benchmarks/notebooks/pitch.ipynb)
- [Original Bach10 preliminary results](../benchmarks/results/pitch/notebook_runs/preliminary_bach10_original/seeded_per_instrument_v1__per_instrument_2__seed_0__methods_pyin+attune+basic_pitch+praat+swiftf0+crepe+spice+rmvpe/rows.csv)
- [Bach10 dataset description](https://labsites.rochester.edu/air/datasets/Bach10%20Dataset_v1.0.pdf)
- [Bach10 v1.1 recording release mirror, pinned revision](https://github.com/flippy-fyp/Bach10_v1.1/tree/2a53cdc6495bb82f03f943b77ca3b11ddf7f5a31)
- [Note notebook](../benchmarks/notebooks/note.ipynb)
- [Vibrato notebook](../benchmarks/notebooks/vibrato.ipynb)
- [Mistake notebook](../benchmarks/notebooks/mistake.ipynb)
- [Native Coco-E renderer timing audit](native-coco-renderer-timing-2026-09-29.md)
- [Current repeat-recovery implementation and validation](repeat-collapsed-alignment-2026-09-30.md)
