# The `extrema` baseline has no published provenance, and it shows

> Historical experiment record. Its one-off probe was replaced by regression
> coverage in `tests/VibratoBenchmarkerTest.py`.

`ExtremaEstimator` in [VibratoBaselines.py](../modules/vibrato/VibratoBaselines.py#L48) is the only method in the harness that is not a port of anything. Every other entry names a source and a commit: `yang_ava_fdm` pins AVA at `77e4dfe`, `mcleod_tartini_prony` pins Tartini at `54e4dbae`, `attune` is the production detector. `extrema` is described in the README as "smooths the pitch contour, finds alternating prominent maxima and minima, and estimates rate and amplitude from each half-cycle" with no citation, and its parameters (35 ms smoothing, 5-cent prominence, 2-15 Hz band) match no published method. Under fair-comparison rule 6 it is currently an unlabelled independent reproduction of nothing in particular, which means a reader cannot tell whether a weak extrema row is a fact about the extrema family or a fact about our code. This document establishes what the published family actually specifies, and measures the gap.

## The extrema family in the literature

The family splits cleanly into a *measurement convention* (voice science, heavily cited, no automatic detection rule) and an *automatic detector* (MIR, sparsely cited, fully specified). No single paper is both, which is why the harness needs two citations rather than one.

| Work | Venue | Cites | What it contributes | Automatic? |
| --- | --- | --- | --- | --- |
| Horii (1989) | J. Voice | 67 | Extrema measurement of vocal vibrato; currently the citation in `paper/vibrato.md` | No |
| **Prame (1994)** | JASA 96(4) | **138** | Rate as the inverse of the interval between successive extrema; mean 6.0 Hz over ten singers | No |
| **Prame (1997)** | JASA 102(1) | **122** | Extent as peak-to-trough in cents, plus intonation as the extrema midline | No |
| **Rossignol et al. (1999)** | DAFx-99 | ~low | The "minima - maxima detection" method: interpolated extrema, a variance-plus-count decision rule, and the extrema midline as the vibrato-free contour | **Yes** |
| Bretos and Sundberg (2003) | J. Voice | 53 | Extrema measurement over sustained crescendo notes | No |
| Pang and Yoon (2005) | Pattern Recognition | 13 | Note-wise detection modelling P(vibrato) from rate, extent, and intonation; paywalled | Yes |
| Ventura, Sousa, and Ferreira (2012) | ISCCSP | 11 | Hybrid: FFT peak for rate, **extrema difference for extent**; relative errors below 0.1% | Yes |
| Yang, Rajab, and Chew (2017) | J. Math. Music | 15 | Not an extrema method, but its **ground truth annotation is Prame's convention verbatim** | n/a |

Citation counts are OpenAlex, which undercounts DAFx proceedings badly. Two further observations shape the recommendation:

**Prame's convention is already the harness's ground truth.** Yang's parameter annotation reads: "Assuming the interval between one peak and one trough is the duration of a half cycle, and the vibrato rate is the inverse of the cycle length, the vibrato extent is the difference between the peak and trough measured in semitones." That is Prame, and it is exactly what the `Areas_and_parameters` tier scores against. So the extrema baseline and the ground truth for the Yang tier are computing the same quantity by construction — worth stating in the paper, because it makes the extrema row the natural floor for that tier rather than an arbitrary competitor.

**Yang's own comparison contains no extrema method.** His Table 1 lists only Herrera-Bonada, Ventura-Sousa-Ferreira, and von Coler-Robel, all frame-wise. Rossignol is cited but dismissed into the note-wise class and never evaluated. So there is no published accuracy number for an extrema detector to check a port against — unlike Driedger, where the port plan can gate on published F-measures. An extrema port can only be validated on synthetic recovery.

## Recommendation

Cite **Prame (1994, 1997)** for the parameter convention and port **Rossignol et al. (1999)** for the detection rule.

If only one name goes in the results table, it is **Rossignol et al. (1999), "Vibrato: detection, estimation, extraction, modification", DAFx-99** — because a benchmark competitor must emit a detection decision, and Rossignol is the only member of the family that publishes one. Prame is by far the more cited (138 + 122 against a DAFx paper), and it is the right citation for the rate and extent arithmetic, which our code already implements correctly. But Prame never automated the decision — his cycles were identified by hand on sonograms — so he cannot be run as a competitor. Ventura et al. is the most accurate of the three and the most recent, but it takes its rate from an FFT peak and only its extent from extrema; porting it as "the extrema method" would misattribute the family. It belongs where the port plan already puts it, as a follow-on to Herrera-Bonada.

Rossignol's method as published, from section 2.5:

- Local maxima of the f0 trajectory are detected and **precisely pinpointed by interpolation**.
- All temporal distances between two successive local maxima are computed, then their **variance**, and the **count** of distances falling in 0.15-0.25 s (4-6.7 Hz).
- The same is done independently for the local minima.
- Vibrato is declared when both variances are low **and** both counts are high — a regularity test, not a per-cycle test.
- Significance is confirmed by a large `Mfreq = mean((max_interp - min_interp) / ((max_interp + min_interp) / 2))`.
- The midline `(max_interp + min_interp) / 2` is the by-product estimate of the vibrato-free f0 contour.
- f0-trajectory methods in that paper use 300 ms windows.

The paper defers exact thresholds to the companion journal article (Rossignol et al., JNMR 1999), so a port must choose the variance and count cutoffs itself and label them as adapter behavior, the same way the Driedger plan handles its unspecified parameters.

## Where our implementation diverges

Eight differences, ordered by how much they distort the reported numbers.

**1. There is no regularity or duration gate at all.** Rossignol's entire decision is a regularity test across many extrema. Ours declares vibrato from a **single half-cycle** in isolation: [VibratoBaselines.py:129-141](../modules/vibrato/VibratoBaselines.py#L129) accepts any adjacent max/min pair whose implied rate is in band and whose peak-to-trough exceeds the prominence, and paints those frames detected. One accidental wiggle is vibrato. Every other method in the harness has such a gate — AVA requires six candidate frames plus 0.25 s pruning, Tartini requires one fitted cycle, `attune` requires `vib_min_cycles` of both fitted rate and observed sign alternations. This is the single largest divergence and it is what the 0.2316 false-alarm figure in `results/vibrato/summary.csv` is measuring.

**2. Extrema are not interpolated.** Rossignol says "precisely pinpointed by interpolation" in his first sentence, and it is the reason the method is usable: on the shared 344.53 fps grid a 5.5 Hz half-cycle spans 31 frames, so integer-grid endpoints put up to +/-3% error on every rate estimate before any noise is considered. The McLeod port already does sub-sample work; this one takes `example.times[right] - example.times[left]` raw at [VibratoBaselines.py:132](../modules/vibrato/VibratoBaselines.py#L132). Parabolic interpolation of both the extremum time and its value is roughly ten lines.

**3. Unvoiced frames become MIDI 0, and the comment justifying it does not apply here.** [VibratoBaselines.py:78-83](../modules/vibrato/VibratoBaselines.py#L78) substitutes `0.0` for NaN and cites AVA's `freqToMidi` sentinel. That is the correct adapter for `yang_ava_fdm`, which is a port of AVA. `extrema` is not a port of AVA and has no reason to inherit its sentinel. The consequence is specific to extrema methods: a rest turns into a 62-to-0 semitone cliff, the 35 ms Savitzky-Golay smoother ramps across it, and `find_peaks` sees enormous shoulder extrema on both sides. The harness's own McLeod adapter uses the opposite and correct policy — `np.interp` across gaps, reported as `filled_unvoiced_frames` in metadata ([McLeodTartiniProny.py:182](../modules/vibrato/McLeodTartiniProny.py#L182)) — and the port plan specifies interpolate-then-flag as the house policy for new ports. Tellingly, `_filled()` at [VibratoBaselines.py:34](../modules/vibrato/VibratoBaselines.py#L34) implements exactly that policy and **is dead code, called from nowhere in the package**. Two estimators in one harness using opposite unvoiced policies also means their rows are not directly comparable.

**4. The rate band is wider than any published band.** Ours is 2-15 Hz. Rossignol's is 4-6.7 Hz, AVA's final decision band is 4-9 Hz, `attune` reports 3-10 Hz. A 15 Hz ceiling admits max/min pairs 12 frames apart, which on a jittery pYIN contour is noise, not vibrato.

**5. The prominence floor is below every published minimum.** Ours is 5 cents peak-to-trough. AVA's decision minimum is 0.10 semitones one-sided, i.e. 20 cents peak-to-peak; `attune` requires 10 cents. pYIN's own frame-to-frame jitter is comparable to 5 cents, so the gate is effectively off. Worse, the threshold is *absolute*, where the extrema literature and our own production detector use an *amplitude-relative* one: `VibratoDetector2._swing_extrema` confirms an extremum only after a reversal of `ALT_PROM_FRAC * amplitude` (0.3 x A0), a hysteresis rule that scales with the oscillation being measured ([VibratoDetector2.py:528](../../algorithms/VibratoDetector2.py#L528)).

**6. Width is measured on the smoothed contour, not the interpolated peak.** [VibratoBaselines.py:133](../modules/vibrato/VibratoBaselines.py#L133) reads `smooth[right] - smooth[left]`, so the reported extent carries the Savitzky-Golay attenuation. Prame's extent is the peak-to-trough of the (interpolated) f0 curve itself. The bias is small at 5.5 Hz with a 35 ms window, but it is a systematic underestimate and it is free to fix once extrema are interpolated.

**7. The pitch center is a low-pass, not the extrema midline.** We take a 0.45 s Savitzky-Golay of the contour ([VibratoBaselines.py:89](../modules/vibrato/VibratoBaselines.py#L89)). Rossignol's center is `(max_interp + min_interp) / 2`, which is the whole point of his by-product — it is what makes the method a note-segmentation front-end. Low impact, since center is excluded from Overall F1, but a Rossignol port should use his definition.

**8. `qualities` is invented.** `min(1, width / (2 * prominence))` at [VibratoBaselines.py:145](../modules/vibrato/VibratoBaselines.py#L145) has no source. Rossignol's natural quality signal is `Mfreq` together with the inverse interval variance. Not scored, so cosmetic, but it should either be Rossignol's or be dropped.

## What the divergences cost, measured

Synthetic contours run in-process, no audio and no pYIN. Script at [probe_extrema_gates.py](probe_extrema_gates.py); the corpus is three 5.5 Hz / 80-cent vibrato notes interleaved with straight notes, portamento slides, and slow intonation drifts, with 5-cent Gaussian jitter standing in for pYIN tracking noise.

| Configuration | Recall | False alarms | Precision | F1 |
| --- | --- | --- | --- | --- |
| **As shipped** (5c prominence, 2-15 Hz, no gate) | 0.973 | **0.466** | 0.511 | 0.670 |
| + Rossignol regularity gate, nothing else changed | 0.973 | **0.038** | 0.927 | **0.949** |
| 5c prominence, 4-9 Hz (AVA band) | 0.823 | 0.223 | 0.649 | 0.726 |
| 5c prominence, 4-6.7 Hz (Rossignol band) | 0.804 | 0.091 | 0.816 | 0.810 |
| 20c prominence (AVA floor), 2-15 Hz | 0.973 | 0.000 | 1.000 | **0.986** |
| 20c prominence, 4-6.7 Hz | 0.804 | 0.000 | 1.000 | 0.891 |

The first two rows are the finding. Adding Rossignol's own decision rule to the extrema we already compute drops false alarms by an order of magnitude **at unchanged recall**. Raising the prominence floor to AVA's published minimum does the same. Nothing about the extrema family produces a 0.47 false-alarm rate; our thresholds and our missing gate do.

Two caveats. Rossignol does not publish his variance and count cutoffs, so the gate row uses cutoffs I chose (at least 4 cycles in the run, coefficient of variation of half-cycle intervals at most 0.25, majority of half-cycles in band) — a port must label these as adapter behavior. And on clean single-note contours the shipped estimator is already fine (rate accuracy 0.986 at zero jitter, 0.946 at 5-cent jitter), so this is specifically a false-alarm and mixed-material problem, which is what the Coco and Yang tiers are.

## A fairness problem for the paper

`paper/vibrato.md` positions extrema-based measurement as the family whose "resolution is bounded by the oscillation it measures", and step 3 of Attune's own method refines its rate seed from local extrema using the amplitude-relative hysteresis rule and a six-extrema support requirement. So the paper compares Attune, whose extrema stage has a relative prominence threshold and a minimum-support rule, against a baseline whose extrema stage has an absolute 5-cent threshold and no support rule. A reviewer who reads both will notice. Porting Rossignol fixes the framing as well as the number: the claim becomes "even with a proper regularity gate, per-half-cycle measurement cannot describe a rate that changes within the cycle", which is the argument the paper actually wants to make and is one the current straw baseline cannot support.

## Suggested change, smallest first

1. Rename `extrema` to `rossignol_minmax`, add the provenance docstring and the DAFx citation, and keep `ExtremaEstimator` as a deprecated alias so existing notebook cells resolve.
2. Replace the MIDI-0 sentinel with `_filled()`, which already exists, and report `filled_unvoiced_frames` in metadata as McLeod does.
3. Add parabolic interpolation of extremum time and value; take width from the interpolated values and center from the interpolated midline.
4. Add the regularity gate: per-run variance of half-cycle intervals, count in band, and `Mfreq`, with the cutoffs documented as adapter behavior.
5. Move the defaults onto Rossignol's published band (4-6.7 Hz) and expose the wider band as a `band_mode="native" | "harness"` switch, mirroring the `window_mode` decision in the Herrera-Bonada plan. Report both if they differ materially.
6. Validation, since no published accuracy number exists to gate against: synthetic recovery of a 5.5 Hz / 80-cent contour to within 0.1 Hz and 5%, matching the existing tests in `test_vibrato_benchmark.py`, plus the `Areas_only` tier from Part 0.5 of the port plan once it lands.

Steps 1-3 are mechanical. Step 4 is the one that changes the results table, and it should be frozen before any scored run, per fair-comparison rule 1.

## Sources

- Rossignol, Depalle, Soumagne, Rodet, and Collette (1999), "Vibrato: detection, estimation, extraction, modification", DAFx-99. https://www.dafx.de/paper-archive/details/N2Kpz2NdMSLXkSnquj5qQw
- Prame (1994), "Measurements of the vibrato rate of ten singers", JASA 96(4), 1979-1984.
- Prame (1997), "Vibrato extent and intonation in professional Western lyric singing", JASA 102(1), 616-621.
- Ventura, Sousa, and Ferreira (2012), "Accurate analysis and visual feedback of vibrato in singing", ISCCSP. https://fe.up.pt/voicestudies/artts/doc/publications/086%20-%20ACCURATE%20ANALYSIS%20AND%20VISUAL%20FEEDBACK%20OF%20VIBRATO%20IN%20SINGING.pdf
- Yang, Rajab, and Chew (2017), "The filter diagonalisation method for music signal analysis: frame-wise vibrato detection and estimation", Journal of Mathematics and Music 11(1). https://luweiyang.com/wp-content/uploads/2017/05/jmm2017_luwei_yang.pdf
- Pang and Yoon (2005), "Automatic detection of vibrato in monophonic music", Pattern Recognition 38(7). https://doi.org/10.1016/j.patcog.2005.01.001
- Horii (1989), "Acoustic analysis of vocal vibrato: a theoretical interpretation of data", Journal of Voice 3(1), 36-43.
