# Step-3 curve-control sweep: the extent gap is knot spacing, not the width penalty

> Historical experiment record. The one-off replay probe was removed after its
> conclusion was confirmed; repeat the supported sweep with
> `VibratoNotebook.run_curve_sec_ablation()` and `VibratoBenchmarker.py`.

Search evidence only. Every row below came from a now-retired one-off replay
of stored Attune frame outputs. The replay matched finished runs to four decimal
places, but its rows were never benchmark results; candidates were confirmed
with the full runner before any default changed.

All rows use `primary_v4` runs, the production edge-value hold, and `vib2_center_smoothness=200`.

## The width penalty is already at its optimum

Changing tier, `vib2_curve_sec` at its `0.4` default, sweeping `vib2_width_smoothness` across four orders of magnitude:

| `vib2_width_smoothness` | Extent F1 | Rate F1 | False alarms |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.3741 | 0.8736 | 0.1888 |
| 0.01 | 0.3851 | 0.8751 | 0.1626 |
| 0.03 | 0.4089 | 0.8822 | 0.0937 |
| 0.10 | 0.4400 | 0.8806 | 0.0937 |
| 0.30 | 0.4442 | 0.8832 | 0.0937 |
| **1.00 (default)** | **0.4429** | **0.8858** | **0.1194** |
| 3.00 | 0.4241 | 0.8815 | 0.1502 |
| 10.00 | 0.4063 | 0.8756 | 0.1876 |
| 30.00 | 0.3956 | 0.8663 | 0.1751 |
| 100.00 | 0.3768 | 0.8435 | 0.2849 |

The curve is flat within `0.1--1.0` and falls off on both sides. Removing the penalty entirely costs `0.069` extent F1 and doubles false alarms, reproducing the earlier full-benchmark `step3_coco_width_0` result. The best row, `0.3`, is worth `+0.0013` extent F1. There is no material gain available on this control, which is consistent with the mechanism: a second-difference penalty on B-spline coefficients does not penalize a straight ramp, so it cannot be what flattens a widening or narrowing profile.

## Knot spacing is the lever

`vib2_curve_sec` sets the interior knot spacing of the cubic B-spline bases carrying both the center and the width curve. Lowering it from `0.4` improves extent F1 on all three tiers and lowers false alarms on both Coco tiers, with `vib2_width_smoothness` left untouched at `1.0`:

| `vib2_curve_sec` | Changing ext F1 | Constant ext F1 | Yang ext F1 | Changing FA | Constant FA | Relative cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.400 (default)** | **0.4429** | **0.4435** | **0.5441** | **0.1194** | **0.1603** | **1.0x** |
| 0.200 | 0.4738 | 0.5001 | 0.5693 | 0.0937 | 0.1311 | ~1.5x |
| 0.150 | 0.4872 | 0.5024 | 0.5898 | 0.0937 | 0.1311 | 1.9x |
| 0.100 | 0.5014 | 0.4993 | 0.5810 | 0.0937 | 0.1311 | 3.0x |
| 0.075 | 0.5093 | 0.5059 | 0.5850 | 0.0937 | 0.1311 | 4.7x |

Rate F1 also improves on both Coco tiers (changing `0.8858` to `0.9081` at `0.075`; constant `0.8866` to `0.9133`) and drifts slightly down on Yang (`0.6745` to `0.6588`). The effect is monotone rather than a knife edge: every spacing below the default beats the default on every tier. Cost is measured single-threaded over the 276 changing cases and is the reason to prefer a middle value.

## Confirmed by full benchmark runs

Four `--attune-curve-sec` values were then run through
`VibratoBenchmarker.py` on the changing tier. Every row reproduces the replay
prediction to four decimals, which is the strongest available check on the
replay method itself:

| `vib2_curve_sec` | Extent F1 | Rate F1 | Overall F1 | False alarms | Audio(s)/Compute(s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 (previous default) | 0.4429 | 0.8858 | 0.6644 | 0.1194 | 44.0 |
| 0.20 | 0.4738 | 0.8872 | 0.6805 | 0.0937 | 34.2 |
| 0.15 | 0.4872 | 0.9026 | 0.6949 | 0.0937 | 31.2 |
| **0.10 (promoted)** | **0.5014** | **0.9058** | **0.7036** | **0.0937** | **21.9** |

The `0.40` control run, invoked with `--methods attune` alone, reproduces the full-method-set preliminary row exactly (`0.4429` / `0.8858` / `0.6644` / `0.1194`), so nothing in the comparison depends on the method set. Overall F1 at `0.10` also passes `driedger_template_benchmark_range`'s `0.6980`, which was the gap that prompted the sweep — and that competitor reaches its extent lead while firing on `1.0000` of straight-tone frames, against Attune's `0.0937`.

Throughput figures in that table are contended: the `0.40` row ran alone to prime the shared cache and the rest ran concurrently. The serial replay ratio is the quotable cost, roughly `3x` slower at `0.10` than at `0.40`. Live analysis is unaffected either way, because the causal basis carries no interior knots.

`vib2_curve_sec` is now `0.1` in `Config`. Only the changing tier has a confirmed run; `0.15` remains within replay noise of `0.10` on the constant/control (`0.5024` versus `0.4993`) and Yang (`0.5898` versus `0.5810`) tiers, so a frozen held-out validation should revisit the pair rather than treat `0.10` as settled. **Every preliminary/full suite summary cached before this change was produced at `0.4` and is now stale for Attune.**

The improvement does not come from better ramp tracking. Per-case regression of estimated width on true width over the widening and narrowing profiles gives a median slope of `0.873`/`0.663` at the default and `0.835`/`0.675` at `0.075` — unchanged. Nor does it come from reduced attenuation: the median estimated-to-true width ratio moves only from `0.920` to `0.924`. It comes from the center curve. A knot every `0.4 s` cannot follow intonation drift inside a note, so drift leaks into the residual the sinusoid is fitted to; a finer center curve absorbs it, leaving a cleaner per-frame amplitude and more cases that pass the fit at all.

Two secondary controls were swept and rejected. `vib2_center_smoothness` is best at its `200` default (`800` and `3200` are each slightly worse at every spacing tested). The `fitted` edge policy is worse than the production hold once knots are fine (`0.4563` versus `0.4843` extent F1 at `vib2_curve_sec=0.2`), so the earlier rejection stands.

## Attenuation is upstream, and it is shared

A separate diagnostic on the same frames explains why the extent tolerance is hard for every contour-based method. Against the commanded pitch bend, the median realized peak-to-peak width is `0.990` in the commanded curve, `0.898` after raw pYIN, `0.934` after HMM smoothing, and `0.914` after Attune's fit — so Attune recovers `0.979` of what reaches it, and most of the loss is in the pitch stage. Every contour-based method sits in the same band (McLeod `0.932`, Rossignol `0.929`, Ventura `0.942`), while audio-domain Driedger at the published bank sits at `0.991`.

The extent tolerance is `0.05` semitones one-sided, i.e. `10` cents peak-to-peak, against true widths of `85--150` cents. A `7--9%` attenuation is therefore `8--14` cents and straddles the pass/fail line. Applying a constant post-hoc gain to the recorded estimates lifts extent F1 from `0.443` to `0.567` at `1.05` for Attune, from `0.573` to `0.621` for Driedger's wide bank, and from `0.479` to `0.578` for McLeod. That every method peaks near the same `1.05` is the point: the residual is a shared front-end and rendering property, not an estimator defect, and it should be characterized rather than tuned away.

## Confirmation runs

The supported confirmation path is `vibrato.ipynb`'s step-3 knot-spacing cell
(`BENCH.run_curve_sec_ablation()`), which drives `VibratoBenchmarker.py
--attune-curve-sec` once per value. The notebook runs spacing values
sequentially by default to avoid nested worker pools.
