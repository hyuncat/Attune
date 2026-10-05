from __future__ import annotations

import os
import tempfile
import threading
import unittest
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
from scipy.io import loadmat

from algorithms.Config import Config
from algorithms.PitchDetector import PitchDetector
from algorithms.PitchSmoother import PitchSmoother
from algorithms.VibratoDetector import VibratoDetector
from app_logic.user.ds.PitchData import Pitch, PitchData
from app_logic.user.ds.VibratoData import VibratoData
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
from benchmarks.modules.vibrato.datasets.CocoDataset import (
    CHANGING_AMPLITUDE_SPAN_RANGE_SEMITONES,
    CHANGING_RATE_SPAN_RANGE_HZ,
    DEFAULT_PROFILES,
    NATIVE_AMPLITUDE_RANGE_SEMITONES,
    NATIVE_RATE_RANGE_HZ,
    CocoDataset,
)
from benchmarks.modules.vibrato.competitors.Attune import Attune
from benchmarks.modules.vibrato.competitors.Driedger import Driedger
from benchmarks.modules.vibrato.competitors.DriedgerDataset import DriedgerDataset
from benchmarks.modules.vibrato.competitors.HerreraBonada import HerreraBonada
from benchmarks.modules.vibrato.competitors.McLeod import McLeod
from benchmarks.modules.vibrato.competitors.Rossignol import Rossignol
from benchmarks.modules.vibrato.competitors.VenturaSousaFerreira import (
    VenturaSousaFerreira,
)
from benchmarks.modules.vibrato.VibratoDetectorBase import (
    CallableDetector,
    VibratoEstimate,
    VibratoExample,
)
from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker
from benchmarks.modules.vibrato.competitors.Yang import Yang, YangBR, YangDT
from benchmarks.modules.vibrato.datasets.SyntheticDataset import SyntheticDataset
from benchmarks.modules.vibrato.datasets.YangDataset import (
    YANG_PITCH_RANGE_OVERRIDES_HZ,
    YangDataset,
)
from benchmarks.modules.vibrato.VibratoNotebook import (
    BENCHMARK_METHODS,
    PRIMARY_COMPARISON_VERSION,
)


class _PreparedFixtureEstimator:
    """Module-level fixture so spawned scoring workers can import it."""

    requires = {"pitch"}
    scores_center = False
    description = "prepared-state fixture"

    def __init__(self, name: str) -> None:
        self.name = name
        self.prepared_case = ""

    def prepare(self, example) -> None:
        self.prepared_case = example.case_id

    def estimate(self, example) -> VibratoEstimate:
        if self.prepared_case != example.case_id:
            raise RuntimeError("prepared estimator state leaked across jobs")
        warnings.warn("spawned fixture warning", RuntimeWarning)
        return VibratoEstimate(
            example.rate_hz.copy(),
            example.width_cents.copy(),
            example.is_vibrato.copy(),
        )


class VibratoBenchmarkerTest(unittest.TestCase):
    def test_primary_defaults_exclude_method_variants(self) -> None:
        available = {
            estimator.name for estimator in VibratoBenchmarker.available_detectors()
        }
        defaults = {
            estimator.name for estimator in VibratoBenchmarker.default_detectors()
        }
        self.assertIn("herrera_bonada_yang_window", available)
        self.assertNotIn("herrera_bonada_yang_window", defaults)
        self.assertIn("herrera_bonada", defaults)
        self.assertIn("rossignol", defaults)
        self.assertIn("driedger_benchmark_range", available)
        self.assertNotIn("driedger_benchmark_range", defaults)
        self.assertIn("rossignol", BENCHMARK_METHODS)
        self.assertIn("driedger_benchmark_range", BENCHMARK_METHODS)
        self.assertEqual(PRIMARY_COMPARISON_VERSION, "primary_v7")

    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = SyntheticDataset.build(
            replicates=1,
            seed=7,
            frame_rate=100.0,
            noise_cents=0.0,
            dropout_probability=0.0,
            outlier_probability=0.0,
        )
        cls.constant = next(
            case for case in cls.examples if case.scenario == "constant"
        )

    @staticmethod
    def isolated_vibrato_example(
        *,
        frame_rate: float = 44_100 / 128,
        duration: float = 4.0,
        rate_hz: float = 5.5,
        width_cents: float = 80.0,
    ) -> VibratoExample:
        times = np.arange(int(round(duration * frame_rate))) / frame_rate
        center = np.full(len(times), 60.0)
        pitch = center + (width_cents / 200.0) * np.sin(2.0 * np.pi * rate_hz * times)
        return VibratoExample(
            case_id="isolated_vibrato",
            scenario="constant",
            split="fixture",
            times=times,
            pitch_midi=pitch,
            center_midi=center,
            rate_hz=np.full(len(times), rate_hz),
            width_cents=np.full(len(times), width_cents),
            is_vibrato=np.ones(len(times), dtype=bool),
        )

    def test_synthetic_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = SyntheticDataset.write_csv(
                self.examples, Path(directory) / "corpus.csv"
            )
            loaded = SyntheticDataset.load_csv(path)
        self.assertEqual(
            [case.case_id for case in loaded],
            sorted(case.case_id for case in self.examples),
        )
        original = {case.case_id: case for case in self.examples}
        for case in loaded:
            np.testing.assert_allclose(
                case.pitch_midi,
                original[case.case_id].pitch_midi,
                equal_nan=True,
            )
            np.testing.assert_allclose(case.rate_hz, original[case.case_id].rate_hz)
            np.testing.assert_array_equal(
                case.score_mask,
                original[case.case_id].score_mask,
            )
            self.assertEqual(
                case.metadata["analysis_group"],
                original[case.case_id].metadata["analysis_group"],
            )
            self.assertEqual(
                case.metadata["analysis_note_bounds"],
                [
                    list(bounds)
                    for bounds in original[case.case_id].metadata[
                        "analysis_note_bounds"
                    ]
                ],
            )

    def test_coco_pitch_range_uses_all_notes_and_exact_bend_extrema(self) -> None:
        fmin, fmax = CocoDataset._annotation_pitch_range(
            [(0.0, 1.0, 55.0), (1.0, 2.0, 60.0)],
            [
                SimpleNamespace(
                    pitch_midi=60,
                    offset_semitones=np.asarray([-0.5, 0.75]),
                )
            ],
        )
        config = Config()
        self.assertAlmostEqual(fmin, config.midi_to_freq(55.0))
        self.assertAlmostEqual(fmax, config.midi_to_freq(60.75))

    def test_yang_pitch_range_comes_from_frequency_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Example-Annotation-new.csv"
            path.write_text(
                "0.0,440.0,0.8,0.787402\n"
                "1.0,0.0,0.5,0.787402\n"
                "2.0,660.0,0.7,0.787402\n",
                encoding="utf-8",
            )
            self.assertEqual(YangDataset._annotation_pitch_range(path), (440.0, 660.0))
            path.write_text("0.0,0.787402,0.8\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no annotated pitch"):
                YangDataset._annotation_pitch_range(path)

    def test_yang_missing_frequency_override_is_scoped_to_one_recording(self) -> None:
        self.assertEqual(
            YANG_PITCH_RANGE_OVERRIDES_HZ,
            {"Huangjiangqin-1": (180.0, 2000.0)},
        )

    def test_ava_fdm_port_recovers_clean_constant_vibrato(self) -> None:
        estimate = YangDT().estimate(self.constant)
        target = self.constant.score_mask
        supported = target & (estimate.rate_hz > 0.0)
        self.assertGreater(float(np.mean(supported[target])), 0.95)
        self.assertLess(float(np.mean(np.abs(estimate.rate_hz[supported] - 5.5))), 1.25)
        self.assertLess(
            float(np.mean(np.abs(estimate.width_cents[supported] - 80.0))), 40.0
        )
        # On a continuous multi-note track, neighboring negative passages let
        # the released candidate grouper retain this positive decision run.
        self.assertGreater(float(np.mean(estimate.detected[target])), 0.95)

    def test_ava_matlab_smooth_and_frame_geometry(self) -> None:
        values = np.array([1.0, 2.0, 8.0, 4.0, 5.0, 12.0, 7.0])
        np.testing.assert_allclose(
            Yang.matlab_smooth(values, span=4),
            [1.0, 11.0 / 3.0, 14.0 / 3.0, 17.0 / 3.0, 7.0, 8.0, 7.0],
        )
        times = np.arange(200, dtype=float) / 100.0
        pitch = 60.0 + 0.4 * np.sin(2.0 * np.pi * 5.5 * times)
        track = Yang.analyze_pitch_track(pitch, times)
        # AVA uses floor(.125*100)=12 samples, floor(12*.25)=3 hop,
        # while pin < length-window, and locates the first frame at window/2.
        self.assertEqual(len(track.times), 63)
        self.assertAlmostEqual(track.times[0], 0.06)
        self.assertAlmostEqual(track.times[-1], 1.92)

    def test_ava_fdm_released_equation_fixture(self) -> None:
        sample_rate = 100.0
        times = np.arange(12, dtype=float) / sample_rate
        frame = 0.4 * np.sin(2.0 * np.pi * 5.5 * times)
        frequencies, amplitudes = Yang.frame_fdm3(
            frame - np.mean(frame),
            sample_rate,
        )
        positive = np.flatnonzero(np.real(frequencies) > 0.0)
        self.assertEqual(len(positive), 1)
        index = int(positive[0])
        self.assertAlmostEqual(float(np.real(frequencies[index])), 7.67816684, places=5)
        self.assertAlmostEqual(
            float(2.0 * abs(amplitudes[index])), 0.24658105, places=5
        )

    def test_ava_unstable_roots_are_rejected_without_runtime_warning_spam(self) -> None:
        frame = np.full(43, 60.0)
        frame[0] = 0.0
        frame -= np.mean(frame)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frequencies, amplitudes = Yang.frame_fdm3(frame, 344.53125)
        runtime = [
            warning
            for warning in caught
            if issubclass(warning.category, RuntimeWarning)
        ]
        self.assertEqual(runtime, [])
        self.assertTrue(np.all(np.isfinite(frequencies)))
        self.assertTrue(np.all(np.isfinite(amplitudes)))

    def test_ava_decision_tree_defaults_and_cleanup(self) -> None:
        rates = np.array([5.0, 5.0, 2.0, 5.0, 5.0])
        extents = np.full(5, 0.2)
        np.testing.assert_array_equal(
            Yang.decision_tree(rates, extents),
            [True, True, True, True, True],
        )
        np.testing.assert_array_equal(
            Yang.decision_tree(np.array([5.0, 2.0, 5.0]), np.full(3, 0.2)),
            [False, False, False],
        )

    def test_ava_candidate_and_duration_pruning(self) -> None:
        times = np.arange(20, dtype=float) * 0.05
        decisions = np.zeros(20, dtype=bool)
        decisions[1:7] = True
        decisions[10:16] = True
        expected = decisions.copy()
        np.testing.assert_array_equal(Yang.candidate_mask(decisions, times), expected)
        # The native harness repairs AVA's zero/single-passage indexing bug.
        # Keep the Python benchmark on that same intended grouping behavior.
        self.assertTrue(Yang.candidate_mask(np.ones(20, dtype=bool), times).all())

    def test_ava_bayes_kdes_match_native_matlab_reference(self) -> None:
        detector = YangBR()
        native = loadmat(detector.DEFAULT_MODEL_PATH, simplify_cells=True)
        model = detector._load_models()

        rate_grid = np.asarray(native["rateGrid"], dtype=float).reshape(-1)
        extent_grid = np.asarray(native["extentGrid"], dtype=float).reshape(-1)
        rate_keep = (rate_grid >= 2.0) & (rate_grid <= 20.0)
        extent_keep = (extent_grid >= 0.0) & (extent_grid <= 3.0)

        python_rate_posterior = detector._posterior(
            model["pdVR"].pdf(rate_grid[rate_keep]),
            model["pdNR"].pdf(rate_grid[rate_keep]),
        )
        python_extent_posterior = detector._posterior(
            model["pdVA"].pdf(extent_grid[extent_keep]),
            model["pdNA"].pdf(extent_grid[extent_keep]),
        )
        matlab_rate_posterior = np.asarray(
            native["ratePosterior"], dtype=float
        ).reshape(-1)[rate_keep]
        matlab_extent_posterior = np.asarray(
            native["extentPosterior"], dtype=float
        ).reshape(-1)[extent_keep]

        np.testing.assert_allclose(
            python_rate_posterior,
            matlab_rate_posterior,
            rtol=0.0,
            atol=8e-5,
        )
        np.testing.assert_allclose(
            python_extent_posterior,
            matlab_extent_posterior,
            rtol=0.0,
            atol=1.2e-4,
        )

        # Recreate the native export's 60,501-point Cartesian decision grid.
        rate_decision_grid = (rate_grid >= 0.0) & (rate_grid <= 20.0)
        extent_decision_grid = (extent_grid >= 0.0) & (extent_grid <= 3.0)
        rate_indices = np.flatnonzero(
            rate_decision_grid & np.isclose(rate_grid / 0.1, np.round(rate_grid / 0.1))
        )
        extent_indices = np.flatnonzero(
            extent_decision_grid
            & np.isclose(extent_grid / 0.01, np.round(extent_grid / 0.01))
        )
        expected = (
            np.asarray(native["ratePosterior"]).reshape(-1)[rate_indices, None]
            * np.asarray(native["extentPosterior"]).reshape(-1)[None, extent_indices]
        ) >= 0.25
        actual = (
            detector._posterior(
                model["pdVR"].pdf(rate_grid[rate_indices]),
                model["pdNR"].pdf(rate_grid[rate_indices]),
            )[:, None]
            * detector._posterior(
                model["pdVA"].pdf(extent_grid[extent_indices]),
                model["pdNA"].pdf(extent_grid[extent_indices]),
            )[None, :]
        ) >= 0.25
        np.testing.assert_array_equal(actual, expected)

    def test_ava_bayes_estimator_recovers_clean_constant_vibrato(self) -> None:
        estimate = YangBR().estimate(self.constant)
        target = self.constant.score_mask
        self.assertGreater(float(np.mean(estimate.detected[target])), 0.95)
        self.assertEqual(estimate.metadata["decision_rule"], "bayes_rule")
        self.assertIn("frame_br_probability", estimate.metadata)

    def test_ava_python_pipeline_matches_native_fixture_decisions(self) -> None:
        native = pd.read_csv(
            Path(__file__).resolve().parents[1]
            / "provenance"
            / "yang"
            / "ava_native_reference.csv"
        )
        times = np.arange(400, dtype=float) / 100.0
        pitch = np.full(400, 60.0)
        vibrato = (times >= 1.0) & (times < 3.0)
        pitch[vibrato] += 0.4 * np.sin(2.0 * np.pi * 5.5 * (times[vibrato] - 1.0))
        track = Yang.analyze_pitch_track(pitch, times)
        detector = YangBR()
        bayes = detector._evaluate_bayes(track.rate_hz, track.extent_semitones)
        bayes_detected = Yang.candidate_mask(
            bayes.decision_detected,
            track.times,
        )

        np.testing.assert_allclose(track.times, native["time"], atol=1.1e-14)
        np.testing.assert_array_equal(
            np.isfinite(track.rate_hz),
            np.isfinite(native["rate_hz"]),
        )
        finite = np.isfinite(track.rate_hz)
        np.testing.assert_allclose(
            track.extent_semitones[finite],
            native.loc[finite, "extent_semitones"],
            rtol=0.0,
            atol=2e-6,
        )
        np.testing.assert_array_equal(track.detected, native["dt_detected"])
        np.testing.assert_array_equal(
            bayes.raw_detected,
            native["br_raw_detected"],
        )
        np.testing.assert_array_equal(bayes_detected, native["br_detected"])

    def test_attune_worker_flushes_and_resets_between_takes(self) -> None:
        recording = SimpleNamespace(vibrato_data=object(), pitch_data=object())
        detector = VibratoDetector(recording=recording, config=Config())
        processed = threading.Event()
        flush_states = []

        def extend(vibrato_data, pitch_data):
            if detector._stop.is_set():
                flush_states.append(detector._live_fit_state)
            processed.set()

        with patch.object(detector, "extend", side_effect=extend):
            for _ in range(2):
                processed.clear()
                detector.run()
                try:
                    self.assertIsNone(detector._live_fit_state)
                    marker = object()
                    detector._live_fit_state = marker
                    detector.notify()
                    self.assertTrue(processed.wait(2.0))
                finally:
                    detector.stop()
                self.assertIs(flush_states[-1], marker)
                self.assertIsNone(detector._live_fit_state)
                self.assertIsNone(detector._thread)

    def test_attune_recovers_clean_constant_vibrato(self) -> None:
        estimate = Attune().estimate(self.constant)
        target = self.constant.score_mask
        self.assertTrue(estimate.detected[target].all())
        self.assertLess(float(np.mean(np.abs(estimate.rate_hz[target] - 5.5))), 0.1)
        self.assertLess(
            float(np.mean(np.abs(estimate.width_cents[target] - 80.0))), 2.0
        )

    def test_attune_detection_floor_rejects_rate_or_width_below_cutoff(self) -> None:
        detector = VibratoDetector(config=Config())
        self.assertFalse(
            detector._meets_detection_floor(
                np.full(20, 2.99),
                np.full(20, 80.0),
            )
        )
        self.assertFalse(
            detector._meets_detection_floor(
                np.full(20, 5.5),
                np.full(20, 9.99),
            )
        )
        self.assertTrue(
            detector._meets_detection_floor(
                np.full(20, 3.0),
                np.full(20, 10.0),
            )
        )
        self.assertTrue(
            detector._meets_detection_floor(
                np.full(20, 10.0),
                np.full(20, 10.0),
            )
        )
        self.assertFalse(
            detector._meets_detection_floor(
                np.full(20, 10.01),
                np.full(20, 80.0),
            )
        )

    def test_attune_fitted_edge_ablation_keeps_quality_taper(self) -> None:
        rates = np.linspace(4.0, 8.0, 101)
        widths = np.linspace(20.0, 120.0, 101)
        qualities = np.ones(101)

        held_rate, held_width, held_quality = VibratoDetector._stabilize_offline_edges(
            rates,
            widths,
            qualities,
            representative_rate=5.0,
            frame_rate=100.0,
            hold_values=True,
        )
        fitted_rate, fitted_width, fitted_quality = (
            VibratoDetector._stabilize_offline_edges(
                rates,
                widths,
                qualities,
                representative_rate=5.0,
                frame_rate=100.0,
                hold_values=False,
            )
        )

        self.assertNotEqual(held_rate[0], rates[0])
        self.assertNotEqual(held_width[-1], widths[-1])
        np.testing.assert_allclose(fitted_rate, rates)
        np.testing.assert_allclose(fitted_width, widths)
        np.testing.assert_allclose(fitted_quality, held_quality)
        self.assertLess(fitted_quality[0], 1.0)
        self.assertLess(fitted_quality[-1], 1.0)

    def test_attune_seed_admits_attenuated_candidate_then_clamps_amplitude(
        self,
    ) -> None:
        detector = VibratoDetector(config=Config())
        times = np.linspace(0.0, 1.0, 345)
        basis, _, _ = detector._spline_basis(times, causal=False)
        voiced = np.ones(len(times), dtype=bool)

        subtle = 60.0 + 0.08 * np.sin(2.0 * np.pi * 5.5 * times + 0.2)
        seed = detector._seed_rate_and_phase(
            subtle,
            times,
            basis,
            voiced,
            scan_bins=100,
        )
        self.assertIsNotNone(seed)
        self.assertAlmostEqual(seed[2], 0.15)

        below_admission = 60.0 + 0.02 * np.sin(2.0 * np.pi * 5.5 * times + 0.2)
        self.assertIsNone(
            detector._seed_rate_and_phase(
                below_admission,
                times,
                basis,
                voiced,
                scan_bins=100,
            )
        )

    def test_attune_projected_seed_matches_independent_grid_solves(self) -> None:
        config = Config()
        detector = VibratoDetector(config=config)
        frame_rate = config.sr / config.h1
        times = np.arange(int(round(0.4 * frame_rate))) / frame_rate
        values = 60.0 + 0.35 * times + 0.42 * np.sin(2.0 * np.pi * 6.2 * times + 0.31)
        voiced = np.ones(len(times), dtype=bool)
        voiced[31:35] = False
        confidence = np.ones(len(times))
        confidence[:12] = np.linspace(0.3, 1.0, 12)
        basis, _, _ = detector._spline_basis(times, causal=True)

        projected = detector._seed_rate_and_phase(
            values,
            times,
            basis,
            voiced,
            scan_bins=32,
            confidence=confidence,
        )
        self.assertIsNotNone(projected)

        frequencies = np.linspace(
            config.vib2_fit_rate_min_hz,
            config.vib2_fit_rate_max_hz,
            32,
        )
        m = basis.shape[1]
        second = detector._second_difference(m)
        penalty = np.zeros((len(second), m + 2))
        penalty[:, :m] = np.sqrt(config.vib2_center_smoothness) * second
        weights = np.sqrt(confidence[voiced])
        target = np.concatenate(
            (
                values[voiced] * weights,
                np.zeros(len(penalty)),
            )
        )
        reference = None
        for rate in frequencies:
            angle = 2.0 * np.pi * rate * times
            design = np.column_stack(
                (
                    basis,
                    np.cos(angle),
                    np.sin(angle),
                )
            )
            design_fit = np.vstack(
                (
                    design[voiced] * weights[:, None],
                    penalty,
                )
            )
            coefficients, *_ = np.linalg.lstsq(
                design_fit,
                target,
                rcond=None,
            )
            amplitude = float(
                np.hypot(
                    coefficients[m],
                    coefficients[m + 1],
                )
            )
            if not (
                config.vib2_seed_candidate_amplitude_min_semitones
                <= amplitude
                <= config.vib2_seed_candidate_amplitude_max_semitones
            ):
                continue
            residual = target - design_fit @ coefficients
            candidate = (
                float(np.dot(residual, residual)),
                float(rate),
                float(
                    np.arctan2(
                        coefficients[m],
                        coefficients[m + 1],
                    )
                ),
                amplitude,
            )
            if reference is None or candidate[0] < reference[0]:
                reference = candidate
        self.assertIsNotNone(reference)
        _, reference_rate, reference_phase, reference_amplitude = reference
        self.assertAlmostEqual(projected[0], reference_rate, places=12)
        phase_error = (projected[1] - reference_phase + np.pi) % (2.0 * np.pi) - np.pi
        self.assertAlmostEqual(phase_error, 0.0, places=10)
        self.assertAlmostEqual(projected[2], reference_amplitude, places=10)

    def test_attune_variable_projection_jacobian_matches_finite_difference(
        self,
    ) -> None:
        config = Config()
        detector = VibratoDetector(config=config)
        frame_rate = config.sr / config.h1
        times = np.arange(int(round(0.4 * frame_rate))) / frame_rate
        values = (
            61.0
            + 0.2 * times
            + 0.4 * np.sin(2.0 * np.pi * (5.5 * times + 0.7 * times**2) + 0.2)
        )
        voiced = np.ones(len(times), dtype=bool)
        voiced[20:23] = False
        confidence = np.ones(len(times))
        confidence[:10] = np.linspace(0.25, 1.0, 10)
        basis, _, _ = detector._spline_basis(times, causal=True)
        rate_basis = detector._rate_basis(times)
        parameters = np.array([0.25, 5.4, 5.9, 6.4])

        def residual(at):
            return detector._variable_projection_residual_jacobian(
                at,
                values,
                times,
                rate_basis,
                basis,
                voiced,
                confidence,
            )[1]

        _, _, analytic = detector._variable_projection_residual_jacobian(
            parameters,
            values,
            times,
            rate_basis,
            basis,
            voiced,
            confidence,
        )
        finite_difference = np.empty_like(analytic)
        step = 1e-6
        for parameter_i in range(4):
            above = parameters.copy()
            below = parameters.copy()
            above[parameter_i] += step
            below[parameter_i] -= step
            finite_difference[:, parameter_i] = (residual(above) - residual(below)) / (
                2.0 * step
            )
        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=2e-5,
            atol=3e-6,
        )

    def test_attune_live_warm_start_shifts_phase_and_skips_seed_scan(self) -> None:
        config = Config()
        detector = VibratoDetector(config=config)
        frame_rate = config.sr / config.h1
        frames = int(round(0.4 * frame_rate))
        old_lo, old_hi = 0, frames
        new_lo, new_hi = 17, frames + 17
        parameters = np.array([0.2, 6.0, 6.0, 6.0])
        shifted = detector._shift_live_parameters(
            parameters,
            old_lo,
            old_hi,
            new_lo,
            new_hi,
            frame_rate,
        )
        self.assertIsNotNone(shifted)
        expected_phase = (
            parameters[0] + 2.0 * np.pi * 6.0 * (new_lo - old_lo) / frame_rate + np.pi
        ) % (2.0 * np.pi) - np.pi
        self.assertAlmostEqual(shifted[0], expected_phase, places=12)
        np.testing.assert_allclose(shifted[1:], 6.0, atol=1e-12)

        absolute_times = np.arange(new_lo, new_hi) / frame_rate
        values = 60.0 + 0.4 * np.sin(2.0 * np.pi * 6.0 * absolute_times + parameters[0])
        with patch.object(
            detector,
            "_seed_rate_and_phase",
            side_effect=AssertionError("warm fitting must not run the grid scan"),
        ):
            fit = detector._fit_curves(
                values,
                frame_rate,
                causal=True,
                initial_phase_parameters=shifted,
            )
        self.assertGreater(fit.rates[-1], 0.0)
        self.assertAlmostEqual(fit.rates[-1], 6.0, places=5)

    def test_attune_incremental_live_fit_reuses_previous_phase_model(self) -> None:
        config = Config()
        detector = VibratoDetector(config=config)
        frame_rate = config.sr / config.h1
        history = int(round(config.vib2_live_sec * frame_rate))
        step = int(round(frame_rate / config.vib2_live_analysis_hz))
        frames = history + step
        times = np.arange(frames) / frame_rate
        values = 60.0 + 0.4 * np.sin(2.0 * np.pi * 6.0 * times + 0.2)
        pitches = [
            Pitch(
                time=(i * config.h1 + 0.5 * config.w1) / config.sr,
                volume=0.1,
                unvoiced_prob=0.0,
                live_distance=0.0,
                config=config,
                value=float(value),
            )
            for i, value in enumerate(values)
        ]
        pitch_data = PitchData(config)
        vibrato_data = VibratoData(config)

        original_seed = detector._seed_rate_and_phase
        seed_calls = 0

        def counted_seed(*args, **kwargs):
            nonlocal seed_calls
            seed_calls += 1
            return original_seed(*args, **kwargs)

        with patch.object(
            detector,
            "_seed_rate_and_phase",
            side_effect=counted_seed,
        ):
            pitch_data.write(pitches[:history], start_time=0.0)
            detector.extend(vibrato_data, pitch_data)
            self.assertEqual(seed_calls, 1)
            pitch_data.write(
                pitches[history:],
                start_time=history / frame_rate,
            )
            detector.extend(vibrato_data, pitch_data)

        self.assertEqual(seed_calls, 1)
        self.assertAlmostEqual(
            float(vibrato_data.rates[vibrato_data.computed_until - 1]),
            6.0,
            places=4,
        )

    def test_mcleod_prony_core_recovers_source_fixture(self) -> None:
        frame_rate = 44_100 / 1_024
        window_size = int(np.ceil(0.4 * frame_rate))
        times = np.arange(window_size, dtype=np.float64) / frame_rate
        values = 63.25 + 0.4 * np.sin(2.0 * np.pi * 5.5 * times + 0.37)
        fit = McLeod.fit_single_sine(values)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.rate_hz(frame_rate), 5.5, places=10)
        self.assertAlmostEqual(fit.amplitude_semitones, 0.4, places=10)
        self.assertAlmostEqual(fit.center_midi, 63.25, places=10)
        self.assertLess(fit.mean_squared_error, 1e-20)
        self.assertIsNone(McLeod.fit_single_sine(np.full(window_size, 63.25)))

    def test_mcleod_tartini_adapter_uses_native_geometry(self) -> None:
        estimate = McLeod().estimate(self.constant)
        target = self.constant.score_mask
        supported = target & estimate.detected
        self.assertGreater(float(np.mean(supported[target])), 0.95)
        self.assertLess(
            float(np.mean(np.abs(estimate.rate_hz[supported] - 5.5))),
            0.05,
        )
        self.assertLess(
            float(np.mean(np.abs(estimate.width_cents[supported] - 80.0))),
            6.0,
        )
        self.assertAlmostEqual(
            estimate.metadata["analysis_frame_rate"],
            44_100 / 1_024,
        )
        self.assertEqual(estimate.metadata["window_samples"], 18)

    def test_mcleod_tartini_withholds_width_below_one_cycle(self) -> None:
        times = self.constant.times
        target = self.constant.score_mask
        center = self.constant.center_midi.copy()
        pitch = self.constant.pitch_midi.copy()
        rate = self.constant.rate_hz.copy()
        width = self.constant.width_cents.copy()
        local_time = times[target] - times[target][0]
        pitch[target] = center[target] + 0.4 * np.sin(2.0 * np.pi * 2.0 * local_time)
        rate[target] = 2.0
        width[target] = 80.0
        slow = replace(
            self.constant,
            case_id="slow_mcleod_fixture",
            pitch_midi=pitch,
            rate_hz=rate,
            width_cents=width,
        )
        estimate = McLeod().estimate(slow)
        start, end = slow.scored_time_bounds
        interior = target & (times >= start + 0.25) & (times < end - 0.25)
        self.assertTrue(np.any(estimate.rate_hz[interior] > 0.0))
        self.assertFalse(np.any(estimate.width_cents[interior] > 0.0))
        self.assertFalse(estimate.detected[interior].any())

    def test_rossignol_recovers_clean_constant_vibrato(self) -> None:
        estimate = Rossignol().estimate(self.constant)
        supported = self.constant.score_mask & estimate.detected
        note_indices = np.flatnonzero(self.constant.score_mask)
        note_start = float(self.constant.times[note_indices[0]])
        note_end = float(
            self.constant.times[note_indices[-1]] + 1.0 / self.constant.frame_rate
        )
        interior = (
            self.constant.score_mask
            & (self.constant.times >= note_start + 0.175)
            & (self.constant.times <= note_end - 0.175)
        )
        # Complete 0.35 s portions cannot cross either note boundary, so test
        # recovery over the source-defined valid interior.
        self.assertGreater(float(np.mean(supported[interior])), 0.85)
        self.assertLess(float(np.mean(np.abs(estimate.rate_hz[supported] - 5.5))), 0.1)
        self.assertLess(
            float(np.mean(np.abs(estimate.width_cents[supported] - 80.0))) / 80.0,
            0.05,
        )
        self.assertEqual(estimate.metadata["portion_seconds"], 0.35)
        self.assertEqual(estimate.metadata["minimum_pb"], 0.02)
        self.assertEqual(estimate.metadata["method_owned_smoothing"], "none")
        self.assertEqual(estimate.metadata["rate_limits_hz"], (3.0, 10.0))
        self.assertEqual(estimate.metadata["minimum_width_cents"], 10.0)
        self.assertIn("filled_unvoiced_frames", estimate.metadata)

    def test_rossignol_note_gate_matches_attune_acceptance_floor(self) -> None:
        estimator = Rossignol()
        self.assertFalse(
            estimator._meets_note_gate(np.full(20, 2.9), np.full(20, 80.0))
        )
        self.assertFalse(
            estimator._meets_note_gate(np.full(20, 10.1), np.full(20, 80.0))
        )
        self.assertFalse(estimator._meets_note_gate(np.full(20, 5.5), np.full(20, 9.9)))
        self.assertTrue(estimator._meets_note_gate(np.full(20, 5.5), np.full(20, 10.0)))

    def test_rossignol_confines_analysis_and_interpolation_to_notes(self) -> None:
        frame_rate = 100.0
        times = np.arange(200, dtype=np.float64) / frame_rate
        pitch = np.empty(200, dtype=np.float64)
        pitch[:100] = 60.0 + 0.4 * np.sin(2.0 * np.pi * 5.5 * times[:100])
        pitch[100:] = 72.0
        # The internal dropout is interpolated, but the octave step at 1 s is
        # a hard boundary and cannot become part of either note's envelopes.
        pitch[45:48] = np.nan
        example = VibratoExample(
            case_id="rossignol_note_scope",
            scenario="constant_then_straight",
            split="fixture",
            times=times,
            pitch_midi=pitch,
            center_midi=np.where(times < 1.0, 60.0, 72.0),
            rate_hz=np.where(times < 1.0, 5.5, 0.0),
            width_cents=np.where(times < 1.0, 80.0, 0.0),
            is_vibrato=times < 1.0,
            metadata={
                "analysis_note_bounds": [
                    (0.0, 1.0, 60.0),
                    (1.0, 2.0, 72.0),
                ],
            },
        )
        estimate = Rossignol().estimate(example)
        self.assertTrue(np.all(estimate.detected[45:48]))
        self.assertFalse(np.any(estimate.detected[100:]))
        self.assertFalse(np.any(estimate.detected[times < 0.175]))
        self.assertFalse(np.any(estimate.detected[(times > 0.825) & (times < 1.0)]))
        self.assertEqual(estimate.metadata["filled_unvoiced_frames"], 3)
        self.assertEqual(estimate.metadata["accepted_notes"], 1)

    def test_rossignol_rejects_mixed_controls(self) -> None:
        noisy = SyntheticDataset.build(
            replicates=1,
            seed=7,
            frame_rate=100.0,
            noise_cents=5.0,
            dropout_probability=0.0,
            outlier_probability=0.0,
        )
        raw = VibratoBenchmarker().run(
            noisy,
            [Rossignol()],
            strict=True,
        )
        summary = VibratoBenchmarker.summarize(raw).iloc[0]
        self.assertLess(summary["frame_false_alarm"], 0.05)
        self.assertGreater(summary["frame_precision"], 0.9)

    def test_herrera_bonada_native_stft_recovers_clean_vibrato(self) -> None:
        example = self.isolated_vibrato_example()
        estimate = HerreraBonada().estimate(example)
        supported = estimate.detected
        self.assertGreater(float(np.mean(supported)), 0.85)
        self.assertLess(float(np.mean(np.abs(estimate.rate_hz[supported] - 5.5))), 0.1)
        self.assertLess(
            float(np.mean(np.abs(estimate.width_cents[supported] - 80.0))) / 80.0,
            0.05,
        )
        self.assertEqual(estimate.metadata["window_function"], "periodic Hamming")

    def test_herrera_bonada_exposes_yang_comparison_geometry(self) -> None:
        estimator = HerreraBonada(window_mode="yang_comparison")
        estimate = estimator.estimate(self.isolated_vibrato_example())
        self.assertEqual(estimator.name, "herrera_bonada_yang_window")
        self.assertEqual(estimate.metadata["window_mode"], "yang_comparison")
        self.assertAlmostEqual(estimate.metadata["window_seconds"], 0.125)
        self.assertAlmostEqual(estimate.metadata["hop_fraction"], 0.25)

    def test_vsf_reproduces_published_synthetic_rate_sweep_accuracy(self) -> None:
        for rate_hz in np.arange(4.0, 8.0 + 0.25, 0.5):
            example = self.isolated_vibrato_example(
                duration=4.0,
                rate_hz=float(rate_hz),
                width_cents=100.0,
            )
            estimate = VenturaSousaFerreira().estimate(example)
            supported = estimate.detected
            self.assertGreater(float(np.mean(supported)), 0.5)
            relative_error = float(
                np.mean(np.abs(estimate.rate_hz[supported] - rate_hz)) / rate_hz
            )
            self.assertLess(relative_error, 0.001)

    def test_driedger_template_normalization_and_fast_salience(self) -> None:
        template = Driedger.build_template(5.5, 80.0)
        self.assertAlmostEqual(float(np.sum(template.values[template.values > 0])), 1.0)
        self.assertAlmostEqual(
            float(np.sum(template.values[template.values < 0])), -1.0
        )
        image = np.zeros((80, 100), dtype=np.float32)
        height, width = template.values.shape
        image[20 : 20 + height, 15 : 15 + width][template.positive_support] = 1.0
        salience = Driedger._template_salience(image, template)
        self.assertGreater(float(np.max(salience)), 0.99)

    def test_driedger_frame_salience_matches_literal_equation_5_reduction(self) -> None:
        rng = np.random.default_rng(1729)
        # Deliberately shorter than the largest template so frequency-boundary
        # handling is exercised rather than hidden by a wide spectrogram.
        image = rng.normal(size=(20, 100))
        for rate_hz, extent_cents in ((5.0, 50.0), (7.0, 100.0)):
            template = Driedger.build_template(rate_hz, extent_cents)
            literal = np.max(Driedger._template_salience(image, template), axis=0)
            optimized = Driedger._template_frame_salience(image, template)
            np.testing.assert_array_equal(optimized, literal)

    def test_driedger_default_is_published_detection_bank_and_global_range(
        self,
    ) -> None:
        estimator = Driedger()
        self.assertEqual(estimator.template_mode, "detection")
        self.assertEqual(len(estimator.templates), 30)
        self.assertEqual(estimator.minimum_hz, 196.0)
        self.assertEqual(estimator.maximum_hz, 3000.0)
        self.assertEqual(
            sorted({template.rate_hz for template in estimator.templates}),
            [5.0, 5.5, 6.0, 6.5, 7.0],
        )
        self.assertEqual(
            sorted({template.extent_cents for template in estimator.templates}),
            [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        )

    def test_driedger_benchmark_range_is_separate_superset_row(self) -> None:
        published = Driedger()
        estimator = Driedger(template_mode="benchmark_range")
        self.assertEqual(estimator.name, "driedger_benchmark_range")
        self.assertEqual(estimator.template_mode, "benchmark_range")
        self.assertEqual(len(estimator.templates), 165)
        self.assertEqual(
            estimator.template_rates_hz, tuple(np.arange(3.0, 10.0 + 0.25, 0.5))
        )
        self.assertEqual(
            estimator.template_extents_cents,
            (5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
        )
        published_parameters = {
            (template.rate_hz, template.extent_cents)
            for template in published.templates
        }
        benchmark_parameters = {
            (template.rate_hz, template.extent_cents)
            for template in estimator.templates
        }
        self.assertLess(published_parameters, benchmark_parameters)

    def test_driedger_frequency_range_uses_score_only_with_attune_padding(self) -> None:
        score_backed = replace(
            self.constant,
            metadata={
                **self.constant.metadata,
                "source_midi": "/not/read/score.mid",
                "analysis_note_bounds": [
                    (0.0, 1.0, 55.0),
                    (1.0, 2.0, 72.0),
                ],
            },
        )
        minimum_hz, maximum_hz, source = Driedger._score_padded_frequency_range(
            score_backed
        )
        config = Config()
        self.assertEqual(source, "score_padded")
        self.assertAlmostEqual(minimum_hz, config.midi_to_freq(53.0))
        self.assertAlmostEqual(maximum_hz, config.midi_to_freq(96.0))

        annotation_only = replace(
            score_backed,
            metadata={
                **score_backed.metadata,
                "source_midi": "",
                "analysis_note_bounds": [(0.0, 1.0, 12.0)],
            },
        )
        self.assertEqual(
            Driedger._score_padded_frequency_range(annotation_only),
            (196.0, 3000.0, "global_fallback"),
        )

    def test_audio_only_estimator_is_skipped_without_audio(self) -> None:
        estimator = Driedger(template_mode="detection")
        raw = VibratoBenchmarker().run([self.constant], [estimator], strict=True)
        self.assertTrue(bool(raw.iloc[0]["skipped"]))
        self.assertEqual(raw.iloc[0]["error"], "")
        self.assertIn("audio", raw.iloc[0]["skip_reason"])
        summary = VibratoBenchmarker.summarize(raw).iloc[0]
        self.assertEqual(summary["cases"], 0)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["errors"], 0)

    def test_scoring_holds_numerical_libraries_to_one_thread(self) -> None:
        """Oversubscribed BLAS pools cost Attune more than the fits themselves."""
        # The module-level guard defers to a deliberate export, so assert it
        # left a value behind rather than the specific count. Scoring itself is
        # single-threaded either way.
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertIsNotNone(os.environ.get(name))

        try:
            from threadpoolctl import threadpool_info
        except ImportError:
            self.skipTest("threadpoolctl is required for the runtime thread guard")
        observed: list[list[int]] = []

        def record(example):
            observed.append([layer["num_threads"] for layer in threadpool_info()])
            return VibratoEstimate(
                example.rate_hz.copy(),
                example.width_cents.copy(),
                example.is_vibrato.copy(),
            )

        VibratoBenchmarker().run(
            [self.constant],
            [CallableDetector("thread_fixture", record)],
            strict=True,
        )
        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0], "no numerical library reported a thread pool")
        self.assertEqual(set(observed[0]), {1})

    def test_parallel_scoring_matches_serial_and_isolates_prepared_state(self) -> None:
        examples = [
            replace(self.constant, case_id="parallel-a", metadata={}),
            replace(self.constant, case_id="parallel-b", metadata={}),
        ]
        benchmarker = VibratoBenchmarker()
        serial = benchmarker.summarize(
            benchmarker.run(
                examples,
                [
                    _PreparedFixtureEstimator("fixture-a"),
                    _PreparedFixtureEstimator("fixture-b"),
                ],
                strict=True,
                workers=1,
            )
        )
        parallel_raw = benchmarker.run(
            examples,
            [
                _PreparedFixtureEstimator("fixture-a"),
                _PreparedFixtureEstimator("fixture-b"),
            ],
            strict=True,
            workers=4,
        )
        self.assertEqual(
            parallel_raw.attrs["scoring_backends"],
            ("spawn_process_pool",),
        )
        self.assertTrue(
            parallel_raw["runtime_warnings"]
            .str.contains("spawned fixture warning")
            .all()
        )
        parallel = benchmarker.summarize(parallel_raw)
        ignored = [
            "compute_seconds",
            "audio_per_compute",
            "wall_compute_seconds",
            "audio_per_wall_compute",
        ]
        pd.testing.assert_frame_equal(
            serial.drop(columns=ignored),
            parallel.drop(columns=ignored),
        )

    def test_estimator_warnings_are_retained_in_reports_not_emitted(self) -> None:
        def warns(example):
            warnings.warn("fixture numerical warning", RuntimeWarning)
            return VibratoEstimate(
                example.rate_hz.copy(),
                example.width_cents.copy(),
                example.is_vibrato.copy(),
            )

        estimator = CallableDetector("warning_fixture", warns)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            raw = VibratoBenchmarker().run(
                [self.constant],
                [estimator],
                strict=True,
            )
        self.assertEqual(emitted, [])
        self.assertIn("fixture numerical warning", raw.iloc[0]["runtime_warnings"])

    def test_audio_prepare_is_called_once_per_shared_group_outside_timer(self) -> None:
        calls = 0

        class AudioFixtureEstimator:
            name = "audio_fixture"
            description = "fixture"
            requires = {"audio"}

            def prepare(self, example):
                nonlocal calls
                calls += 1

            def estimate(self, example):
                zeros = np.zeros(len(example.times), dtype=np.float64)
                return VibratoEstimate(zeros, zeros.copy(), zeros.astype(bool))

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "fixture.wav"
            sf.write(audio_path, np.zeros(800), 8_000)
            examples = [
                replace(example, audio_path=str(audio_path))
                for example in self.examples
            ]
            raw = VibratoBenchmarker().run(
                examples,
                [AudioFixtureEstimator()],
                strict=True,
            )
        self.assertEqual(calls, 1)
        self.assertFalse(raw["skipped"].any())

    def test_driedger_release_loader_builds_nine_by_three_audio_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(9):
                base = f"Item{index}_excerpt"
                for condition in ("0dB", "-5dB", "-10dB"):
                    sf.write(
                        root / f"{base}_mix_{condition}.wav",
                        np.zeros(800),
                        8_000,
                    )
                if index == 0:
                    (root / f"{base}_vibrato.csv").write_text(
                        "0.01,1,0.05\n",
                        encoding="utf-8",
                    )
            examples = DriedgerDataset.load(root)
        self.assertEqual(len(examples), 27)
        self.assertEqual({case.scenario for case in examples}, {"0dB", "-5dB", "-10dB"})
        self.assertTrue(any(case.has_vibrato for case in examples))
        self.assertTrue(all(case.audio_path is not None for case in examples))

    def test_production_adapter_preserves_shifted_note_time(self) -> None:
        shifted_metadata = dict(self.constant.metadata)
        shifted_metadata["analysis_note_bounds"] = [
            (start + 8.0, end + 8.0, midi)
            for start, end, midi in self.constant.metadata["analysis_note_bounds"]
        ]
        shifted = replace(
            self.constant,
            times=self.constant.times + 8.0,
            case_id="shifted_constant",
            metadata=shifted_metadata,
        )
        reference = Attune().estimate(self.constant)
        estimate = Attune().estimate(shifted)
        target = self.constant.score_mask
        np.testing.assert_allclose(estimate.rate_hz[target], reference.rate_hz[target])
        np.testing.assert_allclose(
            estimate.width_cents[target],
            reference.width_cents[target],
        )

    def test_curve_metrics_score_missing_estimates_as_zero(self) -> None:
        zeros = np.zeros(len(self.constant.times), dtype=np.float64)
        missing = CallableDetector(
            "missing",
            lambda example: VibratoEstimate(
                zeros.copy(),
                zeros.copy(),
                np.zeros(len(example.times), dtype=bool),
            ),
        )
        raw = VibratoBenchmarker().run([self.constant], [missing], strict=True)
        row = raw.iloc[0]
        self.assertAlmostEqual(row["rate_mae_hz"], 5.5)
        self.assertAlmostEqual(row["amplitude_mae_semitones"], 0.4)
        self.assertEqual(row["active_curve_within_tolerance"], 0.0)

    def test_default_extent_tolerance_rejects_zero_for_point_15_semitones(self) -> None:
        target = self.constant.score_mask
        subtle = replace(
            self.constant,
            case_id="subtle_missing_fixture",
            width_cents=np.where(target, 30.0, 0.0),
        )
        zeros = np.zeros(len(subtle.times), dtype=np.float64)
        missing = CallableDetector(
            "missing_subtle",
            lambda example: VibratoEstimate(
                zeros.copy(),
                zeros.copy(),
                zeros.astype(bool),
            ),
        )
        row = VibratoBenchmarker().run([subtle], [missing], strict=True).iloc[0]
        self.assertAlmostEqual(row["active_amplitude_mae_semitones"], 0.15)
        self.assertEqual(row["active_amplitude_within_tolerance"], 0.0)

    def test_relative_curve_diagnostic_uses_yang_relative_error(self) -> None:
        scaled = CallableDetector(
            "scaled",
            lambda example: VibratoEstimate(
                0.8 * example.rate_hz,
                0.75 * example.width_cents,
                example.is_vibrato.copy(),
            ),
        )
        row = (
            VibratoBenchmarker()
            .run(
                [self.constant],
                [scaled],
                strict=True,
            )
            .iloc[0]
        )
        self.assertAlmostEqual(row["relative_rate_accuracy"], 0.8)
        self.assertAlmostEqual(row["relative_extent_accuracy"], 0.75)
        self.assertAlmostEqual(row["overall_curve_accuracy"], 0.775)
        self.assertEqual(row["aggregate_precision"], 0.0)
        self.assertEqual(row["aggregate_recall"], 0.0)
        self.assertEqual(row["aggregate_f1"], 0.0)

    def test_yang_half_cycle_annotations_build_parameter_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            audio_path = directory / "Example.wav"
            area_path = directory / "Example-Annotation-new.csv"
            extrema_path = directory / "Example-Annotation-Stat.csv"
            audio_path.touch()
            area_path.write_text("0.0,0.0,0.8\n", encoding="utf-8")
            extrema_path.write_text(
                "0.05,0.050000\n0.15\n0.25\n0.35\n0.45\n0.55\n0.65\n0.75\n",
                encoding="utf-8",
            )
            recording = YangDataset.Recording(
                audio_path=audio_path,
                area_path=area_path,
                extrema_path=extrema_path,
                instrument="Violin",
                performer="Fixture",
            )
            times = np.arange(0.0, 1.0, 0.01)
            pitch = 60.0 + 0.2 * np.sin(2.0 * np.pi * 5.0 * times)
            examples = YangDataset._examples_from_annotations(
                recording,
                times,
                pitch,
                pitch_stage="fixture",
            )

        self.assertEqual(len(examples), 1)
        example = examples[0]
        target = example.score_mask
        np.testing.assert_allclose(example.rate_hz[target], 5.0)
        np.testing.assert_allclose(example.width_cents[target], 40.0)
        self.assertEqual(example.metadata["half_cycles"], 7)
        self.assertTrue(example.audio_path.endswith("Example.wav"))
        self.assertEqual(
            example.metadata["note_boundary_source"],
            "annotated_vibrato_spans_pseudo_notes",
        )

    def test_scoring_crop_ignores_context_false_positives(self) -> None:
        target = self.constant.score_mask
        outside_only = CallableDetector(
            "outside_only",
            lambda example: VibratoEstimate(
                np.zeros(len(example.times), dtype=np.float64),
                np.zeros(len(example.times), dtype=np.float64),
                ~target,
            ),
        )
        raw = VibratoBenchmarker().run(
            [self.constant],
            [outside_only],
            strict=True,
        )
        row = raw.iloc[0]
        self.assertEqual(row["_frame_fp"], 0)
        self.assertEqual(row["_frame_fn"], int(np.sum(target)))
        self.assertAlmostEqual(row["scored_seconds"], self.constant.scored_duration)

    def test_shared_contour_is_estimated_once_then_scored_per_case(self) -> None:
        calls = 0

        def estimate(example):
            nonlocal calls
            calls += 1
            zeros = np.zeros(len(example.times), dtype=np.float64)
            return VibratoEstimate(zeros, zeros.copy(), zeros.astype(bool))

        raw = VibratoBenchmarker().run(
            self.examples,
            [CallableDetector("counted", estimate)],
            strict=True,
        )
        self.assertEqual(calls, 1)
        self.assertEqual(len(raw), len(self.examples))
        self.assertTrue(np.all(raw["analysis_reuse_count"] == len(self.examples)))
        self.assertAlmostEqual(
            float(raw["audio_seconds"].sum()),
            self.constant.duration,
        )

    def test_parameter_prf_requires_the_final_detection_gate(self) -> None:
        exact = CallableDetector(
            "exact",
            lambda example: VibratoEstimate(
                example.rate_hz.copy(),
                example.width_cents.copy(),
                np.zeros(len(example.times), dtype=bool),
            ),
        )
        raw = VibratoBenchmarker().run([self.constant], [exact], strict=True)
        row = raw.iloc[0]
        self.assertEqual(row["curve_within_tolerance"], 1.0)
        self.assertEqual(row["rate_mae_hz"], 0.0)
        self.assertEqual(row["amplitude_mae_semitones"], 0.0)
        summary = VibratoBenchmarker.summarize(raw).iloc[0]
        self.assertEqual(summary["overall_curve_accuracy"], 1.0)
        self.assertEqual(summary["frame_accuracy"], 0.0)
        self.assertTrue(np.isnan(summary["aggregate_precision"]))
        self.assertEqual(summary["aggregate_recall"], 0.0)
        self.assertEqual(summary["aggregate_f1"], 0.0)

    def test_parameter_prf_counts_negative_predictions_as_false_positives(self) -> None:
        straight = next(case for case in self.examples if case.scenario == "straight")
        always_vibrato = CallableDetector(
            "always_vibrato",
            lambda example: VibratoEstimate(
                np.full(len(example.times), 5.5, dtype=np.float64),
                np.full(len(example.times), 80.0, dtype=np.float64),
                np.ones(len(example.times), dtype=bool),
            ),
        )
        raw = VibratoBenchmarker().run(
            [self.constant, straight],
            [always_vibrato],
            strict=True,
        )
        straight_row = raw.loc[raw["scenario"] == "straight"].iloc[0]
        self.assertEqual(
            straight_row["_extent_fp"],
            int(np.sum(straight.score_mask)),
        )
        self.assertEqual(
            straight_row["_rate_fp"],
            int(np.sum(straight.score_mask)),
        )
        summary = VibratoBenchmarker.summarize(raw).iloc[0]
        self.assertLess(summary["aggregate_precision"], 1.0)
        self.assertEqual(summary["aggregate_recall"], 1.0)
        self.assertLess(summary["aggregate_f1"], 1.0)

    def test_aggregate_parameter_prf_is_the_component_macro_average(self) -> None:
        extent_only_match = CallableDetector(
            "extent_only_match",
            lambda example: VibratoEstimate(
                example.rate_hz + 2.0,
                example.width_cents.copy(),
                example.is_vibrato.copy(),
            ),
        )
        row = VibratoBenchmarker.summarize(
            VibratoBenchmarker().run(
                [self.constant],
                [extent_only_match],
                strict=True,
            )
        ).iloc[0]
        self.assertEqual(row["extent_precision"], 1.0)
        self.assertEqual(row["extent_recall"], 1.0)
        self.assertEqual(row["extent_f1"], 1.0)
        self.assertEqual(row["rate_precision"], 0.0)
        self.assertEqual(row["rate_recall"], 0.0)
        self.assertEqual(row["rate_f1"], 0.0)
        self.assertEqual(row["aggregate_precision"], 0.5)
        self.assertEqual(row["aggregate_recall"], 0.5)
        self.assertEqual(row["aggregate_f1"], 0.5)

    def test_center_accuracy_is_optional_and_penalizes_missing_capable_estimates(
        self,
    ) -> None:
        zeros = np.zeros(len(self.constant.times), dtype=np.float64)
        supported = CallableDetector(
            "supported_center",
            lambda example: VibratoEstimate(
                example.rate_hz.copy(),
                example.width_cents.copy(),
                example.is_vibrato.copy(),
                example.center_midi + 0.05,
            ),
        )
        supported.scores_center = True
        unsupported = CallableDetector(
            "unsupported_center",
            lambda example: VibratoEstimate(
                example.rate_hz.copy(),
                example.width_cents.copy(),
                example.is_vibrato.copy(),
            ),
        )
        unsupported.scores_center = False
        missing = CallableDetector(
            "missing_center",
            lambda example: VibratoEstimate(
                example.rate_hz.copy(),
                example.width_cents.copy(),
                example.is_vibrato.copy(),
                np.full(len(example.times), np.nan, dtype=np.float64),
            ),
        )
        missing.scores_center = True

        benchmarker = VibratoBenchmarker(center_accuracy_tolerance_cents=10.0)
        summary = benchmarker.summarize(
            benchmarker.run(
                [self.constant],
                [supported, unsupported, missing],
                strict=True,
            )
        ).set_index("method")
        self.assertEqual(
            summary.loc["supported_center", "center_within_tolerance"], 1.0
        )
        self.assertAlmostEqual(summary.loc["supported_center", "center_mae_cents"], 5.0)
        self.assertTrue(
            np.isnan(summary.loc["unsupported_center", "center_within_tolerance"])
        )
        self.assertEqual(summary.loc["missing_center", "center_within_tolerance"], 0.0)
        self.assertEqual(summary.loc["missing_center", "center_coverage"], 0.0)

    def test_constant_and_extent_direction_profiles(self) -> None:
        u = np.array([0.0, 1.0])
        constant_rate, constant_amplitude = CocoDataset.profile_curves("constant", u)
        np.testing.assert_allclose(constant_rate, [6.0, 6.0])
        np.testing.assert_allclose(constant_amplitude, [0.5, 0.5])
        widening_rate, widening = CocoDataset.profile_curves("widening", u)
        narrowing_rate, narrowing = CocoDataset.profile_curves("narrowing", u)
        np.testing.assert_allclose(widening_rate, [6.0, 6.0])
        np.testing.assert_allclose(narrowing_rate, [6.0, 6.0])
        np.testing.assert_allclose(widening, [0.15, 1.0])
        np.testing.assert_allclose(narrowing, [1.0, 0.15])

    def test_seeded_profile_parameters_are_reproducible_and_bounded(self) -> None:
        for profile in (
            "constant",
            "accelerating",
            "decelerating",
            "widening",
            "narrowing",
        ):
            with self.subTest(profile=profile):
                first = CocoDataset.sample_profile_parameters(profile, seed=1234)
                repeated = CocoDataset.sample_profile_parameters(profile, seed=1234)
                changed = CocoDataset.sample_profile_parameters(profile, seed=1235)
                self.assertEqual(first, repeated)
                self.assertNotEqual(first, changed)

                rate, amplitude = CocoDataset.profile_curves(
                    profile,
                    np.linspace(0.0, 1.0, 101),
                    parameters=first,
                )
                self.assertGreaterEqual(float(rate.min()), NATIVE_RATE_RANGE_HZ[0])
                self.assertLessEqual(float(rate.max()), NATIVE_RATE_RANGE_HZ[1])
                self.assertGreaterEqual(
                    float(amplitude.min()),
                    NATIVE_AMPLITUDE_RANGE_SEMITONES[0],
                )
                self.assertLessEqual(
                    float(amplitude.max()),
                    NATIVE_AMPLITUDE_RANGE_SEMITONES[1],
                )

                if profile in {"accelerating", "decelerating"}:
                    span = float(rate.max() - rate.min())
                    self.assertGreaterEqual(span, CHANGING_RATE_SPAN_RANGE_HZ[0])
                    self.assertLessEqual(span, CHANGING_RATE_SPAN_RANGE_HZ[1])
                if profile in {"widening", "narrowing"}:
                    span = float(amplitude.max() - amplitude.min())
                    self.assertGreaterEqual(
                        span,
                        CHANGING_AMPLITUDE_SPAN_RANGE_SEMITONES[0],
                    )
                    self.assertLessEqual(
                        span,
                        CHANGING_AMPLITUDE_SPAN_RANGE_SEMITONES[1],
                    )

    def test_balanced_coco_selection_fixes_composition_but_randomizes_records(
        self,
    ) -> None:
        records: list[CocoChorales.Stem] = []
        for ensemble in ("brass", "random", "string", "woodwind"):
            for instrument in ("high", "low"):
                for index in range(20):
                    track = f"{ensemble}_track{index:06d}_{instrument}"
                    records.append(
                        CocoChorales.Stem(
                            split="test",
                            shard="0.tar.bz2",
                            track=track,
                            ensemble=ensemble,
                            stem=f"1_{instrument}",
                            stem_voice=1,
                            f0_voice=0,
                            instrument=instrument,
                            wav_member=f"{track}/stems_audio/1_{instrument}.wav",
                            midi_member=f"{track}/stems_midi/1_{instrument}.mid",
                            metadata_member=f"{track}/metadata.yaml",
                            mix_midi_member=f"{track}/mix.mid",
                            f0_path=f"f0/test/{track}.pickle",
                        )
                    )

        first = CocoChorales.sample_balanced_records(
            records,
            count=16,
            seed=17,
        )
        repeated = CocoChorales.sample_balanced_records(
            records,
            count=16,
            seed=17,
        )
        changed = CocoChorales.sample_balanced_records(
            records,
            count=16,
            seed=18,
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(
            [record.track_id for record in first],
            [record.track_id for record in changed],
        )
        expected = Counter(
            {
                (ensemble, instrument): 2
                for ensemble in ("brass", "random", "string", "woodwind")
                for instrument in ("high", "low")
            }
        )
        self.assertEqual(
            Counter((record.ensemble, record.instrument) for record in first),
            expected,
        )
        self.assertEqual(
            Counter((record.ensemble, record.instrument) for record in changed),
            expected,
        )

    def test_pitch_stage_contour_diagnostics_cover_changing_profiles(self) -> None:
        diagnostics = CocoDataset._pitch_stage_contour_diagnostics(
            np.array([60.0, 60.2, 60.4, 60.6]),
            np.array([60.1, np.nan, 60.3, 72.6]),
        )
        self.assertEqual(diagnostics["dropout_fraction"], 0.25)
        self.assertEqual(diagnostics["octave_error_fraction"], 0.25)
        self.assertAlmostEqual(diagnostics["contour_bias_cents"], 400.0)
        self.assertAlmostEqual(
            diagnostics["contour_mae_cents"],
            1_220.0 / 3.0,
        )

    def test_automatic_yin_window_uses_four_guarded_periods(self) -> None:
        self.assertEqual(CocoDataset.automatic_yin_window_size(196.0), 2048)
        self.assertEqual(CocoDataset.automatic_yin_window_size(116.54094), 4096)
        self.assertEqual(CocoDataset.automatic_yin_window_size(30.0), 16384)

        score_fmin = 55.0
        target_fmin = score_fmin * 2.0 ** (-8.0 / 12.0)
        required_samples = int(np.ceil(4.0 * 44_100 / target_fmin))
        window = CocoDataset.automatic_yin_window_size(score_fmin)
        self.assertEqual(window & (window - 1), 0)
        self.assertGreaterEqual(window, required_samples)
        self.assertLess(window // 2, required_samples)

    def test_automatic_yin_window_rejects_invalid_inputs(self) -> None:
        for fmin in (0.0, -1.0, np.nan):
            with self.subTest(fmin=fmin):
                with self.assertRaises(ValueError):
                    CocoDataset.automatic_yin_window_size(fmin)
        with self.assertRaises(ValueError):
            CocoDataset.automatic_yin_window_size(196.0, padding_semitones=-1.0)
        with self.assertRaises(ValueError):
            CocoDataset.automatic_yin_window_size(196.0, periods=0.0)
        with self.assertRaises(ValueError):
            CocoDataset.automatic_yin_window_size(196.0, minimum=0)

    def test_production_pyin_frontend_and_hmm_share_exact_score_range(self) -> None:
        config = Config(fmin=220.0, fmax=440.0)
        detector = PitchDetector(config=config)
        smoother = PitchSmoother(mode="joint", config=config)

        self.assertAlmostEqual(detector.fmin, config.fmin)
        self.assertAlmostEqual(detector.fmax, config.fmax)
        self.assertAlmostEqual(smoother.fmin, config.fmin)
        self.assertAlmostEqual(smoother.fmax, config.fmax)

    def test_rate_direction_profiles_span_four_to_eight_hz(self) -> None:
        u = np.linspace(0.0, 1.0, 101)
        accelerating_rate, accelerating_amplitude = CocoDataset.profile_curves(
            "accelerating",
            u,
        )
        decelerating_rate, decelerating_amplitude = CocoDataset.profile_curves(
            "decelerating",
            u,
        )

        np.testing.assert_allclose(
            accelerating_rate[[0, -1]],
            [4.0, 8.0],
        )
        np.testing.assert_allclose(
            decelerating_rate[[0, -1]],
            [8.0, 4.0],
        )
        self.assertTrue(np.allclose(accelerating_amplitude, 0.5))
        self.assertTrue(np.allclose(decelerating_amplitude, 0.5))

        config = Config()
        self.assertLessEqual(config.vib2_fit_rate_min_hz, 4.0)
        self.assertGreaterEqual(config.vib2_fit_rate_max_hz, 8.0)
        self.assertLessEqual(
            config.vib2_seed_candidate_amplitude_min_semitones,
            0.15,
        )
        self.assertGreaterEqual(
            config.vib2_seed_candidate_amplitude_max_semitones,
            1.0,
        )
        self.assertEqual(config.vib2_seed_amplitude_min_semitones, 0.15)
        self.assertEqual(config.vib2_seed_amplitude_max_semitones, 1.0)
        self.assertGreaterEqual(
            config.vib2_fit_amplitude_max_semitones,
            1.0,
        )

    def test_yang_injection_range_constrains_positive_profiles(self) -> None:
        u = np.linspace(0.0, 1.0, 101)
        for profile in (
            "constant",
            "accelerating",
            "decelerating",
            "widening",
            "narrowing",
        ):
            rate, amplitude = CocoDataset.profile_curves(
                profile,
                u,
                injection_range="yang",
            )
            self.assertTrue(np.all((4.0 <= rate) & (rate <= 9.0)))
            self.assertTrue(np.all(amplitude >= 0.10))

        rate, amplitude = CocoDataset.profile_curves("none", u, injection_range="yang")
        self.assertTrue(np.all(rate == 0.0))
        self.assertTrue(np.all(amplitude == 0.0))

    def test_coco_injector_writes_pitch_bends_and_exact_curves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.mid"
            output = directory / "injected.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            instrument = pretty_midi.Instrument(program=40)
            instrument.notes.extend(
                [
                    pretty_midi.Note(velocity=90, pitch=69, start=0.0, end=1.0),
                    pretty_midi.Note(velocity=90, pitch=71, start=1.2, end=2.2),
                ]
            )
            midi.instruments.append(instrument)
            midi.write(str(source))
            truths = CocoDataset.inject_midi(
                source,
                output,
                profiles=("constant", "decelerating"),
                max_notes=2,
                seed=0,
            )
            repeated = CocoDataset.inject_midi(
                source,
                directory / "repeated.mid",
                profiles=("constant", "decelerating"),
                max_notes=2,
                seed=0,
            )
            injected = pretty_midi.PrettyMIDI(str(output))

        self.assertEqual(
            [truth.scenario for truth in truths],
            [truth.scenario for truth in repeated],
        )
        self.assertGreater(len(injected.instruments[0].pitch_bends), 100)
        self.assertTrue(
            any(bend.pitch != 0 for bend in injected.instruments[0].pitch_bends)
        )
        for truth, repeated_truth in zip(truths, repeated, strict=True):
            self.assertEqual(truth.parameter_seed, repeated_truth.parameter_seed)
            self.assertEqual(
                truth.profile_parameters, repeated_truth.profile_parameters
            )
            np.testing.assert_allclose(truth.rate_hz, repeated_truth.rate_hz)
            np.testing.assert_allclose(
                truth.amplitude_semitones,
                repeated_truth.amplitude_semitones,
            )
        constant = next(truth for truth in truths if truth.scenario == "constant")
        self.assertGreaterEqual(float(constant.rate_hz[0]), NATIVE_RATE_RANGE_HZ[0])
        self.assertLessEqual(float(constant.rate_hz[0]), NATIVE_RATE_RANGE_HZ[1])
        self.assertGreaterEqual(
            float(constant.amplitude_semitones[0]),
            NATIVE_AMPLITUDE_RANGE_SEMITONES[0],
        )
        self.assertLessEqual(
            float(constant.amplitude_semitones[0]),
            NATIVE_AMPLITUDE_RANGE_SEMITONES[1],
        )

    def test_coco_cases_share_full_stem_and_crop_only_the_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.mid"
            output = directory / "injected.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            instrument = pretty_midi.Instrument(program=40)
            instrument.notes.extend(
                [
                    pretty_midi.Note(velocity=90, pitch=69, start=0.2, end=1.2),
                    pretty_midi.Note(velocity=90, pitch=71, start=1.5, end=2.5),
                ]
            )
            midi.instruments.append(instrument)
            midi.write(str(source))
            truths = CocoDataset.inject_midi(
                source,
                output,
                profiles=("constant", "none"),
                max_notes=2,
                seed=0,
            )

        times = np.arange(300, dtype=np.float64) / 100.0
        pitch_frames = [
            SimpleNamespace(time=time, value=69.0, unvoiced_prob=0.0) for time in times
        ]
        pitch_data = SimpleNamespace(
            data=pitch_frames,
            frames_available=lambda: len(pitch_frames),
            _frame_time=lambda index: times[index],
        )
        cases = CocoDataset._examples_from_pitch_data(
            truths,
            pitch_data,
            SimpleNamespace(unv_thresh=0.5),
            split="test",
            metadata={"track": "fixture", "stem": "violin", "snr": "clean"},
        )
        self.assertEqual(len(cases), 2)
        self.assertEqual(len(cases[0].times), len(times))
        np.testing.assert_array_equal(cases[0].times, cases[1].times)
        np.testing.assert_array_equal(cases[0].pitch_midi, cases[1].pitch_midi)
        np.testing.assert_array_equal(
            cases[0].raw_pitch_midi,
            cases[0].pitch_midi,
        )
        self.assertTrue(
            np.all(np.isfinite(cases[0].commanded_pitch_midi[cases[0].score_mask]))
        )
        constant_case = next(case for case in cases if case.scenario == "constant")
        self.assertIn("raw_pitch_amplitude_gain", constant_case.metadata)
        self.assertIn("smoothed_pitch_rate_error_hz", constant_case.metadata)
        self.assertFalse(np.array_equal(cases[0].score_mask, cases[1].score_mask))
        self.assertEqual(
            cases[0].metadata["analysis_group"],
            cases[1].metadata["analysis_group"],
        )
        self.assertEqual(len(cases[0].metadata["analysis_note_bounds"]), 2)

    def test_reports_are_written(self) -> None:
        benchmarker = VibratoBenchmarker()
        estimators = [
            YangDT(),
            Attune(),
        ]
        raw = benchmarker.run(self.examples, estimators, strict=True)
        summary = benchmarker.summarize(raw)
        self.assertEqual(
            set(summary["method"]), {estimator.name for estimator in estimators}
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = benchmarker.write_reports(raw, directory)
            self.assertTrue(paths["summary"].is_file())
            self.assertTrue(paths["note_macro"].is_file())
            self.assertTrue(paths["raw_outputs"].is_dir())
            public_summary = pd.read_csv(paths["summary"])
            self.assertEqual(
                list(public_summary.columns),
                [
                    "Method",
                    "Overall F1",
                    "Overall Precision",
                    "Overall Recall",
                    "Extent F1",
                    "Extent Precision",
                    "Extent Recall",
                    "Rate F1",
                    "Rate Precision",
                    "Rate Recall",
                    "Center Accuracy (Attune)",
                    "Detection F1",
                    "Detection Precision",
                    "Detection Recall",
                    "Detection Accuracy",
                    "False Alarms",
                    "Skipped",
                    "Errors",
                    "Audio(s)/Compute(s)",
                ],
            )
            self.assertEqual(
                sorted(path.name for path in paths["raw_outputs"].iterdir()),
                sorted(estimator.name for estimator in estimators),
            )
            for estimator in estimators:
                method_dir = paths["raw_outputs"] / estimator.name
                self.assertEqual(
                    sorted(path.name for path in method_dir.iterdir()),
                    [
                        "by_scenario.csv",
                        "by_scenario_note_macro.csv",
                        "cases.csv",
                        "frames.csv",
                        "no_vibrato_by_instrument.csv",
                        "no_vibrato_by_pattern.csv",
                        "no_vibrato_cases.csv",
                        "no_vibrato_false_positive_frames.csv",
                        "no_vibrato_overview.csv",
                    ],
                )
                frames = pd.read_csv(method_dir / "frames.csv")
                self.assertEqual(
                    len(frames),
                    sum(int(np.sum(example.score_mask)) for example in self.examples),
                )
                self.assertTrue(
                    {
                        "commanded_pitch_midi",
                        "raw_pitch_midi",
                        "smoothed_pitch_midi",
                        "estimated_center_midi",
                        "estimated_rate_hz",
                        "estimated_amplitude_semitones",
                        "estimated_width_cents",
                    }.issubset(frames.columns)
                )
                no_vibrato_overview = pd.read_csv(
                    method_dir / "no_vibrato_overview.csv"
                )
                self.assertGreater(
                    int(no_vibrato_overview.loc[0, "negative_cases"]),
                    0,
                )
                no_vibrato_cases = pd.read_csv(method_dir / "no_vibrato_cases.csv")
                self.assertTrue(
                    {
                        "false_positive_rate",
                        "false_positive_runs",
                        "diagnostic_pattern",
                    }.issubset(no_vibrato_cases.columns)
                )

            filtered = raw.loc[raw["method"] == estimators[0].name]
            benchmarker.write_reports(filtered, directory)
            self.assertEqual(
                [path.name for path in paths["raw_outputs"].iterdir()],
                [estimators[0].name],
            )


if __name__ == "__main__":
    unittest.main()
