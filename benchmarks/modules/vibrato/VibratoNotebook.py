"""Notebook-facing orchestration for the unified vibrato benchmark."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display

from algorithms.Config import Config
from algorithms.PitchSmoother import PitchSmoother
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
from benchmarks.modules.vibrato.datasets.CocoRenderer import (
    COCO_SFIZZ_POLICY_VERSION,
    DEFAULT_SOUNDFONTS_ROOT,
    CocoRenderer,
)
from benchmarks.modules.vibrato.datasets.CocoDataset import (
    AUTOMATIC_YIN_WINDOW_MINIMUM,
    AUTOMATIC_YIN_WINDOW_PERIODS,
    CHANGING_AMPLITUDE_SPAN_RANGE_SEMITONES,
    CHANGING_RATE_SPAN_RANGE_HZ,
    NATIVE_AMPLITUDE_RANGE_SEMITONES,
    NATIVE_RATE_RANGE_HZ,
    PROFILE_PARAMETER_SAMPLER_VERSION,
    CocoDataset,
)
from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker


AVERAGE_PROFILES = ("constant", "none")
AVERAGE_SNR_CONDITIONS = ("clean",)
CHANGING_PROFILES = (
    "accelerating",
    "decelerating",
    "widening",
    "narrowing",
    "none",
)
CHANGING_SNR_CONDITIONS = ("clean", "20", "10")
BENCHMARK_METHODS = (
    "rossignol",
    "herrera_bonada",
    "ventura_sousa_ferreira",
    "driedger",
    "driedger_benchmark_range",
    "yang_dt",
    "yang_br",
    "mcleod",
    "attune",
)
# Bump this whenever a shipped estimator/configuration or measurement change
# makes an older summary scientifically stale. v5 adopted
# Config.vib2_curve_sec=0.1; v6 restores worker-count-independent timing with
# process-parallel scoring and per-worker CPU clocks; v7 records timings from
# the projected seed scan and analytic variable-projection Jacobian.
PRIMARY_COMPARISON_VERSION = "primary_v7"
# Attune's shipped 0.1 s knot spacing plus the candidates used by the offline
# replay sweep that selected it; see curve_penalty_sweep.md.
CURVE_SEC_ABLATION_VALUES = (0.4, 0.2, 0.15, 0.1)


@dataclass(frozen=True)
class NotebookConfig:
    """The few experiment choices readers may reasonably want to change."""

    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 4) - 1))
    preliminary_stems: int = 20
    yang_recordings: int = 2
    seed: int = 0
    minimum_note_seconds: float = 0.75
    rate_tolerance_hz: float = 0.5
    extent_tolerance_semitones: float = 0.05
    center_tolerance_cents: float = 25.0
    automatic_yin_window: bool = True
    force_pitch_redetection: bool = False
    force_preliminary_rerun: bool = False
    methods: tuple[str, ...] = BENCHMARK_METHODS


@dataclass(frozen=True)
class SuiteResult:
    output_dir: Path
    summary: pd.DataFrame


class VibratoNotebook:
    @staticmethod
    def _repo_root() -> Path:
        for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
            if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
                return candidate
        raise RuntimeError("could not locate the Attune repository root")

    @staticmethod
    def _csv(values: tuple[str, ...]) -> str:
        return ",".join(values)

    def __init__(
        self,
        config: NotebookConfig | None = None,
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config = config or NotebookConfig()
        self.repo_root = Path(repo_root).resolve() if repo_root else self._repo_root()
        self.python = self.repo_root / "at-venv" / "bin" / "python"
        if not self.python.is_file():
            self.python = Path(sys.executable)

        self.runner = (
            self.repo_root
            / "benchmarks"
            / "modules"
            / "vibrato"
            / "VibratoBenchmarker.py"
        )
        self.results_root = self.repo_root / "benchmarks" / "results" / "vibrato_runs"
        self.coco_artifact_root = (
            self.repo_root / "benchmarks" / "datasets" / "cocochorales_vibrato"
        )
        self.yang_root = self.repo_root / "benchmarks" / "datasets" / "vibrato"
        self.yang_cache_root = self.yang_root / "pitch_data" / "attune"
        self.soundfonts_root = DEFAULT_SOUNDFONTS_ROOT
        self.renderer = CocoRenderer(soundfonts_root=self.soundfonts_root)
        self.renderer.validate(set(self.renderer.PATCHES))
        self.sfizz_render = self.renderer.executable
        self.sampling_tag = (
            f"{COCO_SFIZZ_POLICY_VERSION}__{CocoChorales.BALANCED_SELECTION_POLICY}"
            f"__{PROFILE_PARAMETER_SAMPLER_VERSION}"
        )
        pd.set_option("display.float_format", lambda value: f"{value:.4f}")

    def show_configuration(self) -> pd.DataFrame:
        rows = [
            ("runner", self.runner),
            ("sfizz renderer", self.sfizz_render),
            ("soundfonts", self.soundfonts_root),
            ("workers", self.config.workers),
            ("preliminary stems", self.config.preliminary_stems),
            ("random seed", self.config.seed),
            ("result version", PRIMARY_COMPARISON_VERSION),
            ("Attune curve knot spacing", f"{Config.vib2_curve_sec:g} s"),
            ("pYIN/HMM range", "exact annotated F0 range"),
            (
                "YIN window",
                (
                    f"automatic: {AUTOMATIC_YIN_WINDOW_PERIODS:g} guarded "
                    f"periods, power of two, minimum "
                    f"{AUTOMATIC_YIN_WINDOW_MINIMUM}"
                    if self.config.automatic_yin_window
                    else "production default"
                ),
            ),
        ]
        frame = pd.DataFrame(rows, columns=("setting", "value"))
        display(frame)
        return frame

    def coco_command(
        self,
        *,
        suite_name: str,
        case_set_name: str,
        profiles: tuple[str, ...],
        snr_conditions: tuple[str, ...],
        stem_limit: int | None,
        methods: tuple[str, ...] | None = None,
        extra_args: tuple[str, ...] = (),
        name_suffix: str = "",
        attune_curve_sec: float = Config.vib2_curve_sec,
    ) -> tuple[list[str], Path]:
        result_name = (
            f"{suite_name}_{self.sampling_tag}_{PRIMARY_COMPARISON_VERSION}"
            f"_seed{self.config.seed}"
        )
        if self.config.automatic_yin_window:
            result_name += "_auto_yin_4periods_pow2"
        # Non-default detector settings get their own directory so an ablation
        # can never overwrite or be mistaken for the shipped-default suite.
        result_name += name_suffix
        output_dir = self.results_root / result_name
        artifact_dir = (
            self.coco_artifact_root
            / case_set_name
            / self.sampling_tag
            / f"seed{self.config.seed}"
        )
        command = [
            str(self.python),
            str(self.runner),
            "--coco",
            "--coco-sfizz-render",
            str(self.sfizz_render),
            "--coco-soundfonts-root",
            str(self.soundfonts_root),
            "--workers",
            str(self.config.workers),
            "--seed",
            str(self.config.seed),
            "--methods",
            self._csv(methods or self.config.methods),
            "--coco-profiles",
            self._csv(profiles),
            "--coco-notes-per-stem",
            str(len(profiles)),
            "--coco-snrs",
            self._csv(snr_conditions),
            "--coco-min-note-seconds",
            str(self.config.minimum_note_seconds),
            "--rate-tolerance-hz",
            str(self.config.rate_tolerance_hz),
            "--amplitude-tolerance-semitones",
            str(self.config.extent_tolerance_semitones),
            "--center-tolerance-cents",
            str(self.config.center_tolerance_cents),
            "--attune-curve-sec",
            f"{attune_curve_sec:g}",
            "--no-console-summary",
            "--coco-injection-range",
            "native",
            "--coco-output-root",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
        ]
        command += (
            ["--coco-all-stems"]
            if stem_limit is None
            else ["--coco-stem-limit", str(stem_limit)]
        )
        if self.config.automatic_yin_window:
            command.append("--coco-auto-yin-window")
        if self.config.force_pitch_redetection:
            command.append("--force-pitch")
        command += list(extra_args)
        return command, output_dir

    def run_coco_suite(
        self,
        *,
        suite_name: str,
        case_set_name: str,
        profiles: tuple[str, ...],
        snr_conditions: tuple[str, ...],
        stem_limit: int | None,
        force: bool = False,
    ) -> SuiteResult:
        command, output_dir = self.coco_command(
            suite_name=suite_name,
            case_set_name=case_set_name,
            profiles=profiles,
            snr_conditions=snr_conditions,
            stem_limit=stem_limit,
        )
        summary = self._run_or_reuse(command, output_dir, force=force)
        display(summary)
        return SuiteResult(output_dir=output_dir, summary=summary)

    def _run_or_reuse(
        self,
        command: list[str],
        output_dir: Path,
        *,
        force: bool,
    ) -> pd.DataFrame:
        """Invoke the runner unless its summary already exists."""
        report = output_dir / "summary.csv"
        if force or not report.is_file():
            print("Running:", output_dir.name)
            subprocess.run(command, cwd=self.repo_root, check=True)
        else:
            print("Reusing completed reports:", output_dir)
        return self._cached_summary(output_dir)

    @staticmethod
    def _cached_summary(output_dir: Path) -> pd.DataFrame:
        """Add interval-level Yang accuracy to older cached public tables.

        Average successful cases, excluding unmatched intervals (NaN), just
        like VibratoBenchmarker.summarize. Preserve the saved F1 and timings.
        """
        summary = pd.read_csv(output_dir / "summary.csv")
        labels = {
            "yang_extent_accuracy": "Extent Accuracy (Yang)",
            "yang_rate_accuracy": "Rate Accuracy (Yang)",
        }
        missing = {key: label for key, label in labels.items() if label not in summary}
        if missing:
            values = {}
            for method in summary["Method"]:
                cases = pd.read_csv(output_dir / "raw_outputs" / method / "cases.csv")
                good = cases.loc[
                    ~cases["skipped"].fillna(False).astype(bool)
                    & cases["error"].fillna("").eq("")
                ]
                values[method] = good[list(missing)].mean()
            for key, label in missing.items():
                summary[label] = summary["Method"].map(
                    {method: metrics[key] for method, metrics in values.items()}
                ).round(4)
        for label, after in (
            ("Extent Accuracy (Yang)", "Extent Recall"),
            ("Rate Accuracy (Yang)", "Rate Recall"),
        ):
            values = summary.pop(label)
            summary.insert(summary.columns.get_loc(after) + 1, label, values)
        return summary

    def _run_concurrently(self, jobs: list[tuple[list[str], Path]]) -> None:
        """Run independent benchmark commands concurrently."""
        if not jobs:
            return
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {
                pool.submit(
                    subprocess.run,
                    command,
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                ): output_dir
                for command, output_dir in jobs
            }
            for future in as_completed(futures):
                output_dir = futures[future]
                completed = future.result()
                if completed.returncode:
                    print(completed.stdout)
                    print(completed.stderr, file=sys.stderr)
                    raise subprocess.CalledProcessError(
                        completed.returncode,
                        completed.args,
                    )
                print("Completed:", output_dir.name)

    def run_preliminary_average(self) -> SuiteResult:
        result = self.run_coco_suite(
            suite_name="preliminary_average",
            case_set_name="seeded_constant_controls",
            profiles=AVERAGE_PROFILES,
            snr_conditions=AVERAGE_SNR_CONDITIONS,
            stem_limit=self.config.preliminary_stems,
            force=self.config.force_preliminary_rerun,
        )
        run_config = json.loads((result.output_dir / "run_config.json").read_text())
        dataset = run_config["dataset"]
        assert dataset["coco_track_selection_policy"] == CocoChorales.BALANCED_SELECTION_POLICY
        assert (
            dataset["coco_profile_parameter_sampler"]
            == PROFILE_PARAMETER_SAMPLER_VERSION
        )
        return result

    def run_preliminary_changing(self) -> SuiteResult:
        return self.run_coco_suite(
            suite_name="preliminary_changing",
            case_set_name="seeded_changing_profiles",
            profiles=CHANGING_PROFILES,
            snr_conditions=CHANGING_SNR_CONDITIONS,
            stem_limit=self.config.preliminary_stems,
            force=self.config.force_preliminary_rerun,
        )

    def run_curve_sec_ablation(
        self,
        values: Sequence[float] = CURVE_SEC_ABLATION_VALUES,
        *,
        suite: str = "changing",
        force: bool = False,
        parallel: bool = False,
    ) -> pd.DataFrame:
        """Sweep Attune's center/width knot spacing using cached pitch data."""
        if suite not in {"changing", "average"}:
            raise ValueError("suite must be 'changing' or 'average'")
        changing = suite == "changing"
        jobs = [
            (
                value,
                *self.coco_command(
                    suite_name=f"preliminary_{suite}",
                    case_set_name=(
                        "seeded_changing_profiles"
                        if changing
                        else "seeded_constant_controls"
                    ),
                    profiles=CHANGING_PROFILES if changing else AVERAGE_PROFILES,
                    snr_conditions=(
                        CHANGING_SNR_CONDITIONS if changing else AVERAGE_SNR_CONDITIONS
                    ),
                    stem_limit=self.config.preliminary_stems,
                    methods=("attune",),
                    name_suffix=f"_curve_sec_{value:g}".replace(".", "p"),
                    attune_curve_sec=value,
                ),
            )
            for value in values
        ]
        pending = [
            (command, output_dir)
            for _, command, output_dir in jobs
            if force or not (output_dir / "summary.csv").is_file()
        ]
        if pending:
            # Every value builds the same cases and differs only in a detector
            # setting, so the first invocation is the only one that can still
            # be WRITING the shared render/pitch cache. Run it alone, then let
            # the rest read that cache concurrently.
            first, *rest = pending
            print(f"Priming the shared cache with 1 of {len(pending)} runs")
            self._run_or_reuse(*first, force=True)
            if parallel:
                if rest:
                    print(f"Running the remaining {len(rest)} concurrently")
                    self._run_concurrently(rest)
            else:
                for command, output_dir in rest:
                    self._run_or_reuse(command, output_dir, force=True)
        rows = []
        for value, _, output_dir in jobs:
            summary = pd.read_csv(output_dir / "summary.csv")
            attune = summary[summary.Method == "attune"].iloc[0]
            rows.append(
                {
                    "vib2_curve_sec": value,
                    "Extent F1": attune["Extent F1"],
                    "Rate F1": attune["Rate F1"],
                    "Overall F1": attune["Overall F1"],
                    "False Alarms": attune["False Alarms"],
                    "Audio(s)/Compute(s)": attune["Audio(s)/Compute(s)"],
                    "output_dir": output_dir.name,
                }
            )
        table = pd.DataFrame(rows).sort_values("vib2_curve_sec", ascending=False)
        display(table)
        return table

    def run_all_average(self) -> SuiteResult:
        return self.run_coco_suite(
            suite_name="all_average",
            case_set_name="seeded_constant_controls",
            profiles=AVERAGE_PROFILES,
            snr_conditions=AVERAGE_SNR_CONDITIONS,
            stem_limit=None,
        )

    def run_all_changing(self) -> SuiteResult:
        return self.run_coco_suite(
            suite_name="all_changing",
            case_set_name="seeded_changing_profiles",
            profiles=CHANGING_PROFILES,
            snr_conditions=CHANGING_SNR_CONDITIONS,
            stem_limit=None,
        )

    def run_sfizz_control(self) -> dict[str, Any]:
        output_dir = (
            self.results_root / f"zero_vibrato_control_{COCO_SFIZZ_POLICY_VERSION}"
        )
        command = [
            str(self.python),
            str(self.runner),
            "--coco-control",
            "--coco-control-instrument",
            "violin",
            "--coco-sfizz-render",
            str(self.sfizz_render),
            "--coco-soundfonts-root",
            str(self.soundfonts_root),
            "--output-dir",
            str(output_dir),
        ]
        report_path = output_dir / "zero_vibrato_validation.json"
        if self.config.force_preliminary_rerun or not report_path.is_file():
            print(shlex.join(command))
            subprocess.run(command, cwd=self.repo_root, check=True)
        else:
            print("Reusing completed reports:", output_dir)
        report = json.loads(report_path.read_text())
        assert report["passed"]
        assert report["renderer"]["policy"] == COCO_SFIZZ_POLICY_VERSION
        display(
            pd.DataFrame.from_dict(report["stages"], orient="index")
            .rename_axis("pitch_stage")
            .reset_index()
        )
        display(
            pd.DataFrame(
                [
                    {
                        "instrument": report["instrument"],
                        "pitch_midi": report["pitch_midi"],
                        "audio": report["wav_path"],
                        "library": report["renderer"]["library"],
                        "sfz": report["renderer"]["relative_sfz"],
                        "renderer_identity": report["renderer"]["identity"],
                    }
                ]
            )
        )
        return report

    def show_sampling_invariants(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        coco = CocoChorales()
        records_by_seed = {
            seed: coco.select_records(
                split="test",
                max_tracks=self.config.preliminary_stems,
                seed=seed,
                balanced=True,
            )
            for seed in (self.config.seed, self.config.seed + 1)
        }
        compositions = {
            seed: Counter((record.ensemble, record.instrument) for record in records)
            for seed, records in records_by_seed.items()
        }
        assert compositions[self.config.seed] == compositions[self.config.seed + 1]
        assert Counter(
            record.ensemble for record in records_by_seed[self.config.seed]
        ) == Counter({"brass": 5, "random": 5, "string": 5, "woodwind": 5})
        assert {record.track_id for record in records_by_seed[self.config.seed]} != {
            record.track_id for record in records_by_seed[self.config.seed + 1]
        }
        selection = pd.DataFrame(
            [
                {
                    "seed": seed,
                    "ensemble": ensemble,
                    "instrument": instrument,
                    "stems": count,
                }
                for seed, composition in compositions.items()
                for (ensemble, instrument), count in sorted(composition.items())
            ]
        )

        parameter_rows = []
        for index, profile in enumerate(AVERAGE_PROFILES + CHANGING_PROFILES):
            if profile == "none" and any(
                row["profile"] == "none" for row in parameter_rows
            ):
                continue
            parameters = CocoDataset.sample_profile_parameters(
                profile,
                seed=self.config.seed + index,
            )
            rate, amplitude = CocoDataset.profile_curves(
                profile,
                np.array([0.0, 0.5, 1.0]),
                parameters=parameters,
            )
            if profile != "none":
                assert (
                    NATIVE_RATE_RANGE_HZ[0]
                    <= rate.min()
                    <= rate.max()
                    <= NATIVE_RATE_RANGE_HZ[1]
                )
                assert (
                    NATIVE_AMPLITUDE_RANGE_SEMITONES[0]
                    <= amplitude.min()
                    <= amplitude.max()
                    <= NATIVE_AMPLITUDE_RANGE_SEMITONES[1]
                )
            if profile in {"accelerating", "decelerating"}:
                assert (
                    CHANGING_RATE_SPAN_RANGE_HZ[0]
                    <= np.ptp(rate)
                    <= CHANGING_RATE_SPAN_RANGE_HZ[1]
                )
            if profile in {"widening", "narrowing"}:
                assert (
                    CHANGING_AMPLITUDE_SPAN_RANGE_SEMITONES[0]
                    <= np.ptp(amplitude)
                    <= CHANGING_AMPLITUDE_SPAN_RANGE_SEMITONES[1]
                )
            parameter_rows.append(
                {
                    "profile": profile,
                    "rate_start_hz": rate[0],
                    "rate_middle_hz": rate[1],
                    "rate_end_hz": rate[-1],
                    "amplitude_start_semitones": amplitude[0],
                    "amplitude_end_semitones": amplitude[-1],
                }
            )
        parameters = pd.DataFrame(parameter_rows)
        display(selection)
        display(parameters)
        return selection, parameters

    def show_synthesis_manifest(self, run_config: dict[str, Any]) -> pd.DataFrame:
        manifests = run_config["dataset"].get("coco_synthesis") or {}
        assert manifests, "Coco run is missing sfizz synthesis provenance"
        frame = pd.DataFrame(
            [
                {
                    "instrument": instrument,
                    "library": manifest["library"],
                    "library_version": manifest["library_version"],
                    "sfz": manifest["relative_sfz"],
                    "bend_range_semitones": manifest["pitch_bend_range_semitones"],
                    "zero_vibrato_ccs": ",".join(
                        str(value) for value in manifest["zero_vibrato_ccs"]
                    ),
                    "policy": manifest["policy"],
                    "identity": manifest["identity"][:12],
                }
                for instrument, manifest in manifests.items()
            ]
        )
        assert set(frame["policy"]) == {COCO_SFIZZ_POLICY_VERSION}
        display(frame)
        return frame

    def show_pitch_geometry(self, suite: SuiteResult) -> pd.DataFrame:
        cases = pd.read_csv(suite.output_dir / "raw_outputs" / "attune" / "cases.csv")
        columns = [
            "meta_track",
            "meta_stem",
            "meta_ensemble",
            "meta_instrument",
            "meta_pitch_fmin_hz",
            "meta_pitch_fmax_hz",
            "meta_pyin_autocorrelation_fmin_hz",
            "meta_pyin_autocorrelation_fmax_hz",
            "meta_hmm_state_fmin_hz",
            "meta_hmm_state_fmax_hz",
            "meta_yin_integration_size",
            "meta_yin_window_policy",
            "meta_yin_window_score_fmin_hz",
        ]
        missing = sorted(set(columns) - set(cases))
        assert not missing, f"pitch-geometry metadata is missing: {missing}"
        geometry = (
            cases[columns]
            .drop_duplicates(["meta_track", "meta_stem"])
            .sort_values(["meta_pitch_fmin_hz", "meta_instrument"])
        )

        np.testing.assert_allclose(
            geometry["meta_pyin_autocorrelation_fmin_hz"],
            geometry["meta_pitch_fmin_hz"],
        )
        np.testing.assert_allclose(
            geometry["meta_pyin_autocorrelation_fmax_hz"],
            geometry["meta_pitch_fmax_hz"],
        )
        np.testing.assert_allclose(
            geometry["meta_hmm_state_fmin_hz"],
            geometry["meta_pitch_fmin_hz"],
        )
        hmm_upper = geometry["meta_hmm_state_fmax_hz"]
        annotated_upper = geometry["meta_pitch_fmax_hz"]
        assert np.all(hmm_upper <= annotated_upper * (1.0 + 1e-12))
        assert np.all(
            hmm_upper
            >= annotated_upper * 2.0 ** (-PitchSmoother.RESOLUTION / 12.0)
        )

        if self.config.automatic_yin_window:
            expected = geometry["meta_yin_window_score_fmin_hz"].map(
                CocoDataset.automatic_yin_window_size
            )
            np.testing.assert_array_equal(
                geometry["meta_yin_integration_size"].to_numpy(),
                expected.to_numpy(),
            )
            assert geometry["meta_yin_integration_size"].nunique() > 1
            assert np.any(geometry["meta_yin_integration_size"] >= 4096)

        display(
            geometry.groupby(
                ["meta_yin_integration_size", "meta_instrument"],
                as_index=False,
            ).size()
        )
        display(geometry)
        return geometry

    def show_no_vibrato_analysis(
        self,
        suite: SuiteResult,
        *,
        method: str = "attune",
    ) -> dict[str, pd.DataFrame]:
        method_dir = suite.output_dir / "raw_outputs" / method
        report_paths = {
            "overview": method_dir / "no_vibrato_overview.csv",
            "cases": method_dir / "no_vibrato_cases.csv",
            "by_instrument": method_dir / "no_vibrato_by_instrument.csv",
            "by_pattern": method_dir / "no_vibrato_by_pattern.csv",
            "false_positive_frames": (
                method_dir / "no_vibrato_false_positive_frames.csv"
            ),
        }
        if all(path.is_file() for path in report_paths.values()):
            reports = {name: pd.read_csv(path) for name, path in report_paths.items()}
        else:
            reports = VibratoBenchmarker.no_vibrato_diagnostics(
                pd.read_csv(method_dir / "cases.csv"),
                pd.read_csv(method_dir / "frames.csv"),
            )

        display(reports["overview"])
        if len(reports["by_instrument"]):
            display(
                reports["by_instrument"].sort_values(
                    "false_positive_rate",
                    ascending=False,
                )
            )
        if len(reports["by_pattern"]):
            display(reports["by_pattern"])
        failures = (
            reports["cases"]
            .loc[reports["cases"]["false_positive_frames"] > 0]
            .sort_values(
                ["false_positive_rate", "false_positive_seconds"],
                ascending=False,
            )
        )
        failure_columns = [
            "case_id",
            "meta_ensemble",
            "meta_instrument",
            "meta_yin_integration_size",
            "false_positive_rate",
            "false_positive_runs",
            "false_positive_seconds",
            "edge_false_positive_fraction",
            "median_raw_pitch_error_cents",
            "median_smoothed_pitch_error_cents",
            "smoothed_octave_error_fraction",
            "transition_overlap_fraction",
            "median_estimated_rate_hz",
            "median_estimated_amplitude_semitones",
            "diagnostic_pattern",
        ]
        display(failures[[column for column in failure_columns if column in failures]])
        return reports

    def run_yang_preliminary(self) -> SuiteResult:
        output_dir = (
            self.results_root
            / f"preliminary_yang_parameters_{PRIMARY_COMPARISON_VERSION}"
        )
        command = [
            str(self.python),
            str(self.runner),
            "--yang",
            "--workers",
            str(self.config.workers),
            "--methods",
            self._csv(self.config.methods),
            "--yang-root",
            str(self.yang_root),
            "--yang-cache-root",
            str(self.yang_cache_root),
            "--yang-recording-limit",
            str(self.config.yang_recordings),
            "--rate-tolerance-hz",
            str(self.config.rate_tolerance_hz),
            "--amplitude-tolerance-semitones",
            str(self.config.extent_tolerance_semitones),
            "--center-tolerance-cents",
            str(self.config.center_tolerance_cents),
            "--attune-curve-sec",
            f"{Config.vib2_curve_sec:g}",
            "--no-console-summary",
            "--output-dir",
            str(output_dir),
        ]
        if self.config.force_pitch_redetection:
            command.append("--force-pitch")
        summary = self._run_or_reuse(
            command, output_dir,
            force=self.config.force_preliminary_rerun or self.config.force_pitch_redetection,
        )
        display(summary)
        return SuiteResult(output_dir=output_dir, summary=summary)
