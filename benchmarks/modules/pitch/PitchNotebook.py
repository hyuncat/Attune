"""PitchNotebook implementation and owned benchmark helpers."""

from __future__ import annotations
from benchmarks.modules.pitch.competitors.PYIN import (
    PYINAblationAdapter,
    PYINAdditionDetector,
    PYINGlobalVolumeGateSmoother,
    PYINPraatStyleVoicingSmoother,
)
import math
from collections import Counter
from collections.abc import Collection
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import ClassVar
import mir_eval
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import ndimage
from algorithms.Config import PYIN_DEFAULT_MIN_VOLUME
from algorithms.Config import PYIN_DEFAULT_UNV_THRESH
from algorithms.Config import PYIN_POSTHOC_MIN_VOLUME
from algorithms.Config import PYIN_POSTHOC_UNV_THRESH
from benchmarks.modules.pitch.competitors.Attune import Attune
from benchmarks.modules.pitch.datasets.Bach10 import Bach10
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
from benchmarks.modules.pitch.datasets.URMP import URMP
from benchmarks.paths import RESULTS_ROOT
from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker, SCORING_VERSION
import copy
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import matplotlib.pyplot as plt
from scipy import fft as scipy_fft
from scipy.signal import find_peaks
from scipy.stats import boltzmann
from algorithms.Config import Config
from app_logic.NoteData import NoteData
from app_logic.midi.ScoreData import ScoreData
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
from benchmarks.modules.pitch.competitors.PYIN import PYIN
from benchmarks.modules.pitch.competitors.PYIN import PYINPitchDetector
from benchmarks.modules.pitch.competitors.PYIN import PYINPitchSmoother
from benchmarks.modules.pitch.datasets.AudioAnnot import AudioAnnot
from benchmarks.modules.pitch.sweeps.DecoupledVoicing import PraatInspiredVoicingDecoder
from benchmarks.modules.pitch.sweeps.DecoupledVoicing import (
    PraatInspiredVoicingParameters,
)
from benchmarks.modules.pitch.sweeps.DecoupledVoicing import VoicingFeatures


@dataclass(frozen=True)
class _PyinVariant:
    """One post-hoc comparison or streaming-factorial cell."""

    name: str
    label: str
    prominence: bool = False
    volume_gate: bool = False
    unvoiced_gate: bool = False
    praat_controller: bool = False

    @staticmethod
    def streaming_variants() -> tuple[Variant, ...]:
        variants: list[_PyinVariant] = []
        for prominence in (False, True):
            for volume_gate in (False, True):
                for unvoiced_gate in (False, True):
                    additions = [
                        name
                        for enabled, name in (
                            (prominence, "prominence"),
                            (volume_gate, "volume"),
                            (unvoiced_gate, "unvoiced"),
                        )
                        if enabled
                    ]
                    suffix = "_".join(additions) if additions else "baseline"
                    label = "pYIN realtime"
                    if additions:
                        label += " + " + " + ".join(additions)
                    variants.append(
                        _PyinVariant(
                            name=f"pyin_rt_{suffix}",
                            label=label,
                            prominence=prominence,
                            volume_gate=volume_gate,
                            unvoiced_gate=unvoiced_gate,
                        )
                    )
        return tuple(variants)


POSTHOC_VARIANTS = (
    _PyinVariant("pyin_volume_gate", "pYIN + volume gate", volume_gate=True),
    _PyinVariant(
        "pyin_volume_praat_voicing",
        "pYIN + volume + Praat-style voicing",
        volume_gate=True,
        unvoiced_gate=True,
        praat_controller=True,
    ),
)
STREAMING_VARIANTS = _PyinVariant.streaming_variants()
VARIANTS = POSTHOC_VARIANTS
VARIANT_BY_NAME = {
    variant.name: variant for variant in (*POSTHOC_VARIANTS, *STREAMING_VARIANTS)
}
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class PitchNotebook:
    """Notebook configuration, presentation, and compatibility entry points."""

    Variant = _PyinVariant

    @dataclass(frozen=True)
    class NotebookConfig:
        """The few experiment choices readers may reasonably want to change."""

        workers: int = field(default_factory=PitchBenchmarker.default_pitch_workers)
        streaming_workers: int = field(
            default_factory=PitchBenchmarker.default_streaming_workers
        )
        preliminary_tracks_per_instrument: int = 2
        coco_per_stratum: int | None = None
        seed: int = 0
        methods: tuple[str, ...] = ()
        include_slow_methods: bool = False
        use_cache: bool = True
        force_rerun: bool = False
        degradation_snrs: tuple[float, ...] = (math.inf, 20.0, 15.0, 10.0, 5.0)

    @dataclass(frozen=True)
    class SuiteResult:
        """One suite's per-track rows plus the per-method table drawn from them."""

        name: str
        path: Path
        rows: pd.DataFrame
        summary: pd.DataFrame

    @dataclass(frozen=True)
    class ProfileSpec:
        datasets: tuple[str, ...]
        per_instrument: int | None
        max_tracks: int | None
        materialize: bool
        bootstrap_samples: int

    @dataclass(frozen=True)
    class StreamingGrid:
        """One track sampled on the live Attune capture schedule."""

        audio: np.ndarray
        config: Config
        duration: float
        sample_rate: int
        frame_size: int
        integration_size: int
        hop_size: int
        starts: np.ndarray
        times: np.ndarray

    @dataclass(frozen=True)
    class StreamingTrackResult:
        """One variant's frame-restarted estimates and timing diagnostics."""

        estimate: PitchDetectorBase.PitchEstimate
        duration: float
        sample_rate: int
        frame_size: int
        integration_size: int
        hop_size: int
        latencies: np.ndarray
        call_cpus: np.ndarray
        call_walls: np.ndarray

    @dataclass(frozen=True)
    class StreamingThresholdEvidence:
        """One causal pYIN pass reused by every voicing threshold.

        ``frequencies`` is the strongest frame-local candidate pitch before either explicit
        gate. ``unvoiced_probabilities`` and ``volumes`` are the frame-local cues
        the threshold sweep masks without rerunning CMNDF or candidate decoding.
        """

        times: np.ndarray
        frequencies: np.ndarray
        volumes: np.ndarray
        unvoiced_probabilities: np.ndarray
        duration: float
        sample_rate: int
        frame_size: int
        integration_size: int
        hop_size: int

    @staticmethod
    def _default_streaming_workers() -> int:
        configured = os.environ.get("ATTUNE_PYIN_ABLATION_STREAM_WORKERS")
        if configured is not None:
            return int(configured)
        logical_cpus = max(1, os.cpu_count() or 1)
        return min(4, max(1, logical_cpus // 2))

    @dataclass(frozen=True)
    class PyinAblationConfig:
        """The few experiment choices a notebook reader may change."""

        profile: str = field(
            default_factory=lambda: os.environ.get(
                "ATTUNE_PYIN_ABLATION_PROFILE", "balanced"
            )
            .strip()
            .lower()
        )
        tracks_per_instrument: int = 2
        seed: int = 0
        bootstrap_samples: int = 10000
        use_cache: bool = True
        force: bool = field(
            default_factory=lambda: os.environ.get("ATTUNE_PYIN_ABLATION_FORCE", "0")
            == "1"
        )
        experiment_version: str = "pyin_voicing_controllers_v2_128hop"
        volume_floor_ratio: float = field(
            default_factory=lambda: float(
                os.environ.get(
                    "ATTUNE_PYIN_ABLATION_VOLUME_FLOOR_RATIO",
                    str(PYIN_DEFAULT_MIN_VOLUME),
                )
            )
        )
        volume_ceiling_percentile: float = field(
            default_factory=lambda: float(
                os.environ.get("ATTUNE_PYIN_ABLATION_VOLUME_CEILING_PERCENTILE", "95")
            )
        )
        unvoiced_threshold: float = field(
            default_factory=lambda: float(
                os.environ.get(
                    "ATTUNE_PYIN_ABLATION_UNVOICED_THRESHOLD",
                    str(PYIN_DEFAULT_UNV_THRESH),
                )
            )
        )
        praat_switch_cost: float = field(
            default_factory=lambda: float(
                os.environ.get("ATTUNE_PYIN_ABLATION_PRAAT_SWITCH_COST", "0.02")
            )
        )
        streaming_seconds: float = field(
            default_factory=lambda: float(
                os.environ.get("ATTUNE_PYIN_ABLATION_STREAM_SECONDS", "12")
            )
        )
        streaming_workers: int = field(
            default_factory=lambda: PitchNotebook._default_streaming_workers()
        )
        cache_root: Path = field(
            default_factory=lambda: Path(
                os.environ.get(
                    "ATTUNE_PYIN_ABLATION_CACHE_ROOT",
                    RESULTS_ROOT / "pitch" / "pyin_ablation",
                )
            )
        )

        def __post_init__(self) -> None:
            if not np.isfinite(self.streaming_seconds) or self.streaming_seconds <= 0:
                raise ValueError("streaming_seconds must be positive and finite")
            if self.streaming_workers < 1:
                raise ValueError("streaming_workers must be at least 1")
            if (
                not np.isfinite(self.volume_floor_ratio)
                or not 0.0 <= self.volume_floor_ratio <= 1.0
            ):
                raise ValueError("volume_floor_ratio must be between zero and one")
            if (
                not np.isfinite(self.volume_ceiling_percentile)
                or not 0.0 < self.volume_ceiling_percentile <= 100.0
            ):
                raise ValueError(
                    "volume_ceiling_percentile must be in the interval (0, 100]"
                )
            if (
                not np.isfinite(self.unvoiced_threshold)
                or not 0.0 < self.unvoiced_threshold < 1.0
            ):
                raise ValueError("unvoiced_threshold must be between zero and one")
            if not np.isfinite(self.praat_switch_cost) or self.praat_switch_cost < 0:
                raise ValueError("praat_switch_cost must be finite and non-negative")

    class PyinAblation:
        """Select tracks, run all variants, and produce paired reports."""

        METRICS: ClassVar[tuple[str, ...]] = (
            "Overall Accuracy",
            "Raw Pitch Accuracy",
            "Raw Chroma Accuracy",
            "Voicing Recall",
            "Voicing False Alarm",
        )
        METRIC_DIRECTION: ClassVar[dict[str, float]] = {
            "Overall Accuracy": 1.0,
            "Raw Pitch Accuracy": 1.0,
            "Raw Chroma Accuracy": 1.0,
            "Voicing Recall": 1.0,
            "Voicing False Alarm": -1.0,
        }
        POSTHOC_CONTRASTS: ClassVar[tuple[tuple[str, str, str, str], ...]] = (
            (
                "Praat-style global voicing",
                "global volume gate present",
                "pyin_volume_praat_voicing",
                "pyin_volume_gate",
            ),
        )
        STREAMING_CONTRASTS: ClassVar[tuple[tuple[str, str, str, str], ...]] = tuple(
            (
                (component, context, on_name, off_name)
                for component, switch in (
                    ("Prominence", "prominence"),
                    ("Volume gate", "volume_gate"),
                    ("Global unvoiced gate", "unvoiced_gate"),
                )
                for other in STREAMING_VARIANTS
                if not getattr(other, switch)
                for enabled in STREAMING_VARIANTS
                if all(
                    (
                        getattr(enabled, field_name)
                        == (
                            True if field_name == switch else getattr(other, field_name)
                        )
                        for field_name in ("prominence", "volume_gate", "unvoiced_gate")
                    )
                )
                for context, on_name, off_name in (
                    (
                        ", ".join(
                            (
                                f"{name}={('on' if getattr(other, field_name) else 'off')}"
                                for name, field_name in (
                                    ("prominence", "prominence"),
                                    ("volume", "volume_gate"),
                                    ("unvoiced", "unvoiced_gate"),
                                )
                                if field_name != switch
                            )
                        ),
                        enabled.name,
                        other.name,
                    ),
                )
            )
        )
        STREAMING_VERSION: ClassVar[str] = "attune_recording_grid_v6_no_hmm"
        THRESHOLD_SWEEP_VERSION: ClassVar[str] = "accuracy_frontier_v2_no_hmm_128hop"
        DEFAULT_UNVOICED_THRESHOLDS: ClassVar[tuple[float, ...]] = (
            0.85,
            0.9,
            0.925,
            0.95,
            0.96,
            0.97,
            0.975,
            0.98,
            0.985,
            0.99,
        )
        DEFAULT_PRAAT_SWITCH_COSTS: ClassVar[tuple[float, ...]] = (
            0.0,
            0.005,
            0.01,
            0.02,
            0.04,
        )

        def __init__(self, config: PyinAblationConfig | None = None) -> None:
            from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

            self.config = config or PitchNotebook.PyinAblationConfig()
            self.spec = self._profile_spec()
            floor_tag = f"{self.config.volume_floor_ratio:g}".replace(".", "p")
            ceiling_tag = f"{self.config.volume_ceiling_percentile:g}".replace(".", "p")
            unvoiced_tag = f"{self.config.unvoiced_threshold:g}".replace(".", "p")
            switch_tag = f"{self.config.praat_switch_cost:g}".replace(".", "p")
            self.run_root = (
                Path(self.config.cache_root)
                / self.config.experiment_version
                / self.config.profile
                / f"floor_{floor_tag}__ceiling_p{ceiling_tag}__unvoiced_{unvoiced_tag}__switch_{switch_tag}"
            )
            self.benchmarker = PitchBenchmarker(
                PitchBenchmarker.Options(
                    datasets=self.spec.datasets,
                    per_instrument=self.spec.per_instrument,
                    max_tracks=self.spec.max_tracks,
                    seed=self.config.seed,
                    materialize=self.spec.materialize,
                    use_cache=self.config.use_cache,
                    force_reanalysis=self.config.force,
                    quiet_runtime=True,
                )
            )
            self._tracks = None

        def _profile_spec(self) -> ProfileSpec:
            per_instrument = int(self.config.tracks_per_instrument)
            profiles = {
                "smoke": PitchNotebook.ProfileSpec(
                    datasets=("bach10-mf0-synth",),
                    per_instrument=None,
                    max_tracks=1,
                    materialize=False,
                    bootstrap_samples=1000,
                ),
                "balanced": PitchNotebook.ProfileSpec(
                    datasets=("bach10-mf0-synth", URMP.name, CocoChorales.name),
                    per_instrument=per_instrument,
                    max_tracks=None,
                    materialize=True,
                    bootstrap_samples=self.config.bootstrap_samples,
                ),
                "urmp": PitchNotebook.ProfileSpec(
                    datasets=(URMP.name,),
                    per_instrument=per_instrument,
                    max_tracks=None,
                    materialize=False,
                    bootstrap_samples=self.config.bootstrap_samples,
                ),
                "full": PitchNotebook.ProfileSpec(
                    datasets=(*AudioAnnot.NAMES, URMP.name, CocoChorales.name),
                    per_instrument=None,
                    max_tracks=None,
                    materialize=True,
                    bootstrap_samples=self.config.bootstrap_samples,
                ),
            }
            try:
                return profiles[self.config.profile]
            except KeyError as exc:
                raise ValueError(
                    f"unknown profile {self.config.profile!r}; choose from {tuple(profiles)}"
                ) from exc

        @property
        def tracks(self):
            if self._tracks is None:
                self._tracks = self.benchmarker.tracks()
                if not self._tracks:
                    raise RuntimeError(
                        "No benchmark tracks found for the selected profile."
                    )
            return self._tracks

        def show_configuration(self) -> pd.DataFrame:
            frame = pd.DataFrame(
                {
                    "setting": (
                        "profile",
                        "datasets",
                        "tracks per instrument",
                        "max tracks/dataset",
                        "materialize Coco",
                        "bootstrap draws",
                        "volume floor ratio",
                        "global ceiling percentile",
                        "global unvoiced threshold",
                        "Praat-style switch cost",
                        "streaming excerpt seconds",
                        "streaming workers",
                        "force",
                        "run root",
                        "streaming run root",
                    ),
                    "value": (
                        self.config.profile,
                        ", ".join(self.spec.datasets),
                        self.spec.per_instrument,
                        self.spec.max_tracks,
                        self.spec.materialize,
                        self.spec.bootstrap_samples,
                        self.config.volume_floor_ratio,
                        self.config.volume_ceiling_percentile,
                        self.config.unvoiced_threshold,
                        self.config.praat_switch_cost,
                        self.config.streaming_seconds,
                        self.config.streaming_workers,
                        self.config.force,
                        self.run_root,
                        self.streaming_run_root,
                    ),
                }
            )
            display(frame)
            return frame

        @property
        def streaming_run_root(self) -> Path:
            seconds = f"{self.config.streaming_seconds:g}".replace(".", "p")
            return (
                self.run_root / "streaming" / self.STREAMING_VERSION / f"max_{seconds}s"
            )

        @property
        def threshold_sweep_root(self) -> Path:
            return self.run_root / "threshold_sweep" / self.THRESHOLD_SWEEP_VERSION

        @property
        def streaming_threshold_sweep_root(self) -> Path:
            seconds = f"{self.config.streaming_seconds:g}".replace(".", "p")
            return self.threshold_sweep_root / "streaming" / f"max_{seconds}s"

        @staticmethod
        def _validated_thresholds(values: tuple[float, ...]) -> tuple[float, ...]:
            thresholds = tuple(sorted({float(value) for value in values}))
            if not thresholds or any(
                (
                    not np.isfinite(value) or not 0.0 < value < 1.0
                    for value in thresholds
                )
            ):
                raise ValueError("unvoiced thresholds must be finite and in (0, 1)")
            return thresholds

        @staticmethod
        def _validated_switch_costs(values: tuple[float, ...]) -> tuple[float, ...]:
            costs = tuple(sorted({float(value) for value in values}))
            if not costs or any(
                (not np.isfinite(value) or value < 0.0 for value in costs)
            ):
                raise ValueError(
                    "Praat-style switch costs must be finite and non-negative"
                )
            return costs

        @staticmethod
        def _validated_volume_floor_ratios(
            values: tuple[float, ...]
        ) -> tuple[float, ...]:
            ratios = tuple(sorted({float(value) for value in values}))
            if not ratios or any(
                (not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in ratios)
            ):
                raise ValueError("volume-floor ratios must be finite and in [0, 1]")
            return ratios

        def check_contract(self) -> None:
            """Fail fast if a refactor changes either experimental design."""
            reference = PYINAblationAdapter(
                VARIANT_BY_NAME["pyin_rt_baseline"],
                self.run_root,
                volume_floor_ratio=self.config.volume_floor_ratio,
                volume_ceiling_percentile=self.config.volume_ceiling_percentile,
                unvoiced_threshold=self.config.unvoiced_threshold,
                praat_switch_cost=self.config.praat_switch_cost,
            )
            recording = reference.recording_for(reference.DEFAULT_CONFIG)
            if type(recording.pitch_detector) is not PYINPitchDetector:
                raise AssertionError("pyin no longer uses the exact reference detector")
            if type(recording.pitch_smoother) is not PYINPitchSmoother:
                raise AssertionError("pyin no longer uses the exact reference HMM")
            expected = {
                (prominence, volume, unvoiced)
                for prominence in (False, True)
                for volume in (False, True)
                for unvoiced in (False, True)
            }
            actual = [
                (variant.prominence, variant.volume_gate, variant.unvoiced_gate)
                for variant in STREAMING_VARIANTS
            ]
            if len(actual) != 8 or set(actual) != expected:
                raise AssertionError(f"unexpected ablation switches: {actual}")
            prominence = PYINAdditionDetector(
                config=reference.DEFAULT_CONFIG, prominence=True
            )
            acf = np.zeros(prominence.max_period + 1)
            periods = np.asarray(
                [
                    prominence.min_period + 5,
                    prominence.min_period + 25,
                    prominence.min_period + 45,
                ]
            )
            acf[periods] = (10.0, 1.0, 4.0)
            picked = prominence._prominent_peak_indices(acf)
            expected_peak = np.asarray([periods[0] - prominence.min_period])
            if not np.array_equal(picked, expected_peak):
                raise AssertionError("prominence fixture did not prune weak peaks")
            gated = PYINAblationAdapter(
                VARIANT_BY_NAME["pyin_volume_gate"],
                self.run_root,
                volume_floor_ratio=self.config.volume_floor_ratio,
                volume_ceiling_percentile=self.config.volume_ceiling_percentile,
                unvoiced_threshold=self.config.unvoiced_threshold,
                praat_switch_cost=self.config.praat_switch_cost,
            ).recording_for(reference.DEFAULT_CONFIG)
            if type(gated.pitch_smoother) is not PYINGlobalVolumeGateSmoother:
                raise AssertionError("volume gate is not attached after the pYIN HMM")
            controlled = PYINAblationAdapter(
                VARIANT_BY_NAME["pyin_volume_praat_voicing"],
                self.run_root,
                volume_floor_ratio=self.config.volume_floor_ratio,
                volume_ceiling_percentile=self.config.volume_ceiling_percentile,
                unvoiced_threshold=self.config.unvoiced_threshold,
                praat_switch_cost=self.config.praat_switch_cost,
            ).recording_for(reference.DEFAULT_CONFIG)
            if type(controlled.pitch_smoother) is not PYINPraatStyleVoicingSmoother:
                raise AssertionError("Praat-style global controller is not attached")
            print("Post-hoc voicing and 2×2×2 streaming contracts passed.")

        def selection(self) -> pd.DataFrame:
            frame = pd.DataFrame(
                [
                    {
                        "dataset": track.dataset,
                        "track_id": track.track_id,
                        "instrument": track.metadata.get("instrument", ""),
                    }
                    for track in self.tracks
                ]
            )
            if self.config.profile in {"balanced", "urmp"}:
                counts = frame.groupby(["dataset", "instrument"]).size()
                for dataset in self.spec.datasets:
                    dataset_counts = counts.loc[dataset]
                    if not len(dataset_counts) or set(dataset_counts) != {
                        self.config.tracks_per_instrument
                    }:
                        raise AssertionError(
                            f"{dataset} is not {self.config.tracks_per_instrument}/instrument: {dataset_counts.to_dict()}"
                        )
            return frame

        def show_selection(self) -> pd.DataFrame:
            frame = self.selection()
            counts = (
                frame.groupby(["dataset", "instrument"], dropna=False)
                .size()
                .rename("tracks")
                .to_frame()
            )
            display(counts)
            print(
                f"{len(frame)} unique tracks × {len(POSTHOC_VARIANTS)} post-hoc or {len(STREAMING_VARIANTS)} streaming variants"
            )
            return frame

        def _streaming_grid(self, example: PitchExample) -> StreamingGrid:
            """Reproduce the live app's capture buffer, hop, and timestamps."""
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            production = AttuneRealtime()
            audio, sample_rate = example.audio(production.DEFAULT_CONFIG.sr)
            duration = min(
                float(self.config.streaming_seconds), len(audio) / float(sample_rate)
            )
            audio = np.ascontiguousarray(
                audio[: int(math.floor(duration * sample_rate))], dtype=np.float32
            )
            duration = len(audio) / float(sample_rate)
            config = production.config_for(
                example.fmin,
                example.fmax,
                sr=int(sample_rate),
                unv_thresh=self.config.unvoiced_threshold,
            )
            detector = production.recording_for(config).pitch_detector
            frame_size = int(detector.FRAME_SIZE)
            integration_size = int(detector.INTEGRATION_SIZE)
            hop_size = int(detector.HOP_SIZE)
            if sample_rate != production.DEFAULT_CONFIG.sr:
                raise AssertionError(
                    f"streaming audio is {sample_rate} Hz, expected {production.DEFAULT_CONFIG.sr} Hz"
                )
            if integration_size != production.DEFAULT_CONFIG.w1:
                raise AssertionError(
                    f"integration size is {integration_size}, expected {production.DEFAULT_CONFIG.w1}"
                )
            if hop_size != production.DEFAULT_CONFIG.h1:
                raise AssertionError(
                    f"hop size is {hop_size}, expected {production.DEFAULT_CONFIG.h1}"
                )
            if len(audio) < frame_size:
                raise ValueError(
                    f"{example.track_id}: {duration:.3f}s streaming excerpt is shorter than the production {frame_size}-sample capture frame"
                )
            starts = np.arange(0, len(audio) - frame_size + 1, hop_size, dtype=np.int64)
            raw_times = 0.5 * integration_size / float(sample_rate) + np.arange(
                starts.size, dtype=np.float64
            ) * hop_size / float(sample_rate)
            times = PitchDetectorBase.PitchEstimate._uniform_grid(raw_times)
            return PitchNotebook.StreamingGrid(
                audio=audio,
                config=config,
                duration=duration,
                sample_rate=int(sample_rate),
                frame_size=frame_size,
                integration_size=integration_size,
                hop_size=hop_size,
                starts=starts,
                times=times,
            )

        @staticmethod
        def _streaming_detector(
            variant: Variant, config: Config
        ) -> tuple[PYINPitchDetector, None]:
            detector_kwargs = {
                "config": config,
                "frame_length": int(config.w1),
                "center": False,
            }
            detector = (
                PYINAdditionDetector(prominence=variant.prominence, **detector_kwargs)
                if variant.prominence or variant.volume_gate
                else PYINPitchDetector(**detector_kwargs)
            )
            return (detector, None)

        @staticmethod
        def _decode_streaming_frame(
            detector: PYINPitchDetector,
            smoother: None,
            frame: np.ndarray,
            config: Config,
            timestamp: float,
        ) -> tuple[float, float, float]:
            """Select a raw candidate without any HMM or implicit voicing gate."""
            pitch = detector.detect_pitch(frame, start_time=timestamp)
            raw_unvoiced_probability = float(pitch.unvoiced_prob)
            frequency = (
                float(
                    config.midi_to_freq(
                        max(pitch.candidate_pitches, key=lambda item: item[1])[0]
                    )
                )
                if pitch.candidate_pitches
                else 0.0
            )
            return (frequency, float(pitch.volume), raw_unvoiced_probability)

        @staticmethod
        def _apply_streaming_gates(
            frequency: float,
            volume: float,
            unvoiced_probability: float,
            variant: Variant,
            config: Config,
            *,
            volume_floor_ratio: float,
            running_volume_ceiling: float,
        ) -> tuple[float, float]:
            """Apply only causal gates and return output plus updated peak RMS."""
            if variant.volume_gate:
                running_volume_ceiling = max(running_volume_ceiling, volume)
                if volume < volume_floor_ratio * running_volume_ceiling:
                    return (0.0, running_volume_ceiling)
            if variant.unvoiced_gate and unvoiced_probability >= config.unv_thresh:
                return (0.0, running_volume_ceiling)
            return (frequency, running_volume_ceiling)

        def _streaming_cache_path(
            self, example: PitchExample, variant: Variant
        ) -> Path:
            from benchmarks.modules.pitch.PitchCache import PitchCache

            return PitchCache.ablation_streaming_cache_path(self, example, variant)

        def _read_streaming_cache(self, path: Path) -> StreamingTrackResult | None:
            from benchmarks.modules.pitch.PitchCache import PitchCache

            return PitchCache.ablation_read_streaming_cache(self, path)

        def _write_streaming_cache(
            self, path: Path, result: StreamingTrackResult
        ) -> None:
            from benchmarks.modules.pitch.PitchCache import PitchCache

            return PitchCache.ablation_write_streaming_cache(self, path, result)

        def _streaming_threshold_evidence_path(
            self, example: PitchExample, *, prominence: bool
        ) -> Path:
            from benchmarks.modules.pitch.PitchCache import PitchCache

            return PitchCache.ablation_streaming_threshold_evidence_path(
                self, example, prominence=prominence
            )

        def _read_streaming_threshold_evidence(
            self, path: Path, grid: StreamingGrid
        ) -> StreamingThresholdEvidence | None:
            from benchmarks.modules.pitch.PitchCache import PitchCache

            return PitchCache.ablation_read_streaming_threshold_evidence(
                self, path, grid
            )

        def _write_streaming_threshold_evidence(
            self, path: Path, evidence: StreamingThresholdEvidence
        ) -> None:
            from benchmarks.modules.pitch.PitchCache import PitchCache

            return PitchCache.ablation_write_streaming_threshold_evidence(
                self, path, evidence
            )

        def _streaming_threshold_evidence(
            self, example: PitchExample, *, prominence: bool, grid: StreamingGrid
        ) -> StreamingThresholdEvidence:
            """Compute one causal frontend pass for an entire threshold grid."""
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            path = self._streaming_threshold_evidence_path(
                example, prominence=prominence
            )
            cached = self._read_streaming_threshold_evidence(path, grid)
            if cached is not None:
                return cached
            variant = next(
                (
                    candidate
                    for candidate in STREAMING_VARIANTS
                    if candidate.prominence == prominence
                    and candidate.volume_gate
                    and (not candidate.unvoiced_gate)
                )
            )
            detector, smoother = self._streaming_detector(variant, grid.config)
            frames = np.lib.stride_tricks.sliding_window_view(
                grid.audio, grid.integration_size
            )[:: grid.hop_size][: len(grid.starts)]
            if len(frames) != len(grid.starts):
                raise AssertionError("threshold evidence does not match capture starts")
            self._decode_streaming_frame(
                detector, smoother, frames[0], grid.config, float(grid.times[0])
            )
            frequencies = np.empty(len(frames), dtype=np.float64)
            volumes = np.empty(len(frames), dtype=np.float64)
            unvoiced_probabilities = np.empty(len(frames), dtype=np.float64)
            for index, (timestamp, frame) in enumerate(zip(grid.times, frames)):
                frequency, volume, unvoiced_probability = self._decode_streaming_frame(
                    detector, smoother, frame, grid.config, float(timestamp)
                )
                frequencies[index] = frequency
                volumes[index] = volume
                unvoiced_probabilities[index] = unvoiced_probability
            evidence = PitchNotebook.StreamingThresholdEvidence(
                times=grid.times.copy(),
                frequencies=PitchDetectorBase.constrain_freqs_to_range(
                    frequencies, example.fmin, example.fmax
                ),
                volumes=volumes,
                unvoiced_probabilities=unvoiced_probabilities,
                duration=grid.duration,
                sample_rate=grid.sample_rate,
                frame_size=grid.frame_size,
                integration_size=grid.integration_size,
                hop_size=grid.hop_size,
            )
            self._write_streaming_threshold_evidence(path, evidence)
            return evidence

        def _stream_variant(
            self, example: PitchExample, variant: Variant, grid: StreamingGrid
        ) -> StreamingTrackResult:
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            path = self._streaming_cache_path(example, variant)
            cached = self._read_streaming_cache(path)
            if cached is not None:
                expected_geometry = (
                    grid.sample_rate,
                    grid.frame_size,
                    grid.integration_size,
                    grid.hop_size,
                )
                cached_geometry = (
                    cached.sample_rate,
                    cached.frame_size,
                    cached.integration_size,
                    cached.hop_size,
                )
                if (
                    cached_geometry == expected_geometry
                    and math.isclose(
                        cached.duration, grid.duration, rel_tol=0.0, abs_tol=1e-12
                    )
                    and (len(cached.estimate.times) == len(grid.times))
                    and np.array_equal(cached.estimate.times, grid.times)
                ):
                    return cached
            detector, smoother = self._streaming_detector(variant, grid.config)
            all_frames = np.lib.stride_tricks.sliding_window_view(
                grid.audio, grid.integration_size
            )[:: grid.hop_size]
            frames = all_frames[: len(grid.starts)]
            if len(frames) != len(grid.starts):
                raise AssertionError("analysis frames do not match capture starts")
            self._decode_streaming_frame(
                detector, smoother, frames[0], grid.config, float(grid.times[0])
            )
            frequencies = np.empty(len(frames), dtype=np.float64)
            call_cpus = np.empty(len(frames), dtype=np.float64)
            call_walls = np.empty(len(frames), dtype=np.float64)
            latencies = np.empty(len(frames), dtype=np.float64)
            compute_time = 0.0
            wall_compute_time = 0.0
            simulated_worker_ready = 0.0
            running_volume_ceiling = 0.0
            for index, (start, timestamp, frame) in enumerate(
                zip(grid.starts, grid.times, frames)
            ):

                def decode_and_gate(frame=frame, timestamp=timestamp) -> float:
                    nonlocal running_volume_ceiling
                    frequency, volume, unvoiced_probability = (
                        self._decode_streaming_frame(
                            detector, smoother, frame, grid.config, float(timestamp)
                        )
                    )
                    gated, running_volume_ceiling = self._apply_streaming_gates(
                        frequency,
                        volume,
                        unvoiced_probability,
                        variant,
                        grid.config,
                        volume_floor_ratio=self.config.volume_floor_ratio,
                        running_volume_ceiling=running_volume_ceiling,
                    )
                    return gated

                frequency, cpu, wall = PitchDetectorBase.measure(decode_and_gate)
                frequencies[index] = PitchDetectorBase.constrain_freqs_to_range(
                    [frequency], example.fmin, example.fmax
                )[0]
                available = (int(start) + grid.frame_size) / grid.sample_rate
                simulated_worker_ready = max(available, simulated_worker_ready) + cpu
                latencies[index] = max(0.0, simulated_worker_ready - float(timestamp))
                call_cpus[index] = cpu
                call_walls[index] = wall
                compute_time += cpu
                wall_compute_time += wall
            estimate = PitchDetectorBase.PitchEstimate.build(
                grid.times,
                frequencies,
                compute_time,
                metadata={
                    "compute_clock": PitchDetectorBase.COMPUTE_CLOCK,
                    "wall_pitch_compute_time": wall_compute_time,
                },
            )
            result = PitchNotebook.StreamingTrackResult(
                estimate=estimate,
                duration=grid.duration,
                sample_rate=grid.sample_rate,
                frame_size=grid.frame_size,
                integration_size=grid.integration_size,
                hop_size=grid.hop_size,
                latencies=latencies,
                call_cpus=call_cpus,
                call_walls=call_walls,
            )
            self._write_streaming_cache(path, result)
            return result

        @staticmethod
        def _percentile(values: np.ndarray, percentile: float) -> float:
            return (
                float(np.percentile(values, percentile))
                if values.size
                else float("nan")
            )

        def _streaming_row(
            self, example: PitchExample, variant: Variant, result: StreamingTrackResult
        ) -> dict[str, Any]:
            reference_mask = example.ref_times <= result.duration + 1e-09
            cropped = replace(
                example,
                ref_times=example.ref_times[reference_mask],
                ref_freqs=example.ref_freqs[reference_mask],
            )
            row = self.benchmarker.score(variant.name, cropped, result.estimate)
            deadline = result.hop_size / result.sample_rate
            lookahead = (
                result.frame_size - 0.5 * result.integration_size
            ) / result.sample_rate
            queue_delays = np.maximum(
                0.0, result.latencies - lookahead - result.call_cpus
            )
            compute_time = result.estimate.compute_seconds
            row.update(
                {
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "prominence": variant.prominence,
                    "volume_gate": variant.volume_gate,
                    "volume_floor_ratio": (
                        self.config.volume_floor_ratio
                        if variant.volume_gate
                        else float("nan")
                    ),
                    "volume_ceiling_percentile": float("nan"),
                    "volume_reference": (
                        "causal running peak" if variant.volume_gate else "off"
                    ),
                    "unvoiced_gate": variant.unvoiced_gate,
                    "unvoiced_threshold": (
                        self.config.unvoiced_threshold
                        if variant.unvoiced_gate
                        else float("nan")
                    ),
                    "unvoiced_reference": (
                        "raw frame-local pYIN probability"
                        if variant.unvoiced_gate
                        else "off"
                    ),
                    "result_label": f"{example.dataset}_streaming",
                    "audio_seconds": result.duration,
                    "realtime_factor": (
                        result.duration / compute_time
                        if compute_time > 0
                        else float("nan")
                    ),
                    "execution_mode": "Streaming",
                    "stream_adapter": "Frame-restarted pYIN on Attune recording grid",
                    "latency_clock": "dedicated-worker process CPU simulation",
                    "streaming_workers": self.config.streaming_workers,
                    "sample_rate_hz": result.sample_rate,
                    "frame_size_samples": result.frame_size,
                    "integration_size_samples": result.integration_size,
                    "analysis_size_samples": result.integration_size,
                    "hop_size_samples": result.hop_size,
                    "update_interval_ms": deadline * 1000.0,
                    "algorithmic_lookahead_ms": lookahead * 1000.0,
                    "mean_service_time_ms": (
                        float(np.mean(result.call_cpus)) * 1000.0
                        if result.call_cpus.size
                        else float("nan")
                    ),
                    "mean_no_queue_latency_ms": (
                        lookahead + float(np.mean(result.call_cpus))
                    )
                    * 1000.0,
                    "p95_service_time_ms": self._percentile(
                        result.call_cpus * 1000.0, 95
                    ),
                    "service_utilization": (
                        float(np.mean(result.call_cpus)) / deadline
                        if result.call_cpus.size
                        else float("nan")
                    ),
                    "mean_observed_wall_time_ms": (
                        float(np.mean(result.call_walls)) * 1000.0
                        if result.call_walls.size
                        else float("nan")
                    ),
                    "median_queue_delay_ms": self._percentile(
                        queue_delays * 1000.0, 50
                    ),
                    "p95_queue_delay_ms": self._percentile(queue_delays * 1000.0, 95),
                    "median_output_latency_ms": self._percentile(
                        result.latencies * 1000.0, 50
                    ),
                    "p95_output_latency_ms": self._percentile(
                        result.latencies * 1000.0, 95
                    ),
                    "deadline_miss_rate": (
                        float(np.mean(result.call_cpus > deadline))
                        if result.call_cpus.size
                        else float("nan")
                    ),
                }
            )
            return row

        def _validate_streaming_rows(self, results: pd.DataFrame) -> None:
            expected_variants = {variant.name for variant in STREAMING_VARIANTS}
            geometry = (
                "sample_rate_hz",
                "frame_size_samples",
                "integration_size_samples",
                "hop_size_samples",
            )
            for (dataset, track_id), group in results.groupby(
                ["dataset", "track_id"], sort=False
            ):
                actual_variants = set(group["variant"])
                if actual_variants != expected_variants:
                    raise RuntimeError(
                        f"{dataset}/{track_id}: streaming variants are {sorted(actual_variants)}, expected {sorted(expected_variants)}"
                    )
                drifted = [
                    column
                    for column in geometry
                    if group[column].nunique(dropna=False) != 1
                ]
                if drifted:
                    raise RuntimeError(
                        f"{dataset}/{track_id}: variants used different streaming geometry for {drifted}"
                    )

        def _run_streaming_example(self, example: PitchExample) -> list[dict[str, Any]]:
            """Run one track serially so all eight variants share one exact grid."""
            grid = self._streaming_grid(example)
            return [
                self._streaming_row(
                    example, variant, self._stream_variant(example, variant, grid)
                )
                for variant in STREAMING_VARIANTS
            ]

        def run_streaming(self) -> pd.DataFrame:
            """Evaluate tracks concurrently while preserving serial frame order."""
            from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

            examples = [
                self.benchmarker.dataset(track.dataset).example(track)
                for track in self.tracks
            ]
            workers = min(self.config.streaming_workers, len(examples))
            print(
                f"Running {len(examples)} streaming track(s) × {len(STREAMING_VARIANTS)} variants with {workers} worker(s)."
            )
            rows: list[dict[str, Any]] = []
            progress = PitchBenchmarker.Progress(compact=True)
            try:
                if workers == 1:
                    for example in examples:
                        rows.extend(self._run_streaming_example(example))
                        progress.update_compact(
                            len(examples) * len(STREAMING_VARIANTS),
                            "pYIN streaming",
                            example.track_id,
                            count=len(STREAMING_VARIANTS),
                        )
                else:
                    context = multiprocessing.get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=workers, mp_context=context
                    ) as pool:
                        futures = {
                            pool.submit(
                                PitchNotebook._streaming_track_worker,
                                self.config,
                                example,
                            ): example
                            for example in examples
                        }
                        for future in as_completed(futures):
                            example = futures[future]
                            rows.extend(future.result())
                            progress.update_compact(
                                len(examples) * len(STREAMING_VARIANTS),
                                "pYIN streaming",
                                example.track_id,
                                count=len(STREAMING_VARIANTS),
                            )
            finally:
                progress.finish()
            for row in rows:
                row["streaming_workers"] = workers
            results = (
                pd.DataFrame(rows)
                .sort_values(["dataset", "track_id", "variant"])
                .reset_index(drop=True)
            )
            expected = len(examples) * len(STREAMING_VARIANTS)
            if len(results) != expected:
                raise RuntimeError(
                    f"expected {expected} streaming rows, received {len(results)}"
                )
            self._validate_streaming_rows(results)
            self.streaming_run_root.mkdir(parents=True, exist_ok=True)
            path = self.streaming_run_root / "rows.csv"
            results.to_csv(path, index=False)
            print(f"Wrote {len(results)} streaming rows to {path}")
            return results

        def _streaming_threshold_rows_for_example(
            self,
            example: PitchExample,
            thresholds: tuple[float, ...],
            prominences: tuple[bool, ...],
            volume_modes: tuple[bool, ...],
            volume_floor_ratios: tuple[float, ...],
        ) -> list[dict[str, Any]]:
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            grid = self._streaming_grid(example)
            reference_mask = example.ref_times <= grid.duration + 1e-09
            cropped = replace(
                example,
                ref_times=example.ref_times[reference_mask],
                ref_freqs=example.ref_freqs[reference_mask],
            )
            rows: list[dict[str, Any]] = []
            for prominence in prominences:
                evidence = self._streaming_threshold_evidence(
                    example, prominence=prominence, grid=grid
                )
                for volume_gate in volume_modes:
                    ratios: tuple[float | None, ...] = (
                        volume_floor_ratios if volume_gate else (None,)
                    )
                    for volume_floor_ratio in ratios:
                        allowed_by_volume = (
                            evidence.volumes
                            >= float(volume_floor_ratio)
                            * np.maximum.accumulate(evidence.volumes)
                            if volume_floor_ratio is not None
                            else np.ones(len(evidence.times), dtype=bool)
                        )
                        for threshold in thresholds:
                            voiced = allowed_by_volume & (
                                evidence.unvoiced_probabilities < float(threshold)
                            )
                            estimate = PitchDetectorBase.PitchEstimate.build(
                                evidence.times,
                                np.where(voiced, evidence.frequencies, 0.0),
                                0.0,
                            )
                            row = self.benchmarker.score(
                                "pyin_streaming_threshold_sweep", cropped, estimate
                            )
                            row.update(
                                {
                                    "prominence": prominence,
                                    "volume_gate": volume_gate,
                                    "volume_floor_ratio": (
                                        float(volume_floor_ratio)
                                        if volume_floor_ratio is not None
                                        else float("nan")
                                    ),
                                    "unvoiced_threshold": float(threshold),
                                    "execution_mode": "Streaming threshold sweep",
                                    "sample_rate_hz": evidence.sample_rate,
                                    "frame_size_samples": evidence.frame_size,
                                    "integration_size_samples": evidence.integration_size,
                                    "hop_size_samples": evidence.hop_size,
                                }
                            )
                            rows.append(row)
            return rows

        def run_streaming_threshold_sweep(
            self,
            thresholds: tuple[float, ...] | None = None,
            *,
            dataset: str = "urmp",
            prominences: tuple[bool, ...] = (False, True),
            volume_modes: tuple[bool, ...] = (False, True),
            volume_floor_ratios: tuple[float, ...] | None = None,
        ) -> pd.DataFrame:
            """Sweep causal thresholds while computing pYIN evidence only once.

            Every threshold for a candidate mode sees identical decoded pitches,
            RMS values, frame times, and raw pYIN unvoiced probabilities. This is
            an accuracy search; latency belongs to the fixed streaming factorial
            because masking another threshold adds no material service time.
            """
            from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

            thresholds = self._validated_thresholds(
                self.DEFAULT_UNVOICED_THRESHOLDS if thresholds is None else thresholds
            )
            prominences = tuple(dict.fromkeys((bool(value) for value in prominences)))
            volume_modes = tuple(dict.fromkeys((bool(value) for value in volume_modes)))
            if not prominences or not volume_modes:
                raise ValueError("at least one prominence and volume mode is required")
            volume_floor_ratios = self._validated_volume_floor_ratios(
                (self.config.volume_floor_ratio,)
                if volume_floor_ratios is None
                else volume_floor_ratios
            )
            examples = [
                self.benchmarker.dataset(track.dataset).example(track)
                for track in self.tracks
                if track.dataset == dataset
            ]
            if not examples:
                raise ValueError(f"dataset {dataset!r} is absent from this profile")
            workers = min(self.config.streaming_workers, len(examples))
            rows: list[dict[str, Any]] = []
            progress = PitchBenchmarker.Progress(compact=True)
            try:
                if workers == 1:
                    for example in examples:
                        rows.extend(
                            self._streaming_threshold_rows_for_example(
                                example,
                                thresholds,
                                prominences,
                                volume_modes,
                                volume_floor_ratios,
                            )
                        )
                        progress.update_compact(
                            len(examples),
                            "pYIN causal unvoiced-threshold sweep",
                            example.track_id,
                            count=1,
                        )
                else:
                    context = multiprocessing.get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=workers, mp_context=context
                    ) as pool:
                        futures = {
                            pool.submit(
                                PitchNotebook._streaming_threshold_track_worker,
                                self.config,
                                example,
                                thresholds,
                                prominences,
                                volume_modes,
                                volume_floor_ratios,
                            ): example
                            for example in examples
                        }
                        for future in as_completed(futures):
                            example = futures[future]
                            rows.extend(future.result())
                            progress.update_compact(
                                len(examples),
                                "pYIN causal unvoiced-threshold sweep",
                                example.track_id,
                                count=1,
                            )
            finally:
                progress.finish()
            results = (
                pd.DataFrame(rows)
                .sort_values(
                    [
                        "dataset",
                        "track_id",
                        "prominence",
                        "volume_gate",
                        "volume_floor_ratio",
                        "unvoiced_threshold",
                    ]
                )
                .reset_index(drop=True)
            )
            expected = (
                len(examples)
                * len(thresholds)
                * len(prominences)
                * sum(
                    (
                        len(volume_floor_ratios) if enabled else 1
                        for enabled in volume_modes
                    )
                )
            )
            if len(results) != expected:
                raise RuntimeError(
                    f"expected {expected} streaming threshold rows, received {len(results)}"
                )
            self.streaming_threshold_sweep_root.mkdir(parents=True, exist_ok=True)
            path = self.streaming_threshold_sweep_root / f"{dataset}_rows.csv"
            results.to_csv(path, index=False)
            print(f"Wrote {len(results)} streaming threshold rows to {path}")
            return results

        def streaming_grid_summary(self, results: pd.DataFrame) -> pd.DataFrame:
            self._validate_streaming_rows(results)
            per_track = results.drop_duplicates(["dataset", "track_id"])
            return per_track.groupby("dataset", sort=False).agg(
                tracks=("track_id", "nunique"),
                sample_rate_hz=("sample_rate_hz", "first"),
                integration_size_samples=("integration_size_samples", "first"),
                hop_size_samples=("hop_size_samples", "first"),
                update_interval_ms=("update_interval_ms", "first"),
                min_capture_frame_samples=("frame_size_samples", "min"),
                max_capture_frame_samples=("frame_size_samples", "max"),
            )

        def show_streaming_grid(self, results: pd.DataFrame) -> pd.DataFrame:
            summary = self.streaming_grid_summary(results)
            display(summary.style.format({"update_interval_ms": "{:.3f}"}))
            self.streaming_run_root.mkdir(parents=True, exist_ok=True)
            summary.to_csv(self.streaming_run_root / "grid.csv")
            return summary

        def streaming_summary(self, results: pd.DataFrame) -> pd.DataFrame:
            rows: list[dict[str, Any]] = []
            for variant in STREAMING_VARIANTS:
                selected = results.loc[results["variant"] == variant.name]
                compute_time = float(selected["pitch_compute_time"].sum())
                row = {
                    "variant": variant.name,
                    **{
                        metric: float(selected[metric].mean())
                        for metric in self.METRICS
                    },
                    "Audio / compute": (
                        float(selected["audio_seconds"].sum()) / compute_time
                        if compute_time > 0
                        else float("nan")
                    ),
                    "Deadline miss rate": float(selected["deadline_miss_rate"].mean()),
                    "Algorithmic look-ahead (ms)": float(
                        selected["algorithmic_lookahead_ms"].mean()
                    ),
                    "Mean no-queue latency (ms)": float(
                        selected["mean_no_queue_latency_ms"].mean()
                    ),
                    "Median output latency (ms)": float(
                        selected["median_output_latency_ms"].mean()
                    ),
                    "P95 output latency (ms)": float(
                        selected["p95_output_latency_ms"].mean()
                    ),
                    "Mean service time (ms)": float(
                        selected["mean_service_time_ms"].mean()
                    ),
                    "Service / deadline": float(selected["service_utilization"].mean()),
                    "Median queue delay (ms)": float(
                        selected["median_queue_delay_ms"].mean()
                    ),
                    "P95 queue delay (ms)": float(
                        selected["p95_queue_delay_ms"].mean()
                    ),
                }
                rows.append(row)
            return pd.DataFrame(rows).set_index("variant")

        def show_streaming_summary(self, results: pd.DataFrame) -> pd.DataFrame:
            summary = self.streaming_summary(results)
            display(summary.style.format("{:.4f}"))
            self.streaming_run_root.mkdir(parents=True, exist_ok=True)
            summary.to_csv(self.streaming_run_root / "summary.csv")
            return summary

        def show_streaming_effects(self, results: pd.DataFrame) -> pd.DataFrame:
            effects = self.effects(results, self.STREAMING_CONTRASTS)
            display(
                effects.style.format(
                    {
                        "Feature on": "{:.4f}",
                        "Feature off": "{:.4f}",
                        "Gain": "{:+.4f}",
                        "CI low": "{:+.4f}",
                        "CI high": "{:+.4f}",
                    }
                )
            )
            self.streaming_run_root.mkdir(parents=True, exist_ok=True)
            effects.to_csv(self.streaming_run_root / "paired_effects.csv", index=False)
            return effects

        def run(self) -> pd.DataFrame:
            from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

            rows: list[dict[str, Any]] = []
            progress = PitchBenchmarker.Progress(compact=True)
            try:
                for track in self.tracks:
                    dataset = self.benchmarker.dataset(track.dataset)
                    example = dataset.example(track)
                    for variant in POSTHOC_VARIANTS:
                        adapter = PYINAblationAdapter(
                            variant,
                            self.run_root,
                            volume_floor_ratio=self.config.volume_floor_ratio,
                            volume_ceiling_percentile=self.config.volume_ceiling_percentile,
                            unvoiced_threshold=self.config.unvoiced_threshold,
                            praat_switch_cost=self.config.praat_switch_cost,
                        )
                        estimate = adapter.estimate(
                            example,
                            use_cache=self.config.use_cache,
                            force_reanalysis=self.config.force,
                        )
                        row = self.benchmarker.score(variant.name, example, estimate)
                        row.update(
                            {
                                "variant": variant.name,
                                "variant_label": variant.label,
                                "prominence": variant.prominence,
                                "volume_gate": variant.volume_gate,
                                "volume_floor_ratio": (
                                    self.config.volume_floor_ratio
                                    if variant.volume_gate
                                    else float("nan")
                                ),
                                "volume_ceiling_percentile": (
                                    self.config.volume_ceiling_percentile
                                    if variant.volume_gate
                                    else float("nan")
                                ),
                                "volume_reference": (
                                    "global centered-RMS percentile"
                                    if variant.volume_gate
                                    else "off"
                                ),
                                "unvoiced_gate": variant.unvoiced_gate,
                                "unvoiced_threshold": (
                                    self.config.unvoiced_threshold
                                    if variant.praat_controller
                                    else float("nan")
                                ),
                                "praat_controller": variant.praat_controller,
                                "praat_switch_cost": (
                                    self.config.praat_switch_cost
                                    if variant.praat_controller
                                    else float("nan")
                                ),
                            }
                        )
                        rows.append(row)
                        progress.update_compact(
                            len(self.tracks) * len(POSTHOC_VARIANTS),
                            f"pYIN post-hoc {variant.name}",
                            example.track_id,
                            count=1,
                        )
            finally:
                progress.finish()
            result = (
                pd.DataFrame(rows)
                .sort_values(["dataset", "track_id", "variant"])
                .reset_index(drop=True)
            )
            expected = len(self.tracks) * len(POSTHOC_VARIANTS)
            if len(result) != expected:
                raise RuntimeError(
                    f"expected {expected} scored rows, received {len(result)}"
                )
            self.run_root.mkdir(parents=True, exist_ok=True)
            result.to_csv(self.run_root / "rows.csv", index=False)
            print(f"Wrote {len(result)} paired rows to {self.run_root / 'rows.csv'}")
            return result

        def _posthoc_threshold_rows_for_example(
            self,
            example: PitchExample,
            thresholds: tuple[float, ...],
            switch_costs: tuple[float, ...],
            volume_floor_ratios: tuple[float, ...],
        ) -> list[dict[str, Any]]:
            from benchmarks.modules.pitch.PitchCache import PitchCache
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            adapter = PYINAblationAdapter(
                VARIANT_BY_NAME["pyin_volume_gate"],
                self.run_root,
                volume_floor_ratio=self.config.volume_floor_ratio,
                volume_ceiling_percentile=self.config.volume_ceiling_percentile,
                unvoiced_threshold=self.config.unvoiced_threshold,
                praat_switch_cost=self.config.praat_switch_cost,
            )
            config = adapter.config_for(example.fmin, example.fmax)
            hit = adapter.cache(example).read(PitchCache.RAW, config)
            if hit is None:
                adapter.estimate(
                    example,
                    use_cache=self.config.use_cache,
                    force_reanalysis=self.config.force,
                )
                hit = adapter.cache(example).read(PitchCache.RAW, config)
            if hit is None:
                raise RuntimeError(
                    f"raw pYIN evidence is unavailable for {example.track_id}"
                )
            raw_pitches = list(hit[0].data)
            base_smoother = PYINPitchSmoother(config=config)
            states = base_smoother.decode(raw_pitches)
            present = np.asarray(
                [pitch is not None for pitch in raw_pitches], dtype=bool
            )
            times = np.asarray(
                [float(pitch.time) for pitch in raw_pitches if pitch is not None],
                dtype=np.float64,
            )
            decoded_frequencies = np.asarray(
                base_smoother.bin_freqs[states % base_smoother.n_pitch_bins],
                dtype=np.float64,
            )
            decoded_frequencies = PitchDetectorBase.constrain_freqs_to_range(
                decoded_frequencies, example.fmin, example.fmax
            )
            rows: list[dict[str, Any]] = []
            for volume_floor_ratio in volume_floor_ratios:
                for threshold in thresholds:
                    for switch_cost in switch_costs:
                        controller = PYINPraatStyleVoicingSmoother(
                            config=config,
                            relative_floor=volume_floor_ratio,
                            ceiling_percentile=self.config.volume_ceiling_percentile,
                            unvoiced_threshold=threshold,
                            switch_cost=switch_cost,
                        )
                        voiced = controller._controller_mask(raw_pitches)
                        estimate = PitchDetectorBase.PitchEstimate.build(
                            times,
                            np.where(voiced, decoded_frequencies, 0.0)[present],
                            0.0,
                        )
                        row = self.benchmarker.score(
                            "pyin_posthoc_threshold_sweep", example, estimate
                        )
                        row.update(
                            {
                                "unvoiced_threshold": float(threshold),
                                "praat_switch_cost": float(switch_cost),
                                "volume_gate": True,
                                "volume_floor_ratio": float(volume_floor_ratio),
                                "volume_ceiling_percentile": self.config.volume_ceiling_percentile,
                                "execution_mode": "Post-hoc threshold sweep",
                            }
                        )
                        rows.append(row)
            return rows

        def run_posthoc_threshold_sweep(
            self,
            thresholds: tuple[float, ...] | None = None,
            *,
            switch_costs: tuple[float, ...] | None = None,
            volume_floor_ratios: tuple[float, ...] | None = None,
            dataset: str = "urmp",
        ) -> pd.DataFrame:
            """Search the global controller without repeating pitch detection."""
            from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

            thresholds = self._validated_thresholds(
                self.DEFAULT_UNVOICED_THRESHOLDS if thresholds is None else thresholds
            )
            switch_costs = self._validated_switch_costs(
                self.DEFAULT_PRAAT_SWITCH_COSTS
                if switch_costs is None
                else switch_costs
            )
            volume_floor_ratios = self._validated_volume_floor_ratios(
                (self.config.volume_floor_ratio,)
                if volume_floor_ratios is None
                else volume_floor_ratios
            )
            examples = [
                self.benchmarker.dataset(track.dataset).example(track)
                for track in self.tracks
                if track.dataset == dataset
            ]
            if not examples:
                raise ValueError(f"dataset {dataset!r} is absent from this profile")
            rows: list[dict[str, Any]] = []
            progress = PitchBenchmarker.Progress(compact=True)
            try:
                for example in examples:
                    rows.extend(
                        self._posthoc_threshold_rows_for_example(
                            example, thresholds, switch_costs, volume_floor_ratios
                        )
                    )
                    progress.update_compact(
                        len(examples), "pYIN post-hoc threshold sweep", example.track_id
                    )
            finally:
                progress.finish()
            results = (
                pd.DataFrame(rows)
                .sort_values(
                    ["dataset", "track_id", "unvoiced_threshold", "praat_switch_cost"]
                )
                .reset_index(drop=True)
            )
            expected = (
                len(examples)
                * len(thresholds)
                * len(switch_costs)
                * len(volume_floor_ratios)
            )
            if len(results) != expected:
                raise RuntimeError(
                    f"expected {expected} post-hoc threshold rows, received {len(results)}"
                )
            self.threshold_sweep_root.mkdir(parents=True, exist_ok=True)
            path = self.threshold_sweep_root / f"{dataset}_posthoc_rows.csv"
            results.to_csv(path, index=False)
            print(f"Wrote {len(results)} post-hoc threshold rows to {path}")
            return results

        def threshold_summary(
            self, results: pd.DataFrame, *, streaming: bool
        ) -> pd.DataFrame:
            dimensions = (
                [
                    "prominence",
                    "volume_gate",
                    "volume_floor_ratio",
                    "unvoiced_threshold",
                ]
                if streaming
                else ["volume_floor_ratio", "unvoiced_threshold", "praat_switch_cost"]
            )
            missing = [column for column in dimensions if column not in results]
            if missing:
                raise ValueError(f"threshold rows are missing {missing}")
            grouped = results.groupby(dimensions, sort=True, dropna=False)
            summary = grouped[list(self.METRICS)].mean()
            summary.insert(0, "Tracks", grouped["track_id"].nunique())
            return summary

        def show_threshold_summary(
            self, results: pd.DataFrame, *, streaming: bool
        ) -> pd.DataFrame:
            summary = self.threshold_summary(results, streaming=streaming)
            display(summary.style.format({metric: "{:.4f}" for metric in self.METRICS}))
            output_root = (
                self.streaming_threshold_sweep_root
                if streaming
                else self.threshold_sweep_root
            )
            output_root.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_root / "summary.csv")
            return summary

        def promotion_test(
            self,
            candidates: pd.DataFrame,
            competitor_result: Any,
            *,
            competitor: str,
            candidate_columns: tuple[str, ...],
        ) -> pd.DataFrame:
            """Paired OA/RPA superiority test on the exact shared track set.

            This is an exploratory promotion gate. A selected setting must still be
            frozen and rerun on the full benchmark; confidence intervals computed
            after choosing the best row from this same table are not confirmatory.
            """
            competitor_rows = getattr(competitor_result, "rows", competitor_result)
            if not isinstance(competitor_rows, pd.DataFrame):
                raise TypeError("competitor_result must be a DataFrame or SuiteResult")
            reference = competitor_rows.loc[
                competitor_rows["model"] == competitor,
                ["dataset", "track_id", "Overall Accuracy", "Raw Pitch Accuracy"],
            ].copy()
            if reference.empty:
                raise ValueError(f"competitor {competitor!r} is absent")
            if reference.duplicated(["dataset", "track_id"]).any():
                raise ValueError("competitor rows are not unique per dataset/track")
            reports: list[dict[str, Any]] = []
            grouper: str | list[str] = (
                candidate_columns[0]
                if len(candidate_columns) == 1
                else list(candidate_columns)
            )
            for key, group in candidates.groupby(grouper, sort=True, dropna=False):
                values = (key,) if len(candidate_columns) == 1 else tuple(key)
                paired = group.merge(
                    reference,
                    on=["dataset", "track_id"],
                    suffixes=("_candidate", "_competitor"),
                    validate="one_to_one",
                )
                if paired.empty:
                    continue
                report = dict(zip(candidate_columns, values))
                report["Competitor"] = competitor
                report["Tracks"] = len(paired)
                clear = True
                for metric_index, metric in enumerate(
                    ("Overall Accuracy", "Raw Pitch Accuracy")
                ):
                    differences = (
                        paired[f"{metric}_candidate"].to_numpy()
                        - paired[f"{metric}_competitor"].to_numpy()
                    )
                    low, high = self._stratified_bootstrap_ci(
                        differences,
                        paired["dataset"].to_numpy(),
                        draws=self.spec.bootstrap_samples,
                        seed=self.config.seed + metric_index,
                    )
                    short = "OA" if metric == "Overall Accuracy" else "RPA"
                    report[f"Candidate {short}"] = paired[f"{metric}_candidate"].mean()
                    report[f"Competitor {short}"] = paired[
                        f"{metric}_competitor"
                    ].mean()
                    report[f"{short} margin"] = differences.mean()
                    report[f"{short} CI low"] = low
                    report[f"{short} CI high"] = high
                    clear &= low > 0.0
                report["Clear OA+RPA win"] = clear
                reports.append(report)
            if not reports:
                raise ValueError("candidate and competitor rows share no tracks")
            return (
                pd.DataFrame(reports)
                .sort_values(
                    ["Clear OA+RPA win", "OA margin", "RPA margin"], ascending=False
                )
                .reset_index(drop=True)
            )

        def show_promotion_test(
            self,
            candidates: pd.DataFrame,
            competitor_result: Any,
            *,
            competitor: str,
            candidate_columns: tuple[str, ...],
        ) -> pd.DataFrame:
            report = self.promotion_test(
                candidates,
                competitor_result,
                competitor=competitor,
                candidate_columns=candidate_columns,
            )
            numeric = {
                column: "{:+.4f}" if "margin" in column or "CI" in column else "{:.4f}"
                for column in report.columns
                if column
                not in {*candidate_columns, "Competitor", "Tracks", "Clear OA+RPA win"}
            }
            display(report.style.format(numeric))
            return report

        def summary(self, results: pd.DataFrame) -> pd.DataFrame:
            order = [variant.name for variant in POSTHOC_VARIANTS]
            return (
                results.groupby("variant", sort=False)[list(self.METRICS)]
                .mean()
                .reindex(order)
            )

        def show_summary(self, results: pd.DataFrame) -> pd.DataFrame:
            summary = self.summary(results)
            display(summary.style.format("{:.4f}"))
            summary.to_csv(self.run_root / "summary.csv")
            return summary

        def dataset_results(self, results: pd.DataFrame, dataset: str) -> pd.DataFrame:
            selected = results.loc[results["dataset"] == dataset].copy()
            if selected.empty:
                raise ValueError(f"results contain no {dataset!r} tracks")
            return selected

        def show_dataset_summary(
            self, results: pd.DataFrame, dataset: str, *, streaming: bool = False
        ) -> pd.DataFrame:
            selected = self.dataset_results(results, dataset)
            summary = (
                self.streaming_summary(selected)
                if streaming
                else self.summary(selected)
            )
            display(summary.style.format("{:.4f}"))
            output_root = self.streaming_run_root if streaming else self.run_root
            output_root.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_root / f"{dataset}_summary.csv")
            return summary

        def show_dataset_effects(
            self, results: pd.DataFrame, dataset: str, *, streaming: bool = False
        ) -> pd.DataFrame:
            selected = self.dataset_results(results, dataset)
            contrasts = (
                self.STREAMING_CONTRASTS if streaming else self.POSTHOC_CONTRASTS
            )
            effects = self.effects(selected, contrasts)
            display(
                effects.style.format(
                    {
                        "Feature on": "{:.4f}",
                        "Feature off": "{:.4f}",
                        "Gain": "{:+.4f}",
                        "CI low": "{:+.4f}",
                        "CI high": "{:+.4f}",
                    }
                )
            )
            output_root = self.streaming_run_root if streaming else self.run_root
            output_root.mkdir(parents=True, exist_ok=True)
            effects.to_csv(output_root / f"{dataset}_paired_effects.csv", index=False)
            return effects

        @staticmethod
        def _stratified_bootstrap_ci(
            values: np.ndarray, strata: np.ndarray, *, draws: int, seed: int
        ) -> tuple[float, float]:
            rng = np.random.default_rng(seed)
            boot_sum = np.zeros(draws, dtype=np.float64)
            count = 0
            for stratum in np.unique(strata):
                group = values[strata == stratum]
                indices = rng.integers(0, len(group), size=(draws, len(group)))
                boot_sum += group[indices].sum(axis=1)
                count += len(group)
            low, high = np.quantile(boot_sum / count, [0.025, 0.975])
            return (float(low), float(high))

        def effects(
            self,
            results: pd.DataFrame,
            contrasts: tuple[tuple[str, str, str, str], ...] | None = None,
        ) -> pd.DataFrame:
            reports: list[dict[str, Any]] = []
            pair_keys = ["dataset", "track_id"]
            contrasts = self.POSTHOC_CONTRASTS if contrasts is None else contrasts
            expected_pairs = results[pair_keys].drop_duplicates().shape[0]
            for contrast_index, (
                component,
                context,
                feature_on,
                feature_off,
            ) in enumerate(contrasts):
                on = results.loc[
                    results["variant"] == feature_on, [*pair_keys, *self.METRICS]
                ]
                off = results.loc[
                    results["variant"] == feature_off, [*pair_keys, *self.METRICS]
                ]
                paired = on.merge(
                    off, on=pair_keys, suffixes=("_on", "_off"), validate="one_to_one"
                )
                if len(paired) != expected_pairs:
                    raise RuntimeError(
                        f"incomplete pair set for {feature_on} vs {feature_off}"
                    )
                strata = paired["dataset"].to_numpy()
                for metric_index, metric in enumerate(self.METRICS):
                    gain = self.METRIC_DIRECTION[metric] * (
                        paired[f"{metric}_on"].to_numpy()
                        - paired[f"{metric}_off"].to_numpy()
                    )
                    low, high = self._stratified_bootstrap_ci(
                        gain,
                        strata,
                        draws=self.spec.bootstrap_samples,
                        seed=self.config.seed + 100 * contrast_index + metric_index,
                    )
                    evidence = (
                        "favours addition"
                        if low > 0
                        else "favours reference" if high < 0 else "inconclusive"
                    )
                    reports.append(
                        {
                            "Component": component,
                            "Context": context,
                            "Metric": metric,
                            "Tracks": len(gain),
                            "Feature on": paired[f"{metric}_on"].mean(),
                            "Feature off": paired[f"{metric}_off"].mean(),
                            "Gain": gain.mean(),
                            "CI low": low,
                            "CI high": high,
                            "Evidence": evidence,
                        }
                    )
            return pd.DataFrame(reports)

        def show_effects(self, results: pd.DataFrame) -> pd.DataFrame:
            effects = self.effects(results)
            display(
                effects.style.format(
                    {
                        "Feature on": "{:.4f}",
                        "Feature off": "{:.4f}",
                        "Gain": "{:+.4f}",
                        "CI low": "{:+.4f}",
                        "CI high": "{:+.4f}",
                    }
                )
            )
            effects.to_csv(self.run_root / "paired_effects.csv", index=False)
            return effects

        def plot(self, results: pd.DataFrame):
            summary = self.summary(results)
            metrics = ("Overall Accuracy", "Raw Pitch Accuracy", "Voicing False Alarm")
            labels = [variant.label for variant in VARIANTS]
            figure, axes = plt.subplots(
                1, len(metrics), figsize=(14, 3.8), constrained_layout=True
            )
            colors = ("#315f72", "#d29b62")
            for axis, metric in zip(axes, metrics):
                axis.bar(labels, summary[metric], color=colors)
                axis.set_title(metric)
                axis.tick_params(axis="x", rotation=25)
                axis.grid(axis="y", alpha=0.25)
            figure.suptitle(
                f"pYIN post-hoc voicing controller — {self.config.profile} profile"
            )
            path = self.run_root / "metrics.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.show()
            print(path)
            return figure

    @staticmethod
    def _streaming_track_worker(
        config: PyinAblationConfig, example: PitchExample
    ) -> list[dict[str, Any]]:
        """Spawn-safe entry point for one independently replayed track."""
        return PitchNotebook.PyinAblation(config)._run_streaming_example(example)

    @staticmethod
    def _streaming_threshold_track_worker(
        config: PyinAblationConfig,
        example: PitchExample,
        thresholds: tuple[float, ...],
        prominences: tuple[bool, ...],
        volume_modes: tuple[bool, ...],
        volume_floor_ratios: tuple[float, ...],
    ) -> list[dict[str, Any]]:
        """Spawn-safe entry point for one threshold-sweep evidence pass."""
        return PitchNotebook.PyinAblation(config)._streaming_threshold_rows_for_example(
            example, thresholds, prominences, volume_modes, volume_floor_ratios
        )

    "Runs the suites the notebook narrates, reusing completed runs."
    AUDIO_CORPORA: ClassVar[tuple[str, ...]] = (
        Bach10.name,
        "mdb-stem-synth",
        "mdb-melody-synth",
    )
    URMP_CORPUS: ClassVar[str] = URMP.name
    PRELIMINARY_CORPUS: ClassVar[str] = Bach10.name
    STRATUM_KEYS: ClassVar[tuple[str, ...]] = ("Ensemble", "Instrument")
    ATTUNE_CACHE_VARIANT: ClassVar[str] = (
        f"attune_pyin_plus_ordinary__live_unv_{PYIN_DEFAULT_UNV_THRESH:g}_volume_{PYIN_DEFAULT_MIN_VOLUME:g}__posthoc_unv_{PYIN_POSTHOC_UNV_THRESH:g}_volume_{PYIN_POSTHOC_MIN_VOLUME:g}".replace(
            ".", "p"
        )
    )

    def __init__(self, config: NotebookConfig | None = None) -> None:
        self.config = config or PitchNotebook.NotebookConfig()
        self.runs_root = RESULTS_ROOT / "pitch" / "notebook_runs"
        pd.set_option("display.float_format", lambda value: f"{value:.4f}")

    @property
    def methods(self) -> list[str]:
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        if self.config.methods:
            return list(self.config.methods)
        detectors = (
            PitchBenchmarker.available_detectors()
            if self.config.include_slow_methods
            else PitchBenchmarker.default_detectors()
        )
        return [detector.name for detector in detectors]

    def show_configuration(self) -> pd.DataFrame:
        rows = [
            ("workers", self.config.workers),
            ("streaming workers", self.config.streaming_workers),
            (
                "preliminary tracks per instrument",
                self.config.preliminary_tracks_per_instrument,
            ),
            (
                "full CocoChorales stems per stratum",
                self.config.coco_per_stratum or "all",
            ),
            ("random seed", self.config.seed),
            ("methods", ", ".join(self.methods)),
            ("estimate caches", "reused" if self.config.use_cache else "ignored"),
            (
                "degradation SNRs",
                ", ".join(
                    (self.snr_label(snr) for snr in self.config.degradation_snrs)
                ),
            ),
            ("run cache", self.runs_root),
        ]
        frame = pd.DataFrame(rows, columns=("setting", "value"))
        display(frame)
        return frame

    def show_availability(self) -> pd.DataFrame:
        """Which methods this environment can actually run, before any long cell."""
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        options = self.options()
        rows = []
        for method in self.methods:
            try:
                with PitchBenchmarker.silence_runtime():
                    detector = PitchBenchmarker.detector_for(method, options)
                    detector.ensure_available()
                status, detail = ("OK", "")
            except PitchDetectorBase.Unavailable as exc:
                status, detail = ("MISSING", str(exc).splitlines()[-1].strip())
            rows.append(
                {
                    "method": method,
                    "status": status,
                    "description": detector.description,
                    "detail": detail,
                }
            )
        frame = pd.DataFrame(rows)
        display(frame)
        return frame

    @staticmethod
    def _read_run(path: Path) -> pd.DataFrame | None:
        from benchmarks.modules.pitch.PitchCache import PitchCache

        return PitchCache.notebook_read_run(path)

    def options(self, **overrides: Any) -> PitchBenchmarker.Options:
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        base = PitchBenchmarker.Options(
            seed=self.config.seed,
            use_cache=self.config.use_cache,
            per_stratum=self.config.coco_per_stratum,
            quiet_runtime=True,
        )
        return replace(base, **overrides)

    @staticmethod
    def _resolve_methods(
        methods: Collection[str] | None, available: Sequence[str]
    ) -> list[str]:
        """Return a validated selection in the suite's canonical order."""
        if methods is None:
            return list(available)
        requested = {methods} if isinstance(methods, str) else set(methods)
        if not requested:
            raise ValueError("methods cannot be empty; omit it to select all methods")
        unknown = requested.difference(available)
        if unknown:
            choices = ", ".join(available)
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"unknown or disabled method(s): {names}; choose from {choices}"
            )
        return [method for method in available if method in requested]

    @staticmethod
    def _merge_method_rows(
        existing: pd.DataFrame | None,
        fresh: pd.DataFrame,
        replaced_methods: Sequence[str],
        method_order: Sequence[str],
    ) -> pd.DataFrame:
        """Replace selected methods while retaining every other cached row."""
        frames: list[pd.DataFrame] = []
        if existing is not None and (not existing.empty):
            if "model" not in existing.columns:
                raise ValueError("cached suite rows have no model column")
            frames.append(existing.loc[~existing["model"].isin(replaced_methods)])
        if not fresh.empty:
            frames.append(fresh)
        if not frames:
            return pd.DataFrame()
        rows = pd.concat(frames, ignore_index=True, sort=False)
        if "model" in rows.columns:
            rows = rows.loc[rows["model"].isin(method_order)]
        row_key = [
            column
            for column in ("model", "dataset", "track_id")
            if column in rows.columns
        ]
        if row_key:
            rows = rows.drop_duplicates(subset=row_key, keep="last")
        if "model" in rows.columns:
            order = {method: index for index, method in enumerate(method_order)}
            rows = rows.assign(
                _method_order=rows["model"].map(order).fillna(len(order))
            )
            sort_columns = ["_method_order"] + [
                column for column in ("dataset", "track_id") if column in rows.columns
            ]
            rows = rows.sort_values(sort_columns, kind="stable").drop(
                columns="_method_order"
            )
        return rows.reset_index(drop=True)

    @staticmethod
    def _stale_cached_methods(
        rows: pd.DataFrame | None, methods: Sequence[str]
    ) -> list[str]:
        """Return local methods whose saved suite rows predate their algorithm."""
        from benchmarks.modules.pitch.PitchCache import PitchCache

        if rows is None or rows.empty or "model" not in rows.columns:
            return []
        stale: list[str] = []
        if "attune" in methods:
            attune_rows = rows.loc[rows["model"] == "attune"]
            if (
                attune_rows.empty
                or "pitch_cache_version" not in attune_rows.columns
                or (
                    not (
                        pd.to_numeric(
                            attune_rows["pitch_cache_version"], errors="coerce"
                        )
                        == PitchCache.VERSION
                    ).all()
                )
            ):
                stale.append("attune")
        return stale

    def _run_methods(
        self,
        methods: Sequence[str],
        datasets: tuple[str, ...],
        max_tracks: int | None,
        force_reanalysis: bool,
        **option_overrides: Any,
    ) -> pd.DataFrame:
        """Run and report one method subset with a shared suite configuration."""
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        if not methods:
            return pd.DataFrame()
        benchmarker = PitchBenchmarker(
            self.options(
                datasets=datasets,
                max_tracks=max_tracks,
                force_reanalysis=force_reanalysis,
                **option_overrides,
            )
        )
        jobs, counts = benchmarker.plan(
            methods, benchmarker.tracks(), workers=self.config.workers
        )
        for method, count in counts.items():
            if force_reanalysis:
                print(f"  {method}: explicitly recomputing {count['total']} tracks")
            elif benchmarker.options.use_cache:
                print(
                    f"  {method}: {count['cached']}/{count['total']} cached predictions available for rescoring; only cache misses or incompatible configurations run detection"
                )
            else:
                print(
                    f"  {method}: cache reads disabled; computing {count['total']} tracks"
                )
        rows, errors, skipped = benchmarker.run(
            jobs,
            workers=self.config.workers,
            verbose=False,
            progress=True,
            compact_progress=True,
        )
        for method, reason in sorted(skipped.items()):
            print(f"  skipped {method}: {reason.splitlines()[0]}")
        for method, dataset, track_id, _ in errors:
            print(f"  error {method} / {dataset} / {track_id}")
        return rows

    def run_suite(
        self,
        name: str,
        datasets: tuple[str, ...],
        max_tracks: int | None,
        methods: Collection[str] | None = None,
        force: bool = False,
        cache_variant: str | None = None,
        degradation_summary: bool = False,
        **option_overrides: Any,
    ) -> SuiteResult:
        """Run one suite, optionally forcing only selected methods.

        ``methods`` is a strict force-rerun filter, not a display filter. Cached
        rows for unselected configured methods remain in the saved and displayed
        result, but absent unselected methods are not computed. Omitting it
        preserves the historical ``force=True`` behavior and recomputes every
        configured method.
        """
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        all_methods = self.methods
        run_dir = self.runs_root / name
        if cache_variant:
            run_dir /= cache_variant
        path = run_dir / "rows.csv"
        force = bool(force or self.config.force_rerun)
        if methods is not None and (not force):
            raise ValueError("methods is only meaningful with force=True")
        forced_methods = self._resolve_methods(methods, all_methods)
        cached_rows = self._read_run(path)
        stale_methods = self._stale_cached_methods(cached_rows, all_methods)
        unscored_methods = (
            []
            if cached_rows is None
            else [
                method
                for method, group in cached_rows.groupby("model", sort=False)
                if method in all_methods
                and (not PitchBenchmarker.current_scores(group))
            ]
        )
        if (
            not force
            and cached_rows is not None
            and (not stale_methods)
            and (not unscored_methods)
        ):
            rows = cached_rows
            print("Reusing completed run:", path.parent)
        else:
            run_methods = (
                forced_methods
                if force
                else (
                    list(dict.fromkeys(stale_methods + unscored_methods))
                    if cached_rows is not None
                    else all_methods
                )
            )
            if force and set(run_methods) != set(all_methods):
                print(f"Running: {name} (forcing {', '.join(run_methods)})")
            elif stale_methods:
                print(f"Running: {name} (refreshing stale {', '.join(stale_methods)})")
            elif unscored_methods:
                print(f"Rescoring saved predictions for frame-weighted metrics: {name}")
            elif cached_rows is None and (not force):
                print(
                    f"No summary at {path}; rebuilding scores from prediction caches."
                )
            else:
                print("Running:", name)
            scoring_only = [
                m
                for m in unscored_methods
                if m not in run_methods or (not force and m not in stale_methods)
            ]
            run_methods = [m for m in run_methods if m not in scoring_only]
            rescored = (
                self._run_methods(
                    scoring_only,
                    datasets,
                    max_tracks,
                    force_reanalysis=False,
                    **option_overrides,
                )
                if scoring_only
                else pd.DataFrame()
            )
            fresh = (
                self._run_methods(
                    run_methods,
                    datasets,
                    max_tracks,
                    force_reanalysis=force,
                    **option_overrides,
                )
                if run_methods
                else pd.DataFrame()
            )
            fresh = pd.concat([fresh, rescored], ignore_index=True)
            rows = self._merge_method_rows(
                cached_rows, fresh, run_methods + scoring_only, all_methods
            )
            if rows.empty:
                print(f"{name} produced no rows; see the errors above.")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                rows.to_csv(path, index=False)
                print("Reports:", path.parent)
        reporter = PitchBenchmarker(self.options())
        summary = (
            reporter.summarize_degradation(
                rows, all_methods, self.config.degradation_snrs
            )
            if degradation_summary
            else reporter.summarize(rows, all_methods)
        )
        display(summary)
        return PitchNotebook.SuiteResult(
            name=name, path=path, rows=rows, summary=summary
        )

    def run_preliminary(
        self,
        force: bool = False,
        methods: Collection[str] | None = None,
        cache_methods: Collection[str] | None = None,
    ) -> SuiteResult:
        """Two seeded original Bach10 recordings per instrument; separate synth caches."""
        return self.run_suite(
            "preliminary_bach10_original",
            datasets=(self.PRELIMINARY_CORPUS,),
            max_tracks=None,
            per_stratum=None,
            per_instrument=self.config.preliminary_tracks_per_instrument,
            cache_variant=self.preliminary_cache_variant(methods=cache_methods),
            force=force,
            methods=methods,
        )

    def run_preliminary_coco(
        self,
        force: bool = False,
        methods: Collection[str] | None = None,
        cache_methods: Collection[str] | None = None,
    ) -> SuiteResult:
        return self.run_suite(
            "preliminary_coco",
            datasets=(CocoChorales.name,),
            max_tracks=None,
            per_stratum=None,
            per_instrument=self.config.preliminary_tracks_per_instrument,
            materialize=True,
            cache_variant=self.preliminary_cache_variant(methods=cache_methods),
            force=force,
            methods=methods,
        )

    def run_preliminary_urmp(
        self,
        force: bool = False,
        methods: Collection[str] | None = None,
        cache_methods: Collection[str] | None = None,
    ) -> SuiteResult:
        """Two seeded real URMP stems for every represented instrument."""
        return self.run_suite(
            "preliminary_urmp",
            datasets=(self.URMP_CORPUS,),
            max_tracks=None,
            per_stratum=None,
            per_instrument=self.config.preliminary_tracks_per_instrument,
            cache_variant=self.preliminary_cache_variant(
                self.ATTUNE_CACHE_VARIANT, methods=cache_methods
            ),
            force=force,
            methods=methods,
        )

    def run_streaming_comparison(
        self,
        instruments: tuple[str, ...] = ("flute", "violin", "cello", "trumpet"),
        tracks_per_instrument: int | None = 1,
        dataset: str = URMP_CORPUS,
        max_seconds: float = math.inf,
        force: bool = False,
        methods: Collection[str] | None = None,
    ) -> SuiteResult:
        """Low-latency replay of Attune, frame-restarted pYIN, and peers.

        Full audio is replayed by default; set ``max_seconds`` to cap each track.

        Attune and Praat receive an exactly matched app frame. ``pyin_framewise``
        calls the local pYIN port once per left-aligned frame but restarts its
        Viterbi state, so it is explicitly distinct from the full-track
        ``pyin`` row. The neural methods use short model-native center
        frames with 32--200 ms of look-ahead; no method reruns over a 1--2
        second bounded history. Attune uses ordinary pYIN observations with the
        shipped causal confidence/running-peak RMS gate, while every method
        receives the same score-derived pitch range.
        The default four instruments keep this a quick design check. Pass
        ``instruments=()`` to sample one stem from every represented URMP
        instrument.
        """
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        from benchmarks.modules.pitch.PitchCache import PitchCache

        all_methods = list(PitchBenchmarker.PitchStreaming.METHODS)
        options = self.options(
            datasets=(dataset,),
            materialize=dataset == CocoChorales.name,
            instruments=instruments,
            per_instrument=tracks_per_instrument,
            max_tracks=None,
        )
        instrument_tag = "all" if not instruments else "-".join(instruments)
        variant = f"{PitchBenchmarker.PitchStreaming.VERSION}__methods_{'+'.join(all_methods)}__instruments_{instrument_tag}__per_instrument_{tracks_per_instrument}__seed_{self.config.seed}__max_{max_seconds:g}s__unv_{PYIN_DEFAULT_UNV_THRESH:g}__dynamic_floor_{PYIN_DEFAULT_MIN_VOLUME:g}".replace(
            ".", "p"
        )
        path = self.runs_root / f"streaming_{dataset}" / variant / "rows.csv"
        force = bool(force or self.config.force_rerun)
        if methods is not None and (not force):
            raise ValueError("methods is only meaningful with force=True")
        forced_methods = self._resolve_methods(methods, all_methods)
        cached_rows = self._read_run(path)
        stale_scores = (
            []
            if cached_rows is None
            else [
                method
                for method, group in cached_rows.groupby("model", sort=False)
                if not PitchBenchmarker.current_scores(group)
                or not all(
                    (
                        PitchCache.has_latency_samples(row)
                        for row in group.to_dict("records")
                    )
                )
            ]
        )
        if not force and cached_rows is not None and (not stale_scores):
            rows = cached_rows
            print("Reusing completed run:", path.parent)
        else:
            benchmarker = PitchBenchmarker(options)
            tracks = benchmarker.tracks()
            examples = [
                benchmarker.dataset(track.dataset).example(track) for track in tracks
            ]
            run_methods = list(
                dict.fromkeys(
                    (
                        forced_methods
                        if force
                        else stale_scores if cached_rows is not None else all_methods
                    )
                    + stale_scores
                )
            )
            print(
                "Running low-latency live replay:",
                f"{len(examples)} excerpt(s), {', '.join(run_methods)}",
            )
            runner = PitchBenchmarker.PitchStreaming(
                options,
                PitchBenchmarker.StreamingConfig(
                    workers=self.config.streaming_workers, max_seconds=max_seconds
                ),
            )
            fresh = runner.run(
                examples,
                methods=run_methods,
                checkpoint_dir=path.parent / "checkpoints",
                force=force,
            )
            rows = self._merge_method_rows(cached_rows, fresh, run_methods, all_methods)
            present = set(rows["model"]) if "model" in rows.columns else set()
            missing = [
                method
                for method in all_methods
                if method not in present and method not in run_methods
            ]
            if missing:
                print("Filling missing cached methods:", ", ".join(missing))
                recovered = runner.run(
                    examples,
                    methods=missing,
                    checkpoint_dir=path.parent / "checkpoints",
                )
                rows = self._merge_method_rows(rows, recovered, missing, all_methods)
            if not rows.empty:
                path.parent.mkdir(parents=True, exist_ok=True)
                rows.to_csv(path, index=False)
                print("Report:", path)
        reporter = PitchBenchmarker(options)
        summary = reporter.summarize(rows, all_methods)
        display(summary)
        shown = reporter.display(rows)
        diagnostic_columns = [
            "Transition VFA",
            "Deep-rest VFA",
            "Median Onset Acquisition (ms)",
            "Median Release Hangover (ms)",
            "P95 False-alarm Burst (ms)",
            "Median Output Latency (ms)",
            "P95 Output Latency (ms)",
            "Deadline Miss Rate",
        ]
        diagnostic_columns = [
            column for column in diagnostic_columns if column in shown.columns
        ]
        diagnostics = shown.set_index(["model", "Track ID"])[diagnostic_columns]
        display(diagnostics)
        matched_geometry_columns = [
            "Frame Size (samples)",
            "Integration Size (samples)",
            "Hop Size (samples)",
            "Sample Rate (Hz)",
        ]
        for _track_id, track_rows in shown.groupby("Track ID", sort=False):
            matched = track_rows[
                track_rows["model"].isin(
                    PitchBenchmarker.PitchStreaming.MATCHED_METHODS
                )
            ]
            if any(
                (matched[column].nunique() != 1 for column in matched_geometry_columns)
            ):
                raise AssertionError("pYIN and Praat did not receive matched frames")
        geometry_columns = [
            "Stream Adapter",
            "Frame Size (samples)",
            "Integration Size (samples)",
            "Hop Size (samples)",
            "Model Hop Size (samples)",
            "Sample Rate (Hz)",
            "Update Interval (ms)",
            "Algorithmic Look-ahead (ms)",
        ]
        geometry_columns = [
            column for column in geometry_columns if column in shown.columns
        ]
        geometry = (
            shown[["model", "Track ID", "Instrument", *geometry_columns]]
            .drop_duplicates()
            .set_index(["model", "Track ID"])
        )
        display(geometry)
        return PitchNotebook.SuiteResult(
            name=f"streaming_{dataset}", path=path, rows=rows, summary=summary
        )

    def show_pooled_streaming_latency(self, *results: SuiteResult) -> pd.DataFrame:
        """Pool individual output updates across the selected recordings, by method."""
        from benchmarks.modules.pitch.PitchCache import PitchCache

        rows = pd.concat([result.rows for result in results], ignore_index=True)
        if rows.duplicated(["dataset", "track_id", "model"]).any():
            raise ValueError("Duplicate recordings in pooled streaming latency")
        pooled = []
        for method, group in rows.groupby("model", sort=False):
            samples = np.concatenate(
                [
                    PitchCache.read_latency_samples(row)
                    for row in group.to_dict("records")
                ]
            )
            pooled.append(
                {
                    "model": method,
                    "recordings": len(group),
                    "output_updates": samples.size,
                    "pooled_p95_output_latency_ms": (
                        float(np.percentile(samples, 95))
                        if samples.size
                        else float("nan")
                    ),
                }
            )
        table = pd.DataFrame(pooled).set_index("model")
        display(table)
        return table

    def run_preliminary_degraded_coco(
        self,
        force: bool = False,
        methods: Collection[str] | None = None,
        cache_methods: Collection[str] | None = None,
    ) -> SuiteResult:
        """The seeded per-instrument Coco draw at every configured SNR."""
        result = self.run_suite(
            "preliminary_degraded_coco",
            datasets=(CocoChorales.name,),
            max_tracks=None,
            per_stratum=None,
            per_instrument=self.config.preliminary_tracks_per_instrument,
            materialize=True,
            cache_variant=self.degradation_cache_variant(
                max_tracks=None,
                per_instrument=self.config.preliminary_tracks_per_instrument,
                methods=cache_methods,
            ),
            degradation_summary=True,
            noise_snrs=self.config.degradation_snrs,
            force=force,
            methods=methods,
        )
        return result

    def run_all_audio(
        self, force: bool = False, methods: Collection[str] | None = None
    ) -> SuiteResult:
        return self.run_suite(
            "all_audio_original_bach10",
            datasets=self.AUDIO_CORPORA,
            max_tracks=None,
            force=force,
            methods=methods,
        )

    def run_all_urmp(
        self, force: bool = False, methods: Collection[str] | None = None
    ) -> SuiteResult:
        """All isolated real-instrument stems in the local URMP download."""
        return self.run_suite(
            "all_urmp",
            datasets=(self.URMP_CORPUS,),
            max_tracks=None,
            cache_variant=self.ATTUNE_CACHE_VARIANT,
            force=force,
            methods=methods,
        )

    def run_all_coco(
        self, force: bool = False, methods: Collection[str] | None = None
    ) -> SuiteResult:
        return self.run_suite(
            "all_coco",
            datasets=(CocoChorales.name,),
            max_tracks=None,
            force=force,
            methods=methods,
        )

    def run_all_degraded_coco(
        self,
        force: bool = False,
        methods: Collection[str] | None = None,
        cache_methods: Collection[str] | None = None,
    ) -> SuiteResult:
        """Every selected Coco stem at every SNR, through every method."""
        result = self.run_suite(
            "all_degraded_coco",
            datasets=(CocoChorales.name,),
            max_tracks=None,
            cache_variant=self.degradation_cache_variant(
                max_tracks=None, per_instrument=None, methods=cache_methods
            ),
            degradation_summary=True,
            noise_snrs=self.config.degradation_snrs,
            force=force,
            methods=methods,
        )
        return result

    @staticmethod
    def snr_label(snr: float) -> str:
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        return PitchBenchmarker.snr_label(snr)

    def preliminary_cache_variant(
        self, *extra: str, methods: Collection[str] | None = None
    ) -> str:
        """Separate suite rows by the deterministic sampling contract."""
        cache_methods = tuple(methods) if methods is not None else self.methods
        method_tag = "+".join(cache_methods)
        parts = [
            CocoChorales.PER_INSTRUMENT_SELECTION_POLICY,
            f"per_instrument_{self.config.preliminary_tracks_per_instrument}",
            f"seed_{self.config.seed}",
            f"methods_{method_tag}",
            *extra,
        ]
        return "__".join(parts)

    def degradation_cache_variant(
        self,
        max_tracks: int | None,
        per_instrument: int | None,
        methods: Collection[str] | None = None,
    ) -> str:
        """Keep completed rows isolated when methods/SNRs/sampling change."""
        if per_instrument is not None:
            sample = f"{CocoChorales.PER_INSTRUMENT_SELECTION_POLICY}__per_instrument_{per_instrument}"
            strata = "none"
        else:
            sample = "all" if max_tracks is None else str(max_tracks)
            strata = self.config.coco_per_stratum or "all"
        snrs = "-".join(
            (self.snr_label(snr).lower() for snr in self.config.degradation_snrs)
        )
        cache_methods = tuple(methods) if methods is not None else self.methods
        method_tag = "+".join(cache_methods)
        return f"tracks_{sample}__strata_{strata}__seed_{self.config.seed}__snr_{snrs}__methods_{method_tag}"

    def show_preliminary_sampling(self) -> pd.DataFrame:
        """Prove equal quotas, repeatability, and seed-sensitive identities."""
        per_instrument = self.config.preliminary_tracks_per_instrument

        def selections(seed: int) -> dict[str, list[tuple[str, str]]]:
            bach = Bach10(per_instrument=per_instrument, seed=seed).tracks()
            urmp = URMP(per_instrument=per_instrument, seed=seed).tracks()
            coco = CocoChorales(per_instrument=per_instrument, seed=seed)
            records = coco.select_records()
            return {
                "Bach10 original": [
                    (track.track_id, str(track.metadata["instrument"]))
                    for track in bach
                ],
                "URMP": [
                    (track.track_id, str(track.metadata["instrument"]))
                    for track in urmp
                ],
                "CocoChorales": [
                    (record.track_id, record.instrument) for record in records
                ],
            }

        selected = selections(self.config.seed)
        repeated = selections(self.config.seed)
        next_seed = selections(self.config.seed + 1)
        rows: list[dict[str, Any]] = []
        for corpus, items in selected.items():
            counts = Counter((instrument for _, instrument in items))
            if not counts or set(counts.values()) != {per_instrument}:
                raise AssertionError(
                    f"{corpus} preliminary draw is not {per_instrument}/instrument: {dict(sorted(counts.items()))}"
                )
            rows.append(
                {
                    "corpus": corpus,
                    "seed": self.config.seed,
                    "instruments": len(counts),
                    "tracks per instrument": per_instrument,
                    "total tracks": len(items),
                    "same seed repeats": items == repeated[corpus],
                    "next seed changes tracks": {track_id for track_id, _ in items}
                    != {track_id for track_id, _ in next_seed[corpus]},
                }
            )
        frame = pd.DataFrame(rows).set_index("corpus")
        display(frame)
        return frame

    def show_by_degradation(self, result: SuiteResult) -> pd.DataFrame | None:
        """Per-method curves across clean/noisy conditions, in configured order."""
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        rows = result.rows
        if "snr_label" not in rows.columns:
            print(f"{result.name} carries no audio-degradation metadata.")
            return None
        table = PitchBenchmarker(self.options()).summarize_degradation(
            rows, self.methods, self.config.degradation_snrs
        )
        display(table)
        return table

    def show_false_alarm_diagnostics(
        self,
        result: SuiteResult,
        methods: tuple[str, ...] = ("pyin", "attune", "praat"),
    ) -> pd.DataFrame:
        """Locate false alarms relative to reference note boundaries and gaps.

        This reads the estimates already produced by ``result``. It does not
        rerun a detector, which keeps the diagnostic safe to execute while a
        larger URMP suite is in progress.
        """
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        if result.rows.empty or "track_id" not in result.rows:
            raise ValueError(f"{result.name} has no per-track rows")
        track_ids = set(result.rows["track_id"].astype(str))
        dataset = URMP()
        tracks = [track for track in dataset.tracks() if track.track_id in track_ids]
        available = set(result.rows.get("model", pd.Series(dtype=str)).astype(str))
        selected = [method for method in methods if method in available]
        counters: dict[str, Counter] = {method: Counter() for method in selected}
        for track in tracks:
            example = dataset.example(track)
            for method in selected:
                detector = PitchBenchmarker.detector_for(method, self.options())
                if not detector.has_cache(dataset, track):
                    continue
                estimate = detector.estimate(example, use_cache=True)
                ref_v, _, est_v, _ = mir_eval.melody.to_cent_voicing(
                    example.ref_times, example.ref_freqs, estimate.times, estimate.freqs
                )
                reference = np.asarray(ref_v, dtype=bool)
                estimated = np.asarray(est_v, dtype=bool)
                hop = self._median_hop(example.ref_times)
                unvoiced = ~reference
                false_alarm = unvoiced & estimated
                distance = ndimage.distance_transform_edt(unvoiced) * hop
                gap_labels, gap_count = ndimage.label(unvoiced)
                gap_sizes = np.bincount(gap_labels.ravel()) * hop
                gap_duration = (
                    gap_sizes[gap_labels] if gap_count else np.zeros_like(distance)
                )
                counter = counters[method]
                counter["tracks"] += 1
                counter["unvoiced"] += int(np.count_nonzero(unvoiced))
                counter["false_alarm"] += int(np.count_nonzero(false_alarm))
                for milliseconds in (50, 100, 250):
                    counter[f"near_{milliseconds}"] += int(
                        np.count_nonzero(
                            false_alarm & (distance <= milliseconds / 1000.0)
                        )
                    )
                for milliseconds in (250, 500):
                    counter[f"gap_{milliseconds}"] += int(
                        np.count_nonzero(
                            false_alarm & (gap_duration <= milliseconds / 1000.0)
                        )
                    )
                deep = unvoiced & (distance > 0.25)
                counter["deep"] += int(np.count_nonzero(deep))
                counter["deep_false_alarm"] += int(np.count_nonzero(false_alarm & deep))
        rows: list[dict[str, Any]] = []
        for method in selected:
            count = counters[method]
            false_alarms = count["false_alarm"]
            rows.append(
                {
                    "model": method,
                    "tracks": count["tracks"],
                    "false-alarm frames": false_alarms,
                    "strict VFA": self._ratio(false_alarms, count["unvoiced"]),
                    "FA within 50 ms": self._ratio(count["near_50"], false_alarms),
                    "FA within 100 ms": self._ratio(count["near_100"], false_alarms),
                    "FA within 250 ms": self._ratio(count["near_250"], false_alarms),
                    "FA in gaps <=250 ms": self._ratio(count["gap_250"], false_alarms),
                    "FA in gaps <=500 ms": self._ratio(count["gap_500"], false_alarms),
                    "deep-rest VFA": self._ratio(
                        count["deep_false_alarm"], count["deep"]
                    ),
                }
            )
        frame = pd.DataFrame(rows).set_index("model")
        display(frame)
        return frame

    def show_method_comparison(
        self, *results: SuiteResult, methods: Collection[str] | None = None
    ) -> pd.DataFrame:
        """Rank selected cached methods separately within each supplied suite."""
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        selected = tuple(methods or self.methods)
        frames: list[pd.DataFrame] = []
        metric_order = [
            "Overall Accuracy",
            "Voicing False Alarm",
            "Voicing Recall",
            "Raw Pitch Accuracy",
            "Raw Chroma Accuracy",
        ]
        for result in results:
            rows = result.rows.loc[result.rows["model"].isin(selected)]
            if rows.empty:
                continue
            grouped = rows.groupby("model", sort=False)
            table = PitchBenchmarker.pool_scores(rows)[metric_order]
            table.insert(0, "Tracks", grouped["track_id"].nunique())
            table.insert(
                0,
                "OA Rank",
                table["Overall Accuracy"]
                .rank(method="min", ascending=False)
                .astype(int),
            )
            table.insert(0, "Suite", result.name)
            frames.append(table.reset_index(names="Method"))
        if not frames:
            raise ValueError("none of the requested methods is present in the suites")
        comparison = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["Suite", "OA Rank", "Method"], kind="stable")
            .set_index(["Suite", "OA Rank", "Method"])
        )
        display(comparison)
        return comparison

    def show_rpa_decomposition(
        self, *results: SuiteResult, methods: Collection[str] | None = None
    ) -> pd.DataFrame:
        """Split headline RPA loss into missed voicing and wrong-pitch terms.

        For each method, ``1 - RPA`` decomposes exactly as::

            (1 - Voicing Recall) + (Voicing Recall - RPA)

        The first term is reference-voiced audio the method suppressed as
        unvoiced. The second is reference-voiced audio the method emitted with
        an error larger than mir_eval's 50-cent RPA tolerance. ``RPA / VR`` is
        included as an intuitive headline ratio, not as a separate mir_eval
        metric.
        """
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        selected = tuple(methods or self.methods)
        frames: list[pd.DataFrame] = []
        for result in results:
            rows = result.rows.loc[result.rows["model"].isin(selected)]
            if rows.empty:
                continue
            grouped = rows.groupby("model", sort=False)
            table = PitchBenchmarker.pool_scores(rows)[
                ["Raw Pitch Accuracy", "Voicing Recall"]
            ]
            table.insert(0, "Tracks", grouped["track_id"].nunique())
            table["Missed-voicing share"] = 1.0 - table["Voicing Recall"]
            table["Wrong-pitch share"] = (
                table["Voicing Recall"] - table["Raw Pitch Accuracy"]
            )
            table["RPA / Voicing Recall"] = np.divide(
                table["Raw Pitch Accuracy"],
                table["Voicing Recall"],
                out=np.full(len(table), np.nan, dtype=np.float64),
                where=table["Voicing Recall"].to_numpy() > 0.0,
            )
            table.insert(0, "Suite", result.name)
            frames.append(table.reset_index(names="Method"))
        if not frames:
            raise ValueError("none of the requested methods is present in the suites")
        decomposition = pd.concat(frames, ignore_index=True).set_index(
            ["Suite", "Method"]
        )
        display(decomposition)
        return decomposition

    def show_attune_live_voicing_thresholds(
        self,
        result: SuiteResult,
        thresholds: tuple[float, ...] = (0.55, 0.65, 0.75, 0.85, 0.9),
    ) -> pd.DataFrame:
        """Re-score cached real-time evidence under stricter voicing gates."""
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        from benchmarks.modules.pitch.PitchCache import PitchCache

        track_ids = set(result.rows["track_id"].astype(str))
        dataset = URMP()
        tracks = [track for track in dataset.tracks() if track.track_id in track_ids]
        rows: list[dict[str, float]] = []
        detector = Attune()
        for track in tracks:
            example = dataset.example(track)
            base_config = detector.config_for(example.fmin, example.fmax)
            hit = PitchCache(dataset.pitch_cache_path(track)).read(
                PitchCache.RAW, base_config
            )
            if hit is None:
                continue
            pitch_data, _ = hit
            for threshold in thresholds:
                config = detector.config_for(
                    example.fmin, example.fmax, unv_thresh=float(threshold)
                )
                est_times, est_freqs = detector.melody(pitch_data, config)
                metrics = PitchBenchmarker.score_frames(
                    example.ref_times, example.ref_freqs, est_times, est_freqs
                )
                rows.append({"unvoiced threshold": float(threshold), **metrics})
        if not rows:
            raise ValueError("no current Attune real-time caches matched this result")
        frame = PitchBenchmarker.pool_scores(pd.DataFrame(rows), "unvoiced threshold")
        display(frame)
        return frame

    def show_attune_live_volume_floors(
        self,
        result: SuiteResult,
        floors_dbfs: tuple[float | None, ...] = (
            None,
            -72.0,
            -66.0,
            -60.0,
            -57.0,
            -54.0,
            -51.0,
            -48.0,
            -45.0,
            -42.0,
        ),
    ) -> pd.DataFrame:
        """Re-score cached raw pYIN behind an absolute centered-RMS floor.

        ``Pitch.volume`` is the integration window's centered RMS before pYIN
        peak-normalizes the frame. Converting it to dBFS makes this gate causal:
        unlike a fraction of the eventual track maximum, it requires no future
        samples. ``None`` retains the ungated stage-1 baseline.
        """
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        from benchmarks.modules.pitch.PitchCache import PitchCache

        track_ids = set(result.rows["track_id"].astype(str))
        dataset = URMP()
        tracks = [track for track in dataset.tracks() if track.track_id in track_ids]
        rows: list[dict[str, float | str]] = []
        detector = Attune()
        for track in tracks:
            example = dataset.example(track)
            config = detector.config_for(example.fmin, example.fmax)
            hit = PitchCache(dataset.pitch_cache_path(track)).read(
                PitchCache.RAW, config
            )
            if hit is None:
                continue
            pitch_data, _ = hit
            est_times, est_freqs = detector.melody(pitch_data, config)
            volumes = self._pitch_volumes(pitch_data)
            if volumes.size != est_freqs.size or not volumes.size:
                continue
            dbfs = 20.0 * np.log10(np.maximum(volumes, np.finfo(np.float64).tiny))
            for floor in floors_dbfs:
                gated_freqs = (
                    est_freqs
                    if floor is None
                    else np.where(dbfs >= float(floor), est_freqs, 0.0)
                )
                metrics = PitchBenchmarker.score_frames(
                    example.ref_times, example.ref_freqs, est_times, gated_freqs
                )
                rows.append(
                    {
                        "Volume Floor (dBFS)": (
                            "Off" if floor is None else f"{float(floor):g}"
                        ),
                        "track_id": track.track_id,
                        **metrics,
                    }
                )
        if not rows:
            raise ValueError("no current Attune real-time caches matched this result")
        source = pd.DataFrame(rows)
        grouped = source.groupby("Volume Floor (dBFS)", sort=False)
        frame = PitchBenchmarker.pool_scores(source, "Volume Floor (dBFS)")
        frame.insert(0, "Tracks", grouped["track_id"].nunique())
        display(frame)
        return frame

    @staticmethod
    def _pitch_volumes(pitch_data: Any) -> np.ndarray:
        """Frame RMS in the same order ``Attune.melody`` emits frames."""
        return np.asarray(
            [
                float(getattr(pitch, "volume", 0.0) or 0.0)
                for pitch in pitch_data.data
                if pitch is not None
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _median_hop(times: np.ndarray) -> float:
        differences = np.diff(np.asarray(times, dtype=np.float64))
        differences = differences[np.isfinite(differences) & (differences > 0)]
        return float(np.median(differences)) if differences.size else 0.01

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    def show_by_stratum(self, result: SuiteResult) -> pd.DataFrame | None:
        """Annotated corpus rows broken out per (ensemble, instrument)."""
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        table = PitchBenchmarker.display(result.rows)
        if not set(self.STRATUM_KEYS).issubset(table.columns):
            print(f"{result.name} carries no ensemble/instrument metadata.")
            return None
        grouped = table.groupby(["model", *self.STRATUM_KEYS])
        summary = PitchBenchmarker.pool_scores(table, ["model", *self.STRATUM_KEYS])
        summary.insert(0, "n", grouped.size())
        display(summary)
        return summary

    def show_leaderboard(self, *results: SuiteResult) -> pd.DataFrame:
        """One row per (suite, method), ranked by overall accuracy."""
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        frames = [
            PitchBenchmarker(self.options())
            .summarize(result.rows, self.methods)
            .assign(suite=result.name)
            .set_index("suite", append=True)
            for result in results
        ]
        table = pd.concat(frames).sort_values("Overall Accuracy", ascending=False)
        display(table)
        return table

    def run_paper_evaluation(self, mode: str) -> tuple[SuiteResult, ...]:
        """Recommended clean 449-track selection, shared by offline/online runs.

        Explicitly invoked by an opt-in notebook cell; never starts implicitly
        during reporting or paired tests. Uses estimate/checkpoint caches.
        """
        if mode not in ("Offline", "Streaming"):
            raise ValueError("mode must be Offline or Streaming")
        results = []
        for dataset, per_instrument in (
            (Bach10.name, None),
            (URMP.name, None),
            (CocoChorales.name, 20),
        ):
            if mode == "Streaming":
                result = self.run_streaming_comparison(
                    dataset=dataset,
                    instruments=(),
                    tracks_per_instrument=per_instrument,
                )
            else:
                result = self.run_suite(
                    "paper_offline_" + dataset,
                    datasets=(dataset,),
                    max_tracks=None,
                    per_instrument=per_instrument,
                    per_stratum=None,
                    materialize=dataset == CocoChorales.name,
                    cache_variant=f"seed_{self.config.seed}__per_instrument_{per_instrument}",
                )
            results.append(result)
        return tuple(results)

    def show_paired_analysis(
        self,
        *results: SuiteResult,
        mode: str,
        methods: Collection[str],
        n_resamples: int = 9999,
        source_groups: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Independent offline/online test families, plus frame-pooled tables.

        Optional source_groups maps dataset-qualified source-piece IDs to a
        shared ID when the same composition occurs in multiple datasets.
        Reports are written next to the input run manifests, never over them.
        """
        from benchmarks.modules.pitch.PitchBenchmarker import (
            PitchBenchmarker,
            SCORING_VERSION,
        )

        if mode not in ("Offline", "Streaming"):
            raise ValueError("mode must be Offline or Streaming")
        rows = pd.concat([result.rows for result in results], ignore_index=True)
        if "execution_mode" not in rows or not rows.execution_mode.eq(mode).all():
            raise ValueError(f"Expected only {mode} rows with current scoring metadata")
        if "degradation" in rows and (
            not rows.degradation.fillna("clean").isin(("clean", "none")).all()
        ):
            raise ValueError(
                "Use clean runs for the main paired analysis; analyze noise separately"
            )
        if source_groups:
            rows["source_piece"] = rows["source_piece"].replace(source_groups)
        metrics = (
            ("Overall Accuracy",)
            if mode == "Offline"
            else ("Overall Accuracy", "Voicing Recall", "Voicing False Alarm")
        )
        tests, paired_rows = PitchBenchmarker.paired_comparisons(
            rows,
            methods=methods,
            metrics=metrics,
            n_resamples=n_resamples,
            seed=self.config.seed,
        )
        headline = PitchBenchmarker.pool_scores(paired_rows)
        per_dataset = PitchBenchmarker.pool_scores(paired_rows, ["dataset", "model"])
        coverage = (
            paired_rows.loc[paired_rows.model.eq("attune")]
            .groupby("dataset")
            .agg(
                tracks=("track_id", "nunique"),
                source_pieces=("source_piece", "nunique"),
                frames=("frame_count", "sum"),
                voiced_frames=("voiced_frame_count", "sum"),
                unvoiced_frames=("unvoiced_frame_count", "sum"),
            )
        )
        import hashlib
        import json

        manifest = {
            "mode": mode,
            "methods": list(methods),
            "seed": self.config.seed,
            "resamples": n_resamples,
            "scoring_version": SCORING_VERSION,
            "source_groups": source_groups or {},
            "inputs": [str(result.path) for result in results],
        }
        signature = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()[:12]
        folder = self.runs_root / "paired_analysis" / mode.lower() / signature
        folder.mkdir(parents=True, exist_ok=True)
        if mode == "Streaming" and "latency_samples_path" in paired_rows:
            latency = self.show_pooled_streaming_latency(
                PitchNotebook.SuiteResult(
                    "paired_streaming", folder, paired_rows, headline
                )
            )
            latency.to_csv(folder / "pooled_latency.csv")
        elif mode == "Streaming":
            print(
                "Pooled latency unavailable: legacy rows lack per-update samples. Rerun streaming with v8 to collect them."
            )
        tests.to_csv(folder / "paired_tests.csv", index=False)
        headline.to_csv(folder / "pooled_metrics.csv")
        per_dataset.to_csv(folder / "dataset_metrics.csv")
        coverage.to_csv(folder / "coverage.csv")
        paired_rows.to_csv(folder / "paired_rows.csv", index=False)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"{mode}: frame-pooled scores on the common completed recording set; Holm correction within this mode. Confidence intervals are unadjusted."
        )
        display(coverage)
        display(headline.reindex(list(methods)))
        display(tests)
        print("Reports:", folder)
        return tests
