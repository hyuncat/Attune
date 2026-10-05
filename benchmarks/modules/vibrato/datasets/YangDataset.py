"""Shared cases from Yang et al.'s parameter-annotated recordings."""

from __future__ import annotations

import csv
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app_logic.user.ds.AudioData import AudioData
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoExample


# This released file alone omits the frequency column. The performance is The
# Moon Reflected on the Second Spring on the lower G-D Erquan setup; 180-2000 Hz
# safely contains an independent broad-range audit of the bundled recording and
# the instrument's shifted register without returning to the generic 80-2000 Hz
# guess previously applied to every Yang file.
YANG_PITCH_RANGE_OVERRIDES_HZ = {
    "Huangjiangqin-1": (180.0, 2000.0),
}


class YangDataset:
    """Shared real-audio corpus from Yang et al.'s released annotations."""

    @dataclass(frozen=True)
    class Recording:
        audio_path: Path
        area_path: Path
        extrema_path: Path
        instrument: str
        performer: str
        pitch_range_override_hz: tuple[float, float] | None = None
        pitch_range_source: str = "yang_area_annotation_frequency_column"

        @property
        def recording_id(self) -> str:
            return self.audio_path.stem

        @property
        def pitch_range_hz(self) -> tuple[float, float]:
            if self.pitch_range_override_hz is not None:
                return self.pitch_range_override_hz
            return YangDataset._annotation_pitch_range(self.area_path)

    @dataclass(frozen=True)
    class _PitchJob:
        index: int
        recording: YangDataset.Recording
        cache_root: str
        smooth_pitch: bool
        force: bool

    @staticmethod
    def _numeric_column(path: str | Path, index: int) -> np.ndarray:
        """Read one numeric column while tolerating the release's ragged CSVs."""
        values: list[float] = []
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) <= index or not row[index].strip():
                    raise ValueError(
                        f"missing column {index + 1} at {path}:{line_number}"
                    )
                values.append(float(row[index]))
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _annotation_pitch_range(path: str | Path) -> tuple[float, float]:
        """Read the released vibrato-area frequency annotations in column two."""
        frequencies: list[float] = []
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                # The parameter-capable area format is
                # start, frequency, duration, label. Huangjiangqin-1 is a released
                # three-column exception with no frequency metadata.
                if len(row) < 4 or not row[1].strip():
                    continue
                frequency = float(row[1])
                if np.isfinite(frequency) and frequency > 0.0:
                    frequencies.append(frequency)
        if not frequencies:
            raise ValueError(f"no annotated pitch frequencies in {path}")
        return float(min(frequencies)), float(max(frequencies))

    @classmethod
    def discover(
        cls,
        dataset_root: str | Path,
    ) -> list[YangDataset.Recording]:
        """Find recordings that have both area and half-cycle annotations."""
        root = Path(dataset_root)
        parameter_root = root / "Areas_and_parameters"
        if not parameter_root.is_dir() and root.name == "Areas_and_parameters":
            parameter_root = root
        if not parameter_root.is_dir():
            raise FileNotFoundError(
                f"Yang Areas_and_parameters directory was not found under {root}"
            )

        recordings: list[YangDataset.Recording] = []
        suffix = "-Annotation-Stat.csv"
        for extrema_path in sorted(parameter_root.rglob(f"*{suffix}")):
            stem = extrema_path.name.removesuffix(suffix)
            audio_path = extrema_path.with_name(f"{stem}.wav")
            area_candidates = (
                extrema_path.with_name(f"{stem}-Annotation-new.csv"),
                extrema_path.with_name(f"{stem}-Annotation.csv"),
            )
            area_path = next((path for path in area_candidates if path.is_file()), None)
            if not audio_path.is_file() or area_path is None:
                continue
            relative = extrema_path.relative_to(parameter_root)
            instrument = relative.parts[0] if len(relative.parts) >= 1 else "unknown"
            performer = relative.parts[1] if len(relative.parts) >= 2 else "unknown"
            try:
                pitch_range = cls._annotation_pitch_range(area_path)
                pitch_range_source = "yang_area_annotation_frequency_column"
            except ValueError:
                pitch_range = YANG_PITCH_RANGE_OVERRIDES_HZ.get(stem)
                if pitch_range is None:
                    raise
                pitch_range_source = "yang_recording_erquan_safe_override"
                # This is a frozen, recording-specific dataset adapter rather than
                # a runtime failure. The source and exact range are retained in
                # each case/run manifest, so repeating it on stderr only clutters
                # notebook output.
            recording = cls.Recording(
                audio_path=audio_path,
                area_path=area_path,
                extrema_path=extrema_path,
                instrument=instrument,
                performer=performer,
                pitch_range_override_hz=pitch_range,
                pitch_range_source=pitch_range_source,
            )
            recordings.append(recording)
        if not recordings:
            raise FileNotFoundError(
                f"no parameter-capable Yang recordings were found under {parameter_root}"
            )
        return recordings

    @staticmethod
    def _balanced_recording_limit(
        recordings: Sequence[YangDataset.Recording],
        limit: int | None,
    ) -> list[YangDataset.Recording]:
        """Round-robin performers so a two-file smoke run covers erhu and violin."""
        if limit is None or limit >= len(recordings):
            return list(recordings)
        if limit <= 0:
            raise ValueError("recording limit must be positive or None")
        groups: dict[tuple[str, str], list[YangDataset.Recording]] = {}
        for recording in recordings:
            groups.setdefault(
                (recording.instrument, recording.performer),
                [],
            ).append(recording)
        selected: list[YangDataset.Recording] = []
        depth = 0
        ordered_groups = [groups[key] for key in sorted(groups)]
        while len(selected) < limit:
            added = False
            for group in ordered_groups:
                if depth < len(group):
                    selected.append(group[depth])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            depth += 1
        return selected

    @staticmethod
    def _pitch_arrays(pitch_data, config) -> tuple[np.ndarray, np.ndarray]:
        pitches = pitch_data.data[: pitch_data.frames_available()]
        times = np.asarray(
            [
                pitch.time if pitch is not None else pitch_data._frame_time(index)
                for index, pitch in enumerate(pitches)
            ],
            dtype=np.float64,
        )
        values = np.asarray(
            [
                (
                    pitch.value
                    if (
                        pitch is not None
                        and pitch.value != -1
                        and pitch.unvoiced_prob < config.unv_thresh
                    )
                    else np.nan
                )
                for pitch in pitches
            ],
            dtype=np.float64,
        )
        return times, values

    @classmethod
    def _examples_from_annotations(
        cls,
        recording: YangDataset.Recording,
        times: np.ndarray,
        pitch_midi: np.ndarray,
        *,
        pitch_stage: str,
        pitch_metadata: dict[str, Any] | None = None,
    ) -> list[VibratoExample]:
        """Build shared-contour cases from manual areas and half-cycle marks."""
        times = np.asarray(times, dtype=np.float64)
        pitch_midi = np.asarray(pitch_midi, dtype=np.float64)
        if len(times) != len(pitch_midi) or len(times) < 2:
            raise ValueError("Yang pitch times and values must share length >= 2")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("Yang pitch times must be strictly increasing")

        starts = cls._numeric_column(recording.area_path, 0)
        durations = cls._numeric_column(recording.area_path, 2)
        extrema = cls._numeric_column(recording.extrema_path, 0)
        finite = np.isfinite(pitch_midi)
        if int(np.sum(finite)) < 2:
            raise ValueError(
                f"fewer than two voiced pitch frames in {recording.audio_path}"
            )
        filled_pitch = np.interp(times, times[finite], pitch_midi[finite])

        truths: list[dict[str, float | int | np.ndarray]] = []
        for area_index, (start, duration) in enumerate(zip(starts, durations)):
            end = float(start + duration)
            target = (times >= start) & (times < end)
            marks = extrema[(extrema >= start) & (extrema <= end)]
            if int(np.sum(target)) < 2 or len(marks) < 2:
                continue
            half_periods = np.diff(marks)
            marker_pitch = np.interp(marks, times, filled_pitch)
            valid_half_cycles = (
                np.isfinite(half_periods)
                & (half_periods > 0.0)
                & np.isfinite(marker_pitch[:-1])
                & np.isfinite(marker_pitch[1:])
            )
            if not np.any(valid_half_cycles):
                continue
            half_cycle_rates = 0.5 / half_periods[valid_half_cycles]
            half_cycle_extents_cents = 100.0 * np.abs(
                np.diff(marker_pitch)[valid_half_cycles]
            )
            center = float(np.median(filled_pitch[target]))
            truths.append(
                {
                    "area_index": area_index,
                    "start": float(start),
                    "end": end,
                    "target": target,
                    "rate_hz": float(np.mean(half_cycle_rates)),
                    # VibratoExample stores full peak-to-peak width in cents.  Yang's
                    # manual peak-to-trough difference is the same full-extent unit.
                    "width_cents": float(np.mean(half_cycle_extents_cents)),
                    "center_midi": center,
                    "half_cycles": int(np.sum(valid_half_cycles)),
                }
            )
        if not truths:
            raise ValueError(
                f"no usable parameter annotations in {recording.extrema_path}"
            )

        n = len(times)
        truth_rate = np.zeros(n, dtype=np.float64)
        truth_width = np.zeros(n, dtype=np.float64)
        truth_center = np.full(n, np.nan, dtype=np.float64)
        truth_vibrato = np.zeros(n, dtype=np.bool_)
        for truth in truths:
            target = np.asarray(truth["target"], dtype=np.bool_)
            truth_rate[target] = float(truth["rate_hz"])
            truth_width[target] = float(truth["width_cents"])
            truth_center[target] = float(truth["center_midi"])
            truth_vibrato[target] = True

        note_bounds = [
            (
                float(truth["start"]),
                float(truth["end"]),
                float(truth["center_midi"]),
            )
            for truth in truths
        ]
        examples: list[VibratoExample] = []
        for truth in truths:
            area_index = int(truth["area_index"])
            examples.append(
                VibratoExample(
                    case_id=f"{recording.recording_id}__vibrato_{area_index:03d}",
                    scenario=f"{recording.instrument.lower()}_annotated_vibrato",
                    split="yang_preliminary",
                    times=times,
                    pitch_midi=pitch_midi,
                    center_midi=truth_center,
                    rate_hz=truth_rate,
                    width_cents=truth_width,
                    is_vibrato=truth_vibrato,
                    evaluation_mask=np.asarray(truth["target"], dtype=np.bool_),
                    audio_path=str(recording.audio_path),
                    metadata={
                        **(pitch_metadata or {}),
                        "family": "yang_moon_reflected_parameters",
                        "analysis_group": recording.recording_id,
                        "analysis_note_bounds": note_bounds,
                        "continuous_context": True,
                        "instrument": recording.instrument,
                        "performer": recording.performer,
                        "recording": recording.recording_id,
                        "audio_path": str(recording.audio_path),
                        "area_annotation_path": str(recording.area_path),
                        "extrema_annotation_path": str(recording.extrema_path),
                        "pitch_stage": pitch_stage,
                        "half_cycles": int(truth["half_cycles"]),
                        "target_start_time": float(truth["start"]),
                        "target_end_time": float(truth["end"]),
                        "parameter_truth": "manual_extrema_times_plus_common_pyin_pitch",
                        "note_boundary_source": "annotated_vibrato_spans_pseudo_notes",
                    },
                )
            )
        return examples

    @staticmethod
    def _build_recording(
        job: YangDataset._PitchJob,
    ) -> tuple[int, list[VibratoExample]]:
        benchmarker = AttuneRealtime()
        # The released area rows provide the expected frequency for each annotated
        # vibrato span. Establish that range before constructing the detector, just
        # as the pitch benchmark does with its reference-frequency annotations.
        fmin, fmax = job.recording.pitch_range_hz
        config = benchmarker.config_for(fmin, fmax)
        take = benchmarker.recording_for(config)
        take.audio_data = AudioData(
            audio_filepath=str(job.recording.audio_path),
            config=config,
        )
        cache_path = (
            Path(job.cache_root)
            / job.recording.instrument
            / job.recording.performer
            / f"{job.recording.recording_id}.pitch.pkl.xz"
        )
        benchmarker.load_or_detect_pitches(
            take,
            cache_path=cache_path,
            smooth=job.smooth_pitch,
            use_cache=not job.force,
            write_cache=True,
        )
        times, pitch_midi = YangDataset._pitch_arrays(take.pitch_data, config)
        examples = YangDataset._examples_from_annotations(
            job.recording,
            times,
            pitch_midi,
            pitch_stage="attune_adhoc" if job.smooth_pitch else "attune_realtime",
            pitch_metadata={
                "pitch_range_source": job.recording.pitch_range_source,
                "pitch_fmin_hz": float(config.fmin),
                "pitch_fmax_hz": float(config.fmax),
                "yin_integration_size": int(config.w1),
            },
        )
        return job.index, examples

    @classmethod
    def build(
        cls,
        *,
        dataset_root: str | Path,
        cache_root: str | Path,
        max_recordings: int | None = 2,
        smooth_pitch: bool = True,
        force: bool = False,
        workers: int = 1,
    ) -> list[VibratoExample]:
        """Build a preliminary real-audio parameter corpus from Yang's release."""
        if workers <= 0:
            raise ValueError("workers must be positive")
        recordings = cls._balanced_recording_limit(
            cls.discover(dataset_root),
            max_recordings,
        )
        jobs = [
            cls._PitchJob(
                index=index,
                recording=recording,
                cache_root=str(cache_root),
                smooth_pitch=smooth_pitch,
                force=force,
            )
            for index, recording in enumerate(recordings)
        ]
        completed: list[tuple[int, list[VibratoExample]]] = []
        if workers == 1 or len(jobs) == 1:
            completed = [cls._build_recording(job) for job in jobs]
        else:
            try:
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(jobs)),
                    mp_context=context,
                ) as pool:
                    futures = [pool.submit(cls._build_recording, job) for job in jobs]
                    for future in as_completed(futures):
                        completed.append(future.result())
            except (OSError, PermissionError) as error:
                warnings.warn(
                    f"parallel Yang pitch preparation is unavailable ({error}); "
                    "falling back to sequential processing",
                    RuntimeWarning,
                    stacklevel=2,
                )
                completed = [cls._build_recording(job) for job in jobs]
        return [
            example
            for _, examples in sorted(completed, key=lambda item: item[0])
            for example in examples
        ]
