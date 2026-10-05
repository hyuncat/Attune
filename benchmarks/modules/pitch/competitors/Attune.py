"""Benchmark adapter for Attune's promoted ordinary-candidate pYIN+ method.

An adapter, not an implementation: :meth:`detect_stages` builds a real
``Recording`` and calls ``algorithms/PitchDetector.py`` and
``algorithms/PitchSmoother.py`` — the exact objects the app runs — so a benchmark
row can never drift from what Attune ships. Everything else here is timing,
range plumbing, and the melody conversion mir_eval wants.

The detector computes ordinary pYIN candidates once. The benchmark reports the
completed-audio global-confidence result, while the raw cache stage retains the
causal confidence/running-peak path used by Practice. Both modes share the same
frontend pass.
"""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
import numpy as np
from algorithms.Config import (
    Config,
    PYIN_DEFAULT_MIN_VOLUME,
    PYIN_DEFAULT_UNV_THRESH,
    PYIN_POSTHOC_MIN_VOLUME,
    PYIN_POSTHOC_UNV_THRESH,
)
from algorithms.PitchDetector import PitchDetector
from algorithms.PitchSmoother import PitchSmoother, VoicingSmoother
from app_logic.midi.ScoreData import ScoreData
from app_logic.NoteData import NoteData
from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData
from app_logic.user.ds.Recording import Recording
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class Attune(PitchDetectorBase):
    """Attune pYIN+: causal live observations, completed-audio smoothing."""

    name = "attune"
    description = "Attune pYIN+ promoted production pipeline"
    input_sr = None
    smooth: ClassVar[bool] = True
    DEFAULT_CONFIG: ClassVar[Config] = Config(
        sr=44100,
        w1=1024 * 4,
        h1=128,
        fmin=196.0,
        fmax=3000.0,
        tuning=440.0,
        unv_thresh=PYIN_DEFAULT_UNV_THRESH,
        min_volume=PYIN_DEFAULT_MIN_VOLUME,
        posthoc_unv_thresh=PYIN_POSTHOC_UNV_THRESH,
        posthoc_min_volume=PYIN_POSTHOC_MIN_VOLUME,
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config_overrides: dict[str, Any] = {}
        self.algorithm_verbose = False

    @property
    def stage(self) -> str:
        return PitchCache.stage_name(self.smooth)

    def config_for(self, fmin: float, fmax: float, **overrides: Any) -> Config:
        return replace(
            self.DEFAULT_CONFIG,
            **{
                **self.config_overrides,
                **overrides,
                "fmin": float(fmin),
                "fmax": float(fmax),
            },
        )

    @staticmethod
    def range_from_midi(
        midi_numbers: Iterable[float], tuning: float = 440.0
    ) -> tuple[float, float]:
        numbers = np.asarray(list(midi_numbers), dtype=np.float64)
        if numbers.size == 0:
            return (196.0, 3000.0)
        to_hz = lambda midi: tuning * 2 ** ((midi - 69) / 12.0)
        return (to_hz(numbers.min()), to_hz(numbers.max()))

    def recording_for(
        self,
        config: Config,
        score_notes: NoteData | None = None,
        score_data: ScoreData | None = None,
    ) -> Recording:
        """A Recording wired to the same detector/smoother instances the app uses."""
        recording = (
            Recording(score_data=score_data, config=config)
            if score_data is not None
            else Recording(config=config)
        )
        recording.pitch_detector = PitchDetector(recording=recording)
        recording.pitch_smoother = PitchSmoother(recording=recording)
        recording.voicing_smoother = VoicingSmoother(recording=recording)
        if score_data is None and score_notes is not None:
            recording.score_data.note_datas = {0: score_notes}
            recording.score_data.active_instrument = 0
            recording.score_data.clip = None
            recording.active_instrument = 0
        return recording

    def detect_stages(
        self, recording: Recording, smooth: bool = True
    ) -> PitchCache.Stages:
        """Run the production detector (and smoother), timing each separately."""
        audio = recording.audio_data.read_all()
        verbose = self.algorithm_verbose
        raw_pitches, detector_seconds, detector_wall_seconds = self.measure(
            lambda: recording.pitch_detector.detect_pitches(
                audio, show_progress=verbose, verbose=verbose
            )
        )
        origin = recording.audio_data.t_origin
        stages = PitchCache.Stages()
        stages.data[PitchCache.RAW] = self._pitch_data(
            raw_pitches, recording.config, origin
        )
        stages.timing[PitchCache.RAW] = {
            "pitch_cache_version": PitchCache.VERSION,
            "pitch_detector_compute_time": detector_seconds,
            "pitch_smoother_compute_time": 0.0,
            "pitch_compute_time": detector_seconds,
            "wall_pitch_detector_compute_time": detector_wall_seconds,
            "wall_pitch_smoother_compute_time": 0.0,
            "wall_pitch_compute_time": detector_wall_seconds,
            "compute_clock": self.COMPUTE_CLOCK,
        }
        if not smooth:
            return stages

        def smooth_completed_audio():
            tracked = recording.pitch_smoother.smooth(raw_pitches, verbose=verbose)
            return recording.voicing_smoother.smooth(tracked, verbose=verbose)

        smoothed, smoother_seconds, smoother_wall_seconds = self.measure(
            smooth_completed_audio
        )
        stages.data[PitchCache.SMOOTHED] = self._pitch_data(
            smoothed, recording.config, origin
        )
        stages.timing[PitchCache.SMOOTHED] = {
            "pitch_cache_version": PitchCache.VERSION,
            "pitch_smoother_method": getattr(
                recording.pitch_smoother,
                "METHOD_VERSION",
                type(recording.pitch_smoother).__name__,
            ),
            "pitch_detector_compute_time": detector_seconds,
            "pitch_smoother_compute_time": smoother_seconds,
            "pitch_compute_time": detector_seconds + smoother_seconds,
            "wall_pitch_detector_compute_time": detector_wall_seconds,
            "wall_pitch_smoother_compute_time": smoother_wall_seconds,
            "wall_pitch_compute_time": detector_wall_seconds + smoother_wall_seconds,
            "compute_clock": self.COMPUTE_CLOCK,
        }
        return stages

    def load_or_detect_pitches(
        self,
        recording: Recording,
        cache_path: Path | str,
        smooth: bool = True,
        use_cache: bool = True,
        write_cache: bool = True,
    ) -> dict[str, Any]:
        """Fill ``recording.pitch_data`` from cache, detecting only if needed."""
        cache = PitchCache(cache_path)
        stage = PitchCache.stage_name(smooth)
        if use_cache and (hit := cache.read(stage, recording.config)) is not None:
            recording.pitch_data, timing = hit
            return timing
        stages = self.detect_stages(recording, smooth=smooth)
        recording.pitch_data = stages.data[stage]
        if write_cache:
            cache.write(stages)
        return stages.timing[stage]

    def has_pitch_cache(self, cache_path: Path | str, smooth: bool = True) -> bool:
        return PitchCache(cache_path).has(PitchCache.stage_name(smooth))

    def load_pitch_data(
        self, cache_path: Path | str, config: Config, smooth: bool = True
    ) -> tuple[PitchData, dict[str, Any]]:
        stage = PitchCache.stage_name(smooth)
        hit = PitchCache(cache_path).read(stage, config)
        if hit is None:
            raise ValueError(f"pitch cache has no {stage!r} data: {cache_path}")
        return hit

    @staticmethod
    def melody(pitch_data: PitchData, config: Config) -> Melody:
        """Frames as mir_eval's exact pYIN frequency bins and voiced mask."""
        times: list[float] = []
        freqs: list[float] = []
        bins_per_semitone = int(np.ceil(1.0 / PitchSmoother.RESOLUTION))
        base_midi = float(config.freq_to_midi(config.fmin))
        for pitch in pitch_data.data:
            if pitch is None:
                continue
            times.append(pitch.time)
            voiced = pitch.value != -1 and pitch.unvoiced_prob < config.unv_thresh
            if not voiced:
                freqs.append(0.0)
                continue
            pitch_bin = int(
                np.round((float(pitch.value) - base_midi) * bins_per_semitone)
            )
            freqs.append(
                float(config.fmin) * 2.0 ** (pitch_bin / (12 * bins_per_semitone))
            )
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(freqs, dtype=np.float64),
        )

    def cache(self, example: PitchExample) -> PitchCache:
        return PitchCache(example.stage_cache_path)

    def has_cache(self, dataset: Any, track: Any) -> bool:
        return PitchCache(dataset.pitch_cache_path(track)).has_current_timing(
            self.stage, self.COMPUTE_CLOCK
        )

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        """Run the shipped detector on an in-memory segment.

        ``estimate`` remains the cache-aware whole-track adapter. This smaller
        surface is used by bounded-context replay, where every method receives
        the same incrementally revealed audio rather than a dataset path.
        """
        config = self.config_for(fmin, fmax, sr=int(sr))
        recording = self.recording_for(config)
        recording.audio_data = self._audio_data_from_samples(audio, config)
        stages = self.detect_stages(recording, smooth=self.smooth)
        times, freqs = self.melody(stages.data[self.stage], config)
        return (times, self.constrain_freqs_to_range(freqs, fmin, fmax))

    def estimate(
        self,
        example: PitchExample,
        use_cache: bool = True,
        force_reanalysis: bool = False,
    ) -> PitchEstimate:
        config = self.config_for(example.fmin, example.fmax)
        cache = self.cache(example)
        if (
            use_cache
            and (not force_reanalysis)
            and ((hit := cache.read(self.stage, config)) is not None)
        ):
            pitch_data, timing = hit
            if timing.get("compute_clock") == self.COMPUTE_CLOCK:
                return self.constrain_estimate_to_range(
                    PitchDetectorBase.PitchEstimate.build(
                        *self.melody(pitch_data, config),
                        timing["pitch_compute_time"],
                        from_cache=True,
                        metadata=timing,
                    ),
                    example.fmin,
                    example.fmax,
                )
        recording = self.recording_for(config)
        recording.audio_data = self._audio_data(example, config)
        stages = self.detect_stages(recording, smooth=True)
        if use_cache:
            cache.write(stages)
        return self.constrain_estimate_to_range(
            PitchDetectorBase.PitchEstimate.build(
                *self.melody(stages.data[self.stage], config),
                stages.timing[self.stage]["pitch_compute_time"],
                metadata=stages.timing[self.stage],
            ),
            example.fmin,
            example.fmax,
        )

    @staticmethod
    def _audio_data(example: PitchExample, config: Config) -> AudioData:
        samples, _ = example.audio(config.sr)
        return Attune._audio_data_from_samples(samples, config)

    @staticmethod
    def _audio_data_from_samples(audio_samples, config: Config) -> AudioData:
        samples = np.ascontiguousarray(audio_samples, dtype=np.float32)
        audio = AudioData(length=0, config=config)
        audio.data = samples
        audio.sr = int(config.sr)
        audio.capacity = samples.size
        audio.end_index = samples.size
        audio.t_origin = 0.0
        return audio

    @staticmethod
    def _pitch_data(pitches: list, config: Config, origin: float) -> PitchData:
        pitch_data = PitchData(config=config)
        pitch_data.data = pitches
        pitch_data.t_origin = origin
        if origin:
            for pitch in pitch_data.data:
                if pitch is not None:
                    pitch.time += origin
        return pitch_data


class AttuneRealtime(Attune):
    """Legacy benchmark alias for historical result readers."""

    name = "attune_realtime"
    description = "Legacy Attune causal stage"
    smooth = False


class AttuneAdHoc(Attune):
    """Legacy benchmark alias for historical result readers."""

    name = "attune_adhoc"
    description = "Legacy Attune completed-audio stage"
    smooth = True
