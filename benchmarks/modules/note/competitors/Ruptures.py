from __future__ import annotations
from app_logic.user.ds.PitchData import PitchData
from typing import Any
from typing import Sequence
import numpy as np
import ruptures as rpt
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData
from app_logic.user.ds.PitchData import Pitch
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase


class Ruptures(NoteDetectorBase):
    """Ruptures change-point competitors and their parameter variants."""

    def detect(
        self,
        ruptures_algorithm: str = "pelt",
        algorithm: str | None = None,
        model: str = "l2",
        cost: str | None = None,
        pen: float | None = None,
        pitch_step_semitones: float | None = None,
        min_note_length_factor: float | None = None,
        min_silence_duration_ms: float | None = None,
        jump: int | None = None,
        width: int | None = None,
        n_bkps: int | None = None,
        oracle_note_count: bool = False,
        exclude_transitions: bool = False,
        do_transitions: bool | None = None,
        features: Sequence[str] | None = None,
        standardize_features: bool | None = None,
        merge_adjacent: bool = False,
        **_unused: Any,
    ) -> NoteData:
        if algorithm is not None:
            ruptures_algorithm = algorithm
        ruptures_algorithm = ruptures_algorithm.lower()
        if ruptures_algorithm not in self.RUPTURES_ALGORITHMS:
            raise ValueError(f"unknown ruptures algorithm: {ruptures_algorithm!r}")
        if do_transitions is not None:
            exclude_transitions = bool(do_transitions)
        self._prepare_transition_flags(exclude_transitions)
        self._configure_note_segmentation(
            min_note_length_factor=min_note_length_factor,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        min_size = self._pelt_min_size_from_score()
        runs = self._pelt_runs(self.recording.pitch_data)
        if not runs:
            return NoteData()
        penalty = (
            self._pelt_penalty(min_size, pitch_step_semitones)
            if pen is None
            else float(pen)
        )
        pelt_jump = self._pelt_jump(jump)
        feature_names = tuple(features or self._default_features(cost or model))
        if standardize_features is None:
            standardize_features = len(feature_names) > 1
        budgets = self._run_breakpoint_budgets(
            runs,
            min_size=min_size,
            n_bkps=n_bkps,
            oracle_note_count=oracle_note_count,
            algorithm=ruptures_algorithm,
        )
        bkps_by_run: list[list[int]] = []
        for run, run_n_bkps in zip(runs, budgets):
            signal = self._feature_matrix(
                run, feature_names=feature_names, standardize=bool(standardize_features)
            )
            bkps_by_run.append(
                self._predict_bkps(
                    signal,
                    algorithm=ruptures_algorithm,
                    model=cost or model,
                    min_size=min_size,
                    jump=pelt_jump,
                    penalty=penalty,
                    width=width,
                    n_bkps=run_n_bkps,
                )
            )
        return self._notes_from_run_breakpoints(
            runs, bkps_by_run, merge_adjacent=merge_adjacent
        )

    def _predict_bkps(
        self,
        signal: np.ndarray,
        algorithm: str,
        model: str,
        min_size: int,
        jump: int,
        penalty: float,
        width: int | None,
        n_bkps: int | None,
    ) -> list[int]:
        n_frames = len(signal)
        if n_frames < 2 * min_size:
            return [n_frames]
        try:
            if algorithm == "kernelcpd":
                kernel = "linear" if model == "l2" else model
                algo = rpt.KernelCPD(kernel=kernel, min_size=min_size, jump=jump).fit(
                    signal
                )
                bkps = (
                    algo.predict(n_bkps=int(n_bkps))
                    if n_bkps is not None
                    else algo.predict(pen=penalty)
                )
            elif algorithm == "bottomup":
                algo = rpt.BottomUp(model=model, min_size=min_size, jump=jump).fit(
                    signal
                )
                bkps = (
                    algo.predict(n_bkps=int(n_bkps))
                    if n_bkps is not None
                    else algo.predict(pen=penalty)
                )
            elif algorithm == "window":
                window_width = self._window_width(width, min_size, n_frames)
                algo = rpt.Window(
                    width=window_width, model=model, min_size=min_size, jump=jump
                ).fit(signal)
                bkps = (
                    algo.predict(n_bkps=int(n_bkps))
                    if n_bkps is not None
                    else algo.predict(pen=penalty)
                )
            elif algorithm == "dynp":
                if not n_bkps:
                    return [n_frames]
                algo = rpt.Dynp(model=model, min_size=min_size, jump=jump).fit(signal)
                bkps = algo.predict(n_bkps=int(n_bkps))
            else:
                algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
                bkps = algo.predict(pen=penalty)
        except (rpt.exceptions.BadSegmentationParameters, ValueError):
            return [n_frames]
        return self._sanitize_bkps(bkps, n_frames)

    @staticmethod
    def _sanitize_bkps(bkps: Sequence[int], n_frames: int) -> list[int]:
        clean = sorted({int(b) for b in bkps if 0 < int(b) <= n_frames})
        if not clean or clean[-1] != n_frames:
            clean.append(n_frames)
        return clean

    @staticmethod
    def _window_width(width: int | None, min_size: int, n_frames: int) -> int:
        if n_frames <= 2:
            return 2
        candidate = int(width or max(2 * min_size, 5 * min_size))
        candidate = max(2, min(candidate, n_frames - 1))
        return candidate

    def _run_breakpoint_budgets(
        self,
        runs: Sequence[Sequence[Pitch]],
        min_size: int,
        n_bkps: int | None,
        oracle_note_count: bool,
        algorithm: str,
    ) -> list[int | None]:
        if algorithm != "dynp" and n_bkps is None:
            return [None] * len(runs)
        if oracle_note_count and n_bkps is None:
            expected_notes = self._expected_score_note_count()
            total_budget = max(0, expected_notes - len(runs))
        else:
            total_budget = max(0, int(n_bkps or 0))
        if total_budget <= 0:
            return [0] * len(runs)
        lengths = np.asarray([len(run) for run in runs], dtype=float)
        weights = lengths / max(float(lengths.sum()), 1.0)
        raw = weights * total_budget
        budgets = np.floor(raw).astype(int)
        for idx in np.argsort(raw - budgets)[::-1][: total_budget - int(budgets.sum())]:
            budgets[idx] += 1
        capped: list[int] = []
        for budget, run in zip(budgets, runs):
            max_bkps = max(0, len(run) // max(1, min_size) - 1)
            capped.append(min(int(budget), max_bkps))
        return capped

    def _expected_score_note_count(self) -> int:
        try:
            score_notes = self.recording.score_data.clipped_note_data(
                channel=self.recording.active_instrument
            )
        except (AttributeError, KeyError, TypeError):
            return 0
        return len(score_notes.read(i=0, j=len(score_notes.times), clean=True))

    @staticmethod
    def _default_features(model: str) -> tuple[str, ...]:
        if model == "normal":
            return ("pitch", "delta_pitch", "volume", "voiced_prob")
        return ("pitch",)

    def _feature_matrix(
        self, pitches: Sequence[Pitch], feature_names: Sequence[str], standardize: bool
    ) -> np.ndarray:
        pitch_values = np.asarray(
            [float(self._frame_pitch(p)) for p in pitches], dtype=float
        )
        deltas = (
            np.gradient(pitch_values)
            if len(pitch_values) > 1
            else np.zeros_like(pitch_values)
        )
        columns = []
        for name in feature_names:
            if name == "pitch":
                values = pitch_values
            elif name == "delta_pitch":
                values = deltas
            elif name == "volume":
                values = np.asarray([float(p.volume) for p in pitches], dtype=float)
            elif name == "voiced_prob":
                values = np.asarray(
                    [1.0 - float(p.unvoiced_prob) for p in pitches], dtype=float
                )
            elif name == "confidence":
                values = np.asarray([self._confidence(p) for p in pitches], dtype=float)
            elif name == "unvoiced_prob":
                values = np.asarray(
                    [float(p.unvoiced_prob) for p in pitches], dtype=float
                )
            else:
                raise ValueError(f"unknown ruptures feature: {name!r}")
            columns.append(values)
        signal = np.column_stack(columns).astype(float)
        if standardize:
            means = signal.mean(axis=0)
            stds = signal.std(axis=0)
            stds[stds < 1e-09] = 1.0
            signal = (signal - means) / stds
        return signal

    def _notes_from_run_breakpoints(
        self,
        runs: Sequence[Sequence[Pitch]],
        bkps_by_run: Sequence[Sequence[int]],
        merge_adjacent: bool,
    ) -> NoteData:
        note_data = NoteData()
        note_index = 0
        for pitches, bkps in zip(runs, bkps_by_run):
            prev = 0
            first_segment_in_run = True
            for bkp in bkps:
                end = min(int(bkp), len(pitches))
                if end <= prev:
                    continue
                segment = list(pitches[prev:end])
                midi_num = self._pelt_segment_pitch(segment)
                start_time = self.detector.get_boundary_time(pitches, prev)
                end_time = self.detector.get_boundary_time(pitches, end)
                if end_time <= start_time:
                    prev = end
                    continue
                last = (
                    note_data.read_note(i=len(note_data.times) - 1)
                    if note_data.times and (not first_segment_in_run)
                    else None
                )
                if merge_adjacent and self._same_note_pitch(last, midi_num):
                    last.end_time = end_time
                else:
                    note_data.write_note(
                        Note(
                            i=note_index,
                            start_time=start_time,
                            end_time=end_time,
                            midi_num=midi_num,
                        )
                    )
                    note_index += 1
                prev = end
                first_segment_in_run = False
        return self._reindex(note_data)

    RUPTURES_ALGORITHMS = {"pelt", "kernelcpd", "bottomup", "window", "dynp"}

    def _configure_note_segmentation(
        self,
        min_note_length_factor: float | None = None,
        min_silence_duration_ms: float | None = None,
    ) -> None:
        """Apply benchmark overrides through the recording's Config."""
        changed = False
        if min_note_length_factor is not None:
            self.config.min_note_length_factor = float(min_note_length_factor)
            changed = True
        if min_silence_duration_ms is not None:
            self.config.min_silence_duration_ms = float(min_silence_duration_ms)
            changed = True
        if changed:
            self.recording.update_config(self.config)

    def _pelt_min_size_from_score(self) -> int:
        return self.config.min_note_pitch_frames(
            factor=self.config.min_note_length_factor
        )

    def _pelt_penalty(
        self, min_size: int, pitch_step_semitones: float | None = None
    ) -> float:
        pitch_step = (
            self.config.pitch_thresh
            if pitch_step_semitones is None
            else float(pitch_step_semitones)
        )
        return 0.5 * int(min_size) * pitch_step**2

    def _pelt_jump(self, jump: int | None = None) -> int:
        return max(1, int(1 if jump is None else jump))

    def _pelt_runs(self, pitch_data: PitchData) -> list[list[Pitch]]:
        return self.recording.note_detector.get_pitch_runs(pitch_data.data)

    def _pelt_segment_pitch(self, pitches: Sequence[Pitch]) -> list[float]:
        med = self._get_median_pitches(pitches, n_candidates=3)
        return med if med[0] != -1 else [-1.0, -1.0, -1.0]

    def _same_note_pitch(self, note: Note | None, midi_num: Sequence[float]) -> bool:
        if note is None or not note.midi_num or (not midi_num):
            return False
        a, b = (note.midi_num[0], midi_num[0])
        if a == -1 and b == -1:
            return True
        if a == -1 or b == -1:
            return False
        return abs(a - b) < self.config.pitch_thresh

    @staticmethod
    def _confidence(pitch: Pitch) -> float:
        if pitch is None or not pitch.candidate_pitches:
            return 0.0
        return float(pitch.candidate_pitches[0][1])
