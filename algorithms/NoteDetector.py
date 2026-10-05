from collections.abc import Sequence

import numpy as np
import ruptures as rpt
from PyQt6.QtCore import QObject

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording
from algorithms.Config import Config


class NoteDetector(QObject):
    def __init__(self, recording: Recording, config: Config=None, parent: QObject=None) -> None:
        super().__init__(parent)

        self.recording = recording
        self.config = config if config else recording.config
        self.verbose = self.config.verbose

    def update_config(self, config: Config):
        """update the config and all relevant parameters"""
        self.config = config
        self.verbose = self.config.verbose

    def get_pitch_runs(self, pitches: list[Pitch]) -> list[list[Pitch]]:
        """Group voiced frames, splitting on majority-confirmed silence gaps."""
        minimum_run_frames = self.config.note_detection_min_frames()
        silence_window_frames = self.silence_window_frames()
        voiced_frames = np.fromiter(
            (
                self.recording.pitch_data.is_voiced_pitch(pitch)
                for pitch in pitches
            ),
            dtype=bool,
            count=len(pitches),
        )
        confirmed_silence = self._majority_silence_mask(
            voiced_frames,
            silence_window_frames,
        )

        pitch_runs: list[list[Pitch]] = []
        current_run: list[Pitch] = []
        for pitch, is_voiced, is_confirmed_silence in zip(
            pitches,
            voiced_frames,
            confirmed_silence,
        ):
            if is_confirmed_silence:
                if current_run:
                    if len(current_run) >= minimum_run_frames:
                        pitch_runs.append(current_run)
                    current_run = []
            elif is_voiced:
                current_run.append(pitch)

        if len(current_run) >= minimum_run_frames:
            pitch_runs.append(current_run)
        return pitch_runs

    def detect_notes(self, pitches: list[Pitch]) -> NoteData:
        """Segment smoothed pitches with the fixed production configuration."""
        if self.verbose:
            print("[NoteDetector] detecting notes (linear KernelCPD)", flush=True)

        pitch_runs = self.get_pitch_runs(pitches)

        if not pitch_runs:
            if self.verbose:
                print("[NoteDetector] done: 0 runs, 0 note(s)", flush=True)
            return NoteData()

        nd = NoteData()
        note_idx = 0  # global across runs so note.id == sorted position
        for pitches in pitch_runs:
            # get list of breakpoints (note-end indices)
            run = [p for p in pitches if p]
            signal = np.asarray([p.value for p in run], dtype=float).reshape(-1, 1)
            breakpoints = self.segment_breakpoints(signal)
            breakpoints = self._merge_same_pitch_segments(signal, breakpoints)

            # retrace the notes from the breakpoints
            prev_idx = 0
            for pt in breakpoints:
                end_idx = min(int(pt), len(run))
                if end_idx <= prev_idx:
                    continue
                midi_num = float(np.median(signal[prev_idx:end_idx]))
                start_time = self.get_boundary_time(run, prev_idx)
                end_time = self.get_boundary_time(run, end_idx)

                if end_time <= start_time:
                    prev_idx = end_idx
                    continue

                nd.write_note(
                    Note(
                        i=note_idx,
                        start_time=start_time,
                        end_time=end_time,
                        midi_num=[midi_num], #todo: make this a float not a list
                ))
                note_idx += 1
                prev_idx = end_idx

        return nd

    @staticmethod
    def _merge_same_pitch_segments(signal, breakpoints):
        """Consolidate contiguous fragments with at most 0.5 semitones of spread.

        A pitch-driven change point alone is not evidence of a new articulation
        when the original segment medians stay within a half-semitone range.
        Track the whole group's extrema to prevent gradual pitch drift from
        chaining arbitrarily far. Groups are built greedily from left to right.
        Keep the outer boundaries; detect_notes recomputes the median from the
        original frames, without rounding or averaging segment medians.
        Runs separated by confirmed silence never pass through this together.
        Score repetitions are still handled by RepeatSplitter afterwards.
        """
        merged = []
        previous_end = 0
        group_min = group_max = None
        for end in breakpoints:
            pitch = float(np.median(signal[previous_end:end]))
            if merged and max(group_max, pitch) - min(group_min, pitch) <= 0.5:
                merged[-1] = end
                group_min = min(group_min, pitch)
                group_max = max(group_max, pitch)
            else:
                merged.append(end)
                group_min = group_max = pitch
            previous_end = end
        return merged

    def segment_breakpoints(
        self,
        signal: np.ndarray,
        segment_count: int | None = None,
        min_length_seconds: float | None = None,
    ) -> list[int] | None:
        """Segment one pitch run with the production linear KernelCPD model.

        Normal note detection lets the configured penalty choose the number of
        segments. Mistake correction supplies ``segment_count`` so the same
        model returns exactly the number of notes required by a candidate split.
        It may also supply an explicit correction-only minimum duration without
        changing the production detector's capped score-relative default.
        """
        frame_count = len(signal)
        if min_length_seconds is None:
            min_frames = self.config.note_detection_min_frames()
        else:
            frame_rate = self.config.sr / self.config.h1
            min_frames = max(
                1,
                int(np.ceil(
                    max(0.0, float(min_length_seconds)) * frame_rate
                )),
            )
        if segment_count is not None:
            if segment_count < 1 or frame_count < segment_count * min_frames:
                return None
            if segment_count == 1:
                return [frame_count]
        elif frame_count < 2 * min_frames:
            return [frame_count]

        # A pitch step of pitch_thresh across two minimum-length regions is
        # neutral between one and two segments at this penalty.
        penalty = 0.5 * min_frames * (self.config.pitch_thresh ** 2)
        try:
            model = rpt.KernelCPD(
                kernel="linear",
                min_size=min_frames,
                jump=1,
            ).fit(signal)
            breakpoints = model.predict(
                n_bkps=segment_count - 1
                if segment_count is not None
                else None,
                pen=penalty if segment_count is None else None,
            )
        except (rpt.exceptions.BadSegmentationParameters, ValueError):
            return None if segment_count is not None else [frame_count]

        if (
            segment_count is not None
            and len(breakpoints) != segment_count
        ):
            return None
        return [int(point) for point in breakpoints]


    def silence_window_frames(self) -> int:
        """Convert a decoded-silence duration to an odd majority window.

        At the default 128/44100-second hop, 40 ms rounds to 13 frames and
        requires seven unvoiced frames to confirm a silence span.
        """
        frame_rate = self.config.sr / self.config.h1
        target = (
            max(0.0, float(self.config.min_silence_duration_ms))
            * frame_rate
            / 1000.0
        )
        frames = max(1, int(round(target)))
        if frames % 2:
            return frames
        candidates = (max(1, frames - 1), frames + 1)
        return min(
            candidates,
            key=lambda candidate: (abs(candidate - target), candidate),
        )

    @staticmethod
    def _majority_silence_mask(voiced: np.ndarray, window: int) -> np.ndarray:
        """Mark spans covered by windows containing a majority of silence."""
        n_frames = len(voiced)
        silence_mask = np.zeros(n_frames, dtype=bool)
        if n_frames < window:
            return silence_mask

        required_silence_frames = window // 2 + 1
        silence_frame_indices = np.flatnonzero(~voiced)
        if len(silence_frame_indices) < required_silence_frames:
            return silence_mask

        pending_gap: tuple[int, int] | None = None

        # Any majority-sized group of silent frames that fits within one window
        # confirms the span from its first through its last silent decision.
        n_groups = len(silence_frame_indices) - required_silence_frames + 1
        for group_start in range(n_groups):
            group_stop = group_start + required_silence_frames
            confirmed_start = int(silence_frame_indices[group_start])
            confirmed_stop = int(silence_frame_indices[group_stop - 1]) + 1
            if confirmed_stop - confirmed_start > window:
                continue

            # Adjacent qualifying groups represent one confirmed silence event.
            if pending_gap is None:
                pending_gap = (confirmed_start, confirmed_stop)
            elif confirmed_start <= pending_gap[1]:
                pending_gap = (
                    pending_gap[0],
                    max(pending_gap[1], confirmed_stop),
                )
            else:
                silence_mask[pending_gap[0]:pending_gap[1]] = True
                pending_gap = (confirmed_start, confirmed_stop)

        if pending_gap is not None:
            silence_mask[pending_gap[0]:pending_gap[1]] = True
        return silence_mask


    @staticmethod
    def get_boundary_time(pitches: Sequence[Pitch], index: int) -> float:
        """Place a boundary midway between adjacent retained pitch frames."""
        if index <= 0:
            return pitches[0].time
        if index >= len(pitches):
            return pitches[-1].time
        return 0.5 * (pitches[index - 1].time + pitches[index].time)


# legacy transition detection
# precise note onsets may require more delicate thought than just a slope threshold
# @gordon good luck :)
class TransitionDetector:
    def __init__(self, recording: Recording):
        self.recording = recording
        self.config = recording.config

    def get_slope(self, pitches: list[Pitch]):
        """get slope of all voiced pitches in the window"""
        # select only voiced x and y values
        mask  = np.array([p.unvoiced_prob < self.config.unv_thresh if p else False for p in pitches]) # boolean mask

        all_x = np.linspace(start=0, stop=len(pitches), num=len(pitches))
        x_voiced = all_x[mask]
        y_voiced = np.array([p.value for p, m in zip(pitches, mask) if m])

        if x_voiced.size == 0:
            return 0.0, 0.0

        # get slope + intercept of only voiced pitches
        A = np.vstack([x_voiced, np.ones_like(x_voiced)]).T
        slope, intercept = np.linalg.lstsq(A, y_voiced, rcond=None)[0]

        return slope, intercept

    def detect_transitions(self, pitches: list[Pitch]):
        """Marks pitches as transitions in-place if slope of the window is too-steep
        (ie, moves up/down >0.5 semitones per window)
        """
        WINDOW = 9
        HOP = 7
        SLOPE_THRESH = 0.5 / WINDOW
        self.clear_transitions(pitches)
        if len(pitches) < WINDOW:
            return

        frames = np.lib.stride_tricks.sliding_window_view(pitches, window_shape=WINDOW)[::HOP]

        for frame in frames:
            if not frame[0]:
                continue
            slope, _ = self.get_slope(frame)
            if abs(slope) > SLOPE_THRESH:
                for p in frame:
                    if p:
                        p.is_transition = True

    @staticmethod
    def clear_transitions(pitches: list[Pitch]):
        """Remove derived transition flags, including ones restored from cache."""
        for pitch in pitches:
            if pitch:
                pitch.is_transition = False
