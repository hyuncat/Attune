from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np


def _bootstrap_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
            return candidate
    raise RuntimeError("could not locate Attune repo root")


_BOOTSTRAP_ROOT = _bootstrap_repo_root()
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from app_logic.NoteData import Note, NoteData  # noqa: E402
from benchmarks.modules.note.NoteBenchmarker import (
    NoteBenchmarker,
    PathLike,
)  # noqa: E402
from benchmarks.paths import REPO_ROOT, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()
ROOT = REPO_ROOT
MistakeType = Literal["substitution", "deletion", "insertion", "short", "long"]
_SOURCE_SCORE_ID_UNSET = object()


class TruthEvent(TypedDict, total=False):
    type: MistakeType
    score_note_id: int
    time: float
    original_duration: float
    performed_duration: float
    duration_error: float


class MistakeInjector:
    """
    Interface for injecting mistakes into scores, symbolically or re-synthesized through audio.
    Mirrors the PolyTune methods 'augment_mistakes' and 'add_screwups' sampling from
    'lambda_occur ~ U(0.1, 0.4)' and applying +/-50% count jitter to select error indices.

    Attune extends the original pitch-edit injection with explicit short/long
    events. Those events preserve nominal onset and pitch, sample within 0.5–1.5
    times the original duration, and require a salient absolute duration change.
    Ineligible short notes remain unchanged, with the skipped edit recorded.
    Output is always monophonic: insertions and duration overruns delay the
    remaining notes; deletions advance the suffix by the deleted duration.
    allow_overlap controls the preliminary span sampler only.
    Nominal onsets are retained for independent final net-truth matching.
    Inserted notes differ from both final performed neighbors; same-pitch
    extensions belong to the duration-error test category instead.

    Rk: When synthesized, the first and last notes are artificially longer due to reverb.
    We hard-correct for this by truncating any excess for those two edge notes only, and likewise
    we also omit injecting further mistakes into those.
    """

    # Match NoteBenchmarker.notedata_to_pm: 220 PPQ, default 120 BPM.
    MIDI_TICK_SECONDS = 0.5 / 220

    POLYTUNE_LAMBDA_RANGE = (0.1, 0.4)
    POLYTUNE_PITCH_STD = 1.0
    POLYTUNE_DURATION_MEAN = 1.0
    POLYTUNE_DURATION_STD = 0.02
    POLYTUNE_TIMING_MEAN_MS = 0.0
    POLYTUNE_TIMING_STD_MS = 300.0
    POLYTUNE_SCREWUP_TYPES = tuple(range(16))
    SHORT_DURATION_CODE = 4
    LONG_DURATION_CODE = 5
    DURATION_ERROR_RANGE_SEC = (0.30, 0.60)
    DURATION_FACTOR_RANGE = (0.5, 1.5)
    DURATION_ERROR_MIN_SEC = 0.30

    def __init__(
        self,
        lambda_range: tuple[float, float] = POLYTUNE_LAMBDA_RANGE,
        pitch_offset_std: float = POLYTUNE_PITCH_STD,
        duration_mean: float = POLYTUNE_DURATION_MEAN,
        duration_std: float = POLYTUNE_DURATION_STD,
        timing_mean_ms: float = POLYTUNE_TIMING_MEAN_MS,
        timing_std_ms: float = POLYTUNE_TIMING_STD_MS,
        duration_error_range_sec: tuple[float, float] | None = None,
        allow_overlap: bool = True,
        protect_boundary_notes: bool = True,
        mistake_rate: float | None = None,
        weights: Sequence[float] | None = None,
        screwup_type_weights: Sequence[float] | None = None,
        out_dir: PathLike | None = None,
        duration_factor_range: tuple[float, float] = DURATION_FACTOR_RANGE,
        duration_error_min_sec: float = DURATION_ERROR_MIN_SEC,
    ) -> None:
        if mistake_rate is not None:
            lambda_range = (float(mistake_rate), float(mistake_rate))
        self.lambda_range = (float(lambda_range[0]), float(lambda_range[1]))

        if screwup_type_weights is not None:
            if len(screwup_type_weights) != len(self.POLYTUNE_SCREWUP_TYPES):
                raise ValueError("screwup_type_weights must have length 16")
            self.screwup_type_weights = self._normalize_weights(screwup_type_weights)
        elif weights is not None:
            # Backwards compatible 3-vector:
            #   (substitution, deletion, insertion)
            # Duration-aware 5-vector:
            #   (substitution, deletion, insertion, short, long)
            if len(weights) not in {3, 5}:
                raise ValueError("weights must have length 3 or 5")
            probs = np.zeros(16, dtype=float)
            probs[1] = float(weights[0])
            probs[0] = float(weights[1])
            probs[3] = float(weights[2])
            if len(weights) == 5:
                probs[self.SHORT_DURATION_CODE] = float(weights[3])
                probs[self.LONG_DURATION_CODE] = float(weights[4])
            self.screwup_type_weights = self._normalize_weights(probs)
        else:
            self.screwup_type_weights = tuple([1.0 / 16.0] * 16)

        self.pitch_offset_std = float(pitch_offset_std)
        self.duration_mean = float(duration_mean)
        self.duration_std = float(duration_std)
        self.timing_mean_ms = float(timing_mean_ms)
        self.timing_std_ms = float(timing_std_ms)
        low, high = map(float, duration_factor_range)
        if not np.isfinite([low, high]).all() or not 0 < low < 1 < high:
            raise ValueError("duration_factor_range must satisfy 0 < low < 1 < high")
        self.duration_factor_range = (low, high)
        self.duration_error_min_sec = float(duration_error_min_sec)
        if (
            not np.isfinite(self.duration_error_min_sec)
            or self.duration_error_min_sec <= 0
        ):
            raise ValueError("duration_error_min_sec must be finite and positive")
        # Explicit opt-in only, for historical absolute-duration experiments.
        self.duration_error_range_sec = None
        if duration_error_range_sec is not None:
            error_min, error_max = map(float, duration_error_range_sec)
            if (
                not np.isfinite([error_min, error_max]).all()
                or not 0 < error_min <= error_max
            ):
                raise ValueError(
                    "duration_error_range_sec must be positive and ordered"
                )
            self.duration_error_range_sec = (error_min, error_max)
        self.allow_overlap = allow_overlap
        self.protect_boundary_notes = protect_boundary_notes
        self.out_dir = (
            Path(out_dir)
            if out_dir is not None
            else ROOT / "benchmarks" / "datasets" / "mistake-db"
        )
        self.last_metadata: dict[str, Any] = {}

    # @staticmethod
    def _normalize_weights(self, weights: Sequence[float]) -> tuple[float, ...]:
        arr = np.asarray(list(weights), dtype=float)
        if np.any(arr < 0):
            raise ValueError("weights must be non-negative")
        total = float(arr.sum())
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        return tuple(float(x / total) for x in arr)

    @staticmethod
    def _write_note_unique(note_data: NoteData, note: Note) -> None:
        while note.start_time in note_data.data:
            note.start_time += 1e-6
            note.end_time += 1e-6
        note_data.write_note(note)

    @staticmethod
    def _copy_note(
        note: Note,
        note_id: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        midi_num: Sequence[float] | None = None,
        source_score_id: Any = _SOURCE_SCORE_ID_UNSET,
    ) -> Note:
        out = Note(
            i=note.id if note_id is None else note_id,
            start_time=float(note.start_time if start_time is None else start_time),
            end_time=float(note.end_time if end_time is None else end_time),
            midi_num=list(note.midi_num if midi_num is None else midi_num),
            velocity=note.velocity,
            instrument=note.instrument,
        )
        if source_score_id is _SOURCE_SCORE_ID_UNSET:
            source_score_id = getattr(note, "source_score_id", note.id)
        out.source_score_id = source_score_id
        return out

    def _sample_pitch_delta(self, random_generator: np.random.Generator) -> int:
        """GitHub generator samples int(normal(0, stdev_pitch_delta))."""
        for _ in range(50):
            delta = int(random_generator.normal(loc=0.0, scale=self.pitch_offset_std))
            if delta <= -1 or delta >= 1:
                return delta
        return int(random_generator.choice([-1, 1]))

    def _sample_changed_pitch(
        self, original: float, random_generator: np.random.Generator
    ) -> int:
        """Put inserted/substituted pitches on the MIDI semitone grid.

        Preserve a nonzero change even at MIDI 0/127: clipping an outward
        offset must not silently recreate the original pitch.
        """
        base = self._clamp_midi(original)
        delta = self._sample_pitch_delta(random_generator)
        changed = self._clamp_midi(base + delta)
        if changed == base:
            changed = self._clamp_midi(base - delta)
        assert isinstance(changed, int) and changed != base
        return changed

    def _sample_timing_delta(self, random_generator: np.random.Generator) -> float:
        return float(
            random_generator.normal(
                loc=self.timing_mean_ms / 1000.0,
                scale=self.timing_std_ms / 1000.0,
            )
        )

    def _sample_duration(
        self, duration: float, random_generator: np.random.Generator
    ) -> float:
        variance = self.duration_std**2
        if duration <= 0 or variance <= 0 or self.duration_mean <= 0:
            return max(1e-6, duration)
        shape = self.duration_mean**2 / variance
        scale = variance / self.duration_mean
        duration_var = random_generator.gamma(shape, scale)
        out = max(0.5 * duration, duration * duration_var)
        return float(min(out, 2.0 * duration))

    def _duration_mistake_span(
        self,
        notes: Sequence[Note],
        note_index: int,
        requested_type: Literal["short", "long"],
        random_generator: np.random.Generator,
    ) -> tuple[Literal["short", "long"], float, float] | None:
        """Change duration while holding onset and pitch fixed.

        Sample uniformly over representable MIDI durations in the requested
        direction, excluding changes below the salience floor. Return None when
        the ratio bounds and floor cannot both hold. The final monophonic pass
        handles overruns by delaying the suffix, never clipping this duration.
        An explicitly supplied absolute range retains the historical sampler.
        """
        note = notes[note_index]
        start = float(note.start_time)
        original_duration = max(1e-6, note.duration())
        if self.duration_error_range_sec is None:
            low, high = (
                factor * original_duration for factor in self.duration_factor_range
            )
            if requested_type == "short":
                high = original_duration - self.duration_error_min_sec
            else:
                low = original_duration + self.duration_error_min_sec
            tick = self.MIDI_TICK_SECONDS
            first = max(1, int(np.ceil(low / tick - 1e-9)))
            last = int(np.floor(high / tick + 1e-9))
            if first > last:
                return None
            duration = int(random_generator.integers(first, last + 1)) * tick
            return requested_type, start, start + duration
        error_min, error_max = self.duration_error_range_sec
        mistake_type = requested_type
        if mistake_type == "short" and original_duration <= error_min + 1e-6:
            mistake_type = "long"

        if mistake_type == "short":
            maximum = min(error_max, original_duration - 1e-6)
            error = float(random_generator.uniform(error_min, maximum))
            duration = original_duration - error
        else:
            error = float(random_generator.uniform(error_min, error_max))
            duration = original_duration + error

        end = start + duration
        if not self.allow_overlap and note_index < len(notes) - 1:
            end = min(end, float(notes[note_index + 1].start_time))
        return mistake_type, start, end

    @staticmethod
    def _clamp_midi(midi_num: float) -> int:
        return int(np.clip(round(midi_num), 0, 127))

    @staticmethod
    def _protect_span(
        notes: Sequence[Note],
        note_index: int,
        start: float,
        end: float,
    ) -> tuple[float, float]:
        """Keep generated events off the first and last notes.

        Interior mistakes remain free to overlap other interior notes, but the
        adjacent boundary windows are preserved so synth attack/release artifacts
        do not contaminate the first/last note benchmark anchors.
        """
        if not notes or note_index <= 0 or note_index >= len(notes) - 1:
            return start, end

        duration = max(1e-6, end - start)
        if note_index == 1:
            min_start = notes[0].end_time
            if start < min_start:
                start = min_start
                end = start + duration
        if note_index == len(notes) - 2:
            max_end = notes[-1].start_time
            if end > max_end:
                end = max_end
                start = end - duration
        if start < 0:
            end -= start
            start = 0.0
        if end <= start:
            end = start + 1e-6
        return float(start), float(end)

    def _choose_error_indices(
        self,
        note_count: int,
        random_generator: np.random.Generator,
    ) -> tuple[float, set[int]]:
        lambda_value = float(
            random_generator.uniform(self.lambda_range[0], self.lambda_range[1])
        )
        if self.protect_boundary_notes and note_count > 2:
            candidates = np.arange(1, note_count - 1)
        elif self.protect_boundary_notes:
            candidates = np.asarray([], dtype=int)
        else:
            candidates = np.arange(note_count)
        if candidates.size == 0:
            return lambda_value, set()
        base_size = min(int(np.ceil(lambda_value * candidates.size)), candidates.size)
        half_range = base_size // 2
        size_adjustment = int(random_generator.integers(-half_range, half_range + 1))
        size = max(0, min(candidates.size, base_size + size_adjustment))
        selected = random_generator.choice(candidates, size=size, replace=False)
        return lambda_value, {int(i) for i in selected}

    def _timed_span(
        self,
        notes: Sequence[Note],
        note_index: int,
        random_generator: np.random.Generator,
    ) -> tuple[float, float]:
        note = notes[note_index]
        original_start = float(note.start_time)
        original_end = float(note.end_time)
        original_duration = max(1e-6, original_end - original_start)
        duration = self._sample_duration(original_duration, random_generator)
        start = original_start + self._sample_timing_delta(random_generator)

        if not self.allow_overlap:
            if 0 < note_index < len(notes) - 1:
                prev_start = notes[note_index - 1].start_time
                next_start = notes[note_index + 1].start_time
                attempts = 0
                while (
                    (start - original_start >= next_start - original_start)
                    or (start - original_start <= prev_start - original_start)
                ) and attempts < 100:
                    start = original_start + self._sample_timing_delta(random_generator)
                    attempts += 1
            elif note_index == 0 and len(notes) > 1:
                next_start = notes[1].start_time
                attempts = 0
                while (
                    start - original_start >= next_start - original_start
                    and attempts < 100
                ):
                    start = original_start + self._sample_timing_delta(random_generator)
                    attempts += 1
            elif note_index == len(notes) - 1 and len(notes) > 1:
                prev_start = notes[-2].start_time
                attempts = 0
                while (
                    start - original_start <= prev_start - original_start
                    and attempts < 100
                ):
                    start = original_start + self._sample_timing_delta(random_generator)
                    attempts += 1

        end = start + duration
        if not self.allow_overlap:
            if note_index > 0:
                start = max(start, notes[note_index - 1].end_time)
            if note_index < len(notes) - 1:
                end = min(end, notes[note_index + 1].start_time)
            if end <= start:
                end = start + 1e-6

        start, end = self._protect_span(notes, note_index, start, end)
        # PolyTune applies onset/duration jitter as performance realism only: the
        # shifted note still lands in midi_correct_notes (never labeled a timing
        # mistake), and their evaluator scores onset+pitch with offset_ratio=None.
        # So we emit NO timing truth here -- the jittered span is all we return.
        return start, end

    def inject(
        self,
        reference_notes: NoteData,
        random_generator: np.random.Generator | None = None,
    ) -> tuple[NoteData, list[TruthEvent]]:
        random_generator = (
            random_generator
            if random_generator is not None
            else np.random.default_rng()
        )
        notes = reference_notes.read(i=0, j=len(reference_notes.times))
        lambda_value, error_indices = self._choose_error_indices(
            len(notes),
            random_generator,
        )
        selected_metadata: list[dict[str, Any]] = []
        out_notes: list[Note] = []
        performance_notes = NoteData()
        truth: list[TruthEvent] = []
        deleted_duration = 0.0
        deleted_before = {}
        for note_index, note in enumerate(notes):
            deleted_before[note.id] = deleted_duration
            output_start = len(out_notes)
            if note_index not in error_indices:
                out_notes.append(self._copy_note(note))
                continue

            screwup_type = int(
                random_generator.choice(
                    self.POLYTUNE_SCREWUP_TYPES,
                    p=self.screwup_type_weights,
                )
            )
            selected_metadata.append(
                {
                    "score_note_index": int(note_index),
                    "score_note_id": int(note.id),
                    "screwup_type": screwup_type,
                }
            )

            if screwup_type == 0:
                deleted_duration += note.duration()
                truth.append(
                    dict(
                        type="deletion",
                        score_note_id=note.id,
                        time=note.start_time,
                    )
                )
                continue

            if screwup_type in {self.SHORT_DURATION_CODE, self.LONG_DURATION_CODE}:
                requested_type: Literal["short", "long"] = (
                    "short" if screwup_type == self.SHORT_DURATION_CODE else "long"
                )
                span = self._duration_mistake_span(
                    notes,
                    note_index,
                    requested_type,
                    random_generator,
                )
                if span is None:
                    out_notes.append(self._copy_note(note))
                    selected_metadata[-1][
                        "skipped_reason"
                    ] = "duration_bounds_cannot_meet_salience_floor"
                    continue
                duration_type, start_time, end_time = span
                original_duration = max(1e-6, note.duration())
                performed_duration = max(1e-6, end_time - start_time)
                out_notes.append(
                    self._copy_note(
                        note,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
                if self.duration_error_range_sec is None:
                    out_notes[-1].duration_error_ticks = round(
                        performed_duration / self.MIDI_TICK_SECONDS
                    )
                truth.append(
                    dict(
                        type=duration_type,
                        score_note_id=note.id,
                        time=note.start_time,
                        original_duration=float(original_duration),
                        performed_duration=float(performed_duration),
                        duration_error=float(performed_duration - original_duration),
                    )
                )
                selected_metadata[-1]["emitted_type"] = duration_type
                continue

            start_time, end_time = self._timed_span(
                notes,
                note_index,
                random_generator,
            )

            if screwup_type == 1:
                new_pitch = self._sample_changed_pitch(
                    note.midi_num[0], random_generator
                )
                out_notes.append(
                    self._copy_note(
                        note,
                        start_time=start_time,
                        end_time=end_time,
                        midi_num=[new_pitch],
                    )
                )
                truth.append(
                    dict(
                        type="substitution",
                        score_note_id=note.id,
                        time=note.start_time,
                    )
                )
                continue

            if screwup_type == 2:
                new_pitch = self._sample_changed_pitch(
                    note.midi_num[0], random_generator
                )
                original_duration = max(1e-6, note.duration())
                initial_duration = original_duration / 8.0 + random_generator.uniform(
                    low=-original_duration / 32.0,
                    high=(original_duration / 8.0) * 3.0,
                )
                initial_duration = max(
                    1e-6, min(initial_duration, end_time - start_time)
                )
                wrong_end = start_time + initial_duration
                out_notes.append(
                    self._copy_note(
                        note,
                        start_time=start_time,
                        end_time=wrong_end,
                        midi_num=[new_pitch],
                        source_score_id=None,
                    )
                )
                if wrong_end < end_time:
                    out_notes.append(
                        self._copy_note(
                            note,
                            start_time=wrong_end,
                            end_time=end_time,
                        )
                    )
                truth.append(
                    dict(
                        type="insertion", time=start_time, _performed_index=output_start
                    )
                )
                continue

            if screwup_type == 3:
                timed_note = self._copy_note(
                    note,
                    start_time=start_time,
                    end_time=end_time,
                )
                out_notes.append(timed_note)
                extra_duration = self._sample_duration(
                    max(1e-6, timed_note.duration()),
                    random_generator,
                )
                # A new sequential note occupies its own time after the host.
                # The final timeline pass delays every later note as needed.
                extra_start = float(timed_note.end_time)
                extra_end = extra_start + extra_duration
                inserted_pitch = self._sample_changed_pitch(
                    timed_note.midi_num[0], random_generator
                )
                out_notes.append(
                    self._copy_note(
                        timed_note,
                        start_time=extra_start,
                        end_time=extra_end,
                        midi_num=[inserted_pitch],
                        source_score_id=None,
                    )
                )
                truth.append(
                    dict(
                        type="insertion",
                        time=extra_start,
                        _performed_index=len(out_notes) - 1,
                    )
                )
                continue

            # Remaining GitHub placeholder codes apply the jittered span as
            # realism but emit no truth event. Codes 4 and 5 are now explicit
            # Attune short/long extensions handled above.
            out_notes.append(
                self._copy_note(note, start_time=start_time, end_time=end_time)
            )

        # Check final neighbors, after substitutions/deletions have changed them.
        # Repair only inserted pitches, never the neighboring score notes or the
        # timeline. Left-to-right repair also handles adjacent insertions: later
        # repairs exclude the pitch already assigned to their left neighbor.
        inserted_indices = sorted(
            e["_performed_index"] for e in truth if e["type"] == "insertion"
        )
        source_pitches = {n.id: n.midi_num for n in notes}
        pitch_repairs = []
        for i in inserted_indices:
            inserted = out_notes[i]
            neighbors = [
                out_notes[j] for j in (i - 1, i + 1) if 0 <= j < len(out_notes)
            ]
            forbidden = {self._clamp_midi(p) for n in neighbors for p in n.midi_num}
            forbidden.update(self._clamp_midi(p) for p in source_pitches[inserted.id])
            original = int(inserted.midi_num[0])
            if original in forbidden:
                allowed = [p for p in range(128) if p not in forbidden]
                distance = min(abs(p - original) for p in allowed)
                nearest = [p for p in allowed if abs(p - original) == distance]
                inserted.midi_num = [
                    nearest[int(random_generator.integers(len(nearest)))]
                ]
                pitch_repairs.append(
                    dict(
                        performed_index=i,
                        old_pitch=original,
                        new_pitch=inserted.midi_num[0],
                    )
                )
        for i in inserted_indices:
            if any(
                out_notes[i].midi_num[0] in out_notes[j].midi_num
                for j in (i - 1, i + 1)
                if 0 <= j < len(out_notes)
            ):
                raise ValueError("Inserted pitch matches a final performed neighbor")

        # Preserve nominal score positions for net-truth matching before
        # inserting time. This also handles long-duration edits without mixing
        # pitches: an overrun delays all subsequent events, never clips a note.
        added_delay = 0.0
        cursor = 0.0
        end_tick = 0
        # Keep each insertion next to its host, even when sampled durations
        # extend beyond the next nominal onset.
        ordered = out_notes
        for output_index, note in enumerate(ordered):
            note.comparison_time = float(note.start_time)
            shift = added_delay - deleted_before[note.id]
            correction = max(
                0.0, cursor - (note.start_time + shift), -(note.start_time + shift)
            )
            added_delay += correction
            shift += correction
            # Zero-tick notes serialize note-off before note-on and can hang
            # until a later note of the same pitch. Quantize before synthesis,
            # keeping every event at least one tick long and nonoverlapping.
            nominal_start_tick = round(
                (note.start_time + shift) / self.MIDI_TICK_SECONDS
            )
            start_tick = max(end_tick, nominal_start_tick)
            end_tick = max(
                start_tick + 1,
                round((note.end_time + shift) / self.MIDI_TICK_SECONDS)
                + start_tick
                - nominal_start_tick,
            )
            if hasattr(note, "duration_error_ticks"):
                end_tick = start_tick + note.duration_error_ticks
            note.start_time = start_tick * self.MIDI_TICK_SECONDS
            note.end_time = end_tick * self.MIDI_TICK_SECONDS
            note.timeline_delay = note.start_time - note.comparison_time
            cursor = note.end_time
            note.id = output_index
            self._write_note_unique(performance_notes, note)

        # Keep returned event times on the final performed timeline. The
        # nominal score positions remain available separately for auditing.
        for event in truth:
            if event["type"] == "deletion":
                continue
            if event["type"] == "insertion":
                candidates = [ordered[event.pop("_performed_index")]]
            else:
                candidates = [
                    n
                    for n in performance_notes.data.values()
                    if n.source_score_id == event.get("score_note_id")
                ]
            if len(candidates) != 1:
                raise ValueError(
                    "Cannot relink injected event to final monophonic timeline"
                )
            event["score_time"] = float(event["time"])
            event["time"] = float(candidates[0].start_time)
            if event["type"] in {"short", "long"}:
                event["performed_duration"] = float(candidates[0].duration())
                event["duration_error"] = (
                    event["performed_duration"] - event["original_duration"]
                )

        self.last_metadata = {
            "method": "polytune-github-add_screwups",
            "lambda": lambda_value,
            "lambda_range": list(self.lambda_range),
            "screwup_types": list(self.POLYTUNE_SCREWUP_TYPES),
            "screwup_type_weights": list(self.screwup_type_weights),
            "pitch_offset_std": self.pitch_offset_std,
            "changed_pitch_grid": "integer MIDI semitones; nonzero offset from rounded source",
            "insertion_pitch_policy": "distinct_from_final_neighbors_v1",
            "inserted_note_indices": inserted_indices,
            "insertion_pitch_repairs": pitch_repairs,
            "duration_mean": self.duration_mean,
            "duration_std": self.duration_std,
            "duration_error_policy": (
                "relative_salient_v1"
                if self.duration_error_range_sec is None
                else "legacy_absolute"
            ),
            "duration_factor_range": list(self.duration_factor_range),
            "duration_error_min_sec": self.duration_error_min_sec,
            "duration_error_range_sec": (
                list(self.duration_error_range_sec)
                if self.duration_error_range_sec is not None
                else None
            ),
            "timing_mean_ms": self.timing_mean_ms,
            "timing_std_ms": self.timing_std_ms,
            "allow_overlap": False,
            "span_sampler_allow_overlap": self.allow_overlap,
            "timeline_protocol": "monophonic_edits_v3",
            "midi_tick_seconds": self.MIDI_TICK_SECONDS,
            "protect_boundary_notes": self.protect_boundary_notes,
            "selected_note_indices": sorted(error_indices),
            "selected": selected_metadata,
        }
        return performance_notes, truth

    def synth(self, performance_notes: NoteData, name: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        midi_path = self.out_dir / f"{name}.mid"
        NoteBenchmarker.notedata_to_pm(performance_notes).write(str(midi_path))
        return NoteBenchmarker().synth_midi(midi_path, out_dir=self.out_dir, force=True)
