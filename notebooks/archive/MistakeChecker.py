from __future__ import annotations

from copy import copy

import numpy as np

from algorithms.Config import Config
from app_logic.Alignment import Alignment, Mistake
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.Recording import Recording


Edit = tuple[list[Note], list[Note], float]


class MistakeChecker:
    """Repair segmentation errors when doing so lowers string-edit cost."""

    COST_EPSILON = 1e-9

    def __init__(
        self, recording: Recording=None, config: Config=None, verbose: bool=False):
        self.recording = recording
        self.config = recording.config if recording else config
        self.pd = self.nd = self.alignment = None
        self.verbose = verbose

    def update_config(self, config: Config):
        self.config = config

    def check_mistakes(self, recording: Recording=None, verbose: bool=False) -> tuple[NoteData, Alignment]:
        """Apply correction passes while total string-edit cost decreases."""
        if recording is not None:
            self.recording = recording
            self.config = recording.config
        rec = self.recording
        if rec is None or rec.alignment is None:
            return None, None

        old_verbose = self.verbose
        if verbose is None:
            verbose = self.verbose

        # Cached Config values may predate the current score fit. Correction
        # segmentation must always use the current score-note durations.
        rec.update_min_note_length()
        self.pd = rec.pitch_data
        note_data = rec.note_data
        alignment = rec.alignment
        detector = rec.mistake_detector
        current_cost = detector.get_alignment_cost(alignment)
        notes_changed = False

        if verbose:
            print(f"initial alignment cost: {current_cost:.3f}")

        # the essential loop
        while True:
            new_data, new_alignment, edit_count = self._check_mistakes(
                note_data,
                alignment,
            )
            new_cost = detector.get_alignment_cost(new_alignment)
            if verbose:
                print(
                    f" > {edit_count} edit(s), alignment cost: "
                    f"{new_cost:.3f}"
                )

            if (edit_count == 0) or (new_cost >= current_cost - self.COST_EPSILON):
                break
            
            # update current state for the next iteration
            note_data = new_data
            alignment = new_alignment
            current_cost = new_cost
            notes_changed = True

        # update recording with final note_data and mistakes
        rec.note_data = note_data
        rec.alignment = alignment
        if notes_changed:
            rec.reindex_mistakes()
            rec.recompute_vibrato(note_aware=True)
        return note_data, alignment

    def _check_mistakes(self, note_data: NoteData, alignment: Alignment) -> tuple[NoteData, Alignment, int]:
        """Build one non-overlapping split/merge pass, then realign it."""
        self.nd = note_data
        self.alignment = alignment

        def consecutive_groups(mistakes: list[Mistake]) -> list[list[Mistake]]:
            groups: list[list[Mistake]] = []
            for mistake in sorted(mistakes, key=lambda item: item.pair_index):
                if (
                    groups
                    and mistake.pair_index == groups[-1][-1].pair_index + 1
                ):
                    groups[-1].append(mistake)
                else:
                    groups.append([mistake])
            return groups

        pitch_mistakes = alignment.pitch_mistakes
        deletion_groups = consecutive_groups(
            [m for m in pitch_mistakes if m.type == "deletion"]
        )
        insertion_groups = consecutive_groups(
            [m for m in pitch_mistakes if m.type == "insertion"]
        )

        proposals: list[tuple[Edit, list[Mistake]]] = []
        work = [
            *[(self.handle_deletion, group) for group in deletion_groups],
            *[(self.handle_insertion, group) for group in insertion_groups],
        ]
        for handler, group in work:
            edit = handler(group)
            if edit is not None:
                proposals.append((edit, group))
            elif self.verbose:
                print(
                    f"  [check_mistakes] no {group[0].type} edit "
                    f"for pairs {[mistake.pair_index for mistake in group]}"
                )

        # When two otherwise-valid proposals need the same source note, keep the
        # one with the larger reduction in the detector's edit cost.
        proposals.sort(key=lambda item: item[0][2], reverse=True)
        edits: list[Edit] = []
        edited_notes: set[int] = set()
        for edit, group in proposals:
            removed, added, saving = edit
            removed_ids = {id(note) for note in removed}
            if removed_ids & edited_notes:
                if self.verbose:
                    print(
                        "  [check_mistakes] skipped lower-saving overlapping "
                        f"{group[0].type} edit"
                    )
                continue
            edits.append(edit)
            edited_notes.update(removed_ids)
            if self.verbose:
                print(
                    f"  [check_mistakes] {group[0].type} pairs "
                    f"{[mistake.pair_index for mistake in group]} "
                    f"(saves {saving:.3f}): remove "
                    f"{[round(n.start_time, 2) for n in removed]} -> add "
                    f"{[round(n.start_time, 2) for n in added]}"
                )

        if not edits:
            return note_data, alignment, 0

        removed_ids = {
            id(note)
            for removed, _, _ in edits
            for note in removed
        }
        notes = [
            copy(note)
            for note in note_data.data.values()
            if id(note) not in removed_ids
        ]
        for _, replacements, _ in edits:
            notes.extend(replacements)
        notes.sort(key=lambda note: note.start_time)

        new_data = NoteData()
        for note_id, note in enumerate(notes):
            note.id = note_id
            new_data.write_note(note)

        score_notes = self.recording.score_data.clipped_note_data(
            channel=self.recording.active_instrument
        )
        new_alignment = self.recording.mistake_detector.detect_mistakes(
            user_notes=new_data,
            score_notes=score_notes,
            verbose=self.verbose,
        )
        return new_data, new_alignment, len(edits)

    def handle_deletion(self, mistakes: Mistake | list[Mistake]) -> Edit | None:
        """Recover deletions from neighbors, repeated runs, or edge pitch."""
        mistakes = [mistakes] if isinstance(mistakes, Mistake) else mistakes
        mistakes = sorted(mistakes, key=lambda mistake: mistake.pair_index)
        deleted = [
            mistake.midi_note
            for mistake in mistakes
            if mistake.midi_note is not None
        ]
        if self.pd is None or not deleted:
            return None

        detector = self.recording.mistake_detector
        pairs = self.alignment.pairs
        first_index = mistakes[0].pair_index
        last_index = mistakes[-1].pair_index
        previous = pairs[first_index - 1] if first_index > 0 else (None, None)
        following = (
            pairs[last_index + 1]
            if last_index + 1 < len(pairs)
            else (None, None)
        )
        partition_cache: dict[tuple, list[Note] | None] = {}

        def region_for(
            hosts: list[Note],
            region_bounds: tuple[float, float] | None = None,
        ) -> Note:
            template = hosts[0]
            start_time, end_time = region_bounds or (
                min(host.start_time for host in hosts),
                max(host.end_time for host in hosts),
            )
            return Note(
                i=-1,
                start_time=start_time,
                end_time=end_time,
                midi_num=[template.midi_num[0]],
                velocity=template.velocity,
                instrument=template.instrument,
            )

        def split_option(hosts: list[Note], anchors: list[tuple[Note, Note]], 
                         score_targets: list[Note], recovered_count: int, 
                         label: str, region_bounds: tuple[float, float]=None,
        ) -> dict | None:
            segment_count = len(anchors) + recovered_count
            region = region_for(hosts, region_bounds)
            cache_key = (
                "kernel",
                tuple(id(host) for host in hosts),
                segment_count,
                region.start_time,
                region.end_time,
            )
            if cache_key not in partition_cache:
                partition_cache[cache_key] = self._partition_region(
                    region,
                    segment_count,
                )
            split = partition_cache[cache_key]
            if split is None:
                return None

            old_cost = (
                sum(
                    detector.get_substitution_cost(user_note, score_note)
                    for user_note, score_note in anchors
                )
                + sum(
                    detector.get_deletion_cost(target)
                    for target in deleted
                )
            )
            local_alignment = detector.get_string_edit_alignment(
                split, score_targets,
            )
            paired_scores = {
                id(score_note)
                for user_note, score_note in local_alignment.pairs
                if user_note is not None and score_note is not None
            }
            if (any(id(score_note) not in paired_scores for _, score_note in anchors) 
                or any(u and not s for u, s in local_alignment.pairs)):
                return None
            new_cost = detector.get_alignment_cost(local_alignment)
            saving = old_cost - new_cost
            if self.verbose:
                print(
                    f"      [split?] {label}, recover {recovered_count}/"
                    f"{len(deleted)} deletion(s): "
                    f"{old_cost:.3f} -> {new_cost:.3f}"
                )
            if saving <= self.COST_EPSILON:
                return None
            return {
                "removed": hosts,
                "added": split,
                "saving": saving,
            }

        def repeated_split_option(hosts: list[Note], anchors: list[tuple[Note, Note]], 
            score_targets: list[Note], label: str, region_bounds: tuple[float, float]=None,
        ) -> dict | None:
            """Recover every deletion using score-timed repeated-note splits."""
            has_repeat = any(
                left.midi_num[0] == right.midi_num[0]
                for left, right in zip(score_targets, score_targets[1:])
            )
            if not has_repeat:
                return None

            region = region_for(hosts, region_bounds)
            cache_key = (
                "repeat",
                tuple(id(host) for host in hosts),
                tuple(id(target) for target in score_targets),
                region.start_time,
                region.end_time,
            )
            if cache_key not in partition_cache:
                partition_cache[cache_key] = self._partition_repeated_region(
                    region,
                    score_targets,
                )
            split = partition_cache[cache_key]
            if split is None:
                return None

            old_cost = (
                sum(
                    detector.get_substitution_cost(user_note, score_note)
                    for user_note, score_note in anchors
                )
                + sum(
                    detector.get_deletion_cost(target)
                    for target in deleted
                )
            )
            local_alignment = detector.get_string_edit_alignment(
                split,
                score_targets,
            )
            # This path constructs one segment for every target. If string edit
            # would still prefer a gap operation, it is not a valid recovery.
            if any(
                user_note is None or score_note is None
                for user_note, score_note in local_alignment.pairs
            ):
                return None

            new_cost = detector.get_alignment_cost(local_alignment)
            saving = old_cost - new_cost
            if self.verbose:
                print(
                    f"      [repeat split?] {label}, recover "
                    f"{len(deleted)}/{len(deleted)} deletion(s): "
                    f"{old_cost:.3f} -> {new_cost:.3f}"
                )
            if saving <= self.COST_EPSILON:
                return None
            return {
                "removed": hosts,
                "added": split,
                "saving": saving,
            }

        previous_user, previous_score = previous
        following_user, following_score = following
        options = []
        if previous_user is not None and previous_score is not None:
            max_recovered = min(
                len(deleted),
                self._segment_capacity([previous_user]) - 1,
            )
            for recovered_count in range(1, max_recovered + 1):
                option = split_option(
                    [previous_user],
                    [(previous_user, previous_score)],
                    [previous_score, *deleted],
                    recovered_count,
                    "previous",
                )
                if option is not None:
                    options.append(option)
            option = repeated_split_option(
                [previous_user],
                [(previous_user, previous_score)],
                [previous_score, *deleted],
                "previous",
            )
            if option is not None:
                options.append(option)

        if following_user is not None and following_score is not None:
            max_recovered = min(
                len(deleted),
                self._segment_capacity([following_user]) - 1,
            )
            for recovered_count in range(1, max_recovered + 1):
                option = split_option(
                    [following_user],
                    [(following_user, following_score)],
                    [*deleted, following_score],
                    recovered_count,
                    "following",
                )
                if option is not None:
                    options.append(option)
            option = repeated_split_option(
                [following_user],
                [(following_user, following_score)],
                [*deleted, following_score],
                "following",
            )
            if option is not None:
                options.append(option)

        if (
            previous_user is not None
            and previous_score is not None
            and following_user is not None
            and following_score is not None
        ):
            # A short played note can be discarded by initial segmentation and
            # survive only as pitch frames in the gap between its neighbors.
            max_recovered = min(
                len(deleted),
                self._segment_capacity(
                    [previous_user, following_user],
                )
                - 2,
            )
            for recovered_count in range(1, max_recovered + 1):
                option = split_option(
                    [previous_user, following_user],
                    [
                        (previous_user, previous_score),
                        (following_user, following_score),
                    ],
                    [previous_score, *deleted, following_score],
                    recovered_count,
                    "previous+following",
                )
                if option is not None:
                    options.append(option)
            option = repeated_split_option(
                [previous_user, following_user],
                [
                    (previous_user, previous_score),
                    (following_user, following_score),
                ],
                [previous_score, *deleted, following_score],
                "previous+following",
            )
            if option is not None:
                options.append(option)

        # Repeated score pitches contain no pitch changepoint. Extend only
        # through the immediately adjacent equal-pitch score run, then use
        # KernelCPD between distinct pitch blocks and score-proportional timing
        # within each repeated block.
        if previous_user is not None and previous_score is not None:
            previous_run = [previous]
            index = first_index - 2
            while index >= 0:
                user_note, score_note = pairs[index]
                if (
                    user_note is None
                    or score_note is None
                    or score_note.midi_num[0] != previous_score.midi_num[0]
                ):
                    break
                previous_run.append((user_note, score_note))
                index -= 1
            previous_run.reverse()
            if len(previous_run) > 1:
                option = repeated_split_option(
                    [user_note for user_note, _ in previous_run],
                    previous_run,
                    [
                        *[score_note for _, score_note in previous_run],
                        *deleted,
                    ],
                    "previous repeated run",
                )
                if option is not None:
                    options.append(option)

        if following_user is not None and following_score is not None:
            following_run = [following]
            index = last_index + 2
            while index < len(pairs):
                user_note, score_note = pairs[index]
                if (
                    user_note is None
                    or score_note is None
                    or score_note.midi_num[0] != following_score.midi_num[0]
                ):
                    break
                following_run.append((user_note, score_note))
                index += 1
            if len(following_run) > 1:
                option = repeated_split_option(
                    [user_note for user_note, _ in following_run],
                    following_run,
                    [
                        *deleted,
                        *[score_note for _, score_note in following_run],
                    ],
                    "following repeated run",
                )
                if option is not None:
                    options.append(option)

        # Initial note detection intentionally drops voiced runs shorter than
        # its minimum note length. At score edges there is no outer detected
        # neighbor, so include otherwise-unassigned edge pitch in the one
        # available neighbor's correction region.
        voiced_bounds = self.pd.get_voiced_range()
        if (
            voiced_bounds is not None
            and first_index == 0
            and following_user is not None
            and following_score is not None
            and voiced_bounds[0] < following_user.start_time
        ):
            edge_bounds = (voiced_bounds[0], following_user.end_time)
            max_recovered = min(
                len(deleted),
                self._segment_capacity(
                    [following_user],
                    region_bounds=edge_bounds,
                )
                - 1,
            )
            for recovered_count in range(1, max_recovered + 1):
                option = split_option(
                    [following_user],
                    [(following_user, following_score)],
                    [*deleted, following_score],
                    recovered_count,
                    "leading pitch+following",
                    region_bounds=edge_bounds,
                )
                if option is not None:
                    options.append(option)
            option = repeated_split_option(
                [following_user],
                [(following_user, following_score)],
                [*deleted, following_score],
                "leading pitch+following",
                region_bounds=edge_bounds,
            )
            if option is not None:
                options.append(option)

        if (
            voiced_bounds is not None
            and last_index == len(pairs) - 1
            and previous_user is not None
            and previous_score is not None
            and voiced_bounds[1] > previous_user.end_time
        ):
            edge_bounds = (previous_user.start_time, voiced_bounds[1])
            max_recovered = min(
                len(deleted),
                self._segment_capacity(
                    [previous_user],
                    region_bounds=edge_bounds,
                )
                - 1,
            )
            for recovered_count in range(1, max_recovered + 1):
                option = split_option(
                    [previous_user],
                    [(previous_user, previous_score)],
                    [previous_score, *deleted],
                    recovered_count,
                    "previous+trailing pitch",
                    region_bounds=edge_bounds,
                )
                if option is not None:
                    options.append(option)
            option = repeated_split_option(
                [previous_user],
                [(previous_user, previous_score)],
                [previous_score, *deleted],
                "previous+trailing pitch",
                region_bounds=edge_bounds,
            )
            if option is not None:
                options.append(option)

        if not options:
            return None

        selected = max(options, key=lambda option: option["saving"])
        return (
            selected["removed"],
            selected["added"],
            selected["saving"],
        )

    def _segment_capacity(self, notes: list[Note], 
                          region_bounds: tuple[float, float]=None) -> int:
        """Maximum fixed-count KernelCPD segments supported by a region."""
        if not notes or self.pd is None:
            return 0
        start_time, end_time = region_bounds or (
            min(note.start_time for note in notes),
            max(note.end_time for note in notes),
        )
        frames = self.pd.read(
            start_time=start_time,
            end_time=end_time,
            clean=True,
        )
        min_frames = self.config.min_note_pitch_frames(
            factor=self.config.min_note_length_factor,
        )
        return len(frames) // min_frames

    def _crosses_blocking_silence(
        self,
        source_notes: list[Note],
        replacements: list[Note],
    ) -> bool:
        """Whether a replacement joins notes across a note-length silence.

        Measure decoded silence from the pitch frames themselves. Note endpoints
        are only a fallback: their frame-centre convention can move a borderline
        gap by several milliseconds, and the score-relative threshold can move
        by a similar amount while the score fit stabilizes.
        """
        ordered = sorted(source_notes, key=lambda note: note.start_time)
        minimum_silence = self.config.min_note_seconds(
            factor=self.config.min_note_length_factor,
        )
        frame_duration = self.config.h1 / self.config.sr
        # Silence is decoded through a finite majority window, so durations
        # closer than that window are indistinguishable at the detector's own
        # temporal resolution. This also prevents a few milliseconds of tempo
        # refitting from changing whether the same physical silence is crossed.
        silence_resolution = max(
            frame_duration,
            self.config.min_silence_duration_ms / 1000.0,
        )
        pitch_data = self.pd

        def decoded_silence_duration(left: Note, right: Note) -> float:
            if pitch_data is not None:
                # Note boundaries and Pitch.time are frame-centre timestamps.
                # pitch_curve() uses that same coordinate system; read() is
                # intentionally frame-start-indexed for streaming writes.
                _, values = pitch_data.pitch_curve(
                    left.end_time,
                    right.start_time,
                )
                longest_run = current_run = 0
                for is_voiced in np.isfinite(values):
                    if is_voiced:
                        current_run = 0
                    else:
                        current_run += 1
                        longest_run = max(longest_run, current_run)
                if longest_run:
                    return longest_run * frame_duration

            # NoteDetector stores the final voiced frame's time as a run's end.
            # Include that frame interval in the endpoint-only fallback.
            return max(
                0.0,
                right.start_time - left.end_time + frame_duration,
            )

        for left, right in zip(ordered, ordered[1:]):
            gap_duration = decoded_silence_duration(left, right)
            if (
                gap_duration + silence_resolution
                < minimum_silence - self.COST_EPSILON
            ):
                continue
            if any(
                replacement.start_time <= left.end_time + self.COST_EPSILON
                and replacement.end_time >= right.start_time - self.COST_EPSILON
                for replacement in replacements
            ):
                return True
        return False

    def _partition_region(
        self,
        note: Note,
        segment_count: int,
    ) -> list[Note] | None:
        """Split one note with fixed-count production KernelCPD segmentation."""
        pitches = self.pd.read(
            start_time=note.start_time,
            end_time=note.end_time,
            clean=True,
        )
        if not pitches:
            return None

        signal = np.asarray(
            [pitch.value for pitch in pitches],
            dtype=float,
        ).reshape(-1, 1)
        breakpoints = self.recording.note_detector.segment_breakpoints(
            signal,
            segment_count=segment_count,
        )
        if breakpoints is None:
            return None

        indices = [0, *breakpoints]
        boundaries = [note.start_time]
        boundaries.extend(
            self.recording.note_detector.get_boundary_time(pitches, index)
            for index in breakpoints[:-1]
        )
        boundaries.append(note.end_time)

        split = []
        for start, end, start_time, end_time in zip(
            indices,
            indices[1:],
            boundaries,
            boundaries[1:],
        ):
            if end <= start or end_time <= start_time:
                return None
            split.append(
                Note(
                    i=-1,
                    start_time=float(start_time),
                    end_time=float(end_time),
                    midi_num=[float(np.median(signal[start:end, 0]))],
                    velocity=note.velocity,
                    instrument=note.instrument,
                )
            )
        return split if len(split) == segment_count else None

    def _partition_repeated_region(
        self,
        note: Note,
        score_targets: list[Note],
    ) -> list[Note] | None:
        """Use pitch changes between blocks and score timing within repeats."""
        if len(score_targets) < 2:
            return None

        score_blocks: list[list[Note]] = []
        for target in score_targets:
            if (
                score_blocks
                and target.midi_num[0] == score_blocks[-1][-1].midi_num[0]
            ):
                score_blocks[-1].append(target)
            else:
                score_blocks.append([target])
        if all(len(block) == 1 for block in score_blocks):
            return None

        pitches = self.pd.read(
            start_time=note.start_time,
            end_time=note.end_time,
            clean=True,
        )
        if not pitches:
            return None

        signal = np.asarray(
            [pitch.value for pitch in pitches],
            dtype=float,
        ).reshape(-1, 1)
        if len(score_blocks) == 1:
            block_indices = [0, len(pitches)]
            block_boundaries = [note.start_time, note.end_time]
        else:
            breakpoints = self.recording.note_detector.segment_breakpoints(
                signal,
                segment_count=len(score_blocks),
            )
            if breakpoints is None:
                return None
            block_indices = [0, *breakpoints]
            block_boundaries = [note.start_time]
            block_boundaries.extend(
                self.recording.note_detector.get_boundary_time(pitches, index)
                for index in breakpoints[:-1]
            )
            block_boundaries.append(note.end_time)

        min_duration = self.config.min_note_seconds(
            factor=self.config.min_note_length_factor,
        )
        split: list[Note] = []
        for block_index, (
            targets,
            frame_start,
            frame_end,
            block_start,
            block_end,
        ) in enumerate(
            zip(
                score_blocks,
                block_indices,
                block_indices[1:],
                block_boundaries,
                block_boundaries[1:],
            )
        ):
            target_durations = [target.duration() for target in targets]
            target_total = sum(target_durations)
            if target_total <= 0 or block_end <= block_start:
                return None

            boundaries = [block_start]
            elapsed = 0.0
            for duration in target_durations[:-1]:
                elapsed += duration
                boundaries.append(
                    block_start
                    + (block_end - block_start) * elapsed / target_total
                )
            boundaries.append(block_end)

            block_pitches = pitches[frame_start:frame_end]
            for target_index, (start_time, end_time) in enumerate(
                zip(boundaries, boundaries[1:])
            ):
                if end_time - start_time < min_duration - self.COST_EPSILON:
                    return None
                segment_pitches = [
                    pitch
                    for pitch in block_pitches
                    if pitch.time >= start_time
                    and (
                        pitch.time < end_time
                        or (
                            block_index == len(score_blocks) - 1
                            and target_index == len(targets) - 1
                            and pitch.time <= end_time
                        )
                    )
                ]
                if not segment_pitches:
                    return None
                split.append(
                    Note(
                        i=-1,
                        start_time=float(start_time),
                        end_time=float(end_time),
                        midi_num=[
                            float(np.median(
                                [pitch.value for pitch in segment_pitches]
                            ))
                        ],
                        velocity=note.velocity,
                        instrument=note.instrument,
                    )
                )

        return split if len(split) == len(score_targets) else None

    def handle_insertion(
        self,
        mistakes: Mistake | list[Mistake],
    ) -> Edit | None:
        """Resegment insertion groups while retaining any supported note count."""
        mistakes = [mistakes] if isinstance(mistakes, Mistake) else mistakes
        mistakes = sorted(mistakes, key=lambda mistake: mistake.pair_index)
        inserted = [
            mistake.user_note
            for mistake in mistakes
            if mistake.user_note is not None
        ]
        if self.pd is None or not inserted:
            return None

        detector = self.recording.mistake_detector
        pairs = self.alignment.pairs
        first_index = mistakes[0].pair_index
        last_index = mistakes[-1].pair_index
        previous = pairs[first_index - 1] if first_index > 0 else (None, None)
        following = (
            pairs[last_index + 1]
            if last_index + 1 < len(pairs)
            else (None, None)
        )
        partition_cache: dict[
            tuple[tuple[int, ...], int],
            list[Note] | None,
        ] = {}

        def resegment_option(
            hosts: list[Note],
            anchors: list[tuple[Note, Note]],
            score_targets: list[Note],
            retained_count: int,
            label: str,
        ) -> dict | None:
            segment_count = len(score_targets) + retained_count
            cache_key = (
                tuple(id(host) for host in hosts),
                segment_count,
            )
            if cache_key not in partition_cache:
                template = hosts[0]
                region = Note(
                    i=-1,
                    start_time=min(host.start_time for host in hosts),
                    end_time=max(host.end_time for host in hosts),
                    midi_num=[template.midi_num[0]],
                    velocity=template.velocity,
                    instrument=template.instrument,
                )
                partition_cache[cache_key] = self._partition_region(
                    region,
                    segment_count,
                )
            split = partition_cache[cache_key]
            if split is None:
                return None
            if self._crosses_blocking_silence(hosts, split):
                if self.verbose:
                    print(
                        f"      [merge?] {label}: rejected across "
                        "minimum-note-length silence"
                    )
                return None

            old_cost = (
                sum(
                    detector.get_substitution_cost(user_note, score_note)
                    for user_note, score_note in anchors
                )
                + sum(
                    detector.get_insertion_cost(note) for note in inserted
                )
            )
            local_alignment = detector.get_string_edit_alignment(
                split,
                score_targets,
            )
            # This handler may retain insertions, but it must not manufacture a
            # score deletion while trying to remove over-segmentation.
            if any(user_note is None for user_note, _ in local_alignment.pairs):
                return None
            new_cost = detector.get_alignment_cost(local_alignment)
            saving = old_cost - new_cost
            if self.verbose:
                print(
                    f"      [merge?] {label}, retain {retained_count}/"
                    f"{len(inserted)} insertion(s): "
                    f"{old_cost:.3f} -> {new_cost:.3f}"
                )
            if saving <= self.COST_EPSILON:
                return None
            return {
                "removed": hosts,
                "added": split,
                "saving": saving,
            }

        previous_user, previous_score = previous
        following_user, following_score = following
        options = []
        if previous_user is not None and previous_score is not None:
            hosts = [previous_user, *inserted]
            max_retained = min(
                len(inserted) - 1,
                self._segment_capacity(hosts) - 1,
            )
            for retained_count in range(max_retained + 1):
                option = resegment_option(
                    hosts,
                    [(previous_user, previous_score)],
                    [previous_score],
                    retained_count,
                    "previous",
                )
                if option is not None:
                    options.append(option)

        if following_user is not None and following_score is not None:
            hosts = [*inserted, following_user]
            max_retained = min(
                len(inserted) - 1,
                self._segment_capacity(hosts) - 1,
            )
            for retained_count in range(max_retained + 1):
                option = resegment_option(
                    hosts,
                    [(following_user, following_score)],
                    [following_score],
                    retained_count,
                    "following",
                )
                if option is not None:
                    options.append(option)

        if (
            previous_user is not None
            and previous_score is not None
            and following_user is not None
            and following_score is not None
        ):
            hosts = [previous_user, *inserted, following_user]
            max_retained = min(
                len(inserted) - 1,
                self._segment_capacity(hosts) - 2,
            )
            for retained_count in range(max_retained + 1):
                option = resegment_option(
                    hosts,
                    [
                        (previous_user, previous_score),
                        (following_user, following_score),
                    ],
                    [previous_score, following_score],
                    retained_count,
                    "previous+following",
                )
                if option is not None:
                    options.append(option)

        if not options:
            return None

        selected = max(options, key=lambda option: option["saving"])
        return (
            selected["removed"],
            selected["added"],
            selected["saving"],
        )
