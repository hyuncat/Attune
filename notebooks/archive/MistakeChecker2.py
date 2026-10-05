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
        self,
        recording: Recording = None,
        config: Config = None,
        verbose: bool = False,
    ):
        self.recording = recording
        self.config = recording.config if recording else config
        self.pd = self.nd = self.alignment = None
        self.verbose = verbose

    def update_config(self, config: Config):
        self.config = config

    def check_mistakes(
        self,
        recording: Recording = None,
        verbose: bool | None = None,
    ) -> tuple[NoteData | None, Alignment | None]:
        """Apply correction passes while total string-edit cost decreases."""
        if recording is not None:
            self.recording = recording
            self.config = recording.config
        rec = self.recording
        if rec is None or rec.alignment is None:
            return None, None

        if verbose is not None:
            self.verbose = verbose

        # Cached Config values may predate the current score fit. Correction
        # segmentation must always use the current score-note durations.
        rec.update_min_note_length()
        self.pd = rec.pitch_data
        note_data = rec.note_data
        alignment = rec.alignment
        detector = rec.mistake_detector
        current_cost = detector.get_alignment_cost(alignment)
        notes_changed = False

        if self.verbose:
            print(f"initial alignment cost: {current_cost:.3f}")

        while True:
            new_data, new_alignment, edit_count = self._check_mistakes(
                note_data,
                alignment,
            )
            new_cost = detector.get_alignment_cost(new_alignment)
            if self.verbose:
                print(
                    f" > {edit_count} edit(s), alignment cost: "
                    f"{new_cost:.3f}"
                )

            if (
                edit_count == 0
                or new_cost >= current_cost - self.COST_EPSILON
            ):
                break

            note_data = new_data
            alignment = new_alignment
            current_cost = new_cost
            notes_changed = True

        rec.note_data = note_data
        rec.alignment = alignment
        if notes_changed:
            rec.reindex_mistakes()
            rec.recompute_vibrato(note_aware=True)
        return note_data, alignment

    def _check_mistakes(
        self,
        note_data: NoteData,
        alignment: Alignment,
    ) -> tuple[NoteData, Alignment, int]:
        """Build one non-overlapping split/merge pass, then realign it."""
        self.nd = note_data
        self.alignment = alignment

        def consecutive_groups(
            mistakes: list[Mistake],
        ) -> list[list[Mistake]]:
            groups: list[list[Mistake]] = []
            for mistake in sorted(
                mistakes,
                key=lambda item: item.pair_index,
            ):
                if (
                    groups
                    and mistake.pair_index
                    == groups[-1][-1].pair_index + 1
                ):
                    groups[-1].append(mistake)
                else:
                    groups.append([mistake])
            return groups

        pitch_mistakes = alignment.pitch_mistakes
        deletion_groups = consecutive_groups(
            [
                mistake
                for mistake in pitch_mistakes
                if mistake.type == "deletion"
            ]
        )
        insertion_groups = consecutive_groups(
            [
                mistake
                for mistake in pitch_mistakes
                if mistake.type == "insertion"
            ]
        )

        proposals: list[tuple[Edit, list[Mistake]]] = []
        work = [
            *[
                (self.handle_deletion, group)
                for group in deletion_groups
            ],
            *[
                (self.handle_insertion, group)
                for group in insertion_groups
            ],
        ]
        for handler, group in work:
            edit = handler(group)
            if edit is not None:
                proposals.append((edit, group))
            elif self.verbose:
                print(
                    f"  [check_mistakes] no {group[0].type} edit "
                    f"for pairs "
                    f"{[mistake.pair_index for mistake in group]}"
                )

        # When two proposals need the same source note, keep the one with the
        # greater reduction in the detector's edit cost.
        proposals.sort(key=lambda item: item[0][2], reverse=True)
        edits: list[Edit] = []
        edited_notes: set[int] = set()
        for edit, group in proposals:
            removed, added, saving = edit
            removed_ids = {id(note) for note in removed}
            if removed_ids & edited_notes:
                if self.verbose:
                    print(
                        "  [check_mistakes] skipped lower-saving "
                        f"overlapping {group[0].type} edit"
                    )
                continue
            edits.append(edit)
            edited_notes.update(removed_ids)
            if self.verbose:
                print(
                    f"  [check_mistakes] {group[0].type} pairs "
                    f"{[mistake.pair_index for mistake in group]} "
                    f"(saves {saving:.3f}): remove "
                    f"{[round(note.start_time, 2) for note in removed]} "
                    f"-> add "
                    f"{[round(note.start_time, 2) for note in added]}"
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

    def _matched_window(self, mistakes: list[Mistake]) -> list[tuple[Note, Note]] | None:
        """Alignment rows from the nearest matched L through matched R."""
        pairs = self.alignment.pairs
        left_index = mistakes[0].pair_index - 1
        while left_index >= 0:
            user_note, score_note = pairs[left_index]
            if user_note is not None and score_note is not None:
                break
            left_index -= 1

        right_index = mistakes[-1].pair_index + 1
        while right_index < len(pairs):
            user_note, score_note = pairs[right_index]
            if user_note is not None and score_note is not None:
                break
            right_index += 1

        if left_index < 0 or right_index >= len(pairs):
            return None
        return pairs[left_index:right_index + 1]

    def _deletion_window(self, mistakes: list[Mistake]) -> tuple[
        list[tuple[Note | None, Note | None]],
        bool,
        bool,
    ] | None:
        """Deletion window plus flags for its matched left/right boundaries."""
        pairs = self.alignment.pairs
        left_index = mistakes[0].pair_index - 1
        while left_index >= 0:
            user_note, score_note = pairs[left_index]
            if user_note is not None and score_note is not None:
                break
            left_index -= 1

        right_index = mistakes[-1].pair_index + 1
        while right_index < len(pairs):
            user_note, score_note = pairs[right_index]
            if user_note is not None and score_note is not None:
                break
            right_index += 1

        has_left = left_index >= 0
        has_right = right_index < len(pairs)
        if not has_left and not has_right:
            return None

        # At an alignment boundary, include every unmatched edge row. Its
        # acoustic boundary comes from the first/last voiced pitch below.
        window_start = left_index if has_left else 0
        window_end = right_index + 1 if has_right else len(pairs)
        return pairs[window_start:window_end], has_left, has_right

    @staticmethod
    def _window_parts(window: list[tuple[Note | None, Note | None]]) -> tuple[
        list[Note],
        list[Note],
        list[tuple[Note, Note]],
        list[Note],
        list[Note],
    ]:
        hosts = [
            user_note
            for user_note, _ in window
            if user_note is not None
        ]
        score_targets = [
            score_note
            for _, score_note in window
            if score_note is not None
        ]
        anchors = [
            (user_note, score_note)
            for user_note, score_note in window
            if user_note is not None and score_note is not None
        ]
        deleted = [
            score_note
            for user_note, score_note in window
            if user_note is None and score_note is not None
        ]
        inserted = [
            user_note
            for user_note, score_note in window
            if user_note is not None and score_note is None
        ]
        return hosts, score_targets, anchors, deleted, inserted

    @staticmethod
    def _region_for(hosts: list[Note]) -> Note:
        template = hosts[0]
        return Note(
            i=-1,
            start_time=min(note.start_time for note in hosts),
            end_time=max(note.end_time for note in hosts),
            midi_num=[template.midi_num[0]],
            velocity=template.velocity,
            instrument=template.instrument,
        )

    def handle_deletion(self, mistakes: Mistake | list[Mistake]) -> Edit | None:
        """Resegment a matched L-to-R or voiced-edge deletion region."""
        mistakes = (
            [mistakes]
            if isinstance(mistakes, Mistake)
            else mistakes
        )
        mistakes = sorted(
            mistakes,
            key=lambda mistake: mistake.pair_index,
        )
        window_info = self._deletion_window(mistakes)
        if self.pd is None or window_info is None:
            return None
        window, has_left, has_right = window_info

        (
            hosts,
            score_targets,
            anchors,
            deleted,
            _,
        ) = self._window_parts(window)
        minimum_hosts = 2 if has_left and has_right else 1
        if (
            len(hosts) < minimum_hosts
            or not score_targets
            or not deleted
        ):
            return None

        detector = self.recording.mistake_detector
        region = self._region_for(hosts)
        voiced_bounds = self.pd.get_voiced_range(
            include_transitions=False,
        )
        if voiced_bounds is None:
            return None
        if not has_left:
            region.start_time = min(
                region.start_time,
                voiced_bounds[0],
            )
        if not has_right:
            region.end_time = max(
                region.end_time,
                voiced_bounds[1],
            )
        if region.end_time <= region.start_time:
            return None

        old_cost = detector.get_alignment_cost(window)
        options: list[dict] = []
        region_label = (
            "L-to-R"
            if has_left and has_right
            else "voiced-start-to-R"
            if has_right
            else "L-to-voiced-end"
        )

        def add_option(
            split: list[Note] | None,
            label: str,
        ) -> None:
            if split is None:
                return
            local_alignment = detector.get_string_edit_alignment(
                split,
                score_targets,
            )
            paired_scores = {
                id(score_note)
                for user_note, score_note in local_alignment.pairs
                if user_note is not None and score_note is not None
            }
            if any(
                id(score_note) not in paired_scores
                for _, score_note in anchors
            ):
                return
            if any(
                user_note is not None and score_note is None
                for user_note, score_note in local_alignment.pairs
            ):
                return

            new_cost = detector.get_alignment_cost(local_alignment)
            saving = old_cost - new_cost
            if self.verbose:
                print(
                    f"      [split?] {region_label} {label}: "
                    f"{old_cost:.3f} -> {new_cost:.3f}"
                )
            if saving > self.COST_EPSILON:
                options.append(
                    {
                        "removed": hosts,
                        "added": split,
                        "saving": saving,
                    }
                )

        capacity = self._segment_capacity([region])
        maximum_recovered = min(
            len(deleted),
            capacity - len(hosts),
        )
        for recovered_count in range(1, maximum_recovered + 1):
            add_option(
                self._partition_region(
                    region,
                    len(hosts) + recovered_count,
                ),
                f"recover {recovered_count}/{len(deleted)}",
            )

        # Equal-pitch repetitions contain no acoustic changepoint. KernelCPD
        # locates the distinct-pitch blocks, and score-duration proportions
        # place boundaries within each repeated block.
        add_option(
            self._partition_repeated_region(region, score_targets),
            "score-timed repeated notes",
        )

        if not options:
            return None
        selected = max(options, key=lambda option: option["saving"])
        return (
            selected["removed"],
            selected["added"],
            selected["saving"],
        )

    def handle_insertion(self,mistakes: Mistake | list[Mistake]) -> Edit | None:
        """Resegment only the complete matched-L through matched-R region."""
        mistakes = (
            [mistakes]
            if isinstance(mistakes, Mistake)
            else mistakes
        )
        mistakes = sorted(
            mistakes,
            key=lambda mistake: mistake.pair_index,
        )
        window = self._matched_window(mistakes)
        if self.pd is None or window is None:
            return None

        (
            hosts,
            score_targets,
            _,
            _,
            inserted,
        ) = self._window_parts(window)
        if len(hosts) < 3 or len(score_targets) < 2 or not inserted:
            return None

        detector = self.recording.mistake_detector
        region = self._region_for(hosts)
        old_cost = detector.get_alignment_cost(window)
        options: list[dict] = []
        maximum_retained = min(
            len(inserted) - 1,
            self._segment_capacity(hosts) - len(score_targets),
        )

        for retained_count in range(maximum_retained + 1):
            segment_count = len(score_targets) + retained_count
            split = self._partition_region(region, segment_count)
            if split is None:
                continue
            if self._crosses_blocking_silence(hosts, split):
                if self.verbose:
                    print(
                        "      [merge?] L-to-R rejected across "
                        "minimum-note-length silence"
                    )
                continue

            local_alignment = detector.get_string_edit_alignment(
                split,
                score_targets,
            )
            # Over-segmentation correction may retain insertions, but must not
            # create a new score deletion.
            if any(
                user_note is None
                for user_note, _ in local_alignment.pairs
            ):
                continue

            new_cost = detector.get_alignment_cost(local_alignment)
            saving = old_cost - new_cost
            if self.verbose:
                print(
                    f"      [merge?] L-to-R, retain "
                    f"{retained_count}/{len(inserted)} insertion(s): "
                    f"{old_cost:.3f} -> {new_cost:.3f}"
                )
            if saving > self.COST_EPSILON:
                options.append(
                    {
                        "removed": hosts,
                        "added": split,
                        "saving": saving,
                    }
                )

        if not options:
            return None
        selected = max(options, key=lambda option: option["saving"])
        return (
            selected["removed"],
            selected["added"],
            selected["saving"],
        )

    def _segment_capacity(self, notes: list[Note]) -> int:
        """Maximum fixed-count KernelCPD segments supported by a region."""
        if not notes or self.pd is None:
            return 0
        frames = self.pd.read(
            start_time=min(note.start_time for note in notes),
            end_time=max(note.end_time for note in notes),
            clean=True,
        )
        min_frames = max(
            1,
            int(np.ceil(
                self._correction_min_length()
                * self.config.sr
                / self.config.h1
            )),
        )
        return len(frames) // min_frames

    def _correction_min_length(self) -> float:
        """Correction minimum: detector threshold capped at 150 ms."""
        return min(
            max(
                0.0,
                float(self.config.mistake_correction_min_length),
            ),
            self.config.min_note_seconds(
                factor=self.config.min_note_length_factor,
            ),
        )

    #rk: check if this can be removed with something which feeds the segmenter with runs
    def _crosses_blocking_silence(
        self,
        source_notes: list[Note],
        replacements: list[Note],
    ) -> bool:
        """Whether a replacement joins notes across a note-length silence."""
        ordered = sorted(
            source_notes,
            key=lambda note: note.start_time,
        )
        minimum_silence = self._correction_min_length()
        frame_duration = self.config.h1 / self.config.sr
        silence_resolution = max(
            frame_duration,
            self.config.min_silence_duration_ms / 1000.0,
        )

        def decoded_silence_duration(
            left: Note,
            right: Note,
        ) -> float:
            if self.pd is not None:
                _, values = self.pd.pitch_curve(
                    left.end_time,
                    right.start_time,
                )
                longest_run = current_run = 0
                for is_voiced in np.isfinite(values):
                    if is_voiced:
                        current_run = 0
                    else:
                        current_run += 1
                        longest_run = max(
                            longest_run,
                            current_run,
                        )
                if longest_run:
                    return longest_run * frame_duration
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
                replacement.start_time
                <= left.end_time + self.COST_EPSILON
                and replacement.end_time
                >= right.start_time - self.COST_EPSILON
                for replacement in replacements
            ):
                return True
        return False

    def _partition_region(
        self,
        note: Note,
        segment_count: int,
    ) -> list[Note] | None:
        """Split one region with fixed-count production KernelCPD."""
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
            min_length_seconds=self._correction_min_length(),
        )
        if breakpoints is None:
            return None

        indices = [0, *breakpoints]
        boundaries = [note.start_time]
        boundaries.extend(
            self.recording.note_detector.get_boundary_time(
                pitches,
                index,
            )
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
                    midi_num=[
                        float(np.median(signal[start:end, 0]))
                    ],
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
                and target.midi_num[0]
                == score_blocks[-1][-1].midi_num[0]
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
            breakpoints = (
                self.recording.note_detector.segment_breakpoints(
                    signal,
                    segment_count=len(score_blocks),
                    min_length_seconds=self._correction_min_length(),
                )
            )
            if breakpoints is None:
                return None
            block_indices = [0, *breakpoints]
            block_boundaries = [note.start_time]
            block_boundaries.extend(
                self.recording.note_detector.get_boundary_time(
                    pitches,
                    index,
                )
                for index in breakpoints[:-1]
            )
            block_boundaries.append(note.end_time)

        minimum_duration = self._correction_min_length()
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
            target_durations = [
                target.duration()
                for target in targets
            ]
            target_total = sum(target_durations)
            if target_total <= 0 or block_end <= block_start:
                return None

            boundaries = [block_start]
            elapsed = 0.0
            for duration in target_durations[:-1]:
                elapsed += duration
                boundaries.append(
                    block_start
                    + (block_end - block_start)
                    * elapsed
                    / target_total
                )
            boundaries.append(block_end)

            block_pitches = pitches[frame_start:frame_end]
            for target_index, (
                start_time,
                end_time,
            ) in enumerate(
                zip(boundaries, boundaries[1:])
            ):
                if (
                    end_time - start_time
                    < minimum_duration - self.COST_EPSILON
                ):
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
                            float(
                                np.median(
                                    [
                                        pitch.value
                                        for pitch in segment_pitches
                                    ]
                                )
                            )
                        ],
                        velocity=note.velocity,
                        instrument=note.instrument,
                    )
                )

        return (
            split
            if len(split) == len(score_targets)
            else None
        )
