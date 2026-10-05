"""Region refinement: pitch blocks first, score-guided repetitions second.

Checker 2 supplies only the shared pass acceptance, frame partitioning, and
silence/minimum-duration primitives. Retained for explicit legacy comparisons;
production selects RepeatSplitter.
"""
from copy import copy

import numpy as np

from notebooks.archive.MistakeChecker2 import MistakeChecker as Checker2
from app_logic.NoteData import NoteData


class MistakeChecker(Checker2):
    def _good_pair(self, pair):
        user, score = pair
        return (user is not None and score is not None
                and self.recording.mistake_detector.get_pitch_distance(user, score)
                < self.config.pitch_tolerance)

    def _regions(self, alignment):
        """Maximal non-good runs; substitutions participate but do not trigger."""
        pairs = alignment.pairs
        start = 0
        while start < len(pairs):
            if self._good_pair(pairs[start]):
                start += 1
                continue
            end = start
            while end < len(pairs) and not self._good_pair(pairs[end]):
                end += 1
            run = pairs[start:end]
            if any(u is None or s is None for u, s in run):
                left, right = start > 0, end < len(pairs)
                # As in Checker 2, only deletion regions may use a voiced edge.
                if (left and right) or ((left or right) and any(u is None for u, _ in run)):
                    yield pairs[start - int(left):end + int(right)], left, right
            start = end

    @staticmethod
    def _score_blocks(scores):
        groups = []
        for score in scores:
            if groups and groups[-1][-1].midi_num[0] == score.midi_num[0]:
                groups[-1].append(score)
            else:
                groups.append([score])
        blocks = []
        for group in groups:
            block = copy(group[0])
            block.end_time = group[-1].end_time
            blocks.append(block)
        return blocks, groups

    def _repeat_splits(self, block, targets):
        """Propose one additional score-proportional boundary at a time."""
        durations = np.asarray([s.duration() for s in targets], dtype=float)
        if len(targets) < 2 or durations.sum() <= 0:
            return
        for fraction in np.cumsum(durations)[:-1] / durations.sum():
            boundary = block.start_time + block.duration() * float(fraction)
            if min(boundary - block.start_time, block.end_time - boundary) < self._correction_min_length():
                continue
            yield boundary

    def _refine_region(self, window, left, right):
        detector = self.recording.mistake_detector
        hosts, scores, _, _, _ = self._window_parts(window)
        if not hosts or not scores:
            return None
        region = self._region_for(hosts)
        if not (left and right):
            bounds = self.pd.get_voiced_range(include_transitions=False)
            if bounds is None:
                return None
            if not left:
                region.start_time = min(region.start_time, bounds[0])
            if not right:
                region.end_time = max(region.end_time, bounds[1])
        blocks, groups = self._score_blocks(scores)
        low, high = sorted((len(hosts), len(scores)))
        # Search pitch-block counts once, including counts needed after repeats
        # are collapsed. This is a staged heuristic, not a joint optimum.
        best = None
        for count in range(min(low, len(blocks)), min(high, self._segment_capacity([region])) + 1):
            notes = self._partition_region(region, count)
            if notes is None or self._crosses_blocking_silence(hosts, notes):
                continue
            alignment = detector.get_string_edit_alignment(notes, blocks)
            paired = {id(s) for u, s in alignment.pairs if u is not None and s is not None}
            if (left and id(blocks[0]) not in paired) or (right and id(blocks[-1]) not in paired):
                continue
            cost = detector.get_alignment_cost(alignment)
            if best is None or cost < best[0] - self.COST_EPSILON:
                best = cost, notes, alignment
        if best is None:
            return None
        _, notes, block_alignment = best
        # Build reusable proportional cut positions inside the selected blocks.
        # Greedy additions are assessed against the original score, never the
        # collapsed score. Unmatched pitch blocks receive no inferred repeats.
        group_by_id = {id(b): g for b, g in zip(blocks, groups)}
        cuts = []
        for user, score in block_alignment.pairs:
            if user is not None and score is not None:
                for boundary in self._repeat_splits(user, group_by_id[id(score)]):
                    cuts.append(boundary)

        anchor_ids = {id(window[0][1])} if left else set()
        if right:
            anchor_ids.add(id(window[-1][1]))

        def evaluate(candidate):
            if self._crosses_blocking_silence(hosts, candidate):
                return float('inf')
            alignment = detector.get_string_edit_alignment(candidate, scores)
            paired = {id(s) for u, s in alignment.pairs if u is not None and s is not None}
            if not anchor_ids <= paired:
                return float('inf')
            return detector.get_alignment_cost(alignment)

        cost = evaluate(notes)
        while cuts and len(notes) < high:
            choice = None
            for boundary in cuts:
                for index, note in enumerate(notes):
                    if not note.start_time < boundary < note.end_time:
                        continue
                    if min(boundary - note.start_time, note.end_time - boundary) < self._correction_min_length():
                        continue
                    split = []
                    for start, end in ((note.start_time, boundary), (boundary, note.end_time)):
                        frames = self.pd.read(start_time=start, end_time=end, clean=True)
                        if not frames:
                            break
                        part = copy(note)
                        part.start_time, part.end_time = start, end
                        part.base_start_time, part.base_end_time = start, end
                        part.midi_num = [float(np.median([p.value for p in frames]))]
                        split.append(part)
                    if len(split) != 2:
                        continue
                    candidate = notes[:index] + split + notes[index + 1:]
                    new_cost = evaluate(candidate)
                    if new_cost < cost - self.COST_EPSILON and (choice is None or new_cost < choice[0]):
                        choice = new_cost, candidate, boundary
            if choice is None:
                break
            cost, notes, boundary = choice
            cuts.remove(boundary)
        saving = detector.get_alignment_cost(window) - cost
        if low <= len(notes) <= high and saving > self.COST_EPSILON:
            return hosts, notes, saving
        return None

    def _check_mistakes(self, note_data, alignment):
        self.nd, self.alignment = note_data, alignment
        proposals = []
        for window, left, right in self._regions(alignment):
            proposal = self._refine_region(window, left, right)
            if proposal is not None:
                proposals.append(proposal)
        removed_ids, replacements = set(), []
        edits = 0
        for removed, added, saving in sorted(proposals, key=lambda p: p[2], reverse=True):
            ids = {id(n) for n in removed}
            if ids & removed_ids:
                continue
            removed_ids.update(ids)
            replacements.extend(added)
            edits += 1
        if not edits:
            return note_data, alignment, 0
        notes = [copy(n) for n in note_data.data.values() if id(n) not in removed_ids]
        notes.extend(replacements)
        result = NoteData()
        for index, note in enumerate(sorted(notes, key=lambda n: n.start_time)):
            note.id = index
            result.write_note(note)
        score = self.recording.score_data.clipped_note_data(channel=self.recording.active_instrument)
        updated = self.recording.mistake_detector.detect_mistakes(result, score)
        return result, updated, edits
