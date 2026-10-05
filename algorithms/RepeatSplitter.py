"""Recover missing score repeats by splitting their already-matched user notes."""
from copy import deepcopy

from app_logic.NoteData import Note, NoteData


class RepeatSplitter:
    """One local recovery pass on the original-note alignment."""
    refit_score_alignment = False

    def __init__(self, recording=None, config=None, verbose=False):
        self.recording = recording
        self.config = recording.config if recording is not None else config
        self.verbose = verbose
        self.proposals = []

    def update_config(self, config):
        self.config = config

    def check_mistakes(self, recording=None, verbose=None):
        if recording is not None:
            self.recording = recording
            self.config = recording.config
        rec = self.recording
        if rec is None or rec.alignment is None:
            return None, None
        if verbose is not None:
            self.verbose = verbose
        score = rec.score_data.clipped_note_data(channel=rec.active_instrument)
        notes, rec.alignment, self.proposals = self.recover_repeats(
            [rec.note_data.data[t] for t in rec.note_data.times],
            [score.data[t] for t in score.times], rec.mistake_detector, rec.alignment)
        if notes is not None:
            rec.note_data = notes
            rec.recompute_vibrato(note_aware=True)
        rec.reindex_mistakes()
        return rec.note_data, rec.alignment

    @staticmethod
    def groups(notes, detected=False, max_gap=.05):
        """Group contiguous equal pitches; score chords must agree in full."""
        result = []
        for note in notes:
            pitch = lambda n: round(n.midi_num[0]) if detected else tuple(n.midi_num)
            if (result and pitch(result[-1][-1]) == pitch(note)
                    and -.001 <= note.start_time - result[-1][-1].end_time <= max_gap):
                result[-1].append(note)
            else:
                result.append([note])
        return result

    @staticmethod
    def duration_cost(pairs, detector):
        """Local recovery objective: duration error for matches, existing gap costs."""
        return sum(detector.get_deletion_cost(score) if user is None else
                   detector.get_insertion_cost(user) if score is None else
                   abs(user.duration() - score.duration())
                   for user, score in pairs)

    @staticmethod
    def _better(candidate, previous):
        return (previous is None or candidate[0] < previous[0] - 1e-9
                or (abs(candidate[0] - previous[0]) <= 1e-9
                    and candidate[1] < previous[1]))

    @classmethod
    def recover_repeat_block(cls, block, detector):
        """Return (cost including fees, cut count, assignments) for a local block.

        An assignment is (user, original match index, start, stop, time segments).
        Score notes between assignments remain deleted. Each recovered deletion pays
        its exact former deletion cost; matches pay duration error only (no pitch
        or onset cost). Ties prefer fewer cuts. Original matches
        must stay inside their assigned consecutive score subset.
        """
        scores = [score for _, score in block]
        anchors = [(i, user) for i, (user, _) in enumerate(block) if user is not None]
        deletion = [0.]
        for score in scores:
            deletion.append(deletion[-1] + detector.get_deletion_cost(score))

        # Consumed score count -> (penalized cost, new boundary count, assignments).
        states = {0: (0., 0, [])}
        for i, (anchor, user) in enumerate(anchors):
            following = anchors[i + 1][0] if i + 1 < len(anchors) else len(scores)
            updated = {}
            for consumed, (cost, cuts, path) in states.items():
                # Missing notes before this assignment may remain missing.
                for start in range(consumed, anchor + 1):
                    skipped_cost = deletion[start] - deletion[consumed]
                    for stop in range(anchor + 1, following + 1):
                        subset = scores[start:stop]
                        if len(subset) == 1:
                            segments = [(user.start_time, user.end_time)]
                        else:
                            duration = sum(score.duration() for score in subset)
                            if duration <= 0:
                                continue
                            boundaries, elapsed = [user.start_time], 0.
                            for score in subset[:-1]:
                                elapsed += score.duration()
                                boundaries.append(user.start_time + user.duration() * elapsed / duration)
                            boundaries.append(user.end_time)
                            segments = list(zip(boundaries, boundaries[1:]))
                            if any(b - a <= .03 for a, b in segments):
                                continue
                        candidates = [Note(-1, a, b, list(user.midi_num)) for a, b in segments]
                        # All subset members except the original match were deleted.
                        fee = (deletion[stop] - deletion[start]
                               - detector.get_deletion_cost(scores[anchor]))
                        local_cost = cls.duration_cost(zip(candidates, subset), detector)
                        candidate = (cost + skipped_cost + local_cost + fee,
                                     cuts + len(subset) - 1,
                                     path + [(user, anchor, start, stop, segments)])
                        if cls._better(candidate, updated.get(stop)):
                            updated[stop] = candidate
            states = updated
        best = None
        for consumed, (cost, cuts, path) in states.items():
            candidate = (cost + deletion[-1] - deletion[consumed], cuts, path)
            if cls._better(candidate, best):
                best = candidate
        return best

    @classmethod
    def recover_repeats(cls, user_notes, score_notes, detector, alignment):
        """Refine only matched-note/deletion blocks inside same-pitch score groups.

        Existing matches anchor an ordered partition of the original score notes.
        Each user note can cover its matched score note plus neighboring deletions;
        it cannot cross another matched user note, an insertion, or a substitution.
        A small local DP chooses partial, complete, or no recovery, charging the
        original deletion cost for every recovered missing score note.
        Outside pairs, existing onsets, pitches, and the fitted timeline stay fixed.
        """
        group_for = {n.id: index for index, group in enumerate(cls.groups(score_notes))
                     if len(group) > 1 for n in group}
        pairs = alignment.pairs
        output, proposals = [], []
        next_id = max((n.id for n in user_notes), default=-1) + 1
        changed = False
        index = 0
        while index < len(pairs):
            user, score = pairs[index]
            group = group_for.get(score.id) if score is not None else None
            if group is None or (user is not None and detector.is_pitch_substitution(user, score)):
                output.append(pairs[index])
                index += 1
                continue
            end = index
            while end < len(pairs):
                u, s = pairs[end]
                if (s is None or group_for.get(s.id) != group
                        or (u is not None and detector.is_pitch_substitution(u, s))):
                    break
                end += 1
            block = pairs[index:end]
            anchors = [(j, u) for j, (u, _) in enumerate(block) if u is not None]
            missing = sum(u is None for u, _ in block)
            if not missing or not anchors:
                output.extend(block)
                index = end
                continue
            scores = [s for _, s in block]
            before = cls.duration_cost(block, detector)
            candidate = cls.recover_repeat_block(block, detector)
            accepted = candidate is not None and candidate[1] > 0 and candidate[0] < before - 1e-9
            cuts = []
            if accepted:
                consumed = 0
                for u, anchor, start, stop, segments in candidate[2]:
                    output.extend((None, score) for score in scores[consumed:start])
                    consumed = stop
                    if stop - start == 1:
                        output.append((u, scores[start]))
                        continue
                    for offset, (a, b) in enumerate(segments):
                        note = deepcopy(u)
                        # Retain the original ID on the piece covering its old match.
                        note.id = u.id if start + offset == anchor else next_id
                        if note.id == next_id:
                            next_id += 1
                        note.start_time = note.base_start_time = a
                        note.end_time = note.base_end_time = b
                        note.info = None
                        output.append((note, scores[start + offset]))
                        if offset:
                            cuts.append((u.id, a))
                output.extend((None, score) for score in scores[consumed:])
                changed = True
            else:
                output.extend(block)
            proposals.append(dict(user_ids=[u.id for _, u in anchors], score_ids=[s.id for s in scores],
                                  status='split' if accepted else 'unchanged', cuts=cuts,
                                  recovered_notes=len(cuts),
                                  recovery_fee=sum(detector.get_deletion_cost(scores[j])
                                                   for _, anchor, start, stop, _ in candidate[2]
                                                   for j in range(start, stop) if j != anchor) if accepted else 0.,
                                  before_cost=before, candidate_cost=candidate[0] if candidate else None))
            index = end
        if not changed:
            return None, alignment, proposals
        notes = NoteData()
        for note in user_notes:
            if note.midi_num[0] == -1:
                notes.write_note(note)
        for user, _ in output:
            if user is not None:
                notes.write_note(user)
        return notes, detector.alignment_from_pairs(output), proposals

    @classmethod
    def repeat_split(cls, user_notes, score_notes, detector):
        """Standalone adapter; production reuses the original fitted alignment."""
        alignment = detector.get_string_edit_alignment(
            [n for n in user_notes if n.midi_num[0] != -1], score_notes)
        result, _, proposals = cls.recover_repeats(user_notes, score_notes, detector, alignment)
        if result is None:
            result = NoteData()
            for note in user_notes:
                result.write_note(note)
        return result, proposals
