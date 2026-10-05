"""Frozen pre-2026-09-30 repeat refiner for paired and historical benchmarks.

Do not use as the production checker or update it with new splitting behavior.
"""
from copy import deepcopy
from app_logic.NoteData import Note, NoteData


def groups(notes, detected=False, max_gap=.05):
    result = []
    for note in notes:
        pitch = lambda n: round(n.midi_num[0]) if detected else tuple(n.midi_num)
        if (result and pitch(result[-1][-1]) == pitch(note)
                and -.001 <= note.start_time-result[-1][-1].end_time <= max_gap):
            result[-1].append(note)
        else:
            result.append([note])
    return result


def repeat_split(user_notes, score_notes, detector):
    """Return copied notes and an auditable list of accepted/rejected proposals.

    Group detected repetitions too, solely for the coarse alignment, so existing
    articulations are not penalized as insertions against one collapsed score.
    Use the existing score fit, without fitting each candidate to itself.
    """
    users, scores = groups(user_notes, True), groups(score_notes)
    def collapse(group, i):
        return Note(i, group[0].start_time, group[-1].end_time,
                    list(group[0].midi_num))
    us = [collapse(g, i) for i, g in enumerate(users)]
    ss = [collapse(g, i) for i, g in enumerate(scores)]
    alignment = detector.get_string_edit_alignment(us, ss)
    cuts, proposals = {}, []

    def divided_notes(selected_cuts):
        result = NoteData()
        for note in user_notes:
            boundaries = [note.start_time, *sorted(set(selected_cuts.get(note.id, []))), note.end_time]
            for a, b in zip(boundaries, boundaries[1:]):
                n = deepcopy(note)
                n.id = len(result.times)
                n.start_time = n.base_start_time = a
                n.end_time = n.base_end_time = b
                result.write_note(n)
        return result

    def full_cost(data):
        pairs = detector.get_string_edit_alignment([data.data[t] for t in data.times], score_notes)
        return detector.get_alignment_cost(pairs)

    result = divided_notes(cuts)
    cost = full_cost(result)
    for u, s in alignment.pairs:
        if u is None or s is None or len(scores[s.id]) < 2:
            continue
        ug, sg = users[u.id], scores[s.id]
        detail = dict(user_ids=[n.id for n in ug], score_ids=[n.id for n in sg])
        reason = None
        if any(abs(n.midi_num[0]-s.midi_num[0]) > .5 for n in ug):
            reason = 'pitch mismatch'
        elif len(ug) >= len(sg):
            reason = 'already segmented'
        elif sum(n.duration() for n in sg) <= 0:
            reason = 'zero score duration'
        proposed = []
        before_cost, candidate_cost = cost, None
        if reason is None:
            total_duration = sum(n.duration() for n in sg)
            elapsed = 0.
            for sn in sg[:-1]:
                elapsed += sn.duration()
                t = u.start_time + u.duration() * elapsed / total_duration
                # Existing detected onsets are authoritative; never move them.
                if any(abs(n.start_time-t) <= .1 for n in ug[1:]):
                    continue
                containing = next((n for n in ug if n.start_time+.03 < t < n.end_time-.03), None)
                if containing is None:
                    reason = 'boundary outside supported span'
                    break
                proposed.append((containing.id, t))
            if reason is None:
                candidate_cuts = deepcopy(cuts)
                for i, t in proposed:
                    candidate_cuts.setdefault(i, []).append(t)
                candidate = divided_notes(candidate_cuts)
                candidate_cost = full_cost(candidate)
                if candidate_cost > cost + 1e-9:
                    reason = 'alignment cost increased'
                else:
                    cuts, result, cost = candidate_cuts, candidate, candidate_cost
        proposals.append(dict(**detail, status=reason or 'split', cuts=proposed,
                              before_cost=before_cost, candidate_cost=candidate_cost, after_cost=cost))
    return result, proposals



class MistakeChecker:
    """Apply the validated repeat-only pass without iterative score refitting."""

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
        notes, self.proposals = repeat_split(
            [rec.note_data.data[t] for t in rec.note_data.times],
            [score.data[t] for t in score.times], rec.mistake_detector)
        changed = any(p['status'] == 'split' and p['cuts'] for p in self.proposals)
        if changed:
            # Cached popup timing/vibrato descriptors predate these boundaries.
            for note in notes.data.values():
                note.info = None
            rec.note_data = notes
            rec.detect_mistakes()
            rec.reindex_mistakes()
            rec.recompute_vibrato(note_aware=True)
        if self.verbose:
            count = sum(len(p['cuts']) for p in self.proposals if p['status'] == 'split')
            print(f'Repeat refinement: {count} accepted boundary cuts')
        return rec.note_data, rec.alignment
