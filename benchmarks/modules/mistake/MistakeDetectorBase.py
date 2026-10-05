"""MistakeDetectorBase implementation and owned benchmark helpers."""

from __future__ import annotations
from abc import ABC, abstractmethod


class MistakeDetectorBase(ABC):
    """Shared detector contract and the data used to compare methods."""

    THREAD_ENV = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    )

    @abstractmethod
    def preflight(self):
        """Validate pinned dependencies and return model identity."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, performance, score, directory):
        """Estimate mistake events from shared performance and score inputs."""
        raise NotImplementedError

    @staticmethod
    def truth_events(truth, reference, performed):
        from copy import copy
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
        from scipy.optimize import linear_sum_assignment

        "Canonicalize NET truth, never the raw injection history."
        score = reference.notes_by_id()
        events = []
        for event in truth:
            kind = event["type"]
            if kind in ("deletion", "substitution"):
                note = score[event["score_note_id"]]
                events.append(
                    dict(kind="missed", onset=note.start_time, pitch=note.midi_num[0])
                )
            if kind in ("insertion", "substitution"):
                note = performed.data[event["time"]]
                events.append(
                    dict(kind="extra", onset=note.start_time, pitch=note.midi_num[0])
                )
        return events

    @staticmethod
    def with_reference_score_ids(mistakes, reference, aligned_score):
        from copy import copy
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
        from scipy.optimize import linear_sum_assignment

        "Relink an unchanged score sequence after tempo fitting, without parser IDs.\n\n    MIDI IDs are assigned across channels (including regenerated metronome\n    notes). Tempo changes can therefore change IDs or make them alias another\n    original note. Score order and pitches remain fixed in this benchmark.\n    Keep fitted timings for duration diagnostics, but use original IDs for\n    symbolic scoring and original score-audio onsets for missed events.\n    "
        original = [reference.data[t] for t in reference.times]
        fitted = [aligned_score.data[t] for t in aligned_score.times]
        if len(original) != len(fitted) or any(
            (a.midi_num != b.midi_num for a, b in zip(original, fitted))
        ):
            raise ValueError(
                "Cannot map fitted score: note count or pitch sequence changed"
            )
        correspondence = {id(b): a for a, b in zip(original, fitted)}
        result = []
        for mistake in mistakes:
            mapped = copy(mistake)
            if mistake.midi_note is not None:
                source = correspondence.get(id(mistake.midi_note))
                if source is None:
                    raise ValueError(
                        "Alignment refers to a note outside the current fitted score"
                    )
                mapped.midi_note = copy(mistake.midi_note)
                mapped.midi_note.id = source.id
            result.append(mapped)
        return result

    @staticmethod
    def predicted_events(mistakes, reference):
        from copy import copy
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
        from scipy.optimize import linear_sum_assignment

        score = reference.notes_by_id()
        events = []
        for mistake in mistakes:
            if mistake.type in ("deletion", "substitution"):
                note = score[mistake.midi_note.id]
                events.append(
                    dict(kind="missed", onset=note.start_time, pitch=note.midi_num[0])
                )
            if mistake.type in ("insertion", "substitution"):
                note = mistake.user_note
                events.append(
                    dict(kind="extra", onset=note.start_time, pitch=note.midi_num[0])
                )
        return events

    @staticmethod
    def score_events(predicted, truth, onset_tolerance=0.1, pitch_tolerance=0.5):
        from copy import copy
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
        from scipy.optimize import linear_sum_assignment

        "Maximum one-to-one onset+pitch matching, separately for each error class."
        counts = {}
        for kind in ("missed", "extra"):
            predicted_kind = [e for e in predicted if e["kind"] == kind]
            true_kind = [e for e in truth if e["kind"] == kind]
            valid = np.zeros((len(predicted_kind), len(true_kind)), dtype=bool)
            for i, pred in enumerate(predicted_kind):
                for j, ref in enumerate(true_kind):
                    valid[i, j] = (
                        abs(pred["onset"] - ref["onset"]) <= onset_tolerance
                        and abs(pred["pitch"] - ref["pitch"]) <= pitch_tolerance
                    )
            matches = maximum_bipartite_matching(csr_matrix(valid), perm_type="column")
            tp = int(np.count_nonzero(matches >= 0))
            counts[f"audio_{kind}"] = (
                tp,
                len(predicted_kind) - tp,
                len(true_kind) - tp,
            )
        counts["audio_pitch"] = tuple(
            (sum((c[i] for c in counts.values())) for i in range(3))
        )
        return counts

    @staticmethod
    def net_mistakes(
        reference,
        performed,
        *,
        onset_tolerance=0.1,
        pitch_tolerance=0.5,
        duration_tolerance=0.25,
        score_onsets=None,
    ):
        from copy import copy
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
        from scipy.optimize import linear_sum_assignment

        "One-to-one, onset-gated matching of final notes to score notes.\n\n    Maximize pitch-correct matches first, then remaining same-slot matches;\n    break ties by onset distance. No injector lineage or tested aligner is used.\n    Unmatched notes are extra/missed; differing-pitch pairs are substitutions.\n    Optional score_onsets undo the known insertion/deletion-induced time shifts for truth\n    matching only. Event times stay on the actual performed timeline. The onset\n    gate is fixed independently of scoring sweeps.\n    "
        score = [reference.data[t] for t in reference.times]
        played = [performed.data[t] for t in performed.times]
        n, m = (len(score), len(played))
        k = min(n, m)
        cost = np.zeros((n + m, n + m))
        cost[:n, :m] = (k + 2) ** 3
        for i, s in enumerate(score):
            for j, u in enumerate(played):
                distance = abs(
                    s.start_time - (score_onsets or {}).get(u.id, u.start_time)
                )
                if distance <= onset_tolerance:
                    correct = (
                        min((abs(u.midi_num[0] - p) for p in s.midi_num))
                        < pitch_tolerance
                    )
                    benefit = k + 2 if correct else 1.0
                    cost[i, j] = -benefit + distance / max(onset_tolerance, 1e-12) / (
                        k + 2
                    )
        si, ui = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(si, ui) if i < n and j < m and (cost[i, j] < 0)]
        matched_s, matched_u = ({i for i, _ in pairs}, {j for _, j in pairs})
        truth, audit = ([], [])
        for i, j in pairs:
            s, u = (score[i], played[j])
            audit.append(dict(score_note_id=int(s.id), performed_note_id=int(u.id)))
            if min((abs(u.midi_num[0] - p) for p in s.midi_num)) >= pitch_tolerance:
                truth.append(
                    dict(
                        type="substitution",
                        score_note_id=int(s.id),
                        time=float(u.start_time),
                    )
                )
            delta = u.duration() - s.duration()
            if abs(delta) > duration_tolerance:
                truth.append(
                    dict(
                        type="long" if delta > 0 else "short",
                        score_note_id=int(s.id),
                        time=float(u.start_time),
                        duration_error=float(delta),
                    )
                )
        truth.extend(
            (
                dict(type="deletion", score_note_id=int(s.id), time=float(s.start_time))
                for i, s in enumerate(score)
                if i not in matched_s
            )
        )
        truth.extend(
            (
                dict(type="insertion", time=float(u.start_time))
                for j, u in enumerate(played)
                if j not in matched_u
            )
        )
        return (truth, audit)

    @staticmethod
    def timeline_score_onsets(payload, performed):
        from copy import copy
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
        from scipy.optimize import linear_sum_assignment

        "Validate the generated time map against the serialized performed MIDI."
        if payload.get("injector", {}).get("timeline_protocol") not in {
            "monophonic_edits_v2",
            "monophonic_edits_v3",
        }:
            return None
        entries = payload.get("performance_timeline")
        ordered = [performed.data[t] for t in performed.times]
        if (
            entries is None
            or len(entries) != len(ordered)
            or len({e["id"] for e in entries}) != len(entries)
        ):
            raise ValueError("Missing or inconsistent monophonic performance time map")
        ordered = [performed.data[t] for t in performed.times]
        if any(
            (a.end_time > b.start_time + 1e-08 for a, b in zip(ordered, ordered[1:]))
        ):
            raise ValueError(
                "Generated MIDI is polyphonic despite the monophonic protocol"
            )
        entries = sorted(entries, key=lambda e: e["onset"])
        for e, n in zip(entries, ordered):
            if (
                abs(n.start_time - e["onset"]) > 0.01
                or n.midi_num[0] != e["pitch"]
                or n.end_time <= n.start_time
                or ("end" in e and abs(n.end_time - e["end"]) > 0.01)
            ):
                raise ValueError("Performance time map does not match serialized MIDI")
        injector = payload.get("injector", {})
        if injector.get("insertion_pitch_policy") == "distinct_from_final_neighbors_v1":
            for i in injector["inserted_note_indices"]:
                if not 0 <= i < len(ordered):
                    raise ValueError(
                        "Inserted note index is outside the serialized performance"
                    )
                if any(
                    (
                        ordered[i].midi_num[0] in ordered[j].midi_num
                        for j in (i - 1, i + 1)
                        if 0 <= j < len(ordered)
                    )
                ):
                    raise ValueError(
                        "Serialized inserted pitch matches a performed neighbor"
                    )
        return {n.id: float(e["score_onset"]) for e, n in zip(entries, ordered)}

    @staticmethod
    def prf(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        return (precision, recall, f1)
