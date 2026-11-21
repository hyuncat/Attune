from app_logic.JsonHandler import JsonHandler
from app_logic.user.ds.Recording import Recording
from ui.Colors import Colors
from ui.score.NotePopupSV import NotePopupSV


class ScoreAnnotations:
    """Builds the mistake-annotation payload the ScoreViewer pushes into its JS
    layer (set_mistake_annotations): colored score notes, insertion markers,
    per-note volumes, and the NotePopupSV popup payloads. Everything is keyed
    by drift-free score-note INDICES — the web layer only renders and reports
    clicks; it never infers note identity from Verovio seconds."""

    @classmethod
    def build(cls, rec: Recording) -> dict:
        score_nd = rec.score_data.note_datas.get(rec.active_instrument)
        if score_nd is None:
            return {"notes": {}, "insertions": [], "noteMeta": {}, "volumes": {}}

        score_notes = score_nd.read(i=0, j=len(score_nd.times))
        score_index = JsonHandler._note_index_maps(score_notes)
        volume_scale = (rec.pitch_data.volume_color_scale()
                        if rec.pitch_data is not None else None)

        def sidx(note):
            return JsonHandler._lookup_note_index(note, score_index)

        notes: dict[str, list[dict]] = {}
        note_meta: dict[str, dict] = {}
        volumes: dict[str, dict] = {}
        insertion_slots: dict[tuple[int | None, int | None], dict] = {}

        for idx, score_note in enumerate(score_notes):
            note_meta[str(idx)] = {
                "seekTime": float(score_note.start_time),
                # chord pitches, low->high irrelevant here: JS pairs them with the
                # rendered noteheads to fit its midi->y map for insertion markers
                "midis": [float(m) for m in score_note.midi_num
                          if m is not None and m >= 0],
            }
            volume = cls._score_note_volume(rec, score_note, volume_scale)
            if volume is not None:
                volumes[str(idx)] = volume

        mistakes = list(rec.alignment.pitch_mistakes) + list(rec.alignment.timing_mistakes)

        for mistake in mistakes:
            if mistake is None or mistake.is_overridden():
                continue
            payload = NotePopupSV.mistake_payload(mistake)
            if mistake.type == "insertion":
                flanks = rec.alignment.flanking_score_notes(mistake)
                if flanks is None:
                    continue
                left_idx = sidx(flanks[0]) if flanks[0] is not None else None
                right_idx = sidx(flanks[1]) if flanks[1] is not None else None
                if left_idx is None and right_idx is None:
                    continue
                entry = insertion_slots.setdefault(
                    (left_idx, right_idx),
                    {
                        "leftIndex": left_idx,
                        "rightIndex": right_idx,
                        "seekTime": rec.alignment.insertion_seek_time(mistake),
                        "mistakes": [],
                        "midis": [],
                    },
                )
                entry["mistakes"].append(payload)
                # several insertions share one marker: clicks seek the FIRST one
                entry["seekTime"] = min(
                    entry["seekTime"],
                    rec.alignment.insertion_seek_time(mistake),
                )
                midi = NotePopupSV.primary_midi(mistake.user_note)
                if midi is not None:
                    entry["midis"].append(midi)
                volume = cls._user_note_volume(rec, mistake.user_note, volume_scale)
                if volume is not None:
                    entry["volume"] = volume
                continue

            idx = sidx(mistake.midi_note)
            if idx is None:
                continue
            notes.setdefault(str(idx), []).append(payload)

        # the marker's vertical position is the mean of the slot's played pitches
        insertions = []
        for entry in insertion_slots.values():
            midis = entry.pop("midis")
            if midis:
                entry["midi"] = sum(midis) / len(midis)
            insertions.append(entry)

        return {
            "notes": notes,
            "insertions": insertions,
            "noteMeta": note_meta,
            "volumes": volumes,
        }

    # --- per-note volume payloads (the colored volume dots under the staff) ---
    @classmethod
    def _score_note_volume(
        cls,
        rec: Recording,
        score_note,
        volume_scale,
    ) -> dict | None:
        """The volume payload of the user note this score note aligned to."""
        if rec.alignment is None or score_note is None:
            return None
        user_note = rec.alignment.get_match(score_note=score_note)
        return cls._user_note_volume(rec, user_note, volume_scale)

    @classmethod
    def _user_note_volume(
        cls,
        rec: Recording,
        user_note,
        volume_scale,
    ) -> dict | None:
        if user_note is None or rec.pitch_data is None:
            return None
        level = rec.pitch_data.mean_level_db(user_note.start_time, user_note.end_time)
        frac = volume_scale.fraction_db(level) if volume_scale is not None else None
        if frac is None:
            return None
        return {
            "frac": float(frac),
            # score-dimmed (unlike GuitarHero's dots) so the cursor reads on top
            "color": Colors.css_rgb(Colors.viridis(frac, dim=True)),
        }
