from dataclasses import dataclass

import numpy as np


@dataclass
class NoteInfo:
    """Per-note descriptors (pitch/cents, timing, vibrato, relative volume).
    Generated on demand from a Recording's pitch frames + alignment via
    analyze(), and cached on the inspected Note (Note.info)."""

    # Volume is shown RELATIVE to this take only — never as an absolute number.
    # A computer mic's level is uncalibrated (mic gain/distance/AGC all unknown),
    # and an absolute dBFS reading disagrees with the color ramp (a note can be
    # loud FOR THIS TAKE yet low in absolute dBFS). Normalizing to the take's own
    # [quietest, loudest] is gain-invariant and matches the ramp, so volume_frac
    # is the only volume descriptor — the readout is the color bar, no number.

    note_name: str
    cents: float                            # offset from the nearest semitone
    onset: float                            # app-time (sec)
    duration: float                         # sec
    onset_mistake: str | None = None        # "early" / "late" / None
    duration_mistake: str | None = None     # "long" / "short" / None
    vibrato_rate_hz: float | None = None
    vibrato_width_cents: float | None = None  # full peak-to-peak sinusoid width
    volume_frac: float | None = None        # 0..1 bar position in the take's dBFS range; None = no data

    @classmethod
    def analyze(cls, recording, note) -> "NoteInfo":
        midi = float(note.midi_num[0])
        onset_mistake, duration_mistake = cls._timing_mistakes(recording, note)
        rate, width = cls._vibrato(recording, note)
        return cls(
            note_name=note.get_note_name(),
            cents=(midi - round(midi)) * 100.0,
            onset=note.start_time,
            duration=note.duration(),
            onset_mistake=onset_mistake,
            duration_mistake=duration_mistake,
            vibrato_rate_hz=rate,
            vibrato_width_cents=width,
            volume_frac=cls._volume(recording, note),
        )

    @staticmethod
    def _timing_mistakes(recording, note) -> tuple[str | None, str | None]:
        """This note's onset ("early"/"late") and duration ("long"/"short")
        mistakes from the alignment, if analyze() has produced them."""
        onset_mistake = duration_mistake = None
        alignment = getattr(recording, "alignment", None)
        for m in (alignment.mistakes_for(note, "timing") if alignment else []):
            if m.type in ("early", "late"):
                onset_mistake = m.type
            elif m.type in ("long", "short"):
                duration_mistake = m.type
        return onset_mistake, duration_mistake

    @staticmethod
    def _vibrato(recording, note) -> tuple[float | None, float | None]:
        """Summarize the same instantaneous track drawn by VibratoWidget."""
        vibrato_data = getattr(recording, "vibrato_data", None)
        if vibrato_data is None:
            return None, None
        return vibrato_data.note_summary(note)

    @classmethod
    def _volume(cls, recording, note) -> float | None:
        """0..1 bar position within the take's dBFS range (None when the note
        carries no measurable volume): the note's mean level placed across the
        take's [quietest, loudest] frame dBFS — the same normalization the plot
        dots and ScoreViewer dots color across (mirrored inline here since
        app_logic can't import ui.Colors)."""
        pd = recording.pitch_data
        vol = pd.mean_volume(note.start_time, note.end_time)
        vmin, vmax = pd.volume_range_db()
        if vol <= 0 or vmin is None or vmax is None or vmax <= vmin:
            return None
        frac = (20.0 * np.log10(vol) - vmin) / (vmax - vmin)
        return float(min(max(frac, 0.0), 1.0))
