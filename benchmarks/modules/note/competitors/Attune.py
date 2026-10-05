from __future__ import annotations
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase


class Attune(NoteDetectorBase):
    """Production score-conditioned note detection and audio-only diagnostic."""

    def detect(self, **kwargs):
        self.recording.detect_notes()
        return self.recording.note_data

    @classmethod
    def predict_task(cls, task, config):
        from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker

        recording, cfg, adapter, conditioning = cls.recording_for_task(task, config)
        timings = {}
        from benchmarks.modules.note.NoteCache import NoteCache

        timings.update(
            NoteCache.attune_frontend(
                task, config, recording, cfg, adapter, conditioning
            )
        )
        started = cls.cpu_seconds()
        recording.transition_detector.clear_transitions(recording.pitch_data.data)
        if conditioning:
            recording.resize_score(to_span="pitch", include_transitions=False)
            recording.update_min_note_length()
        notes = recording.note_detector.detect_notes(recording.pitch_data.data)
        timings["segmentation_cpu_seconds"] = cls.cpu_seconds() - started
        if conditioning:
            recording.note_data = notes
            started = cls.cpu_seconds()
            recording.align_score_and_refine()
            timings["refinement_cpu_seconds"] = cls.cpu_seconds() - started
            notes = recording.note_data
        iv, hz = NoteBenchmarker.notedata_to_intervals(notes, cfg)
        return (iv, hz, timings)
