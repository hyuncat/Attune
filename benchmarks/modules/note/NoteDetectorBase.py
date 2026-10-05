from __future__ import annotations
import resource
import time
import importlib.util
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Sequence
import numpy as np
import pretty_midi
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.PitchData import PitchData
from app_logic.user.ds.Recording import Recording


class NoteDetectorBase:
    """Shared recording adapter, detector dispatch and note conversion helpers."""

    class _BasicPitchOptionalBackendFilter(logging.Filter):
        _PREFIXES = (
            "Coremltools is not installed.",
            "tflite-runtime is not installed.",
            "onnxruntime is not installed.",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            return not any(
                (record.getMessage().startswith(prefix) for prefix in self._PREFIXES)
            )

    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.detector = recording.note_detector
        self.config = recording.config

    def apply_transition_postprocess(self, note_data: NoteData) -> NoteData:
        """Legacy benchmark-only transition cleanup.

        Production no longer marks or removes transition frames. This remains
        available only for explicit historical baseline comparisons.
        """
        pitch_data = self.recording.pitch_data
        notes = NoteDetectorBase.clone_note_data(note_data)
        self._recompute_note_pitches(notes, pitch_data)
        return self._prune_transition_notes(notes, pitch_data)

    @staticmethod
    def _configure_external_model_environment() -> None:
        mpl_dir = Path(tempfile.gettempdir()) / "attune-matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
        NoteDetectorBase._suppress_external_dependency_warnings()
        NoteDetectorBase._patch_legacy_collections_abc()
        NoteDetectorBase._patch_legacy_numpy_aliases()

    @staticmethod
    def _suppress_external_dependency_warnings() -> None:
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API\\..*",
            category=UserWarning,
        )
        root_logger = logging.getLogger()
        if not any(
            (
                isinstance(
                    log_filter, NoteDetectorBase._BasicPitchOptionalBackendFilter
                )
                for log_filter in root_logger.filters
            )
        ):
            root_logger.addFilter(NoteDetectorBase._BasicPitchOptionalBackendFilter())

    @staticmethod
    def _patch_legacy_collections_abc() -> None:
        """Expose moved ABCs for older optional deps under Python 3.12.

        PyPI madmom still imports MutableSequence from collections during
        CREPE Notes repeated-note splitting. Python 3.12 only exposes those ABCs
        from collections.abc, so patch the old names before upstream imports.
        """
        import collections
        import collections.abc

        for name in (
            "Callable",
            "Iterable",
            "Mapping",
            "MutableMapping",
            "MutableSequence",
            "Sequence",
        ):
            if not hasattr(collections, name) and hasattr(collections.abc, name):
                setattr(collections, name, getattr(collections.abc, name))

    @staticmethod
    def _patch_legacy_numpy_aliases() -> None:
        """Expose NumPy aliases removed after 1.20 for older madmom releases."""
        aliases = {
            "bool": bool,
            "complex": complex,
            "float": float,
            "int": int,
            "object": object,
            "str": str,
        }
        for name, value in aliases.items():
            if name not in np.__dict__:
                setattr(np, name, value)

    @staticmethod
    def _module_available(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    @staticmethod
    def _frame_pitch(pitch: Pitch | None):
        if pitch is None:
            return None
        return pitch.value

    def _is_note_frame(self, pitch: Pitch | None) -> bool:
        return (
            pitch is not None
            and pitch.value != -1
            and (pitch.unvoiced_prob < self.config.unv_thresh)
            and (not getattr(pitch, "is_transition", False))
        )

    def _recompute_note_pitches(
        self, note_data: NoteData, pitch_data: PitchData
    ) -> None:
        for note in note_data.data.values():
            if note is None or not note.midi_num or note.midi_num[0] == -1:
                continue
            frames = pitch_data.read(
                start_time=note.start_time, end_time=note.end_time, clean=False
            )
            kept = [p for p in frames if self._is_note_frame(p)]
            med = self._get_median_pitches(kept)
            if med[0] != -1:
                note.midi_num = med

    def _prune_transition_notes(
        self, note_data: NoteData, pitch_data: PitchData, frac_thresh: float = 0.5
    ) -> NoteData:
        survivors = []
        for note in note_data.read(i=0, j=len(note_data.times)):
            voiced = pitch_data.read(
                start_time=note.start_time, end_time=note.end_time, clean=True
            )
            n_trans = sum((1 for p in voiced if p.is_transition))
            if voiced and n_trans > frac_thresh * len(voiced):
                continue
            survivors.append(note)
        out = NoteData()
        for idx, note in enumerate(survivors):
            copied = NoteDetectorBase.copy_note(note, note_id=idx)
            out.write_note(copied)
        return out

    @staticmethod
    def _normalize_method(method: str) -> str:
        aliases = {
            "pelt": "ruptures",
            "ruptures_pelt": "ruptures",
            "slope-window": "slope_window",
            "window-slope-aware": "slope_window",
            "slope_aware_window": "slope_window",
            "tony": "tony_pyin",
            "pyin-tony": "tony_pyin",
            "pyin_tony": "tony_pyin",
            "tony-pyin": "tony_pyin",
            "tony_pyin": "tony_pyin",
            "crepe": "crepe_notes",
            "crepe-notes": "crepe_notes",
            "basic-pitch": "basic_pitch",
        }
        return aliases.get(method, method)

    def _prepare_transition_flags(self, exclude_transitions: bool) -> None:
        if exclude_transitions:
            self.recording.transition_detector.detect_transitions(
                self.recording.pitch_data.data
            )
            return
        for pitch in self.recording.pitch_data.data:
            if pitch is not None:
                pitch.is_transition = False

    def _trim_note_edges_by_volume(
        self, note_data: NoteData, floor_ratio: float
    ) -> NoteData:
        frame_dt = self.config.h1 / self.config.sr
        min_seconds = getattr(self.config, "min_note_length", 0.03)
        try:
            score_note_data = self.recording.score_data.clipped_note_data(
                channel=self.recording.active_instrument
            )
            min_seconds = score_note_data.get_min_note_length(
                default=float(min_seconds), clean=True
            )
        except (AttributeError, KeyError, TypeError):
            pass
        min_seconds = max(0.0, float(min_seconds) * self.config.min_note_length_factor)
        out = NoteData()
        for idx, note in enumerate(note_data.read(i=0, j=len(note_data.times))):
            frames = self.recording.pitch_data.read(
                start_time=note.start_time, end_time=note.end_time, clean=False
            )
            valid = [p for p in frames if p is not None]
            if not valid:
                out.write_note(NoteDetectorBase.copy_note(note, note_id=idx))
                continue
            volumes = np.asarray([float(p.volume) for p in valid], dtype=float)
            peak = float(volumes.max(initial=0.0))
            if peak <= 0:
                out.write_note(NoteDetectorBase.copy_note(note, note_id=idx))
                continue
            keep = np.flatnonzero(volumes >= peak * floor_ratio)
            if keep.size == 0:
                out.write_note(NoteDetectorBase.copy_note(note, note_id=idx))
                continue
            start = float(valid[int(keep[0])].time)
            end = float(valid[int(keep[-1])].time) + frame_dt
            if end - start < min_seconds:
                out.write_note(NoteDetectorBase.copy_note(note, note_id=idx))
                continue
            trimmed = NoteDetectorBase.copy_note(note, note_id=idx)
            trimmed.start_time = start
            trimmed.end_time = end
            out.write_note(trimmed)
        return out

    def _audio_path_for_external_model(self) -> str:
        candidate = getattr(self.recording, "audio_filepath", None)
        if candidate is not None and Path(candidate).exists():
            return str(candidate)
        import soundfile as sf

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        sf.write(
            handle.name,
            self.recording.audio_data.read_all(),
            int(self.recording.audio_data.sr),
        )
        return handle.name

    def _write_external_audio(self, audio_path: Path) -> None:
        import soundfile as sf

        sf.write(
            str(audio_path),
            self.recording.audio_data.read_all(),
            int(self.recording.audio_data.sr),
        )

    @staticmethod
    def _notedata_from_midi(midi_path: Path, origin: float = 0.0) -> NoteData:
        pretty_midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        raw_notes = sorted(
            (
                pretty_midi_note
                for instrument in pretty_midi_data.instruments
                for pretty_midi_note in instrument.notes
            ),
            key=lambda pretty_midi_note: (
                pretty_midi_note.start,
                pretty_midi_note.end,
                pretty_midi_note.pitch,
            ),
        )
        note_data = NoteData()
        for idx, pretty_midi_note in enumerate(raw_notes):
            if pretty_midi_note.end <= pretty_midi_note.start:
                continue
            start_time = float(pretty_midi_note.start) + origin
            while start_time in note_data.data:
                start_time = float(np.nextafter(start_time, float("inf")))
            note_data.write_note(
                Note(
                    i=idx,
                    start_time=start_time,
                    end_time=float(pretty_midi_note.end) + origin,
                    midi_num=[int(pretty_midi_note.pitch)],
                    velocity=int(pretty_midi_note.velocity),
                )
            )
        return note_data

    @staticmethod
    def _reindex(note_data: NoteData) -> NoteData:
        out = NoteData()
        for idx, note in enumerate(note_data.read(i=0, j=len(note_data.times))):
            copied = NoteDetectorBase.copy_note(note, note_id=idx)
            out.write_note(copied)
        return out

    @staticmethod
    def copy_note(note: Note, note_id: int | None = None) -> Note:
        """Return a detached Note copy for benchmark post-processing."""
        return Note(
            i=note.id if note_id is None else note_id,
            start_time=float(note.start_time),
            end_time=float(note.end_time),
            midi_num=list(note.midi_num),
            velocity=note.velocity,
            instrument=note.instrument,
        )

    @staticmethod
    def clone_note_data(note_data: NoteData) -> NoteData:
        out = NoteData()
        for idx, note in enumerate(note_data.read(i=0, j=len(note_data.times))):
            out.write_note(NoteDetectorBase.copy_note(note, note_id=idx))
        return out

    @staticmethod
    def competitor(method):
        from benchmarks.modules.note.competitors.Attune import Attune
        from benchmarks.modules.note.competitors.BasicPitch import BasicPitch
        from benchmarks.modules.note.competitors.CrepeNotes import CrepeNotes
        from benchmarks.modules.note.competitors.Ruptures import Ruptures
        from benchmarks.modules.note.competitors.SlopeWindow import SlopeWindow
        from benchmarks.modules.note.competitors.Tony import Tony

        key = NoteDetectorBase._normalize_method(method)
        if key == "mt3":
            raise NotImplementedError("MT3 requires a separate model/runtime setup")
        try:
            return {
                "attune": Attune,
                "attune-audio-only": Attune,
                "ruptures": Ruptures,
                "slope_window": SlopeWindow,
                "basic_pitch": BasicPitch,
                "crepe_notes": CrepeNotes,
                "tony_pyin": Tony,
            }[key]
        except KeyError:
            raise ValueError(f"unknown note benchmark method: {method!r}") from None

    def detect(self, method="ruptures", **kwargs):
        return self.competitor(method)(self.recording).detect(**kwargs)

    @classmethod
    def predict_task(cls, task, config):
        raise NotImplementedError(
            f"{cls.__name__} does not support audio-only evaluation"
        )

    @staticmethod
    def recording_for_task(task, config):
        import pretty_midi
        import tempfile
        from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
        from benchmarks.modules.pitch.competitors.Attune import Attune

        method = task["method"]
        adapter = Attune()
        conditioning = method == "attune"
        if conditioning:
            source = pretty_midi.PrettyMIDI(task["score"])
            instruments = [i for i in source.instruments if not i.is_drum]
            ins = instruments[task["score_part"]]
            pitches = [n.pitch for n in ins.notes]
            if not pitches:
                raise ValueError("Empty score part")
            fmin, fmax = (task["pitch_fmin"], task["pitch_fmax"])
        else:
            fmin, fmax = (config.audio_only_fmin, config.audio_only_fmax)
        cfg = adapter.config_for(fmin, fmax)
        cfg.verbose = False
        cfg.min_note_length = config.audio_only_min_note
        if conditioning:
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory) / "score.mid"
                source.instruments = [ins]
                source.write(str(p))
                recording = adapter.recording_for(
                    cfg, score_data=OneInstrumentScoreData(p)
                )
        else:
            recording = adapter.recording_for(cfg)
        recording.audio_filepath = Path(task["audio"])
        return (recording, cfg, adapter, conditioning)

    @staticmethod
    def cpu_seconds():
        """All process threads plus reaped child CPU (e.g. Sonic Annotator)."""
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time() + child.ru_utime + child.ru_stime

    @staticmethod
    def event_arrays(events):
        """Keep every polyphonic prediction, including identical onset times."""
        import numpy as np

        a = np.asarray(events, dtype=float).reshape(-1, 3)
        return (a[:, :2], 440.0 * 2.0 ** ((a[:, 2] - 69.0) / 12.0))

    def _get_median_pitches(
        self, pitches: Sequence[Pitch], n_candidates: int = 3
    ) -> list[float]:
        medians = [-1.0] * n_candidates
        voiced = [
            p
            for p in pitches
            if p is not None
            and p.value != -1
            and (p.unvoiced_prob < self.config.unv_thresh)
        ]
        if not voiced:
            return medians
        columns = [[] for _ in range(n_candidates)]
        for pitch in voiced:
            candidates = pitch.candidate_pitches or [(pitch.value, 1.0)]
            for i, (midi, _prob) in enumerate(candidates[:n_candidates]):
                if midi != -1:
                    columns[i].append(float(midi))
        for i, values in enumerate(columns):
            if values:
                medians[i] = float(np.median(values))
        return medians
