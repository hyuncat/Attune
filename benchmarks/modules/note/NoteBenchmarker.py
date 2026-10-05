from __future__ import annotations
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from benchmarks.modules.note.NoteCache import NoteCache
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import TypeAlias
import mir_eval
import numpy as np
import numpy.typing as npt
import pretty_midi
from benchmarks.paths import DATASETS_ROOT
from benchmarks.paths import REPO_ROOT
from benchmarks.paths import RESULTS_ROOT
from benchmarks.paths import ensure_repo_on_path
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData
from app_logic.midi.ScoreData import ScoreData
from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.Recording import Recording
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
from collections.abc import Sequence
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
import argparse
import multiprocessing
import queue as queue_mod
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from concurrent.futures.process import BrokenProcessPool
import math
import traceback
import contextlib
import hashlib
import importlib.metadata
import multiprocessing as mp
import platform
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import itertools
import pandas as pd

ensure_repo_on_path()
ROOT = REPO_ROOT
PathLike = str | Path
NoteMethodConfig: TypeAlias = dict[str, Any]


class NoteMethods:

    @staticmethod
    def _cpd_note_methods() -> dict[str, NoteMethodConfig]:
        """Change-point-detection note families.

        Every change-point method goes through the benchmark adapter so the rows
        differ ONLY in the search/cost model. Kernel-CPD kernels map through
        ruptures' cost name: linear<-l2, gaussian<-rbf, cosine. ``dynp`` needs a
        target segment count, which it takes from the score (a real input in-app),
        so it is the one score-informed / "oracle" row.

        ``pelt-l2`` used to carry ``current_notedetector=True`` (route to production
        ``Recording.detect_notes()``). That row stopped measuring PELT once the
        production detector became KernelCPD — it silently duplicated
        kernelcpd-linear, and it was the only row skipping the transition
        post-pass, so it was never comparable to the rest anyway.
        """
        bases: list[tuple[str, NoteMethodConfig]] = [
            ("pelt-l1", dict(method="ruptures", ruptures_algorithm="pelt", model="l1")),
            ("pelt-l2", dict(method="ruptures", ruptures_algorithm="pelt", model="l2")),
            (
                "pelt-rbf",
                dict(method="ruptures", ruptures_algorithm="pelt", model="rbf"),
            ),
            (
                "dynp",
                dict(
                    method="ruptures",
                    ruptures_algorithm="dynp",
                    model="l2",
                    oracle_note_count=True,
                ),
            ),
            (
                "kernelcpd-linear",
                dict(method="ruptures", ruptures_algorithm="kernelcpd", model="l2"),
            ),
            (
                "kernelcpd-gaussian",
                dict(method="ruptures", ruptures_algorithm="kernelcpd", model="rbf"),
            ),
            (
                "kernelcpd-cosine",
                dict(method="ruptures", ruptures_algorithm="kernelcpd", model="cosine"),
            ),
            (
                "bottomup",
                dict(method="ruptures", ruptures_algorithm="bottomup", model="l2"),
            ),
            (
                "window-base",
                dict(method="ruptures", ruptures_algorithm="window", model="l2"),
            ),
            ("window-slope-aware", dict(method="slope_window")),
        ]
        methods: dict[str, NoteMethodConfig] = {}
        for label, base_config in bases:
            config = dict(base_config)
            config.setdefault("current_notedetector", False)
            config.update(
                exclude_transitions=False,
                postprocess_transitions=False,
                benchmark_group="cpd",
                base_method=label,
            )
            if config.get("current_notedetector", False):
                config["postprocess_transitions"] = False
            methods[label] = config
        return methods

    @staticmethod
    def _moreover_note_methods() -> dict[str, NoteMethodConfig]:
        """Non-change-point baselines. These bring their own segmentation and are
        scored as-is: the transition re-median / prune post-pass is deliberately NOT
        applied (per spec, only the CPD families get it)."""
        return {
            "basic-pitch": dict(
                method="basic_pitch",
                postprocess_transitions=False,
                external_baseline=True,
                benchmark_group="moreover",
                base_method="basic-pitch",
            ),
            "tony": dict(
                method="tony_pyin",
                postprocess_transitions=False,
                external_baseline=True,
                benchmark_group="moreover",
                base_method="tony",
            ),
            "crepe-notes": dict(
                method="crepe_notes",
                postprocess_transitions=False,
                external_baseline=True,
                uses_onsets=True,
                benchmark_group="moreover",
                base_method="crepe-notes",
            ),
        }

    @staticmethod
    def _default_note_methods(
        all_methods: dict[str, NoteMethodConfig]
    ) -> dict[str, NoteMethodConfig]:
        """The curated matrix used to narrow the field: all CPD families plus the
        moreover baselines. Same set feeds the 10-song shortlist and the full run;
        ``--method`` / ``--no-external`` prune it further per invocation."""
        return dict(all_methods)


_ALL_NOTE_METHODS = {
    **NoteMethods._cpd_note_methods(),
    **NoteMethods._moreover_note_methods(),
}
_EXTERNAL_NOTE_METHODS = {"basic-pitch", "tony", "crepe-notes"}


class NoteBenchmarker:
    ETUDE_DATASETS: ClassVar[list[str]] = ["kayser", "wohlfahrt"]
    ETUDE_DATASET_ALIASES: ClassVar[dict[str, str]] = {"wolfhart": "wohlfahrt"}
    ALL_NOTE_METHODS: ClassVar[dict[str, NoteMethodConfig]] = _ALL_NOTE_METHODS
    NOTE_METHODS: ClassVar[dict[str, NoteMethodConfig]] = (
        NoteMethods._default_note_methods(_ALL_NOTE_METHODS)
    )
    OPTIONAL_NOTE_METHODS: ClassVar[dict[str, NoteMethodConfig]] = {
        "mt3": dict(
            method="mt3",
            external_baseline=True,
            benchmark_group="external",
            base_method="mt3",
        )
    }
    SOUNDFONT_PATH: ClassVar[Path] = REPO_ROOT / "resources" / "MuseScore_General.sf3"
    SYNTH_SR: ClassVar[int] = 44100

    def __init__(self, onset_tolerance: float = 0.05) -> None:
        self.attune = AttuneRealtime()
        self.DATASETS = DATASETS_ROOT
        self.RESULTS = RESULTS_ROOT
        self.MISTAKE_DIR = DATASETS_ROOT / "mistake-db"
        self.RESULTS.mkdir(parents=True, exist_ok=True)
        self.onset_tolerance = onset_tolerance

    @property
    def algorithm_verbose(self) -> bool:
        return self.attune.algorithm_verbose

    @algorithm_verbose.setter
    def algorithm_verbose(self, value: bool) -> None:
        self.attune.algorithm_verbose = bool(value)

    @staticmethod
    def corpus_dir_for_midi(midi_path: PathLike) -> Path:
        midi_path = Path(midi_path)
        return (
            midi_path.parent.parent
            if midi_path.parent.name == "midi"
            else midi_path.parent
        )

    @classmethod
    def dataset_name_for_midi(cls, midi_path: PathLike) -> str:
        return cls.corpus_dir_for_midi(midi_path).name

    @classmethod
    def synth_audio_dir_for_midi(cls, midi_path: PathLike) -> Path:
        return cls.corpus_dir_for_midi(midi_path) / "synth_audio"

    @staticmethod
    def notedata_to_pm(
        note_data: NoteData, program: int = 40, velocity: int = 90
    ) -> "pretty_midi.PrettyMIDI":
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)
        for start in note_data.times:
            note = note_data.data[start]
            if (
                not note.midi_num
                or note.midi_num[0] == -1
                or note.end_time <= note.start_time
            ):
                continue
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(note.velocity or velocity),
                    pitch=int(note.midi_num[0]),
                    start=float(note.start_time),
                    end=float(note.end_time),
                )
            )
        pm.instruments.append(instrument)
        return pm

    def synth_midi(
        self,
        midi_path: PathLike,
        out_dir: PathLike | None = None,
        gain: float = 1.0,
        force: bool = False,
    ) -> Path:
        """Render a MIDI etude to WAV with fluidsynth, reusing an up-to-date file."""
        midi_path = Path(midi_path)
        out_dir = (
            Path(out_dir)
            if out_dir is not None
            else self.synth_audio_dir_for_midi(midi_path)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (midi_path.stem + ".wav")
        if (
            not force
            and out.exists()
            and (out.stat().st_mtime >= midi_path.stat().st_mtime)
        ):
            return out
        executable = shutil.which("fluidsynth")
        if executable is None and sys.platform == "darwin":
            for directory in ("/opt/homebrew/bin", "/usr/local/bin"):
                executable = shutil.which("fluidsynth", path=directory)
                if executable is not None:
                    break
        if executable is None:
            raise RuntimeError(
                "FluidSynth executable not found on the Python process PATH or in standard macOS Homebrew locations. If already installed, add its bin directory to the notebook kernel's PATH."
            )
        subprocess.run(
            [
                executable,
                "-ni",
                "-R",
                "0",
                "-C",
                "0",
                "-g",
                str(gain),
                "-F",
                str(out),
                "-r",
                str(self.SYNTH_SR),
                str(self.SOUNDFONT_PATH),
                str(midi_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out

    def result_csv_path(self, algorithm: str, dataset: str) -> Path:
        return self.RESULTS / algorithm / f"{dataset.replace('/', '_')}.csv"

    def write_dataset_result(
        self, df: "pd.DataFrame", algorithm: str, dataset: str, index: bool = True
    ) -> Path:
        out_path = self.result_csv_path(algorithm, dataset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=index)
        return out_path

    @staticmethod
    def _limit(items: list[Any], max_tracks: int | None) -> list[Any]:
        return items if max_tracks is None else items[:max_tracks]

    @staticmethod
    def notedata_to_intervals(
        note_data: NoteData, config, drop_rests: bool = True
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert NoteData to mir_eval-style intervals and frequencies."""
        notes = note_data.read(i=0, j=len(note_data.times)) if note_data.times else []
        rows, frequencies_hz = ([], [])
        for note in notes:
            if drop_rests and (not note.midi_num or note.midi_num[0] == -1):
                continue
            if note.end_time <= note.start_time:
                continue
            rows.append([note.start_time, note.end_time])
            frequencies_hz.append(config.midi_to_freq(note.midi_num[0]))
        if not rows:
            return (np.zeros((0, 2)), np.zeros((0,)))
        return (np.asarray(rows, dtype=float), np.asarray(frequencies_hz, dtype=float))

    def iter_etudes(self, dataset: str) -> Iterator[tuple[str, Path]]:
        """Returns iterable of (title, midi_path) for each etude in the given dataset."""
        source_dataset = self.ETUDE_DATASET_ALIASES.get(dataset, dataset)
        dataset_dir = self.DATASETS / "violin-etudes" / source_dataset
        midi_dir = dataset_dir / "midi"
        search_dir = midi_dir if midi_dir.exists() else dataset_dir
        for mid in sorted(search_dir.glob("*.mid")):
            yield (mid.stem, mid)

    @staticmethod
    def etude_corpus_dir_for_midi(midi_path: PathLike) -> Path:
        return NoteBenchmarker.corpus_dir_for_midi(midi_path)

    @staticmethod
    def detect_notes(
        recording: Recording,
        method: str = "pelt",
        model: str = "l2",
        do_transitions: bool = True,
        jump: int | None = None,
        **kwargs: Any,
    ) -> NoteData:
        method_config = dict(kwargs)
        method_config["method"] = method
        if method == "pelt":
            method_config["method"] = "ruptures"
            method_config.setdefault("ruptures_algorithm", "pelt")
        method_config.setdefault("model", model)
        if do_transitions is not None:
            method_config.setdefault("exclude_transitions", bool(do_transitions))
        if jump is not None:
            method_config.setdefault("jump", jump)
        notes = NoteDetectorBase(recording).detect(**method_config)
        recording.note_data = notes
        return notes

    @staticmethod
    def prepare_for_note_detection(
        recording: Recording,
        resize_score_to_pitch: bool = True,
        detect_transitions: bool = False,
    ) -> Recording:
        if detect_transitions:
            recording.transition_detector.detect_transitions(recording.pitch_data.data)
        if resize_score_to_pitch:
            recording.resize_score(
                to_span="pitch", include_transitions=not detect_transitions
            )
        recording.update_min_note_length()
        return recording

    @staticmethod
    def detect_recording_notes(recording: Recording) -> NoteData:
        """Run the production note pipeline exactly as the app does."""
        recording.detect_notes()
        return recording.note_data

    def detect_recording_notes_timed(
        self, recording: Recording
    ) -> tuple[NoteData, float]:
        start = time.perf_counter()
        notes = self.detect_recording_notes(recording)
        return (notes, time.perf_counter() - start)

    def detect_notes_timed(
        self,
        recording: Recording,
        method: str = "pelt",
        model: str = "l2",
        do_transitions: bool = True,
        jump: int | None = None,
        **kwargs: Any,
    ) -> tuple[NoteData, float]:
        start = time.perf_counter()
        notes = self.detect_notes(
            recording,
            method=method,
            model=model,
            do_transitions=do_transitions,
            jump=jump,
            **kwargs,
        )
        return (notes, time.perf_counter() - start)

    def analyze_recording(
        self,
        recording: Recording,
        method: str = "pelt",
        model: str = "l2",
        truncate: bool = False,
        **_unused,
    ) -> dict[str, float]:
        del method, model
        recording.reset_analysis()
        _, note_compute_time = self.detect_recording_notes_timed(recording)
        if truncate:
            recording.trim_end(mark_unsaved=False)
        return {"note_compute_time": note_compute_time}

    @staticmethod
    def _latency_offset(
        reference_onsets: npt.NDArray[np.float64],
        estimated_onsets: npt.NDArray[np.float64],
    ) -> float:
        """Robust constant detector latency: the median signed gap from each
        estimated onset to its NEAREST reference onset. This is a pure
        translation (no scale), so it can't drift interior notes — it only
        absorbs a fixed pitch/note-detector lag. Outlier onsets wash out in the
        median."""
        ref = np.sort(np.asarray(reference_onsets, dtype=float))
        est = np.asarray(estimated_onsets, dtype=float)
        if ref.size == 0 or est.size == 0:
            return 0.0
        if ref.size == 1:
            return float(ref[0] - np.median(est))
        idx = np.clip(np.searchsorted(ref, est), 1, ref.size - 1)
        left, right = (ref[idx - 1], ref[idx])
        nearest = np.where(np.abs(est - left) <= np.abs(est - right), left, right)
        return float(np.median(nearest - est))

    @staticmethod
    def _trim_boundary_notes(detected: NoteData, reference: NoteData) -> None:
        """NoteData form of `_trim_boundaries`: in-place clamp the FIRST/LAST
        VOICED detected notes to the reference (MIDI) note DURATIONS, anchoring
        the reliable inner edge and trimming the swelled outer one (synth
        attack/release). Used by the mistake pipeline, which aligns/checks the
        NoteData directly (the note benchmark trims the mir_eval intervals).
          - first note: start := end - ref_first_duration
          - last note:  end   := start + ref_last_duration
        No global shift — the detected take keeps its absolute time so resize /
        mistake detection still anchor it against the score."""

        def voiced(nd: NoteData) -> list[Note]:
            notes = nd.read(i=0, j=len(nd.times)) if nd.times else []
            return [n for n in notes if n.midi_num and n.midi_num[0] != -1]

        det_voiced, ref_voiced = (voiced(detected), voiced(reference))
        if not det_voiced or not ref_voiced:
            return
        det_voiced[0].start_time = det_voiced[0].end_time - ref_voiced[0].duration()
        det_voiced[-1].end_time = det_voiced[-1].start_time + ref_voiced[-1].duration()
        all_notes = detected.read(i=0, j=len(detected.times))
        detected.load_data({n.start_time: n for n in all_notes})

    def load_or_detect_notes(
        self,
        recording: Recording,
        cache_path: PathLike,
        model: str = "l2",
        write_cache: bool = True,
    ) -> tuple[NoteData, float]:
        """Load cached PELT notes if present, else detect (PELT) and cache the
        RAW notes. Caching is decoupled from trimming: callers apply
        `_trim_boundary_notes` after, so the cache holds the honest detector
        output. Returns (notes, note_compute_time) — the cached compute time on a
        hit, so timing reporting stays meaningful."""
        cache_path = Path(cache_path)
        if cache_path.exists():
            recording.note_data, metadata = self.load_note_data(cache_path)
            return (recording.note_data, float(metadata.get("note_compute_time", 0.0)))
        del model
        notes, note_compute_time = self.detect_recording_notes_timed(recording)
        if write_cache:
            self.save_note_data(
                notes,
                cache_path,
                metadata={
                    "note_compute_time": note_compute_time,
                    "method": "recording.detect_notes",
                },
            )
        return (notes, note_compute_time)

    @staticmethod
    def _trim_boundaries(
        est_intervals: npt.NDArray[np.float64], ref_intervals: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Force the FIRST/LAST detected notes to the reference (MIDI) note
        DURATIONS, anchoring the reliable inner edge and trimming the outer one.
        The synth's attack/release swells those two notes past their true length
        even with reverb/chorus off, and they have no neighbour to mask them.
          - first note: keep its END, pull its START to end - ref_first_duration
          - last note:  keep its START, pull its END to start + ref_last_duration
        Pure per-note trim (interior notes untouched); the caller's pad then
        slides everything up so all start times are >= 0. Returns a COPY so the
        shared estimated array is never mutated."""
        if est_intervals.shape[0] == 0 or ref_intervals.shape[0] == 0:
            return est_intervals
        out = est_intervals.copy()
        first_end, last_start = (out[0, 1], out[-1, 0])
        out[0, 0] = first_end - (ref_intervals[0, 1] - ref_intervals[0, 0])
        out[-1, 1] = last_start + (ref_intervals[-1, 1] - ref_intervals[-1, 0])
        return out

    @staticmethod
    def _normalized_method_name(method_config: NoteMethodConfig) -> str:
        method = method_config.get("method", "pelt")
        if method == "pelt":
            return "ruptures"
        if method == "basic-pitch":
            return "basic_pitch"
        if method == "tony-pyin":
            return "tony_pyin"
        if method == "crepe-notes":
            return "crepe_notes"
        return str(method)

    @classmethod
    def _method_uses_onsets(cls, method_config: NoteMethodConfig) -> bool:
        method = cls._normalized_method_name(method_config)
        return bool(method_config.get("uses_onsets", False) or method == "crepe_notes")

    @classmethod
    def _method_needs_audio(cls, method_config: NoteMethodConfig) -> bool:
        """True when a method reads the raw waveform. Pure change-point methods
        run on the cached pitch track alone, so their recordings skip audio."""
        method = cls._normalized_method_name(method_config)
        return bool(
            method_config.get("current_notedetector", False)
            or method_config.get("external_baseline", False)
            or method in {"basic_pitch", "tony_pyin", "crepe_notes"}
        )

    @classmethod
    def _base_detection_config(
        cls, method_config: NoteMethodConfig
    ) -> NoteMethodConfig:
        config = dict(method_config)
        for key in (
            "benchmark_group",
            "base_method",
            "current_notedetector",
            "external_baseline",
            "postprocess_transitions",
            "uses_onsets",
            "refined_with_onsets",
            "refine_with_onsets",
        ):
            config.pop(key, None)
        config["method"] = cls._normalized_method_name(config)
        if config["method"] == "ruptures":
            config.setdefault("ruptures_algorithm", "pelt")
        return config

    @staticmethod
    def _jsonable_method_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return [NoteBenchmarker._jsonable_method_value(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): NoteBenchmarker._jsonable_method_value(v)
                for k, v in sorted(value.items())
            }
        return repr(value)

    @classmethod
    def _method_cache_key(cls, method_config: NoteMethodConfig) -> str:
        base = cls._base_detection_config(method_config)
        payload = {
            key: cls._jsonable_method_value(value)
            for key, value in sorted(base.items())
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def _method_row_metadata(
        cls, label: str, method_config: NoteMethodConfig
    ) -> dict[str, Any]:
        method = cls._normalized_method_name(method_config)
        return {
            "method_label": label,
            "detector_family": method,
            "benchmark_group": method_config.get("benchmark_group", method),
            "base_method": method_config.get("base_method", label),
            "ruptures_algorithm": method_config.get(
                "ruptures_algorithm", "pelt" if method == "ruptures" else None
            ),
            "ruptures_model": method_config.get("model"),
            "ruptures_cost": method_config.get("cost", method_config.get("model")),
            "refined_with_onsets": False,
            "transition_excluding": (
                bool(method_config.get("exclude_transitions"))
                if "exclude_transitions" in method_config
                else None
            ),
            "oracle_note_count": bool(method_config.get("oracle_note_count", False)),
            "external_baseline": bool(method_config.get("external_baseline", False)),
            "postprocess_transitions": bool(
                method_config.get("postprocess_transitions", False)
            ),
            "uses_onsets": cls._method_uses_onsets(method_config),
            "current_notedetector": bool(
                method_config.get("current_notedetector", False)
            ),
        }

    def _detect_notes_for_method(
        self,
        recording: Recording,
        method_config: NoteMethodConfig,
        method_cache: dict[str, tuple[NoteData, float]],
    ) -> tuple[NoteData, dict[str, float]]:
        cache_key = self._method_cache_key(method_config)
        if bool(method_config.get("current_notedetector", False)):
            cache_key = f"recording.detect_notes:{cache_key}"
            if cache_key not in method_cache:
                recording.reset_analysis()
                notes, note_time = self.detect_recording_notes_timed(recording)
                method_cache[cache_key] = (
                    NoteDetectorBase.clone_note_data(notes),
                    note_time,
                )
            notes, note_time = method_cache[cache_key]
            recording.note_data = NoteDetectorBase.clone_note_data(notes)
            return (
                recording.note_data,
                {
                    "base_note_compute_time": note_time,
                    "onset_refinement_compute_time": 0.0,
                    "transition_postprocess_compute_time": 0.0,
                    "note_compute_time": note_time,
                },
            )
        if cache_key not in method_cache:
            base_config = self._base_detection_config(method_config)
            base_notes, base_time = self.detect_notes_timed(recording, **base_config)
            method_cache[cache_key] = (
                NoteDetectorBase.clone_note_data(base_notes),
                base_time,
            )
        base_notes, base_time = method_cache[cache_key]
        notes = NoteDetectorBase.clone_note_data(base_notes)
        postprocess_time = 0.0
        if bool(method_config.get("postprocess_transitions", False)):
            start = time.perf_counter()
            notes = NoteDetectorBase(recording).apply_transition_postprocess(notes)
            postprocess_time = time.perf_counter() - start
        recording.note_data = notes
        return (
            notes,
            {
                "base_note_compute_time": base_time,
                "onset_refinement_compute_time": 0.0,
                "transition_postprocess_compute_time": postprocess_time,
                "note_compute_time": base_time + postprocess_time,
            },
        )

    @staticmethod
    def _failed_method_row(error: Exception) -> dict[str, Any]:
        return {
            "Precision": np.nan,
            "Recall": np.nan,
            "F-measure": np.nan,
            "Average Overlap Ratio": np.nan,
            "Estimated Notes": 0,
            "note_compute_time": np.nan,
            "base_note_compute_time": np.nan,
            "onset_refinement_compute_time": np.nan,
            "error": f"{type(error).__name__}: {error}",
        }

    REALTIME_COL: ClassVar[str] = "Audio(s)/Compute(s)"
    NOTE_COMPUTE_COL: ClassVar[str] = "Note Compute Time (s)"
    NOTE_QUALITY_METRICS: ClassVar[list[str]] = ["F-measure", "Precision", "Recall"]
    NOTE_RESULT_COLUMNS: ClassVar[list[str]] = [
        "Track ID",
        "F-measure",
        "Precision",
        "Recall",
        "Average Overlap Ratio",
        "Estimated Notes",
        "Reference Notes",
        REALTIME_COL,
        NOTE_COMPUTE_COL,
        "Split",
        "Track",
        "Ensemble",
        "Instrument",
        "Voice",
        "error",
    ]

    def note_result_csv_path(self, method: str, dataset: str) -> Path:
        safe_dataset = dataset.replace("/", "_")
        safe_method = method.replace("/", "_")
        return (
            self.RESULTS / "note" / "raw_outputs" / safe_method / f"{safe_dataset}.csv"
        )

    @property
    def note_summary_csv_path(self) -> Path:
        return self.RESULTS / "note" / "note_benchmarks.csv"

    def display_note_columns(self, df: "pd.DataFrame") -> "pd.DataFrame":
        out = df.copy()
        if out.index.name in (None, "track_id"):
            out.index.name = "Track ID"
        rename_candidates = {
            "track_id": "Track ID",
            "method_label": "Method",
            "realtime_factor": self.REALTIME_COL,
            "note_compute_time": self.NOTE_COMPUTE_COL,
            "split": "Split",
            "track": "Track",
            "ensemble": "Ensemble",
            "instrument": "Instrument",
            "voice": "Voice",
        }
        rename = {
            src: dst
            for src, dst in rename_candidates.items()
            if src in out.columns and (dst not in out.columns or src == dst)
        }
        if rename:
            out = out.rename(columns=rename)
        out = out.drop(
            columns=[
                c
                for c in (
                    "audio_seconds",
                    "base_note_compute_time",
                    "onset_refinement_compute_time",
                    "transition_postprocess_compute_time",
                    "onset_compute_time",
                    "model",
                    "dataset",
                )
                if c in out.columns
            ],
            errors="ignore",
        )
        ordered = [c for c in self.NOTE_RESULT_COLUMNS if c in out.columns]
        remaining = [c for c in out.columns if c not in ordered]
        return out[ordered + remaining]

    def write_note_result(
        self, df: "pd.DataFrame", method: str, dataset: str, index: bool = True
    ) -> Path:
        out_path = self.note_result_csv_path(method, dataset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.display_note_columns(df).to_csv(out_path, index=index)
        return out_path

    def _audio_seconds(self, wav_path: PathLike | None) -> float:
        if wav_path is None:
            return float("nan")
        try:
            import soundfile as sf

            info = sf.info(str(wav_path))
            return float(info.frames) / float(info.samplerate)
        except Exception:
            return float("nan")

    @staticmethod
    def _timing_columns(
        note_compute_time: float, audio_seconds: float
    ) -> dict[str, Any]:
        realtime = (
            audio_seconds / note_compute_time
            if note_compute_time
            and note_compute_time > 0
            and np.isfinite(audio_seconds)
            else float("nan")
        )
        return {
            "note_compute_time": float(note_compute_time),
            "audio_seconds": float(audio_seconds),
            "realtime_factor": float(realtime),
        }

    def _eval_intervals(
        self,
        ref_iv: npt.NDArray[np.float64],
        ref_pi: npt.NDArray[np.float64],
        est_iv: npt.NDArray[np.float64],
        est_pi: npt.NDArray[np.float64],
        *,
        align: str,
        latency_align: bool,
        trim_boundaries: bool,
        onset_tolerance: float,
    ) -> tuple[float, float, float, float]:
        """Shared identity/resize interval scoring: constant latency shift +
        boundary trim + non-negative pad, then mir_eval overlap PRF."""
        if align == "identity" and latency_align and est_iv.size:
            est_iv = est_iv + self._latency_offset(ref_iv[:, 0], est_iv[:, 0])
        if trim_boundaries:
            est_iv = self._trim_boundaries(est_iv, ref_iv)
        mins = [0.0]
        if est_iv.size:
            mins.append(float(est_iv.min()))
        if ref_iv.size:
            mins.append(float(ref_iv.min()))
        pad = -min(mins)
        if pad:
            ref_iv = ref_iv + pad
            est_iv = est_iv + pad
        return mir_eval.transcription.precision_recall_f1_overlap(
            ref_iv,
            ref_pi,
            est_iv,
            est_pi,
            onset_tolerance=onset_tolerance,
            offset_ratio=None,
        )

    def _prepare_note_recording(
        self,
        primary_path: PathLike,
        reference_path: PathLike,
        align: str = "identity",
        needs_audio: bool = True,
    ) -> tuple[Recording, npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Build a Recording with cached pitches + score reference for one etude.

        Returns (recording, reference_intervals, reference_pitches, audio_seconds).
        The etude WAV is synthesized from the reference MIDI; pitch data is loaded
        from (or written to) the shared pyin_smooth cache — pitch detection time is
        NOT charged to the note benchmark. Subclasses (CocoChorales) override this
        with their own audio/reference/pitch-cache loading. ``needs_audio`` lets the
        override skip the waveform load for pure change-point methods (the etude
        path always has audio from synthesis, so it ignores the hint)."""
        midi_path = Path(reference_path)
        score_data = OneInstrumentScoreData(midi_path)
        config = self.attune.config_for(
            *self.attune.range_from_midi(score_data.midi_numbers)
        )
        recording = self.attune.recording_for(config, score_data=score_data)
        wav_path = self.synth_midi(midi_path)
        recording.audio_data = AudioData(
            audio_filepath=str(wav_path), config=recording.config
        )
        recording.audio_filepath = wav_path
        self.attune.load_or_detect_pitches(
            recording,
            cache_path=PitchCache.path_for(
                self.etude_corpus_dir_for_midi(midi_path), midi_path.stem
            ),
            smooth=True,
            write_cache=True,
        )
        self.prepare_for_note_detection(
            recording, resize_score_to_pitch=align == "resize", detect_transitions=False
        )
        ref_iv, ref_pi = self.notedata_to_intervals(
            recording.score_data.clipped_note_data(channel=recording.active_instrument),
            recording.config,
        )
        return (recording, ref_iv, ref_pi, self._audio_seconds(wav_path))

    def score_note_track(
        self,
        primary_path: PathLike,
        reference_path: PathLike,
        method_label: str,
        method_config: NoteMethodConfig,
        *,
        align: str = "identity",
        latency_align: bool = True,
        trim_boundaries: bool = True,
        onset_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Detect + score ONE method on ONE track, returning a single flat row.

        This is the note analogue of one pitch benchmark row: the
        parallel runner calls it once per (method, track). The row carries the note
        detection compute time and Audio(s)/Compute(s) — never pitch time."""
        onset_tolerance = (
            self.onset_tolerance if onset_tolerance is None else onset_tolerance
        )
        recording, ref_iv, ref_pi, audio_seconds = self._prepare_note_recording(
            primary_path,
            reference_path,
            align,
            needs_audio=self._method_needs_audio(method_config),
        )
        row_meta = self._method_row_metadata(method_label, method_config)
        try:
            notes, note_timing = self._detect_notes_for_method(
                recording, method_config, method_cache={}
            )
        except Exception as exc:
            return {
                **row_meta,
                "Precision": np.nan,
                "Recall": np.nan,
                "F-measure": np.nan,
                "Average Overlap Ratio": np.nan,
                "Estimated Notes": 0,
                "Reference Notes": int(len(ref_iv)),
                **self._timing_columns(float("nan"), audio_seconds),
                "onset_compute_time": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if align == "resize" and notes.times:
            recording.resize_score(to_span="note")
            ref_iv, ref_pi = self.notedata_to_intervals(
                recording.score_data.clipped_note_data(
                    channel=recording.active_instrument
                ),
                recording.config,
            )
        est_iv, est_pi = self.notedata_to_intervals(notes, recording.config)
        note_compute_time = note_timing["note_compute_time"]
        if len(est_iv) == 0:
            precision = recall = f_measure = aor = 0.0
        else:
            precision, recall, f_measure, aor = self._eval_intervals(
                ref_iv,
                ref_pi,
                est_iv,
                est_pi,
                align=align,
                latency_align=latency_align,
                trim_boundaries=trim_boundaries,
                onset_tolerance=onset_tolerance,
            )
        return {
            **row_meta,
            "Precision": float(precision),
            "Recall": float(recall),
            "F-measure": float(f_measure),
            "Average Overlap Ratio": float(aor),
            "Estimated Notes": int(len(est_iv)),
            "Reference Notes": int(len(ref_iv)),
            **self._timing_columns(note_compute_time, audio_seconds),
            "onset_compute_time": 0.0,
            "error": None,
        }

    def bench_note_track(
        self,
        midi_path: PathLike,
        onset_tolerance: float = None,
        methods: dict[str, NoteMethodConfig] = None,
        align: str = "identity",
        latency_align: bool = True,
        trim_boundaries: bool = True,
        write_note_cache: bool = True,
        progress: bool = False,
    ) -> dict[str, Any]:
        """Benchmark note detection on one synthesized etude.

        `align` controls how the detected notes are placed against the reference:
          - "identity" (default): the WAV is rendered FROM the reference MIDI, so
            audio-time already EQUALS MIDI-time and the true alignment is the
            identity. No tempo rescale runs (it would only re-fit the timebase
            from two noisy detected endpoints and inject drift). With
            `latency_align` we shift the detected notes by a single constant to
            absorb detector lag. This isolates note-DETECTION quality.
          - "resize": runs the app's resize_score(to_span="note") pipeline
            (perform.py::analyze) per method to score the full alignment-aware
            flow as the app actually runs it.
        `trim_boundaries` clamps the first/last detected notes to the reference
        MIDI durations (see _trim_boundaries) to neutralize the synth's
        attack/release swell on those two notes.
        `write_note_cache` auto-saves the RAW PELT-L2 notes to
        <corpus>/note_data/<stem>.note.json so the mistake benchmarker can reuse
        them instead of re-running PELT.
        """
        effective_onset_tolerance = (
            self.onset_tolerance if onset_tolerance is None else onset_tolerance
        )
        effective_methods = self._effective_note_methods(methods)
        midi_path = Path(midi_path)
        score_data = OneInstrumentScoreData(midi_path)
        config = self.attune.config_for(
            *self.attune.range_from_midi(score_data.midi_numbers)
        )
        recording = self.attune.recording_for(config, score_data=score_data)
        wav_path = self.synth_midi(midi_path)
        recording.audio_data = AudioData(
            audio_filepath=str(wav_path), config=recording.config
        )
        recording.audio_filepath = wav_path
        pitch_timing = self.attune.load_or_detect_pitches(
            recording,
            cache_path=PitchCache.path_for(
                self.etude_corpus_dir_for_midi(midi_path), midi_path.stem
            ),
            smooth=True,
            write_cache=True,
        )
        self.prepare_for_note_detection(
            recording, resize_score_to_pitch=align == "resize", detect_transitions=False
        )

        def read_reference():
            return self.notedata_to_intervals(
                recording.score_data.clipped_note_data(
                    channel=recording.active_instrument
                ),
                recording.config,
            )

        reference_intervals, reference_pitches = read_reference()
        out: dict[str, Any] = {"_reference_notes": len(reference_intervals)}
        method_cache: dict[str, tuple[NoteData, float]] = {}
        for label, method_config in effective_methods.items():
            method_start = time.perf_counter()
            if progress:
                print(f"    note method: {label}", flush=True)
            row_base = {
                **self._method_row_metadata(label, method_config),
                **pitch_timing,
            }
            try:
                notes, note_timing = self._detect_notes_for_method(
                    recording, method_config, method_cache=method_cache
                )
            except Exception as exc:
                out[label] = {
                    **row_base,
                    **self._failed_method_row(exc),
                    "onset_compute_time": 0.0,
                }
                if progress:
                    print(
                        f"      error after {time.perf_counter() - method_start:.1f}s: {exc}",
                        flush=True,
                    )
                continue
            note_compute_time = note_timing["note_compute_time"]
            if (
                write_note_cache
                and row_base["current_notedetector"]
                and (method_config.get("model") == "l2")
            ):
                cache_key = (
                    f"recording.detect_notes:{self._method_cache_key(method_config)}"
                )
                base_notes, base_time = method_cache[cache_key]
                self.save_note_data(
                    base_notes,
                    self.note_cache_path(
                        self.etude_corpus_dir_for_midi(midi_path), midi_path.stem
                    ),
                    metadata={
                        "note_compute_time": base_time,
                        "model": "l2",
                        "method": "recording.detect_notes",
                    },
                )
            if align == "resize":
                if notes.times:
                    recording.resize_score(to_span="note")
                reference_intervals, reference_pitches = read_reference()
            estimated_intervals, estimated_pitches = self.notedata_to_intervals(
                notes, recording.config
            )
            if len(estimated_intervals) == 0:
                out[label] = {
                    **row_base,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F-measure": 0.0,
                    "Average Overlap Ratio": 0.0,
                    "Estimated Notes": 0,
                    "onset_compute_time": 0.0,
                    **note_timing,
                    "error": None,
                }
                if progress:
                    print(
                        f"      done in {time.perf_counter() - method_start:.1f}s (0 notes)",
                        flush=True,
                    )
                continue
            ref_iv, ref_pi = (reference_intervals, reference_pitches)
            est_iv, est_pi = (estimated_intervals, estimated_pitches)
            if align == "identity" and latency_align:
                est_iv = est_iv + self._latency_offset(ref_iv[:, 0], est_iv[:, 0])
            if trim_boundaries:
                est_iv = self._trim_boundaries(est_iv, ref_iv)
            mins = [0.0, float(est_iv.min())]
            if ref_iv.size:
                mins.append(float(ref_iv.min()))
            pad = -min(mins)
            if pad:
                ref_iv = ref_iv + pad
                est_iv = est_iv + pad
            precision, recall, f_measure, average_overlap_ratio = (
                mir_eval.transcription.precision_recall_f1_overlap(
                    ref_iv,
                    ref_pi,
                    est_iv,
                    est_pi,
                    onset_tolerance=effective_onset_tolerance,
                    offset_ratio=None,
                )
            )
            out[label] = {
                **row_base,
                "Precision": precision,
                "Recall": recall,
                "F-measure": f_measure,
                "Average Overlap Ratio": average_overlap_ratio,
                "Estimated Notes": len(est_iv),
                "onset_compute_time": 0.0,
                **note_timing,
                "error": None,
            }
            if progress:
                print(
                    f"      done in {time.perf_counter() - method_start:.1f}s (F={f_measure:.3f}, notes={len(est_iv)})",
                    flush=True,
                )
        return out

    def bench_note_dataset(
        self,
        dataset: str,
        max_tracks: int | None = None,
        onset_tolerance: float | None = None,
        methods: dict[str, NoteMethodConfig] = None,
        align: str = "identity",
        latency_align: bool = True,
        trim_boundaries: bool = True,
        write_note_cache: bool = True,
        verbose: bool = True,
        write: bool = False,
    ) -> pd.DataFrame:
        import pandas as pd

        effective_methods = self._effective_note_methods(methods)
        tracks = self._limit(list(self.iter_etudes(dataset)), max_tracks)
        rows = []
        for i, (title, midi_path) in enumerate(tracks):
            track_result = self.bench_note_track(
                midi_path,
                onset_tolerance=onset_tolerance,
                methods=effective_methods,
                align=align,
                latency_align=latency_align,
                trim_boundaries=trim_boundaries,
                write_note_cache=write_note_cache,
                progress=verbose,
            )
            reference_note_count = track_result.pop("_reference_notes")
            for label, method_result in track_result.items():
                row = {
                    "dataset": dataset,
                    "track": title,
                    "method": label,
                    "Reference Notes": reference_note_count,
                }
                row.update(method_result)
                rows.append(row)
            if verbose:
                print(
                    f"[{dataset}] {i + 1}/{len(tracks)} {title[:32]:32s} reference={reference_note_count}"
                )
        df = pd.DataFrame(rows)
        if write:
            self.write_dataset_result(df, "note", dataset, index=False)
        return df

    def _effective_note_methods(
        self, methods: dict[str, NoteMethodConfig] | None
    ) -> dict[str, NoteMethodConfig]:
        return {
            label: dict(method_config)
            for label, method_config in (methods or self.NOTE_METHODS).items()
        }

    note_cache_path = staticmethod(NoteCache.note_cache_path)
    _compatible_note_end_time = staticmethod(NoteCache._compatible_note_end_time)
    save_note_data = staticmethod(NoteCache.save_note_data)
    load_note_data = staticmethod(NoteCache.load_note_data)
    _note_from_payload = staticmethod(NoteCache._note_from_payload)
    _note_to_payload = staticmethod(NoteCache._note_to_payload)


class OneInstrumentScoreData(ScoreData):
    """ScoreData wrapper that flattens all different channels to a single instrument."""

    def __init__(self, midi_path: PathLike, collapse_simultaneous: bool = True) -> None:
        source_path = Path(midi_path)
        note_data = self.pretty_midi_to_notedata(
            pretty_midi.PrettyMIDI(str(source_path)),
            collapse_simultaneous=collapse_simultaneous,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            normalized_path = Path(temp_dir) / source_path.name
            NoteBenchmarker.notedata_to_pm(note_data).write(str(normalized_path))
            super().__init__(normalized_path)
        self.filepath = source_path
        self.title = source_path.stem

    @property
    def note_data(self) -> NoteData:
        return self.clipped_note_data(channel=self.active_instrument)

    @property
    def midi_numbers(self) -> list[int]:
        return self.notedata_midi_numbers(self.note_data)

    @staticmethod
    def pretty_midi_to_notedata(
        pretty_midi_data: pretty_midi.PrettyMIDI, collapse_simultaneous: bool = True
    ) -> NoteData:
        raw_notes = sorted(
            (
                pretty_midi_note
                for instrument in pretty_midi_data.instruments
                for pretty_midi_note in instrument.notes
            ),
            key=lambda pretty_midi_note: (
                pretty_midi_note.start,
                -pretty_midi_note.pitch,
            ),
        )
        note_data, note_index, last_start_time = (NoteData(), 0, None)
        for pretty_midi_note in raw_notes:
            if (
                collapse_simultaneous
                and last_start_time is not None
                and (abs(pretty_midi_note.start - last_start_time) < 0.001)
            ):
                continue
            note_data.write_note(
                Note(
                    i=note_index,
                    start_time=float(pretty_midi_note.start),
                    end_time=float(pretty_midi_note.end),
                    midi_num=[int(pretty_midi_note.pitch)],
                    velocity=int(pretty_midi_note.velocity),
                )
            )
            last_start_time = pretty_midi_note.start
            note_index += 1
        return note_data

    @staticmethod
    def notedata_midi_numbers(note_data: NoteData) -> list[int]:
        return [
            note_data.data[t].midi_num[0]
            for t in note_data.times
            if note_data.data[t].midi_num and note_data.data[t].midi_num[0] != -1
        ]


class CocoNoteBenchmarker(NoteBenchmarker):
    """Note-detection benchmark over materialized CocoChorales stems."""

    def __init__(
        self,
        root: PathLike | None = None,
        f0_fps: float | None = None,
        onset_tolerance: float = 0.05,
    ) -> None:
        super().__init__(onset_tolerance=onset_tolerance)
        self.coco = CocoChorales(root=root, f0_fps=f0_fps)

    def local_midi_path(self, record: CocoChorales.Stem) -> Path | None:
        stems = self.coco.root / "main_dataset" / record.split / record.track
        candidates = [
            self.coco.materialized_midi_path(record),
            stems / "stems_midi" / f"{record.stem}.mid",
            stems / "stems_MIDI" / f"{record.stem}.mid",
        ]
        return next((p for p in candidates if p.exists()), None)

    def records_to_note_tracks(
        self, records: Sequence[CocoChorales.Stem]
    ) -> list[tuple[str, Path, Path]]:
        """(track_id, wav, midi) for stems whose audio AND stems_midi are present."""
        tracks: list[tuple[str, Path, Path]] = []
        for record in records:
            wav = self.coco.local_wav_path(record)
            midi = self.local_midi_path(record)
            if wav is None or midi is None:
                continue
            tracks.append((record.track_id, wav, midi))
        return tracks

    def iter_note_tracks(
        self, dataset: str = "test"
    ) -> Iterator[tuple[str, Path, Path]]:
        yield from self.records_to_note_tracks(
            self.coco.load_or_build_manifest(dataset)
        )

    def has_note_input_cache(self, wav_path: PathLike) -> bool:
        """Whether the pyin_smooth pitch cache (the note detector's input) exists."""
        return self.attune.has_pitch_cache(
            self.coco.cache_path_for_wav(wav_path), smooth=True
        )

    def _prepare_note_recording(
        self,
        primary_path: PathLike,
        reference_path: PathLike,
        align: str = "identity",
        needs_audio: bool = True,
    ) -> tuple[Recording, npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Assemble a Recording from the cached pyin_smooth pitches + stems_midi.

        The waveform is only loaded when the method actually needs it (external
        transcribers, spectral onsets) or when the pitch cache is missing and the
        track has to be pitch-detected from scratch. Change-point methods run on
        the cached pitch track alone."""
        wav_path = Path(primary_path)
        midi_path = Path(reference_path)
        score_data = OneInstrumentScoreData(midi_path)
        config = self.attune.config_for(
            *self.attune.range_from_midi(score_data.midi_numbers)
        )
        recording = self.attune.recording_for(config, score_data=score_data)
        recording.audio_filepath = wav_path
        cache_path = self.coco.cache_path_for_wav(wav_path)
        if needs_audio or not self.attune.has_pitch_cache(cache_path, smooth=True):
            recording.audio_data = self.coco.load_resampled_audio(wav_path, config.sr)
        self.attune.load_or_detect_pitches(
            recording, cache_path=cache_path, smooth=True, write_cache=True
        )
        self.prepare_for_note_detection(
            recording, resize_score_to_pitch=align == "resize", detect_transitions=False
        )
        ref_iv, ref_pi = self.notedata_to_intervals(
            recording.score_data.clipped_note_data(channel=recording.active_instrument),
            recording.config,
        )
        return (recording, ref_iv, ref_pi, self._audio_seconds(wav_path))

    def note_row_meta(self, wav_path: PathLike) -> dict[str, Any]:
        """Split / ensemble / instrument / voice for grouping the raw CSVs."""
        return self.coco.meta_for_wav(wav_path)


TrackItem = tuple[str, str, str, str]
WorkChunk = tuple[str, int, tuple[TrackItem, ...]]
ChunkResult = tuple[
    str, str, list[dict[str, Any]], list[tuple[str, str, str, str]], float, str | None
]
RunChunkFn = Callable[..., ChunkResult]
__all__ = [
    "NoteBenchmarker",
    "CocoNoteBenchmarker",
    "NoteEvaluation",
    "NotebookConfig",
    "NoteCLI",
    "NoteRunner",
]


class NoteRunner:

    @staticmethod
    def fmt_dur(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"

    @staticmethod
    def parse_shard(text: str | None) -> tuple[int, int] | None:
        if text is None:
            return None
        i, n = (int(x) for x in text.split("/"))
        if not 0 <= i < n:
            raise argparse.ArgumentTypeError(
                f"shard index {i} out of range for {n} shards"
            )
        return (i, n)

    @staticmethod
    def _force_teardown(ex: ProcessPoolExecutor) -> None:
        processes = list((getattr(ex, "_processes", None) or {}).values())
        ex.shutdown(wait=False, cancel_futures=True)
        for child in processes:
            if child.is_alive():
                child.terminate()
        for child in processes:
            child.join(timeout=10)
            if child.is_alive():
                child.kill()

    class ProgressRenderer:
        SPINNER = "|/-\\"

        def __init__(self, enabled: bool = True) -> None:
            self.enabled = enabled
            self.is_tty = enabled and sys.stdout.isatty()
            self.active: dict[int, dict[str, Any]] = {}
            self.live_lines = 0
            self.last_render = 0.0
            self.transport_failed = False

        def close(self) -> None:
            self._clear_live()
            sys.stdout.flush()

        def drain(self, progress_queue: Any | None) -> bool:
            if not self.enabled or progress_queue is None:
                return False
            changed = False
            while True:
                try:
                    event = progress_queue.get_nowait()
                except queue_mod.Empty:
                    break
                except (BrokenPipeError, EOFError, OSError):
                    self.transport_failed = True
                    self.enabled = False
                    self._clear_live()
                    return changed
                self._handle(event)
                changed = True
            if changed or (self.is_tty and self.active):
                self.render()
            return changed

        def _handle(self, event: dict[str, Any]) -> None:
            pid = int(event.get("pid", 0) or 0)
            if event.get("event") == "start":
                self.active[pid] = {
                    "index": int(event.get("index", 0) or 0),
                    "total": int(event.get("total", 0) or 0),
                    "model": str(event.get("model", "")),
                    "track_id": str(event.get("track_id", "")),
                    "started": time.perf_counter(),
                }
                return
            if event.get("event") == "done":
                item = self.active.pop(pid, None) or {
                    "index": int(event.get("index", 0) or 0),
                    "total": int(event.get("total", 0) or 0),
                    "model": str(event.get("model", "")),
                    "track_id": str(event.get("track_id", "")),
                }
                ok = bool(event.get("ok"))
                self._clear_live()
                print(self._line(item, "✓" if ok else "X"), flush=True)

        def render(self) -> None:
            if not self.is_tty:
                return
            now = time.perf_counter()
            if now - self.last_render < 0.08 and self.live_lines:
                return
            self._clear_live()
            lines = [
                self._line(item, self._spinner(now)) for item in self.active.values()
            ]
            if lines:
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
            self.live_lines = len(lines)
            self.last_render = now

        def _clear_live(self) -> None:
            if not self.is_tty or self.live_lines <= 0:
                self.live_lines = 0
                return
            for _ in range(self.live_lines):
                sys.stdout.write("\x1b[1A\x1b[2K")
            sys.stdout.flush()
            self.live_lines = 0

        def _spinner(self, now: float) -> str:
            return self.SPINNER[int(now * 10) % len(self.SPINNER)]

        def _line(self, item: dict[str, Any], status: str) -> str:
            total = int(item.get("total", 0) or 0)
            index = int(item.get("index", 0) or 0)
            model = str(item.get("model", ""))
            title = str(item.get("track_id", ""))
            line = f"[{index}/{total}] {model}: {title} [{status}]"
            width = shutil.get_terminal_size((120, 20)).columns
            if width > 20 and len(line) > width:
                line = line[: width - 1]
            return line

    @staticmethod
    def process_chunks(
        run_chunk: RunChunkFn,
        chunks: Sequence[WorkChunk],
        workers: int,
        batch_size: int,
        watchdog: float,
        max_attempts: int,
        kind: str,
        opts: dict[str, Any],
        verbose: bool,
        progress: bool,
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, str]], dict[str, str]]:
        rows: list[dict[str, Any]] = []
        errors: list[tuple[str, str, str, str]] = []
        skipped: dict[str, str] = {}
        attempts: dict[str, int] = {}
        queue = list(chunks)
        total_tracks = sum((len(items) for _, _, items in chunks))
        completed_tracks = 0
        started = time.perf_counter()
        manager = multiprocessing.Manager() if progress else None
        progress_queue = manager.Queue() if manager is not None else None
        renderer = NoteRunner.ProgressRenderer(enabled=progress)

        def chunk_key(chunk: WorkChunk) -> str:
            model, _, items = chunk
            if not items:
                return f"{model}:empty"
            return f"{model}:{items[0][0]}:{items[0][1]}:{items[-1][0]}:{items[-1][1]}:{len(items)}"

        def log(tag: str, model: str, n: int, dur: float, extra: str = "") -> None:
            nonlocal completed_tracks
            completed_tracks += n
            elapsed = time.perf_counter() - started
            rate = completed_tracks / elapsed if elapsed else 0.0
            eta = (total_tracks - completed_tracks) / rate if rate else 0.0
            print(
                f"[{completed_tracks:>4}/{total_tracks}] {tag:7s} {model:14s} {n:>4} track(s) in {NoteRunner.fmt_dur(dur)} {extra}| elapsed {NoteRunner.fmt_dur(elapsed)} eta {NoteRunner.fmt_dur(eta)}",
                flush=True,
            )

        try:
            while queue:
                batch, queue = (queue[:batch_size], queue[batch_size:])
                runnable: list[WorkChunk] = []
                for chunk in batch:
                    key = chunk_key(chunk)
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] > max_attempts:
                        model, _, items = chunk
                        tb = f"gave up after {max_attempts} attempts (kept hanging/failing)"
                        for dataset, track_id, *_ in items:
                            errors.append((model, dataset, track_id, tb))
                        renderer.close()
                        log("GIVEUP", model, len(items), 0.0)
                    else:
                        runnable.append(chunk)
                if not runnable:
                    continue
                ex = ProcessPoolExecutor(max_workers=workers)
                futs = {
                    ex.submit(
                        run_chunk,
                        chunk,
                        kind,
                        opts,
                        verbose,
                        progress_queue,
                        total_tracks,
                    ): chunk
                    for chunk in runnable
                }
                try:
                    pending = set(futs)
                    last_activity = time.perf_counter()
                    while pending:
                        done_set, pending = wait(
                            pending,
                            timeout=0.1 if progress_queue is not None else watchdog,
                            return_when=FIRST_COMPLETED,
                        )
                        if renderer.drain(progress_queue):
                            last_activity = time.perf_counter()
                        if renderer.transport_failed:
                            progress_queue = None
                        if not done_set:
                            if time.perf_counter() - last_activity < watchdog:
                                continue
                            stuck = [futs[fut] for fut in pending]
                            renderer.close()
                            watchdog_reason = (
                                "no track/chunk progress"
                                if progress
                                else "no chunk finished"
                            )
                            print(
                                f"\n!! watchdog: {watchdog_reason} in {NoteRunner.fmt_dur(watchdog)} -- re-queueing {len(stuck)} in-flight chunk(s).",
                                file=sys.stderr,
                                flush=True,
                            )
                            for model, _, items in stuck:
                                first = items[0][1] if items else "empty"
                                print(
                                    f"   stuck: {model} / {first} ({len(items)} tracks)",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            queue.extend(stuck)
                            break
                        last_activity = time.perf_counter()
                        broke = False
                        for fut in done_set:
                            chunk = futs[fut]
                            model, _, items = chunk
                            try:
                                (
                                    status,
                                    result_model,
                                    chunk_rows,
                                    chunk_errors,
                                    dur,
                                    skip_msg,
                                ) = fut.result()
                            except BrokenProcessPool:
                                broke = True
                                break
                            if status == "skip":
                                rows.extend(chunk_rows)
                                errors.extend(chunk_errors)
                                skipped[result_model] = (
                                    skip_msg or "dependency unavailable"
                                )
                                renderer.close()
                                log("SKIP", result_model, len(items), dur)
                                print(
                                    f"[{result_model}] SKIPPED -- {skipped[result_model]}",
                                    flush=True,
                                )
                                continue
                            rows.extend(chunk_rows)
                            errors.extend(chunk_errors)
                            if status == "ok":
                                if not progress:
                                    log(
                                        "OK",
                                        model,
                                        len(chunk_rows) + len(chunk_errors),
                                        dur,
                                    )
                            else:
                                renderer.close()
                                log("ERR", model, len(items), dur)
                                for _, _, _, tb in chunk_errors:
                                    print(tb.rstrip(), file=sys.stderr, flush=True)
                        if broke:
                            stuck = [chunk, *[futs[fut] for fut in pending]]
                            renderer.close()
                            print(
                                f"\n!! pool broke (a worker died) -- re-queueing {len(stuck)} unfinished chunk(s).",
                                file=sys.stderr,
                                flush=True,
                            )
                            queue.extend(stuck)
                            break
                finally:
                    renderer.drain(progress_queue)
                    renderer.close()
                    NoteRunner._force_teardown(ex)
        finally:
            renderer.close()
            if manager is not None:
                try:
                    manager.shutdown()
                except (BrokenPipeError, EOFError, OSError):
                    pass
        return (rows, errors, skipped)


F0_FPS_DEFAULT = CocoChorales.F0_FPS_DEFAULT
ALL_METHODS = list(_ALL_NOTE_METHODS)
CPD_METHODS = [m for m in ALL_METHODS if m not in _EXTERNAL_NOTE_METHODS]


class NoteCLI:

    @staticmethod
    def _is_missing_dependency(exc: Exception) -> bool:
        """External baselines (basic-pitch / tony / crepe-notes) raise a RuntimeError
        whose message flags an unmet dependency. Treat those as a whole-method skip
        rather than one failure per track, matching the pitch runner's behaviour."""
        text = str(exc).lower()
        return isinstance(
            exc, (RuntimeError, ImportError, ModuleNotFoundError, FileNotFoundError)
        ) and any(
            (
                needle in text
                for needle in (
                    "unavailable",
                    "not found",
                    "no module",
                    "not installed",
                    "missing",
                )
            )
        )

    @staticmethod
    def make_benchmarker(kind: str, opts: dict[str, Any] | None = None):
        opts = opts or {}
        if kind == "coco":
            return CocoNoteBenchmarker(
                root=opts.get("root"),
                f0_fps=float(opts.get("f0_fps", F0_FPS_DEFAULT)),
                onset_tolerance=float(opts.get("onset_tolerance", 0.05)),
            )
        return NoteBenchmarker(onset_tolerance=float(opts.get("onset_tolerance", 0.05)))

    @staticmethod
    def list_work(
        datasets: list[str],
        shard: tuple[int, int] | None = None,
        kind: str = "coco",
        opts: dict[str, Any] | None = None,
    ) -> list[TrackItem]:
        opts = opts or {}
        pb = NoteCLI.make_benchmarker(kind, opts)
        work: list[TrackItem] = []
        max_tracks = opts.get("max_tracks")
        if kind == "coco":
            for ds in datasets:
                records = pb.select_records(
                    split=ds,
                    per_stratum=opts.get("per_stratum"),
                    seed=int(opts.get("seed", 0)),
                    max_tracks=None,
                    ensembles=opts.get("ensembles"),
                    instruments=opts.get("instruments"),
                    rebuild_manifest=False,
                )
                if opts.get("materialize"):
                    to_extract = (
                        records[:max_tracks] if max_tracks is not None else records
                    )
                    written = pb.materialize_records(
                        to_extract, force=bool(opts.get("force_materialize", False))
                    )
                    print(f"materialized {len(written)} file(s) for {ds}", flush=True)
                for track_id, wav, midi in pb.records_to_note_tracks(records):
                    work.append((ds, track_id, str(wav), str(midi)))
        else:
            for ds in datasets:
                for title, midi in pb.iter_etudes(ds):
                    work.append((ds, title, str(midi), str(midi)))
        work.sort()
        if kind == "etude" and opts.get("instruments"):
            wanted = [str(name).lower() for name in opts["instruments"]]
            work = [
                item
                for item in work
                if any((name in item[1].lower() for name in wanted))
            ]
        if max_tracks is not None:
            work = work[: int(max_tracks)]
        if shard is not None:
            i, n = shard
            work = [w for idx, w in enumerate(work) if idx % n == i]
        return work

    @staticmethod
    def chunk_size_for(track_count: int, workers: int, tracks_per_task: int) -> int:
        if track_count <= 0:
            return 1
        if tracks_per_task > 0:
            return tracks_per_task
        return max(1, math.ceil(track_count / max(1, workers)))

    @staticmethod
    def build_chunks(
        methods: list[str], tracks: list[TrackItem], workers: int, tracks_per_task: int
    ) -> list[WorkChunk]:
        chunks: list[WorkChunk] = []
        progress_index = 1
        for method in methods:
            size = NoteCLI.chunk_size_for(len(tracks), workers, tracks_per_task)
            for start in range(0, len(tracks), size):
                chunks.append(
                    (
                        method,
                        progress_index + start,
                        tuple(tracks[start : start + size]),
                    )
                )
            progress_index += len(tracks)
        chunks.sort(
            key=lambda c: (c[0], c[2][0][0] if c[2] else "", c[2][0][1] if c[2] else "")
        )
        return chunks

    @staticmethod
    def run_chunk(
        chunk: WorkChunk,
        kind: str,
        opts: dict[str, Any],
        verbose: bool,
        progress_queue: Any | None = None,
        progress_total: int = 0,
    ) -> tuple[
        str,
        str,
        list[dict[str, Any]],
        list[tuple[str, str, str, str]],
        float,
        str | None,
    ]:
        method, first_progress_index, items = chunk
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        errors: list[tuple[str, str, str, str]] = []

        def progress(
            event: str, index: int, dataset: str, track_id: str, ok: bool | None = None
        ) -> None:
            if progress_queue is None:
                return
            try:
                progress_queue.put(
                    {
                        "event": event,
                        "pid": os.getpid(),
                        "index": index,
                        "total": progress_total,
                        "model": method,
                        "dataset": dataset,
                        "track_id": track_id,
                        "ok": ok,
                    }
                )
            except (BrokenPipeError, EOFError, OSError):
                return

        try:
            bench = NoteCLI.make_benchmarker(kind, opts)
            bench.algorithm_verbose = bool(opts.get("algorithm_verbose", False))
            method_config = bench.ALL_NOTE_METHODS[method]
        except Exception as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            return (
                "err",
                method,
                rows,
                [(method, "", "", tb)],
                time.perf_counter() - started,
                None,
            )
        for i, (dataset, track_id, primary, reference) in enumerate(items, start=1):
            progress_index = first_progress_index + i - 1
            progress("start", progress_index, dataset, track_id)
            try:
                row = bench.score_note_track(
                    primary,
                    reference,
                    method,
                    method_config,
                    align=opts.get("align", "identity"),
                    latency_align=bool(opts.get("latency_align", True)),
                    trim_boundaries=bool(opts.get("trim_boundaries", True)),
                    onset_tolerance=opts.get("onset_tolerance"),
                )
                row["track_id"] = track_id
                row["dataset"] = dataset
                row["method"] = method
                if kind == "coco":
                    row.update(bench.note_row_meta(primary))
                rows.append(row)
                progress(
                    "done",
                    progress_index,
                    dataset,
                    track_id,
                    ok=row.get("error") is None,
                )
                if verbose and progress_queue is None:
                    fval = row.get("F-measure", float("nan"))
                    rtf = row.get("realtime_factor", float("nan"))
                    print(
                        f"[{method}] {i:>4}/{len(items)} {dataset:16s} {track_id[:40]:40s} F={fval:.3f} {rtf:.0f}xRT",
                        flush=True,
                    )
            except Exception as exc:
                progress("done", progress_index, dataset, track_id, ok=False)
                if NoteCLI._is_missing_dependency(exc):
                    return (
                        "skip",
                        method,
                        rows,
                        errors,
                        time.perf_counter() - started,
                        str(exc),
                    )
                tb = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                errors.append((method, dataset, track_id, tb))
                print(
                    f"[{method}] {dataset} / {track_id} ERROR: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
        return ("ok", method, rows, errors, time.perf_counter() - started, None)

    @staticmethod
    def write_raw_outputs(
        pb: NoteBenchmarker, rows: list[dict[str, Any]], kind: str
    ) -> None:
        import pandas as pd

        full = pd.DataFrame(rows)
        for method, method_df in full.groupby("method", sort=False):
            if kind == "coco" and "instrument" in method_df.columns:
                groups = (
                    (f"coco_{instrument}", sub)
                    for instrument, sub in method_df.groupby("instrument", sort=True)
                )
            else:
                groups = (
                    (dataset, sub)
                    for dataset, sub in method_df.groupby("dataset", sort=True)
                )
            for dataset_label, sub in groups:
                out_df = sub.drop(
                    columns=["method", "dataset"], errors="ignore"
                ).set_index("track_id")
                out = pb.write_note_result(out_df, method, dataset_label)
                print(f"wrote {len(out_df)} rows -> {out}")

    @staticmethod
    def write_summary(
        pb: NoteBenchmarker, rows: list[dict[str, Any]], method_order: list[str]
    ) -> None:
        import pandas as pd

        full = pd.DataFrame(rows)
        metric_cols = [c for c in pb.NOTE_QUALITY_METRICS if c in full.columns]
        cols = [*metric_cols, "realtime_factor"]
        table = full.groupby("method")[[c for c in cols if c in full.columns]].mean(
            numeric_only=True
        )
        table.insert(0, "Tracks", full.groupby("method").size())
        table = table.rename(columns={"realtime_factor": pb.REALTIME_COL})
        table = table.reindex(
            [m for m in method_order if m in table.index]
            + [m for m in table.index if m not in method_order]
        )
        table.index.name = "method"
        print(f"\n{'=' * 72}\nnote_benchmarks.csv (mean over tracks)\n{'=' * 72}")
        with pd.option_context(
            "display.float_format", lambda v: f"{v:.4f}", "display.width", 200
        ):
            print(table.to_string())
        out = pb.note_summary_csv_path
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out)
        print(f"\nwrote summary -> {out}")

    @staticmethod
    def select_methods(args: argparse.Namespace) -> list[str]:
        if args.methods:
            unknown = [m for m in args.methods if m not in _ALL_NOTE_METHODS]
            if unknown:
                raise SystemExit(f"unknown methods: {unknown}; choices: {ALL_METHODS}")
            methods = list(args.methods)
        elif args.cpd_only:
            methods = list(CPD_METHODS)
        else:
            methods = list(ALL_METHODS)
        if args.no_external:
            methods = [m for m in methods if m not in _EXTERNAL_NOTE_METHODS]
        return methods

    @staticmethod
    def print_method_matrix() -> None:
        for label, config in _ALL_NOTE_METHODS.items():
            family = config.get("method")
            algo = config.get("ruptures_algorithm", "-")
            model = config.get("model", "-")
            post = config.get("postprocess_transitions", False)
            group = config.get("benchmark_group", "-")
            print(
                f"{label:22s} group={group:9s} family={family:12s} algorithm={algo:9s} model={model:7s} prune+remedian={post}"
            )

    @staticmethod
    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        p.add_argument(
            "--benchmarker",
            choices=["coco", "etude"],
            default="coco",
            help="'coco' runs materialized CocoChorales stems (default); 'etude' runs violin etudes",
        )
        p.add_argument(
            "--datasets",
            nargs="+",
            default=None,
            help="splits/datasets. Default: 'test' for coco, all etude datasets for etude.",
        )
        p.add_argument(
            "--method",
            action="append",
            dest="methods",
            help="method label to run; repeatable; default = the full matrix",
        )
        p.add_argument(
            "--cpd-only",
            action="store_true",
            help="run only the change-point families (no moreover baselines)",
        )
        p.add_argument(
            "--no-external",
            action="store_true",
            help="drop basic-pitch / tony / crepe-notes (unmet deps)",
        )
        p.add_argument(
            "--list-methods",
            action="store_true",
            help="print the method matrix and exit",
        )
        p.add_argument(
            "--workers",
            type=int,
            default=max(1, os.cpu_count() or 4),
            help="process pool size (default: all logical CPUs; BLAS threads capped to 1/worker)",
        )
        p.add_argument(
            "--align",
            choices=["identity", "resize"],
            default="identity",
            help="identity isolates note detection; resize mirrors the app pipeline",
        )
        p.add_argument(
            "--no-latency-align",
            action="store_true",
            help="disable the constant detector-lag shift (identity mode)",
        )
        p.add_argument(
            "--no-trim-boundaries",
            action="store_true",
            help="do not clamp first/last notes to reference durations",
        )
        p.add_argument(
            "--onset-tolerance",
            type=float,
            default=0.05,
            help="mir_eval onset match tolerance in seconds",
        )
        p.add_argument(
            "--instrument",
            action="append",
            dest="instruments",
            help="filter tracks by instrument; repeatable",
        )
        p.add_argument(
            "--ensemble",
            action="append",
            dest="ensembles",
            help="CocoChorales ensemble filter; repeatable",
        )
        p.add_argument(
            "--max-tracks",
            type=int,
            default=None,
            help="cap selected tracks after filtering",
        )
        p.add_argument(
            "--per-stratum",
            type=int,
            default=None,
            help="CocoChorales sample cap per (ensemble,instrument)",
        )
        p.add_argument("--seed", type=int, default=0, help="CocoChorales sampling seed")
        p.add_argument(
            "--materialize",
            action="store_true",
            help="CocoChorales-only: extract selected stems before benchmarking",
        )
        p.add_argument(
            "--force-materialize",
            action="store_true",
            help="CocoChorales-only: overwrite materialized stems",
        )
        p.add_argument(
            "--root", default=None, help="CocoChorales dataset root override"
        )
        p.add_argument(
            "--f0-fps",
            type=float,
            default=F0_FPS_DEFAULT,
            help="CocoChorales f0 frame rate",
        )
        p.add_argument(
            "--shard",
            type=NoteRunner.parse_shard,
            default=None,
            help="run only shard i/N (e.g. 0/2)",
        )
        p.add_argument(
            "--tracks-per-task",
            type=int,
            default=0,
            help="tracks per worker task (0 = auto)",
        )
        p.add_argument(
            "--batch-size",
            type=int,
            default=0,
            help="chunks per fresh process pool (0 = workers*2)",
        )
        p.add_argument(
            "--watchdog",
            type=float,
            default=1200.0,
            help="seconds with no progress before in-flight chunks are re-queued",
        )
        p.add_argument(
            "--max-attempts",
            type=int,
            default=2,
            help="retries for a stuck/failed chunk before giving up",
        )
        p.add_argument(
            "--quiet-tracks",
            action="store_true",
            help="suppress legacy per-track worker lines",
        )
        p.add_argument(
            "--no-progress",
            action="store_true",
            help="disable the live per-track progress display",
        )
        p.add_argument(
            "--algorithm-verbose",
            action="store_true",
            help="let algorithms print their own diagnostics inside workers",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="print the work plan and exit; detect nothing",
        )
        return p.parse_args()

    @staticmethod
    def main() -> int:
        args = NoteCLI.parse_args()
        if args.list_methods:
            NoteCLI.print_method_matrix()
            return 0
        if args.datasets is None:
            args.datasets = (
                ["test"]
                if args.benchmarker == "coco"
                else NoteBenchmarker.ETUDE_DATASETS
            )
        methods = NoteCLI.select_methods(args)
        opts: dict[str, Any] = {
            "root": args.root,
            "f0_fps": args.f0_fps,
            "onset_tolerance": args.onset_tolerance,
            "align": args.align,
            "latency_align": not args.no_latency_align,
            "trim_boundaries": not args.no_trim_boundaries,
            "instruments": args.instruments,
            "ensembles": args.ensembles,
            "max_tracks": args.max_tracks,
            "per_stratum": args.per_stratum,
            "seed": args.seed,
            "materialize": args.materialize and (not args.dry_run),
            "force_materialize": args.force_materialize,
            "algorithm_verbose": args.algorithm_verbose,
        }
        tracks = NoteCLI.list_work(
            args.datasets, args.shard, kind=args.benchmarker, opts=opts
        )
        chunks = NoteCLI.build_chunks(
            methods, tracks, workers=args.workers, tracks_per_task=args.tracks_per_task
        )
        batch_size = (
            args.batch_size if args.batch_size > 0 else max(1, args.workers * 2)
        )
        pb = NoteCLI.make_benchmarker(args.benchmarker, opts)
        cached_input = 0
        if args.benchmarker == "coco":
            cached_input = sum(
                (1 for _, _, wav, _ in tracks if pb.has_note_input_cache(wav))
            )
        print(f"benchmarker: {args.benchmarker} (note detection)")
        print(f"datasets:   {', '.join(args.datasets)}")
        print(f"methods:    {len(methods)} | {', '.join(methods)}")
        if args.instruments:
            print(f"instruments:{' ' * 3}{', '.join(args.instruments)}")
        if args.ensembles:
            print(f"ensembles:  {', '.join(args.ensembles)}")
        print(
            f"align:      {args.align} (latency_align={not args.no_latency_align}, trim={not args.no_trim_boundaries})"
        )
        print(
            f"shard:      {(f'{args.shard[0]}/{args.shard[1]}' if args.shard else 'all')}"
        )
        print(
            f"workers:    {args.workers}  (cpu_count={os.cpu_count()}, BLAS threads capped to 1/worker)"
        )
        print(
            f"tracks:     {len(tracks)} corpus tracks | {len(methods) * len(tracks)} method/track pairs"
        )
        if args.benchmarker == "coco":
            print(
                f"pitch input:{' ' * 1}{cached_input}/{len(tracks)} tracks have a cached pyin_smooth track"
            )
        print(
            f"chunks:     {len(chunks)} total | batch={batch_size} | watchdog={NoteRunner.fmt_dur(args.watchdog)}"
        )
        if args.dry_run:
            print(
                f"\n[dry-run] would process {len(methods) * len(tracks)} method/track pair(s); no detection performed."
            )
            return 0
        if not tracks:
            print("\nno tracks selected.")
            if args.benchmarker == "coco":
                print(
                    "CocoChorales may need --materialize (extract selected stems first)."
                )
            return 1
        if not chunks:
            print("\nnothing to do.")
            return 0
        print(
            f"\nstarting {len(methods) * len(tracks)} method/track pair(s) at {time.strftime('%Y-%m-%d %H:%M:%S')} ...\n",
            flush=True,
        )
        started = time.perf_counter()
        rows, errors, skipped = NoteRunner.process_chunks(
            NoteCLI.run_chunk,
            chunks,
            workers=args.workers,
            batch_size=batch_size,
            watchdog=args.watchdog,
            max_attempts=args.max_attempts,
            kind=args.benchmarker,
            opts=opts,
            verbose=not args.quiet_tracks,
            progress=not args.no_progress,
        )
        total = time.perf_counter() - started
        if rows:
            NoteCLI.write_raw_outputs(pb, rows, args.benchmarker)
            NoteCLI.write_summary(pb, rows, methods)
        print(
            f"\ndone: {len(rows)} rows, {len(errors)} track errors, {len(skipped)} skipped method(s) in {NoteRunner.fmt_dur(total)}"
        )
        if skipped:
            print("skipped methods:")
            for method, reason in sorted(skipped.items()):
                first_line = (
                    reason.splitlines()[0] if reason else "dependency unavailable"
                )
                print(f"  - {method}: {first_line}")
        if errors:
            print("failed tracks:")
            for method, dataset, track_id, _ in errors[:40]:
                print(f"  - {method} / {dataset} / {track_id}")
            if len(errors) > 40:
                print(f"  ... and {len(errors) - 40} more")
            return 1
        return 0


VERSION = "note-preliminary-v3-repeat-recovery"
METHODS = ("attune", "basic-pitch", "crepe-notes", "tony")
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)


@dataclass(frozen=True)
class NotebookConfig:
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 4) - 1))
    neural_workers: int = 6
    preliminary_tracks_per_instrument: int = 2
    seed: int = 0
    max_tracks: int | None = None
    methods: tuple[str, ...] = METHODS
    onset_tolerance: float = 0.05
    pitch_tolerance: float = 50.0
    offset_ratio: float = 0.2
    offset_min_tolerance: float = 0.05
    force: bool = False
    force_methods: tuple[str, ...] = ()
    use_pitch_cache: bool = True
    audio_only_fmin: float = 32.7032
    audio_only_fmax: float = 2093.005
    audio_only_min_note: float = 0.03

    def __post_init__(self):
        if (
            min(
                self.workers,
                self.neural_workers,
                self.preliminary_tracks_per_instrument,
            )
            < 1
        ):
            raise ValueError("Worker and sampling counts must be positive")
        if set(self.force_methods) - {*METHODS, "attune-audio-only"}:
            raise ValueError(f"Unknown forced methods: {self.force_methods}")
        if not self.methods or set(self.methods) - {*METHODS, "attune-audio-only"}:
            raise ValueError(f"Unknown or empty methods: {self.methods}")


class NoteEvaluation:

    def __init__(self, config=None):
        self.config = config or NotebookConfig()
        self.runs_root = RESULTS_ROOT / "note" / "notebook_runs"

    def select_tracks(self, dataset):
        from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
        from benchmarks.modules.pitch.datasets.URMP import URMP
        from benchmarks.modules.pitch.datasets.Bach10 import Bach10

        config = self.config
        result = []

        def pitch_fields(track):
            example = source.example(track)
            return dict(
                pitch_fmin=example.fmin,
                pitch_fmax=example.fmax,
                pitch_annotation=str(track.annot_path),
                pitch_cache=str(example.stage_cache_path),
                crepe_pitch_cache=str(
                    example.estimate_cache_dir / f"crepe__{example.safe_id}.npz"
                ),
            )

        if dataset == "coco":
            source = CocoChorales(
                split="test",
                per_instrument=config.preliminary_tracks_per_instrument,
                seed=config.seed,
            )
            records = source.select_records()
            source.materialize_records(records)
            from benchmarks.modules.note.NoteBenchmarker import CocoNoteBenchmarker

            locator = CocoNoteBenchmarker(root=source.root)
            pitch_tracks = {t.track_id: t for t in source.records_to_tracks(records)}
            for r in records:
                audio, reference = (
                    source.local_wav_path(r),
                    locator.local_midi_path(r),
                )
                if audio is None or reference is None:
                    raise FileNotFoundError(f"Missing selected Coco stem: {r.track_id}")
                result.append(
                    dict(
                        dataset="coco",
                        track_id=r.track_id,
                        group=r.track,
                        instrument=r.instrument,
                        audio=str(audio),
                        reference=str(reference),
                        score=str(reference),
                        score_part=0,
                        **pitch_fields(pitch_tracks[r.track_id]),
                    )
                )
        elif dataset == "urmp":
            source = URMP(
                per_instrument=config.preliminary_tracks_per_instrument,
                seed=config.seed,
            )
            for t in source.tracks():
                ref = t.audio_path.with_name(
                    t.audio_path.name.replace("AuSep_", "Notes_")
                ).with_suffix(".txt")
                if not ref.is_file():
                    raise FileNotFoundError(ref)
                scores = list(t.audio_path.parent.glob("Sco_*.mid"))
                if len(scores) != 1:
                    raise FileNotFoundError(
                        f"Expected one URMP score in {t.audio_path.parent}"
                    )
                result.append(
                    dict(
                        dataset="urmp",
                        track_id=t.track_id,
                        group=t.audio_path.parent.name,
                        instrument=t.metadata["instrument"],
                        audio=str(t.audio_path),
                        reference=str(ref),
                        score=str(scores[0]),
                        score_part=int(t.metadata["voice"]) - 1,
                        **pitch_fields(t),
                    )
                )
        elif dataset == Bach10.name:
            source = Bach10(
                per_instrument=config.preliminary_tracks_per_instrument,
                seed=config.seed,
            )
            for t in source.tracks():
                piece = t.metadata["track"]
                ref = t.audio_path.with_name(f"{piece}-GTNotes.mat")
                score = t.audio_path.with_name(f"{piece}.mid")
                for path in (ref, score):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                result.append(
                    dict(
                        dataset=Bach10.name,
                        track_id=t.track_id,
                        group=piece,
                        instrument=t.metadata["instrument"],
                        audio=str(t.audio_path),
                        reference=str(ref),
                        reference_part=int(t.metadata["f0_row"]),
                        reference_policy="GTNotes frame centers; median fractional MIDI pitch",
                        score=str(score),
                        score_part=int(t.metadata["f0_row"]),
                        **pitch_fields(t),
                    )
                )
        else:
            raise ValueError(dataset)
        if not result:
            raise FileNotFoundError(f"No {dataset} note tracks found")
        return result[: config.max_tracks] if config.max_tracks is not None else result

    def provenance(self, tracks):
        from benchmarks.modules.note.NoteCache import NoteCache

        versions = {}
        for name in (
            "numpy",
            "scipy",
            "librosa",
            "mir_eval",
            "ruptures",
            "basic-pitch",
            "crepe",
            "crepe-notes",
            "tensorflow",
            "pretty_midi",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        files = [
            p
            for directory in (
                "algorithms",
                "app_logic",
                "benchmarks/modules/note",
                "benchmarks/modules/pitch",
            )
            for p in (REPO_ROOT / directory).rglob("*.py")
        ]
        sources = {
            str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)
        }
        inputs = {
            t[k]: hashlib.sha256(Path(t[k]).read_bytes()).hexdigest()
            for t in tracks
            for k in ("audio", "reference", "score", "pitch_annotation")
        }
        settings = asdict(self.config)
        settings.pop("force")
        settings.pop("force_methods")
        return dict(
            version=VERSION,
            config=settings,
            versions=versions,
            sources=sources,
            inputs=inputs,
            cache_functions=NoteCache.function_fingerprints(Path(__file__)),
            python=sys.version,
            platform=platform.platform(),
            cpu_count=os.cpu_count(),
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            threads=1,
            device="CPU",
            score_policy="attune: supplied MIDI score part, pitch-notebook F0 range, duration prior, alignment and repeat recovery; others: audio-only",
            cache_policy="shared PitchCache version + production smoother version + process_cpu timing; no old wall timings",
            compatibility="scipy.signal.gaussian aliases signal.windows.gaussian for Basic Pitch",
            predictions="unshifted, untrimmed; all notes retained",
            tracks=tracks,
        )

    def run_preliminary(self, dataset):
        import pandas as pd
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        from benchmarks.modules.note.NoteCache import NoteCache

        config = self.config
        tracks = self.select_tracks(dataset)
        provenance = self.provenance(tracks)
        reusable = {} if config.force else NoteCache.checkpoint_index(self.runs_root)
        digest = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()[:16]
        tier = "smoke" if config.max_tracks is not None else "preliminary"
        output = self.runs_root / f"{tier}_{dataset}_{digest}"
        output.mkdir(parents=True, exist_ok=True)
        for directory in ("predictions", "checkpoints"):
            (output / directory).mkdir(exist_ok=True)
        (output / "metadata.json").write_text(json.dumps(provenance, indent=2))
        pd.DataFrame(tracks).to_csv(output / "selection.csv", index=False)
        for name in THREAD_ENV:
            os.environ[name] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/attune-note-matplotlib")
        rows = []
        jobs_by_method = {}
        cached_by_method = {}
        for method in config.methods:
            jobs = []
            cached_by_method[method] = []
            for track in tracks:
                job_id = hashlib.sha256(
                    (track["track_id"] + method).encode()
                ).hexdigest()[:20]
                task = {**track, "method": method, "job_id": job_id}
                task["cache_key"] = NoteCache.job_key(provenance, task)
                if method not in config.force_methods and task["cache_key"] in reusable:
                    cached = NoteCache.reuse_checkpoint(
                        reusable[task["cache_key"]], task, output
                    )
                    rows.append(cached)
                    cached_by_method[method].append(cached)
                    continue
                jobs.append((task, config, str(output)))
            jobs_by_method[method] = jobs
        pending = sum((len(jobs) for jobs in jobs_by_method.values()))
        total = len(tracks) * len(config.methods)
        if pending:
            print(
                f"Running: {tier}_{dataset} ({len(rows)} cached, {pending} pending)",
                flush=True,
            )
        else:
            print("Reusing completed run:", output, flush=True)
        progress = PitchBenchmarker.Progress(enabled=True, compact=True)
        try:
            for method, jobs in jobs_by_method.items():
                for row in cached_by_method[method]:
                    progress.update_compact(total, row["method"], row["track_id"])
                if not jobs:
                    continue
                workers = (
                    min(config.workers, config.neural_workers)
                    if method in ("basic-pitch", "crepe-notes")
                    else config.workers
                )
                with mp.get_context("spawn").Pool(
                    min(workers, len(jobs)), maxtasksperchild=1
                ) as pool:
                    for row in pool.imap_unordered(
                        NoteEvaluation._worker, jobs, chunksize=1
                    ):
                        rows.append(row)
                        checkpoint = output / "checkpoints" / f"{row['job_id']}.json"
                        NoteCache.atomic_json(checkpoint, row)
                        progress.update_compact(total, row["method"], row["track_id"])
        finally:
            progress.finish()
        for row in rows:
            if row["status"] != "ok":
                print(f"  error {row['method']} / {dataset} / {row['track_id']}")
        for row in rows:
            row.update(NoteEvaluation.note_event_counts(row))
        frame = pd.DataFrame(rows).sort_values(["method", "track_id"])
        frame.to_csv(output / "rows.csv", index=False)
        summary = self.summarize(frame)
        summary.to_csv(output / "summary.csv", index=False)
        self.paired_comparisons(frame).to_csv(output / "paired.csv", index=False)
        print("Reports:", output, flush=True)
        return frame

    @staticmethod
    def summarize(rows, by_dataset=True):
        """Pool matched-note counts; runtime columns remain per-recording means."""
        import pandas as pd

        result = []
        keys = ["dataset", "method"] if by_dataset else ["method"]
        for identity, data in rows.groupby(keys):
            identity = identity if isinstance(identity, tuple) else (identity,)
            good = data[data.status == "ok"]
            row = dict(zip(keys, identity))
            row.update(
                attempted=len(data),
                succeeded=len(good),
                failed=len(data) - len(good),
                datasets_scored=good.dataset.nunique(),
                complete=len(good) == len(data),
                aggregation="pooled_note_counts",
            )
            counts = [
                NoteEvaluation.note_event_counts(record)
                for record in good.to_dict("records")
            ]
            for suffix in ("", "_offset"):
                tp, fp, fn = [
                    sum((c[f"{name}{suffix}"] for c in counts))
                    for name in ("tp", "fp", "fn")
                ]
                row.update({f"tp{suffix}": tp, f"fp{suffix}": fp, f"fn{suffix}": fn})
                for metric, numerator, denominator in (
                    ("precision", tp, tp + fp),
                    ("recall", tp, tp + fn),
                    ("f1", 2 * tp, 2 * tp + fp + fn),
                ):
                    row[f"{metric}{suffix}"] = (
                        (numerator / denominator if denominator else 0.0)
                        if counts
                        else float("nan")
                    )
            for metric in (
                "cpu_seconds",
                "wall_seconds",
                "execution_cpu_seconds",
                "frontend_cpu_seconds",
                "segmentation_cpu_seconds",
                "refinement_cpu_seconds",
            ):
                row[metric] = good[metric].mean() if metric in good else float("nan")
            row["audio_per_cpu"] = (
                good.audio_seconds.sum() / good.cpu_seconds.sum()
                if len(good) and good.cpu_seconds.sum() > 0
                else float("nan")
            )
            result.append(row)
        return pd.DataFrame(result)

    def paired_comparisons(self, rows):
        """Paired micro-F1 differences; bootstrap whole pieces, pooling counts."""
        import numpy as np
        import pandas as pd

        result = []

        def f1(counts):
            tp, fp, fn = counts
            return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

        for dataset, data in rows.groupby("dataset"):
            good = data[data.status == "ok"].copy()
            if good.empty:
                continue
            counts = pd.DataFrame(
                [
                    NoteEvaluation.note_event_counts(row)
                    for row in good.to_dict("records")
                ],
                index=good.index,
            )
            for column in counts:
                good[column] = counts[column]
            for metric, suffix in (("f1", ""), ("f1_offset", "_offset")):
                columns = [name + suffix for name in ("tp", "fp", "fn")]
                for method in self.config.methods:
                    if method.startswith("attune"):
                        continue
                    left = good[good.method == "attune"].set_index(
                        ["track_id", "group"]
                    )[columns]
                    right = good[good.method == method].set_index(
                        ["track_id", "group"]
                    )[columns]
                    pair = left.join(right, how="inner", lsuffix="_a", rsuffix="_b")
                    if pair.empty:
                        continue
                    groups = np.array(
                        [
                            group.to_numpy().sum(axis=0)
                            for _, group in pair.groupby(level="group")
                        ]
                    )
                    totals = groups.sum(axis=0)
                    delta = f1(totals[:3]) - f1(totals[3:])
                    rng = np.random.default_rng(self.config.seed)
                    draws = []
                    for _ in range(2000):
                        sample = groups[
                            rng.integers(len(groups), size=len(groups))
                        ].sum(axis=0)
                        draws.append(f1(sample[:3]) - f1(sample[3:]))
                    low, high = (
                        np.quantile(draws, [0.025, 0.975])
                        if len(groups) > 1
                        else (np.nan, np.nan)
                    )
                    result.append(
                        dict(
                            dataset=dataset,
                            baseline=method,
                            metric=metric,
                            aggregation="pooled_note_counts",
                            paired_tracks=len(pair),
                            groups=len(groups),
                            delta=delta,
                            ci_low=low,
                            ci_high=high,
                            complete=len(pair) == data.track_id.nunique(),
                        )
                    )
        return pd.DataFrame(result)

    @staticmethod
    def reference_notes(task):
        import numpy as np
        import pretty_midi

        if task["dataset"] == "urmp":
            a = np.loadtxt(task["reference"], ndmin=2)
            if a.shape[1] != 3:
                raise ValueError("URMP Notes must have onset, Hz, duration columns")
            intervals = np.column_stack((a[:, 0], a[:, 0] + a[:, 2]))
            pitches = a[:, 1]
        elif task["dataset"] == "bach10-original":
            from scipy.io import loadmat

            parts = loadmat(task["reference"])["GTNotes"]
            notes = parts[int(task["reference_part"]), 0].ravel()
            intervals, pitches = ([], [])
            for note in notes:
                note = np.asarray(note, dtype=float)
                if (
                    note.ndim != 2
                    or note.shape[0] != 2
                    or note.shape[1] < 2
                    or (not np.isfinite(note).all())
                ):
                    raise ValueError("Invalid Bach10 note matrix")
                frames, midi = note
                if (
                    frames[0] < 1
                    or (frames != np.floor(frames)).any()
                    or (np.diff(frames) != 1).any()
                    or (midi <= 0).any()
                ):
                    raise ValueError("Invalid Bach10 note frames or pitches")
                intervals.append(0.023 + (frames[[0, -1]] - 1) * 0.01)
                pitches.append(440.0 * 2.0 ** ((np.median(midi) - 69.0) / 12.0))
            intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
            pitches = np.asarray(pitches, dtype=float)
        else:
            midi = pretty_midi.PrettyMIDI(task["reference"])
            notes = [
                n for ins in midi.instruments if not ins.is_drum for n in ins.notes
            ]
            intervals = np.array(
                [[n.start, n.end] for n in notes], dtype=float
            ).reshape(-1, 2)
            pitches = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
        if (
            not len(pitches)
            or not np.isfinite(intervals).all()
            or (not np.isfinite(pitches).all())
        ):
            raise ValueError("Empty or non-finite reference")
        if (
            (pitches <= 0).any()
            or (intervals[:, 0] < 0).any()
            or (intervals[:, 1] <= intervals[:, 0]).any()
        ):
            raise ValueError("Invalid reference note")
        order = np.argsort(intervals[:, 0], kind="stable")
        return (intervals[order], pitches[order])

    @staticmethod
    def score_predictions(ref_iv, ref_hz, est_iv, est_hz, config):
        import mir_eval

        out = {}
        for suffix, ratio in (("", None), ("_offset", config.offset_ratio)):
            p, r, f, overlap = mir_eval.transcription.precision_recall_f1_overlap(
                ref_iv,
                ref_hz,
                est_iv,
                est_hz,
                onset_tolerance=config.onset_tolerance,
                pitch_tolerance=config.pitch_tolerance,
                offset_ratio=ratio,
                offset_min_tolerance=config.offset_min_tolerance,
            )
            out.update(
                {
                    f"precision{suffix}": p,
                    f"recall{suffix}": r,
                    f"f1{suffix}": f,
                    f"overlap{suffix}": overlap,
                }
            )
        return out

    @staticmethod
    def note_event_counts(row):
        """Recover exact match counts from full-precision per-track metrics.

        mir_eval recall = matches / reference count. Verify precision as well before
        migrating saved rows, so rounded or incompatible metrics cannot be pooled.
        Matching remains within each recording, never across dataset boundaries.
        """
        import math

        if row["status"] != "ok":
            return {}
        reference, estimated = (
            int(row["reference_notes"]),
            int(row["estimated_notes"]),
        )
        counts = {}
        for suffix in ("", "_offset"):
            recall, precision = (
                float(row[f"recall{suffix}"]),
                float(row[f"precision{suffix}"]),
            )
            tp = round(recall * reference)
            expected_recall = tp / reference if reference else 0.0
            expected_precision = tp / estimated if estimated else 0.0
            if (
                not 0 <= tp <= min(reference, estimated)
                or not math.isclose(recall, expected_recall, abs_tol=1e-10)
                or (not math.isclose(precision, expected_precision, abs_tol=1e-10))
            ):
                raise ValueError("Cannot recover exact note counts from saved metrics")
            counts.update(
                {
                    f"tp{suffix}": tp,
                    f"fp{suffix}": estimated - tp,
                    f"fn{suffix}": reference - tp,
                }
            )
        return counts

    @staticmethod
    @contextlib.contextmanager
    def _worker_log(path):
        """Keep Python and native-library chatter in the per-job log."""
        with path.open("w") as log:
            saved = []
            try:
                for stream in (sys.stdout, sys.stderr):
                    stream.flush()
                for fd in (1, 2):
                    saved.append((fd, os.dup(fd)))
                    os.dup2(log.fileno(), fd)
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    yield
            finally:
                log.flush()
                for fd, original in saved:
                    try:
                        os.dup2(original, fd)
                    finally:
                        os.close(original)

    @staticmethod
    def _worker(payload):
        task, config, output = payload
        import traceback
        from threadpoolctl import threadpool_limits
        import soundfile as sf

        row = {
            **task,
            "status": "error",
            "error": "",
            "compute_clock": "process+reaped_children",
            "timing_scope": "frontend + note-stage CPU; cached frontend retains original CPU cost; Tony uses complete call",
            "score_conditioned": task["method"] == "attune",
        }
        log_path = Path(output) / "logs" / f"{task['job_id']}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with NoteEvaluation._worker_log(log_path):
            try:
                ref_iv, ref_hz = NoteEvaluation.reference_notes(task)
                info = sf.info(task["audio"])
                row["audio_seconds"] = info.duration
                row["reference_notes"] = len(ref_hz)
                with threadpool_limits(limits=1):
                    cpu, wall = (NoteDetectorBase.cpu_seconds(), time.perf_counter())
                    iv, hz, timings = NoteEvaluation._predict(task, config)
                    row["execution_cpu_seconds"] = NoteDetectorBase.cpu_seconds() - cpu
                    row["wall_seconds"] = time.perf_counter() - wall
                row.update(timings)
                row["cpu_seconds"] = (
                    timings["frontend_cpu_seconds"]
                    + timings["segmentation_cpu_seconds"]
                    + timings.get("refinement_cpu_seconds", 0.0)
                    if "segmentation_cpu_seconds" in timings
                    else row["execution_cpu_seconds"]
                )
                row["audio_per_cpu"] = info.duration / row["cpu_seconds"]
                row["estimated_notes"] = len(hz)
                row.update(
                    NoteEvaluation.score_predictions(ref_iv, ref_hz, iv, hz, config)
                )
                row["status"] = "ok"
                row.update(NoteEvaluation.note_event_counts(row))
                prediction = {"intervals": iv.tolist(), "frequencies_hz": hz.tolist()}
                (Path(output) / "predictions" / f"{task['job_id']}.json").write_text(
                    json.dumps(prediction)
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        return row

    @staticmethod
    def _predict(task, config):
        return NoteDetectorBase.competitor(task["method"]).predict_task(task, config)

    @staticmethod
    def paired_significance(
        rows,
        *,
        methods=("attune", "basic-pitch", "crepe-notes", "tony"),
        metrics=("f1",),
        n_resamples=9999,
        seed=0,
        alpha=0.05,
        source_groups=None,
    ):
        """Return Holm-adjusted comparisons and per-dataset completion coverage.

        Use the common successful track intersection for every comparison. Each
        permutation swaps all counts within a source group and recomputes micro-F1;
        method-specific prediction counts mean its denominator must also be swapped.
        Holm covers all requested competitor/metric pairs in this call.
        source_groups optionally maps (dataset, original group) to shared cluster IDs,
        including known compositions shared across datasets.
        """
        methods = tuple(dict.fromkeys(methods))
        metrics = tuple(dict.fromkeys(metrics))
        if "attune" not in methods or len(methods) < 2:
            raise ValueError("Select Attune and at least one competitor")
        if not metrics or set(metrics) - {"f1", "f1_offset"}:
            raise ValueError("Metrics must be f1 and/or f1_offset")
        if not isinstance(n_resamples, int) or n_resamples < 99 or (not 0 < alpha < 1):
            raise ValueError("Use at least 99 resamples and 0 < alpha < 1")
        suffixes = {"f1": "", "f1_offset": "_offset"}
        columns = [name + suffixes[m] for m in metrics for name in ("tp", "fp", "fn")]
        keys = ["dataset", "track_id"]
        required = {*keys, "group", "method", "status", *columns}
        if not required.issubset(rows.columns):
            raise ValueError(
                f"Missing columns: {sorted(required - set(rows.columns))}; rerun result cells"
            )
        selected = rows.loc[rows.method.isin(methods)].copy()
        if selected.empty or selected[keys + ["method"]].isna().any().any():
            raise ValueError("Missing recording identities or no selected results")
        if selected.duplicated([*keys, "method"]).any():
            raise ValueError(
                "Duplicate method/track rows: select one run per recording"
            )
        good = selected.loc[selected.status.eq("ok")].copy()
        if good.empty:
            raise ValueError("No tracks completed by every selected method")
        values = good[columns].to_numpy(float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(
                "Successful rows require finite nonnegative integer note counts"
            )
        good[columns] = values
        if good["group"].isna().any():
            raise ValueError("Successful rows require source groups")
        overrides = source_groups or {}

        def source_id(row):
            original = (row.dataset, row.group)
            if original in overrides:
                return str(overrides[original])
            group = str(row.group)
            if row.dataset == "urmp" and len(group.split("_")) >= 3:
                group = group.split("_")[1]
            return f"{row.dataset}:{group}"

        good["source_piece"] = good.apply(source_id, axis=1)
        coverage = good.groupby(keys).method.nunique()
        common = coverage.index[coverage.eq(len(methods))]
        report = []
        for dataset, data in selected.groupby("dataset", sort=True):
            total = data.track_id.nunique()
            paired = sum((key[0] == dataset for key in common))
            for method in methods:
                attempted = data.loc[data.method.eq(method)]
                report.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        selected_tracks=total,
                        attempted=len(attempted),
                        succeeded=int(attempted.status.eq("ok").sum()),
                        paired_tracks=paired,
                        excluded_tracks=total - paired,
                    )
                )
        if not len(common):
            raise ValueError("No tracks completed by every selected method")
        paired = good.set_index(keys).loc[common].reset_index()
        if paired.groupby(keys).source_piece.nunique().gt(1).any():
            raise ValueError("Paired methods disagree on source groups")
        for metric in metrics:
            suffix = suffixes[metric]
            paired["_reference_count"] = paired["tp" + suffix] + paired["fn" + suffix]
            if paired.groupby(keys)["_reference_count"].nunique().gt(1).any():
                raise ValueError("Paired methods disagree on reference-note counts")
        clusters = sorted(paired.source_piece.unique())
        if len(clusters) < 2:
            raise ValueError("At least two independent source groups are required")

        def f1(counts):
            numerator = 2 * counts[..., 0]
            denominator = numerator + counts[..., 1] + counts[..., 2]
            return np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            )

        records = []
        for method in methods:
            if method == "attune":
                continue
            for metric in metrics:
                cols = [name + suffixes[metric] for name in ("tp", "fp", "fn")]
                a, b = [
                    paired.loc[paired.method.eq(m)]
                    .groupby("source_piece")[cols]
                    .sum()
                    .loc[clusters]
                    .to_numpy(float)
                    for m in ("attune", method)
                ]
                total_a, total_b = (a.sum(axis=0), b.sum(axis=0))
                observed = float(f1(total_a) - f1(total_b))
                exact = 2 ** len(clusters) <= n_resamples
                draws = 2 ** len(clusters) if exact else n_resamples
                swaps = (
                    itertools.product((0, 1), repeat=len(clusters)) if exact else None
                )
                rng = np.random.default_rng(seed)
                extreme = 0
                for start in range(0, draws, 256):
                    count = min(256, draws - start)
                    mask = (
                        np.array(list(itertools.islice(swaps, count)))
                        if exact
                        else rng.integers(0, 2, size=(count, len(clusters)))
                    )
                    change = mask @ (b - a)
                    differences = f1(total_a + change) - f1(total_b - change)
                    extreme += int(
                        np.count_nonzero(np.abs(differences) >= abs(observed) - 1e-12)
                    )
                records.append(
                    dict(
                        method=method,
                        metric=metric,
                        tracks=len(common),
                        source_groups=len(clusters),
                        excluded_tracks=selected[keys].drop_duplicates().shape[0]
                        - len(common),
                        attune_f1=100 * float(f1(total_a)),
                        competitor_f1=100 * float(f1(total_b)),
                        difference_pp=100 * observed,
                        p_value=(
                            extreme / draws if exact else (extreme + 1) / (draws + 1)
                        ),
                        exact=exact,
                        permutations=draws,
                        seed=seed,
                    )
                )
        result = pd.DataFrame(records)
        ordered = result.p_value.sort_values()
        adjusted = np.minimum(
            1,
            np.maximum.accumulate(ordered.to_numpy() * np.arange(len(ordered), 0, -1)),
        )
        result["p_adj"] = pd.Series(adjusted, index=ordered.index)
        result["significant"] = result.p_adj.le(alpha)
        result["alpha"] = alpha
        return (result, pd.DataFrame(report))


if __name__ == "__main__":
    raise SystemExit(NoteCLI.main())
