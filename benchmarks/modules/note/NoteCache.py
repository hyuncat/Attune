from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from typing import Any
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData

PathLike = str | Path
NOTE_CACHE_VERSION = 1
LEGACY_NOTEBOOK_SHA = "502c5753c61a3401f6fa6a6b4f8e6641f24280737b31f8148ec47d9be922fe23"
LEGACY_FUNCTIONS = {
    "cpu_seconds": "ab1ec3cd359e65be3db1d396a097661750c7a2fb36c516d3b985f69e7c37e0f7",
    "reference_notes": "4bd6a82bc509dd92674457c6a609b7ad4737194726f952b87e7753c1dea37044",
    "event_arrays": "497cfbc3409d5c7524631d40b1d679598cc49a41f3eecf26512b6e4c0515f698",
    "score_predictions": "1301e56444a154df1f42f0aab9cbbc8ba2740730cee682523d594b319138032a",
    "_predict": "876e663738e1ba1ed60c6e2138a88a9f71d669667ac9690ff0d87375b7df88d5",
}
REPEAT_RECOVERY_PREDICT_SHA = (
    "bbbb5bccca9d864f4283afd7da571179ed90d0a8df3c8326e6ed5098a28251a4"
)
CREPE_CACHE_PREDICT_SHA = (
    "995dfeb82fafaa5a668231b7873fb57eb95fb727d74c406d5f9c688ba8f117bf"
)
NOTEBOOK_PATH = "benchmarks/modules/note/NoteNotebook.py"


class NoteCache:

    @staticmethod
    def function_fingerprints(path):
        """Fingerprint scoring and shared adapter helpers at their class-owned locations."""
        paths = [Path(path), Path(__file__).with_name("NoteDetectorBase.py")]
        result = {}
        for source in paths:
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in LEGACY_FUNCTIONS:
                    result[node.name] = hashlib.sha256(
                        ast.dump(node, include_attributes=False).encode()
                    ).hexdigest()
        return result

    @staticmethod
    def job_key(provenance, task):
        """Conservative semantic key; never use worker count, method list or git HEAD."""
        sources = provenance["sources"]
        functions = provenance.get("cache_functions")
        if functions is None:
            if sources.get(NOTEBOOK_PATH) != LEGACY_NOTEBOOK_SHA:
                return None
            functions = LEGACY_FUNCTIONS
        method = task["method"]
        if (
            method != "crepe-notes"
            and functions.get("_predict") == CREPE_CACHE_PREDICT_SHA
        ):
            functions = {**functions, "_predict": REPEAT_RECOVERY_PREDICT_SHA}
        if (
            method in ("basic-pitch", "crepe-notes", "tony")
            and functions.get("_predict") == REPEAT_RECOVERY_PREDICT_SHA
        ):
            functions = {**functions, "_predict": LEGACY_FUNCTIONS["_predict"]}
        dependencies = {
            "benchmarks/modules/note/NoteDetectionBaselines.py",
            "benchmarks/modules/note/NoteBenchmarker.py",
            "benchmarks/modules/pitch/competitors/Attune.py",
            "benchmarks/modules/pitch/PitchDetectorBase.py",
            "algorithms/Config.py",
            "app_logic/NoteData.py",
        }
        if method.startswith("attune"):
            dependencies.update(
                (
                    p
                    for p in sources
                    if p.startswith(("algorithms/", "app_logic/"))
                    and "/tests/" not in p
                    and ("/archive/" not in p)
                )
            )
            dependencies.add("benchmarks/modules/pitch/PitchCache.py")
        if method == "crepe-notes":
            dependencies.add("benchmarks/modules/pitch/competitors/Crepe.py")
        if method == "tony":
            dependencies.add("benchmarks/modules/note/Tony.py")
        if "benchmarks/modules/note/NoteDetectorBase.py" in sources:
            if not functions or set(LEGACY_FUNCTIONS) - functions.keys():
                return None
            dependencies.discard("benchmarks/modules/note/NoteDetectionBaselines.py")
            dependencies.discard("benchmarks/modules/note/Tony.py")
            competitor = {
                "attune": "Attune",
                "attune-audio-only": "Attune",
                "basic-pitch": "BasicPitch",
                "crepe-notes": "CrepeNotes",
                "tony": "Tony",
            }.get(method)
            if competitor is None:
                return None
            dependencies.update(
                {
                    "benchmarks/modules/note/NoteDetectorBase.py",
                    "benchmarks/modules/note/NoteCache.py",
                    f"benchmarks/modules/note/competitors/{competitor}.py",
                }
            )
        if any((p not in sources for p in dependencies)):
            return None
        settings = provenance["config"]
        config_keys = [
            "onset_tolerance",
            "pitch_tolerance",
            "offset_ratio",
            "offset_min_tolerance",
        ]
        if method == "attune-audio-only":
            config_keys += ["audio_only_fmin", "audio_only_fmax", "audio_only_min_note"]
        elif method == "attune":
            config_keys += ["audio_only_min_note"]
        input_keys = ["audio", "reference"]
        task_keys = ["dataset", "track_id", "reference_part", "reference_policy"]
        if method == "attune":
            input_keys += ["score", "pitch_annotation"]
            task_keys += ["score_part", "pitch_fmin", "pitch_fmax"]
        if any((task.get(k) not in provenance["inputs"] for k in input_keys)):
            return None
        packages = {"numpy", "scipy", "librosa", "mir_eval", "pretty_midi"}
        packages.update(
            {
                "attune": {"ruptures"},
                "attune-audio-only": {"ruptures"},
                "basic-pitch": {"basic-pitch", "tensorflow"},
                "crepe-notes": {"crepe", "crepe-notes", "tensorflow"},
            }.get(method, set())
        )
        identity = dict(
            schema="note-result-v1",
            method=method,
            functions=functions,
            sources={p: sources[p] for p in sorted(dependencies)},
            settings={k: settings[k] for k in config_keys},
            inputs={k: provenance["inputs"][task[k]] for k in input_keys},
            task={k: task.get(k) for k in task_keys},
            versions={k: provenance["versions"].get(k) for k in sorted(packages)},
            python=provenance["python"],
            platform=provenance["platform"],
            threads=provenance["threads"],
            device=provenance["device"],
        )
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def checkpoint_index(runs_root):
        """Index successful, complete job artifacts; never mutate historical runs."""
        result = {}
        manifests = sorted(
            Path(runs_root).glob("*/metadata.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for manifest in manifests:
            try:
                metadata = json.loads(manifest.read_text())
                for checkpoint in (manifest.parent / "checkpoints").glob("*.json"):
                    try:
                        row = json.loads(checkpoint.read_text())
                        if row.get("status") != "ok":
                            continue
                        prediction = (
                            manifest.parent / "predictions" / f"{row['job_id']}.json"
                        )
                        data = json.loads(prediction.read_text())
                        if not {"intervals", "frequencies_hz"} <= data.keys():
                            continue
                        key = NoteCache.job_key(metadata, row)
                        if key and row.get("cache_key", key) == key:
                            result.setdefault(key, (checkpoint, prediction, row))
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return result

    @staticmethod
    def atomic_json(path, value):
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)

    @staticmethod
    def reuse_checkpoint(candidate, task, output):
        """Copy verified results into the new report; original artifacts remain intact."""
        checkpoint, prediction, row = candidate
        cached = {**row, **task, "cache_reused_from": str(checkpoint)}
        NoteCache.atomic_json(
            Path(output) / "predictions" / f"{task['job_id']}.json",
            json.loads(prediction.read_text()),
        )
        NoteCache.atomic_json(
            Path(output) / "checkpoints" / f"{task['job_id']}.json", cached
        )
        return cached

    @classmethod
    def note_cache_path(cls, corpus_dir: PathLike, track_id: str) -> Path:
        safe_track_id = track_id.replace("/", "_")
        return Path(corpus_dir) / "note_data" / f"{safe_track_id}.note.json"

    @staticmethod
    def _note_to_payload(note: Note) -> dict[str, Any]:
        return {
            "id": int(note.id),
            "start_time": float(note.start_time),
            "end_time": float(note.end_time),
            "midi_num": [float(m) for m in note.midi_num],
            "velocity": None if note.velocity is None else int(note.velocity),
            "instrument": note.instrument,
        }

    @staticmethod
    def _compatible_note_end_time(payload: dict[str, Any], start_time: float) -> float:
        """Accept both current end-time and legacy duration-only caches."""
        if payload.get("end_time") is not None:
            return float(payload["end_time"])
        return start_time + max(0.0, float(payload.get("duration", 0.0)))

    @classmethod
    def _note_from_payload(cls, payload: dict[str, Any]) -> Note:
        start_time = float(payload["start_time"])
        return Note(
            i=int(payload["id"]),
            start_time=start_time,
            end_time=cls._compatible_note_end_time(payload, start_time),
            midi_num=[float(m) for m in payload["midi_num"]],
            velocity=payload.get("velocity"),
            instrument=payload.get("instrument"),
        )

    @classmethod
    def save_note_data(
        cls, note_data: NoteData, cache_path: PathLike, metadata: dict[str, Any] = None
    ) -> Path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        notes = note_data.read(i=0, j=len(note_data.times)) if note_data.times else []
        payload = {
            "version": NOTE_CACHE_VERSION,
            "metadata": metadata or {},
            "notes": [cls._note_to_payload(n) for n in notes],
        }
        with open(cache_path, "w") as fh:
            json.dump(payload, fh)
        return cache_path

    @classmethod
    def load_note_data(cls, cache_path: PathLike) -> tuple[NoteData, dict[str, Any]]:
        with open(cache_path) as fh:
            payload = json.load(fh)
        if payload.get("version") != NOTE_CACHE_VERSION:
            raise ValueError(
                f"Unsupported note cache version: {payload.get('version')}"
            )
        note_data = NoteData()
        for note_payload in payload.get("notes", []):
            note_data.write_note(cls._note_from_payload(note_payload))
        return (note_data, dict(payload.get("metadata") or {}))

    @staticmethod
    def attune_frontend(task, config, recording, cfg, adapter, conditioning):
        from benchmarks.modules.pitch.PitchCache import PitchCache

        timings = {}
        from algorithms.PitchSmoother import PitchSmoother

        cache = PitchCache(task["pitch_cache"])
        hit = None
        if conditioning and config.use_pitch_cache:
            try:
                stages_payload, metadata = cache._payload()
                smooth_meta = metadata.get(PitchCache.SMOOTHED, {})
                compatible = (
                    PitchCache.SMOOTHED in stages_payload
                    and smooth_meta.get("pitch_cache_version") == PitchCache.VERSION
                    and (
                        smooth_meta.get("pitch_smoother_method")
                        == PitchSmoother.METHOD_VERSION
                    )
                    and cache.has_current_timing(
                        PitchCache.SMOOTHED, adapter.COMPUTE_CLOCK
                    )
                )
                if compatible:
                    hit = cache.read(PitchCache.SMOOTHED, cfg)
            except Exception:
                hit = None
        if hit is not None:
            recording.pitch_data, pitch_timing = hit
            timings["pitch_cache_status"] = "hit"
        else:
            import librosa

            samples, _ = librosa.load(task["audio"], sr=cfg.sr, mono=True)
            recording.audio_data = adapter._audio_data_from_samples(samples, cfg)
            stages = adapter.detect_stages(recording)
            recording.pitch_data = stages.data[PitchCache.SMOOTHED]
            pitch_timing = stages.timing[PitchCache.SMOOTHED]
            timings["pitch_cache_status"] = (
                "miss" if conditioning and config.use_pitch_cache else "disabled"
            )
            if conditioning and config.use_pitch_cache:
                cache.write(stages)
        timings["frontend_cpu_seconds"] = float(pitch_timing["pitch_compute_time"])
        timings["frontend_compute_clock"] = pitch_timing["compute_clock"]
        timings["frontend_timing_reused"] = hit is not None
        timings["pitch_cache_version"] = PitchCache.VERSION
        if conditioning and config.use_pitch_cache and cache.path.exists():
            timings["pitch_cache_sha256"] = hashlib.sha256(
                cache.path.read_bytes()
            ).hexdigest()
        return timings

    @staticmethod
    def crepe_frontend(task, config):
        from benchmarks.modules.pitch.competitors.Crepe import Crepe

        timings = {}
        detector = Crepe()
        raw_path = detector.raw_cache_path(task["crepe_pitch_cache"])
        _, frequency, confidence, cpu, wall, hit = detector.frontend(
            task["audio"], raw_path, use_cache=config.use_pitch_cache
        )
        timings.update(
            frontend_cpu_seconds=cpu,
            frontend_wall_seconds=wall,
            frontend_compute_clock=detector.COMPUTE_CLOCK,
            frontend_timing_reused=hit,
            pitch_cache_status=(
                ("hit" if hit else "miss") if config.use_pitch_cache else "disabled"
            ),
            frontend_preprocessing="librosa-mono-16000",
        )
        if config.use_pitch_cache:
            timings["pitch_cache_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        return (frequency, confidence, timings)
