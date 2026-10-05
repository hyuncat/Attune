"""MistakeCache implementation and owned benchmark helpers."""

from __future__ import annotations

NATIVE_SERIAL_RUNNER_SHA256 = (
    "e7d487469f2ef1a01ade0d172b7397c1300568923395c3a22d106aea87077c01"
)
NATIVE_SERIAL_INFERENCE_SHA256 = (
    "a414aeee78c422941322042439fdf17c64bec40df8ce23ea629ae848367dbe1e"
)
NATIVE_CACHE_ONLY_CODE_UPGRADES = {
    "NativeComparison.inference": [
        "35bded9231bacdec212628fd6846be256978a8d9ca4e452ef83d528d68a9a0ba",
        "70a2ac121f123f072d9d0e3c3829bd22a0d9eb6b7d706f8b3b59bce8a4bd889f",
    ],
    "benchmarks/modules/mistake/CaseAssetReuse.py": [
        None,
        "8e4a3c0fe5c79b3dfdc447b3ff1a47e1de967e5b2231ee1fa81c0f6d61b1e162",
    ],
    "benchmarks/modules/mistake/CompetitorComparison.py": [
        "06fd265d4171b7513143e7177373f37bb18fa0e38d0533581703ad593196032a",
        "1ce92c8c322ad88140a648951c49bb2665118bb25fa89e00bc1583b0dd218b0f",
    ],
    "benchmarks/modules/mistake/datasets/MistakeCases.py": [
        "5f168353d2aa06b09335108abc177d757da8a9f6772b0060bb76c51aee601465",
        "6dea0b007e35ce46740ca1d212b6cb159685b478e7cf1d3bc1c54833a6fd5439",
    ],
    "benchmarks/modules/mistake/Parallel.py": [
        "4261dc01202b0b65668d20dfebf5cca8ca52dad39ead12048235e1b8135d88db",
        "d021378ad7557938b24b8779f36d2d1046b8290a870fc520e585784b30501eea",
    ],
    "benchmarks/modules/mistake/competitors/PolyTuneReuse.py": [
        "14560c47efcdd7b9cd54e4933f41694d785934a51f0cce9a74c01b7917df123c",
        "ab1fb8ce8b358ce0210436eed692eaaa716f38d1240fb66c219c1a9f095a4b07",
    ],
}
from importlib.metadata import version
from copy import deepcopy
from benchmarks.modules.mistake.datasets.NativeDatasets import digest
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import pandas as pd

EVALUATION_VERSION = 2
RUNNER = "benchmarks/modules/mistake/MistakeBenchmarker.py"
WORKER = "benchmarks/modules/mistake/competitors/PolyTune.py"
FRAMES = "benchmarks/modules/mistake/competitors/PolyTune.py"
LEGACY_COMPATIBLE = {
    RUNNER: "1bfbbea2c94ffb41091c40f051ee544dcca407fe59b3f6338e557d1c05bcbb7a",
    WORKER: "e8d159f4cf47fe96aea67133c4019ddd0ac68b2b73662eb31101b782fb0c149c",
}
EXECUTION_ONLY_UPGRADES = {
    "benchmarks/modules/mistake/competitors/PolyTune.py": [
        (
            "04d06d4d785b8db1229b7eac8db5dfd7178a0337bb24cc535811ce2dfad7939e",
            "f2bacb87014020c3ea0b2021f4a3c976c45c25574f293782ae2691cd17e3e3a2",
        )
    ],
    "benchmarks/modules/mistake/competitors/PolyTune.py": [
        (
            "48bd057d012bfd82eea00fee850ea1f7f7c6d0cc442a74094abf33e6d7b65b15",
            "bb6782dd4fd26ebbd2786ebe19ef7aaf9ba8298eade7d6b8eb015bc8ccd29123",
        )
    ],
}
SCORER_ID_UPGRADE = (
    "69c73bcbc3a2c3de9c77226854a39f03d50d538c7e5c6a75edcb8f81543f30ab",
    "1b9ea203560b99eced554c50c4e3c24a840dae10e195ec716b5d1d8a478d1cf0",
)
SCORER_PATH = "benchmarks/modules/mistake/MistakeDetectorBase.py"
from functools import lru_cache
import shutil

GENERATION_EQUIVALENT = {
    "MistakeCases.py": [
        "5f168353d2aa06b09335108abc177d757da8a9f6772b0060bb76c51aee601465",
        "6dea0b007e35ce46740ca1d212b6cb159685b478e7cf1d3bc1c54833a6fd5439",
    ],
    "MistakeBenchmarker.py": [
        "c095afff70393db5a5ddbe1d9d150cd48bd13321ac52098293cc9bc4863e82f5",
        "fe997f00a41468373598356d04110fd0e929c105410928733b84cfc11e9e710f",
    ],
}
from benchmarks.modules.mistake.competitors.PolyTune import PolyTune

LADDER_DEFAULT_WORKER_UPGRADE = (
    "7b18b5781efb704dda56ef9cb2969bfef6452dd3bacc24e042a273847b557655",
    "f7e813dd4edb6735dc82c280923757783d0276ecffe86a14aee9d0669b049b0d",
)
WORKER_IMPORT_UPGRADES = {
    "PolyTune": (
        "bb6782dd4fd26ebbd2786ebe19ef7aaf9ba8298eade7d6b8eb015bc8ccd29123",
        "c81758b0f32098e6896ef26de0a766ca5bc6ca90092915c8ede5b8fda724fb3a",
    ),
    "LadderSym": (
        "f7e813dd4edb6735dc82c280923757783d0276ecffe86a14aee9d0669b049b0d",
        "e137d3f3a2bc649a01c066aac1ce56563e25d0259ddb75ec77f4215563360395",
    ),
}


class MistakeCache:
    """Storage adapters and provenance checks for all benchmark methods."""

    @staticmethod
    def cached_native_pitches(recording, performance):
        """Stage cache independent of note segmentation, alignment and error thresholds."""
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        import hashlib
        from benchmarks.paths import REPO_ROOT
        from benchmarks.modules.pitch.PitchCache import PitchCache

        PitchCache = PitchCache
        ignored = {
            "pitch_tolerance",
            "timing_tolerance",
            "min_note_length",
            "min_note_length_factor",
            "min_note_length_cap",
        }
        config = {
            k: v
            for k, v in asdict(recording.config).items()
            if k not in ignored and (not k.startswith("alignment_"))
        }
        code = {
            name: digest(REPO_ROOT / name)
            for name in (
                "algorithms/PitchDetector.py",
                "algorithms/PitchSmoother.py",
                "app_logic/user/ds/Recording.py",
                "benchmarks/modules/pitch/PitchCache.py",
            )
        }
        import inspect

        signature = dict(
            version=1,
            audio_sha256=digest(performance),
            config=config,
            code=code,
            resampler=hashlib.sha256(
                inspect.getsource(MistakeBenchmarker.native_attune_audio).encode()
            ).hexdigest(),
            packages={p: version(p) for p in ("numpy", "scipy", "librosa", "numba")},
        )
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = (
            REPO_ROOT
            / "benchmarks/results/_native_pitch_cache"
            / (key + ".pitch.pkl.xz")
        )
        cache = PitchCache(path)
        hit = cache.read(PitchCache.SMOOTHED, recording.config)
        if hit is not None:
            recording.pitch_data, _ = hit
            recording.pitch_data.end_index = len(recording.pitch_data.data)
            recording.pitches_smoothed = True
            return dict(hit=True, path=str(path), signature=signature)
        recording.detect_pitches()
        stages = PitchCache.Stages()
        stages.data[PitchCache.SMOOTHED] = recording.pitch_data
        stages.timing[PitchCache.SMOOTHED] = {"source": "Recording.detect_pitches"}
        cache.write(stages)
        return dict(hit=False, path=str(path), signature=signature)

    @staticmethod
    def cached_native_notes(recording, pitch_cache, score_midi):
        """Cache unrefined detections so label-threshold changes rerun alignment only."""
        import hashlib
        from benchmarks.paths import REPO_ROOT
        from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker

        config = {
            k: v
            for k, v in asdict(recording.config).items()
            if k not in ("pitch_tolerance", "timing_tolerance")
            and (not k.startswith("alignment_"))
        }
        signature = dict(
            version=1,
            pitch_cache=pitch_cache["path"],
            score_sha256=digest(score_midi),
            config=config,
            code={
                name: digest(REPO_ROOT / name)
                for name in (
                    "algorithms/NoteDetector.py",
                    "app_logic/NoteData.py",
                    "app_logic/user/ds/Recording.py",
                    "benchmarks/modules/note/NoteBenchmarker.py",
                )
            },
        )
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = REPO_ROOT / "benchmarks/results/_native_note_cache" / (key + ".json")
        reader = NoteBenchmarker.__new__(NoteBenchmarker)
        try:
            notes, metadata = reader.load_note_data(path)
            hit = (
                metadata.get("signature") == signature
                and metadata.get("trimmed_boundaries") is False
            )
        except (OSError, ValueError, KeyError):
            hit = False
        if hit:
            recording.transition_detector.clear_transitions(recording.pitch_data.data)
            recording.resize_score(to_span="pitch", include_transitions=False)
            recording.update_min_note_length()
            recording.note_data = notes
            recording.recompute_vibrato(note_aware=True)
        else:
            recording.detect_notes()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            reader.save_note_data(
                recording.note_data,
                temporary,
                metadata=dict(signature=signature, trimmed_boundaries=False),
            )
            temporary.replace(path)
        return dict(hit=hit, path=str(path))

    @staticmethod
    def compatible_contract(previous, current):
        if previous == current:
            return True
        old_body, new_body = (dict(previous), dict(current))
        old_code, new_code = (old_body.pop("code", {}), new_body.pop("code", {}))
        changes = {
            k: [old_code.get(k), new_code.get(k)]
            for k in old_code.keys() | new_code.keys()
            if old_code.get(k) != new_code.get(k)
        }
        if (
            old_body == new_body
            and changes
            and all(
                (
                    NATIVE_CACHE_ONLY_CODE_UPGRADES.get(k) == pair
                    for k, pair in changes.items()
                )
            )
        ):
            return True
        previous = deepcopy(previous)
        code = previous.get("code", {})
        path = "benchmarks/modules/mistake/MistakeBenchmarker.py"
        if (
            code.get(path) != NATIVE_SERIAL_RUNNER_SHA256
            or current["code"].get("NativeComparison.inference")
            != NATIVE_SERIAL_INFERENCE_SHA256
        ):
            return False
        previous["code"] = {
            p: h
            for p, h in code.items()
            if p != path
            and "tests" not in Path(p).parts
            and (not Path(p).name.startswith("test_"))
        }
        previous["code"]["NativeComparison.inference"] = NATIVE_SERIAL_INFERENCE_SHA256
        return previous == current

    @staticmethod
    def reusable_audio_prediction(source, contract, case, method):
        """Reuse neural inference only, validating inputs, weights, environments and adapter code.

        Attune-only frontend changes do not warrant loading either neural model again.
        Metrics are always recomputed, so no old score rows leak into the new run.
        """
        if method == "Attune" or source is None:
            return None
        source = Path(source).resolve()
        metadata_path = source / "run.json"
        if not metadata_path.exists():
            return None
        previous = json.loads(metadata_path.read_text())["contract"]
        for key in ("device", "packages", "ladder_contiguous_inference"):
            if previous.get(key) != contract.get(key):
                raise ValueError(f"Cannot reuse audio predictions: changed {key}")
        if previous["models"].get(method) != contract["models"].get(method):
            raise ValueError(f"Cannot reuse {method}: model identity changed")
        unrelated = {
            f"benchmarks/modules/mistake/{name}.py"
            for name in (
                "MistakeBenchmarker",
                "CompetitorComparison",
                "provenance/InjectedCaseAudit",
                "sweeps/RefinementComparison",
                "provenance/PipelineAudit",
                "provenance/PipelineAuditDetails",
                "NotebookRun",
                "datasets/MistakeCases",
                "CaseAssetReuse",
                "Parallel",
                "competitors/PolyTuneReuse",
                "ResultOverview",
            )
        }

        def adapter_code(code):
            return {
                k: v
                for k, v in code.items()
                if k
                not in (
                    "NativeComparison.inference",
                    "benchmarks/modules/mistake/MistakeBenchmarker.py",
                )
                and k not in unrelated
                and ("tests" not in Path(k).parts)
                and (not Path(k).name.startswith("test_"))
            }

        if adapter_code(previous["code"]) != adapter_code(contract["code"]):
            raise ValueError(
                "Cannot reuse audio predictions: adapter/dependency code changed"
            )
        prior_case = next(
            (
                c
                for c in previous["manifest"]["cases"]
                if c["case_id"] == case["case_id"]
            ),
            None,
        )
        if prior_case is None:
            return None
        if (
            previous["manifest"]["dataset"] != contract["manifest"]["dataset"]
            or prior_case["hashes"] != case["hashes"]
        ):
            raise ValueError("Cannot reuse audio predictions: case assets changed")
        checkpoint = source / "cases" / case["case_id"] / method.lower() / "result.json"
        if not checkpoint.exists():
            return None
        prediction = deepcopy(json.loads(checkpoint.read_text())["prediction"])
        prediction["reused_from"] = dict(
            path=str(checkpoint), sha256=digest(checkpoint)
        )
        return prediction

    @staticmethod
    def atomic_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)

    @staticmethod
    def units(methods):
        return [
            (kind, method)
            for method in methods
            for kind in (
                ("audio",)
                if method in ("PolyTune", "LadderSym")
                else (
                    ("detected",)
                    if method in ("Attune (repeat only)", "Attune (Checker 3)")
                    else ("detected", "oracle_notes")
                )
            )
        ]

    @staticmethod
    def scoped_job(job):
        """Adding a competitor must not invalidate unrelated completed methods."""
        result = dict(job)
        method = job["method"]
        packages = {"numpy", "scipy", "mir_eval"}
        if method.startswith("Parangonar"):
            packages |= {"parangonar", "partitura"}
        if method == "Parangonar TheGlueNote":
            packages.add("TheGlueNote")
        if method in ("Nakamura", "PolyTune", "LadderSym"):
            packages.add(method)
        result["packages"] = {k: v for k, v in job["packages"].items() if k in packages}
        result["code"] = {
            p: h
            for p, h in job["code"].items()
            if (method == "PolyTune" or "/PolyTune" not in p)
            and (method == "LadderSym" or "/LadderSym" not in p)
            and (
                method == "Attune (repeat only)" or not p.endswith("/RepeatSplitter.py")
            )
            and (method == "Nakamura" or not p.endswith("/Nakamura.py"))
        }
        if method in ("PolyTune", "LadderSym"):
            unrelated = {
                "notebooks/archive/MistakeChecker2.py",
                "notebooks/archive/MistakeChecker3.py",
                "algorithms/NoteDetector.py",
                "algorithms/MistakeDetector.py",
                "app_logic/user/ds/Recording.py",
            }
            result["code"] = {
                p: h for p, h in result["code"].items() if p not in unrelated
            }
        return result

    @staticmethod
    def generation_signature(spec):
        result = dict(spec)
        result["code"] = {
            k: (
                "audited_boundary_cache_only_v1"
                if v in GENERATION_EQUIVALENT.get(k, ())
                else v
            )
            for k, v in spec.get("code", {}).items()
        }
        return result

    @staticmethod
    @lru_cache(maxsize=8)
    def asset_candidates(results_root):
        found = []
        for manifest in sorted(
            Path(results_root).glob("*/cases/*/manifest.json"), reverse=True
        ):
            try:
                item = json.loads(manifest.read_text())
                run_path = manifest.parents[2] / "run.json"
                run = json.loads(run_path.read_text()) if run_path.exists() else {}
                found.append((manifest, item, run))
            except (OSError, ValueError):
                continue
        return found

    @staticmethod
    def reuse_case_assets(bench, spec, output):
        from benchmarks.modules.mistake.competitors.PolyTune import PolyTune
        from benchmarks.paths import REPO_ROOT

        wanted = MistakeCache.generation_signature(spec)
        for manifest, old, run in MistakeCache.asset_candidates(
            str(Path(output).resolve().parent)
        ):
            if MistakeCache.generation_signature(old.get("spec", {})) != wanted:
                continue
            if not all((Path(old[k]).is_file() for k in ("audio", "midi", "truth"))):
                continue
            paths = bench.mistake_db_paths(old["dataset"], old["track_id"])
            copied = {}
            for key in ("audio", "midi", "truth"):
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old[key], paths[key])
                copied[key] = dict(source=old[key], sha256=PolyTune.sha256(paths[key]))
            pitch_code = {
                k: v
                for k, v in run.get("code", {}).items()
                if any(
                    (
                        name in k
                        for name in ("PitchDetector", "PitchSmoother", "PitchCache")
                    )
                )
            }
            pitch_ok = all(
                (
                    (REPO_ROOT / k).is_file() and PolyTune.sha256(REPO_ROOT / k) == v
                    for k, v in pitch_code.items()
                )
            )
            defaults = manifest.parents[2] / "production_defaults.json"
            if defaults.exists():
                from dataclasses import asdict
                from algorithms.Config import Config

                ignored = {
                    "pitch_tolerance",
                    "timing_tolerance",
                    "min_note_length",
                    "min_note_length_factor",
                    "min_note_length_cap",
                }
                before = json.loads(defaults.read_text())
                current = asdict(Config())
                pitch_ok &= all(
                    (before.get(k) == v for k, v in current.items() if k not in ignored)
                )
            else:
                pitch_ok = False
            if pitch_ok:
                for suffix in ("", ".mistake-range.json"):
                    source = Path(str(old["pitch_data"]) + suffix)
                    dest = Path(str(paths["pitch_data"]) + suffix)
                    if source.is_file():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, dest)
                        copied["pitch_data" + suffix] = dict(
                            source=str(source), sha256=PolyTune.sha256(dest)
                        )
            for relative in ("clean_score.mid", "score_audio/clean_score.wav"):
                source = manifest.parent / relative
                if source.is_file():
                    dest = bench.MISTAKE_DIR / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
            result = dict(old)
            result.update(
                {
                    k: str(paths[k])
                    for k in ("audio", "midi", "truth", "pitch_data", "note_data")
                }
            )
            result["asset_reuse"] = copied
            return result
        return None

    @staticmethod
    def compatible_worker(previous, current, method, contiguous=False):
        if previous == current:
            return True
        original, relocated = WORKER_IMPORT_UPGRADES.get(method, (None, None))
        if current == relocated:
            current = original
        return previous == current or (
            method == "LadderSym"
            and (not contiguous)
            and ((previous, current) == LADDER_DEFAULT_WORKER_UPGRADE)
        )

    @staticmethod
    @lru_cache(maxsize=8)
    def candidates(
        results_root, identity_json, worker_hash, frames_hash, method="PolyTune"
    ):
        identity = json.loads(identity_json)
        result = []
        for path in Path(results_root).glob("*/run.json"):
            try:
                run = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if run.get("packages", {}).get(method) != identity:
                continue
            if method == "LadderSym" and run.get("ladder_contiguous_inference", False):
                continue
            code = run.get("code", {})
            if (
                not MistakeCache.compatible_worker(
                    code.get(
                        "benchmarks/modules/mistake/competitors/" + method + ".py",
                        code.get("benchmarks/modules/mistake/" + method + "Worker.py"),
                    ),
                    worker_hash,
                    method,
                )
                or code.get(
                    "benchmarks/modules/mistake/competitors/PolyTune.py",
                    code.get("benchmarks/modules/mistake/PolyTuneFrames.py"),
                )
                != frames_hash
            ):
                continue
            for manifest in (path.parent / "cases").glob("*/manifest.json"):
                try:
                    item = json.loads(manifest.read_text())
                except (OSError, ValueError):
                    continue
                prediction = (
                    manifest.parent / method.lower() / (method.lower() + ".json")
                )
                score = manifest.parent / "score_audio/clean_score.wav"
                audio = Path(item["audio"])
                if prediction.is_file() and audio.is_file() and score.is_file():
                    result.append(
                        (audio, score, prediction, manifest.parent / "clean_score.mid")
                    )
        return result

    @staticmethod
    def cached_prediction(task, identity, *, method="PolyTune"):
        if identity is None:
            return None
        directory = Path(task["directory"]).resolve()
        results_root = directory.parents[3]
        code_dir = Path(__file__).parent / "competitors"
        options = MistakeCache.candidates(
            str(results_root),
            json.dumps(identity, sort_keys=True),
            PolyTune.sha256(code_dir / (method + ".py")),
            PolyTune.sha256(code_dir / "PolyTune.py"),
            method,
        )
        performance, score = (
            PolyTune.sha256(task["audio"]),
            PolyTune.sha256(task["score_audio"]),
        )
        for old_audio, old_score, prediction, old_midi in options:
            if (
                PolyTune.sha256(old_audio) != performance
                or PolyTune.sha256(old_score) != score
            ):
                continue
            if method == "LadderSym" and (
                not old_midi.is_file()
                or PolyTune.sha256(old_midi) != PolyTune.sha256(task["score_midi"])
            ):
                continue
            try:
                payload = json.loads(prediction.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(payload.get("events"), list):
                continue
            payload.update(
                inference_cache_hit=True,
                inference_cache_source=str(prediction),
                execution_cpu_seconds=0.0,
                setup_cpu_seconds=0.0,
                setup_seconds=0.0,
                wall_seconds=0.0,
            )
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / (method.lower() + ".json")
            if destination != prediction.resolve():
                import benchmarks.modules.mistake.MistakeCache as _local_MistakeCache

                atomic_json = _local_MistakeCache.MistakeCache.atomic_json
                atomic_json(destination, payload)
            return payload
        return None

    def __init__(self, output, metadata, previous=None):
        self.output = Path(output)
        self.metadata = metadata
        self.previous = previous or {}
        self.legacy = (
            pd.read_csv(self.output / "rows.csv")
            if previous and (self.output / "rows.csv").exists()
            else pd.DataFrame()
        )
        self.checkpoint_index = {}
        for path in (self.output / "checkpoints").glob("*.json"):
            payload = json.loads(path.read_text())
            old = payload["job"]
            key = tuple((old[k] for k in ("source", "seed", "rate", "input", "method")))
            self.checkpoint_index.setdefault(key, []).append(payload)
        self.source_hashes = {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in metadata["sources"]
        }

    def job(self, source, seed, rate, kind, method):
        code = dict(self.metadata["code"])
        return MistakeCache.scoped_job(
            dict(
                version=EVALUATION_VERSION,
                source=source,
                source_hash=self.source_hashes[source],
                seed=int(seed),
                rate=float(rate),
                input=kind,
                method=method,
                tolerances=self.metadata["tolerances"],
                code=code,
                packages=self.metadata["packages"],
                source_info=self.metadata["source_metadata"].get(source, {}),
            )
        )

    def path(self, job):
        key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        return self.output / "checkpoints" / f"{key}.json"

    def valid(self, job, rows):
        metrics = {"audio_pitch", "audio_missed", "audio_extra"}
        if job["method"] not in ("PolyTune", "LadderSym"):
            metrics |= {
                "substitution",
                "deletion",
                "insertion",
                "short",
                "long",
                "pitch",
                "duration",
                "legacy_five_type",
            }
        expected = {(float(t), m) for t in job["tolerances"] for m in metrics}
        keys = [(float(r["tolerance"]), r["metric"]) for r in rows]
        return (
            len(keys) == len(expected)
            and set(keys) == expected
            and all(
                (
                    all(
                        (
                            r[k] == job[k]
                            for k in ("source", "seed", "rate", "input", "method")
                        )
                    )
                    for r in rows
                )
            )
            and (len({r["case_id"] for r in rows}) == 1)
        )

    def save(self, job, rows):
        if not self.valid(job, rows):
            raise ValueError("Refusing to checkpoint incomplete evaluation rows")
        MistakeCache.atomic_json(self.path(job), dict(job=job, rows=rows))

    def load(self, job):
        path = self.path(job)
        if path.exists():
            payload = json.loads(path.read_text())
            if payload["job"] == job and self.valid(job, payload["rows"]):
                return payload["rows"]
            raise ValueError(f"Invalid checkpoint: {path}")
        key = tuple((job[k] for k in ("source", "seed", "rate", "input", "method")))
        for payload in self.checkpoint_index.get(key, []):
            previous_job = MistakeCache.scoped_job(
                json.loads(json.dumps(payload["job"]))
            )
            for name, pairs in EXECUTION_ONLY_UPGRADES.items():
                old_hash = previous_job["code"].get(name)
                new_hash = job["code"].get(name)
                if (old_hash, new_hash) in pairs:
                    previous_job["code"][name] = new_hash
            if (
                job["method"]
                in ("PolyTune", "Parangonar DualDTW", "Parangonar Automatic")
                and (
                    previous_job["code"].get(SCORER_PATH),
                    job["code"].get(SCORER_PATH),
                )
                == SCORER_ID_UPGRADE
            ):
                previous_job["code"][SCORER_PATH] = job["code"][SCORER_PATH]
            if previous_job == job and self.valid(job, payload["rows"]):
                rows = payload["rows"]
                for row in rows:
                    if "cpu_seconds" not in row:
                        row.update(
                            cpu_seconds=None,
                            execution_cpu_seconds=None,
                            timing_version="legacy_wall_only",
                        )
                self.save(job, rows)
                return rows
        if (
            self.previous.get("checkpoint_schema")
            or self.legacy.empty
            or self.previous.get("packages") != self.metadata["packages"]
        ):
            return None
        old_code = self.previous.get("code", {})
        if old_code.get(RUNNER) not in (
            self.metadata["code"][RUNNER],
            LEGACY_COMPATIBLE[RUNNER],
        ):
            return None
        for p, h in job["code"].items():
            if p == FRAMES and old_code.get(WORKER) == LEGACY_COMPATIBLE[WORKER]:
                continue
            if old_code.get(p) != h and old_code.get(p) != LEGACY_COMPATIBLE.get(p, h):
                return None
        frame = self.legacy
        for k in ("source", "seed", "rate", "input", "method"):
            frame = frame[frame[k] == job[k]]
        frame = frame[frame.tolerance.isin(job["tolerances"])]
        rows = frame.to_dict(orient="records")
        if not self.valid(job, rows):
            return None
        manifest = self.output / "cases" / rows[0]["case_id"] / "manifest.json"
        if (
            not manifest.exists()
            or json.loads(manifest.read_text())["spec"]["source_hash"]
            != job["source_hash"]
        ):
            return None
        self.save(job, rows)
        return rows

    @staticmethod
    def completed_results(request):
        """Replay a matching completed run from its own provenance, without inference.

        Explicit cache reuse preserves the algorithms that produced these results,
        even when local implementation files have since changed.
        """
        from benchmarks.modules.mistake.MistakeCache import MistakeCache

        Checkpoints = MistakeCache
        units = MistakeCache.units
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        summarize = MistakeBenchmarker.summarize
        import pandas as pd

        output = Path(request["output"])
        path = output / "run.json"
        if request.get("force_pitch_detection", False) or not path.exists():
            return None
        metadata = json.loads(path.read_text())
        if metadata.get("status") != "complete":
            return None
        if request[
            "stage"
        ] == "audio" and "no truth boundary correction" not in metadata.get(
            "pipeline", ""
        ):
            raise ValueError(
                "Historical run used truth-assisted boundaries. Use a new output directory; old results are preserved."
            )
        wanted = dict(
            sources=[str(Path(p).resolve()) for p in request["sources"]],
            seeds=list(request["seeds"]),
            rates=list(request["rates"]),
            methods=list(request["methods"]),
            tolerances=list(request["tolerances"]),
            source_metadata={
                str(Path(p).resolve()): info
                for p, info in request["source_metadata"].items()
            },
        )
        if any((metadata.get(key) != value for key, value in wanted.items())):
            raise ValueError(
                "Saved run settings differ from this request. Use a new output directory for a new experiment."
            )
        kinds = (
            ("oracle_notes",)
            if request["stage"] == "symbolic"
            else ("detected", "audio")
        )
        checkpoints = Checkpoints(output, metadata, previous=metadata)
        rows = []
        for source in wanted["sources"]:
            for seed in wanted["seeds"]:
                for rate in wanted["rates"]:
                    for kind, method in units(wanted["methods"]):
                        if kind not in kinds:
                            continue
                        job = checkpoints.job(source, seed, rate, kind, method)
                        batch = checkpoints.load(job)
                        if batch is None:
                            raise ValueError(
                                f"Missing or incompatible saved checkpoint: {source}, {seed}, {rate}, {method}. No redetection performed."
                            )
                        rows.extend(batch)
        frame = pd.DataFrame(rows)
        for name, table in [("rows.csv", frame), ("summary.csv", summarize(frame))]:
            temporary = output / (name + ".tmp")
            table.to_csv(temporary, index=False)
            temporary.replace(output / name)
        print(
            f"Reused {len(frame)} saved rows from {output}; no detection or model inference. Results retain their recorded algorithm versions.",
            flush=True,
        )
        return frame

    @staticmethod
    def pipeline_fingerprint():
        from benchmarks.paths import REPO_ROOT

        paths = [
            "notebooks/archive/MistakeChecker2.py",
            "notebooks/archive/MistakeChecker3.py",
            "algorithms/NoteDetector.py",
            "algorithms/MistakeDetector.py",
            "algorithms/Config.py",
            "app_logic/user/ds/Recording.py",
            "benchmarks/modules/mistake/datasets/MistakeInjector.py",
            "benchmarks/modules/mistake/sweeps/RefinementComparison.py",
        ]
        return {
            p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in paths
        }
