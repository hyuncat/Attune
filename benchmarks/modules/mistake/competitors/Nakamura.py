"""Nakamura: adapter, pinned setup, and isolated inference in one class."""

import sys
from pathlib import Path

_root = next((p for p in Path(__file__).resolve().parents if (p / "app.py").is_file()))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pretty_midi
from benchmarks.paths import REPO_ROOT


class Nakamura(MistakeDetectorBase):
    URL = "https://midialignment.github.io/AlignmentTool_v240109.zip"
    ARCHIVE_SHA256 = "cf75af54435c6ad83a7b578b691724866f6df5701e8529d86ae2a3db1e85bddd"
    DEFAULT_ROOT = REPO_ROOT / "benchmarks/datasets/pretrained/nakamura/AlignmentTool"
    SOURCES = dict(
        SprToFmt3x="SprToFmt3x_v170225.cpp",
        Fmt3xToHmm="Fmt3xToHmm_v170225.cpp",
        ScorePerfmMatcher="ScorePerfmMatcher_v170101_2.cpp",
        ErrorDetection="ErrorDetection_v190702.cpp",
        RealignmentMOHMM="RealignmentMOHMM_v170427.cpp",
        MatchToCorresp="MatchToCorresp_v170918.cpp",
    )

    @staticmethod
    def root():
        return Path(
            os.environ.get("ATTUNE_NAKAMURA_ROOT", Nakamura.DEFAULT_ROOT)
        ).resolve()

    @staticmethod
    def preflight():
        directory = Nakamura.root()
        missing = [
            name
            for name in Nakamura.SOURCES
            if not os.access(directory / "Programs" / name, os.X_OK)
        ]
        if missing:
            raise RuntimeError(
                "Nakamura is not compiled. Run: python -m benchmarks.modules.mistake.competitors.Nakamura setup (or set ATTUNE_NAKAMURA_ROOT). Missing: "
                + ", ".join(missing)
            )
        return dict(
            release="v240109",
            binaries={
                name: hashlib.sha256(
                    (directory / "Programs" / name).read_bytes()
                ).hexdigest()
                for name in Nakamura.SOURCES
            },
        )

    @staticmethod
    def write_spr(path, notes):
        lines = []
        for i, note in enumerate(notes):
            if len(note.midi_num) != 1:
                raise ValueError("Nakamura benchmark requires monophonic note events")
            pitch = int(np.rint(note.midi_num[0]))
            if not 0 <= pitch <= 127 or note.end_time <= note.start_time:
                raise ValueError("Invalid pitch or duration for Nakamura")
            lines.append(
                f"{i}\t{note.start_time:.9f}\t{note.end_time:.9f}\t{pretty_midi.note_number_to_name(pitch)}\t{note.velocity or 64}\t80\t0\n"
            )
        Path(path).write_text("".join(lines))

    @staticmethod
    def parse_corresp(text):
        """Return standard alignment records; IDs are validated by the shared adapter."""
        result = []
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            fields = line.split()
            if len(fields) != 10:
                raise ValueError(f"Malformed Nakamura correspondence: {line}")
            user, score = (fields[0], fields[5])
            if user == score == "*":
                raise ValueError("Nakamura returned an empty correspondence")
            item = dict(
                label=(
                    "deletion"
                    if user == "*"
                    else "insertion" if score == "*" else "match"
                )
            )
            if user != "*":
                item["performance_id"] = user
            if score != "*":
                item["score_id"] = score
            result.append(item)
        return result

    @staticmethod
    def align(score_notes, user_notes):
        programs = Nakamura.root() / "Programs"
        with tempfile.TemporaryDirectory(prefix="attune-nakamura-") as tmp:
            directory = Path(tmp)
            Nakamura.write_spr(directory / "score_spr.txt", score_notes)
            Nakamura.write_spr(directory / "user_spr.txt", user_notes)
            commands = [
                ("SprToFmt3x", "score_spr.txt", "score_fmt3x.txt"),
                ("Fmt3xToHmm", "score_fmt3x.txt", "score_hmm.txt"),
                (
                    "ScorePerfmMatcher",
                    "score_hmm.txt",
                    "user_spr.txt",
                    "pre.txt",
                    "0.001",
                ),
                (
                    "ErrorDetection",
                    "score_fmt3x.txt",
                    "score_hmm.txt",
                    "pre.txt",
                    "err.txt",
                    "0",
                ),
                (
                    "RealignmentMOHMM",
                    "score_fmt3x.txt",
                    "score_hmm.txt",
                    "err.txt",
                    "match.txt",
                    "0.3",
                ),
                ("MatchToCorresp", "match.txt", "score_spr.txt", "corresp.txt"),
            ]
            for name, *args in commands:
                result = subprocess.run(
                    [str(programs / name), *args],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode:
                    raise RuntimeError(
                        f"Nakamura {name} failed: {result.stdout}\n{result.stderr}"
                    )
            return Nakamura.parse_corresp((directory / "corresp.txt").read_text())

    @staticmethod
    def setup():
        import argparse
        import hashlib
        from pathlib import Path
        import shutil
        import subprocess
        import tempfile
        import urllib.request
        import zipfile

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--archive", type=Path, help="Use an already downloaded official zip"
        )
        parser.add_argument("--cxx", default="c++")
        args = parser.parse_args()
        directory = Nakamura.root()
        with tempfile.TemporaryDirectory() as tmp:
            archive = args.archive or Path(tmp) / "source.zip"
            if args.archive is None:
                urllib.request.urlretrieve(Nakamura.URL, archive)
            if (
                hashlib.sha256(archive.read_bytes()).hexdigest()
                != Nakamura.ARCHIVE_SHA256
            ):
                raise RuntimeError("Nakamura source checksum mismatch")
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    path = Path(entry.filename)
                    if path.parts[0] != "AlignmentTool" or entry.is_dir():
                        continue
                    if ".." in path.parts or path.is_absolute():
                        raise ValueError("Unsafe source archive path")
                    destination = directory.joinpath(*path.parts[1:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(bundle.read(entry))
        compiler = shutil.which(args.cxx)
        if compiler is None:
            raise RuntimeError(
                "A C++ compiler is required (macOS: Xcode Command Line Tools)"
            )
        (directory / "Programs").mkdir(exist_ok=True)
        for name, source in Nakamura.SOURCES.items():
            subprocess.run(
                [
                    compiler,
                    "-O2",
                    "-std=c++11",
                    str(directory / "Code" / source),
                    "-o",
                    str(directory / "Programs" / name),
                ],
                check=True,
            )
        Nakamura.preflight()
        print(f"Nakamura ready: {directory}")

    def predict(self, performance, score, directory=None):
        return self.align(score, performance)

    @staticmethod
    def cli():
        import sys

        action = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
        if action == "setup":
            Nakamura.setup()
        elif action == "worker" and hasattr(Nakamura, "worker_main"):
            Nakamura.worker_main()
        else:
            raise SystemExit(
                "Usage: python -m benchmarks.modules.mistake.competitors.Nakamura setup [options]"
                + (" | worker [options]" if hasattr(Nakamura, "worker_main") else "")
            )


if __name__ == "__main__":
    Nakamura.cli()
