"""Reproducible headless build of the pinned official Tartini 1.2 source.

Only the explicit substitutions below modify upstream source. No rewritten DSP.
The archive retains upstream copyright headers and GPL-2.0-or-later license.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "20b96c1eec07fc7c2f8722b1e19c6f05c70927c40da1fc8779f6f595fff6790a"
SOURCES = (
    "mytransforms channel notedata analysisdata conversions useful bspline "
    "fast_smooth prony mymatrix myalgo zoomlookup IIR_Filter "
    "FastSmoothedAveragingFilter FixedAveragingFilter GrowingAveragingFilter"
).split()
HEADERS = (
    "mytransforms channel notedata analysisdata conversions useful bspline "
    "fast_smooth prony mymatrix myalgo mystring musicnotes equalloudness "
    "zoomlookup IIR_Filter Filter FastSmoothedAveragingFilter "
    "FixedAveragingFilter GrowingAveragingFilter array1d array2d myassert "
    "large_vector SmartPtr safe_bool"
).split()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Upstream patch no longer applies uniquely: {old!r}")
    return text.replace(old, new)


def prepare(destination: Path, *, safety_patches: bool = True):
    archive = ROOT / "tartini1.2.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Tartini reference archive checksum mismatch")
    wanted = {s + ".cpp" for s in SOURCES} | {s + ".h" for s in HEADERS}
    found = set()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in wanted or not member.isfile():
                continue
            if name in found:
                raise RuntimeError(f"Ambiguous upstream file: {name}")
            found.add(name)
            text = tar.extractfile(member).read().decode("latin1")
            if name == "channel.cpp":
                # UI-only routines: no callers in the pitch engine.
                for start, end in (
                    ('QString Channel::getUniqueFilename()', 'float Channel::averagePitch(int begin, int end)'),
                    ('AnalysisData *Channel::getActiveChannelCurrentChunkData()', 'void Channel::clearFreqLookup()'),
                    ('void Channel::exportChannel(int type, QString typeString)', 'void Channel::doPronyFit(int chunk)'),
                ):
                    first, last = text.index(start), text.index(end)
                    text = text[:first] + text[last:]
                text = re.sub(r'^#include <Q(?:FileDialog|File|TextStream)>\n', '', text, flags=re.M)
                text = replace_once(text, 'gdata->qsettings->value("Analysis/thresholdValue", 93).toInt()', '93')
                # Upstream destructor leaks these two filters on every track.
                text = replace_once(text, '  delete highPassFilter;',
                                    '  delete highPassFilter;\n  delete pitchSmallSmoothingFilter;\n  delete pitchBigSmoothingFilter;')
            if name == "mytransforms.cpp" and safety_patches:
                # Upstream indexes [-1] after silence/no candidate. Undefined
                # behavior has no reference value to reproduce; use zero drift.
                text = replace_once(text, '      if(chunk > 0) {\n        float prevPeriod',
                                    '      if(chunk > 0 && prevAnalysisData->highestCorrelationIndex >= 0) {\n        float prevPeriod')
            (destination / name).write_text(text)
    if wanted != found:
        raise RuntimeError(f"Missing upstream sources: {wanted - found}")
    for name in ("gdata.h", "soundfile.h", "qcolor.h", "bridge.cpp"):
        shutil.copyfile(ROOT / name, destination / name)


def build(*, safety_patches: bool = True) -> Path:
    # Include every shim and build instruction in binary identity. Rebuilds never
    # overwrite loaded libraries, and concurrent processes publish atomically.
    digest = hashlib.sha256()
    for path in sorted(ROOT.iterdir()):
        if path.suffix in {".h", ".cpp", ".py", ".gz"}:
            digest.update(path.read_bytes())
    digest.update(str(safety_patches).encode())
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = shlex.split(os.environ.get("TARTINI_FFTW_FLAGS", ""))
    if not flags:
        try:
            flags = shlex.split(subprocess.check_output(
                ["pkg-config", "--cflags", "--libs", "fftw3f"], text=True, stderr=subprocess.DEVNULL))
        except (FileNotFoundError, subprocess.CalledProcessError):
            flags = ["-lfftw3f"]
            for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
                if (prefix / "include/fftw3.h").exists():
                    flags = [f"-I{prefix}/include", f"-L{prefix}/lib", "-lfftw3f"]
                    break
    digest.update(repr((compiler, flags, sys.platform)).encode())
    # A private user-owned directory, never a shared predictable /tmp library.
    cache = ROOT / ".build" / digest.hexdigest()[:20]
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / ("tartini.dylib" if sys.platform == "darwin" else "tartini.so")
    if library.exists():
        return library
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        work = Path(temporary)
        prepare(work, safety_patches=safety_patches)
        command = compiler + ["-std=c++11", "-Wno-deprecated-declarations", "-O2", "-fPIC", "-shared", "-ffp-contract=off",
                              "-include", "algorithm", "-include", "cstring", "-I", str(work)]
        command += [str(work / (name + ".cpp")) for name in SOURCES]
        command += [str(work / "bridge.cpp"), *flags, "-o", str(work / library.name)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("Tartini requires a C++ compiler and FFTW3 single precision.\n" + result.stderr[-14000:])
        os.replace(work / library.name, library)
    return library


if __name__ == "__main__":
    print(build())
