#!/usr/bin/env python3
"""Rebuild the resources/demo/* recording sidecars from audio + score.

Each demo take is re-analyzed headlessly (detect_pitches -> detect_notes ->
detect_mistakes, mirroring perform.py::analyze) and its `.{name}.json.xz` is
overwritten, so the caches match the current detectors and Pitch serialization.
Recordings are rebuilt in parallel across processes (pitch detection is
CPU-bound).

Benchmark pitch caches are NOT touched here — rebuild those with the harness's
own rewrite path, e.g.:
    python experiments/scripts/benchmark_pitch.py --no-cache --models pyin
"""
from __future__ import annotations

import argparse
import contextlib
import json
import lzma
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # re-runs in each spawned worker so app imports resolve
DEMO_DIR = ROOT / "resources" / "demo"
SCORE_GLOBS = ("*.mxl", "*.musicxml", "*.xml", "*.mid", "*.midi", "*.mei")


def sidecar_meta(sidecar: Path) -> dict | None:
    """Pull only the audio/score/name we need, reading the sidecar's own record.

    We parse just the JSON header, never the pitch frames, so a stale on-disk
    Pitch format can't break enumeration.
    """
    try:
        with lzma.open(sidecar) as fh:
            payload = json.load(fh)
    except Exception:
        return None
    rec = payload.get("recording", {}) or {}
    audio_file = rec.get("audio_file")
    if not audio_file:
        return None
    audio_path = sidecar.parent / audio_file
    score_path = (payload.get("score", {}) or {}).get("path")
    if not score_path or not Path(score_path).exists():
        # stored path moved: fall back to a score sitting next to the audio
        found = [p for g in SCORE_GLOBS for p in sorted(sidecar.parent.glob(g))]
        score_path = str(found[0]) if found else None
    return {
        "sidecar": str(sidecar),
        "audio_path": str(audio_path),
        "score_path": score_path,
        "name": rec.get("name"),
    }


def rebuild_one(meta: dict) -> dict:
    """Worker: full re-analysis of one recording, then save_cache.

    Runs in a spawned process, so every app import happens here. Pipeline stdout
    is muted to keep parallel logs readable.
    """
    t0 = time.perf_counter()
    audio_path, score_path, name = meta["audio_path"], meta["score_path"], meta["name"]
    result = {"audio": audio_path, "ok": False}
    if not Path(audio_path).exists():
        return {**result, "err": "audio file missing"}
    if not score_path or not Path(score_path).exists():
        return {**result, "err": "score file missing"}
    try:
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), \
                contextlib.redirect_stderr(dn):
            from algorithms.Config import Config
            from app_logic.midi.ScoreData import ScoreData
            from app_logic.user.ds.Recording import Recording
            from app_logic.JsonHandler import JsonHandler

            score = ScoreData()
            score.load(score_path)
            rec = Recording(score_data=score, config=Config())
            # load_cache=False: build fresh from audio (correct sr), never trust
            # the stale sidecar we're about to overwrite.
            rec.load_audio(
                audio_path, score_filepath=score_path,
                recording_name=name, load_cache=False,
            )
            rec.detect_pitches()

            # mirror perform.py::analyze, minus the UI refreshes
            rec.reset_analysis()
            rec.detect_notes()
            rec.resize_score(to_span="onset")
            rec.detect_mistakes()
            rec.stabilize_score_alignment()
            rec.reindex_mistakes()
            rec.update_alignment_distances()
            rec.trim_end()

            saved = JsonHandler(rec).save_cache(
                score_filepath=score_path, recording_name=name,
            )
        return {
            **result,
            "ok": bool(saved),
            "notes": len(rec.note_data.times),
            "secs": time.perf_counter() - t0,
            "err": None if saved else "save_cache returned False",
        }
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the pool
        return {
            **result,
            "err": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(),
            "secs": time.perf_counter() - t0,
        }


def _log(r: dict) -> None:
    name = Path(r["audio"]).name
    if r.get("ok"):
        print(f"  ✓ {name:28} {r.get('notes'):>3} notes  {r.get('secs', 0):5.1f}s")
    else:
        print(f"  ✗ {name:28} {r.get('err')}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--demo-dir", default=str(DEMO_DIR))
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 6))
    ap.add_argument("--filter", help="only rebuild sidecars whose path contains this substring")
    ap.add_argument("--serial", action="store_true", help="run in-process (easier to debug)")
    ap.add_argument("--dry-run", action="store_true", help="list the work and exit")
    args = ap.parse_args()

    sidecars = sorted(Path(args.demo_dir).glob("**/.*.json.xz"))
    metas = [m for s in sidecars if (m := sidecar_meta(s))]
    if args.filter:
        metas = [m for m in metas if args.filter in m["sidecar"]]
    if not metas:
        print(f"no demo sidecars found under {args.demo_dir}")
        return

    workers = 1 if args.serial else max(1, args.workers)
    print(f"{len(metas)} recording(s) to rebuild (workers={workers})")
    for m in metas:
        score = Path(m["score_path"]).name if m["score_path"] else "MISSING"
        print(f"  {Path(m['audio_path']).name:28} score={score}")
    if args.dry_run:
        return

    t0 = time.perf_counter()
    results: list[dict] = []
    if args.serial:
        for m in metas:
            results.append(rebuild_one(m))
            _log(results[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(rebuild_one, m): m for m in metas}
            for fut in as_completed(futures):
                results.append(fut.result())
                _log(results[-1])

    ok = sum(1 for r in results if r.get("ok"))
    print(f"\ndone: {ok}/{len(results)} rebuilt in {time.perf_counter() - t0:.1f}s")
    for r in results:
        if not r.get("ok") and r.get("trace"):
            print(f"\n--- {Path(r['audio']).name} ---\n{r['trace']}")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
