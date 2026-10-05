"""Paired replay of URMP repeat recordings: frozen baseline versus current production.

Use saved pre-refinement audio-derived notes from the existing URMP experiment.
No inference, annotation-driven fitting, or changes to historical results.
"""

from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from algorithms.Config import Config
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.Recording import Recording
from algorithms.RepeatSplitter import RepeatSplitter
from notebooks.archive.PreviousRepeatMistakeChecker import (
    MistakeChecker as PreviousChecker,
)
from benchmarks.modules.mistake.provenance.PipelineAudit import note_matches, sequence
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
from benchmarks.paths import REPO_ROOT


def from_rows(rows):
    data = NoteData()
    for i, n in enumerate(rows):
        data.write_note(Note(i, n["onset"], n["end"], [n["pitch"]]))
    return data


def counts(reference, estimates, tolerance):
    tp = len(note_matches(reference, estimates, tolerance))
    return dict(tp=tp, fp=len(estimates) - tp, fn=len(reference) - tp)


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; preserving {output}")
    manifest = json.loads((source / "run.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The source URMP experiment must be complete")
    # Check all required inputs before creating the destination.
    inputs = []
    for item in manifest["selected"]:
        track = item["track"]
        saved_path, midi = source / f"{track}.json", source / f"{track}.mid"
        saved = json.loads(saved_path.read_text())
        score = OneInstrumentScoreData(midi)
        repeat_groups = [
            g for g in RepeatSplitter.groups(sequence(score.note_data)) if len(g) > 1
        ]
        if repeat_groups:
            inputs.append((item, saved, midi, repeat_groups))
    if not inputs:
        raise ValueError("No contiguous score-repeat cases found")
    output.mkdir(parents=True)
    code_files = [
        "algorithms/RepeatSplitter.py",
        "algorithms/MistakeDetector.py",
        "algorithms/Config.py",
        "app_logic/user/ds/Recording.py",
        "notebooks/archive/PreviousRepeatMistakeChecker.py",
        "benchmarks/modules/mistake/sweeps/CollapsedRepeatComparison.py",
    ]
    metadata = dict(
        status="running",
        source=str(source),
        tracks=[x[0]["track"] for x in inputs],
        protocol="Paired frozen pre-refinement audio-derived notes; official score MIDI; "
        "performed annotations used only for evaluation. Before includes one robust "
        "onset fit plus the previous global-cost repeat refiner. After uses production "
        "original-note fitting and local missing-repeat recovery. No injected mistakes.",
        selection="Existing cached URMP cohort with exact score/annotation pitch sequences, "
        "deduplicated audio, and at least one contiguous score repeat. Development data.",
        config=asdict(Config()),
        code_sha256={
            p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()
            for p in code_files
        },
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for item, _, midi, _ in inputs
            for p in (midi, source / f"{item['track']}.json")
        },
    )
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for number, (item, saved, midi, repeat_groups) in enumerate(inputs, 1):
        track = item["track"]
        print(f"{number}/{len(inputs)} {track}", flush=True)
        reference = sequence(from_rows(saved["reference"]))
        # Score/annotation sequences agree in this preselected corpus. Include
        # every member (including the first) of each contiguous repeat group.
        score = sequence(OneInstrumentScoreData(midi).note_data)
        if [n.midi_num for n in score] != [n.midi_num for n in reference]:
            raise ValueError(f"Score/annotation pitch mismatch: {track}")
        index_by_id = {n.id: i for i, n in enumerate(score)}
        # Fixed annotation windows select estimates for regional precision as
        # well as recall; wrong-pitch/excess notes within them count as FP.
        windows = [
            (
                reference[index_by_id[g[0].id]].start_time - 0.1,
                reference[index_by_id[g[-1].id]].end_time + 0.1,
            )
            for g in repeat_groups
        ]
        repeat_reference = [
            n for n in reference if any(a <= n.start_time <= b for a, b in windows)
        ]
        details = {}
        with (output / f"{track}.log").open("w") as log, redirect_stdout(log):
            for method in ("Before", "After"):
                rec = Recording(
                    score_data=OneInstrumentScoreData(midi), config=Config()
                )
                rec.note_data = from_rows(saved["methods"]["No refinement"]["notes"])
                if method == "Before":
                    rec.repeat_splitter = PreviousChecker(recording=rec)
                    rec.resize_score(to_span="onset")
                    rec.detect_mistakes()
                    rec.refit_score_alignment_once()
                    rec.stabilize_score_alignment()
                    rec.reindex_mistakes()
                else:
                    rec.align_score_and_refine()
                estimates = sequence(rec.note_data)
                regional = [
                    n
                    for n in estimates
                    if any(a <= n.start_time <= b for a, b in windows)
                ]
                for scope, refs, ests in [
                    ("Whole repeat-case recordings", reference, estimates),
                    ("Repeat regions", repeat_reference, regional),
                ]:
                    for tolerance in (0.05, 0.1, 0.2):
                        rows.append(
                            dict(
                                track=track,
                                instrument=item["instrument"],
                                method=method,
                                scope=scope,
                                tolerance=tolerance,
                                **counts(refs, ests, tolerance),
                            )
                        )
                details[method] = dict(
                    notes=[
                        dict(onset=n.start_time, end=n.end_time, pitch=n.midi_num[0])
                        for n in estimates
                    ],
                    proposals=rec.repeat_splitter.proposals,
                    pitch_mistakes=len(rec.alignment.pitch_mistakes),
                )
        (output / f"{track}.json").write_text(json.dumps(details, indent=2))
        pd.DataFrame(rows).to_csv(output / "rows.csv", index=False)
    table = pd.DataFrame(rows)
    summary = table.groupby(["scope", "tolerance", "method"])[["tp", "fp", "fn"]].sum()
    summary["precision"] = 100 * summary.tp / (summary.tp + summary.fp)
    summary["recall"] = 100 * summary.tp / (summary.tp + summary.fn)
    summary["f1"] = 200 * summary.tp / (2 * summary.tp + summary.fp + summary.fn)
    summary.to_csv(output / "summary.csv")
    metadata["status"] = "complete"
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    return summary.reset_index()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="benchmarks/results/repeat_refinement_urmp_production"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args.source, args.output).to_string(index=False))
