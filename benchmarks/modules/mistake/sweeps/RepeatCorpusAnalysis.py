"""Summarize completed URMP repeat experiments without inference or alignment."""

from pathlib import Path
import json
from collections import Counter
import pandas as pd
import pretty_midi
from app_logic.NoteData import Note
from benchmarks.modules.mistake.provenance.PipelineAudit import note_matches


def analyze(output):
    output = Path(output)
    rows = pd.read_csv(output / "rows.csv")
    status, changes, eligible_match = Counter(), Counter(), Counter()
    cuts, tolerances = [], []
    eligible_total = 0
    for track in rows.track.unique():
        saved = json.loads((output / (track + ".json")).read_text())
        reference = [
            Note(i, n["onset"], n["end"], [n["pitch"]])
            for i, n in enumerate(saved["reference"])
        ]
        ns = sorted(
            pretty_midi.PrettyMIDI(str(output / (track + ".mid"))).instruments[0].notes,
            key=lambda n: n.start,
        )
        eligible = {
            j
            for j in range(1, len(ns))
            if ns[j].pitch == ns[j - 1].pitch
            and -0.001 <= ns[j].start - ns[j - 1].end <= 0.05
        }
        eligible_total += len(eligible)
        baseline = {j for _, j in saved["methods"]["No refinement"]["matches"]}
        for method, value in saved["methods"].items():
            matched = {j for _, j in value["matches"]}
            eligible_match[method] += len(matched & eligible)
            changes[method + " gained"] += len(matched - baseline)
            changes[method + " lost"] += len(baseline - matched)
            estimates = [
                Note(i, n["onset"], n["end"], [n["pitch"]])
                for i, n in enumerate(value["notes"])
            ]
            for tolerance in [0.05, 0.1, 0.2]:
                count = len(note_matches(reference, estimates, tolerance))
                tolerances.append(
                    dict(
                        track=track,
                        method=method,
                        tolerance=tolerance,
                        tp=count,
                        fp=len(estimates) - count,
                        fn=len(reference) - count,
                    )
                )
        for proposal in saved["methods"]["Repeat only"]["proposals"]:
            status[proposal["status"]] += 1
            if proposal["status"] != "split":
                continue
            for _, onset in proposal["cuts"]:
                nearest = min(
                    saved["repeat_indices"],
                    key=lambda j: abs(reference[j].start_time - onset),
                )
                cuts.append(
                    dict(
                        track=track,
                        onset=onset,
                        nearest_repeat_error=onset - reference[nearest].start_time,
                    )
                )
    pd.DataFrame(tolerances).to_csv(output / "tolerance_rows.csv", index=False)
    summary = (
        pd.DataFrame(tolerances)
        .groupby(["method", "tolerance"])[["tp", "fp", "fn"]]
        .sum()
    )
    summary["f1"] = 200 * summary.tp / (2 * summary.tp + summary.fp + summary.fn)
    summary.to_csv(output / "tolerance_summary.csv")
    result = dict(
        proposal_status=dict(status),
        eligible_repeat_total=eligible_total,
        eligible_repeat_matches=dict(eligible_match),
        matched_changes=dict(changes),
        cuts=cuts,
    )
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "cuts"}, indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output", nargs="?", default="benchmarks/results/repeat_refinement_urmp"
    )
    analyze(parser.parse_args().output)
