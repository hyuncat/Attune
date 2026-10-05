"""Replay saved benchmark cases with the production repeat-only refiner."""

from notebooks.archive.PreviousRepeatMistakeChecker import groups, repeat_split
from app_logic.NoteData import Note, NoteData


def run(output):
    import json
    from pathlib import Path
    from contextlib import redirect_stdout
    import pandas as pd
    from algorithms.Config import Config
    from benchmarks.modules.mistake.provenance.PipelineAudit import (
        sequence,
        snapshot,
        note_matches,
    )
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
    from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData

    output = Path(output)
    dest = output / "pipeline_audit"
    rows = pd.read_csv(output / "rows.csv").drop_duplicates("case_id")
    audited = set(pd.read_csv(dest / "reproduction.csv").case_id)
    rows = rows[rows.case_id.isin(audited)]
    assert len(rows) == len(audited), "Missing saved audit cases"
    bench = MistakeBenchmarker()
    results, details = ([], [])
    with (dest / "repeat_only.log").open("w") as log, redirect_stdout(log):
        for row in rows.itertuples():
            saved = json.loads((dest / (row.case_id + ".json")).read_text())
            manifest = json.loads(
                (output / "cases" / row.case_id / "manifest.json").read_text()
            )
            reference = OneInstrumentScoreData(manifest["source_midi"]).note_data
            performed = sequence(OneInstrumentScoreData(manifest["midi"]).note_data)
            rec = bench.attune.recording_for(
                Config(**json.loads(row.config)),
                score_data=OneInstrumentScoreData(manifest["source_midi"]),
            )
            rec.note_data = NoteData()
            for n in saved["Attune (no refinement)"]["notes"]:
                rec.note_data.write_note(
                    Note(n["id"], n["onset"], n["end"], [n["pitch"]])
                )
            rec.resize_score(to_span="onset")
            rec.detect_mistakes()
            baseline = snapshot(rec, reference, saved["truth"])
            assert all(
                (
                    baseline[k] == saved["Attune (no refinement)"][k]
                    for k in ("tp", "fp", "fn")
                )
            )
            rec.note_data, proposals = repeat_split(
                sequence(rec.note_data),
                sequence(rec.score_data.note_data),
                rec.mistake_detector,
            )
            rec.detect_mistakes()
            state = snapshot(rec, reference, saved["truth"])
            matched = note_matches(performed, sequence(rec.note_data), 0.1)
            results.append(
                dict(
                    case_id=row.case_id,
                    rate=row.rate,
                    instrument=row.instrument,
                    **{k: state[k] for k in ("tp", "fp", "fn")},
                    note_tp=len(matched),
                    note_fp=len(rec.note_data.times) - len(matched),
                    note_fn=len(performed) - len(matched)
                )
            )
            details.append(dict(case_id=row.case_id, proposals=proposals, result=state))
    pd.DataFrame(results).to_csv(dest / "repeat_only.csv", index=False)
    (dest / "repeat_only.json").write_text(json.dumps(details, indent=2))
    table = (
        pd.DataFrame(results)
        .groupby("rate")[["tp", "fp", "fn", "note_tp", "note_fp", "note_fn"]]
        .sum()
    )
    table["mistake_f1"] = 200 * table.tp / (2 * table.tp + table.fp + table.fn)
    table["note_f1"] = (
        200 * table.note_tp / (2 * table.note_tp + table.note_fp + table.note_fn)
    )
    print(table.to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    run(parser.parse_args().output)
