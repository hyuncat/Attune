from __future__ import annotations

"Notebook presentation and compatibility facade for note benchmarks."
from dataclasses import asdict
from benchmarks.paths import REPO_ROOT
from benchmarks.paths import RESULTS_ROOT
from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation
from benchmarks.modules.note.NoteBenchmarker import NotebookConfig
from benchmarks.modules.note.NoteBenchmarker import METHODS
from benchmarks.modules.note.NoteBenchmarker import VERSION


class NoteNotebook(NoteEvaluation):

    def show_configuration(self):
        import pandas as pd
        from IPython.display import display

        table = pd.DataFrame(asdict(self.config).items(), columns=["setting", "value"])
        display(table)
        return table

    def run_preliminary_coco(self):
        return self.run_preliminary("coco")

    def run_preliminary_urmp(self):
        return self.run_preliminary("urmp")

    def run_preliminary_bach10(self):
        return self.run_preliminary("bach10-original")

    @staticmethod
    def historical_audit():
        import pandas as pd

        frames = []
        for path in sorted(
            (RESULTS_ROOT / "note" / "raw_outputs").glob("*/coco_*.csv")
        ):
            frame = pd.read_csv(path)
            frame["saved_method"] = path.parent.name
            frame["source_csv"] = str(path.relative_to(REPO_ROOT))
            frames.append(frame)
        if not frames:
            return (pd.DataFrame(), pd.DataFrame())
        rows = pd.concat(frames, ignore_index=True)
        if rows.duplicated(["saved_method", "Track ID"]).any():
            raise ValueError(
                "Duplicate historical method/track rows; inspect before aggregating"
            )
        summary = rows.groupby("saved_method")["F-measure"].agg(
            ["size", "count", "mean"]
        )
        return (rows, summary.sort_values("mean", ascending=False))


__all__ = ["NoteNotebook", "NotebookConfig", "METHODS", "VERSION"]
