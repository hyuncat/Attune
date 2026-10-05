"""MistakeNotebook implementation and owned benchmark helpers."""

from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys
from algorithms.Config import Config
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import time
import pretty_midi
from benchmarks.modules.mistake.datasets.NativeDatasets import DATASETS
from benchmarks.modules.mistake.datasets.NativeDatasets import SPLIT_SHA256
from benchmarks.modules.mistake.datasets.NativeDatasets import digest
from benchmarks.modules.mistake.datasets.NativeDatasets import download
from benchmarks.modules.mistake.datasets.NativeDatasets import save_json
from benchmarks.modules.mistake.datasets.NativeDatasets import tree
import mir_eval


class MistakeNotebook:
    """Notebook configuration, presentation, and compatibility entry points."""

    @staticmethod
    def completed_results(request):
        from benchmarks.modules.mistake.MistakeCache import MistakeCache

        return MistakeCache.completed_results(request)

    @staticmethod
    def run(request):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        return MistakeBenchmarker.run_request(request)

    @staticmethod
    def comparison_table(
        rows,
        *,
        input_kind="oracle_notes",
        tolerance=0.1,
        rate=0.25,
        metric="audio_pitch",
    ):
        """Pooled mistake-event scores, never mean per-case F1 or note accuracy."""
        selected = rows[
            (rows.input == input_kind)
            & np.isclose(rows.tolerance, tolerance)
            & np.isclose(rows.rate, rate)
            & (rows.metric == metric)
        ]
        if selected.empty:
            raise ValueError(
                f"No results for {input_kind}, {metric}, rate={rate}, tolerance={tolerance}"
            )
        if selected.duplicated(["case_id", "method"]).any():
            raise ValueError("Duplicate case/method results")
        counts = selected.groupby("method")[["tp", "fp", "fn"]].sum()
        counts["Precision %"] = (
            100 * counts.tp / (counts.tp + counts.fp).replace(0, np.nan)
        )
        counts["Recall %"] = (
            100 * counts.tp / (counts.tp + counts.fn).replace(0, np.nan)
        )
        counts["F1 %"] = (
            200 * counts.tp / (2 * counts.tp + counts.fp + counts.fn).replace(0, np.nan)
        )
        counts["Cases"] = selected.groupby("method").case_id.nunique()
        return (
            counts[["Cases", "tp", "fp", "fn", "Precision %", "Recall %", "F1 %"]]
            .sort_values("F1 %", ascending=False)
            .round(1)
        )

    @staticmethod
    def validate_pairing(detected, symbolic, output, symbolic_output=None):
        """Case IDs include implementation hashes; pair by inputs, verifying changed IDs."""
        from hashlib import sha256

        keys = ["source", "seed", "rate"]
        if detected.duplicated(keys).any() or symbolic.duplicated(keys).any():
            raise ValueError(
                "Duplicate symbolic/detected source, seed, rate evaluations"
            )
        left = detected.set_index(keys).case_id
        right = symbolic.set_index(keys).case_id
        if set(left.index) != set(right.index):
            raise ValueError("Symbolic and detected source/seed/rate selections differ")

        def identity(case_id, folder=None):
            candidates = (
                [Path(folder) / "cases" / case_id / "manifest.json"]
                if folder is not None
                else sorted(
                    Path(output).parent.glob(f"*/cases/{case_id}/manifest.json")
                )
            )
            identities = set()
            for manifest in candidates:
                if not manifest.is_file():
                    continue
                case = json.loads(manifest.read_text())
                spec = case["spec"]
                truth_path = manifest.parent / "net_truth.json"
                if not Path(case["midi"]).is_file() or not truth_path.is_file():
                    continue
                truth = json.loads(truth_path.read_text())["net_truth"]
                identities.add(
                    (
                        spec["source_hash"],
                        sha256(Path(case["midi"]).read_bytes()).hexdigest(),
                        json.dumps(truth, sort_keys=True),
                    )
                )
            if len(identities) != 1:
                raise ValueError(
                    f"Cannot verify unique cached inputs/truth for case {case_id}"
                )
            return identities.pop()

        for key in left.index:
            current_id, symbolic_id = (left.loc[key], right.loc[key])
            if current_id == symbolic_id:
                continue
            if identity(current_id, output) != identity(symbolic_id, symbolic_output):
                raise ValueError(
                    f"Symbolic and detected MIDI or labels differ for {key}"
                )

    @staticmethod
    def overview(
        output, tolerance=0.1, rate=0.25, *, symbolic_rows=None, symbolic_output=None
    ):
        output = Path(output)
        run = json.loads((output / "run.json").read_text())
        if run.get("status") != "complete":
            raise ValueError(
                "This run is incomplete; inspect run.json before interpreting it."
            )
        rows = pd.read_csv(output / "rows.csv")
        keys = ["case_id", "input", "method", "tolerance", "metric"]
        if rows.duplicated(keys).any():
            raise ValueError("Duplicate evaluation keys found in rows.csv")
        primary = rows[
            np.isclose(rows.tolerance, tolerance) & (rows.metric == "audio_pitch")
        ]
        injected = primary[np.isclose(primary.rate, rate)]

        def metrics(frame):
            totals = frame.groupby("method")[["tp", "fp", "fn"]].sum()
            totals["Precision %"] = (
                100 * totals.tp / (totals.tp + totals.fp).replace(0, np.nan)
            )
            totals["Recall %"] = (
                100 * totals.tp / (totals.tp + totals.fn).replace(0, np.nan)
            )
            totals["F1 %"] = (
                200
                * totals.tp
                / (2 * totals.tp + totals.fp + totals.fn).replace(0, np.nan)
            )
            return totals

        main = metrics(injected[injected.input != "oracle_notes"])
        leaderboard = main[["Precision %", "Recall %", "F1 %"]].sort_values(
            "F1 %", ascending=False
        )
        exact = injected[injected.input == "oracle_notes"]
        if symbolic_rows is not None:
            exact = symbolic_rows[
                (symbolic_rows.input == "oracle_notes")
                & np.isclose(symbolic_rows.tolerance, tolerance)
                & np.isclose(symbolic_rows.rate, rate)
                & (symbolic_rows.metric == "audio_pitch")
            ]
            for method in set(exact.method) & set(injected.method):
                MistakeNotebook.validate_pairing(
                    injected[
                        (injected.method == method) & (injected.input != "oracle_notes")
                    ],
                    exact[exact.method == method],
                    output,
                    symbolic_output,
                )
            if exact.duplicated(["case_id", "method"]).any():
                raise ValueError("Duplicate symbolic evaluations")
        oracle = metrics(exact)
        extraction = (
            main[["F1 %"]]
            .rename(columns={"F1 %": "Detected notes F1 %"})
            .join(
                oracle[["F1 %"]].rename(columns={"F1 %": "Exact notes F1 %"}),
                how="inner",
            )
        )
        clean = primary[(primary.rate == 0) & (primary.input != "oracle_notes")]
        varying = (
            clean.groupby(["source", "method"])[["tp", "fp", "fn"]]
            .nunique()
            .gt(1)
            .any(axis=1)
        )
        clean_table = clean.groupby("method").agg(
            false_alarms=("fp", "sum"),
            score_notes=("score_notes", "sum"),
            stems=("source", "nunique"),
            evaluations=("case_id", "nunique"),
        )
        clean_table["False alarms / 100 notes"] = (
            100 * clean_table.false_alarms / clean_table.score_notes
        )
        clean_table["Stems with differing seed results"] = varying.groupby(
            level="method"
        ).sum()
        clean_table = clean_table.rename(
            columns={
                "false_alarms": "False alarms",
                "stems": "Clean stems",
                "evaluations": "Clean evaluations",
            }
        )
        clean_table = clean_table[
            [
                "Clean stems",
                "Clean evaluations",
                "False alarms",
                "False alarms / 100 notes",
                "Stems with differing seed results",
            ]
        ]
        return dict(
            leaderboard=leaderboard.round(1),
            extraction=extraction.round(1),
            clean=clean_table.sort_values("False alarms").round(2),
            sources=injected.source.nunique(),
            injected_cases=injected.case_id.nunique(),
            raw_rows=len(rows),
            summary_rows=len(pd.read_csv(output / "summary.csv")),
        )

    @staticmethod
    def native_overview(output, tolerance=0.1):
        """Average over every case, explicitly assigning perfect error-free cases F1=1.

        This reporting-only convention does not rewrite raw undefined event F1 or
        the native-style class means (whose empty classes retain F1=0).
        """
        output = Path(output)
        metadata = json.loads((output / "run.json").read_text())
        frame = pd.read_csv(output / "rows.csv")
        expected = {c["case_id"] for c in metadata["contract"]["manifest"]["cases"]}
        primary = frame[
            (frame.protocol == "common_pooled_events")
            & (frame.metric == "audio_pitch")
            & (frame.tolerance == tolerance)
        ]
        if primary.duplicated(["method", "case_id"]).any():
            raise ValueError("Duplicate case/method rows; cannot average reliably")
        rows = []
        for method in metadata["contract"]["methods"]:
            cases = primary[primary.method == method]
            if set(cases.case_id) - expected:
                raise ValueError("Saved rows contain cases outside the manifest")
            complete = set(cases.case_id) == expected
            tp, fp, fn = cases[["tp", "fp", "fn"]].sum()
            denominator = 2 * tp + fp + fn
            case_denominator = 2 * cases.tp + cases.fp + cases.fn
            case_f1 = (
                2 * cases.tp / case_denominator.where(case_denominator != 0)
            ).fillna(1.0)
            native = frame[
                (frame.method == method)
                & (frame.protocol == "native_style_macro")
                & (frame.metric == "three_class_average")
                & (frame.tolerance == 0.05)
            ]
            native_complete = (
                not native.duplicated("case_id").any()
                and set(native.case_id) == expected
            )
            rows.append(
                {
                    "Method": method,
                    "Cases": f"{len(cases)}/{len(expected)}",
                    "Mean case error F1 (100 ms), %": (
                        100 * case_f1.mean() if complete else float("nan")
                    ),
                    "Pooled error F1 (100 ms), %": (
                        100 * (2 * tp / denominator)
                        if complete and denominator
                        else float("nan")
                    ),
                    "Mean three-class F1 (50 ms), %": (
                        100 * native.f1.mean() if native_complete else float("nan")
                    ),
                    "TP": int(tp),
                    "FP": int(fp),
                    "FN": int(fn),
                }
            )
        table = pd.DataFrame(rows)
        if tolerance != 0.1:
            table = table.rename(
                columns={
                    c: c.replace("100 ms", f"{tolerance * 1000:g} ms")
                    for c in table.columns
                }
            )
        return table.round(1)

    @staticmethod
    def audit_note_audit(notes):
        """Half-open intervals: touching boundaries are not overlaps. No reordering."""
        if not notes:
            return dict(
                notes=0,
                overlap_pairs=0,
                polyphonic_seconds=0.0,
                max_voices=0,
                max_overlap_seconds=0.0,
                delayed_100ms=0,
                max_delay_seconds=0.0,
            )
        pairs = [
            min(a.end, b.end) - max(a.start, b.start)
            for j, a in enumerate(notes)
            for b in notes[j + 1 :]
            if min(a.end, b.end) - max(a.start, b.start) > 1e-07
        ]
        events = sorted([(n.start, 1) for n in notes] + [(n.end, -1) for n in notes])
        voices = maximum = 0
        polyphonic = 0.0
        previous = events[0][0]
        for at, delta in events:
            if voices > 1:
                polyphonic += at - previous
            voices += delta
            maximum = max(maximum, voices)
            previous = at
        cursor, previous_end, delays = (round(notes[0].start * 250), 1000000, [])
        for note in notes:
            onset, end = (round(note.start * 250), round(note.end * 250))
            cursor += max(0, onset - previous_end)
            delays.append(cursor / 250 - note.start)
            cursor += end - onset
            previous_end = end
        return dict(
            notes=len(notes),
            overlap_pairs=len(pairs),
            polyphonic_seconds=polyphonic,
            max_voices=maximum,
            max_overlap_seconds=max(pairs, default=0.0),
            delayed_100ms=sum((d > 0.1 for d in delays)),
            max_delay_seconds=max(delays),
        )

    @staticmethod
    def audit_census(root, splits=("test",), workers=12):
        """All available mirror pieces in requested splits; cache small mix MIDIs only.

        Failures are explicit and never silently removed from the denominator.
        Uses each instrument independently, retaining PrettyMIDI's upstream load order.
        """
        root = Path(root)
        if workers < 1:
            raise ValueError("workers must be positive")
        repo, revision, split_name = DATASETS["coco"]
        split_path = download(
            repo, revision, split_name, root / "split.json", sha256=SPLIT_SHA256["coco"]
        )
        split = json.loads(split_path.read_text())
        available = {
            Path(e["path"]).name
            for e in tree(repo, revision, "score", root / "pieces.json")
            if e["type"] == "directory"
        }
        piece_splits = {
            Path(value).stem: split["split"][key]
            for key, value in split["midi_filename"].items()
        }
        selected = sorted((p for p in available if piece_splits.get(p) in splits))
        if not selected:
            raise ValueError("No pieces selected")

        def inspect(piece):
            path = root / "midi" / (piece + ".mid")
            for attempt in range(3):
                try:
                    download(repo, revision, f"mistake/{piece}/mix.mid", path)
                    midi = pretty_midi.PrettyMIDI(str(path))
                    return [
                        dict(
                            piece=piece,
                            split=piece_splits[piece],
                            part=index,
                            program=instrument.program,
                            sha256=digest(path),
                            **MistakeNotebook.audit_note_audit(instrument.notes),
                        )
                        for index, instrument in enumerate(midi.instruments)
                        if instrument.notes
                    ]
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)

        rows, failures = ([], [])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(inspect, piece): piece for piece in selected}
            for count, future in enumerate(as_completed(futures), 1):
                try:
                    rows.extend(future.result())
                except Exception as error:
                    failures.append(dict(piece=futures[future], error=str(error)))
                if count % 100 == 0 or count == len(selected):
                    print(
                        f"Coco-E MIDI census: {count}/{len(selected)} pieces; {len(failures)} failures",
                        flush=True,
                    )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(["split", "piece", "part"])
        frame.to_csv(root / "stems.csv", index=False)
        save_json(
            root / "metadata.json",
            dict(
                repo=repo,
                revision=revision,
                splits=list(splits),
                selected_pieces=len(selected),
                completed_pieces=frame.piece.nunique() if rows else 0,
                failures=failures,
                workers=workers,
                split_sha256=digest(split_path),
                code_sha256=digest(__file__),
                scope="All available author-mirror pieces in requested splits; not the full original 240k corpus",
                source="mistake/<piece>/mix.mid, each nonempty instrument separately",
                warning="MIDI overlap and hypothetical schedule drift are not verified audio error counts",
            ),
        )
        if failures:
            raise RuntimeError(
                f"{len(failures)} pieces failed; see metadata.json and rerun to retry cached census"
            )
        return frame

    @staticmethod
    def audit_prevalence(frame):
        rows = []
        for split, group in frame.groupby("split"):
            affected = group.max_overlap_seconds.gt(0.004)
            rows.append(
                dict(
                    split=split,
                    pieces=group.piece.nunique(),
                    stems=len(group),
                    overlapping_stems=int(affected.sum()),
                    overlapping_stems_pct=100 * affected.mean(),
                    affected_pieces=group.loc[affected, "piece"].nunique(),
                    overlaps_gt100ms=int(group.max_overlap_seconds.gt(0.1).sum()),
                    delayed_stems_100ms=int(group.delayed_100ms.gt(0).sum()),
                    notes=int(group.notes.sum()),
                    hypothetical_delayed_notes=int(group.delayed_100ms.sum()),
                    polyphonic_seconds=group.polyphonic_seconds.sum(),
                )
            )
        return pd.DataFrame(rows)

    @staticmethod
    def audit_example(manifest, case_id, start, end):
        """Listen and inspect a fixed window; all audio shares the original time axis."""
        import matplotlib.pyplot as plt
        import soundfile as sf
        from IPython.display import Audio, Markdown, display
        from benchmarks.modules.mistake.MistakeNotebook import MistakeNotebook

        sequential_schedule = MistakeNotebook.renderer_sequential_schedule
        case = next((c for c in manifest["cases"] if c["case_id"] == case_id))
        labels = []
        for kind in ("correct", "extra"):
            midi = pretty_midi.PrettyMIDI(case["files"][kind])
            labels.extend(
                (
                    dict(kind=kind, pitch=n.pitch, onset=n.start, end=n.end)
                    for i in midi.instruments
                    for n in i.notes
                )
            )
        visible = pd.DataFrame(
            [n for n in labels if n["onset"] < end and n["end"] > start]
        )
        display(visible.round(4))
        schedule = sequential_schedule(case["rendering"]["performance_midi"])
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 8), sharex=True, constrained_layout=True
        )
        for n in labels:
            if n["onset"] < end and n["end"] > start:
                axes[0].plot(
                    [n["onset"], n["end"]],
                    [n["pitch"]] * 2,
                    color="tab:red" if n["kind"] == "extra" else "tab:blue",
                    linewidth=6,
                )
        for n in schedule:
            if n["onset"] < end and n["end"] > start:
                axes[0].plot(
                    [n["onset"], n["end"]],
                    [n["pitch"] + 0.18] * 2,
                    "k--",
                    linewidth=1.5,
                )
        axes[0].set(
            title=f"{case_id}: MIDI correct (blue), extra (red), sequential hypothesis (dashed)",
            ylabel="MIDI pitch",
        )
        clips = {}
        for ax, (name, path) in zip(
            axes[1:],
            [
                ("Supplied audio", case["rendering"]["original_audio"]["performance"]),
                ("FluidSynth / MuseScore", case["files"]["performance"]),
            ],
        ):
            with sf.SoundFile(path) as audio:
                sr = audio.samplerate
                audio.seek(round(start * sr))
                y = audio.read(round((end - start) * sr), always_2d=True).mean(axis=1)
            clips[name] = (y, sr)
            ax.specgram(
                y,
                NFFT=2048,
                Fs=sr,
                noverlap=1792,
                xextent=(start, start + len(y) / sr),
                cmap="magma",
            )
            for n in visible.to_dict("records"):
                ax.hlines(
                    pretty_midi.note_number_to_hz(n["pitch"]),
                    max(start, n["onset"]),
                    min(end, n["end"]),
                    color="cyan",
                    linewidth=1,
                )
            freqs = pretty_midi.note_number_to_hz(visible.pitch.to_numpy())
            ax.set(
                ylim=(freqs.min() * 0.85, freqs.max() * 1.15), ylabel="Hz", title=name
            )
        axes[-1].set(xlim=(start, end), xlabel="Seconds in original file")
        display(fig)
        plt.close(fig)
        for name, (y, sr) in clips.items():
            display(Markdown(f"**{name} — {start:g}–{end:g} s**"))
            display(Audio(y, rate=sr))
        return clips

    @staticmethod
    def audit_spectral_evidence(manifest, case_id, windows):
        """Relative fundamental-band peaks; not loudness or full transcription."""
        import soundfile as sf

        case = next((c for c in manifest["cases"] if c["case_id"] == case_id))
        rows = []
        for start, end, reference_pitch, target_pitch in windows:
            for name, path in [
                ("Supplied", case["rendering"]["original_audio"]["performance"]),
                ("FluidSynth", case["files"]["performance"]),
            ]:
                with sf.SoundFile(path) as audio:
                    sr = audio.samplerate
                    audio.seek(round(start * sr))
                    y = audio.read(round((end - start) * sr), always_2d=True).mean(
                        axis=1
                    )
                spectrum = abs(np.fft.rfft(y * np.hanning(len(y)), n=65536))
                frequencies = np.fft.rfftfreq(65536, 1 / sr)

                def peak(pitch):
                    f = pretty_midi.note_number_to_hz(pitch)
                    band = (frequencies >= f * 2 ** (-30 / 1200)) & (
                        frequencies <= f * 2 ** (30 / 1200)
                    )
                    return max(float(spectrum[band].max()), 1e-12)

                rows.append(
                    dict(
                        case=case_id,
                        audio=name,
                        start=start,
                        end=end,
                        reference_pitch=reference_pitch,
                        target_pitch=target_pitch,
                        target_relative_db=20
                        * np.log10(peak(target_pitch) / peak(reference_pitch)),
                    )
                )
        return pd.DataFrame(rows)

    @staticmethod
    def renderer_sequential_schedule(midi):
        instruments = [
            i for i in pretty_midi.PrettyMIDI(str(midi)).instruments if i.notes
        ]
        if len(instruments) != 1:
            raise ValueError("Expected one nonempty instrument")
        notes = instruments[0].notes
        cursor = round(notes[0].start * 250)
        previous_offset = 1000000
        result = []
        for note in notes:
            onset, offset = (round(note.start * 250), round(note.end * 250))
            cursor += max(0, onset - previous_offset)
            result.append(
                dict(
                    onset=cursor / 250,
                    end=(cursor + offset - onset) / 250,
                    pitch=note.pitch,
                    label_onset=note.start,
                    label_end=note.end,
                )
            )
            cursor += offset - onset
            previous_offset = offset
        return result

    @staticmethod
    def renderer_inspect(native, soundfont):
        native, soundfont = (Path(native), Path(soundfont))
        manifest = json.loads((soundfont / "manifest.json").read_text())
        rows, boundaries = ([], [])

        def arrays(events):
            return (
                np.array([[e["onset"], e["end"]] for e in events]).reshape(-1, 2),
                np.array([440 * 2 ** ((e["pitch"] - 69) / 12) for e in events]),
            )

        for case in manifest["cases"]:
            hypothesis = MistakeNotebook.renderer_sequential_schedule(
                case["rendering"]["performance_midi"]
            )
            for renderer, root in [("Original", native), ("FluidSynth", soundfont)]:
                saved = json.loads(
                    (
                        root / "cases" / case["case_id"] / "attune/result.json"
                    ).read_text()
                )
                notes = [
                    e
                    for e in saved["prediction"]["events"]
                    if e["kind"] in ("extra", "correct")
                ]
                labels = [
                    e for e in saved["truth"] if e["kind"] in ("extra", "correct")
                ]
                for reference_name, reference in [
                    ("official MIDI", labels),
                    ("sequential schedule hypothesis", hypothesis),
                ]:
                    if renderer == "FluidSynth" and reference_name != "official MIDI":
                        continue
                    for gate in (0.05, 0.1, 0.2):
                        matches = mir_eval.transcription.match_notes(
                            *arrays(reference),
                            *arrays(notes),
                            onset_tolerance=gate,
                            pitch_tolerance=50,
                            offset_ratio=None,
                        )
                        tp = len(matches)
                        rows.append(
                            dict(
                                case=case["case_id"],
                                renderer=renderer,
                                reference=reference_name,
                                gate=gate,
                                tp=tp,
                                fp=len(notes) - tp,
                                fn=len(reference) - tp,
                            )
                        )
                        if gate == 0.2:
                            boundaries.extend(
                                (
                                    dict(
                                        case=case["case_id"],
                                        renderer=renderer,
                                        reference=reference_name,
                                        onset_error_ms=1000
                                        * (notes[j]["onset"] - reference[i]["onset"]),
                                        offset_error_ms=1000
                                        * (notes[j]["end"] - reference[i]["end"]),
                                    )
                                    for i, j in matches
                                )
                            )
        detail = pd.DataFrame(rows)
        summary = detail.groupby(["renderer", "reference", "gate"], as_index=False)[
            ["tp", "fp", "fn"]
        ].sum()
        summary["F1 %"] = 200 * summary.tp / (2 * summary.tp + summary.fp + summary.fn)
        return (summary.round(2), detail, pd.DataFrame(boundaries))
