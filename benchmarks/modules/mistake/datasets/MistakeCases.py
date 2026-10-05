"""CocoChorales sampling and instrument-preserving synthetic mistake cases."""

import hashlib
import json
from pathlib import Path
import pandas as pd
import pretty_midi
from algorithms.Config import Config
from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
from benchmarks.modules.note.NoteBenchmarker import CocoNoteBenchmarker


def select_coco_sources(*, root=None, split="test", per_instrument=1, seed=0):
    """Use the existing note/pitch benchmark sampler, retaining stem provenance."""
    coco = CocoChorales(
        root=root, split=split, per_instrument=per_instrument, seed=seed
    )
    records = coco.select_records()
    if not records:
        raise ValueError(f"No CocoChorales stems selected under {coco.root}")
    locator = CocoNoteBenchmarker(root=coco.root)
    missing = [record for record in records if locator.local_midi_path(record) is None]
    if missing:
        coco.materialize_records(missing)
    rows = []
    for record in records:
        path = locator.local_midi_path(record)
        if path is None:
            raise FileNotFoundError(
                f"Missing selected CocoChorales MIDI: {record.track_id}"
            )
        rows.append(
            dict(
                source=str(path.resolve()),
                dataset="coco",
                split=record.split,
                track_id=record.track_id,
                group=record.track,
                ensemble=record.ensemble,
                instrument=record.instrument,
                stem=record.stem,
            )
        )
    return pd.DataFrame(rows)


def source_program(midi):
    instruments = [i for i in pretty_midi.PrettyMIDI(str(midi)).instruments if i.notes]
    if len(instruments) != 1 or instruments[0].is_drum:
        raise ValueError(
            "Select one pitched CocoChorales stem, not an ensemble mix MIDI."
        )
    return int(instruments[0].program)


def prepare_case(
    midi,
    seed,
    rate,
    output,
    source_info=None,
    *,
    prepare_audio=True,
    prepare_pitches=True,
):
    """Inject into source stem MIDI and render with its original GM instrument."""
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

    midi = Path(midi).resolve()
    program = source_program(midi)
    bench = MistakeBenchmarker()
    timing_tolerance = Config().timing_tolerance
    duration_error_min = max(0.3, timing_tolerance + 0.05)
    spec = dict(
        source=str(midi),
        source_hash=hashlib.sha256(midi.read_bytes()).hexdigest(),
        source_info=source_info or {},
        program=program,
        case_version=5,
        timeline_protocol="monophonic_edits_v3",
        seed=int(seed),
        rate=float(rate),
        weights=[0.2] * 5,
        duration_error_policy="relative_salient_v1",
        duration_factor_range=[0.5, 1.5],
        duration_error_min_sec=duration_error_min,
        duration_tolerance=timing_tolerance,
        timing_std_ms=0.0,
        duration_std=0.0,
        soundfont_hash=hashlib.sha256(
            Path(bench.SOUNDFONT_PATH).read_bytes()
        ).hexdigest(),
        code={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("MistakeInjector.py"),
                Path(__file__).parent.parent / "MistakeBenchmarker.py",
            )
        },
    )
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    bench.MISTAKE_DIR = Path(output) / "cases" / key
    manifest = bench.MISTAKE_DIR / "manifest.json"
    if manifest.exists():
        item = json.loads(manifest.read_text())
    else:
        item = None
    if item is None:
        import benchmarks.modules.mistake.MistakeCache as _api_MistakeCache

        item = _api_MistakeCache.MistakeCache.reuse_case_assets(bench, spec, output)
        if item is not None:
            item.update(case_id=key, spec=spec)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(item, indent=2))
    if item is None or (
        prepare_audio
        and (
            not all(
                (
                    Path(item[k]).exists()
                    for k in (
                        ("audio", "pitch_data") if prepare_pitches else ("audio",)
                    )
                )
            )
        )
    ):
        injector = MistakeInjector(
            mistake_rate=rate,
            weights=(0.2,) * 5,
            timing_std_ms=0.0,
            duration_std=0.0,
            duration_error_min_sec=duration_error_min,
        )
        item = bench.generate_mistake_db_track(
            midi,
            (source_info or {}).get("dataset", "paired"),
            seed,
            injector=injector,
            program=program,
            prepare_audio=prepare_audio,
            prepare_pitches=prepare_pitches,
        )
        item.update(case_id=key, spec=spec)
        manifest.write_text(json.dumps(item, indent=2))
    return (bench, item)
