from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pretty_midi
import soundfile as sf
from app_logic.user.ds.AudioData import AudioData
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime as AttunePitch
from benchmarks.modules.vibrato.competitors.Attune import Attune

PITCH_RAW_STAGE = PitchCache.RAW
PITCH_SMOOTHED_STAGE = PitchCache.SMOOTHED
from benchmarks.modules.vibrato.datasets.CocoDataset import CocoDataset
from benchmarks.modules.vibrato.datasets.CocoRenderer import CocoRenderer


class CocoControl:
    """Render and validate one straight-note Coco/sfizz control."""

    DEFAULT_PITCH = {
        "bassoon": 53,
        "cello": 48,
        "clarinet": 67,
        "double bass": 48,
        "flute": 72,
        "horn": 64,
        "oboe": 69,
        "saxophone": 64,
        "trombone": 60,
        "trumpet": 69,
        "tuba": 48,
        "viola": 60,
        "violin": 69,
    }

    @staticmethod
    def _audio_data(path: Path, config: Any) -> AudioData:
        samples, sample_rate = sf.read(path, always_2d=True)
        if int(sample_rate) != int(config.sr):
            raise ValueError(
                f"unexpected control render rate {sample_rate}; expected {config.sr}"
            )
        mono = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)
        audio = AudioData(config=config)
        audio.data = mono
        audio.sr = int(sample_rate)
        audio.capacity = len(mono)
        audio.end_index = len(mono)
        audio.t_origin = 0.0
        return audio

    @classmethod
    def run(
        cls,
        output_dir: Path,
        *,
        instrument: str = "violin",
        pitch: int | None = None,
        duration: float = 3.0,
        velocity: int = 96,
        sfizz_render: Path | None = None,
        soundfonts_root: Path,
    ) -> dict[str, Any]:
        if instrument not in CocoRenderer.PATCHES:
            raise ValueError(f"unknown Coco instrument: {instrument}")
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        if not 1 <= velocity <= 127:
            raise ValueError("velocity must be in [1, 127]")
        pitch = cls.DEFAULT_PITCH[instrument] if pitch is None else int(pitch)
        if not 0 <= pitch <= 127:
            raise ValueError("pitch must be in [0, 127]")
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        note_start = 0.5
        note_end = note_start + duration
        source_midi = output_dir / "zero_vibrato_source.mid"
        rendered_midi = output_dir / "zero_vibrato_render.mid"
        midi = pretty_midi.PrettyMIDI()
        midi_instrument = pretty_midi.Instrument(program=40)
        midi_instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=note_start, end=note_end
            )
        )
        midi.instruments.append(midi_instrument)
        midi.write(str(source_midi))
        renderer = CocoRenderer(
            executable=sfizz_render, soundfonts_root=soundfonts_root
        )
        renderer.validate({instrument})
        renderer.prepare_midi(source_midi, rendered_midi, instrument=instrument)
        wav_path = renderer.render(
            rendered_midi, instrument=instrument, out_dir=output_dir, force=True
        )
        pitch_pipeline = AttunePitch()
        fmin, fmax = AttunePitch.range_from_midi((pitch - 2, pitch + 2))
        config = pitch_pipeline.config_for(fmin, fmax)
        recording = pitch_pipeline.recording_for(config)
        recording.audio_data = cls._audio_data(wav_path, config)
        stages = pitch_pipeline.detect_stages(recording, smooth=True).data
        zero = np.zeros(2, dtype=np.float64)
        truth = CocoDataset.InjectedNoteTruth(
            case_id="single_note_zero_vibrato",
            scenario="none",
            note_index=0,
            pitch_midi=pitch,
            start=note_start,
            end=note_end,
            bend_times=np.asarray((note_start, note_end), dtype=np.float64),
            rate_hz=zero.copy(),
            amplitude_semitones=zero.copy(),
            offset_semitones=zero.copy(),
            parameter_sampler="explicit_zero_control",
            parameter_seed=0,
            profile_parameters=CocoDataset.VibratoProfileParameters(
                0.0, 0.0, 0.0, 0.0, 0.0
            ),
        )
        detector = Attune()
        stage_results: dict[str, dict[str, float | int | bool | None]] = {}
        failed = False
        for stage in (PITCH_RAW_STAGE, PITCH_SMOOTHED_STAGE):
            example = CocoDataset._examples_from_pitch_data(
                [truth],
                stages[stage],
                config,
                raw_pitch_data=stages[PITCH_RAW_STAGE],
                split="control",
                metadata={
                    "track": "single_note_control",
                    "stem": instrument,
                    "instrument": instrument,
                    "snr": "clean",
                    "audio_path": str(wav_path),
                    "analysis_note_bounds": [(note_start, note_end, float(pitch))],
                },
            )[0]
            estimate = detector.estimate(example)
            scored = example.score_mask
            detected_frames = int(np.sum(estimate.detected & scored))
            voiced_pitch = example.pitch_midi[scored & np.isfinite(example.pitch_midi)]
            voiced_fraction = float(len(voiced_pitch) / max(1, int(np.sum(scored))))
            stage_failed = detected_frames > 0 or voiced_fraction < 0.9
            failed = failed or stage_failed
            stage_results[stage] = {
                "detected_any_vibrato": detected_frames > 0,
                "detected_frames": detected_frames,
                "scored_frames": int(np.sum(scored)),
                "voiced_frames": int(len(voiced_pitch)),
                "voiced_fraction": voiced_fraction,
                "max_reported_rate_hz": float(np.max(estimate.rate_hz[scored])),
                "max_reported_width_cents": float(np.max(estimate.width_cents[scored])),
                "pitch_p95_minus_p05_cents": (
                    float(100.0 * np.subtract(*np.percentile(voiced_pitch, (95, 5))))
                    if len(voiced_pitch)
                    else None
                ),
            }
        report = {
            "passed": not failed,
            "instrument": instrument,
            "pitch_midi": pitch,
            "duration_seconds": duration,
            "velocity": velocity,
            "wav_path": str(wav_path),
            "renderer": renderer.manifest_for(instrument),
            "stages": stage_results,
        }
        (output_dir / "zero_vibrato_validation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report
