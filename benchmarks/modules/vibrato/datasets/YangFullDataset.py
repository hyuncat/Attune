"""All released Yang audio, scored once per complete recording.

Detection truth covers the full recording. Rate/extent truth is available only
for manually marked half cycles. Common input pitch also supplies extent truth,
so this remains an adapted protocol rather than independent extent annotation.
"""
import csv
from pathlib import Path
import numpy as np

from app_logic.user.ds.AudioData import AudioData
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
from benchmarks.modules.vibrato.datasets.YangDataset import YangDataset
from benchmarks.modules.vibrato.datasets.CocoDataset import CocoDataset
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoExample


class YangFullDataset(YangDataset):
    @classmethod
    def discover(cls, dataset_root):
        root = Path(dataset_root)
        recordings = []
        for audio in sorted(root.rglob('*.wav')):
            if not any(part in ('Areas_only','Areas_and_parameters') for part in audio.parts):
                continue
            stem = audio.stem
            folder = (root/'Areas_only/CMMSD/CMMSD_Vibrato_Luwei'
                      if 'CMMSD' in audio.parts else audio.parent)
            area = next((p for p in (folder/f'{stem}-Annotation-new.csv',
                        folder/f'{stem}-Annotation.csv', folder/f'{stem}.txt') if p.exists()), None)
            if area is None:
                raise FileNotFoundError(f'Missing Yang annotation for {audio}')
            extrema = folder/f'{stem}-Annotation-Stat.csv'
            yin = folder/f'{stem}-Yin.csv'
            frequencies = cls._numeric_column(yin, 1) if yin.exists() else np.array([])
            frequencies = frequencies[np.isfinite(frequencies) & (frequencies > 0)]
            if len(frequencies):
                bounds = (float(frequencies.min()), float(frequencies.max()))
                source = 'released_yin_range_plus_8_semitones'
            else:
                bounds = cls._annotation_pitch_range(area) if stem != 'Huangjiangqin-1' else (180.,2000.)
                source = 'area_frequency_range_plus_8_semitones'
            bounds = (bounds[0]*2**(-8/12), bounds[1]*2**(8/12))
            instrument = audio.parent.parent.name if 'Areas_and_parameters' in audio.parts else (audio.parent.name if 'Coler2011' in audio.parts else stem.split('_')[0])
            recordings.append(cls.Recording(audio, area, extrema, instrument,
                audio.parent.name if 'Areas_and_parameters' in audio.parts else 'unknown', bounds, source))
        if not recordings:
            raise FileNotFoundError(f'No Yang audio under {root}')
        return recordings

    @staticmethod
    def areas(recording):
        path = recording.area_path
        if path.suffix == '.txt':
            rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
            return [(float(a[0].replace(',','.')),float(b[0].replace(',','.')))
                    for a,b in zip(rows,rows[1:]) if a[-1]=='vib_on' and b[-1]=='vib_off']
        with path.open(encoding='utf-8-sig') as f:
            return [(float(row[0]), float(row[0])+float(row[2])) for row in csv.reader(f) if row]

    @staticmethod
    def _build_recording(job):
        from algorithms.NoteDetector import NoteDetector
        recording = job.recording
        benchmarker = AttuneRealtime()
        fmin, fmax = recording.pitch_range_hz
        window = CocoDataset.automatic_yin_window_size(fmin, padding_semitones=0.)
        config = benchmarker.config_for(fmin, fmax, w1=window)
        take = benchmarker.recording_for(config)
        take.audio_data = AudioData(audio_filepath=str(recording.audio_path), config=config)
        cache = Path(job.cache_root)/'full_audio_pad8_v1'/recording.instrument/recording.performer/f'{recording.recording_id}.pitch.pkl.xz'
        benchmarker.load_or_detect_pitches(take, cache_path=cache, smooth=job.smooth_pitch,
                                          use_cache=not job.force, write_cache=True)
        times, pitch = YangDataset._pitch_arrays(take.pitch_data, config)
        references = [dict(start=start,end=end) for start,end in YangFullDataset.areas(recording)]
        rate = np.zeros(len(times)); width = rate.copy(); truth = np.zeros(len(times),bool)
        for ref in references:
            mask = (times >= ref['start']) & (times < ref['end'])
            truth[mask] = True; rate[mask] = np.nan; width[mask] = np.nan
        if recording.extrema_path.exists():
            params = YangDataset._examples_from_annotations(recording,times,pitch,pitch_stage='common_pyin')
            for example in params:
                target = example.score_mask
                start, end = example.metadata['target_start_time'], example.metadata['target_end_time']
                r, w = float(example.rate_hz[target][0]), float(example.width_cents[target][0])
                for ref in references:
                    if abs(ref['start']-start)<1e-6 and abs(ref['end']-end)<1e-6:
                        ref.update(rate_hz=r, extent_semitones=w/200.)
                rate[target] = r
                width[target] = w
        notes = NoteDetector(take, config=config).detect_notes(take.pitch_data.data[:take.pitch_data.frames_available()])
        note_bounds = [(float(note.start_time),float(note.end_time),float(note.midi_num[0])) for note in notes.data.values()]
        # Explicit whole-recording fallback is independent of the annotations.
        if not note_bounds:
            note_bounds=[(float(times[0]),float(times[-1]+1/config.sr*config.h1),60.)]
        center = np.full(len(times),np.nan)
        metadata = dict(family='yang_full_audio', recording=recording.recording_id,
            instrument=recording.instrument, performer=recording.performer,
            area_annotation_path=str(recording.area_path),
            extrema_annotation_path=str(recording.extrema_path) if recording.extrema_path.exists() else '',
            analysis_group=str(recording.audio_path), analysis_note_bounds=note_bounds,
            yang_references=references, parameter_annotations=bool(np.any(np.isfinite(rate) & (rate > 0))),
            parameter_truth='manual_extrema_times_plus_common_pyin_pitch',
            note_boundary_source='audio_only_production_note_detector',
            pitch_range_source=recording.pitch_range_source, pitch_fmin_hz=fmin,pitch_fmax_hz=fmax,
            yin_integration_size=window, continuous_context=True)
        return job.index,[VibratoExample(recording.recording_id,recording.instrument,'yang_full_audio',
            times,pitch,center,rate,width,truth,audio_path=str(recording.audio_path),metadata=metadata)]


class YangParameterDataset(YangFullDataset):
    """Complete audio from only the recordings with half-cycle annotations."""

    @classmethod
    def discover(cls, dataset_root):
        recordings = [recording for recording in super().discover(dataset_root)
                      if recording.extrema_path.is_file()]
        if not recordings:
            raise FileNotFoundError(f'No parameter-annotated Yang audio under {dataset_root}')
        return recordings
