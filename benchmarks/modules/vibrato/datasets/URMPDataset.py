"""Preliminary URMP vibrato proxy references, NOT released vibrato annotations.

Li et al. (ISMIR 2017), §2.2/4.1 motivates the 19-piece selection and
ACF peak-spacing rate. Thresholds, eligibility and extent below are our explicit
adaptation: the authors' manual vibrato decisions are not in the local release.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks

from benchmarks.modules.pitch.datasets.URMP import URMP
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoExample

PIECE_IDS = (1, 2, 8, 9, 11, 12, 13, 19, 20, 24, 25, 26, 27, 32, 35, 36, 38, 39, 44)
STRINGS = frozenset(('vn', 'va', 'vc', 'db'))
PROTOCOL = 'urmp19_acf_proxy_v1'
SOURCE = 'https://bochenli.github.io/publication/li2017video.pdf'


@dataclass(frozen=True)
class ReferenceConfig:
    minimum_note_seconds: float = 0.75
    edge_trim_seconds: float = 0.05
    minimum_voiced_fraction: float = 0.95
    maximum_missing_seconds: float = 0.03
    minimum_rate_hz: float = 3.0
    maximum_rate_hz: float = 9.0
    acf_peak_prominence: float = 0.1
    minimum_periodicity: float = 0.3
    minimum_width_cents: float = 10.0


def discover(root):
    """Require the complete paper-defined selection; evaluate string parts only."""
    tracks = URMP(root=root).tracks()
    groups = {}
    for track in tracks:
        groups.setdefault(track.metadata['piece_number'], []).append(track)
    selected = {piece: parts for piece, parts in groups.items()
                if sum(p.metadata['instrument_code'] not in STRINGS for p in parts) <= 1}
    if set(selected) != set(PIECE_IDS):
        raise ValueError(f'Expected all 19 URMP pieces {PIECE_IDS}; found {sorted(selected)}')
    for piece, parts in selected.items():
        expected = parts[0].audio_path.parent.name.split('_')[2:]
        actual = [p.metadata['instrument_code'] for p in parts]
        if sorted(expected) != sorted(actual):
            raise ValueError(f'Incomplete audio/F0 parts for URMP piece {piece}')
    return [p for piece in PIECE_IDS for p in selected[piece]
            if p.metadata['instrument_code'] in STRINGS]


def _peak_positions(values, indices):
    positions = []
    for i in indices:
        denom = values[i-1] - 2*values[i] + values[i+1]
        shift = 0.5*(values[i-1]-values[i+1])/denom if abs(denom) > 1e-12 else 0.
        positions.append(i + np.clip(shift, -0.5, 0.5))
    return np.asarray(positions)


def reference_parameters(times, midi, config):
    """Return eligible-note proxy parameters; None excludes unreliable F0 spans."""
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.
    voiced = np.isfinite(midi)
    if len(times) < 3 or voiced.mean() < config.minimum_voiced_fraction:
        return None
    missing = np.flatnonzero(~voiced)
    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1)+1)
    if max((len(run) for run in runs), default=0)*dt > config.maximum_missing_seconds + 1e-9:
        return None
    x = np.interp(times, times[voiced], midi[voiced])
    x -= x.mean()  # Paper removes DC; deliberately do not detrend slides away.
    energy = float(x @ x)
    rate = width = periodicity = 0.
    if energy > 1e-12:
        acf = correlate(x, x, mode='full', method='fft')[len(x)-1:] / energy
        peaks, _ = find_peaks(acf, prominence=config.acf_peak_prominence)
        # Include zero lag, but do not search only 3–9 Hz: out-of-range periodic
        # modulation must remain a proxy negative rather than force a match.
        peaks = peaks[(peaks > 0) & (peaks < len(x)//2)]
        if len(peaks) >= 2:
            positions = np.r_[0., _peak_positions(acf, peaks)]
            rate = float(1./(np.median(np.diff(positions))*dt))
            periodicity = float(acf[peaks[0]])
            # Local adaptation: median adjacent peak/trough difference in cents.
            distance = max(1, int(0.5/(config.maximum_rate_hz*dt)))
            tops, _ = find_peaks(x, distance=distance, prominence=0.02)
            bottoms, _ = find_peaks(-x, distance=distance, prominence=0.02)
            extrema = sorted([(i, 1) for i in tops] + [(i, -1) for i in bottoms])
            deltas = [abs(x[j]-x[i])*100 for (i,a),(j,b) in zip(extrema,extrema[1:]) if a != b]
            width = float(np.median(deltas)) if deltas else 0.
    positive = (config.minimum_rate_hz <= rate <= config.maximum_rate_hz
                and periodicity >= config.minimum_periodicity
                and width >= config.minimum_width_cents)
    return dict(is_vibrato=bool(positive), rate_hz=rate if positive else 0.,
                width_cents=width if positive else 0., periodicity=periodicity)


def build_track(track, cache_root, reference_config, force_pitch=False):
    from algorithms.NoteDetector import NoteDetector
    from app_logic.user.ds.AudioData import AudioData
    from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
    from benchmarks.modules.vibrato.datasets.CocoDataset import CocoDataset
    from benchmarks.modules.vibrato.datasets.YangDataset import YangDataset

    rt, hz = URMP().reference(track)
    if not np.allclose(np.diff(rt), 0.01, atol=1e-6):
        raise ValueError(f'Unexpected URMP reference grid: {track.track_id}')
    ref_pitch = np.full(len(hz), np.nan)
    ref_pitch[hz > 0] = 69 + 12*np.log2(hz[hz > 0]/440.)
    notes_path = track.audio_path.with_name(track.audio_path.name.replace('AuSep_', 'Notes_').replace('.wav','.txt'))
    notes = np.loadtxt(notes_path, ndmin=2)
    adapter = AttuneRealtime()
    # Oracle-assisted range only; F0 values never enter the estimator contour.
    fmin, fmax = float(hz[hz > 0].min()*2**(-8/12)), float(hz.max()*2**(8/12))
    config = adapter.config_for(fmin, fmax, w1=CocoDataset.automatic_yin_window_size(fmin, padding_semitones=0.))
    take = adapter.recording_for(config)
    take.audio_data = AudioData(audio_filepath=str(track.audio_path), config=config)
    adapter.load_or_detect_pitches(take, Path(cache_root)/f'{track.safe_id}.pitch.pkl.xz',
                                  smooth=True, use_cache=not force_pitch)
    times, pitch = YangDataset._pitch_arrays(take.pitch_data, config)
    detected_notes = NoteDetector(take, config=config).detect_notes(take.pitch_data.data[:take.pitch_data.frames_available()])
    bounds = [(float(n.start_time),float(n.end_time),float(n.midi_num[0])) for n in detected_notes.data.values()]
    rate = np.zeros(len(times)); width = rate.copy()
    truth = np.zeros(len(times), bool); evaluation = truth.copy()
    audit = []
    for i, (start, frequency, duration) in enumerate(notes):
        lo = float(start + reference_config.edge_trim_seconds)
        hi = float(start + duration - reference_config.edge_trim_seconds)
        row = dict(track=track.track_id, piece=track.metadata['piece_number'], note=i,
                   start=lo, end=hi, duration=float(duration), status='short_note')
        if duration >= reference_config.minimum_note_seconds:
            mask = (rt >= lo) & (rt < hi)
            params = reference_parameters(rt[mask], ref_pitch[mask], reference_config)
            row['status'] = 'unreliable_reference_f0'
            if params is not None:
                target = (times >= lo) & (times < hi)
                if not target.any():
                    raise ValueError(f'Audio grid does not cover {track.track_id} note {i}')
                evaluation[target] = True
                truth[target] = params['is_vibrato']
                rate[target] = params['rate_hz']; width[target] = params['width_cents']
                row.update(params, status='scored')
        audit.append(row)
    if not evaluation.any():
        return None, audit
    example = VibratoExample(track.safe_id, track.metadata['instrument'], PROTOCOL,
        times, pitch, np.full(len(times), np.nan), rate, width, truth,
        evaluation_mask=evaluation, audio_path=str(track.audio_path), metadata=dict(
            family=PROTOCOL, instrument=track.metadata['instrument'], recording=track.track_id,
            piece=track.metadata['piece_number'], parameter_annotations=True,
            parameter_truth='automatic_proxy_from_released_URMP_F0_not_manual_vibrato_labels',
            analysis_note_bounds=bounds, note_boundary_source='audio_only_NoteDetector',
            reference_config=asdict(reference_config), pitch_fmin_hz=fmin, pitch_fmax_hz=fmax,
            yin_integration_size=config.w1))
    return example, audit
