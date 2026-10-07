"""Read-only presentation helpers for the vibrato experiment notebook."""
from pathlib import Path
from IPython.display import display
from benchmarks.modules.vibrato.tests.FrameGates import DEFAULT_GATES

def show_frame_results(output):
    import pandas as pd

    # Fixed method order: no automatic promotion of the best tuning-set row.
    FRAME_GATE_OUTPUT = Path(output)
    frame_gate_summary = pd.read_csv(FRAME_GATE_OUTPUT/'comparison.csv')
    _gate_order = ['attune_score_ungated', *[g.name for g in DEFAULT_GATES], 'yang_dt', 'yang_br']
    _gate_columns = ['frame_precision', 'frame_recall', 'frame_f1', 'yang_note_f1',
                     'aggregate_soft_f1', 'extent_soft_f1', 'rate_soft_f1',
                     'yang_split_rate', 'yang_merge_rate', 'yang_spurious_rate']
    _gate_table = frame_gate_summary.set_index('method').reindex(_gate_order)
    display(_gate_table[_gate_columns])
    # Deltas are percentage points against the SAME score-assisted ungated baseline.
    _gate_metrics = ['frame_f1', 'yang_note_f1', 'aggregate_soft_f1']
    display(100 * (_gate_table[_gate_metrics] - _gate_table.loc['attune_score_ungated', _gate_metrics]))
    # A pooled gain must not conceal recording-specific regressions.
    _gate_cases = pd.read_csv(FRAME_GATE_OUTPUT / 'per_recording.csv')
    display(_gate_cases.pivot(index='case_id', columns='method', values='frame_f1').reindex(columns=_gate_order))
    # Raw report throughput excludes the shared fit for Attune; do not compare it with Yang.
    print('Shared fitting costs:', FRAME_GATE_OUTPUT / 'fit_timing.csv')


def fragmentation_table(output):
    import json
    import pickle
    import numpy as np
    import pandas as pd

    DIAGNOSTIC_RUN = Path(output)
    _diag_notes = json.loads((DIAGNOSTIC_RUN / 'note_bounds.json').read_text())
    _diag_frames = pd.read_csv(DIAGNOSTIC_RUN / 'raw_outputs/attune_score_ungated/frames.csv')
    _diag_rows = []
    _diag_bool = lambda s: s.astype(str).str.lower().isin(['true', '1']).to_numpy()
    for _case, _notes in _diag_notes.items():
        _d = _diag_frames[_diag_frames.case_id == _case]
        _t = _d.time.to_numpy()
        _truth, _base = _diag_bool(_d.truth_vibrato), _diag_bool(_d.method_detected)
        _duration = np.zeros(len(_t))
        for _a, _b, _ in _notes:
            _duration[(_t >= _a) & (_t < _b)] = _b - _a
        # Locally generated, trusted fit artifact; no model execution.
        with (DIAGNOSTIC_RUN / 'fits' / f'{_case}.pkl').open('rb') as _f:
            _fit = pickle.load(_f)['estimate']
        _rate, _width = _fit.rate_hz, _fit.width_cents
        assert len(_rate) == len(_d)
        _voiced = np.isfinite(_d.smoothed_pitch_midi.to_numpy())
        _rate_ok = np.isfinite(_rate) & (_rate >= 4) & (_rate <= 9)
        _width_ok = np.isfinite(_width) & (_width >= 20)
        _reasons = {
            'original_fit_rejected_pct_truth': _truth & ~_base,
            'unvoiced_pct_truth': _truth & _base & ~_voiced,
            'rate_rejected_pct_truth': _truth & _base & _voiced & ~_rate_ok,
            'width_rejected_pct_truth': _truth & _base & _voiced & _rate_ok & ~_width_ok,
        }
        _row = dict(case_id=_case, detected_notes=len(_notes),
            median_note_ms=1000*np.median([b-a for a,b,_ in _notes]),
            truth_in_notes_under_150ms_pct=100*np.mean((_duration[_truth] > 0) & (_duration[_truth] < .15)))
        _row.update({key: 100*value.sum()/_truth.sum() for key,value in _reasons.items()})
        _diag_rows.append(_row)
    return pd.DataFrame(_diag_rows).round(2)


def plot_fragmentation(output, case="Huangjiangqin-1", window=(48.5, 52.0)):
    import json
    import pandas as pd
    import numpy as np
    DIAGNOSTIC_RUN = Path(output)
    _diag_notes = json.loads((DIAGNOSTIC_RUN/'note_bounds.json').read_text())
    _diag_frames = pd.read_csv(DIAGNOSTIC_RUN/'raw_outputs/attune_score_ungated/frames.csv')
    _diag_bool = lambda s: s.astype(str).str.lower().isin(['true', '1']).to_numpy()
    import matplotlib.pyplot as plt

    DIAGNOSTIC_CASE = case
    DIAGNOSTIC_WINDOW = window
    _d = _diag_frames[_diag_frames.case_id == DIAGNOSTIC_CASE]
    _t = _d.time.to_numpy()
    _selected = (_t >= DIAGNOSTIC_WINDOW[0]) & (_t <= DIAGNOSTIC_WINDOW[1])
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True, layout='constrained')
    axes[0].plot(_t[_selected], _d.smoothed_pitch_midi.to_numpy()[_selected], lw=1)
    for _a, _b, _ in _diag_notes[DIAGNOSTIC_CASE]:
        if _a < DIAGNOSTIC_WINDOW[1] and _b > DIAGNOSTIC_WINDOW[0]:
            axes[0].axvline(_a, color='tab:red', alpha=.45, lw=.7)
    axes[0].set(ylabel='MIDI pitch', title=f'{DIAGNOSTIC_CASE}: red lines = production note starts')
    axes[1].plot(_t[_selected], _d.estimated_rate_hz.to_numpy()[_selected])
    axes[1].axhspan(4, 9, color='tab:green', alpha=.12)
    axes[1].set(ylabel='Rate (Hz)')
    axes[2].plot(_t[_selected], _d.estimated_width_cents.to_numpy()[_selected])
    axes[2].axhline(20, color='grey', ls='--')
    axes[2].set(ylabel='Width (cents)')
    _masks = [('truth', _diag_bool(_d.truth_vibrato))]
    for _method in ['attune_score_ungated', 'attune_score_local_yang', 'yang_dt']:
        _saved = pd.read_csv(DIAGNOSTIC_RUN / 'raw_outputs' / _method / 'frames.csv')
        _saved = _saved[_saved.case_id == DIAGNOSTIC_CASE]
        np.testing.assert_allclose(_saved.time.to_numpy(), _t)
        _masks.append((_method, _diag_bool(_saved.method_detected)))
    for _i, (_label, _mask) in enumerate(_masks):
        axes[3].fill_between(_t[_selected], _i, _i+.75, where=_mask[_selected], step='mid', alpha=.75)
    axes[3].set(yticks=np.arange(len(_masks))+.375, yticklabels=[label for label,_ in _masks], xlabel='Recording time (s)')
    axes[3].set_xlim(*DIAGNOSTIC_WINDOW)
    plt.show()

