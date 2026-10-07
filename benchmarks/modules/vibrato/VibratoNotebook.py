"""Presentation helpers for the paired vibrato benchmark."""

from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker


class VibratoNotebook(VibratoBenchmarker):
    """Run shared benchmark suites and display their saved metrics."""

    @staticmethod
    def show_results(summary):
        """One compact row per method, with detection and parameter scores separate."""
        from IPython.display import display
        columns = ['method', 'frame_f1', 'frame_false_alarm', 'rate_f1', 'extent_f1',
                   'aggregate_f1', 'aggregate_soft_f1', 'yang_note_f1']
        table = summary[[c for c in columns if c in summary]].copy()
        table['Audio(s)/Compute(s)'] = (
            summary['audio_seconds'] / summary['compute_seconds'].where(summary['compute_seconds'] > 0)
        )
        table['method'] = table['method'].replace({'attune': 'Attune'})
        display(table.round(4))
        print('Scores are fractions (0–1). Full metrics and per-case reports are saved alongside comparison.csv.')
        print('Audio(s)/Compute(s): audio seconds per estimator CPU second (higher is faster); excludes pitch extraction, note segmentation, and score alignment.')

    @staticmethod
    def show_significance(result):
        from IPython.display import display
        display(result[['competitor','metric','clusters','difference_pp','ci_low_pp','ci_high_pp',
                        'p_value','p_holm','significant']].round(4))
