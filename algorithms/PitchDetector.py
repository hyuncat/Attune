"""Production ordinary-candidate pYIN detector.

The promoted live and completed-audio paths share ordinary pYIN CMNDF troughs.
The live decision is causal and applies the selected raw unvoiced-probability
and running-peak RMS gates. Completed-audio decoding retains the same evidence
for its whole-recording controller. No band-pass filter is applied.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QObject, pyqtSignal
from scipy import fft as scipy_fft
from scipy.stats import beta as beta_distribution
from scipy.stats import boltzmann
from tqdm import tqdm

from algorithms.Config import Config
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording


FloatArray = npt.NDArray[np.float64]


class PitchDetector(QObject):
    """Faithful pYIN frontend shared by live and post-hoc processing."""

    pitch_detected = pyqtSignal(float)
    status_changed = pyqtSignal(str)
    detection_finished = pyqtSignal()

    N_THRESHOLDS = 100
    BETA_PARAMETERS = (2.0, 18.0)
    BOLTZMANN_PARAMETER = 2.0
    RESOLUTION = 0.1
    NO_TROUGH_PROB = 0.01
    BATCH_FRAMES = 512

    def __init__(
        self,
        recording: Recording | None = None,
        config: Config | None = None,
        parent: QObject | None = None,
        *,
        center: bool = True,
        pad_mode: str | Callable = "constant",
    ) -> None:
        super().__init__(parent)
        if recording is None and config is None:
            raise ValueError(
                "Must provide either a recording or a config to initialize "
                "the PitchDetector."
            )
        self.recording = recording
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.load_config(config if config is not None else recording.config)

        self.pda_thread: threading.Thread | None = None
        self.offline_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self._drain_on_stop = False
        self._owns_spectrum_worker = False
        self.block = False

    # ---------------------------------------------------------------- config
    def load_config(self, config: Config) -> None:
        """Rebuild the exact pYIN period, threshold, and pitch-bin geometry."""
        self.config = config
        self.SR = int(config.sr)
        self.frame_length = int(config.w1)
        self.INTEGRATION_SIZE = self.frame_length
        self.FRAME_SIZE = self.frame_length
        self.HOP_SIZE = int(config.h1)
        self.fmin = float(config.fmin)
        self.fmax = float(config.fmax)
        # Historical diagnostics use these names; pYIN+ now deliberately has
        # no extra search guard, so they equal the configured bounds.
        self.guarded_fmin = self.fmin
        self.guarded_fmax = self.fmax
        self._check_parameters()

        self.min_period = int(np.floor(self.SR / self.fmax))
        self.max_period = min(
            int(np.ceil(self.SR / self.fmin)),
            self.frame_length - 1,
        )
        # Compatibility names used by diagnostics and the streaming harness.
        self.tau_min = self.min_period
        self.tau_max = self.max_period

        self.n_bins_per_semitone = int(np.ceil(1.0 / self.RESOLUTION))
        self.n_pitch_bins = (
            int(np.floor(
                12
                * self.n_bins_per_semitone
                * np.log2(self.fmax / self.fmin)
            ))
            + 1
        )
        self.bin_freqs = self.fmin * 2.0 ** (
            np.arange(self.n_pitch_bins, dtype=np.float64)
            / (12 * self.n_bins_per_semitone)
        )
        self.bin_midis = np.asarray(
            [config.freq_to_midi(freq) for freq in self.bin_freqs],
            dtype=np.float64,
        )

        self.thresholds = np.linspace(0.0, 1.0, self.N_THRESHOLDS + 1)
        beta_cdf = beta_distribution.cdf(
            self.thresholds,
            self.BETA_PARAMETERS[0],
            self.BETA_PARAMETERS[1],
        )
        self.beta_probs = np.diff(beta_cdf)
        self.max_volume = 0.0

    def re_init(self, config: Config | None = None) -> None:
        if config is not None:
            self.load_config(config)

    def _check_parameters(self) -> None:
        if self.fmax > self.SR / 2:
            raise ValueError(
                f"fmax={self.fmax:.3f} cannot exceed Nyquist frequency "
                f"{self.SR / 2}"
            )
        if self.fmin >= self.fmax:
            raise ValueError(
                f"fmin={self.fmin:.3f} must be less than "
                f"fmax={self.fmax:.3f}"
            )
        if self.fmin <= 0:
            raise ValueError(
                f"fmin={self.fmin:.3f} must be strictly positive"
            )
        if self.SR / self.fmin >= self.frame_length - 1:
            feasible_fmin = self.SR / (self.frame_length - 1)
            feasible_frame_length = int(np.ceil(self.SR / self.fmin) + 1)
            raise ValueError(
                f"fmin={self.fmin:.3f} is too small for "
                f"frame_length={self.frame_length} and sr={self.SR}. "
                f"Either increase to fmin={feasible_fmin:.3f} "
                f"or frame_length={feasible_frame_length}"
            )
        if self.SR / self.fmin >= self.frame_length // 2:
            optimal_fmin = self.SR / (self.frame_length / 2)
            optimal_frame_length = int(np.ceil(self.SR / self.fmin) * 2 + 1)
            warnings.warn(
                f"With fmin={self.fmin:.3f}, sr={self.SR} and "
                f"frame_length={self.frame_length}, less than two periods of "
                "fmin fit into the frame, which can cause inaccurate pitch "
                f"detection. Consider increasing to fmin={optimal_fmin:.3f} "
                f"or frame_length={optimal_frame_length}.",
                stacklevel=3,
            )

    # --------------------------------------------------------------- live API
    def run(self, start_time: float | None = None) -> None:
        """Start causal frame detection on the recording's audio queue."""
        self.stop()
        self.stop_event.clear()
        self._drain_on_stop = False
        self.max_volume = 0.0
        self.recording.a2p_queue.init_start_time(start_time)
        self.recording.spectrum_detector.start()
        self._owns_spectrum_worker = True
        self.pda_thread = threading.Thread(target=self._run, daemon=True)
        self.pda_thread.start()

    def _run(self) -> None:
        while True:
            try:
                stopping = self.stop_event.is_set()
                if stopping and not self._drain_on_stop:
                    break
                samples, frame_start = self.recording.a2p_queue.pop(
                    self.FRAME_SIZE,
                    self.HOP_SIZE,
                    stall=self.block,
                )
                if samples is None:
                    if stopping:
                        break
                    self.stop_event.wait(0.002)
                    continue

                self.recording.spectrum_detector.submit(
                    samples[:self.INTEGRATION_SIZE],
                    frame_start,
                )
                pitch = self.detect_pitch(samples, frame_start)
                pitch.time = (
                    frame_start
                    + 0.5 * self.INTEGRATION_SIZE / self.SR
                )
                self.recording.write_pitch_data([pitch], frame_start)
                # PitchData is addressed on the frame-start grid even though
                # the plotted/analysed Pitch carries the window-centre time.
                self.pitch_detected.emit(frame_start)
            except Exception as exc:  # noqa: BLE001 -- keep the live worker up
                print(f"[PitchDetector] frame skipped due to error: {exc}")

    def stop(self, drain: bool = False) -> None:
        if self.pda_thread and self.pda_thread.is_alive():
            self._drain_on_stop = drain
            self.stop_event.set()
            self.pda_thread.join()
        if self.recording is not None and self._owns_spectrum_worker:
            self.recording.spectrum_detector.stop(drain=True)
            self._owns_spectrum_worker = False
        self._drain_on_stop = False

    def detect_pitches_async(self) -> None:
        if self.offline_thread and self.offline_thread.is_alive():
            return
        self.offline_thread = threading.Thread(
            target=self._detect_pitches_offline,
            daemon=True,
        )
        self.offline_thread.start()

    def _detect_pitches_offline(self) -> None:
        try:
            self.recording.detect_pitches(on_phase=self.status_changed.emit)
        except Exception as exc:  # noqa: BLE001 -- surface worker failures
            print(f"[PitchDetector] offline detection failed: {exc}")
        finally:
            self.detection_finished.emit()

    # ----------------------------------------------------------- pYIN frontend
    @staticmethod
    def _frame_audio(
        audio: npt.ArrayLike,
        frame_length: int,
        hop_length: int,
        *,
        center: bool,
        pad_mode: str | Callable,
    ) -> np.ndarray:
        samples = np.asarray(audio)
        if samples.ndim != 1:
            raise ValueError("PitchDetector currently supports mono audio only")
        if not np.issubdtype(samples.dtype, np.floating):
            raise ValueError("audio data must be floating-point")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio data must be finite")
        if center:
            samples = np.pad(
                samples,
                (frame_length // 2, frame_length // 2),
                mode=pad_mode,
            )
        if samples.size < frame_length:
            raise ValueError(
                f"input is too short ({samples.size}) for "
                f"frame_length={frame_length}"
            )
        return np.lib.stride_tricks.sliding_window_view(
            samples,
            frame_length,
        )[::hop_length].T

    @staticmethod
    def _cumulative_mean_normalized_difference(
        frames: np.ndarray,
        min_period: int,
        max_period: int,
    ) -> np.ndarray:
        """Librosa 0.11-compatible YIN equation-8 implementation."""
        frame_length = frames.shape[-2]
        fft_size = scipy_fft.next_fast_len(2 * frame_length - 1, real=True)
        spectrum = scipy_fft.rfft(frames, n=fft_size, axis=-2)
        power_spectrum = spectrum.real**2 + spectrum.imag**2
        acf_frames = scipy_fft.irfft(
            power_spectrum,
            n=fft_size,
            axis=-2,
        )[..., :max_period + 1, :]

        yin_frames = np.square(frames)
        np.cumsum(yin_frames, out=yin_frames, axis=-2)
        periods = slice(1, max_period + 1)
        yin_frames[..., 0, :] = 0
        yin_frames[..., periods, :] = (
            2
            * (acf_frames[..., 0:1, :] - acf_frames[..., periods, :])
            - yin_frames[..., :periods.stop - 1, :]
        )
        numerator = yin_frames[..., min_period:max_period + 1, :]
        period_range = np.arange(1, max_period + 1).reshape(
            (1,) * (yin_frames.ndim - 2) + (max_period, 1)
        )
        cumulative_mean = (
            np.cumsum(yin_frames[..., periods, :], axis=-2)
            / period_range
        )
        denominator = cumulative_mean[
            ..., min_period - 1:max_period, :
        ]
        return numerator / (denominator + np.finfo(denominator.dtype).tiny)

    @staticmethod
    def _parabolic_interpolation(values: np.ndarray) -> np.ndarray:
        shifts = np.zeros_like(values)
        if values.shape[-2] <= 2:
            return shifts
        previous = values[..., :-2, :]
        current = values[..., 1:-1, :]
        following = values[..., 2:, :]
        a = following + previous - 2 * current
        b = (following - previous) / 2
        interior = shifts[..., 1:-1, :]
        np.divide(-b, a, out=interior, where=np.abs(b) < np.abs(a))
        return shifts

    @staticmethod
    def _localmin(values: np.ndarray) -> npt.NDArray[np.bool_]:
        local = np.zeros_like(values, dtype=bool)
        if values.shape[-2] == 1:
            return local
        local[..., 1:-1, :] = (
            (values[..., 1:-1, :] < values[..., :-2, :])
            & (values[..., 1:-1, :] <= values[..., 2:, :])
        )
        local[..., -1, :] = values[..., -1, :] < values[..., -2, :]
        return local

    def _probabilities_for_troughs(
        self,
        yin_frames: np.ndarray,
        parabolic_shifts: np.ndarray,
        troughs_for_frame: Callable[[int, np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build exact pYIN observations for one candidate-selection rule."""
        yin_probs = np.zeros_like(yin_frames)
        for frame_index, yin_frame in enumerate(yin_frames.T):
            trough_index = np.asarray(
                troughs_for_frame(frame_index, yin_frame),
                dtype=int,
            )
            if trough_index.size == 0:
                continue
            trough_heights = yin_frame[trough_index]
            trough_thresholds = np.less.outer(
                trough_heights,
                self.thresholds[1:],
            )
            trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
            n_troughs = np.count_nonzero(trough_thresholds, axis=0)
            trough_prior = boltzmann.pmf(
                trough_positions,
                self.BOLTZMANN_PARAMETER,
                n_troughs,
            )
            trough_prior[~trough_thresholds] = 0
            probabilities = trough_prior.dot(self.beta_probs)

            global_minimum = int(np.argmin(trough_heights))
            thresholds_below_minimum = np.count_nonzero(
                ~trough_thresholds[global_minimum, :]
            )
            probabilities[global_minimum] += (
                self.NO_TROUGH_PROB
                * np.sum(self.beta_probs[:thresholds_below_minimum])
            )
            yin_probs[trough_index, frame_index] = probabilities

        yin_period, frame_index = np.nonzero(yin_probs)
        period_candidates = self.min_period + yin_period
        period_candidates = (
            period_candidates + parabolic_shifts[yin_period, frame_index]
        )
        f0_candidates = self.SR / period_candidates
        bin_index = (
            12
            * self.n_bins_per_semitone
            * np.log2(f0_candidates / self.fmin)
        )
        bin_index = np.clip(
            np.round(bin_index),
            0,
            self.n_pitch_bins,
        ).astype(int)

        # Retain librosa's inclusive upper clip and subsequent overwrite of the
        # first unvoiced row; this edge behaviour is part of the defended port.
        observations = np.zeros(
            (2 * self.n_pitch_bins, yin_frames.shape[1]),
            dtype=np.float64,
        )
        observations[bin_index, frame_index] = yin_probs[
            yin_period,
            frame_index,
        ]
        voiced_probability = np.clip(
            np.sum(observations[:self.n_pitch_bins], axis=0),
            0,
            1,
        )
        observations[self.n_pitch_bins:, :] = (
            1.0 - voiced_probability
        ) / self.n_pitch_bins
        return observations[:self.n_pitch_bins], voiced_probability

    def _probabilities_from_frames(
        self,
        frames: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centered = frames - np.mean(frames, axis=-2, keepdims=True)
        volumes = np.sqrt(np.mean(np.square(centered), axis=-2)).reshape(-1)
        yin_frames = self._cumulative_mean_normalized_difference(
            frames,
            self.min_period,
            self.max_period,
        )
        parabolic_shifts = self._parabolic_interpolation(yin_frames)

        def ordinary(_frame_index: int, yin_frame: np.ndarray) -> np.ndarray:
            is_trough = self._localmin(yin_frame[:, np.newaxis])[:, 0]
            if yin_frame.size > 1:
                is_trough[0] = yin_frame[0] < yin_frame[1]
            return np.flatnonzero(is_trough)

        observations, voiced_probability = (
            self._probabilities_for_troughs(
                yin_frames,
                parabolic_shifts,
                ordinary,
            )
        )
        return observations, voiced_probability, volumes

    def probabilities(
        self,
        audio: npt.ArrayLike,
        *,
        center: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the one canonical ordinary-pYIN observation set."""
        frames = self._frame_audio(
            audio,
            self.frame_length,
            self.HOP_SIZE,
            center=self.center if center is None else bool(center),
            pad_mode=self.pad_mode,
        )
        observations, voiced_probability, _ = self._probabilities_from_frames(
            frames
        )
        return observations, voiced_probability

    # --------------------------------------------------------- Pitch creation
    def _candidates(
        self,
        observations: np.ndarray,
        frame_index: int,
    ) -> list[tuple[float, float]]:
        nonzero = np.flatnonzero(observations[:, frame_index])
        candidates = [
            (
                float(self.bin_midis[pitch_bin]),
                float(observations[pitch_bin, frame_index]),
            )
            for pitch_bin in nonzero
        ]
        candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        return candidates

    def _live_value(
        self,
        candidates: list[tuple[float, float]],
        voiced_probability: float,
    ) -> tuple[float, float]:
        """Exact one-frame joint-HMM pitch decision plus raw unvoiced mass."""
        unvoiced_probability = 1.0 - float(voiced_probability)
        unvoiced_bin_probability = (
            unvoiced_probability
        ) / self.n_pitch_bins
        if candidates and candidates[0][1] >= unvoiced_bin_probability:
            return float(candidates[0][0]), unvoiced_probability
        return -1.0, unvoiced_probability

    def _pitch_from_observations(
        self,
        observations: np.ndarray,
        voiced_probability: np.ndarray,
        volumes: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> Pitch:
        candidates = self._candidates(observations, frame_index)
        value, unvoiced_prob = self._live_value(
            candidates,
            float(voiced_probability[frame_index]),
        )
        volume = float(volumes[frame_index])
        self.max_volume = max(self.max_volume, volume)
        if (
            unvoiced_prob >= float(self.config.unv_thresh)
            or volume
            < self.max_volume * max(0.0, float(self.config.min_volume))
        ):
            value = -1.0

        score_note = (
            self.recording.score_data.current_note()
            if (
                self.recording is not None
                and self.recording.score_data is not None
            )
            else None
        )
        live_distance = (
            score_note.midi_num[0] - value
            if score_note is not None and value != -1
            else None
        )
        return Pitch(
            time=float(timestamp),
            candidates=candidates,
            value=value,
            volume=volume,
            unvoiced_prob=unvoiced_prob,
            live_distance=live_distance,
            config=self.config,
        )

    def detect_pitch(
        self,
        audio_frame: npt.ArrayLike,
        start_time: float | None = None,
    ) -> Pitch:
        """Detect one causal frame, retaining its evidence for later tracking."""
        samples = np.asarray(audio_frame)
        if samples.size != self.frame_length:
            raise ValueError(
                f"PitchDetector frame must contain {self.frame_length} samples; "
                f"received {samples.size}"
            )
        frames = self._frame_audio(
            samples,
            self.frame_length,
            self.HOP_SIZE,
            center=False,
            pad_mode=self.pad_mode,
        )
        observations = self._probabilities_from_frames(frames)
        return self._pitch_from_observations(
            *observations,
            frame_index=0,
            timestamp=0.0 if start_time is None else float(start_time),
        )

    def detect_pitches(
        self,
        audio: npt.ArrayLike,
        show_progress: bool = False,
        progress_desc: str = "Detecting pitches",
        verbose: bool = False,
    ) -> list[Pitch]:
        """Compute canonical ordinary-pYIN observations over completed audio.

        The librosa-compatible frame grid is processed in bounded chunks so
        long takes do not allocate a multi-gigabyte temporary at the dense
        production hop.
        """
        samples = np.asarray(audio)
        if samples.ndim != 1:
            raise ValueError("PitchDetector currently supports mono audio only")
        if not np.issubdtype(samples.dtype, np.floating):
            raise ValueError("audio data must be floating-point")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio data must be finite")
        if self.center:
            framed_samples = np.pad(
                samples,
                (self.frame_length // 2, self.frame_length // 2),
                mode=self.pad_mode,
            )
        else:
            framed_samples = samples
        if framed_samples.size < self.frame_length:
            return []

        frames = np.lib.stride_tricks.sliding_window_view(
            framed_samples,
            self.frame_length,
        )[::self.HOP_SIZE]
        frame_count = len(frames)
        self.max_volume = 0.0
        output: list[Pitch] = []
        started = time.perf_counter()
        if verbose:
            print(
                f"[PitchDetector] detecting {frame_count} frame(s)",
                flush=True,
            )

        spectrum = self.recording.spectrum_detector if self.recording else None
        raw_spectrum_frames = None
        if spectrum is not None:
            spectrum.start()
            if samples.size >= self.frame_length:
                raw_spectrum_frames = np.lib.stride_tricks.sliding_window_view(
                    samples,
                    self.frame_length,
                )[::self.HOP_SIZE]

        starts = range(0, frame_count, self.BATCH_FRAMES)
        if show_progress:
            starts = tqdm(
                starts,
                total=(frame_count + self.BATCH_FRAMES - 1) // self.BATCH_FRAMES,
                desc=progress_desc,
                leave=False,
                mininterval=0.25,
            )
        try:
            for batch_start in starts:
                batch = frames[
                    batch_start:batch_start + self.BATCH_FRAMES
                ].T
                observations = self._probabilities_from_frames(batch)
                for local_index in range(batch.shape[1]):
                    global_index = batch_start + local_index
                    timestamp = global_index * self.HOP_SIZE / self.SR
                    pitch = self._pitch_from_observations(
                        *observations,
                        frame_index=local_index,
                        timestamp=timestamp,
                    )
                    pitch.live_distance = None
                    output.append(pitch)
                    if (
                        spectrum is not None
                        and raw_spectrum_frames is not None
                        and global_index < len(raw_spectrum_frames)
                    ):
                        spectrum.submit(
                            raw_spectrum_frames[global_index],
                            self.recording.timbre_data.t_origin
                            + global_index * self.HOP_SIZE / self.SR,
                        )
        finally:
            if spectrum is not None:
                spectrum.stop(drain=True)

        if verbose:
            print(
                f"[PitchDetector] done: {len(output)} pitch frame(s) in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )
        return output

    def preprocess_audio(self, audio: npt.ArrayLike) -> tuple[np.ndarray, float]:
        """Return centered samples and centered RMS without filtering.

        pYIN itself consumes the original samples, matching the defended
        frontend. This helper remains for volume diagnostics and tests.
        """
        samples = np.asarray(audio, dtype=float)
        if samples.size == 0:
            return np.array([], dtype=float), 0.0
        centered = samples - np.mean(samples)
        return centered, float(np.sqrt(np.mean(np.square(centered))))
