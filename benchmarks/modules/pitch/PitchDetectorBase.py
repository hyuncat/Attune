"""PitchDetectorBase implementation and owned benchmark helpers."""

from __future__ import annotations
import math
import os
import time
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import TypeVar

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
):
    os.environ.setdefault(_variable, "1")
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Melody = tuple[FloatArray, FloatArray]
ResultT = TypeVar("ResultT")


class PitchDetectorBase(ABC):
    """Shared detector contract and the data used to compare methods."""

    @dataclass(frozen=True)
    class AudioDegradation:
        """Deterministic additive-noise condition shared by every detector.

        Noise is calibrated against the clean signal power during reference-voiced
        frames, then mixed across the entire clip.  This preserves the old SNR
        experiment's important property: rests contain room/microphone-like noise
        while the clean f0 remains the scoring reference.
        """

        VERSION: ClassVar[str] = "white_noise_v1"
        snr_db: float
        seed: int

        def __post_init__(self) -> None:
            if math.isnan(self.snr_db) or self.snr_db == -math.inf:
                raise ValueError("snr_db must be finite or +inf for clean audio")

        @property
        def is_clean(self) -> bool:
            return self.snr_db == math.inf

        @property
        def label(self) -> str:
            return "clean" if self.is_clean else f"{self.snr_db:g}dB"

        @property
        def cache_tag(self) -> str:
            return "clean" if self.is_clean else f"{self.VERSION}_{self.snr_db:g}db"

        def apply(
            self,
            audio: npt.ArrayLike,
            sr: int,
            ref_times: npt.ArrayLike,
            ref_freqs: npt.ArrayLike,
        ) -> npt.NDArray[np.float32]:
            """Mix seeded white noise at the requested voiced-frame SNR."""
            clean = np.asarray(audio, dtype=np.float64).reshape(-1)
            if self.is_clean or clean.size == 0:
                return np.ascontiguousarray(clean, dtype=np.float32)
            times = np.asarray(ref_times, dtype=np.float64).reshape(-1)
            freqs = np.asarray(ref_freqs, dtype=np.float64).reshape(-1)
            if times.size >= 2 and times.size == freqs.size:
                step = float(np.median(np.diff(times)))
                if np.isfinite(step) and step > 0:
                    sample_times = np.arange(clean.size, dtype=np.float64) / float(sr)
                    indices = np.rint((sample_times - times[0]) / step).astype(np.int64)
                    indices = np.clip(indices, 0, freqs.size - 1)
                    voiced = freqs[indices] > 0
                else:
                    voiced = np.ones(clean.size, dtype=bool)
            else:
                voiced = np.ones(clean.size, dtype=bool)
            signal = clean[voiced] if np.any(voiced) else clean
            signal_power = float(np.mean(signal**2)) if signal.size else 0.0
            if not np.isfinite(signal_power) or signal_power <= 0:
                return np.ascontiguousarray(clean, dtype=np.float32)
            noise = np.random.default_rng(self.seed).standard_normal(clean.size)
            noise -= np.mean(noise)
            noise_power = float(np.mean(noise**2))
            if not np.isfinite(noise_power) or noise_power <= 0:
                return np.ascontiguousarray(clean, dtype=np.float32)
            scale = math.sqrt(
                signal_power / 10.0 ** (float(self.snr_db) / 10.0) / noise_power
            )
            return np.ascontiguousarray(clean + scale * noise, dtype=np.float32)

    @dataclass(frozen=True)
    class PitchExample:
        """One track's audio, reference melody, and the range detectors may search.

        Everything a detector needs travels here, cache locations included, so no
        detector ever reaches back into the dataset that built it.
        """

        track_id: str
        dataset: str
        audio_path: Path
        ref_times: FloatArray
        ref_freqs: FloatArray
        fmin: float
        fmax: float
        estimate_cache_dir: Path
        stage_cache_path: Path
        metadata: dict[str, Any] = field(default_factory=dict)
        degradation: PitchDetectorBase.AudioDegradation | None = None

        def __post_init__(self) -> None:
            if len(self.ref_times) != len(self.ref_freqs):
                raise ValueError(
                    f"{self.track_id}: reference times/freqs differ in length"
                )
            if self.fmax <= self.fmin:
                raise ValueError(f"{self.track_id}: fmax must exceed fmin")

        @property
        def safe_id(self) -> str:
            return self.track_id.replace("/", "_")

        @property
        def audio_seconds(self) -> float:
            import soundfile as sf

            try:
                info = sf.info(str(self.audio_path))
                return float(info.frames) / float(info.samplerate)
            except Exception:
                return float("nan")

        def audio(self, sr: int | None) -> tuple[npt.NDArray[np.float32], int]:
            """Load mono audio at ``sr`` (``None`` keeps the file's native rate)."""
            import librosa

            if self.degradation is None or self.degradation.is_clean:
                samples, rate = librosa.load(str(self.audio_path), sr=sr, mono=True)
                return (np.ascontiguousarray(samples, dtype=np.float32), int(rate))
            samples, native_rate = librosa.load(
                str(self.audio_path), sr=None, mono=True
            )
            samples = self.degradation.apply(
                samples, int(native_rate), self.ref_times, self.ref_freqs
            )
            rate = int(native_rate)
            if sr is not None and int(sr) != rate:
                samples = librosa.resample(samples, orig_sr=rate, target_sr=int(sr))
                rate = int(sr)
            return (np.ascontiguousarray(samples, dtype=np.float32), int(rate))

    @dataclass(frozen=True)
    class PitchEstimate:
        """A detector's melody on its own frame grid; 0 Hz = unvoiced."""

        times: FloatArray
        freqs: FloatArray
        compute_seconds: float
        from_cache: bool = False
        metadata: dict[str, Any] = field(default_factory=dict)

        @classmethod
        def build(
            cls,
            times: npt.ArrayLike,
            freqs: npt.ArrayLike,
            compute_seconds: float,
            **kwargs: Any,
        ) -> "PitchEstimate":
            """Normalize a raw (times, freqs) pair into a scoreable estimate."""
            times = np.asarray(times, dtype=np.float64).reshape(-1)
            freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
            if len(times) != len(freqs):
                raise ValueError("estimate times and freqs must share a length")
            if times.size == 0:
                times, freqs = (np.array([0.0]), np.array([0.0]))
            freqs = np.where(np.isfinite(freqs) & (freqs > 0), freqs, 0.0)
            return cls(
                cls._uniform_grid(times), freqs, float(compute_seconds), **kwargs
            )

        @staticmethod
        def _uniform_grid(times: FloatArray) -> FloatArray:
            """Snap near-uniform frame times back onto an exact ``t0 + k*step`` grid.

            A float32 round-trip through the cache jitters the tail by ~1 us, which
            is enough to fail mir_eval's uniform-timescale test and silently switch
            it to a silence-unaware interpolation fallback.
            """
            if times.size < 2:
                return times
            step = (times[-1] - times[0]) / (times.size - 1)
            return times[0] + np.arange(times.size, dtype=np.float64) * step

    "audio -> (times, freqs) with 0 Hz for unvoiced, plus its compute time.\n\n    Subclasses implement :meth:`predict`; :meth:`estimate` adds the shared\n    caching, range clamping, and timing every competitor needs identically.\n    Detectors with their own pipeline (Attune) override :meth:`estimate`.\n    "
    name: str = "base"
    description: str = ""
    install_hint: str = ""
    input_sr: int | None = 16000
    model_fmin: float | None = None
    model_fmax: float | None = None
    COMPUTE_CLOCK: ClassVar[str] = "process_cpu"
    LEGACY_COMPUTE_CLOCK: ClassVar[str] = "legacy_wall"

    class Unavailable(RuntimeError):
        """Libraries or a checkpoint are missing -- skip the detector, don't crash."""

        def __init__(self, detector: str, detail: str = "", hint: str = "") -> None:
            message = f"{detector}: dependency unavailable"
            if detail:
                message += f" ({detail})"
            if hint:
                message += f"\n    -> {hint}"
            super().__init__(message)
            self.detector = detector

    @dataclass(frozen=True)
    class Cache:
        """The per-track estimate npz a crashed run resumes from."""

        directory: Path
        method: str
        track: str

        @property
        def path(self) -> Path:
            return self.directory / f"{self.method}__{self.track}.npz"

        def exists(self) -> bool:
            if not self.path.exists():
                return False
            try:
                with np.load(self.path) as stored:
                    return (
                        "compute_clock" in stored
                        and str(stored["compute_clock"].item())
                        == PitchDetectorBase.COMPUTE_CLOCK
                    )
            except Exception:
                return False

        def read(self) -> PitchEstimate | None:
            if not self.path.exists():
                return None
            try:
                with np.load(self.path) as stored:
                    compute_clock = (
                        str(stored["compute_clock"].item())
                        if "compute_clock" in stored
                        else PitchDetectorBase.LEGACY_COMPUTE_CLOCK
                    )
                    if compute_clock != PitchDetectorBase.COMPUTE_CLOCK:
                        return None
                    return PitchDetectorBase.PitchEstimate.build(
                        stored["times"],
                        stored["freqs"],
                        float(stored["compute_time"]),
                        from_cache=True,
                        metadata={
                            "compute_clock": compute_clock,
                            **({"inference_threads": int(stored["inference_threads"])}
                               if "inference_threads" in stored else {}),
                            "wall_pitch_compute_time": float(
                                stored["wall_compute_time"]
                            ),
                        },
                    )
            except Exception:
                return None

        def write(self, estimate: PitchEstimate) -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            compute_clock = str(
                estimate.metadata.get(
                    "compute_clock", PitchDetectorBase.LEGACY_COMPUTE_CLOCK
                )
            )
            wall_compute_time = float(
                estimate.metadata.get(
                    "wall_pitch_compute_time", estimate.compute_seconds
                )
            )
            np.savez(
                self.path,
                times=np.asarray(estimate.times, dtype=np.float64),
                freqs=np.asarray(estimate.freqs, dtype=np.float32),
                compute_time=np.asarray(estimate.compute_seconds, dtype=np.float64),
                wall_compute_time=np.asarray(wall_compute_time, dtype=np.float64),
                compute_clock=np.asarray(compute_clock, dtype=np.str_),
                **({"inference_threads": int(estimate.metadata["inference_threads"])}
                   if "inference_threads" in estimate.metadata else {}),
            )

    def __init__(
        self, confidence: float | None = None, step_seconds: float = 0.01
    ) -> None:
        self.confidence = confidence
        self.step_seconds = float(step_seconds)
        self._model: Any = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    def ensure_available(self) -> None:
        """Import the deps cheaply so a missing detector is skipped before the loop."""

    @abstractmethod
    def predict(
        self, audio: npt.NDArray[np.float32], sr: int, fmin: float, fmax: float
    ) -> Melody:
        """Estimate a melody from one track's samples."""

    def cache(self, example: PitchExample) -> "PitchDetectorBase.Cache":
        return self.Cache(example.estimate_cache_dir, self.name, example.safe_id)

    def has_cache(self, dataset: Any, track: Any) -> bool:
        """Probe the cache from paths alone, without loading a reference melody."""
        return self.Cache(
            dataset.estimate_cache_dir(track), self.name, track.safe_id
        ).exists()

    def estimate(
        self,
        example: PitchExample,
        use_cache: bool = True,
        force_reanalysis: bool = False,
    ) -> PitchEstimate:
        """Estimate a melody, optionally replacing rather than reading its cache."""
        cache = self.cache(example)
        fmin, fmax = self.clamp_range(example.fmin, example.fmax)
        if (
            use_cache
            and (not force_reanalysis)
            and ((cached := cache.read()) is not None)
        ):
            return self.constrain_estimate_to_range(cached, fmin, fmax)
        audio, sr = example.audio(self.input_sr)
        (times, freqs), compute_seconds, wall_seconds = self.measure(
            lambda: self.predict(audio, sr, fmin, fmax)
        )
        raw_estimate = PitchDetectorBase.PitchEstimate.build(
            times,
            freqs,
            compute_seconds,
            metadata={
                "compute_clock": self.COMPUTE_CLOCK,
                "wall_pitch_compute_time": wall_seconds,
            },
        )
        if use_cache:
            cache.write(raw_estimate)
        return self.constrain_estimate_to_range(raw_estimate, fmin, fmax)

    @staticmethod
    @contextmanager
    def single_threaded_numerics():
        """Keep a benchmark worker's already-loaded numerical pools at one thread.

        The benchmark modules set the usual BLAS/OpenMP environment variables
        before imports, which covers spawned workers.  Notebook and test callers
        can import numpy first, however, so the runtime limit is needed as well.
        It also prevents unintended nested BLAS pools. CREPE's TensorFlow
        inference pool is budgeted separately by the offline runner; process CPU
        seconds include all of its threads, while wall seconds measure throughput.
        """
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            yield
        else:
            with threadpool_limits(limits=1):
                yield

    @classmethod
    def measure(cls, operation: Callable[[], ResultT]) -> tuple[ResultT, float, float]:
        """Return ``(result, worker CPU seconds, diagnostic wall seconds)``.

        Wall clocks include time a heavily parallel worker spends descheduled,
        which previously made Compute Time and Audio/Compute vary with worker
        count.  ``process_time`` is local to this worker and excludes that wait.
        """
        with cls.single_threaded_numerics():
            cpu_started = time.process_time()
            wall_started = time.perf_counter()
            result = operation()
            cpu_elapsed = max(time.process_time() - cpu_started, math.ulp(1.0))
            wall_elapsed = max(time.perf_counter() - wall_started, math.ulp(1.0))
        return (result, cpu_elapsed, wall_elapsed)

    def unavailable(self, exc: BaseException) -> "PitchDetectorBase.Unavailable":
        return self.Unavailable(self.name, repr(exc), self.install_hint)

    def clamp_range(self, fmin: float, fmax: float) -> tuple[float, float]:
        """Fit the track's range inside what this model can represent."""
        low = self.model_fmin if self.model_fmin is not None else fmin
        high = self.model_fmax if self.model_fmax is not None else fmax
        fmin = float(min(max(fmin, low), high))
        fmax = float(max(min(fmax, high), low))
        return (low, high) if fmax <= fmin else (fmin, fmax)

    @staticmethod
    def constrain_freqs_to_range(
        freqs: npt.ArrayLike, fmin: float, fmax: float
    ) -> FloatArray:
        """Apply the shared score/reference-range voicing gate.

        Some APIs (Praat, Basic Pitch, torchcrepe, PENN) can use the range while
        choosing a candidate. Others (reference CREPE, SPICE, RMVPE) expose only
        a decoded F0, and SwiftF0 documents its ``fmin``/``fmax`` arguments as
        output voicing gates. Applying the same final gate here gives every
        competitor the same admissible range without pretending that all model
        families can condition their internal inference on it.
        """
        values = np.asarray(freqs, dtype=np.float64).reshape(-1)
        valid = np.isfinite(values) & (values >= float(fmin)) & (values <= float(fmax))
        return np.where(valid, values, 0.0)

    @classmethod
    def constrain_estimate_to_range(
        cls, estimate: PitchEstimate, fmin: float, fmax: float
    ) -> PitchEstimate:
        """Range-gate both fresh and legacy cached competitor estimates."""
        return PitchDetectorBase.PitchEstimate.build(
            estimate.times,
            cls.constrain_freqs_to_range(estimate.freqs, fmin, fmax),
            estimate.compute_seconds,
            from_cache=estimate.from_cache,
            metadata={
                **estimate.metadata,
                "range_gate_fmin": float(fmin),
                "range_gate_fmax": float(fmax),
            },
        )

    @staticmethod
    def voiced_freqs(freqs: npt.ArrayLike, voiced: npt.ArrayLike) -> FloatArray:
        """Zero out every frame the model did not call voiced."""
        values = np.asarray(freqs, dtype=np.float64).reshape(-1)
        mask = np.asarray(voiced).reshape(-1).astype(bool)
        return np.where(mask & np.isfinite(values) & (values > 0), values, 0.0)

    @staticmethod
    def quiet_tensorflow() -> None:
        """Silence TensorFlow's SavedModel-restore flood (crepe, SPICE, Basic Pitch).

        That noise arrives through ``logging``/``absl``, which the warnings
        filters in ``benchmarks/__init__`` cannot reach; ``TF_CPP_MIN_LOG_LEVEL``
        covers only the C++ side, and only if set before TensorFlow imports.
        """
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            import logging
            import tensorflow as tf

            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            tf.get_logger().setLevel("ERROR")
            tf.autograph.set_verbosity(0)
        except Exception:
            pass
        try:
            from absl import logging as absl_logging

            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass
