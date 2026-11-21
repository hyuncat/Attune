"""Official Tartini 1.2 MPM, executed through a headless native C++ bridge.

The pinned source, explicit compatibility patches, settings, and differences
from McLeod's later thesis are documented in tartini/README.md. Offline returns
revised stored pitches; streaming commits only the newest causal estimate.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import threading
import weakref

import numpy as np

from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.modules.pitch.competitors.tartini.build import build

# FFTW plan creation/destruction is not thread safe. Calls into a given engine
# are serialized too; benchmark worker processes still run independently.
_NATIVE_LOCK = threading.RLock()
_FLOAT_ARRAY = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")


@lru_cache(maxsize=None)
def _library(safety_patches=True):
    with _NATIVE_LOCK:
        lib = ctypes.CDLL(str(build(safety_patches=safety_patches)))
        for name, args, result in (
            ("tartini_create", [], ctypes.c_void_p),
            ("tartini_destroy", [ctypes.c_void_p], None),
            ("tartini_push", [ctypes.c_void_p, _FLOAT_ARRAY], ctypes.c_int),
            ("tartini_frequency", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_float),
            ("tartini_note", [ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
            ("tartini_nsdf", [_FLOAT_ARRAY, _FLOAT_ARRAY], None),
        ):
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
        return lib


def _destroy(lib, handle):
    with _NATIVE_LOCK:
        lib.tartini_destroy(handle)


class TartiniSession:
    """One fresh channel/file, including the reference's persistent filter state."""
    def __init__(self, *, safety_patches=True):
        self.lib = _library(safety_patches)
        with _NATIVE_LOCK:
            self.handle = self.lib.tartini_create()
        self._finalizer = weakref.finalize(self, _destroy, self.lib, self.handle)
        self.count = 0

    def close(self):
        self._finalizer()

    def push(self, samples):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.shape != (McLeod.STREAM_HOP,) or not np.isfinite(samples).all():
            raise ValueError("Tartini expects 1024 finite mono samples per update")
        if not self._finalizer.alive:
            raise RuntimeError("Tartini session is closed")
        with _NATIVE_LOCK:
            self.count = self.lib.tartini_push(self.handle, samples)
            return self.frequency(self.count - 1)

    def frequency(self, chunk, *, visible=True):
        if not self._finalizer.alive or not 0 <= chunk < self.count:
            raise ValueError("No such frame in this Tartini session")
        with _NATIVE_LOCK:
            return float(self.lib.tartini_frequency(self.handle, chunk, int(visible)))

    def frequencies(self, *, visible=True):
        return np.array([self.frequency(i, visible=visible) for i in range(self.count)])


class McLeod(PitchDetectorBase):
    name = "mcleod"
    description = "Tartini 1.2 native MPM (stateful, official source)"
    install_hint = "Install a C++ compiler and single-precision FFTW3 (brew install fftw / libfftw3-dev)."
    input_sr = 44100
    VERSION = "tartini_1_2_native_v1"
    FRAME_SIZE = 2048  # nearest power of two to the default 48 ms at 44.1 kHz
    STREAM_HOP = 1024  # nearest power of two to the default 24 ms
    CUTOFF = 0.93
    SMALL_CUTOFF = 0.4

    def ensure_available(self):
        try:
            _library()
        except (OSError, RuntimeError) as exc:
            raise self.unavailable(exc) from exc

    def cache(self, example):
        return self.Cache(example.estimate_cache_dir, f"{self.name}_{self.VERSION}", example.safe_id)

    def has_cache(self, dataset, track):
        return self.Cache(dataset.estimate_cache_dir(track), f"{self.name}_{self.VERSION}", track.safe_id).exists()

    def reset_streaming(self):
        previous = getattr(self, "_stream", None)
        if previous is not None:
            previous.close()
        self._stream = TartiniSession()

    def close_streaming(self):
        previous = getattr(self, "_stream", None)
        if previous is not None:
            previous.close()
        self._stream = None

    @classmethod
    def nsdf(cls, frame):
        """Reference float32 NSDF at lags 1..1024 (zero lag is omitted)."""
        samples = np.ascontiguousarray(frame, dtype=np.float32)
        if samples.shape != (cls.FRAME_SIZE,) or not np.isfinite(samples).all():
            raise ValueError("Tartini NSDF expects 2048 finite samples")
        result = np.empty(cls.FRAME_SIZE // 2, dtype=np.float32)
        with _NATIVE_LOCK:
            _library().tartini_nsdf(samples, result)
        return result

    def predict_streaming(self, audio, sr, fmin, fmax):
        """Consume one new hop; report its center relative to this input hop.

        The center is the start of the new hop, with 1024 samples look-ahead.
        The native engine may revise history; those revisions are not substituted
        into previously committed streaming benchmark outputs.
        """
        if sr != self.input_sr:
            raise ValueError("Tartini benchmark uses its fixed 44100 Hz profile")
        if getattr(self, "_stream", None) is None:
            self.reset_streaming()
        return np.array([0.0]), np.array([self._stream.push(audio)])

    def predict(self, audio, sr, fmin, fmax):
        """File preprocessing on Tartini's native grid, then read revised history.

        Match SoundFile::preProcess/readChunk: a zero-initialized window, first
        half-window read, then full hops only. No padded partial EOF frame, no
        interpolated 10 ms analysis grid, and no extra synthetic note ending.
        """
        if sr != self.input_sr:
            raise ValueError("Tartini benchmark uses its fixed 44100 Hz profile")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Tartini expects finite mono audio")
        session = TartiniSession()
        try:
            for start in range(0, len(audio) - self.STREAM_HOP + 1, self.STREAM_HOP):
                session.push(audio[start:start + self.STREAM_HOP])
            return np.arange(session.count) * self.STREAM_HOP / sr, session.frequencies()
        finally:
            session.close()
