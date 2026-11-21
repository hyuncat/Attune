"""Librosa's Mel-based SuperFlux approximation, with the common note adapter."""
from benchmarks.modules.note.competitors.LibrosaFlux import LibrosaFlux


class LibrosaSuperFlux(LibrosaFlux):
    superflux = True
