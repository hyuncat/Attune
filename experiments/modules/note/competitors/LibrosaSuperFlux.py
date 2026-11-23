"""Librosa's Mel-based SuperFlux approximation, with the common note adapter."""
from experiments.modules.note.competitors.LibrosaFlux import LibrosaFlux


class LibrosaSuperFlux(LibrosaFlux):
    superflux = True
