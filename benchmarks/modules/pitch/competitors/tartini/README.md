# Tartini 1.2 reference backend

`../McLeod.py` now executes Philip McLeod's original C++ analysis engine instead
of the former independent NumPy MPM approximation. The benchmark row `mcleod`
means **Tartini 1.2, default MPM profile**, at 44.1 kHz.

## Reference and reproducibility

- [Official download page](https://www.cs.otago.ac.nz/graphics/Geoff/tartini/download.html)
- [Pinned source archive](https://www.cs.otago.ac.nz/graphics/Geoff/tartini/download/src/tartini1.2.tar.gz)
- Archive SHA-256: `20b96c1eec07fc7c2f8722b1e19c6f05c70927c40da1fc8779f6f595fff6790a`
- The complete, unmodified archive is included, with its GPL-2.0-or-later license.
  `build.py` extracts only named source files, verifies the archive, and applies
  individually checked substitutions. Generated source and binaries are not
  committed. Upstream copyright headers remain intact.

Requirements: a C++11 compiler and **single-precision FFTW3** (`fftw3f`). On macOS,
`brew install fftw`; on Debian/Ubuntu, install `libfftw3-dev` and a C++ compiler.
`pkg-config` is preferred; Homebrew include/library paths are also detected.
`CXX` and `TARTINI_FFTW_FLAGS` can specify other installations. Run:

```sh
at-venv/bin/python benchmarks/modules/pitch/competitors/tartini/build.py
QT_QPA_PLATFORM=offscreen at-venv/bin/python -m unittest benchmarks.modules.pitch.tests.McLeodTest
```

The library is compiled once into ignored `.build/` storage. The build key covers
archive, bridge, settings facades, build script and compiler/link flags. Native
setup is warmed outside streaming update timing. There is no approximation
fallback if the native dependency is unavailable.

## Exact reference boundary

The original `MyTransforms`, `Channel`, `AnalysisData`, `NoteData`, IIR filter,
interpolation, aggregation, note-change/backtracking and supporting math routines
are compiled from the archive. Single-precision FFTW and original C++ arithmetic
are retained. `gdata.h` supplies the original clean-install defaults;
`soundfile.h` replaces Qt/device/file transport. GUI export dialogs, filenames,
and active-channel selection are omitted; they have no DSP callers.

Two explicit upstream fixes are applied:

1. `calculateAnalysisData` accesses the previous frame's candidate at index -1
   when that frame has no candidates. The guard uses zero period drift in this
   otherwise undefined case. Valid candidate paths are unchanged.
2. `Channel`'s destructor releases its two otherwise leaked smoothing filters.

No mathematical pitch routine is translated to Python. The test suite also
builds the archive without the invalid-index guard and compares the stored and
live outputs bit-for-bit on inputs where upstream behavior is defined. This
validates the guard, not a claim to reproduce undefined memory accesses. FFTW
versions, CPU and compiler differences can still cause floating-point differences;
the wrapper is not a byte-identical build of the 2007 GUI executable.

## Settings and transport

Settings follow `global/gdata.cpp::updateQuickRefSettings`, `getAnalysisBufferSize`,
`getAnalysisStepSize`, and the `Channel` constructor:

| Setting | Pinned profile |
| --- | --- |
| Analysis mode | MPM, the release default |
| Sample rate | 44100 Hz; benchmark loader resamples input |
| Frame / hop | 2048 / 1024 samples (48 / 24 ms rounded to powers of two) |
| NSDF | Lags 1 through 1024; zero lag omitted |
| Candidate gate | Raw peak >= 0.4; interpolated height clamped to [-1, 1] |
| Relative cutoff | 0.93, with the release's note-level selection logic |
| Equal-loudness option | On: this release actually uses its 150 Hz high-pass IIR |
| Automatic noise floor | On, including upstream RMS-to-dB and score formulas |
| Harmonic analysis | On, as in the default application |
| Detailed per-sample pitch | Off, as in the default application |
| Note tracking | On, including warped NSDF aggregation and retrospective revisions |

The bridge follows `SoundFile::readChunk` / `finishRecordChunk`: zero-initialized
buffers, an initial half-window, full hops, filtering only new samples, clipping
filtered samples to [-1, 1], and advancing the transport cursor **after** analysis.
Even that cursor ordering is preserved because aggregation reads it. Native file
preprocessing discards a partial final hop; the wrapper does too. A file shorter
than one hop has no estimates. There is no synthetic end-of-note flush.

The benchmark frequency is the stored `fundamentalFreq` when the original
`Channel::isVisibleChunk` gate accepts the frame, otherwise zero. This is the
pitch-contour voicing policy, not the ungated text export (which exports MIDI
pitch and RMS). `TartiniSession.frequency(..., visible=False)` exposes stored raw
frequencies for audits. The benchmark's common score-range gate still applies
**after** native inference and is not a Tartini algorithm setting.

## Live versus post hoc

Both modes feed the same stateful engine one hop at a time. Each hop yields a
center timestamp `i * 1024 / 44100`; the first window includes 1024 leading zeros.
It becomes available one hop later: **23.22 ms acquisition look-ahead**, plus
measured compute/backlog. Each benchmark track has a fresh engine, filter,
noise-floor and note history. Warm-up cannot leak into a scored track.

- Streaming commits the newest frame's estimate at each update. Tartini also
  revises earlier frames internally; these revisions consume compute time, but
  are not used to replace already-scored live outputs with future information.
- Offline reads the stored sequence after the final complete hop, retaining all
  retrospective octave revisions. It uses the native grid, not independent
  zero-padded frames on a 10 ms grid.

An octave-ambiguity regression shows this distinction: the initial live output
is about 440 Hz, but the final stored pitch for that same frame is about 220 Hz.
Timing measures native analysis plus the headless bridge, not Qt display, device
I/O, or disk export latency.

## Thesis versus released source

Philip McLeod's *Fast, Accurate Pitch Detection Tools for Music Analysis* (2008),
§7.2.3 (pp. 113–114), explains revising a note's earlier octave estimates during
real-time processing. §10.1 (pp. 137–139) outlines a later pipeline with the
middle/outer-ear filter, modified-cepstrum warped aggregation, SNAC/WSNAC, and
optional detailed pitch. The conclusion (p. 155) explicitly says that the version
containing all thesis algorithms would be released after marking.

The linked 1.2 archive is the June 2007 release. Its default MPM uses warped
**NSDF** aggregation. Its alternative `MPM_MODIFIED_CEPSTRUM` branch selects from
frame-local cepstrum estimates and explicitly skips `recalcNotePitches`; it is
not the thesis's modified-cepstrum aggregate pipeline. These references therefore
cannot both be reproduced by one identical implementation. This backend pins
the user's linked executable source; it does **not** claim to implement the
unreleased thesis pipeline. A thesis implementation should have its own method
name and validation, rather than silently modifying this reference.

## Cache and validation

Adapter version `tartini_1_2_native_v1` changes only McLeod prediction cache and
streaming checkpoint identities. Offline and streaming summary rows lacking this
version are refreshed on the next normal suite run; peer rows remain reusable.
Historical prediction/checkpoint files are retained. The notebook's usual summary
merge replaces stale McLeod summary rows when that suite is rerun.

Tests cover NSDF versus its direct definition, silence transitions, input/EOF
policy, native timestamps, state isolation, live versus revised octaves,
unchanged upstream outputs on defined inputs, and stale summary detection.
No corpus benchmark was rerun as part of this change.

Validation on 2026-10-12: 17 focused native/reference, CREPE-policy, and notebook
cache regressions passed. Running the broader notebook test file also exposed
four process-worker fixture errors (three non-dataclass fake examples and one
fake audio file escaping its parent-process mock). Those unrelated fixtures were
left unchanged; no corpus data or saved benchmark results were regenerated.
