# Attune desktop packaging

The build starts at `app.py`. It does not copy the repository into the app.
`requirements-app.txt` contains the direct runtime packages; the original
`requirements.txt` remains the development/research environment. Pip resolves
transitive dependencies, and PyInstaller follows imports plus its package hooks.
Some dependencies ship optional tools; `bundle_config.EXCLUDES` prevents unused
notebook, plotting, ML, and alternate Qt backends from entering the app.

The initial spec targets **macOS**, matching the current CoreAudio synth driver.
Build with Python 3.12 on the architecture you want to ship. A Windows release
also needs a platform-specific synth driver, native binaries, and its own spec.

## Build

From the repository root:

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install -r requirements-build.txt
.venv-build/bin/python scripts/audit_bundle.py
.venv-build/bin/python -m PyInstaller --noconfirm Attune.spec
```

Output: `dist/Attune.app`. The spec builds it directly, without a duplicate
one-folder distribution. Allow several GB of free space for the environment,
native-library processing cache, and finished app.

The **build machine** needs FluidSynth, ffmpeg, and ffprobe (for example, from
Homebrew). Users do not. The spec bundles them and their non-system shared-library
dependencies. Override discovery with absolute paths if necessary:

```sh
export ATTUNE_FLUIDSYNTH=/opt/homebrew/lib/libfluidsynth.dylib
export ATTUNE_FFMPEG=/opt/homebrew/bin/ffmpeg
export ATTUNE_FFPROBE=/opt/homebrew/bin/ffprobe
```

Use matching architectures. A full Homebrew FFmpeg build has many codec libraries;
for a smaller release, supply an audio-focused FFmpeg build and matching ffprobe.
Do not manually discard linked libraries. The M4A demos and M4A imports use pydub
and require both executables. PyInstaller's native dependency analysis relocates
their dylibs. The runtime hook puts bundled executables first on PATH and binds
pyfluidsynth to its bundled dylib, independent of Homebrew on the user's machine.

## Included repository data

`packaging/bundle_config.py` is the source of truth; `scripts/audit_bundle.py`
prints the exact file list and the static local import closure without importing
the GUI. Python modules such as `resources/program_map.py` are collected as code.

- `resources/MuseScore_General.sf3`.
- Only the 14 icons currently referenced by the app (no unused alternate PNGs).
- The five local Verovio HTML/CSS/JS/WASM files under `ui/score/verovio`.
- Existing demos, excluding top-level `fugue*` entries.
- In `scales/major`: score files, `flute_scale_good`, `flute_scale_bad`, and their
  hidden `.json.xz` / legacy `.json.gz` / `.json` analysis sidecars only.
- In `scales/minor`: score files, `major_singing`, and its hidden sidecars only.
- No `.DS_Store` or Python cache directories.

Source demos are never changed by packaging. In frozen builds, the app seeds
`~/Library/Application Support/Attune/scales` and opens it as the library root.
The first copy restores the audio timestamps used by the saved analyses.
Subsequent launches preserve edits, renames, and deletions. Older installations'
editable copies are still accessible by opening their `demos-v1` folder manually.
Source execution opens `resources/demo/scales`.

Numba remains included for the accelerated pitch smoother. Its decorated module
is also collected as source to support JIT caching, with a writable cache under
`~/Library/Caches/Attune/numba`.

## Release validation

Launch `dist/Attune.app` from Finder and from a directory outside the repository.
Load Demos: check both scores and all three recordings, existing cache results,
score rendering, and M4A playback. Run Analyze, inspect vibrato/timbre, play MIDI,
record in Perform and Practice, then save/reopen a take. Repeat on a clean Mac
without Python, Homebrew, or the source repository. Review
`build/Attune/warn-Attune.txt` and `xref-Attune.html` for missing/extra modules.
Never treat a successful build as a complete audio-device/GUI validation.

The Mac app icon is `packaging/Attune.icns`. For public distribution, set the
release version and configure appropriate Developer ID
signing and hardened-runtime entitlements, notarization, and a DMG/ZIP. Setting
`ATTUNE_CODESIGN_IDENTITY` alone is not a complete notarization workflow. This
configuration defaults to a local development build.

PyInstaller configuration references:
[spec files](https://pyinstaller.org/en/stable/spec-files.html),
[package hooks](https://pyinstaller.org/en/stable/hooks.html).
