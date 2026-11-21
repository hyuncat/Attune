# Attune Web

A web deployment of Attune, scoped to **upload-only** for now (no live
in-browser recording, no real-time Practice-mode gating — see project
history for why). This reuses the existing Python analysis pipeline
(`algorithms/`, `app_logic/`) unchanged via a thin FastAPI wrapper, and a
Svelte frontend that mirrors the desktop app's actual layout (toolbar, left
sidebar, center score + pitch overlay, right mistake table, transport bar).

**Current status**: functionally complete for the upload → analyze →
review workflow. Upload a score + recording, get real notation rendering,
mistake detection with adjustable pitch/timing tolerance, a live per-frame
pitch overlay, MIDI playback of the score with instrument/metronome muting,
and Range/Tuning controls that default to the score's own pitch span.
Mistakes are reachable from all three of the desktop app's interaction
surfaces: the mistake table (click-to-highlight, mirrors GuitarHero's
highlight-and-pan behavior), colored inline annotations directly on the
rendered score (click a note/insertion marker for a popup - see
`annotations.js`, ported from `ui/score/ScoreAnnotations.py`), and clicking a
user note in the pitch overlay itself (a GuitarHero-style popup with
pitch/cents/onset/duration/volume - see `NoteOverlay.svelte`, ported from
`ui/guitarhero/GuitarHero.py`'s `select_note`). Pitch, timing, and volume are
all wired up: a "Colors:" dropdown next to the score switches its annotation
color mode (mirrors `perform.py`'s `score_color_mode`), and GuitarHero's own
pitch/volume dot-coloring toggle is ported too (`colors.js` has the viridis
ramp + take-relative volume math, ported from `ui/Colors.py` +
`PitchData.mean_volume`/`volume_range_db`).

The right column also has a per-note inspector (`NotePanel.svelte`, ported
from `ui/note/NotePanel.py`) below the mistake table, continuously tracking
whichever note the playback cursor is currently inside and charting it - both
tabs are fully wired: Volume (dBFS curve + pitch contour, `VolumeChart.svelte`,
ported from `ui/note/VolumeWidget.py`) and Vibrato (Width/Speed metric
toggle, viridis-colored scatter+line, `VibratoChart.svelte`, ported from
`ui/note/VibratoWidget.py`). Vibrato itself is computed in Python
unconditionally as a side effect of note detection (`Recording.detect_notes()`
already called `recompute_vibrato()` before this - the web port was only
ever missing the serialization step) and shipped as a `"vibrato"` key
`/analyze` adds to its response after `to_cache_payload()` returns
(`JsonHandler._vibrato_to_payload`) - deliberately NOT part of the shared
desktop cache format, since vibrato is never persisted there either (cheap
to recompute on load). All the windowing/smoothing (the 3-point median,
per-note vs. take-wide ranges) happens client-side in `noteCurve.js`, off
the raw grid, the same architecture as the Volume/pitch curves. The
GuitarHero-style click popup's Vibrato row (`NoteOverlay.svelte`) is filled
in from the same data too.

The score cursor is barline-anchored, not approximate: `scoreTimeMap.js`
ports `ui/time/ScoreTimeMap.py` faithfully (piecewise-linear interpolation
between the app's MIDI timeline and Verovio's own, which drifts from it over
a piece), anchored from `/notedata`'s `measure_onsets_og`/`bpm_og`/
`transpose_offset` fields paired with `ScoreViewer.svelte`'s
`getMeasureTimemap()`. Owned as a plain instance in `App.svelte` (not a
shared singleton, not inside `ScoreViewer.svelte`) - mirrors `perform.py`
owning `_time_map` rather than `ScoreViewer.py`, keeping the viewer a dumb
Verovio wrapper with no score-policy knowledge. This fixed a real, previously
-shipped bug in the process, not just the approximate-click gap this was
originally scoped to close: the ongoing playback-cursor push
(`scoreViewer.setPlaybackTime`) was sending raw app time completely
unconverted, silently drifting from what's actually sounding on any
recording where Verovio's timeline had diverged from the MIDI one -
desktop always converts this (`perform.py:504`), the web port never had.
Also fixed along the way: `ui/score/verovio/viewer.js`'s `window.loadScore`
now queues a request and flushes it once Verovio's WASM finishes
initializing, instead of silently dropping it - a same-origin iframe host
can call `loadScore` moments after page load, well before the async WASM
init completes, and the old behavior meant the score could silently never
load for the rest of the session with no retry.

Not yet done: deployment (still runs locally only), and a few toolbar stubs
(Settings, Save, Clip) that are either dead ends in the desktop app itself
(Settings) or real features not yet built (Save, Clip) — see inline comments
in `Toolbar.svelte` for the current state of each.

## Running it

**One command, one terminal:**

```bash
cd web
npm run dev:all
```

This starts both the backend (`uvicorn`, port 8000) and frontend (`vite`,
usually port 5173 but falls back automatically if that's taken) as a single
process group, with output prefixed `[backend]`/`[frontend]` so the two
stay visually distinct. `Ctrl+C` stops both cleanly.

Open the URL the `[frontend]` lines print, upload a score
(`resources/demo/scales/major/major.mxl` is a good one to start with) and a
recording (`resources/demo/scales/major/ashwin.wav`), and click Analyze.

First-time setup (skip if you've already done this once):

```bash
cd web
npm install
npm run sync-verovio   # copies resources/verovio -> web-resources/
npm run sync-icons     # copies resources/icons -> web-resources/
npm run sync-synth     # copies the MuseScore soundfont + js-synthesizer's WASM runtime -> web-resources/
python3.14 -m venv ../venv           # if you don't already have one at the project root
source ../venv/bin/activate
pip install -r api/requirements.txt
```

**If you'd rather run the two processes separately** (e.g. to restart just
the backend without touching the frontend, or to watch their logs in
separate windows), that still works exactly as before:

```bash
# Terminal 1 — from the project root
source venv/bin/activate
uvicorn web.api.analyze_api:app --app-dir . --port 8000

# Terminal 2
cd web
npm run dev
```

## Prerequisites

- **Python 3.x** — verified working with 3.14. Avoid very new Python
  releases without a moment's checking first: `praat-parselmouth` (a
  dependency of the *desktop* app, deliberately excluded from
  `web/api/requirements.txt` since nothing `/analyze` touches imports it)
  fails to build from source on 3.14, which is exactly the kind of thing to
  watch for on a bleeding-edge interpreter.
- **Node.js** — verified working with v22.19.0 / npm 10.9.3.
- **ffmpeg** on your `PATH` — required by `pydub` to decode `.m4a` uploads
  (phone voice-memo recordings). Without it, `.m4a` uploads will fail even
  though `pip install` succeeds; `.wav`/`.flac`/`.ogg`/`.aif`/`.aiff` don't
  need it (handled by `soundfile` directly).

## Notes on setup

Run the backend from the **Attune project root** (not from `web/api/` —
the app needs to import `algorithms/` and `app_logic/`, which live at the
root); `npm run backend` (and therefore `npm run dev:all`) already handles
this by `cd`-ing up a level before invoking `uvicorn`.

Note on the `python3.14 -m venv` pin: `python -m venv` accepts multiple
directory arguments and creates a venv in each one, so if you add anything
after the target directory on that line — including a trailing `#`
comment, if whatever runs the command doesn't treat `#` as a shell comment
— it'll silently create extra venv-shaped folders named after each word.
Keep that line exactly as written, with nothing appended.

`python3.14` specifically, not a generic `python3` — that's the only
version this has actually been installed and run against. The pinned
versions in `requirements.txt` (scipy, PyQt6, etc.) haven't been verified
against older 3.x interpreters, and a plain `python3` could silently
resolve to something else on a machine with multiple Python versions
installed. If you only have an older 3.x available, it may well work, but
treat it as unverified rather than assumed-fine.

Verify the backend's up on its own:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

One thing worth knowing if you look at `requirements.txt`: it includes
`PyQt6`. That's not a mistake — `algorithms/PitchDetector.py` and
`algorithms/NoteDetector.py` subclass `QObject` for Qt signals the desktop
app uses, so importing them pulls in PyQt6 even though this API never
touches any GUI code. It imports and runs fine headless, no display
needed. It is real dead weight for a deployed container, though (see task
#6 in project notes) — worth trimming when this gets containerized.

CORS is set to allow any `localhost`/`127.0.0.1` port (not a fixed one) —
deliberately, since Vite's dev port isn't stable on every machine. Dev-only;
tighten this to the real deployed frontend origin before shipping.

**`npm run sync-verovio` / `sync-icons` / `sync-synth` are not optional on
a fresh checkout.** They copy assets from the project root
(`resources/verovio`, `resources/icons`, `resources/MuseScore_General.sf3`
+ the js-synthesizer WASM runtime) into `web/web-resources/`, which Vite
serves as static files. `web/web-resources/` is gitignored (generated, not
committed), so skipping any of these means the corresponding feature 404s
at runtime — the score viewer, the toolbar/mistake-table icons, or MIDI
playback, respectively. None of the three run automatically on `npm run
dev` — re-run them manually if the underlying files in `resources/` ever
change.

## Layout

```
web/
├── api/                      # FastAPI backend
│   ├── analyze_api.py        # POST /analyze, /notedata, /realign
│   └── requirements.txt
├── scripts/
│   ├── sync-verovio.mjs      # ui/score/verovio -> web-resources/verovio
│   ├── sync-icons.mjs        # resources/icons -> web-resources/icons
│   └── sync-synth.mjs        # soundfont + js-synthesizer WASM -> web-resources/synth
├── src/
│   ├── App.svelte            # top-level layout: toolbar/sidebar/center/results/transport
│   ├── Toolbar.svelte        # Upload/Settings/Save/Clip/Playback/Tempo/Metronome
│   ├── RecordingTree.svelte  # left sidebar: score + recording tree
│   ├── SettingsPanel.svelte  # left sidebar: Instrument/Range/Tuning/Transpose
│   ├── ScoreViewer.svelte    # Verovio notation iframe
│   ├── NoteOverlay.svelte    # GuitarHero-equivalent pitch overlay
│   ├── ResultsView.svelte    # mistake table + tolerance controls
│   ├── NotePanel.svelte       # per-note inspector (Volume/Vibrato), below ResultsView
│   ├── VolumeChart.svelte     # NotePanel's Volume tab chart
│   ├── VibratoChart.svelte    # NotePanel's Vibrato tab chart
│   ├── TransportBar.svelte   # play/pause/seek/Analyze
│   ├── StatusBar.svelte
│   ├── sessionState.svelte.js  # shared reactive app state (upload, analysis, mistakes)
│   ├── playback.svelte.js      # MIDI playback engine (js-synthesizer + SMF)
│   ├── smf.js                  # builds a Standard MIDI File from note_data
│   ├── mistakes.js             # client-side mistake classification + note-name<->MIDI/Hz
│   ├── annotations.js          # client-side port of ScoreAnnotations.py - builds the
│   │                            # score-note-indexed mistake payload ScoreViewer's iframe renders
│   ├── colors.js                # viridis ramp + volume math, ported from ui/Colors.py + PitchData
│   ├── noteCurve.js             # "note under the cursor" window + curve extraction for NotePanel
│   ├── scoreTimeMap.js          # barline-anchored app<->Verovio time conversion, ported from ScoreTimeMap.py
│   ├── noteDataCache.js        # content-hash-keyed cache for /notedata responses
│   ├── realign.js              # debounced /realign calls for pitch-tolerance changes
│   ├── theme.css               # dark theme, extracted from qdarktheme's real stylesheet
│   └── main.js
├── web-resources/            # generated by the sync-* scripts, gitignored
├── JSPitchDetector.js        # JS port of the real-time pitch detector - NOT
│                              # used yet (deferred; only needed if/when live
│                              # Practice-mode recording comes back)
├── index.html
├── package.json
└── vite.config.js
```

## Landing page and `/demo/` deployment

The Vite build now has two HTML entry points: `index.html` (landing page) and
`demo/index.html` (the existing Svelte app). Both development and production
serve the landing page at `/` and the app at `/demo/`. Shared assets remain at
`/assets/`, `/icons/`, `/verovio/`, and `/synth/`; deploy the entire `web/dist`
directory at the domain root. Do not set Vite's base to `/demo/` for this
multi-page build.

Run `npm run build:deploy`, then `npm run preview` to check the production
output. `npm run dev` also serves both pages. Both Mac download links point to the v0.1.0 GitHub Release ZIP. To override the destination, set `VITE_MAC_DOWNLOAD_URL` before building.

### Existing AWS Amplify deployment

The repository-root `amplify.yml` installs dependencies in `web`, runs all asset
sync scripts and the production build, then publishes `web/dist`. It assumes
Amplify's build starts at the repository root (a standard single-app setup).
If the existing Amplify app uses a monorepo app root of `web`, keep that setup's
working directory and use `dist` as its artifact directory instead.

Keep the existing Amplify `VITE_API_URL` environment variable. It is embedded
at build time, so changes require a frontend rebuild. When unset, development
uses `http://localhost:8000`; production uses the ECS endpoint recovered from
the currently deployed public frontend:

`https://at-c19b0e98085b4847b318cac88b97b75d.ecs.us-east-1.on.aws`

On 2026-11-21, this endpoint's `/health` returned HTTP 503. Restore the ECS
service before expecting score upload and analysis to work. This is separate
from the frontend route changes.

In Amplify Hosting's **Rewrites and redirects**, remove any unconditional SPA
rewrite that sends all paths to `/index.html`. This is now a multi-page site.
Place these rules before any fallback:

| Source | Target | Status |
| --- | --- | --- |
| `/demo` | `/demo/` | 301 |
| `/demo/` | `/demo/index.html` | 200 |

Allow real static files to be served directly. There are no client-side nested
demo routes requiring a catch-all rewrite. Check `/`, `/demo`, `/demo/`, and
`/verovio/viewer.html` after deployment, including refreshing `/demo/` directly.

Attach `myattune.org` in Amplify's custom-domain settings and redirect `www` to
the apex domain. DNS and certificate validation must complete before the custom
domain works. The Python API's `ATTUNE_ALLOWED_ORIGINS` environment variable is a
comma-separated list; its default includes the current Amplify origin plus
`https://myattune.org` and `https://www.myattune.org`. Deploy the backend change
before testing uploads from the custom domain. CORS origins never include
`/demo` because URL paths are not part of an origin.

### Bundled example and landing screenshot

The demo automatically opens the C major scale with `flute_scale_bad.m4a`
and precomputed analysis. These static assets are committed in `web/sample`;
loading, inspecting, and playing this example does not require the Python API.
Uploading files, rerunning Analyze, and server realignment still use the API.
**Reset demo** restores the bundled files and analysis. An upload made while
the sample is downloading takes priority over the automatic load.

`npm run dev`, `npm run build`, and `npm run build:deploy` copy the sample and
landing screenshot into Vite's public directory via `sync-site`. To regenerate
the example after pipeline changes, run from the repository root:

```bash
at-venv/bin/python web/scripts/export-sample.py
```

This reads the existing flute recording and cached pitch track, runs note
analysis and alignment, and exports the fitted score timeline with the results.
It does not alter the original audio or sidecar. The export strips local paths
and cache metadata. Website builds do not require Python.

`web/marketing/attune-desktop.png` is the user-supplied desktop app screenshot
used on the landing page.

### Plot controls

The pitch plot and note-inspection charts resize with their panels. The pitch
plot includes note labels and time gridlines: scroll over the plot to zoom both axes, over the left pitch labels to zoom
pitch only, or over the bottom time labels to zoom time only. Drag to pan. Double-click or use Reset view to restore the
full take. Clicking the plot seeks; clicking a note also opens its details.
The transport slider supports dragging and keyboard arrows; playback pauses
while scrubbing and resumes on release if it was playing beforehand.
Instrument menus use the score's MIDI program names (the bundled web example
uses Flute, program 73, and the title C Major Scale); the metronome is excluded from
the analysis instrument selector.

Volume and Vibrato support the same axis-aware wheel zoom and drag-to-pan
controls as the pitch plot. Double-click a chart to reset; selecting a new note
or changing the vibrato metric also restores the default view.
