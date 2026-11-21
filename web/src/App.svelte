<script>
  // Ports app.py's QMainWindow layout: Toolbar (top) -> horizontal splitter
  // [left: RecordingTree + SettingsWidget | center: ScoreViewer + GuitarHero
  // (Perform tab's vertical splitter) | right: MistakeWidget + ToleranceWidget]
  // -> transport row -> StatusBar. Practice tab (live pitch-matching) is out
  // of scope (upload-only, no live recording), so only Perform's layout is
  // ported - there's nothing to put in a second tab.
  import ScoreViewer from "./ScoreViewer.svelte";
  import NoteOverlay from "./NoteOverlay.svelte";
  import ResultsView from "./ResultsView.svelte";
  import NotePanel from "./NotePanel.svelte";
  import Toolbar from "./Toolbar.svelte";
  import RecordingTree from "./RecordingTree.svelte";
  import SettingsPanel from "./SettingsPanel.svelte";
  import TransportBar from "./TransportBar.svelte";
  import StatusBar from "./StatusBar.svelte";
  import { session } from "./sessionState.svelte.js";
  import { playback } from "./playback.svelte.js";
  import { ScoreTimeMap } from "./scoreTimeMap.js";
  import { onMount } from "svelte";

  let scoreViewer;
  // Rendered-system height (CSS px), pulled from the iframe after each load -
  // mirrors perform.py's _fit_score_viewer_height, which sizes the Qt
  // splitter pane to the single rendered line so nothing needs to scroll.
  // legendHeight is measured via bind:clientHeight so the fit accounts for
  // the row below the score, same as desktop adding
  // _score_legend_row.sizeHint().height().
  let scoreFitHeight = $state(null);
  let legendHeight = $state(0);

  // Owned here (not a shared singleton, not inside ScoreViewer.svelte) -
  // mirrors perform.py owning _time_map rather than ScoreViewer.py: the
  // widget stays a dumb Verovio wrapper with no score-policy knowledge
  // (tempo/transpose), and App.svelte plays the same "host tab" role
  // perform.py does. Only this file's two call sites below need it today.
  let timeMap = new ScoreTimeMap();

  function scoreInfo() {
    return {
      bpm: session.noteData?.bpm,
      bpmOg: session.noteData?.bpm_og,
      transposeOffset: session.noteData?.transpose_offset,
    };
  }

  // Mirrors perform.py's _refresh_score_mistakes: push score-note-indexed
  // mistake markers + the active color mode into the viewer. A plain
  // function (not an effect body) so the ready catch-up below can also call
  // it directly - mirrors perform.py's on_score_viewer_loaded re-pushing
  // everything once the JS API becomes ready.
  function pushAnnotations() {
    if (!scoreViewer?.ready) return;
    scoreViewer.setAnnotationColorMode(session.scoreColorMode);
    scoreViewer.setMistakeAnnotations(session.annotationsPayload);
  }

  // session.annotationsPayload is a $derived.by in sessionState.svelte.js
  // that already tracks every mutation feeding it (see its comment there for
  // why that's colocated rather than hand-wired) - this effect only needs to
  // watch that one collapsed value plus scoreColorMode, instead of the 8
  // separate fields the old effect read directly.
  $effect(() => {
    session.annotationsPayload;
    session.scoreColorMode;
    pushAnnotations();
  });

  // Mirrors perform.py's on_note_clicked/on_annotation_clicked: clicking a
  // note or mistake marker in the score seeks the transport there (ignored
  // while playing, matching desktop's scrub-only guard).
  //
  // Annotation clicks already carry app/MIDI time (from noteMeta.seekTime -
  // built off the same NoteData timeline playback.seek() uses) - no map
  // needed, matching desktop's on_annotation_clicked taking app_sec directly.
  function onAnnotationClicked(appSec) {
    if (playback.isPlaying) return;
    playback.seek(appSec);
  }

  // Plain note clicks only carry Verovio's OWN rendered-SVG time, which
  // drifts from MIDI time - convert via the barline map first (mirrors
  // perform.py's _score_note_start_from_viewer), then snap to the nearest
  // rendered score-note onset so the shared transport lands on a real
  // NoteData start time.
  function onNoteClicked(viewerSec) {
    if (playback.isPlaying) return;
    const appT = timeMap.appTime(viewerSec, scoreInfo());
    const starts = (session.scoreNotesActive ?? []).map((n) => n[1]);
    if (!starts.length) {
      playback.seek(appT);
      return;
    }
    const nearest = starts.reduce((best, t) =>
      Math.abs(t - appT) < Math.abs(best - appT) ? t : best
    );
    playback.seek(nearest);
  }

  // Drives the score cursor from the real playback clock - mirrors
  // app.py's time_changed / perform.py's move_views calling
  // ScoreViewer.set_playback_time directly on every WallClock tick. A plain
  // function, called imperatively from playback.onTick's subscription (see
  // onMount below) rather than a $effect reading playback.currentTime: this
  // needs to fire on every ~100ms tick, and reactive tracking through the
  // scoreViewer binding proved unreliable in practice for exactly that case
  // (the score never advanced past the first line during playback).
  function pushPlaybackTime() {
    if (!scoreViewer?.ready) return;
    scoreViewer.setPlaybackTime(timeMap.viewerTime(playback.currentTime, scoreInfo()));
  }

  function base64ToBytes(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  // Loads the real uploaded score into Verovio the moment /notedata returns
  // it (musicxml_b64 - see analyze_api.py), independent of Analyze ever
  // running, mirroring the desktop app's score-independent score loading. A
  // plain function, called imperatively from session.onNoteDataLoaded's
  // subscription (see onMount below) - fired once at the real point
  // noteData changes, rather than derived from watching
  // session.noteData?.musicxml_b64 reactively.
  //
  // Re-anchors the barline time map right after (mirrors perform.py's
  // refresh_score_viewer -> _rebuild_time_map) - simpler here than desktop's
  // version, though: ScoreViewer.svelte's getMeasureTimemap() is a same-
  // origin, synchronous iframe call (unlike desktop's genuinely async Qt
  // runJavaScript-with-callback), and loadScore() itself runs Verovio's
  // render synchronously start to finish, so the timemap is already valid
  // the instant loadScore() returns - no async _store callback needed.
  let lastLoadedMusicXml = null;
  function pushScoreLoad(b64) {
    if (!scoreViewer?.ready || !b64 || b64 === lastLoadedMusicXml) return;
    lastLoadedMusicXml = b64;
    // onRendered fires once the score has real geometry (ScoreViewer.svelte
    // retries internally if the first render comes back empty) - mirrors
    // ScoreViewer.py's _emit_content_height feeding perform.py's
    // _fit_score_viewer_height.
    scoreViewer.loadScore(base64ToBytes(b64), (height) => {
      scoreFitHeight = height;
      const veroOnsets = scoreViewer.getMeasureTimemap();
      const appOnsets = session.noteData?.measure_onsets_og;
      if (veroOnsets?.length && appOnsets?.length) {
        timeMap.setAnchors(appOnsets, veroOnsets);
      } else {
        timeMap.clear();
      }
      pushAnnotations();
      pushPlaybackTime();
    });
  }

  // ScoreViewer's onReady catch-up push - mirrors perform.py's
  // on_score_viewer_loaded, which re-pushes score/annotations/playback-time
  // once the JS API becomes ready. Needed because the imperative
  // subscriptions above only fire on a NEW mutation; without this, a
  // mutation that happened before the iframe finished loading (e.g. a score
  // picked while Verovio's WASM was still initializing) would have its push
  // silently no-op (pushX's `scoreViewer?.ready` guard) with nothing left to
  // re-trigger it afterward.
  function onScoreViewerReady() {
    pushScoreLoad(session.noteData?.musicxml_b64);
    pushAnnotations();
    pushPlaybackTime();
  }

  // Imperative subscriptions, registered once - see pushPlaybackTime's
  // comment for why this isn't a reactive $effect. playback.svelte.js and
  // sessionState.svelte.js call these back directly at their real point of
  // mutation (a clock tick/seek, or a freshly-loaded score).
  onMount(() => {
    const offTick = playback.onTick(pushPlaybackTime);
    const offNoteData = session.onNoteDataLoaded((noteData) => pushScoreLoad(noteData?.musicxml_b64));
    session.loadSample();
    return () => {
      offTick();
      offNoteData();
    };
  });
</script>

<div class="app-shell">
  {#if session.sampleError}<div role="alert">{session.sampleError} Use Upload or Reset demo to try again.</div>{/if}
  <div class="demo-header"><a href="/">Attune</a><span>Web demo · Upload your own files or explore the example</span><button onclick={() => session.loadSample()} disabled={session.sampleLoading || session.analyzeStatus === "loading"}>{session.sampleLoading ? "Loading example…" : "Reset demo"}</button><a href="https://github.com/hyuncat/Attune">Source on GitHub</a></div>
  <Toolbar />

  <div class="main-splitter">
    <aside class="left-column">
      <RecordingTree />
      <SettingsPanel />
    </aside>

    <section class="center-column">
      <div class="score-pane" style={`flex: 0 1 ${(scoreFitHeight ?? 200) + legendHeight + 4}px; min-height: 180px; max-height: 45%;`}>
        <div class="score-viewer-wrap">
          <ScoreViewer bind:this={scoreViewer} {onNoteClicked} {onAnnotationClicked} onReady={onScoreViewerReady} />
        </div>
        <div class="score-legend-row" bind:clientHeight={legendHeight}>
          <span class="legend-swatches">
            {#if session.scoreColorMode === "volume"}
              <span class="legend-item"><span class="swatch volume-swatch"></span>volume</span>
            {:else if session.scoreColorMode === "timing"}
              <span class="legend-item"><span class="swatch" style="background:rgb(198,30,0)"></span>error</span>
            {:else}
              <span class="legend-item"><span class="swatch" style="background:rgb(198,30,0)"></span>wrong note</span>
              <span class="legend-item"><span class="swatch" style="background:rgb(212,130,0)"></span>off-pitch</span>
            {/if}
          </span>
          <label class="color-mode-picker">
            Colors:
            <select bind:value={session.scoreColorMode}>
              <option value="pitch">Pitch</option>
              <option value="timing">Timing</option>
              <option value="volume">Volume</option>
            </select>
          </label>
        </div>
      </div>
      <div class="overlay-pane">
        {#if session.analysisResult && session.scoreNotesActive}
          <NoteOverlay
            scoreNotes={session.scoreNotesActive}
            userNotes={session.analysisResult.note_data}
            pairs={session.currentPairs}
            pitchMistakes={session.pitchMistakes}
            timingMistakes={session.timingMistakes}
            pitchFrames={session.analysisResult.pitch_data?.pitches}
            vibratoPoints={session.analysisResult.vibrato?.points}
            vibMinCycles={session.analysisResult.config?.vib_min_cycles}
            pitchTolerance={session.pitchTolerance}
            currentTime={playback.currentTime}
            selectedMistake={session.selectedMistake}
            selectedMistakeOverridden={session.selectedMistake ? session.overridden.has(session.overrideKey(session.selectedMistake)) : false}
            onSeek={(t) => playback.seek(t)}
          />
        {:else}
          <div class="overlay-placeholder">
            Upload a score and a recording, then click Analyze to see the
            pitch overlay here.
          </div>
        {/if}
      </div>
    </section>

    <aside class="right-column">
      <div class="results-pane">
        <ResultsView />
      </div>
      <div class="note-panel-pane">
        <NotePanel
          userNotesActive={session.analysisResult?.note_data}
          pitchFrames={session.analysisResult?.pitch_data?.pitches}
          vibratoPoints={session.analysisResult?.vibrato?.points}
          currentTime={playback.currentTime}
        />
      </div>
    </aside>
  </div>

  <TransportBar />
  <StatusBar />
</div>

<style>
  .demo-header { display: flex; align-items: center; gap: 20px; padding: 10px 16px; background: #172535; color: #dfe9f0; font-size: 13px; flex-shrink: 0; }
  .demo-header button { background: #263d50; color: #edf4f8; border: 1px solid #6b879c; border-radius: 4px; padding: 5px 10px; cursor: pointer; }
  .demo-header button:hover:not(:disabled) { background: #36546c; }
  .demo-header button:disabled { background: #253340; color: #b4c0ca; cursor: wait; }
  .demo-header a { color: #a6dbc9; text-decoration: none; }
  .demo-header span { flex: 1; }
  @media (max-width: 650px) { .demo-header span { font-size: 11px; } .demo-header { gap: 10px; } }
  .app-shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  .main-splitter {
    flex: 1;
    display: flex;
    min-height: 0;
  }
  /* QMainWindow::separator is 4px, in the shared border-gray (a static
     approximation here - no drag interaction, just the real handle width
     and color instead of a 1px pane border). Panes themselves are plain
     QWidgets, so they take the base window background, not the
     input-family surface gray. */
  .left-column {
    width: 240px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    border-right: 4px solid var(--border);
    background: var(--bg-window);
    min-width: 180px;
    max-width: 320px;
  }
  .center-column {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  .score-pane {
    flex: 1;
    min-height: 200px;
    display: flex;
    flex-direction: column;
  }
  .score-viewer-wrap {
    flex: 1;
    min-height: 0;
  }
  /* perform.py's _build_score_legend_row: swatches (left) + a right-aligned
     "Colors:" dropdown, same row shape as GuitarHero's own legend below. */
  .score-legend-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 2px 8px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    flex-shrink: 0;
  }
  .legend-swatches {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
  }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 0.3rem;
  }
  .volume-swatch {
    background: linear-gradient(to right, rgb(61, 1, 76), rgb(53, 74, 125), rgb(30, 131, 126), rgb(85, 181, 88), rgb(228, 208, 33));
  }
  .color-mode-picker {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  .color-mode-picker select {
    background: var(--bg-input, #2a2b30);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 4px;
    font-size: 0.8rem;
  }
  .overlay-pane {
    flex: 1;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    /* center children (the placeholder box, or NoteOverlay once analyzed)
       vertically in whatever height this pane ends up with, rather than
       pinning them to the top - .overlay-placeholder itself isn't flex:1
       anymore (see its own comment), so without this it defaults to
       flex-start and sits right under the border instead of centered. */
    padding: 8px;
    overflow: auto;
    border-top: 4px solid var(--border);
  }
  .overlay-placeholder {
    /* NOT flex:1 - .overlay-pane grows to absorb whatever height
       .score-pane's auto-fit no longer needs, so stretching this box to
       fill 100% of an oversized .overlay-pane left a huge empty void with
       one line of text floating in it. Sizing to its own content instead
       (padding + text, no stretch) keeps it a normal, proportioned box that
       sits at the top of .overlay-pane (a column flex container's default
       main-axis alignment) - any leftover .overlay-pane height beyond that
       is just plain background, not part of this box. */
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
    background: var(--overlay-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .right-column {
    width: 296px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    border-left: 4px solid var(--border);
    background: var(--bg-window);
    min-width: 240px;
    max-width: 420px;
  }
  /* app.py's right_column QSplitter: MistakeWidget+ToleranceWidget take the
     slack, NotePanel keeps a fixed size below it (setStretchFactor(1, 0)). */
  .results-pane {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
  .note-panel-pane {
    flex: 1;
    min-height: 220px;
    border-top: 4px solid var(--border);
    overflow: hidden;
  }
</style>
