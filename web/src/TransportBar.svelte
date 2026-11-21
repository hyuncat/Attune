<script>
  // Ports app.py's init_slider_layout transport row exactly: play button,
  // record button, time label, slider, Analyze button - a persistent
  // full-width row below the main splitter, not inside a tab.
  //
  // Play/slider drive real MIDI playback (playback.svelte.js). Record stays
  // a stub - live audio capture is out of scope for this upload-only web
  // app, not something task #3 covers. Analyze is real - moved here from
  // the old standalone upload panel, since on the desktop app it's the
  // last control in this same row, not a separate form.
  import { session } from "./sessionState.svelte.js";
  import { playback } from "./playback.svelte.js";

  const ICONS = "/icons";

  let canAnalyze = $derived(
    !!session.scoreFile && !!session.audioFile && session.analyzeStatus !== "loading"
  );
  let canPlay = $derived(!!session.noteData && !playback.loading);

  // mirrors app.py's update_time_label's format_time exactly: "MM:SS.s"
  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
  }

  function togglePlay() {
    if (playback.isPlaying) playback.pause();
    else playback.play();
  }

  let scrubbing = false;
  let resumeAfterScrub = false;
  function startScrub() {
    scrubbing = true;
    resumeAfterScrub = playback.isPlaying;
    if (resumeAfterScrub) playback.pause();
  }
  function finishScrub() {
    if (!scrubbing) return;
    scrubbing = false;
    if (resumeAfterScrub) playback.play();
  }
  function handleSeek(e) {
    playback.seek(parseFloat(e.target.value));
  }
</script>

<svelte:window onpointerup={finishScrub} onpointercancel={finishScrub} />

<div class="transport-bar">
  <button
    class="icon-btn"
    onclick={togglePlay}
    disabled={!canPlay}
    title={playback.error || (playback.loading ? "Loading audio engine + soundfont..." : "Play/pause")}
  >
    {#if playback.loading}
      <span class="spinner" aria-label="Loading"></span>
    {:else}
      <img src="{ICONS}/{playback.isPlaying ? 'pause' : 'play'}.png" alt={playback.isPlaying ? "Pause" : "Play"} />
    {/if}
  </button>
  {#if playback.loading}
    <span class="loading-label">Loading audio engine…</span>
  {/if}
  <button class="icon-btn" disabled title="Recording isn't supported in the web version (upload-only)">
    <img src="{ICONS}/record.png" alt="Record" />
  </button>
  <span class="time-label">{formatTime(playback.currentTime)} / {formatTime(playback.duration)}</span>
  <input
    class="transport-slider"
    type="range"
    min="0"
    max={playback.duration || 0}
    step="0.05"
    value={playback.currentTime}
    oninput={handleSeek}
    onpointerdown={startScrub}
    aria-label="Playback position"
    disabled={!session.noteData || playback.duration <= 0}
  />
  <button class="analyze-btn" onclick={() => session.runAnalyze()} disabled={!canAnalyze}>
    {session.analyzeStatus === "loading" ? "Analyzing..." : "Analyze"}
  </button>
</div>

{#if session.analyzeStatus === "error"}
  <p class="error">{session.analyzeError}</p>
{/if}
{#if playback.error}
  <p class="error">Playback: {playback.error}</p>
{/if}

<style>
  /* self.transport_widget is a plain QWidget (not a QToolBar/QStatusBar), so
     it takes the base QWidget background - same color as the rest of the
     window, not a distinct "raised" panel. The hairline top border is a
     minimal web-only affordance since the real app gets its visual
     separation for free from the splitter/tab edge above this row. */
  .transport-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: var(--bg-window);
    border-top: 1px solid var(--border);
  }
  /* play/record are real QPushButtons (icon-only, fixed 26x26): transparent
     at rest, accent-tinted hover/pressed, bordered in the shared gray. */
  .icon-btn {
    width: 26px;
    height: 26px;
    padding: 4px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .icon-btn:hover:not(:disabled) {
    background: var(--accent-hover-bg);
  }
  .icon-btn:active:not(:disabled) {
    background: var(--accent-pressed-bg);
  }
  .icon-btn img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .icon-btn:disabled img {
    opacity: 0.5;
  }
  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid var(--text-disabled);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .loading-label {
    color: var(--text-secondary);
    font-size: 0.8rem;
    white-space: nowrap;
  }
  .time-label {
    min-width: 100px;
    color: var(--text);
    font-family: monospace;
    font-size: 0.85rem;
  }
  /* QSlider: 4px groove, filled portion in accent, unfilled in a dark blue
     (not neutral gray) - accent-color alone can't express the two-tone
     track, so the groove/thumb are hand-styled instead of relying on it. */
  .transport-slider {
    flex: 1;
    -webkit-appearance: none;
    appearance: none;
    height: 26px;
    min-width: 40px;
    cursor: pointer;
    touch-action: none;
    background: transparent;
  }
  .transport-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    margin-top: -6px;
    border-radius: 8px;
    background: var(--accent);
  }
  .transport-slider::-webkit-slider-runnable-track { height: 4px; background: var(--slider-track-unfilled); border-radius: 2px; }
  .transport-slider::-moz-range-track { height: 4px; background: var(--slider-track-unfilled); }
  .transport-slider::-moz-range-thumb { width: 16px; height: 16px; border: 0; border-radius: 50%; background: var(--accent); }
  .transport-slider:disabled {
    opacity: 0.5;
  }
  /* QPushButton: border in the shared gray, accent-colored TEXT (not
     white), transparent at rest, accent-tinted hover/pressed backgrounds. */
  .analyze-btn {
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
    flex-shrink: 0;
  }
  .analyze-btn:hover:not(:disabled) {
    background: var(--accent-hover-bg);
  }
  .analyze-btn:active:not(:disabled) {
    background: var(--accent-pressed-bg);
  }
  .analyze-btn:disabled {
    color: var(--text-disabled);
    cursor: default;
  }
  .error {
    color: var(--danger);
    font-size: 0.85rem;
    margin: 4px 10px;
  }
</style>
