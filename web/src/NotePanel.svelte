<script>
  // Ports ui/note/NotePanel.py: the right-column per-note inspector, sitting
  // below the mistake table in its own row (app.py's right_column splitter:
  // MistakeWidget+ToleranceWidget on top, NotePanel below). A combo picks
  // which graph shows the note under the playback cursor; orchestration
  // only - each graph (VolumeChart, VibratoChart) owns its own rendering.
  import { noteContaining, volumeLevelStats } from "./noteCurve.js";
  import VolumeChart from "./VolumeChart.svelte";
  import VibratoChart from "./VibratoChart.svelte";

  let { userNotesActive, pitchFrames, vibratoPoints, currentTime } = $props();

  const TABS = ["Volume", "Vibrato"];
  const HELP = {
    Volume: "Shows how the recorded level changes during the note, relative to "
      + "the average played level in this take. 0 dB is average; positive "
      + "values are louder and negative values softer. The average excludes "
      + "unvoiced frames and silence. The grey line is the pitch contour.",
    Vibrato: "Vibrato over the note under the cursor. Speed is the "
      + "oscillation rate in Hz; Width is the full peak-to-peak pitch "
      + "excursion in cents. Dot colors use the minimum and "
      + "maximum across the whole recording for direct note-to-note "
      + "comparison. The grey line is the pitch contour.",
  };

  let activeTab = $state("Volume");
  let volumeStats = $derived(volumeLevelStats(pitchFrames));
  let currentNote = $derived(noteContaining(userNotesActive, currentTime));
</script>

<div class="note-panel">
  <div class="note-panel-header">
    <select bind:value={activeTab}>
      {#each TABS as tab}
        <option value={tab}>{tab}</option>
      {/each}
    </select>
    <span class="help-icon" title={HELP[activeTab]}>?</span>
  </div>

  {#if activeTab === "Volume"}
    <VolumeChart note={currentNote} {pitchFrames} {volumeStats} {currentTime} />
  {:else}
    <VibratoChart note={currentNote} {vibratoPoints} {pitchFrames} {currentTime} />
  {/if}
</div>

<style>
  .note-panel {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px;
    height: 100%;
    box-sizing: border-box;
    overflow: hidden;
  }
  .note-panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }
  .note-panel-header select {
    background: var(--bg-input, #2a2b30);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 4px;
    font-size: 0.8rem;
  }
  .help-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 1px solid var(--text-secondary);
    color: var(--text-secondary);
    font-size: 0.65rem;
    cursor: help;
    margin-left: auto;
  }
</style>
