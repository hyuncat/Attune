<script>
  // Ports ui/note/VibratoWidget.py: vibrato speed/width over the note under
  // the cursor. Width is the full peak-to-peak excursion in cents.
  // the y value; dot color encodes the OTHER metric, ramped against the
  // whole take's own range for note-to-note comparison. The pitch contour is
  // underlaid, matching VolumeChart. NotePanel.svelte owns picking which
  // note/tab is active; this only draws - including its own metric combo
  // (desktop puts that in NotePanel's shared header row instead; keeping it
  // here instead keeps NotePanel a plain orchestrator with no per-graph
  // knowledge, a cleaner boundary for the same behavior).
  import { vibratoCurve, vibratoGlobalRange, pitchContour, contourToBand, paddedRange } from "./noteCurve.js";
  import { viridis, cssRgb } from "./colors.js";

  let { note, vibratoPoints, pitchFrames, currentTime } = $props();

  const WIDTH = 380;
  const HEIGHT = 140;
  const Y_PADDING = 0.15;
  const DEFAULT_RANGES = { Width: [0, 200], Speed: [0, 10] };

  let metric = $state("Width");
  let widthMode = $derived(metric === "Width");

  let window = $derived(note ? [note.startTime, note.endTime] : null);

  let curve = $derived.by(() => {
    if (!window) return [];
    return vibratoCurve(vibratoPoints, window[0], window[1]);
  });

  let yDomain = $derived(paddedRange(...DEFAULT_RANGES[metric], Y_PADDING));

  // dot color = the metric NOT currently on the y-axis, ramped against the
  // take's own range - mirrors _render's colors/color_range selection.
  let colorMetric = $derived(widthMode ? "rate" : "width");
  let colorRange = $derived(vibratoGlobalRange(vibratoPoints, colorMetric));

  function colorFor(rate, width) {
    const value = widthMode ? rate : width;
    if (!colorRange || colorRange[1] <= colorRange[0]) return cssRgb(viridis(0.5));
    const frac = Math.max(0, Math.min(1, (value - colorRange[0]) / (colorRange[1] - colorRange[0])));
    return cssRgb(viridis(frac));
  }

  let contourPoints = $derived.by(() => {
    if (!window) return [];
    return contourToBand(pitchContour(pitchFrames, window[0], window[1]), yDomain[0], yDomain[1]);
  });

  function xPos(t) {
    if (!window) return 0;
    const span = window[1] - window[0] || 1;
    return ((t - window[0]) / span) * WIDTH;
  }
  function yPos(v) {
    const [y0, y1] = yDomain;
    const span = y1 - y0 || 1;
    return HEIGHT - ((v - y0) / span) * HEIGHT;
  }

  function toPath(points, xKey, yKey) {
    let d = "";
    let started = false;
    for (const p of points) {
      const v = p[yKey];
      if (v == null || !Number.isFinite(v)) {
        started = false;
        continue;
      }
      const cmd = started ? "L" : "M";
      d += `${cmd}${xPos(p[xKey]).toFixed(1)},${yPos(v).toFixed(1)} `;
      started = true;
    }
    return d.trim();
  }

  let contourPath = $derived(toPath(contourPoints, "time", "y"));

  let visiblePoints = $derived.by(() => {
    return curve
      .map((p) => ({
        time: p.time,
        value: widthMode ? p.width : p.rate,
        color: colorFor(p.rate, p.width),
      }))
      .filter((p) => Number.isFinite(p.value));
  });

  let curvePath = $derived(toPath(visiblePoints, "time", "value"));
  let timelineX = $derived(window ? xPos(Math.min(Math.max(currentTime, window[0]), window[1])) : null);

  const GRADIENT_LABELS = { Width: ["slow", "fast"], Speed: ["narrow", "wide"] };
</script>

<div class="vibrato-wrap">
  <div class="metric-row">
    <select bind:value={metric}>
      <option value="Width">Width</option>
      <option value="Speed">Speed</option>
    </select>
    <span class="legend">
      <span class="gradient-swatch"></span>
      {GRADIENT_LABELS[metric][0]} → {GRADIENT_LABELS[metric][1]}
    </span>
  </div>

  {#if !note}
    <div class="chart-blank">No note under the cursor</div>
  {:else}
    <svg viewBox="0 0 {WIDTH} {HEIGHT}" class="chart" role="img" aria-label="Vibrato over the current note">
      {#if contourPath}
        <path d={contourPath} class="contour-line" />
      {/if}
      {#if curvePath}
        <path d={curvePath} class="vibrato-line" />
      {/if}
      {#each visiblePoints as p}
        <circle cx={xPos(p.time)} cy={yPos(p.value)} r="2.5" fill={p.color} />
      {/each}
      {#if timelineX != null}
        <line x1={timelineX} y1="0" x2={timelineX} y2={HEIGHT} class="timeline" />
      {/if}
      <text x="4" y="12" class="axis-label">{widthMode ? "cents" : "Hz"}</text>
      <text x={WIDTH - 4} y={HEIGHT - 4} class="axis-label" text-anchor="end">Time (s)</text>
    </svg>
  {/if}
</div>

<style>
  .vibrato-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
  }
  .metric-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    color: var(--text-secondary);
  }
  .metric-row select {
    background: var(--bg-input, #2a2b30);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 4px;
    font-size: 0.75rem;
  }
  .legend {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .gradient-swatch {
    display: inline-block;
    width: 40px;
    height: 8px;
    border-radius: 2px;
    background: linear-gradient(to right, rgb(68, 1, 84), rgb(59, 82, 139), rgb(33, 145, 140), rgb(94, 201, 98), rgb(253, 231, 37));
  }
  .chart {
    width: 100%;
    height: 116px;
    background: rgb(20, 20, 25);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .chart-blank {
    width: 100%;
    height: 116px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgb(20, 20, 25);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 0.8rem;
  }
  /* ui.Colors.NOTE_CONTOUR_RGB['vibrato'] = (95,100,110), same as volume's */
  .contour-line {
    fill: none;
    stroke: rgb(95, 100, 110);
    stroke-width: 1.5;
  }
  /* pg.mkPen(*Colors.viridis(0.55), 230, width=4) - a fixed mid-viridis line;
     the DOTS carry the real per-point color. */
  .vibrato-line {
    fill: none;
    stroke: rgb(45, 156, 132);
    stroke-width: 2.5;
    opacity: 0.9;
  }
  .timeline {
    stroke: rgb(0, 255, 0);
    stroke-width: 1.5;
  }
  .axis-label {
    fill: rgb(230, 230, 235);
    font-size: 9px;
  }
</style>
