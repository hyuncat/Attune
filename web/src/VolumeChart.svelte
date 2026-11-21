<script>
  import { createChartViewport } from "./chartViewport.svelte.js";
  import ChartAxes from "./ChartAxes.svelte";
  // Ports ui/note/VolumeWidget.py: the cursor-note's level relative to the take's average,
  // with the pitch contour underlaid for context, windowed to the note's own
  // [startTime, endTime] span. NotePanel.svelte owns picking which note/tab
  // is active; this only draws.
  import { volumeCurveDb, pitchContour, contourToBand, paddedRange } from "./noteCurve.js";

  let { note, pitchFrames, volumeStats, currentTime } = $props();

  let panelWidth = $state(280);
  let panelHeight = $state(240);
  const WIDTH = $derived(Math.max(120, panelWidth - 60));
  const HEIGHT = $derived(Math.max(80, panelHeight - 52));
  const Y_PADDING = 0.15;

  // One reference and scale for all notes in the take; leave breathing room
  // around zero even when the performance is nearly constant in level.
  let yDomain = $derived.by(() => {
    if (!volumeStats) return [-24, 24];
    const { referenceDb, min, max } = volumeStats;
    return paddedRange(Math.min(-6, min - referenceDb),
                       Math.max(6, max - referenceDb), Y_PADDING);
  });

  let window = $derived(note ? [note.startTime, note.endTime] : null);

  let contourPoints = $derived.by(() => {
    if (!window) return [];
    return contourToBand(pitchContour(pitchFrames, window[0], window[1]), yDomain[0], yDomain[1]);
  });

  let curvePoints = $derived.by(() => {
    if (!window || !volumeStats) return [];
    const ref = volumeStats.referenceDb;
    return volumeCurveDb(pitchFrames, window[0], window[1], yDomain[0] + ref)
      .map((point) => ({ ...point, db: point.db - ref }));
  });


  const clipId = $props.id();
  const viewport = createChartViewport(() => ({ x: window ?? [0, 1], y: yDomain, width: WIDTH, height: HEIGHT }));
  $effect(() => { note?.startTime; note?.endTime; pitchFrames; volumeStats; viewport.reset(); });

  function xPos(t) {
    if (!window) return 0;
    const [x0, x1] = viewport.x;
    return ((t - x0) / (x1 - x0 || 1)) * WIDTH;
  }
  function yPos(v) {
    const [y0, y1] = viewport.y;
    const span = y1 - y0 || 1;
    return HEIGHT - ((v - y0) / span) * HEIGHT;
  }

  // Builds an SVG path, breaking into a new subpath (M) after every null -
  // mirrors PlotDataItem(connect="finite")'s gap behavior.
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
  let curvePath = $derived(toPath(curvePoints, "time", "db"));
  let timelineX = $derived(window ? xPos(Math.min(Math.max(currentTime, window[0]), window[1])) : null);
</script>

<div class="chart-wrap" bind:clientWidth={panelWidth} bind:clientHeight={panelHeight}>
  {#if !note}
    <div class="chart-blank">No note under the cursor</div>
  {:else}
    <svg viewBox="0 0 {WIDTH + 58} {HEIGHT + 50}" class="chart" role="img" aria-label="Volume over the current note" onwheel={viewport.wheel} onpointerdown={viewport.down} onpointermove={viewport.move} onpointerup={viewport.up} onpointercancel={viewport.up} onlostpointercapture={viewport.up} ondblclick={viewport.reset}>
      <title>Scroll over the plot to zoom both axes; scroll over labels to zoom one axis. Drag to pan. Double-click to reset.</title>
      <defs><clipPath id={clipId}><rect width={WIDTH} height={HEIGHT} /></clipPath></defs>
      <g transform="translate(46,20)">
      <ChartAxes width={WIDTH} height={HEIGHT} xRange={viewport.x} yRange={viewport.y} label="Relative level (dB)" />
      <g clip-path={`url(#${clipId})`}>
      {#if contourPath}
        <path d={contourPath} class="contour-line" />
      {/if}
      {#if volumeStats}
        <line x1="0" y1={yPos(0)} x2={WIDTH} y2={yPos(0)} class="average-line" />
        <text x={WIDTH - 4} y={yPos(0) - 4} class="axis-label" text-anchor="end">0 · Average</text>

      {/if}
      {#if curvePath}
        <path d={curvePath} class="volume-line" />
      {/if}
      {#if timelineX != null}
        <line x1={timelineX} y1="0" x2={timelineX} y2={HEIGHT} class="timeline" />
      {/if}


      </g></g>
    </svg>
  {/if}
</div>

<style>
  .chart-wrap {
    width: 100%;
    flex: 1;
    min-height: 0;
  }
  .chart {
    display: block;
    touch-action: none;
    user-select: none;
    cursor: grab;
    width: 100%;
    height: 100%;
    background: rgb(20, 20, 25);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .chart-blank {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgb(20, 20, 25);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 0.8rem;
  }
  /* ui.Colors.NOTE_CONTOUR_RGB['volume'] = (95,100,110) */
  .contour-line {
    fill: none;
    stroke: rgb(95, 100, 110);
    stroke-width: 1.5;
  }
  /* ui.Colors.NOTE_VOLUME_RGB = (94,201,98) */
  .average-line {
    stroke: rgb(230, 230, 235);
    stroke-opacity: 0.45;
    stroke-dasharray: 5 4;
  }
  .volume-line {
    fill: none;
    stroke: rgb(94, 201, 98);
    stroke-width: 2.5;
  }
  /* ui.Colors.NOTE_TIMELINE_RGB = (0,255,0), mirrors GuitarHero's own cursor */
  .timeline {
    stroke: rgb(0, 255, 0);
    stroke-width: 1.5;
  }
  .axis-label {
    fill: rgb(230, 230, 235);
    font-size: 11px;
  }
</style>
