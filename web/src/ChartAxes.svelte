<script>
  let { width, height, xRange, yRange, label } = $props();
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const format = value => Number(value.toFixed(2)).toString();
</script>
{#each ticks as f}
  <line x1={f * width} x2={f * width} y1="0" y2={height} />
  <line x1="0" x2={width} y1={f * height} y2={f * height} />
  {#if f === 0 || f === 0.5 || f === 1}
    <text x={f * width} y={height + 16} text-anchor={f === 0 ? 'start' : f === 1 ? 'end' : 'middle'}>{format(xRange[0] + f * (xRange[1] - xRange[0]))}</text>
  {/if}
  <text x="-6" y={height * (1 - f) + 4} text-anchor="end">{format(yRange[0] + f * (yRange[1] - yRange[0]))}</text>
{/each}
<text x="0" y="-7">{label}</text>
<text x={width} y={height + 28} text-anchor="end">Time (s)</text>
<style>
  line { stroke: #adb6c2; stroke-opacity: 0.16; pointer-events: none; }
  text { fill: #e6e6eb; font-size: 10px; pointer-events: none; }
</style>
