// Shared axis-aware navigation for the per-note analysis charts.
export function createChartViewport(getBounds) {
  let xView = $state(null);
  let yView = $state(null);
  let drag = null;
  const xRange = () => xView ?? getBounds().x;
  const yRange = () => yView ?? getBounds().y;
  function reset() { xView = null; yView = null; drag = null; }
  function point(event) {
    const r = event.currentTarget.getBoundingClientRect();
    const { width, height } = getBounds();
    return [(event.clientX - r.left) * (width + 58) / r.width - 46,
      (event.clientY - r.top) * (height + 50) / r.height - 20];
  }
  function zoom(range, fraction, factor, base) {
    const span = Math.max(0.0001, Math.min((base[1] - base[0]) * 20, (range[1] - range[0]) * factor));
    const anchor = range[0] + fraction * (range[1] - range[0]);
    return [anchor - span * fraction, anchor + span * (1 - fraction)];
  }
  function wheel(event) {
    const [x, y] = point(event);
    const b = getBounds();
    const overX = x >= 0 && x <= b.width, overY = y >= 0 && y <= b.height;
    const inPlot = overX && overY;
    const xAxis = overX && y > b.height, yAxis = x < 0 && overY;
    if (!inPlot && !xAxis && !yAxis) return;
    event.preventDefault();
    const factor = Math.exp(Math.max(-1, Math.min(1, event.deltaY * 0.002)));
    if (inPlot || xAxis) xView = zoom(xRange(), x / b.width, factor, b.x);
    if (inPlot || yAxis) yView = zoom(yRange(), 1 - y / b.height, factor, b.y);
  }
  function down(event) {
    if (event.button !== 0) return;
    const [x, y] = point(event), b = getBounds();
    if (x < 0 || x > b.width || y < 0 || y > b.height) return;
    drag = { point: [x, y], x: [...xRange()], y: [...yRange()] };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function move(event) {
    if (!drag) return;
    const [x, y] = point(event), b = getBounds();
    xView = drag.x.map(v => v - (x - drag.point[0]) / b.width * (drag.x[1] - drag.x[0]));
    yView = drag.y.map(v => v + (y - drag.point[1]) / b.height * (drag.y[1] - drag.y[0]));
  }
  function up() { drag = null; }
  return { get x() { return xRange(); }, get y() { return yRange(); }, reset, wheel, down, move, up };
}
