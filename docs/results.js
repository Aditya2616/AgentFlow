// results.js: draws the fault dose-response chart and table from results.json (written by the evaluation).
const ARMS = ["off", "retry", "full"];
const LABEL = { off: "no verifier", retry: "retry only", full: "full verifier" };
const KIND = { transient: "Transient tool failures", silent: "Silently wrong results" };
const COLOR = { off: "#8a8d94", retry: "#e9e9e4", full: "#d7ff3f" };
const DASH = { off: "5 4", retry: "", full: "" };
const NS = "http://www.w3.org/2000/svg";
const $ = (s) => document.querySelector(s);

function pick(cells, kind, rate, arm) {
  return cells.find((c) => c.arm === arm && c.fault_rate === rate && (rate === 0 ? c.kind === "clean" : c.kind === kind));
}

function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function draw(svg, data, kind) {
  svg.replaceChildren();
  const W = 640, H = 300, L = 46, T = 14, PW = W - L - 16, PH = H - T - 48;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const rates = [...new Set(data.cells.map((c) => c.fault_rate).concat([0]))].sort((a, b) => a - b);
  const xmax = Math.max(...rates) || 1;
  const X = (r) => L + (PW * r) / xmax, Y = (v) => T + PH * (1 - v);
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: L, x2: L + PW, y1: Y(g), y2: Y(g), stroke: "rgba(237,236,230,.1)" }, svg);
    const t = el("text", { x: L - 8, y: Y(g) + 4, "text-anchor": "end", fill: "#8a8d94", "font-size": 11 }, svg);
    t.textContent = `${g * 100}%`;
  }
  for (const r of rates) {
    const t = el("text", { x: X(r), y: T + PH + 18, "text-anchor": "middle", fill: "#8a8d94", "font-size": 11 }, svg);
    t.textContent = `${Math.round(r * 100)}%`;
  }
  const lab = el("text", { x: L + PW / 2, y: H - 8, "text-anchor": "middle", fill: "#8a8d94", "font-size": 11 }, svg);
  lab.textContent = "injected fault rate per tool call";
  for (const arm of ARMS) {
    const pts = rates.map((r) => pick(data.cells, kind, r, arm)).filter(Boolean);
    if (pts.length < 2) continue;
    const band = pts.map((p) => `${X(p.fault_rate)},${Y(p.hi)}`).concat(pts.slice().reverse().map((p) => `${X(p.fault_rate)},${Y(p.lo)}`));
    el("polygon", { points: band.join(" "), fill: COLOR[arm], opacity: 0.1 }, svg);
    const ys = pts.map((p) => `${X(p.fault_rate)},${Y(p.k / p.n)}`).join(" ");
    const pl = el("polyline", { points: ys, fill: "none", stroke: COLOR[arm], "stroke-width": 2 }, svg);
    if (DASH[arm]) pl.setAttribute("stroke-dasharray", DASH[arm]);
    for (const p of pts) el("circle", { cx: X(p.fault_rate), cy: Y(p.k / p.n), r: 3.4, fill: COLOR[arm] }, svg);
  }
}

function table(data, kind) {
  const rows = [];
  const rates = [...new Set(data.cells.map((c) => c.fault_rate).concat([0]))].sort((a, b) => a - b);
  for (const r of rates) for (const arm of ARMS) {
    const c = pick(data.cells, kind, r, arm); if (!c) continue;
    rows.push(`<tr><td>${Math.round(r * 100)}%</td><td>${LABEL[arm]}</td><td>${Math.round((100 * c.k) / c.n)}%` +
      ` <span class="ci">${c.k}/${c.n} [${Math.round(100 * c.lo)}, ${Math.round(100 * c.hi)}]</span></td>` +
      `<td>${c.model_calls.toFixed(1)}</td><td>${c.tool_calls.toFixed(1)}</td></tr>`);
  }
  return `<table><thead><tr><th>fault rate</th><th>arm</th><th>success [95% CI]</th><th>model calls</th><th>tool calls</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

export async function initResults() {
  let data;
  try { data = await (await fetch("results.json")).json(); } catch { data = null; }
  const root = $("#results");
  if (!data || !data.cells.length) { $("#results-empty").hidden = false; return false; }
  const svg = $("#dose-chart");
  let kind = "transient";
  const render = () => {
    draw(svg, data, kind);
    $("#dose-table").innerHTML = table(data, kind);
    for (const b of document.querySelectorAll("#kind-seg button")) b.setAttribute("aria-selected", String(b.dataset.kind === kind));
    $("#kind-note").textContent = kind === "transient"
      ? "The tool call raises a timeout. A plain retry fixes it, so retry-only should keep up with the full verifier."
      : "The tool succeeds but returns a wrong result: no search hits, zero SQL rows, empty Python output or a cut-off document. Only checking the result can notice that.";
  };
  $("#kind-seg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { kind = b.dataset.kind; render(); } });
  $("#results-meta").textContent = `${data.model}, ${data.device} ${data.dtype}. Measured by agentflow ablate; numbers are read from results.json.`;
  render();
  return root;
}
