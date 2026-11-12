// graph.js: the live graph. Planner on the left, five tools on the right, verifier between.
// Packets only ever move because a real trace event said so (see engine.js).

import { TOOLS } from "./engine.js";

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
const TONE = { go: "var(--lime)", ok: "var(--lime)", warn: "var(--ember)", bad: "var(--ember)", pass: "var(--lime)", retry: "var(--ember)", replan: "var(--ember)", fail: "var(--ember)" };

export class Graph {
  constructor(host) {
    this.host = host;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.visible = true;
    this.speed = 1;
    this.queue = [];      // packets waiting for the previous one to land
    this.flying = [];
    this.nodes = {};
    this.pos = {};
    this.svg = el("svg", { class: "graph-svg", role: "img", "aria-label": "Agent graph: planner, five tools and a verifier" });
    host.appendChild(this.svg);
    this.build();
    this.layout();
    new ResizeObserver(() => this.layout()).observe(host);
    new IntersectionObserver((e) => { this.visible = e[0].isIntersecting; if (this.visible) this.kick(); }).observe(host);
    this.raf = 0;
  }

  build() {
    const s = this.svg;
    this.gEdges = el("g", { class: "edges" }, s);
    this.gNodes = el("g", { class: "nodes" }, s);
    this.gPackets = el("g", { class: "packets" }, s);
    this.edges = { back: el("path", { class: "edge back" }, this.gEdges), out: {}, in: {} };
    for (const t of TOOLS) {
      this.edges.out[t] = el("path", { class: "edge" }, this.gEdges);
      this.edges.in[t] = el("path", { class: "edge" }, this.gEdges);
    }
    // planner
    const p = el("g", { class: "node planner", "data-node": "planner" }, this.gNodes);
    this.nodes.planner = { g: p, ring: el("circle", { class: "ring" }, p), body: el("circle", { class: "body" }, p),
      label: el("text", { class: "lbl strong", "text-anchor": "middle", "dominant-baseline": "central" }, p),
      sub: el("text", { class: "sub", "text-anchor": "middle" }, p) };
    this.nodes.planner.label.textContent = "PLAN";
    // verifier
    const v = el("g", { class: "node verifier", "data-node": "verifier" }, this.gNodes);
    this.nodes.verifier = { g: v, ring: el("rect", { class: "ring" }, v), body: el("rect", { class: "body" }, v),
      label: el("text", { class: "lbl", "text-anchor": "middle" }, v), sub: el("text", { class: "sub", "text-anchor": "middle" }, v) };
    this.nodes.verifier.label.textContent = "VERIFY";
    // tools
    for (const t of TOOLS) {
      const g = el("g", { class: "node tool", "data-node": t }, this.gNodes);
      this.nodes[t] = { g, ring: el("rect", { class: "ring" }, g), body: el("rect", { class: "body" }, g),
        label: el("text", { class: "lbl", "text-anchor": "middle", "dominant-baseline": "central" }, g),
        sub: el("text", { class: "sub", "text-anchor": "middle" }, g) };
      this.nodes[t].label.textContent = t;
    }
  }

  layout() {
    const W = this.host.clientWidth || 600;
    const H = this.host.clientHeight || 360;
    this.W = W; this.H = H;
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const narrow = W < 520;
    const pillW = narrow ? 66 : 84, pillH = narrow ? 30 : 32, r = narrow ? 24 : 30, dia = narrow ? 18 : 22;
    const padY = pillH / 2 + 18;
    const cy = H / 2;
    this.pos.planner = { x: W * (narrow ? 0.14 : 0.12), y: cy };
    this.pos.verifier = { x: W * 0.5, y: cy };
    TOOLS.forEach((t, i) => { this.pos[t] = { x: W - pillW / 2 - (narrow ? 12 : 28), y: padY + (i * (H - 2 * padY)) / (TOOLS.length - 1) }; });
    const P = this.pos, N = this.nodes;
    set(N.planner.ring, { cx: P.planner.x, cy: P.planner.y, r: r + 8 });
    set(N.planner.body, { cx: P.planner.x, cy: P.planner.y, r });
    set(N.planner.label, { x: P.planner.x, y: P.planner.y });
    set(N.planner.sub, { x: P.planner.x, y: P.planner.y + r + 18 });
    const d = dia * 2;
    for (const k of ["ring", "body"]) {
      const pad = k === "ring" ? 8 : 0;
      set(N.verifier[k], { x: P.verifier.x - dia - pad, y: P.verifier.y - dia - pad, width: d + 2 * pad, height: d + 2 * pad,
        transform: `rotate(45 ${P.verifier.x} ${P.verifier.y})`, rx: 4 });
    }
    set(N.verifier.label, { x: P.verifier.x, y: P.verifier.y + dia + 28 });
    set(N.verifier.sub, { x: P.verifier.x, y: P.verifier.y + dia + 44 });
    for (const t of TOOLS) {
      const p = P[t];
      for (const k of ["ring", "body"]) {
        const pad = k === "ring" ? 5 : 0;
        set(N[t][k], { x: p.x - pillW / 2 - pad, y: p.y - pillH / 2 - pad, width: pillW + 2 * pad, height: pillH + 2 * pad, rx: (pillH + 2 * pad) / 2 });
      }
      set(N[t].label, { x: p.x, y: p.y });
      set(N[t].sub, { x: p.x, y: p.y + pillH / 2 + 13 });
      this.edges.out[t].setAttribute("d", `M${P.planner.x + r},${P.planner.y} L${p.x - pillW / 2},${p.y}`);
      this.edges.in[t].setAttribute("d", `M${p.x - pillW / 2},${p.y} L${P.verifier.x + dia * 1.2},${P.verifier.y}`);
    }
    this.edges.back.setAttribute("d", `M${P.verifier.x - dia * 1.2},${P.verifier.y} L${P.planner.x + r},${P.planner.y}`);
  }

  // ---- state setters used by the engine's cues
  reset() {
    this.queue = []; this.flying.forEach((f) => f.node.remove()); this.flying = [];
    for (const n of Object.values(this.nodes)) { n.g.classList.remove("hot", "ok", "bad", "go", "warn", "think"); n.sub.textContent = ""; }
    this.setSub("verifier", "");
  }
  setSub(node, text, tone) { const n = this.nodes[node]; if (!n) return; n.sub.textContent = text; n.sub.setAttribute("class", "sub " + (tone || "")); }
  think(on, phase, step) {
    const g = this.nodes.planner.g;
    g.classList.toggle("think", on);
    if (on) this.setSub("planner", phase === "plan" ? "planning" : phase === "act" ? `filling args ${step || ""}` : "answering");
    else this.setSub("planner", "");
  }
  pulse(node, tone, instant) {
    const n = this.nodes[node]; if (!n) return;
    n.g.classList.remove("ok", "bad", "go", "warn");
    n.g.classList.add(tone === "go" ? "go" : tone === "ok" ? "ok" : tone === "bad" ? "bad" : "warn");
    if (this.reduced || instant) return;
    n.g.classList.remove("hot"); void n.g.getBoundingClientRect(); n.g.classList.add("hot");
  }
  verdict(v, forced) {
    this.setSub("verifier", forced ? "pass-through" : v, v === "pass" ? "ok" : "bad");
    const g = this.nodes.verifier.g;
    g.classList.remove("ok", "bad", "go", "warn");
    g.classList.add(v === "pass" || forced ? "ok" : "bad");
  }
  toolDone(tool, ms, ok) { this.setSub(tool, ms != null ? `${ms} ms` : "", ok ? "" : "bad"); }

  // ---- packets
  packet(from, to, tone, instant) {
    if (instant || this.reduced) return;
    this.queue.push({ from, to, tone });
    this.kick();
  }
  kick() {
    if (!this.raf && this.visible) this.raf = requestAnimationFrame((t) => this.tick(t));
  }
  tick(now) {
    this.raf = 0;
    if (!this.flying.length && this.queue.length) this.launch(now);
    for (const f of this.flying) {
      const u = Math.min(1, (now - f.t0) / f.dur);
      const e = ease(u);
      const x = f.a.x + (f.b.x - f.a.x) * e, y = f.a.y + (f.b.y - f.a.y) * e;
      f.dot.setAttribute("transform", `translate(${x} ${y})`);
      f.trail.setAttribute("x2", x); f.trail.setAttribute("y2", y);
      f.trail.style.opacity = String(0.8 * (1 - Math.max(0, u - 0.7) / 0.3));
      if (u >= 1) { f.done = true; f.node.remove(); }
    }
    this.flying = this.flying.filter((f) => !f.done);
    if (this.flying.length || this.queue.length) this.raf = requestAnimationFrame((t) => this.tick(t));
  }
  launch(now) {
    const q = this.queue.shift();
    const a = this.edgePoint(q.from, q.to, true), b = this.edgePoint(q.from, q.to, false);
    const color = TONE[q.tone] || TONE.go;
    const node = el("g", { class: "packet" }, this.gPackets);
    const trail = el("line", { class: "trail", x1: a.x, y1: a.y, x2: a.x, y2: a.y, stroke: color }, node);
    const dot = el("g", {}, node);
    el("circle", { r: 10, fill: color, opacity: 0.18 }, dot);
    el("circle", { r: 4, fill: color }, dot);
    const dist = Math.hypot(b.x - a.x, b.y - a.y);
    const dur = Math.max(260, Math.min(620, 220 + dist * 0.9)) / this.speed;
    this.flying.push({ a, b, node, trail, dot, t0: now, dur });
  }
  edgePoint(from, to, start) {
    const A = this.pos[from], B = this.pos[to];
    if (!A || !B) return start ? A || B || { x: 0, y: 0 } : B || A;
    return start ? A : B;
  }
}

function set(node, attrs) { for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v); }
